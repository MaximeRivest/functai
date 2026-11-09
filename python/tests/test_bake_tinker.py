"""Baking on Tinker from Python, end to end against tools/fake_tinker.py through
Tinker's official SDK (the fake's model is a lookup table that learns): the plan
and its price, each method, checks choosing the checkpoint and temperature,
pictures, resuming, and the model called through ``fn.using(lm=...)``. Offline:
the student is a toy one (contract/cases/training's tokenizer and templates)."""

import json
import struct
import sys
import threading
import zlib
from typing import Literal

import pytest

pytest.importorskip("tinker")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

import functai  # noqa: E402
import functai.bake  # noqa: E402
from contract_support import CONTRACT  # noqa: E402
from functai import ai  # noqa: E402
from functai.bake.examples import BakeError  # noqa: E402
from lm15.types import ImagePart, image  # noqa: E402

sys.path.insert(0, str(CONTRACT.parent / "tools"))
import fake_tinker  # noqa: E402

CASES = CONTRACT / "cases" / "training"


@ai
def capital(country: str) -> str:
    """The capital city of the country."""


@ai
def mood(text: str) -> Literal["happy", "sad"]:
    """The mood of the message."""


@ai
def where(screen: ImagePart, target: str) -> Literal["top", "bottom"]:
    """The corner the target is in."""


PAIRS = [("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Chile", "Santiago")]


def rows(n=4):
    return [{"country": c, "result": r} for c, r in PAIRS for _ in range(n)]


TEST5 = [{"country": c, "result": r} for c, r in PAIRS]


@pytest.fixture(scope="module", autouse=True)
def no_pytorch():
    """Training on Tinker needs no PyTorch: these tests run as if it were not installed (the
    environment has it, for local training). The SDK decides at import whether it has PyTorch,
    so its flag is set too: a sparse tensor made dense through PyTorch failed live this way."""
    from tinker.types import tensor_data
    mp = pytest.MonkeyPatch()
    mp.setitem(sys.modules, "torch", None)
    mp.setattr(tensor_data, "_HAVE_TORCH", False)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    home = tmp_path_factory.mktemp("bake-tinker")
    mp = pytest.MonkeyPatch()
    mp.setenv("XDG_CACHE_HOME", str(home / "cache"))
    mp.setenv("FUNCTAI_PRICES", "off")
    tok = json.loads((CASES / "toy-tokenizer.json").read_text())
    tok["added_tokens"].append({"id": 298, "content": "<|unused_200054|>", "single_word": False, "lstrip": False,
                                "rstrip": False, "normalized": False, "special": True})
    files = home / "toy"
    files.mkdir()
    (files / "tokenizer.json").write_text(json.dumps(tok))
    (files / "chat_template.jinja").write_text((CASES / "qwen-like.jinja").read_text())
    pics = home / "inkling-toy"
    pics.mkdir()
    (pics / "tokenizer.json").write_text(json.dumps(tok))
    (pics / "chat_template.jinja").write_text((CASES / "inkling-like.jinja").read_text().replace("<|pic|>", "<|unused_200054|>"))
    (pics / "processor_config.json").write_text(json.dumps({"image_processor": {"size": {"width": 40, "height": 40}}}))
    price = {"prefill": "$0.30", "cached_prefill": "$0.03", "sample": "$1.00", "train": "$1.50", "context": "8K"}
    (home / "cache" / "functai" / "prices").mkdir(parents=True)
    (home / "cache" / "functai" / "prices" / "tinker-models.json").write_text(json.dumps(
        [{"tinker_id": "toy/Toy-1", **price}, {"tinker_id": "thinkingmachines/Inkling-Toy", **price}]))
    server, url, fake = fake_tinker.serve(api_key="tml-test")
    model = {"vocab": 299, "prefill": 0.3, "cached": 0.03, "sample": 1.0, "train": 1.5, "context": 8192,
             "state_gb": 0.01, "sampler_gb": 0.005}
    fake.models.update({"toy/Toy-1": {**model, "patch": None}, "thinkingmachines/Inkling-Toy": {**model, "patch": 40}})
    yield {"home": home, "files": files, "pics": pics, "url": url, "fake": fake}
    server.shutdown()
    mp.undo()


def opts(env, **o):
    return {"where": "tinker", "tinker": {"api_key": "tml-test", "base_url": env["url"]}, "log": None,
            "student_files": str(env["files"]), **o}


def test_the_plan_prices_the_bake_and_spends_nothing(env):
    sessions = dict(env["fake"].sessions)
    plan = functai.bake.bake(capital, rows(), **opts(env, student="toy/Toy-1", test=TEST5, plan_only=True,
                                                     run_folder=env["home"] / "plan"))
    text = str(plan)
    assert "bake capital → toy/Toy-1 on Tinker (sft, LoRA r32)" in text
    assert "20 train · 5 held out" in text and "sent to Thinking Machines" in text
    assert plan.dollars["total"] > 0
    assert env["fake"].sessions == sessions


def test_sft_learns_checks_choose_and_calls_run_on_tinker(env):
    from contract_support import assert_valid, validator
    folder = env["home"] / "sft"
    b = functai.bake.bake(capital, rows(), **opts(env, student="toy/Toy-1", test=TEST5, run_folder=folder))
    assert_valid(validator("baked"), b.meta, "baked.json")
    w = b.meta["weights"]
    assert w["service"] == "tinker" and "sampler_weights/best-step" in w["uri"] and w["state_uri"].endswith("-state")
    assert b.meta["template"]["source"] == "files"
    assert b.meta["training"]["best"]["score"] == 1 and b.meta["sampling"]["temperature"] == 0
    assert b.meta["training"]["checks"][0]["scores"]["0.0"] < 1
    assert capital.using(lm=b)("Peru") == "Lima"
    losses = [r["loss"] for r in map(json.loads, (folder / "metrics.jsonl").read_text().splitlines()) if "loss" in r]
    assert losses[-1] < losses[0] / 3
    assert not any("/weights/step-" in p for p in env["fake"].checkpoints)
    again = functai.bake.bake(capital, rows(), **opts(env, student="toy/Toy-1", test=TEST5, run_folder=folder))
    assert again.meta["weights"]["uri"] == w["uri"]
    loaded = functai.bake.load(b.path)
    loaded.on("tinker", tinker={"api_key": "tml-test", "base_url": env["url"]}, files=str(env["files"]))
    assert capital.using(lm=loaded)("Japan") == "Tokyo"


def test_criteria_decide_by_their_name_not_by_exact_match(env):
    only_lima = functai.Criterion("only_lima", "verdict", judge=lambda answer: answer == "Lima")
    b = functai.bake.bake(capital, rows(), **opts(env, student="toy/Toy-1", test=TEST5, criteria=[only_lima],
                                                 run_folder=env["home"] / "criteria", epochs=2))
    best = b.meta["training"]["best"]
    assert best["score"] == pytest.approx(0.2), "one case of five says Lima: the criterion's 0.2, not exact match's 1"
    assert b.meta["training"]["criteria"] == {"only_lima": pytest.approx(0.2)}


def test_spent_is_what_tinker_billed_this_runs_sessions_and_none_while_its_records_are_not_in(env):
    import re
    folder = env["home"] / "billed"
    b = functai.bake.bake(capital, rows(), **opts(env, student="toy/Toy-1", test=TEST5, run_folder=folder, epochs=1))
    spent = b.meta["training"]["spent"]
    sessions = json.loads((folder / "run.json").read_text())["sessions"]
    assert len(sessions) == 1
    usage = env["fake"].usage
    ours = sum(u["estimated_cost_usd"] for u in usage if u["session_id"] in sessions)
    assert 0 < ours < sum(u["estimated_cost_usd"] for u in usage), "earlier tests' bakes spent on the same account"
    assert spent["billed"] == pytest.approx(ours, abs=1e-12)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:00:00Z", spent["billed_through"])
    env["fake"].options["billing_lag"] = True
    try:
        later = functai.bake.bake(capital, rows(), **opts(env, student="toy/Toy-1", test=TEST5,
                                                          run_folder=env["home"] / "billed-later", epochs=1))
    finally:
        env["fake"].options["billing_lag"] = False
    s2 = later.meta["training"]["spent"]
    assert s2["billed"] is None and s2["billed_through"] is None, "not billed yet is not $0"
    assert s2["estimated"] > 0


def test_a_check_whose_calls_fail_stops_the_bake_a_failure_is_not_a_score_of_0(env):
    folder = env["home"] / "check-fails"
    env["fake"].options["fail"] = {"/api/v1/asample": 1}     # the first check's first call
    try:
        with pytest.raises(Exception, match="failed on purpose"):
            functai.bake.bake(capital, rows(), **opts(env, student="toy/Toy-1", test=TEST5, run_folder=folder))
    finally:
        env["fake"].options["fail"] = {}
    assert json.loads((folder / "run.json").read_text())["state"] == "failed"


def test_a_run_that_fails_resumes_from_its_last_saved_state(env):
    folder = env["home"] / "resume"
    o = opts(env, student="toy/Toy-1", test=TEST5, run_folder=folder, epochs=4, eval_every=100, batch=8)
    fake = env["fake"]
    earlier = set(fake.runs)
    done = threading.Event()

    def watch():
        while not done.is_set():
            new = [r for k, r in fake.runs.items() if k not in earlier and r["opt"].get("steps", 0) >= 4]
            if new:
                fake.options["fail"] = {"/api/v1/forward_backward": 1}
                return
            done.wait(0.002)
    t = threading.Thread(target=watch, daemon=True)
    t.start()
    with pytest.raises(Exception):
        functai.bake.bake(capital, rows(), **o)
    done.set()
    run = json.loads((folder / "run.json").read_text())
    assert run["state"] == "failed" and run["states"], "a state was saved before the failure"
    b = functai.bake.bake(capital, rows(), **o)
    assert json.loads((folder / "run.json").read_text())["state"] == "done"
    assert b.meta["training"]["steps"] == 8
    assert any("/weights/step-" in (r.get("loaded_from") or "") for r in fake.runs.values())


def _share(env, student, uri):
    import tinker
    from functai.bake.functions import Entry
    from functai.bake.service import _model_input, conversation
    sc = tinker.ServiceClient(api_key="tml-test", base_url=env["url"])
    s = sc.create_sampling_client(model_path=uri)
    s = s.result() if hasattr(s, "result") and not isinstance(s, tinker.SamplingClient) else s
    msgs, pics, _ = conversation(Entry.build(capital), {"country": "Peru"}, None)
    prompt = student.render(msgs, pics).prompt
    r = s.sample(prompt=_model_input(prompt, pics), num_samples=100,
                 sampling_params=tinker.types.SamplingParams(max_tokens=32, temperature=1.0, stop=[student.stop_token],
                                                              seed=7)).result()
    return sum("Lima" in student.read_reply(list(q.tokens)) for q in r.sequences) / 100


def test_rl_moves_the_student_towards_rewarded_answers(env):
    mixed = [{"country": "Peru", "result": "Lima"}] * 4 + [{"country": "Peru", "result": "Cusco"}] * 4 + \
        [{"country": "France", "result": "Paris"}] * 4
    start = functai.bake.bake(capital, mixed, **opts(env, student="toy/Toy-1", test=[{"country": "France", "result": "Paris"}],
                                                     run_folder=env["home"] / "rl-start", epochs=4, batch=4, lr=4e-5,
                                                     eval_every=100))
    is_lima = functai.Criterion("is_lima", "verdict", judge=lambda answer: answer == "Lima")
    rl = functai.bake.bake(capital, [{"country": "Peru"}, {"country": "Peru", "result": "Lima"}],
                           **opts(env, method="rl", start=start, reward=is_lima, group=8, epochs=6, batch=1,
                                  test=[{"country": "Peru", "result": "Lima"}], run_folder=env["home"] / "rl",
                                  temperatures=[1.0]))
    assert rl.meta["training"]["method"] == "rl" and rl.meta["training"]["trained_on"] == ["is_lima"]
    metrics = [json.loads(x) for x in (env["home"] / "rl" / "metrics.jsonl").read_text().splitlines()]
    assert any("agreement" in m for m in metrics), "how alike the answers are is measured at checks"
    student = start._runner.student
    before = _share(env, student, start.meta["weights"]["uri"])
    after = _share(env, student, rl.meta["weights"]["uri"])
    assert 0 < before < 0.5, f"the start answers Lima only sometimes ({before})"
    assert after > before + 0.4, f"Lima {before} → {after}"


def test_rl_with_the_energy_score_runs(env):
    start = functai.bake.bake(capital, rows(2), **opts(env, student="toy/Toy-1", test=TEST5,
                                                       run_folder=env["home"] / "energy-start", epochs=1, lr=3e-5))
    b = functai.bake.bake(capital, rows(2), **opts(env, method="rl", start=start, test=TEST5,
                                                   reward=functai.bake.rewards.energy(lambda a, b: 0.0 if a == b else 1.0),
                                                   group=4, epochs=1, run_folder=env["home"] / "energy",
                                                   temperatures=[1.0]))
    checks = b.meta["training"]["checks"]
    assert len(checks) >= 2 and all(isinstance(c["scores"]["1.0"], float) for c in checks)


def test_prefer_moves_the_student_to_the_chosen_answer(env):
    mixed = [{"country": "Peru", "result": "Lima"}] * 3 + [{"country": "Peru", "result": "Cusco"}] * 3 + \
        [{"country": "France", "result": "Paris"}] * 3
    start = functai.bake.bake(capital, mixed, **opts(env, student="toy/Toy-1", test=[{"country": "France", "result": "Paris"}],
                                                     run_folder=env["home"] / "dpo-start", epochs=2, lr=4e-5))
    pairs = [{"inputs": {"country": "Peru"}, "chosen": {"result": "Lima"}, "rejected": {"result": "Cusco"}}] * 4
    b = functai.bake.bake(capital, [], **opts(env, method="prefer", start=start, pairs=pairs, beta=1.0, lr=5e-4, epochs=3,
                                              batch=2, test=[{"country": "Peru", "result": "Lima"}],
                                              run_folder=env["home"] / "dpo"))
    losses = [m["loss"] for m in map(json.loads, (env["home"] / "dpo" / "metrics.jsonl").read_text().splitlines())
              if "loss" in m]
    assert losses[-1] < losses[0]
    assert capital.using(lm=b, temperature=0)("Peru") == "Lima"


def test_decision_gives_probabilities_and_answers_the_likeliest(env):
    data = [{"text": t, "result": r} for t, r in [("What a lovely day", "happy"), ("I lost my keys", "sad"),
                                                  ("We won!", "happy"), ("It rained all week", "sad")] for _ in range(3)]
    b = functai.bake.bake(mood, data, **opts(env, method="decision", student="toy/Toy-1", test=data[::3],
                                             run_folder=env["home"] / "decision", epochs=3))
    assert b.meta["training"]["decision"]["choices"] == ["happy", "sad"]
    assert mood.using(lm=b)("I lost my keys") == "sad" and mood.using(lm=b)("We won!") == "happy"
    req = {"model": "s", "messages": [{"role": "user", "parts": [{"type": "text", "text": "<text>\nWe won!\n</text>\n"}]}]}
    probs = b._runner.probabilities(req)
    assert abs(sum(p for _c, p in probs) - 1) < 1e-9


def test_distill_against_a_teacher_with_the_same_tokenizer_and_refuses_another(env):
    b = functai.bake.bake(capital, rows(1), **opts(env, method="distill", student="toy/Toy-1", teacher="toy/Toy-1",
                                                   test=TEST5, run_folder=env["home"] / "distill", epochs=1, batch=5,
                                                   group=2, max_tokens=16))
    assert b.meta["training"]["method"] == "distill"
    other = env["home"] / "other"
    other.mkdir(exist_ok=True)
    tok = json.loads((env["files"] / "tokenizer.json").read_text())
    tok["added_tokens"].append({"id": 299, "content": "<|other|>", "single_word": False, "lstrip": False,
                                "rstrip": False, "normalized": False, "special": True})
    (other / "tokenizer.json").write_text(json.dumps(tok))
    (other / "chat_template.jinja").write_text((env["files"] / "chat_template.jinja").read_text())
    with pytest.raises(BakeError, match="tokenizer is not"):
        functai.bake.bake(capital, rows(1), **opts(env, method="distill", student="toy/Toy-1", teacher="Qwen/Qwen3-8B",
                                                   test=TEST5, run_folder=env["home"] / "distill-bad", epochs=1,
                                                   max_tokens=16,
                                                   student_files=lambda i: str(env["files"] if i == "toy/Toy-1" else other)))


def _png(w, h):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes([(y * 7) % 256]) * w for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_pictures_train_with_their_token_counts_and_calls_send_them(env):
    screens = [(_png(80, 41), "top"), (_png(120, 40), "bottom"), (_png(41, 81), "top")]
    data = [{"screen": image(data=p, media_type="image/png"), "target": "OK", "result": r} for p, r in screens for _ in range(3)]
    b = functai.bake.bake(where, data, **opts(env, student="thinkingmachines/Inkling-Toy", student_files=str(env["pics"]),
                                              test=data[::3], run_folder=env["home"] / "pictures", epochs=3))
    assert b.meta["template"]["picture"]["rule"] == "grid"
    assert where.using(lm=b)(image(data=screens[1][0], media_type="image/png"), "OK") == "bottom"
