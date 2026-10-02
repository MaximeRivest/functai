"""Generative students: examples, plans, runs, trainers, runners, adoption.

Offline: a 2-layer chat model with random weights and Qwen3.5's tokenizer and
chat template (from the local Hugging Face cache), trained on the CPU in its
own process; Tinker through a fake service client; teachers are FakeRouters."""

import json
import random
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")

import functai  # noqa: E402
from conftest import FakeRouter  # noqa: E402
from functai import ai, module  # noqa: E402
from functai.bake import BakeError  # noqa: E402

QWEN = "Qwen/Qwen3.5-0.8B"


def _have(name):
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(name, local_files_only=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have(QWEN), reason=f"{QWEN}'s tokenizer is not in the local cache")

PAIRS = [("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima"), ("Chile", "Santiago"), ("Kenya", "Nairobi")]


@ai
def capital(country: str, style: str) -> str:
    """The capital city of the country."""


@ai
def flag(country: str, style: str) -> str:
    """The main colour of the country's flag."""


def rows(n=30, style="short"):
    return [{"country": c, "style": style, "result": p} for _ in range(n) for c, p in PAIRS]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    path = tmp_path_factory.mktemp("tiny") / "chat"
    tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                      pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(autouse=True)
def no_price_lookups(monkeypatch):
    monkeypatch.setenv("FUNCTAI_PRICES", "off")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


TRAIN = dict(local_files_only=True, log=None, device="cpu", lora=False, lr=2e-3, epochs=10)


# ------------------------------------------------------------------ what a student reads


def test_fixed_inputs_are_left_out_checked_and_refused_when_they_vary(tiny):
    ex = functai.bake.examples(capital, rows(2), student=tiny, fixed={"style": "short"}, log=None)
    msgs = ex.rows[0]["messages"]
    assert "style" not in json.dumps(msgs) and "<country>" in msgs[1]["content"]
    assert msgs[-1] == {"role": "assistant", "content": "<result>\nParis\n</result>"}
    with pytest.raises(BakeError, match="another value"):
        functai.bake.examples(capital, rows(1) + rows(1, style="long"), fixed={"style": "short"}, log=None)
    with pytest.raises(BakeError, match="no function here has an input"):
        functai.bake.examples(capital, rows(2), fixed={"colour": "red"}, log=None)


def test_derived_inputs_must_follow_their_source(tiny):
    data = [{"country": c, "style": f"about {c}", "result": p} for c, p in PAIRS] * 3
    ex = functai.bake.examples(capital, data, student=tiny, derived={"style": "country"}, log=None)
    entry = ex.entries["capital"]
    assert entry.derived["style"]["from"] == "country" and len(entry.derived["style"]["values"]) == 5
    bad = data + [{"country": "France", "style": "other", "result": "Paris"}]
    with pytest.raises(BakeError, match="does not decide it"):
        functai.bake.examples(capital, bad, derived={"style": "country"}, log=None)


def test_examples_carry_the_exact_tokens_and_round_trip(tiny, tmp_path):
    from functai.bake.dataset import Examples
    from functai.bake.template import load_tokenizer, prompt_ids
    data = rows(4)
    for i, r in enumerate(data):
        r["teacher"], r["w"] = ("opus" if i % 2 else "astra"), (2.0 if i % 2 else 1.0)
    ex = functai.bake.examples(capital, data, student=tiny, tags="teacher", weights="w", log=None)
    r = ex.rows[0]
    tok = load_tokenizer(tiny)
    assert list(r["input_ids"][:r["answer_start"]]) == prompt_ids(tok, r["messages"][:-1], {"enable_thinking": False})
    assert tok.decode(r["input_ids"][r["answer_start"]:]).startswith("<result>\nParis\n</result><|im_end|>")
    assert {x["tag"] for x in ex.rows} == {"opus", "astra"} and {x["weight"] for x in ex.rows} == {1.0, 2.0}
    s = ex.stats()
    assert s["rows"] == 20 and s["answer"]["max"] >= s["answer"]["median"] > 0
    for name in ("ex.parquet", "ex.jsonl", "ex.jsonl.gz"):
        ex.save(tmp_path / name)
        back = Examples.load(tmp_path / name)
        assert len(back) == 20 and back.student == tiny and list(back.rows[3]["input_ids"]) == list(ex.rows[3]["input_ids"])
        assert back.entries["capital"]["layout"]["name"] == ex.entries["capital"].layout["name"]
    assert len(ex.filter(lambda x: x["tag"] == "opus")) == 10
    hf = ex.to_hf()
    assert set(hf.column_names) >= {"messages", "prompt", "completion", "input_ids"}


def test_rows_without_answers_go_to_the_teacher(tiny):
    r = FakeRouter(responder=lambda req: "<result>\nTEACHER\n</result>")
    functai.configure(lm="gpt-4.1-mini", client=r)
    data = rows(2)
    for x in data[:4]:
        del x["result"]
    ex = functai.bake.examples(capital, data, student=tiny, log=None)
    answers = [x["messages"][-1]["content"] for x in ex.rows]
    assert answers.count("<result>\nTEACHER\n</result>") == 4 and len(r.requests) == 4
    assert ex.info["labeling"]["rows"] == 4
    assert sorted({x["source"] for x in ex.rows}) == ["data", "teacher"]


@module
def atlas(country: str):
    return [capital(country, "short"), flag(country, "short")]


def test_a_program_gives_examples_of_every_function_inside(tiny):
    def answer(req):
        return "<result>\nCAP\n</result>" if "capital city" in req.system else "<result>\nRED\n</result>"
    functai.configure(lm="gpt-4.1-mini", client=FakeRouter(responder=answer))
    ex = functai.bake.examples(atlas, [{"country": c} for c, _ in PAIRS] * 3, student=tiny, log=None)
    assert ex.stats()["functions"] == {"capital": 15, "flag": 15}
    assert {x["messages"][-1]["content"] for x in ex.rows} == {"<result>\nCAP\n</result>", "<result>\nRED\n</result>"}


# ------------------------------------------------------------------ batches by tokens


def test_batches_by_tokens_pack_or_group_and_weights_repeat():
    from functai.bake.trainers.here_worker import grouped_bins, packed_bins, repeat_by_weight
    rng = random.Random(0)
    lengths = {i: rng.randint(10, 400) for i in range(300)}
    bins = packed_bins(list(lengths), lengths, 1000)
    assert sorted(i for b in bins for i in b) == list(range(300))
    assert all(sum(lengths[i] for i in b) <= 1000 for b in bins)
    assert len(bins) <= sum(lengths.values()) / 1000 * 1.1 + 1           # best-fit decreasing wastes little
    groups = grouped_bins(list(lengths), lengths, 1000)
    assert sorted(i for b in groups for i in b) == list(range(300))
    assert all(max(lengths[i] for i in b) * len(b) <= 1000 or len(b) == 1 for b in groups)
    rep = repeat_by_weight(list(range(1000)), [2.5] * 1000, seed=0)
    assert 2400 < len(rep) < 2600 and min(rep.count(i) for i in range(1000)) >= 2


def test_packed_batches_restart_positions_and_learn_only_answers():
    from functai.bake.trainers.here_worker import Collate
    c = Collate(None, [], pad=0, packed=True)
    out = c.collate([(list(range(1, 6)), 3), (list(range(10, 14)), 1)])
    assert out["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 2, 3]]
    assert out["labels"].tolist() == [[-100, -100, -100, 4, 5, -100, 11, 12, 13]]
    padded = Collate(None, [], pad=0, packed=False).collate([([1, 2, 3], 2), ([7, 8], 1)])
    assert padded["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert padded["labels"].tolist() == [[-100, -100, 3], [-100, 8, -100]]


def test_the_schedule_warms_up_holds_and_decays():
    from functai.bake.trainers.tinker import wsd
    lr = [wsd(s, 100, 0.03, 0.2) for s in range(100)]
    assert lr[0] < lr[1] < lr[2] == 1.0 and lr[79] == 1.0 and lr[80] < 1.0 and lr[99] == 0.0
    assert all(a >= b for a, b in zip(lr[80:], lr[81:]))


# ------------------------------------------------------------------ choosing where


def _fake_places(monkeypatch, table):
    """Trainers that answer from ``table``: {place: (ok, seconds, dollars, set_up)}."""
    from functai.bake import trainers
    from functai.bake.recipe import recipe

    def make(name):
        ok, secs, dollars, set_up = table[name]

        class T(trainers.Trainer):
            def set_up(self):
                return set_up

            def estimate(self, job):
                return trainers.Estimate(ok, [] if ok else [f"{name} cannot"], secs, dollars,
                                         {**recipe(job), "precision": "bf16", "micro_tokens": 1024, "accumulate": 1,
                                          "devices": ["cpu"], "packing": False}, summary=f"{name} summary")
        T.name = name
        return T()
    real = trainers.trainer
    monkeypatch.setattr(trainers, "trainer", lambda n: make(n) if isinstance(n, str) and n in table else real(n))


def test_auto_takes_here_when_it_can_else_the_cheapest_service(tiny, home, monkeypatch):
    _fake_places(monkeypatch, {"here": (True, 600, None, True), "tinker": (True, None, 3.0, True),
                               "prime": (True, None, 2.0, True)})
    p = functai.bake.plan(capital, rows(), student=tiny, local_files_only=True, log=None)
    assert p.where == "here" and "free" in p.reason
    _fake_places(monkeypatch, {"here": (False, None, None, True), "tinker": (True, None, 3.0, True),
                               "prime": (True, None, 2.0, True)})
    p = functai.bake.plan(capital, rows(), student=tiny, local_files_only=True, log=None)
    assert p.where == "prime" and "cheapest" in p.reason and "tinker summary" in repr(p)
    functai.configure(bake_where=["tinker", "prime"])
    assert functai.bake.plan(capital, rows(), student=tiny, local_files_only=True, log=None).where == "tinker"
    functai.configure(bake_where=None)
    _fake_places(monkeypatch, {"here": (False, None, None, True), "tinker": (True, None, 3.0, False),
                               "prime": (False, None, None, False)})
    with pytest.raises(BakeError, match="nowhere to train"):
        functai.bake.plan(capital, rows(), student=tiny, local_files_only=True, log=None)


def test_auto_picks_a_head_when_it_can_and_a_student_otherwise(tiny, home):
    from typing import Literal

    @ai
    def mood(text: str) -> Literal["happy", "sad"]:
        """Mood."""
    from functai.bake import _all_finite
    assert _all_finite(mood) and not _all_finite(capital)
    p = mood.bake([{"text": "x", "result": "happy"}] * 20, student=tiny, plan_only=True, local_files_only=True,
                  log=None, device="cpu")
    assert type(p).__name__ == "Plan"            # plan_only asks for a generative plan


# ------------------------------------------------------------------ training here, for real (on the CPU)


@pytest.fixture(scope="module")
def baked(tiny, tmp_path_factory):
    folder = tmp_path_factory.mktemp("run") / "capital"
    b = capital.bake(rows(), student=tiny, fixed={"style": "short"}, run_folder=folder,
                     metric=lambda row, pred: pred.result == row["result"], **TRAIN)
    return b


def test_a_student_trains_here_and_answers_through_its_function(baked):
    r = baked.report
    assert r.functions[0].readable == 1.0 and r.functions[0].score >= 0.6
    assert baked.meta["functai_baked"] == 2 and baked.meta["weights"]["form"] == "merged"
    from contract_support import assert_valid, validator
    assert_valid(validator("baked"), baked.meta, "baked.json")
    t = baked.meta["training"]
    assert t["steps"] > 0 and t["history"] and t["eval_loss"] < 2.0
    fast = capital.using(lm=baked)
    assert fast("Peru", "short") == "Lima"
    with pytest.raises(BakeError, match="fixed to one value"):
        fast("Peru", "long")


def test_judge_scores_each_group_apart(baked):
    test = rows(2)
    for i, r in enumerate(test):
        r["teacher"] = "opus" if i % 2 else "astra"
    rep = functai.bake.judge(baked, capital, test, metric=lambda row, pred: pred.result == row["result"],
                             by="teacher", num_threads=4)
    g = rep.functions[0].groups
    assert set(g) == {"opus", "astra"} and g["opus"]["rows"] == 5 and "opus" in repr(rep)


def test_the_call_sends_the_training_tokens(baked):
    from functai.bake.dataset import Examples
    from functai.bake.functions import chat_messages
    from functai.bake.template import prompt_ids
    run = functai.bake.run(baked.meta["run"]["folder"])
    ex = Examples.load(run.folder / "examples.parquet")
    row = next(r for r in ex.rows if "Chile" in r["messages"][1]["content"])
    req = capital.using(lm=baked).render("Chile", "short")
    ids = prompt_ids(baked.tokenizer(), chat_messages(req), baked.template.kwargs)
    assert ids == list(row["input_ids"][:row["answer_start"]])


def test_the_run_folder_has_its_story(baked):
    run = functai.bake.run(baked.meta["run"]["folder"])
    assert run.state == "done" and (run.folder / "plan.json").exists() and run.checkpoints()
    recs = run.records()
    assert any("eval_loss" in r for r in recs) and any("loss" in r for r in recs)
    assert any(x.folder == run.folder for x in functai.bake.runs(run.folder.parent))
    ck = run.checkpoint()
    assert ck.meta["weights"]["form"] == "merged" and "step" in ck.name


def test_running_the_same_bake_again_finds_it_done(baked, tiny):
    again = capital.bake(rows(), student=tiny, fixed={"style": "short"}, run_folder=baked.meta["run"]["folder"],
                         **TRAIN)
    assert again.path == baked.path


def test_a_stopped_run_resumes_from_its_checkpoint(tiny, home):
    run = capital.bake(rows(), student=tiny, wait=False, report=False, **{**TRAIN, "epochs": 16})
    t0 = time.time()
    while not run.checkpoints() and time.time() - t0 < 180:
        time.sleep(0.5)
    run.stop()
    assert run.state == "stopped" and run.checkpoints()
    first = run.checkpoints()[-1]
    run.resume()
    b = run.wait(show=False)
    assert b.meta["training"]["steps"] > first
    log = (run.folder / "log.txt").read_text()
    assert log.count("==== ") >= 2


def test_adopt_checks_the_tokens(baked, tmp_path):
    from functai.bake.dataset import Examples
    run = functai.bake.run(baked.meta["run"]["folder"])
    ex = Examples.load(run.folder / "examples.parquet")
    adopted = functai.bake.adopt(baked.path / "model", capital, examples=ex, path=tmp_path / "adopted")
    assert capital.using(lm=adopted)("Kenya", "short") == "Nairobi"
    other = tmp_path / "other"
    import shutil
    shutil.copytree(baked.path / "model", other)
    cfg = json.loads((other / "tokenizer_config.json").read_text())
    tpl = (other / "chat_template.jinja")
    if tpl.exists():
        tpl.write_text("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                       "{% if add_generation_prompt %}assistant: {% endif %}")
    else:
        cfg["chat_template"] = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        (other / "tokenizer_config.json").write_text(json.dumps(cfg))
    with pytest.raises(BakeError, match="tokenize differently"):
        functai.bake.adopt(other, capital, examples=ex, path=tmp_path / "refused")


def test_export_writes_examples_and_recipes(tiny, home):
    run = capital.bake(rows(), student=tiny, where="export", local_files_only=True, log=None)
    out = run.folder / "export"
    assert run.state == "exported"
    for f in ("examples.parquet", "examples.jsonl", "train_trl.py", "axolotl.yaml", "README.md"):
        assert (out / f).exists(), f
    compile((out / "train_trl.py").read_text(), "train_trl.py", "exec")
    assert "roles_to_train: [assistant]" in (out / "axolotl.yaml").read_text()


def test_format_1_is_refused_with_the_way_forward(tmp_path):
    (tmp_path / "baked.json").write_text(json.dumps({"functai_baked": 1, "kind": "generative"}))
    with pytest.raises(BakeError, match="bake it again"):
        functai.bake.load(tmp_path)


# ------------------------------------------------------------------ Tinker (a fake service)


class _Future:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


class _FakeTinker:
    """Tinker's clients, recording what bake sends."""
    sent = []
    saved = []

    def __init__(self, *a, **k):
        pass

    def create_lora_training_client(self, base_model, rank=32, seed=None, **k):
        return _FakeTraining(base_model, rank)

    def create_training_client_from_state_with_optimizer(self, path, **k):
        c = _FakeTraining("resumed", 0)
        c.resumed_from = path
        return c

    def create_training_client_from_state(self, path, **k):
        return _FakeTraining("loaded", 0)

    def create_sampling_client(self, model_path=None, **k):
        return _FakeSampling(model_path)


class _FakeTraining:
    def __init__(self, base, rank):
        self.base, self.rank, self.steps = base, rank, 0
        self.model_id = "run-1"

    def forward_backward(self, data, loss):
        import tinker
        _FakeTinker.sent.append(data)
        outs = [{"logprobs": tinker.types.TensorData(data=[-0.5] * len(d.loss_fn_inputs["weights"].tolist()),
                                                     dtype="float32")} for d in data]
        return _Future(type("O", (), {"loss_fn_outputs": outs})())

    def forward(self, data, loss):
        return self.forward_backward(data, loss)

    def optim_step(self, params):
        self.steps += 1
        return _Future(None)

    def save_state(self, name, **k):
        _FakeTinker.saved.append(name)
        return _Future(type("S", (), {"path": f"tinker://run-1/weights/{name}"})())

    def save_weights_for_sampler(self, name, **k):
        return _Future(type("S", (), {"path": f"tinker://run-1/sampler_weights/{name}"})())


class _FakeSampling:
    def __init__(self, path):
        self.path = path

    def sample(self, prompt, num_samples, sampling_params):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
        ids = tok("<result>\nLima\n</result>", add_special_tokens=False)["input_ids"] + \
            [tok.convert_tokens_to_ids("<|im_end|>")]
        seq = type("Q", (), {"tokens": ids, "stop_reason": "stop"})()
        return _Future(type("R", (), {"sequences": [seq]})())


def test_tinker_trains_on_our_tokens_and_runs_there(home, monkeypatch):
    if not _have("Qwen/Qwen3.5-4B"):
        pytest.skip("Qwen3.5-4B's config is not in the local cache")
    import tinker
    from functai.bake import prices
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setattr(tinker, "ServiceClient", _FakeTinker)
    monkeypatch.setattr(prices, "tinker_models", lambda: {"Qwen/Qwen3.5-4B": {"train": 0.74, "context": 65536}})
    _FakeTinker.sent.clear()
    p = functai.bake.plan(capital, rows(), student="Qwen/Qwen3.5-4B", where="tinker", local_files_only=True,
                          log=None)
    assert p.estimate.dollars and "$" in repr(p)
    # inline: the run in a thread of this process, where the fake client is (its own process would not see it)
    b = capital.bake(rows(), student="Qwen/Qwen3.5-4B", where="tinker", local_files_only=True, log=None,
                     report=False, epochs=1, inline=True)
    d = _FakeTinker.sent[0][0]
    w = d.loss_fn_inputs["weights"].tolist()
    assert w[0] == 0.0 and w[-1] == 1.0 and 0 < sum(w) < len(w)              # the reply only
    assert b.meta["weights"]["form"] == "remote" and b.meta["weights"]["uri"].startswith("tinker://")
    from contract_support import assert_valid, validator
    assert_valid(validator("baked"), b.meta, "baked.json")
    assert capital.using(lm=b)("Peru", "short") == "Lima" and b.runner.name == "tinker"
