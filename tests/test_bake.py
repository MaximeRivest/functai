"""Baking: training a head model for an AI function, using it, escalating.

Offline and on the CPU: BERT-tiny (4.4M parameters) from the local Hugging Face
cache, a made-up task a small model learns in seconds, a fake teacher."""

import dataclasses
import json
import os
import random
import threading
from typing import Literal

import lm15
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import functai  # noqa: E402
from conftest import FakeRouter  # noqa: E402
from functai import _ai, ai  # noqa: E402
from functai.bake import BakeError, load  # noqa: E402

TINY = "google/bert_uncased_L-2_H-128_A-2"


def _have_tiny() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(TINY, local_files_only=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_tiny(), reason=f"{TINY} is not in the local Hugging Face cache")

POS = ["great", "loved", "excellent", "wonderful", "amazing", "fantastic"]
NEG = ["awful", "hated", "terrible", "boring", "horrible", "bad"]
NEU = ["okay", "fine", "average", "decent", "ordinary", "so-so"]
THINGS = ["movie", "pizza", "service", "book", "hotel", "phone", "concert"]


def reviews(n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        label, words = rng.choice([("positive", POS), ("negative", NEG), ("neutral", NEU)])
        spam = rng.random() < 0.3
        text = f"The {rng.choice(THINGS)} was {rng.choice(words)}." + (" BUY NOW at cheap-deals!" if spam else "")
        out.append({"text": text, "result": label, "spam": spam})
    return out


@ai
def sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """The sentiment of the review."""


@dataclasses.dataclass
class Review:
    sentiment: Literal["positive", "negative", "neutral"]
    spam: bool


@ai
def review(text: str) -> Review:
    """Read the review."""


@ai(module="cot")
def thoughtful(text: str) -> Literal["positive", "negative", "neutral"]:
    """The sentiment of the review."""


OPTS = dict(student=TINY, device="cpu", local_files_only=True, log=None, epochs=6)


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    return sentiment.bake(reviews(600), path=tmp_path_factory.mktemp("b") / "sentiment", **OPTS)


def test_bake_trains_tests_and_reports(baked):
    r = baked.report
    assert r.rows["test"] == 60 and r.rows["validation"] == 54 and r.rows["train"] == 486
    assert r.fields[0].accuracy > 0.95 and r.label_source == "the data's labels"
    assert r.speed["rows_per_second"] > 0 and r.training["passes_run"] >= 1
    text = repr(r)
    assert "accuracy" in text and "answering only when sure" in text
    files = sorted(p.relative_to(baked.path).as_posix() for p in baked.path.rglob("*") if p.is_file())
    assert "baked.json" in files and "model/model.safetensors" in files and "tokenizer/tokenizer.json" in files
    meta = json.loads((baked.path / "baked.json").read_text())
    assert meta["fields"] == [{"name": "result", "keys": ["positive", "negative", "neutral"],
                               "values": ["positive", "negative", "neutral"]}]
    assert meta["layout"]["template"][-1] == {"role": "user", "text": "{text}"}   # the input alone, no prompt


def test_the_function_runs_on_the_weights(baked):
    fast = sentiment.using(lm=baked)
    assert fast("The pizza was amazing.") == "positive"
    pred = fast("The hotel was horrible.", all=True)
    assert pred.result == "negative" and pred.measured_by == {"result": "provider_classification"}
    assert set(pred.probabilities["result"]) == {"positive", "negative", "neutral"}
    assert pred.confidence == pred.probabilities["result"]["negative"] > 0.5
    request = functai.inspect_history(1)[0].request
    assert request.system is None and request.messages[-1].parts[0].text == "The hotel was horrible."


def test_a_baked_model_reads_its_own_layout_whatever_the_function_says(baked):
    templated = sentiment.using(template=[functai.system("Be brief. {instruction}"), functai.user("Review: {text}")])
    pred = templated.using(lm=baked)("The book was great.", all=True)
    assert pred.result == "positive"
    assert functai.inspect_history(1)[0].request.messages[-1].parts[0].text == "The book was great."


def test_hidden_reasoning_is_not_asked_of_a_head(tmp_path, baked):
    # thoughtful has module="cot": on a head model it answers without a reasoning output
    assert thoughtful.using(lm=baked)("The movie was great.") == "positive"


def test_a_changed_function_is_refused(baked):
    @ai
    def two_way(text: str) -> Literal["positive", "negative"]:
        """The sentiment of the review."""
    with pytest.raises(BakeError, match="has changed since"):
        two_way.using(lm=baked)("x")


def test_concurrent_calls_are_batched_and_answers_go_to_the_right_caller(baked):
    fast = sentiment.using(lm=baked)
    texts = [f"The phone was {w}." for w in (POS + NEG + NEU) * 20]
    expected = {**{f"The phone was {w}.": "positive" for w in POS}, **{f"The phone was {w}.": "negative" for w in NEG},
                **{f"The phone was {w}.": "neutral" for w in NEU}}
    got = [None] * len(texts)

    def work(k):
        for i in range(k, len(texts), 16):
            got[i] = fast(texts[i])
    threads = [threading.Thread(target=work, args=(k,)) for k in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert got == [expected[t] for t in texts]


def test_predict_is_the_fast_path_for_tables(baked):
    out = baked.predict([{"text": "The service was awful."}, {"text": "The book was okay."}])
    assert [r["result"] for r in out] == ["negative", "neutral"]
    assert 0 < out[0]["result__confidence"] <= 1 and set(out[0]["result__probs"]) == {"positive", "negative", "neutral"}


def test_loading_checks_the_files(baked, tmp_path):
    copy = baked.save(tmp_path / "copy")
    assert load(copy.path)("x") if False else load(copy.path).name == "sentiment"
    (copy.path / "tokenizer" / "tokenizer.json").write_text("{}")
    with pytest.raises(BakeError, match="changed since baking"):
        load(copy.path)


# ------------------------------------------------------------------ escalation


def test_unsure_answers_go_to_a_bigger_model(baked):
    r = FakeRouter(responder=lambda req: "<result>\nneutral\n</result>")
    functai.configure(lm="gpt-4.1-mini", client=r)
    safe = sentiment.using(lm=baked, escalate_to="gpt-4.1", escalate_below=1.0)
    pred = safe("Something happened.", all=True)
    assert pred.escalated and pred.first.escalated is False and pred.first.confidence < 1.0
    assert r.requests[-1].model == "gpt-4.1" and pred.result == "neutral"
    sure = sentiment.using(lm=baked, escalate_to="gpt-4.1", escalate_below=0.01)
    n = len(r.requests)
    assert sure("The pizza was great.") == "positive" and len(r.requests) == n


def test_escalating_to_an_ai_function_and_no_escalation_loop(baked):
    @ai
    def judge(text: str) -> Literal["positive", "negative", "neutral"]:
        """Judge carefully."""
    r = FakeRouter(responder=lambda req: "<result>\npositive\n</result>")
    functai.configure(lm="gpt-4.1-mini", client=r, escalate_to="gpt-4.1", escalate_below=1.0)   # global, too
    pred = sentiment.using(lm=baked, escalate_to=judge)("Something happened.", all=True)
    assert pred.escalated and pred.result == "positive"
    assert "Judge carefully." in r.requests[-1].system


def test_escalation_needs_a_measured_first_answer():
    r = FakeRouter(responder=lambda req: "<result>\npositive\n</result>")
    functai.configure(lm="gpt-4.1-mini", client=r)
    with pytest.raises(ValueError, match="measures its confidence"):
        sentiment.using(escalate_to="gpt-4.1")("x")


def test_threshold_from_the_report(baked):
    t = baked.report.threshold(0.9)
    assert t is not None and 0 < t["threshold"] <= 1 and t["accuracy"] >= 0.9


# ------------------------------------------------------------------ labels and teachers


def jev_like(req):
    """A teacher that measures: a Jev-style data part with a distribution."""
    text = req.messages[-1].parts[0].text
    label = "positive" if any(w in text for w in POS) else "negative" if any(w in text for w in NEG) else "neutral"
    dist = {k: (0.8 if k == label else 0.1) for k in ("positive", "negative", "neutral")}
    return lm15.Response(id=None, model=req.model, message=lm15.Message.assistant(
        [lm15.DataPart(value={"result": label}, probabilities={"result": dist}, method="provider_classification")]),
        finish_reason="stop", usage=lm15.Usage(input_tokens=10, output_tokens=0, total_tokens=10))


def test_a_measuring_teacher_gives_soft_labels_and_the_report_compares(tmp_path, monkeypatch):
    from functai import models
    monkeypatch.setattr(models, "JUDGMENT_ONLY", models.JUDGMENT_ONLY | {"typesafe"})
    functai.configure(client=FakeRouter(responder=jev_like, provider="typesafe"))
    rows = reviews(400, seed=3)
    b = sentiment.bake(rows, teacher="jev-latest", labels="teacher", path=tmp_path / "t",
                       prices={"teacher": (0.042, 0.0), "gpu_per_hour": 0.12}, **OPTS)
    r = b.report
    assert r.label_source == "teacher (soft)" and r.labeling["rows"] == r.rows["train"] + r.rows["validation"]
    assert r.fields[0].teacher_accuracy == 1.0 and r.fields[0].agreement is not None
    assert r.labeling["dollars"] == pytest.approx(r.labeling["rows"] * 10 * 0.042 / 1e6)
    assert any("can at best match it" in n for n in r.notes) or r.fields[0].accuracy > r.fields[0].teacher_accuracy


def test_a_text_teacher_gives_hard_labels_for_unlabeled_rows(tmp_path):
    def text_teacher(req):
        t = req.messages[-1].parts[0].text
        label = "positive" if any(w in t for w in POS) else "negative" if any(w in t for w in NEG) else "neutral"
        return f"<result>\n{label}\n</result>"
    functai.configure(lm="gpt-4.1-mini", client=FakeRouter(responder=text_teacher))
    rows = reviews(400, seed=4)
    for r in rows[:300]:
        del r["result"]                                  # only 100 rows carry labels
    b = sentiment.bake(rows, teacher="gpt-4.1", compare_teacher=False, path=tmp_path / "h", **OPTS)
    assert b.report.label_source == "the data's labels and teacher (hard)" and b.report.labeling["rows"] == 300


def test_without_labels_or_teacher_bake_says_what_to_do(tmp_path):
    rows = [{"text": "x"}] * 50
    with pytest.raises(BakeError, match="teacher="):
        sentiment.bake(rows, path=tmp_path / "n", **OPTS)
    with pytest.raises(BakeError, match="not one of the answers"):
        sentiment.bake([{"text": "x", "result": "angry"}] * 50, path=tmp_path / "n2", **OPTS)


# ------------------------------------------------------------------ what a head can answer


def test_a_record_of_finite_answers_gets_one_answer_layer_each(tmp_path):
    rows = [{"text": r["text"], "result": Review(r["result"], r["spam"])} for r in reviews(600, seed=5)]
    b = review.bake(rows, path=tmp_path / "r", **OPTS)
    assert [f.name for f in b.fields] == ["result.sentiment", "result.spam"] and b.meta["architecture"] == "multi-head"
    got = review.using(lm=b)("The hotel was awful. BUY NOW at cheap-deals!")
    assert got == Review("negative", True)
    pred = review.using(lm=b)("The book was great.", all=True)
    assert set(pred.probabilities) == {"result.sentiment", "result.spam"} and pred.confidence > 0.5
    assert load(b.path).predict([{"text": "The book was great."}])[0]["result.spam"] is False


def test_open_ended_outputs_tools_and_optionals_are_refused_with_the_way_forward(tmp_path):
    @ai
    def summary(text: str) -> str:
        """Summarize."""

    def lookup(q: str) -> str:
        """Look up."""
        return q

    @ai(tools=[lookup])
    def tooly(text: str) -> Literal["a", "b"]:
        """Pick."""

    @ai
    def extra(text: str) -> Literal["a", "b"]:
        """Pick."""
        why: str = _ai["why"]
        return _ai

    with pytest.raises(BakeError, match="method='sft'"):
        summary.bake([{"text": "x", "result": "y"}] * 20, path=tmp_path / "1", **OPTS)
    with pytest.raises(BakeError, match="uses tools"):
        tooly.bake([{"text": "x", "result": "a"}] * 20, path=tmp_path / "2", **OPTS)
    with pytest.raises(BakeError, match="why"):
        extra.bake([{"text": "x", "result": "a", "why": "z"}] * 20, path=tmp_path / "3", **OPTS)


def test_bake_needs_torch_only_when_used():
    import subprocess
    import sys
    code = ("import sys; sys.modules['torch'] = None; import functai; "
            "from functai.bake import Baked, bake; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ})
    assert out.stdout.strip() == "ok", out.stderr


# ------------------------------------------------------------------ saving a program with its weights


def test_a_program_on_baked_weights_saves_loads_and_verifies(baked, tmp_path, monkeypatch):
    import importlib
    import sys
    import textwrap
    root = tmp_path / "proj"
    root.mkdir()
    (root / "baked_prog.py").write_text(textwrap.dedent(f"""
        from typing import Literal
        import functai
        from functai import ai

        FAST = functai.bake.load({str(baked.path)!r})


        @ai
        def careful(text: str) -> Literal["positive", "negative", "neutral"]:
            \"\"\"Judge the sentiment carefully.\"\"\"


        @ai(lm=FAST, escalate_to=careful, escalate_below=0.9)
        def sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
            \"\"\"The sentiment of the review.\"\"\"
    """))
    monkeypatch.syspath_prepend(str(root))
    prog = importlib.import_module("baked_prog")
    report = functai.check(prog.sentiment)
    assert report.ok, report
    assert {"torch", "transformers", "safetensors"} <= set(report.requirements)
    assert "on baked:sentiment" in repr(report)
    target = tmp_path / "saved"
    functai.save(prog.sentiment, target)
    assert (target / "models" / "sentiment" / "baked.json").exists()
    manifest = json.loads((target / "functai.json").read_text())
    node = manifest["nodes"]["baked_prog:sentiment"]["ai"]
    assert node["settings"]["lm"] == {"baked": "sentiment"}
    assert node["settings"]["escalate_to"] == {"node": "baked_prog:careful"}
    assert node["fingerprints"]["answers"] and "positive" in node["fingerprints"]["answers"][0]["result"]

    del sys.modules["baked_prog"]
    r = FakeRouter(responder=lambda req: "<result>\nneutral\n</result>")
    functai.configure(lm="gpt-4.1-mini", client=r)
    loaded = functai.load(target, trust=True)
    assert loaded("The pizza was amazing.") == "positive" and not r.requests       # the weights answered
    assert loaded._settings["lm"].path == (target / "models" / "sentiment").resolve()
    assert loaded.using(escalate_below=1.0)("Something happened.", all=True).escalated   # unsure: careful answers
    assert "Judge the sentiment carefully." in r.requests[-1].system
    assert functai.verify(target, trust=True, fresh=False).ok

    weights = target / "models" / "sentiment" / "model" / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-8] + b"\0" * 8)
    with pytest.raises(functai.LoadRefused, match="changed since it was saved"):
        functai.load(target, trust=True)


# ------------------------------------------------------------------ generative students


def _tiny_chat_model(path):
    """A 2-layer chat model with random weights and Qwen3.5's tokenizer and chat template."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
                      tie_word_embeddings=True, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                      pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def _have_qwen_tokenizer() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
        return True
    except Exception:
        return False


@ai
def capital(country: str) -> str:
    """The capital city of the country."""


@pytest.mark.skipif(not _have_qwen_tokenizer(), reason="Qwen3.5's tokenizer is not in the local cache")
def test_a_generative_student_learns_the_layouts_reply_and_keeps_its_layout(tmp_path):
    from functai.bake.sft import chat_messages, example_ids, prompt_ids, template_kwargs
    student = _tiny_chat_model(tmp_path / "tiny")
    rows = [{"country": c, "result": p} for c, p in [("France", "Paris"), ("Japan", "Tokyo"), ("Peru", "Lima")]] * 5
    b = capital.bake(rows, test=rows[:3], method="sft", student=student, device="cpu", epochs=2, lora=False,
                     path=tmp_path / "b", local_files_only=True, log=None)
    assert b.kind == "generative" and b.meta["layout"]["name"] == "functai_xml"
    h = b.report.training["history"]
    assert len(h) == 2 and h[1]["train_loss"] < h[0]["train_loss"]
    # the training target is the layout's reply, and the loss starts after the prompt
    tok = b._tokenizer
    request = capital.using(lm=b).render("Japan")
    msgs = chat_messages(lm15.serde.request_to_dict(request))
    assert msgs[0]["role"] == "system" and "Reply in exactly this form" in msgs[0]["content"]
    full, start = example_ids(tok, msgs, "<result>\nTokyo\n</result>", template_kwargs(tok))
    assert full[:start] == prompt_ids(tok, msgs, template_kwargs(tok))
    assert tok.decode(full[start:]).startswith("<result>\nTokyo\n</result>")
    # a later template on the function does not change what the student reads
    templated = capital.using(template=[functai.system("Answer. {instruction}"), functai.user("{country}")])
    assert templated.using(lm=b).render("Japan").system == request.system
    # it runs through functai (an untrained 2-layer model's reply is noise; the path is what is tested)
    try:
        capital.using(lm=b, retries=0)("Japan")
    except Exception as exc:  # noqa: BLE001 — an unreadable reply is expected from random weights
        assert "parse" in str(exc) or "Refusal" in type(exc).__name__
    assert functai.inspect_history(1)[0].model == "baked:capital"
    reloaded = load(b.path)
    assert reloaded.kind == "generative" and reloaded.meta["max_new_tokens"] >= 16
