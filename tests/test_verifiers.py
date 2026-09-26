"""The verifiers harness and the rows taskset, offline: units, then real
vf-eval runs against a fake model server. Skipped without verifiers."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers.v1")

import lmcc  # noqa: E402

import functai_verifiers as fv  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from verifiers_program import sentiment  # noqa: E402


def config(**kw):
    return fv.FunctaiHarnessConfig(id="functai-verifiers", program="verifiers_program:sentiment", **kw)


def test_the_program_runs_on_its_layout_replayed_verbatim():
    assert fv.load_program(config()).adapter.name == "functai_xml"
    fn = fv.load_program(config(adapter="line"))
    assert fn.adapter.name == "sentiment_line" and fn.adapter.replay == "verbatim"
    assert fv.load_program(config(adapter="chat")).adapter.name == "functai_chat"
    with pytest.raises(KeyError, match="line"):
        fv.load_program(config(adapter="nope"))


def test_an_adapter_can_be_an_lmcc_json_file(tmp_path):
    from verifiers_program import ADAPTERS
    path = tmp_path / "a.json"
    path.write_text(json.dumps(ADAPTERS["line"].dump()))
    assert fv.load_program(config(adapter=str(path))).adapter.name == "sentiment_line"


def test_inputs_travel_in_the_user_turn():
    assert fv.decode_inputs(sentiment, fv.user_turn(text="great")) == {"text": "great"}
    assert fv.decode_inputs(sentiment, "plain text works for one text input") == {
        "text": "plain text works for one text input"}


def test_strip_reasoning_removes_an_inline_think_block():
    from verifiers.v1.types import AssistantMessage, Response

    def response(content):
        return Response(id="r", created=0, model="m", finish_reason="stop", message=AssistantMessage(content=content))
    assert fv.strip_reasoning(response("<think>hm</think>\nSentiment: positive")).message.content == "Sentiment: positive"
    assert fv.strip_reasoning(response("Sentiment: positive")) is None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def fake_model():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(HERE / "fake_openai_rows.py"), str(port)])
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/v1"
    proc.terminate()


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    words = ["great", "awful", "okay", "loved", "hated", "fine", "amazing", "terrible", "average"]
    path = tmp_path_factory.mktemp("rows") / "rows.jsonl"
    labels = {"great": "positive", "loved": "positive", "amazing": "positive", "awful": "negative",
              "hated": "negative", "terrible": "negative"}
    path.write_text("\n".join(json.dumps({"text": f"The {thing} was {w}.", "result": labels.get(w, "neutral")})
                              for thing in ("film", "meal") for w in words))
    return path


def _vf_eval(fake_model, rows, tmp_path, *extra):
    exe = Path(sys.executable).parent / "vf-eval"
    run = subprocess.run(
        [str(exe), "functai-rows", "--env.taskset.program", "verifiers_program:sentiment",
         "--env.taskset.data", str(rows), "--env.agent.harness.id", "functai-verifiers",
         "--env.agent.harness.program", "verifiers_program:sentiment", *extra,
         "--model", "fake", "--client.base-url", fake_model, "--client.api-key-var", "FAKE_KEY",
         "--num-tasks", "18", "--no-push", "--no-rich", "--output-dir", str(tmp_path)],
        env={**os.environ, "FAKE_KEY": "x", "PYTHONPATH": str(HERE)}, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    return [json.loads(line) for line in next(tmp_path.rglob("traces.jsonl")).read_text().splitlines()]


@pytest.mark.parametrize("adapter", [None, "line"])
def test_vf_eval_rollouts_score_and_record_each_turn(fake_model, rows, adapter, tmp_path):
    from verifiers.v1.trace import Trace
    extra = ["--env.agent.harness.adapter", adapter] if adapter else []
    records = _vf_eval(fake_model, rows, tmp_path, *extra)
    assert len(records) == 18 and all(r["ok"] for r in records)
    traces = [r["traces"][0] for r in records]
    assert all(len(Trace.model_validate(t).branches) == 1 for t in traces)
    recorded = [t["info"]["functai"]["turns"][0] for t in traces]
    unreadable = [x for x in recorded if "refusal" in x]
    assert unreadable and len(unreadable) < len(recorded)             # the fake's non-answers, recorded
    for t, x in zip(traces, recorded):                               # the fake is right when it answers
        assert t["rewards"]["agrees"]["score"] == (0.0 if "refusal" in x else 1.0)
    replies = [m for t in traces for m in Trace.model_validate(t).branches[0].messages if m.role == "assistant"]
    assert all(m.reasoning_content is None for m in replies)         # the teacher's thinking was dropped


def test_teacher_rollouts_export_as_sft_rows(fake_model, rows, tmp_path):
    _vf_eval(fake_model, rows, tmp_path)
    out = fv.sft_rows(next(tmp_path.rglob("traces.jsonl")))
    assert out and len(out) < 18                                      # unreadable turns skipped
    for row in out:
        assert [m["role"] for m in row["prompt"]] == ["system", "user"]
        assert row["completion"][0]["role"] == "assistant" and "<result>" in row["completion"][0]["content"]
        assert "reasoning_content" not in row["completion"][0]


_ = lmcc


def test_an_environment_package_carries_the_saved_program_and_runs(fake_model, rows, tmp_path):
    import tomllib
    from functai.bake import prime
    data = [json.loads(line) for line in rows.read_text().splitlines()]
    root = prime.env_package(sentiment, data, tmp_path / "env", name="sentiment-rows")
    assert (root / "sentiment_rows" / "saved" / "functai.json").exists()
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert "functai[prime]" in project["project"]["dependencies"]
    exe = Path(sys.executable).parent / "vf-eval"
    run = subprocess.run(
        [str(exe), "sentiment-rows", "--env.agent.harness.id", "functai-verifiers",
         "--env.agent.harness.program", "sentiment_rows.program:sentiment",
         "--model", "fake", "--client.base-url", fake_model, "--client.api-key-var", "FAKE_KEY",
         "--num-tasks", "6", "--no-push", "--no-rich", "--output-dir", str(tmp_path / "out")],
        env={**os.environ, "FAKE_KEY": "x", "PYTHONPATH": str(root)},       # the test's own code is not on the path
        capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-3000:]
    records = [json.loads(x) for x in next((tmp_path / "out").rglob("traces.jsonl")).read_text().splitlines()]
    assert len(records) == 6 and all(r["ok"] for r in records)

    text = prime.config(sentiment, env="me/sentiment-rows", model="Qwen/Qwen3.5-0.8B", loss="sft",
                        teacher="deepseek/deepseek-v4.1-flash", teacher_sampling={"reasoning_effort": "low"})
    cfg = tomllib.loads(text)
    assert cfg["loss"] == "sft" and cfg["rollouts_per_example"] == 1 and cfg["teacher"]["model"]
    assert cfg["env"][0]["harness"] == {"id": "functai-verifiers", "program": "sentiment_rows.program:sentiment"}
    with pytest.raises(ValueError, match="teacher"):
        prime.config(sentiment, env="me/x", model="m", loss="sft")
