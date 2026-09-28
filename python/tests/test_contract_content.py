"""contract/cases/content and contract/cases/saw: which values a call record
keeps (calls.md, *Content*), and what a call saw (calls.md, *Saw*).

content/: the harness defines a real AI function with the case's fields
(its own ``log_content``), sets the host's layers as a program would (a
``with configure(...)`` block, ``configure(...)``, the environment), asks
the call log which fields it keeps, and writes the case's whole record
through the call log's own restriction."""

import pytest

import functai
from functai import _ai, ai, calllog
from contract_support import assert_valid, case_files, load, validator

CONTENT = case_files("content")
SAW = case_files("saw")
CALL = validator("call")


def ci_log(job: int) -> str:
    """The CI log of a job."""
    return "upload_test: timeout after 30 s"


def triage_build(fields, own):
    """An AI function with the case's fields: two inputs, an output before the
    answer, reasoning added (module="cot") and, with ``calls``, a tool."""
    settings = {"module": "cot"}
    if "calls" in fields["added"]:
        settings["tools"] = [ci_log]
    if own is not None:
        settings["log_content"] = own

    def triage_build(transcript: str, question: str) -> str:
        """Is the build broken for a real reason?"""
        summary: str = _ai
        return _ai

    return ai(**settings)(triage_build)


@pytest.fixture
def environment(monkeypatch):
    def set_(value):
        if value is None:
            monkeypatch.delenv(calllog.ENV_CONTENT, raising=False)
        else:
            monkeypatch.setenv(calllog.ENV_CONTENT, value)
    return set_


@pytest.mark.parametrize("path", CONTENT, ids=lambda p: p.stem)
def test_content_case(path, environment):
    case = load(path)
    fields, expect = case["fields"], case["expect"]
    environment(case["environment"])
    layers = {x["where"]: x["log_content"] for x in case["layers"]}
    assert len(layers) == len(case["layers"])
    try:
        fn = triage_build(fields, layers.get("own"))
        if "configure" in layers:
            functai.configure(log_content=layers["configure"])
        block = functai.configure(log_content=layers["block"]) if "block" in layers else None
    except functai.LogContentError as err:
        assert {"refuses": err.code, "field": err.field} == expect
        return
    assert "refuses" not in expect, "a setting that should refuse was taken"
    ins, outs, added = fn._fields()
    assert (ins, outs, added) == (fields["inputs"], fields["outputs"], fields["added"])
    if block is not None:
        with block:
            kept = calllog.kept_fields(ins, outs, added, [s.get("log_content") for _w, s in calllog._layers(fn)])
    else:
        kept = calllog.kept_fields(ins, outs, added, [s.get("log_content") for _w, s in calllog._layers(fn)])
    written = calllog.restrict(case["record"], ins, outs, kept)
    assert_valid(CALL, written, path.stem)
    assert written == expect["record"]


def test_a_block_holds_every_call_inside_it(fake, tmp_path, environment):
    """The layers as calls meet them: a host's block keeps the transcript out of
    every call inside it, whatever the function's own setting says."""
    environment(None)
    fake("<reasoning>\nflaky\n</reasoning>\n<summary>\nA flaky test.\n</summary>\n<result>\nno\n</result>")
    fn = triage_build({"added": ["reasoning"]}, {"transcript": True})
    with functai.configure(log_calls=tmp_path, log_content={"transcript": False}):
        fn("Ana: red again.", "Real?")
    [rec] = [__import__("json").loads(line) for f in tmp_path.rglob("*.jsonl") for line in f.read_text().splitlines()]
    assert_valid(CALL, rec)
    assert rec["content"] is False and rec["omitted"] == {"inputs": ["transcript"], "outputs": ["reasoning"]}
    assert rec["inputs"] == {"question": "Real?"} and rec["outputs"] == {"summary": "A flaky test.", "result": "no"}
    assert "Ana" not in __import__("json").dumps(rec)
    assert all(not {"request", "response", "request_hash"} & set(ex) for ex in rec["exchanges"])


@pytest.mark.parametrize("path", SAW, ids=lambda p: p.stem)
def test_saw_case(path):
    case = load(path)
    if case["kind"] == "shown":
        pytest.skip("saw/ shown: the turn an entry stands for; Python does not show earlier calls again from the "
                    "call log yet (contract/README.md: languages that replay context)")
    for rec in case["records"]:
        assert_valid(CALL, rec, path.stem)
    for q in case["queries"]:
        try:
            got = {"saw": calllog.saw(q["call"], case["records"])}
        except functai.SawError as err:
            got = {"unknown": err.code, "call": err.call}
        assert got == q["expect"], q
        try:
            calllog.check_kept(q["call"], case["records"])
        except functai.SawError as err:
            keeps = {"refuses": err.code, "call": err.call}
        else:
            keeps = {"ok": True}
        assert keeps == q["keeps"], q


def test_the_contract_has_content_and_saw_cases():
    assert len(CONTENT) >= 20 and len(SAW) >= 16
