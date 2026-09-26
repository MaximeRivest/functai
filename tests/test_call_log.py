"""The call log: every call as a line of JSON, ratings, and rows with known answers.

The records are checked against contract/schema, the reading rules against
contract/cases (what a TypeScript implementation must also pass)."""

import json
import os
import socket
import stat
import threading
import warnings
from pathlib import Path
from typing import Literal

import lm15
import lmcc
import pytest

import functai
from conftest import FakeRouter
from functai import _ai, ai, calllog, module

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "contract"
TEAMS = ("shipping", "billing", "product")


def team_reply(req):
    text = str(req.messages[-1])
    label = "billing" if ("charged" in text or "refund" in text) else "product" if "broke" in text else "shipping"
    return f"<result>\n{label}\n</result>"


@pytest.fixture(autouse=True)
def no_env(monkeypatch, tmp_path):
    """No log unless a test asks for one; the default folder is a temporary one."""
    for var in (calllog.ENV_FOLDER, calllog.ENV_CONTENT, calllog.ENV_CALLER):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    calllog._warned.clear()


@pytest.fixture
def log(tmp_path):
    """A log folder, on for every call."""
    folder = tmp_path / "calls"
    functai.configure(log_calls=folder)
    return folder


@pytest.fixture
def schemas():
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource
    docs = {name: json.loads((CONTRACT / "schema" / f"{name}.schema.json").read_text()) for name in ("call", "rating")}
    registry = Registry().with_resources([(d["$id"], Resource.from_contents(d)) for d in docs.values()])
    return {name: jsonschema.Draft202012Validator(d, registry=registry) for name, d in docs.items()}


def lines(folder):
    return [json.loads(line) for f in sorted(Path(folder).rglob("*.jsonl")) for line in f.read_text().splitlines()]


def logged(folder):
    return [r for r in lines(folder) if "functai_call" in r]


def valid(schemas, records):
    for rec in records:
        kind = "call" if "functai_call" in rec else "rating"
        errors = sorted(schemas[kind].iter_errors(rec), key=lambda e: list(e.path))
        assert not errors, f"{kind} record breaks the schema: {errors[0].message} at {list(errors[0].path)}"
    return records


@ai
def team(message: str) -> Literal["shipping", "billing", "product"]:
    """Which team should answer this customer message?"""


# ------------------------------------------------------------------ off by default, on when asked


def test_nothing_is_logged_unless_asked_but_every_call_has_an_id(fake, tmp_path):
    fake(responder=team_reply)
    p = team("My parcel never came.", all=True)
    assert calllog._UUID.match(p.call_id) and p.call_id[14] == "7"          # a UUIDv7
    assert not (tmp_path / "xdg").exists()


def test_the_environment_turns_it_on_and_a_function_can_refuse(fake, monkeypatch, tmp_path):
    fake(responder=team_reply)
    monkeypatch.setenv(calllog.ENV_FOLDER, "1")
    team("My parcel never came.")
    default = tmp_path / "xdg" / "functai" / "calls"
    assert [r["outputs"] for r in logged(default)] == [{"result": "shipping"}]
    assert calllog.default_folder() == default

    private = team.using(log_calls=False)
    private("I was charged twice.")
    assert len(logged(default)) == 1

    monkeypatch.setenv(calllog.ENV_FOLDER, str(tmp_path / "elsewhere"))
    with functai.configure(log_calls=True):          # True: on, where the environment says
        team("The kettle broke.")
    assert len(logged(tmp_path / "elsewhere")) == 1


def test_log_settings_are_checked_when_given():
    with pytest.raises(TypeError, match="log_calls is a folder"):
        functai.configure(log_calls=3)
    with pytest.raises(ValueError, match="empty"):
        functai.configure(log_calls=" ")
    with pytest.raises(TypeError, match="log_content is True or False"):
        functai.configure(log_content="no")
    with pytest.raises(TypeError, match="caller is a dict"):
        functai.configure(caller="me")
    with pytest.raises(TypeError, match="JSON values"):
        functai.configure(caller={"when": object()})


# ------------------------------------------------------------------ what a call record holds


def test_a_call_record_holds_the_typed_call_and_its_exchange(fake, log, schemas):
    fake(responder=team_reply)
    p = team("I was charged twice for one order.", all=True)
    [rec] = valid(schemas, logged(log))
    assert rec["id"] == p.call_id == rec["root"] and rec["parent"] is None
    assert rec["program"]["name"] == "team" and rec["program"]["kind"] == "ai"
    assert rec["program"]["module"] == __name__ and rec["program"]["answer"] == "result"
    assert rec["program"]["version"] == team.version
    assert rec["program"]["signature"] == lmcc.signature_fingerprint(team.signature)
    assert rec["program"]["file"] == __file__
    assert rec["inputs"] == {"message": "I was charged twice for one order."}
    assert rec["outputs"] == {"result": "billing"} and "returned" not in rec
    assert rec["sizes"] == {"inputs": {"message": 36}, "outputs": {"result": 9}}
    assert rec["error"] is None and rec["model"] == "gpt-4.1-mini"
    assert rec["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    [ex] = rec["exchanges"]
    assert ex["provider"] == "openai" and ex["finish"] == "stop" and ex["cached"] is False
    assert "Which team should answer" in json.dumps(ex["request"]) and "billing" in json.dumps(ex["response"])
    assert rec["process"]["language"] == "python" and rec["process"]["pid"] == os.getpid()
    assert rec["caller"] == {}


def test_without_content_only_sizes_times_and_tokens(fake, log, schemas):
    fake("<result>\nnot a team\n</result>", responder=None)
    quiet = team.using(log_content=False, retries=0)
    with pytest.raises(lmcc.Refusal):
        quiet("my password is hunter2")
    [rec] = valid(schemas, logged(log))
    assert rec["content"] is False
    assert not {"inputs", "outputs", "returned", "probabilities"} & set(rec)
    assert rec["sizes"]["inputs"] == {"message": len('"my password is hunter2"')}
    assert rec["error"]["type"] == "Refusal" and rec["error"]["code"].startswith("parse-")
    assert "message" not in rec["error"]                     # it can quote the reply
    assert "hunter2" not in json.dumps(rec)


def test_failures_retries_and_the_cache_are_exchanges(fake, log, schemas):
    busy = lm15.RateLimitError("slow down", provider="openai")
    busy.retry_after = 0.001
    fake(busy, "<result>\nshipping\n</result>", "<result>\nbilling\n</result>")
    cached = team.using(cache_replies=True)
    assert cached("Where is my parcel?") == "shipping"
    assert cached("Where is my parcel?") == "shipping"          # from the cache
    first, second = valid(schemas, logged(log))
    assert [e.get("error", {}).get("type") for e in first["exchanges"]] == ["RateLimitError", None]
    assert first["usage"]["input_tokens"] == 3                  # only replies count
    assert second["exchanges"][0]["cached"] is True and second["exchanges"][0]["seconds"] == 0


def test_a_module_is_the_parent_of_the_calls_it_makes(fake, log, schemas):
    fake(responder=team_reply)

    @module
    def route_all(messages: list) -> list:
        return [team(m) for m in messages]

    assert route_all(["charged twice", "parcel late"]) == ["billing", "shipping"]
    a, b, parent = valid(schemas, logged(log))                 # children end first
    assert parent["program"]["kind"] == "module" and parent["outputs"] == {"result": ["billing", "shipping"]}
    assert parent["inputs"] == {"messages": ["charged twice", "parcel late"]}
    assert parent["exchanges"] == [] and parent["model"] is None and parent["usage"] == {}
    assert a["parent"] == b["parent"] == parent["id"] == a["root"] == parent["root"]
    assert parent["program"]["version"] == route_all.version


def test_a_tool_that_calls_an_ai_function_makes_a_child_call(fake, log, schemas):
    def which_team(message: str) -> str:
        """The team for a message."""
        return team(message)

    @ai(tools=[which_team])
    def helper(question: str) -> str:
        """Answer the customer."""

    call = lm15.ToolCallPart(id="c1", name="which_team", input={"message": "I was charged twice"})
    fake([call], "<result>\nbilling\n</result>", "<result>\nBilling will help you.\n</result>")
    helper("Who helps me?")
    child, outer = valid(schemas, logged(log))
    assert child["parent"] == outer["id"] and child["program"]["name"] == "team"
    assert len(outer["exchanges"]) == 2 and len(child["exchanges"]) == 1


def test_a_body_that_changes_the_answer_keeps_what_it_returned(fake, log, schemas):
    @ai
    def score(text: str) -> float:
        """A score between 0 and 1."""
        s: float = _ai
        return max(0.0, min(1.0, s))

    fake("<s>\n1.7\n</s>")
    assert score("great") == 1.0
    [rec] = valid(schemas, logged(log))
    assert rec["outputs"] == {"s": 1.7} and rec["returned"] == 1.0


def test_an_unreadable_reply_kept_as_a_turn_is_a_failure_with_empty_outputs(fake, log, schemas):
    fake("no tags at all")
    keep = team.using(on_unreadable="record", retries=0)
    p = keep("Where is it?", all=True)
    [rec] = valid(schemas, logged(log))
    assert p.refusal is not None and rec["outputs"] == {} and rec["error"]["type"] == "Refusal"


def test_a_measured_answer_carries_its_confidence_and_escalation(monkeypatch, log, schemas):
    from functai import models
    monkeypatch.setattr(models, "JUDGMENT_ONLY", models.JUDGMENT_ONLY | {"typesafe"})

    def measured(req):
        if req.model == "openai:gpt-4.1":                       # the model escalated to
            return "<result>\nbilling\n</result>"
        dist = {"shipping": 0.5, "billing": 0.3, "product": 0.2}
        return lm15.Response(id=None, model=req.model, message=lm15.Message.assistant(
            [lm15.DataPart(value={"result": "shipping"}, probabilities={"result": dist},
                           method="provider_classification")]),
            finish_reason="stop", usage=lm15.Usage(input_tokens=10, output_tokens=0, total_tokens=10))

    functai.configure(client=FakeRouter(responder=measured, provider="typesafe"))
    unsure = team.using(lm="typesafe:jev", escalate_to="openai:gpt-4.1", escalate_below=0.9)
    assert unsure("charged twice?") == "billing"
    [rec] = valid(schemas, logged(log))
    assert rec["escalated"] is True and rec["model"] == "openai:gpt-4.1"
    assert [(e["model"], e["provider"]) for e in rec["exchanges"]] == [("typesafe:jev", "typesafe"),
                                                                        ("openai:gpt-4.1", "openai")]
    sure = team.using(lm="typesafe:jev")
    sure("charged twice?")
    rec = valid(schemas, logged(log))[-1]
    assert rec["confidence"] == 0.5 and rec["probabilities"]["result"]["billing"] == 0.3


def test_values_without_a_json_form_are_described(fake, log, schemas):
    class Opaque:
        def __repr__(self):
            return "Opaque(" + "x" * 5000 + ")"

    @ai
    def echo(thing) -> str:
        """Repeat."""

    fake("<result>\nok\n</result>")
    echo(Opaque())
    [rec] = valid(schemas, logged(log))
    described = rec["inputs"]["thing"]
    assert described["$type"].endswith("Opaque") and len(described["$repr"]) == 2000


def test_a_huge_record_loses_its_messages_then_its_values(fake, log, schemas, monkeypatch):
    monkeypatch.setattr(calllog, "MAX_LINE", 3000)
    fake(responder=team_reply)
    team("x" * 1000)
    team("y" * 4000)
    small, huge = valid(schemas, logged(log))
    assert small["truncated"] is True and "request" not in small["exchanges"][0] and "inputs" in small
    assert huge["truncated"] is True and "inputs" not in huge and huge["sizes"]["inputs"]["message"] == 4002


def test_code_run_with_exec_is_logged_as_main(fake, log):
    fake(responder=team_reply)
    ns = {"ai": ai, "Literal": Literal}
    exec("@ai\ndef routed(message: str) -> Literal['shipping', 'billing']:\n    'Which team?'\n", ns)
    ns["routed"]("charged twice")
    [rec] = logged(log)
    assert rec["program"]["module"] == "__main__" and len(functai.calls(ns["routed"]).collect()) == 1


# ------------------------------------------------------------------ never in the way


def test_a_folder_that_cannot_be_written_warns_once_and_calls_go_on(fake, tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    fake(responder=team_reply)
    functai.configure(log_calls=blocked)
    with pytest.warns(UserWarning, match="calls go on, unlogged"):
        assert team("charged twice") == "billing"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert team("parcel late") == "shipping"                 # no second warning


def test_files_are_per_process_per_day_and_private(fake, log):
    fake(responder=team_reply)
    team("parcel late")
    [path] = list(log.rglob("*.jsonl"))
    assert calllog._DAY.match(path.parent.name)
    assert path.name.startswith(f"{socket.gethostname()}-{os.getpid()}-") and path.suffix == ".jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_threads_write_whole_lines(fake, log, schemas):
    fake(responder=team_reply)
    threads = [threading.Thread(target=lambda: [team("charged " * 200) for _ in range(20)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(valid(schemas, logged(log))) == 160


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_a_forked_child_writes_its_own_file(fake, log):
    fake(responder=team_reply)
    team("parcel late")
    pid = os.fork()
    if pid == 0:
        try:
            team("charged twice")
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    team("broke")
    assert len(list(log.rglob("*.jsonl"))) == 2
    assert {r["process"]["pid"] for r in logged(log)} == {os.getpid(), pid}


def test_ids_are_uuid7_in_order():
    ids = [calllog.new_id() for _ in range(5000)]
    assert ids == sorted(ids) and len(set(ids)) == 5000
    assert all(calllog._UUID.match(i) and i[14] == "7" and i[19] in "89ab" for i in ids[:50])


# ------------------------------------------------------------------ who called


def test_the_caller_comes_from_the_environment_and_blocks(fake, log, monkeypatch):
    fake(responder=team_reply)
    monkeypatch.setenv(calllog.ENV_CALLER, json.dumps({"kind": "notebook", "notebook": "/n/triage.md"}))
    with functai.configure(caller={"cell": "c3"}):
        team("parcel late")
    monkeypatch.setenv(calllog.ENV_CALLER, "{not json")
    with pytest.warns(UserWarning, match="not a JSON object"):
        team("parcel late")
    first, second = logged(log)
    assert first["caller"] == {"kind": "notebook", "notebook": "/n/triage.md", "cell": "c3"}
    assert second["caller"] == {}


def test_evaluation_and_optimization_calls_say_so(fake, log):
    fake(responder=team_reply)
    rows = [{"message": "charged twice", "result": "billing"}, {"message": "parcel late", "result": "shipping"},
            {"message": "it broke", "result": "product"}, {"message": "refund please", "result": "billing"}]
    ev = functai.evaluate(team, rows, num_threads=4)
    recs = logged(log)
    assert len(recs) == 4 and {r["caller"]["evaluation"] for r in recs} == {ev.run}
    copy = team.using()
    copy.opt(trainset=rows)
    opt = [r for r in logged(log)[4:]]
    assert opt and all("optimization" in r["caller"] for r in opt)
    assert len({r["caller"]["optimization"] for r in opt}) == 1
    team("parcel late")
    purposes = functai.calls(team).pull("purpose")
    assert set(purposes[:4]) == {"evaluation"} and set(purposes[4:-1]) == {"optimization"} and purposes[-1] == "use"


# ------------------------------------------------------------------ versions


def test_a_version_names_what_is_sent_besides_the_inputs():
    @ai
    def a(message: str) -> str:
        """Summarize."""

    @ai
    def b(message: str) -> str:
        """Summarize in one line."""

    v = a.version
    assert v.startswith("sha256:") and len(v) == 71 and a.version == v          # stable, and needs no model
    assert b.version != v                                                     # the instruction
    assert a.using(lm="gpt-4.1-nano", temperature=0.3).version == v           # where it runs is not the program
    assert a.using(adapter="json").version != v                               # the layout
    assert a.using(module="cot").version != v
    c = a.using()
    c.demos = [{"message": "a long text", "result": "short"}]
    assert c.version != v                                                     # the worked examples
    c.demos = []
    assert c.version == v


def test_a_module_version_follows_its_code_and_its_functions():
    @ai
    def inner(x: str) -> str:
        """Shorten."""

    @module
    def outer(x: str) -> str:
        return inner(x)

    v = outer.version
    assert outer.version == v
    inner.demos = [{"x": "long", "result": "short"}]
    assert outer.version != v                                                 # optimizing inside is a new version
    inner.demos = []
    assert outer.version == v


def test_a_saved_and_loaded_program_keeps_its_version_and_says_where_it_came_from(tmp_path, monkeypatch, schemas):
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "support_kit.py").write_text(
        "from typing import Literal\nfrom functai import ai\n\n\n"
        "@ai(temperature=0)\n"
        "def team(message: str) -> Literal['shipping', 'billing', 'product']:\n"
        '    """Which team should answer this customer message?"""\n')
    import importlib
    kit = importlib.import_module("support_kit")
    functai.save(kit.team, tmp_path / "saved")
    manifest = json.loads((tmp_path / "saved" / "functai.json").read_text())
    loaded = functai.load(tmp_path / "saved", trust=True)
    assert loaded.version == kit.team.version
    from functai import saved
    spec = kit.team._spec()
    plan, past = saved.probe_plan(kit.team, spec, kit.team._effective())
    sent = saved.request_fingerprint(saved.probe_request(kit.team, spec, plan, past, saved._sample_inputs(spec)))
    assert manifest["nodes"]["support_kit:team"]["ai"]["fingerprints"]["requests"][0] == sent   # the folder names it
    functai.configure(lm="gpt-4.1-mini", client=FakeRouter(responder=team_reply), log_calls=tmp_path / "log")
    loaded("charged twice")
    [rec] = valid(schemas, logged(tmp_path / "log"))
    assert rec["program"]["module"] == "support_kit"
    assert rec["program"]["saved"] == "sha256:" + __import__("hashlib").sha256(
        (tmp_path / "saved" / "functai.json").read_bytes()).hexdigest()


# ------------------------------------------------------------------ rating, and rows with known answers


def test_rate_writes_a_rating_next_to_the_call(fake, log, schemas):
    fake(responder=team_reply)
    p = team("My parcel never came, refund me.", all=True)
    r = functai.rate(p, "wrong", answer="shipping", note="A lost parcel is shipping.", reasons=["wrong team"])
    assert r["verdict"] == "wrong" and r["answer"] == "shipping" and r["call"] == p.call_id
    assert r["by"] == calllog._process_info()["user"]
    assert functai.rate(p, answer="shipping")["verdict"] == "wrong"           # a correction says wrong
    assert functai.rate(p.call_id, True)["verdict"] == "right"
    assert functai.rate({"call": p.call_id}, None)["verdict"] is None         # withdrawn
    valid(schemas, lines(log))
    with pytest.raises(TypeError, match="say whether the answer is right"):
        functai.rate(p)
    with pytest.raises(ValueError, match="needs no correction"):
        functai.rate(p, "right", answer="billing")
    with pytest.raises(ValueError, match="verdict is"):
        functai.rate(p, "meh")
    with pytest.raises(TypeError, match="rate what"):
        functai.rate("not an id", "right")


def test_rate_needs_a_log(fake):
    fake(responder=team_reply)
    p = team("x", all=True)
    with pytest.raises(ValueError, match="written to the call log"):
        functai.rate(p, "right")


def test_rated_calls_are_rows_evaluate_and_opt_take(fake, log):
    fake(responder=team_reply)
    a = team("My parcel never came, refund me.", all=True)        # says billing
    b = team("The kettle broke on day one.", all=True)            # says product
    c = team("Where is my order?", all=True)                      # says shipping
    team("Unrated message")
    functai.rate(a, "wrong", answer="shipping")
    functai.rate(b, "right")
    functai.rate(c, "wrong")                                       # no right answer: left out
    with pytest.warns(UserWarning, match="1 marked wrong without the right answer"):
        rows = functai.rated(team)
    got = rows.select("message", "result", "rating").collect().to_dicts()
    assert got == [{"message": "My parcel never came, refund me.", "result": "shipping", "rating": "wrong"},
                   {"message": "The kettle broke on day one.", "result": "product", "rating": "right"}]
    assert rows.columns[-7:] == ["rating", "rated_by", "origin", "disputed", "sample", "version", "call"]
    ev = functai.evaluate(team, rows)
    assert ev.score == 0.5                                         # the correction is what it gets wrong
    copy = team.using()
    copy.opt(trainset=rows)
    assert copy.demos


def test_rated_uses_the_current_signature_and_each_persons_latest(fake, log):
    fake(responder=team_reply)
    p = team("charged twice", all=True)

    @ai
    def team_v2(message: str, urgent: bool) -> Literal["shipping", "billing", "product"]:   # another signature
        """Which team?"""

    functai.rate(p, "right", by="ana")
    functai.rate(p, "wrong", answer="shipping", by="ben")
    functai.rate(p, "right", by="ben")                             # ben changes his mind
    rows = functai.rated(team).collect().to_dicts()
    assert [(r["result"], r["rated_by"], r["disputed"]) for r in rows] == [("billing", "ben", False)]
    functai.rate(p, "wrong", answer="product", by="ana")
    [row] = functai.rated(team).collect().to_dicts()
    assert (row["result"], row["rated_by"], row["disputed"]) == ("product", "ana", True)
    [row] = functai.rated(team, by="ben").collect().to_dicts()
    assert (row["result"], row["disputed"]) == ("billing", False)
    functai.rate(p, None, by="ana")
    functai.rate(p, None, by="ben")
    with pytest.raises(ValueError, match="no rated calls of team"):
        functai.rated(team)
    team_v2.__name__ = "team"                                      # same name, new inputs
    with pytest.raises(ValueError, match="no rated calls"):
        functai.rated(team_v2)


def test_calls_is_the_log_as_a_table(fake, log):
    fake(responder=team_reply)
    p = team("charged twice", all=True)
    team("parcel late")
    functai.rate(p, "right")
    t = functai.calls(team)
    assert t.columns[:2] == ["message", "pred_result"]
    got = t.select("message", "pred_result", "rating", "model").collect().to_dicts()
    assert got == [{"message": "charged twice", "pred_result": "billing", "rating": "right", "model": "gpt-4.1-mini"},
                   {"message": "parcel late", "pred_result": "shipping", "rating": None, "model": "gpt-4.1-mini"}]
    everything = functai.calls()
    assert everything.columns[:3] == ["program", "inputs", "outputs"]
    assert json.loads(everything.collect().to_dicts()[0]["inputs"]) == {"message": "charged twice"}
    assert len(functai.calls(team, since="1d").collect()) == 2
    with pytest.raises(ValueError, match="no calls"):
        functai.calls(team, since="2999-01-01")
    with pytest.raises(ValueError, match="since is"):
        functai.calls(team, since="last tuesday")


def test_reading_skips_partial_lines_and_unknown_records(fake, log):
    fake(responder=team_reply)
    team("parcel late")
    day = next(log.iterdir())
    (day / "other-writer.jsonl").write_text('{"functai_call": 2, "id": "x"}\n{"hello": 1}\n{"functai_call": 1, "id"')
    found, ratings = calllog.read(log)
    assert len(found) == 1 and ratings == []


# ------------------------------------------------------------------ the contract


def cases():
    return sorted((CONTRACT / "cases").glob("*.json"))


@pytest.mark.parametrize("path", cases(), ids=lambda p: p.stem)
def test_contract_case(path, schemas):
    case = json.loads(path.read_text())
    valid(schemas, case["records"])
    calls_ = [r for r in case["records"] if "functai_call" in r]
    ratings = [r for r in case["records"] if "functai_rating" in r]
    rows, left = calllog.rated_rows(calls_, ratings, **case["rated"])
    assert rows == case["expect"]["rows"]
    assert left == case["expect"]["left_out"]


def test_the_contract_has_cases():
    assert len(cases()) >= 5


def test_columns_about_a_call_never_hide_its_data(fake, log):
    @ai
    def describe(model: str, version: str) -> str:
        """One sentence about this product model and version."""

    fake("<result>\nA kettle.\n</result>")
    p = describe("K-100", "2", all=True)
    functai.rate(p, "right")
    got = functai.calls(describe).collect().to_dicts()[0]
    assert (got["model"], got["version"]) == ("K-100", "2")
    assert got["_model"] == "gpt-4.1-mini" and got["_version"] == describe.version
    row = functai.rated(describe).collect().to_dicts()[0]
    assert (row["model"], row["version"], row["_version"]) == ("K-100", "2", describe.version)
    assert functai.rated(describe).columns[-7:] == ["rating", "rated_by", "origin", "disputed", "sample",
                                                    "_version", "call"]
