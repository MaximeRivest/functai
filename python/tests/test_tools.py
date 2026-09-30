"""Stage 4: tools that say what they do, asking before they run, waiting turns, resuming them.

Offline: a fake provider. The contract is contract/tools.md."""

import threading
import time

import lm15
import pytest

import functai
from functai import ai, calllog, module, stores
from functai.stores import MemoryConversations

XML = "<result>\n{}\n</result>"
written = []
calls_to_read = []


@functai.tool(effects="reads")
def read_note(name: str) -> str:
    """The text of one note."""
    calls_to_read.append(name)
    return f"text of {name}"


@functai.tool(effects="changes")
def write_note(name: str, text: str) -> str:
    """Replace a note's text."""
    written.append((name, text))
    return f"wrote {name}"


def undeclared(name: str) -> str:
    """A plain function: its effects are unknown."""
    written.append(("undeclared", name))
    return "done"


def results(req):
    """The tool results of this turn: after the last message that asks something (earlier turns of a
    conversation hold their own)."""
    asks = [i for i, m in enumerate(req.messages) if m.role == "user"
            and any(type(p).__name__ == "TextPart" for p in m.parts)]
    start = asks[-1] if asks else 0
    return [p for m in req.messages[start:] for p in m.parts if type(p).__name__ == "ToolResultPart"]


def result_text(part):
    content = part.content
    return "".join(getattr(c, "text", "") for c in content) if isinstance(content, (list, tuple)) else str(content)


def script(*steps):
    """A model that asks for these tool calls, one per reply, then answers with the last result's text."""
    def respond(req):
        n = len(results(req))
        if n < len(steps):
            name, args = steps[n]
            return [lm15.ToolCallPart(id=f"c{n + 1}", name=name, input=args)]
        return XML.format("done: " + result_text(results(req)[-1]))
    return respond


@pytest.fixture(autouse=True)
def clear():
    written.clear()
    calls_to_read.clear()
    yield


@ai(tools=[read_note, write_note])
def gardener(request: str) -> str:
    """Tidy the user's notes as they ask."""


TWO = (("read_note", {"name": "a.md"}), ("write_note", {"name": "b.md", "text": "x"}))


def test_a_tool_says_what_it_does():
    assert read_note.effects == "reads" and write_note.effects == "changes"
    assert read_note("x.md") == "text of x.md"                              # still the function
    assert functai.tools.effects_of(undeclared) is None
    assert functai.tools.effects_of(gardener) is None                       # an AI function whose tool changes things

    @ai
    def reader(q: str) -> str:
        """Read."""
    assert functai.tools.effects_of(reader) == "reads"
    with pytest.raises(ValueError):
        functai.tool(effects="deletes")(undeclared)


def test_no_approval_by_default(fake):
    fake(responder=script(*TWO))
    assert gardener("tidy") == "done: wrote b.md" and written == [("b.md", "x")]


def test_a_function_is_asked_about_tools_that_change_things(fake):
    r = fake(responder=script(*TWO))
    asked = []

    def ask(approval):
        asked.append(approval)
        return "not that file"

    assert gardener.using(approve=ask)("tidy") == "done: The person did not allow this call. Reason: not that file"
    assert written == [] and calls_to_read == ["a.md"]                      # reads ran, the change did not
    [a] = asked
    assert (a.name, a.invocation, a.effects, a.path) == ("write_note", 2, "changes", "gardener/write_note")
    assert "did not allow" in result_text(results(r.requests[-1])[-1])      # the model is told


def test_rules(fake):
    a = functai.Approval("c", 1, "t1", "refund", {}, "changes", "support/answer/refund")
    r = functai.Approval("c", 2, "t2", "order_status", {}, "reads", "support/answer/order_status")
    u = functai.Approval("c", 3, "t3", "mystery", {}, None, "support/mystery")
    ask = functai.tools.asks
    assert [ask("changes", x) for x in (a, r, u)] == [True, False, True]     # unknown counts as changes
    assert [ask("all", x) for x in (a, r, u)] == [True, True, True]
    assert [ask(["answer/refund"], x) for x in (a, r, u)] == [True, False, False]
    assert [ask(["order_status"], x) for x in (a, r, u)] == [False, True, False]
    assert not ask(None, a)
    for bad in ("sometimes", 3, [""]):
        with pytest.raises((TypeError, ValueError)):
            functai.configure(approve=bad)


def test_a_plain_call_with_a_rule_has_nobody_to_ask(fake):
    fake(responder=script(*TWO))
    with pytest.raises(functai.ApprovalError) as err:
        gardener.using(approve="changes")("tidy")
    assert err.value.code == "approval-required" and written == []
    assert err.value.approval.name == "write_note"


def test_a_stream_waits_for_an_answer_here(fake):
    fake(responder=script(*TWO))
    s = gardener.stream("tidy", approve="changes")
    for e in s.events():
        if e.kind == "approval":
            assert (e.name, e.invocation, e.to) == ("write_note", 2, "owner")
            s.approve(e)
            break
    assert s.result == "done: wrote b.md" and written == [("b.md", "x")]
    kinds = [e.kind for e in s.events()]
    assert kinds.index("approval") < kinds.index("approved") < len(kinds) - 1


def test_a_denied_stream_tells_the_model(fake):
    fake(responder=script(*TWO))
    s = gardener.stream("tidy", approve="all")
    threading.Thread(target=lambda: (time.sleep(0.2), s.approve(1), time.sleep(0.2), s.deny(2, "no"))).start()
    assert s.result.startswith("done: The person did not allow this call") and written == []


def test_tool_calls_are_numbered_and_calls_inside_a_tool_carry_it(fake, tmp_path):
    @ai
    def read_tracking(log: str) -> str:
        """Where the parcel is."""

    @functai.tool(effects="reads")
    def order_status(order: str) -> str:
        """Where an order is."""
        return read_tracking(f"raw log of {order}")

    @ai(tools=[order_status], log_calls=tmp_path)
    def answer(message: str) -> str:
        """Answer."""

    def respond(req):
        if "Where the parcel is" in str(req.system):
            return XML.format("Leeds")
        n = len(results(req))
        if n < 2:
            return [lm15.ToolCallPart(id="call_1", name="order_status", input={"order": f"B-{n}"})]  # ids repeat
        return XML.format("in Leeds")

    fake(responder=respond)
    with functai.configure(log_calls=tmp_path):
        s = answer.stream("where?")
        assert s.result == "in Leeds"
    events = list(s.events())
    assert [e.invocation for e in events if e.kind == "tool_call"] == [1, 2]
    assert [e.invocation for e in events if e.kind == "tool_result"] == [1, 2]
    assert [e.invocation for e in events if e.kind == "started" and e.function == "read_tracking"] == [1, 2]
    recs = calllog.read(tmp_path)[0]
    assert sorted(r.get("invocation") for r in recs if r["program"]["name"] == "read_tracking") == [1, 2]
    top = next(r for r in recs if r["program"]["name"] == "answer")
    assert [st["kind"] for st in top["steps"]] == ["model", "tool", "model", "tool", "model"]


def test_a_required_journal_waits_only_before_tools_that_change_things(fake, monkeypatch):
    fake(responder=script(*TWO))
    barriers = []
    real = calllog.tool_barrier
    monkeypatch.setattr(calllog, "tool_barrier", lambda seq=None: (barriers.append(seq), real(seq))[1])
    with functai.configure(journal=functai.Journal(functai.MemoryStore(), required=True)):
        assert gardener("tidy") == "done: wrote b.md"
    assert len(barriers) == 1                                              # before write_note, not read_note


# ------------------------------------------------------------------ waiting turns (vignette 4, 11)


def test_a_turn_waits_and_is_answered_from_another_process(fake, tmp_path):
    r = fake(responder=script(*TWO))
    chat = gardener.conversation("g", store=tmp_path, approve="changes")
    with pytest.raises(functai.Waiting) as err:
        chat("tidy")
    [a] = err.value.approvals
    assert (a.name, a.input, a.path) == ("write_note", {"name": "b.md", "text": "x"}, "gardener/write_note")
    assert chat.turns[-1].state == "waiting" and written == []
    asked = len(r.requests)
    stores._folders.clear()                                                 # another process, after a restart
    later = gardener.conversation("g", store=tmp_path, approve="changes")
    turn = later.turns[-1]
    assert turn.approve(turn.waiting[0], by="maxime") == "done: wrote b.md"
    assert len(r.requests) - asked == 1                                    # no model answer paid for twice
    assert written == [("b.md", "x")] and calls_to_read == ["a.md"]       # nothing run twice
    done = later.turns[-1]
    assert done.state == "done" and done.result == "done: wrote b.md"
    kinds = [(e["kind"], e["writer"]) for e in done.events()]
    assert kinds.count(("started", 1)) == 1 and ("started", 2) not in kinds  # one log, continued by writer 2
    assert kinds[-1] == ("done", 2) and ("approved", 2) in kinds


def test_a_waiting_turn_can_be_denied_or_abandoned(fake):
    fake(responder=script(*TWO))
    chat = gardener.conversation(approve="changes")
    with pytest.raises(functai.Waiting):
        chat("tidy")
    assert chat.turns[-1].deny(reason="not today") == "done: The person did not allow this call. Reason: not today"
    with pytest.raises(functai.Waiting):
        chat("again")
    with pytest.raises(functai.ConversationError) as err:
        chat("a third")                                                     # the head waits: answer it first
    assert err.value.code == "conversation-busy"
    chat.turns[-1].abandon()
    assert chat.turns[-1].state == "abandoned" and written == []


def test_a_tool_that_may_have_run_is_never_run_again_on_its_own(fake):
    r = fake(responder=script(*TWO))
    chat = gardener.conversation("crash")
    chat("tidy")                                                            # ran to its end: every record kept
    records = chat.store.read("crash")
    # the process died just after write_note started: what a store keeps of it then
    cut = next(i for i, rec in enumerate(records) if rec["kind"] == "tool" and rec["state"] == "started")
    past = calllog._iso(time.time() - 60)
    kept = [{k: v for k, v in rec.items() if k != "seq"} for rec in records[:cut + 1]]
    for rec in kept:
        if rec["kind"] == "lease":
            rec["until"] = past
    crashed = MemoryConversations()
    crashed.append("crash", kept)
    written.clear()
    asked = len(r.requests)
    chat2 = gardener.conversation("crash", store=crashed)
    turn = chat2.turns[-1]
    assert turn.state == "interrupted"
    [u] = turn.unfinished
    assert (u["name"], u["invocation"]) == ("write_note", 2)
    with pytest.raises(functai.ConversationError) as err:
        turn.resume()
    assert err.value.code == "turn-unfinished"
    assert turn.resume(results={2: "wrote b.md (checked by hand)"}) == "done: wrote b.md (checked by hand)"
    assert written == [] and len(r.requests) - asked == 1                  # not rerun; one new model answer
    again = gardener.conversation("crash2", store=MemoryConversations())
    assert again is not None


def test_a_pause_three_levels_down(fake):
    @functai.tool(effects="changes")
    def refund(order: str, amount: float) -> str:
        """Refund part of an order."""
        written.append(("refund", order))
        return "refunded"

    @ai(tools=[refund])
    def answer(message: str) -> str:
        """Answer."""

    @ai
    def topic(message: str) -> str:
        """What is it about?"""

    @module
    def support(message: str) -> str:
        return answer(message + " / " + topic(message))

    def respond(req):
        if "What is it about" in str(req.system):
            return XML.format("shipping")
        return script(("refund", {"order": "B-2210", "amount": 12.5}))(req)

    r = fake(responder=respond)
    chat = support.conversation(approve=["answer/refund"])
    with pytest.raises(functai.Waiting) as err:
        chat("late parcel")
    assert err.value.approvals[0].path == "support/answer/refund"
    asked = len(r.requests)
    assert chat.turns[-1].approve() == "done: refunded"
    assert len(r.requests) - asked == 1 and written == [("refund", "B-2210")]   # topic not asked again


# ------------------------------------------------------------------ what is written passes the contract's schemas


def test_what_a_waiting_and_resumed_turn_writes_passes_the_schemas(fake, tmp_path):
    import json as _json
    from contract_support import assert_valid, validator
    fake(responder=script(*TWO))
    functai.configure(log_calls=tmp_path / "log")
    chat = gardener.conversation("schemas", store=tmp_path / "store", approve="changes")
    with pytest.raises(functai.Waiting):
        chat("tidy")
    chat.turns[-1].approve()
    conv, event, call = validator("conversation"), validator("event"), validator("call")
    for line in (tmp_path / "store" / "conversations" / "schemas.jsonl").read_text().splitlines():
        rec = _json.loads(line)
        assert_valid(conv, rec, rec["kind"])
    for path in (tmp_path / "store" / "trees").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            e = _json.loads(line)
            assert_valid(event, e, e["kind"])
    for rec in calllog.read(tmp_path / "log")[0]:
        assert_valid(call, rec, rec["program"]["name"])
    kinds = [_json.loads(x)["kind"] for x in (tmp_path / "store" / "conversations" / "schemas.jsonl")
             .read_text().splitlines()]
    assert {"program", "turn", "lease", "reply", "tool", "waiting", "approval", "ended"} <= set(kinds)
