"""A call tree's log of events through real calls (contract/streaming.md):
one numbering per tree, the kept form given to observers and journals, the
required journal's barriers, and a follower reading what a store kept."""

import json
import warnings

import lm15
import pytest

import functai
from functai import _ai, ai, calllog, eventlog, module
from functai.errors import EventRefused
from contract_support import assert_valid, validator

EVENT = validator("event")
CALL = validator("call")
XML = "<result>\n{}\n</result>"


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for var in (calllog.ENV_FOLDER, calllog.ENV_CONTENT, calllog.ENV_CALLER):
        monkeypatch.delenv(var, raising=False)


def records(folder):
    return [json.loads(line) for f in sorted(folder.rglob("*.jsonl")) for line in f.read_text().splitlines()]


@ai
def haiku(topic: str) -> str:
    """A haiku about the topic."""


@ai(module="cot")
def triage(transcript: str, question: str) -> str:
    """Is the build broken for a real reason?"""
    summary: str = _ai
    return _ai


def lookup(order: str) -> str:
    """Where an order is."""
    return f"{order} is in Leeds."


@ai(tools=[lookup])
def helper(question: str) -> str:
    """Help with the order."""


class FlakyStore(eventlog.MemoryStore):
    """A store whose appends fail when ``fails(event)`` says how: "down" (no
    answer, nothing kept), "lost" (kept, the answer lost), "refused"."""

    def __init__(self, fails):
        super().__init__()
        self.fails = fails
        self.sent = []

    def append(self, event):
        self.sent.append(event["seq"])
        how = self.fails(event)
        if how == "down":
            raise ConnectionError("down")
        if how == "refused":
            raise EventRefused("event-conflict", "another writer has it")
        answer = super().append(event)
        if how == "lost":
            raise TimeoutError("the answer was lost")
        return answer

    def extend(self, events):
        """A batch meets the same failures: down or refused when any event would be, lost after keeping."""
        events = list(events)
        self.sent.extend(e["seq"] for e in events)
        hows = [self.fails(e) for e in events]
        if "down" in hows:
            raise ConnectionError("down")
        if "refused" in hows:
            raise EventRefused("event-conflict", "another writer has it",
                               event=eventlog.position(events[hows.index("refused")]))
        answer = super().extend(events)
        if "lost" in hows:
            raise TimeoutError("the answer was lost")
        return answer


def test_every_request_is_an_event_and_one_numbering_holds_the_tree(fake, tmp_path):
    fake("no tags", XML.format("Snow."))
    s = haiku.using(log_calls=tmp_path).stream("snow")
    events = [e.to_dict() for e in s.events()]
    for e in events:
        assert_valid(EVENT, e)
    assert [e["kind"] for e in events if e["kind"] != "text"] == ["started", "request", "retry", "request", "done"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["after"] is None and all(e["after"] == {"writer": 1, "seq": e["seq"] - 1} for e in events[1:])
    [rec] = records(tmp_path)
    assert_valid(CALL, rec)
    # law 8: the requests are the record's exchanges, in order
    assert len(rec["exchanges"]) == len([e for e in events if e["kind"] == "request"]) == 2
    assert rec["saw"] == [] and rec["program"]["interface"] == rec["program"]["signature"]
    assert all(ex["request_hash"].startswith("sha256:") for ex in rec["exchanges"])
    assert events[0]["program"] == rec["program"] and events[0]["inputs"] == rec["inputs"]


def test_a_stream_opened_inside_a_tree_shows_the_tree_s_numbers(fake):
    @module
    def blurb(topic: str) -> str:
        inner = haiku.stream(topic)
        seen.extend(inner.events())
        return inner.result

    seen = []
    fake(XML.format("Snow on the cedar."))
    outer = blurb.stream("snow")
    whole = list(outer.events())
    assert outer.result == "Snow on the cedar."
    tree = outer.call_id
    assert {e.tree for e in seen} == {tree} and seen[0].kind == "started" and seen[0].function == "haiku"
    assert seen[0].after is None and seen[0].seq == 2                      # law 7: its first event, the tree's seq
    assert [e.seq for e in seen] == [e.seq for e in whole if e.function == "haiku"]  # the outer stream sees them too
    plain = []
    fake(XML.format("Snow."))

    @module
    def quiet(topic: str) -> str:                                          # nobody watches the tree
        inner = haiku.stream(topic)
        plain.extend(inner.events())
        return inner.result

    quiet("snow")
    assert plain[0].seq == 2 and plain[0].after is None and plain[0].tree != plain[0].call


def test_observers_get_the_kept_form(fake):
    seen, host = [], []
    fake([lm15.ThinkingPart("The transcript says flaky."),
          lm15.TextPart("<reasoning>\nflaky\n</reasoning>\n<summary>\nA flaky test.\n</summary>\n<result>\nno\n"
                        "</result>")])
    quiet = triage.using(observers=[seen], log_content={"transcript": False})
    with functai.configure(observers=[host]):
        quiet("Ana: it is red again.", "Real?")
    eventlog.drain()
    assert [e["kind"] for e in seen] == [e["kind"] for e in host]         # a program's observer beside the host's
    for e in seen:
        assert_valid(EVENT, e)
    started = seen[0]
    assert started["content"] is False and started["inputs"] == {"question": "Real?"}
    assert started["omitted"] == {"inputs": ["transcript"], "outputs": ["reasoning"]}
    assert not [e for e in seen if e["kind"] == "thinking"]
    assert {e["field"] for e in seen if e["kind"] == "text"} == {"summary", "result"}
    assert "Ana" not in json.dumps(seen)
    assert seen[0]["after"] is None and all(b["after"] == {"writer": 1, "seq": a["seq"]} for a, b in zip(seen, seen[1:]))
    assert seen[-1]["kind"] == "done" and seen[-1]["value"] == "no"


def test_an_observer_that_fails_warns_once_and_gets_nothing_more(fake):
    calls = []

    def broken(event):
        calls.append(event)
        raise RuntimeError("the socket closed")

    fake(responder=lambda req: XML.format("Snow."))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with functai.configure(observers=[broken]):
            assert haiku("snow") == "Snow."
            assert haiku("rain") == "Snow."
        eventlog.drain()
    assert len(calls) == 1 and len([w for w in caught if "observer" in str(w.message)]) == 1


def test_a_best_effort_journal_keeps_whole_trees_and_a_follower_reads_them(fake):
    store = functai.MemoryStore()

    @module
    def support(message: str) -> str:
        return haiku(message)

    fake(XML.format("Snow."))
    with functai.configure(journal=store):
        s = support.stream("snow")
        assert s.result == "Snow."
    eventlog.drain()
    [tree] = store.trees()
    assert tree == s.call_id and store.finished(tree)
    kept = store.read(tree)
    for e in kept:
        assert_valid(EVENT, e)
    reader = functai.Follower()
    assert [reader.receive(e) for e in kept] == ["kept"] * len(kept)
    watched = eventlog.Replay()
    for e in s.events():
        watched.apply(e.to_dict())
    assert reader.state(tree) == watched.state


def test_a_best_effort_journal_that_fails_never_stops_the_call(fake):
    store = FlakyStore(lambda e: "down")
    fake(XML.format("Snow."))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with functai.configure(journal=store):
            assert haiku("snow") == "Snow."
        eventlog.drain()
    assert [w for w in caught if "best-effort journal" in str(w.message)]


def test_a_required_journal_that_does_not_keep_the_start_runs_nothing(fake, tmp_path):
    store = FlakyStore(lambda e: "down")
    router = fake(XML.format("Snow."))
    with functai.configure(journal=functai.Journal(store, required=True), log_calls=tmp_path):
        with pytest.raises(functai.JournalError) as err:
            haiku("snow")
    assert router.requests == []                                          # the code did not run
    assert err.value.code == "journal-end" and err.value.journal == "unknown"
    assert err.value.outcome.error.code == "journal-barrier"
    assert err.value.settle() == "not-kept"
    [rec] = records(tmp_path)
    assert_valid(CALL, rec)
    assert rec["error"]["code"] == "journal-barrier" and rec["journal"] == "unknown"


def test_a_required_journal_keeps_a_tool_call_before_the_tool_runs(fake):
    ran = []

    def cancel(order: str) -> str:
        """Cancel an order."""
        ran.append(order)
        return "cancelled"

    @ai(tools=[cancel])
    def agent(request: str) -> str:
        """Do what is asked."""

    store = FlakyStore(lambda e: "down" if e["kind"] == "tool_call" else None)
    fake([lm15.ToolCallPart(id="c1", name="cancel", input={"order": "B-2210"})], XML.format("Done."))
    with functai.configure(journal=functai.Journal(store, required=True)):
        with pytest.raises(functai.JournalError) as err:
            agent("Cancel B-2210.")
    assert ran == []                                                      # the tool did not run
    assert err.value.code == "journal-end"                                # nothing after the tool call is kept
    assert err.value.outcome.error.code == "journal-barrier"


def test_a_required_journal_that_cannot_confirm_the_end_keeps_the_outcome(fake, tmp_path):
    store = FlakyStore(lambda e: "lost" if e["kind"] == "done" else None)
    fake(responder=lambda req: XML.format("Snow."))
    with functai.configure(journal=functai.Journal(store, required=True, retries=0), log_calls=tmp_path):
        s = haiku.stream("snow")
        with pytest.raises(functai.JournalError) as err:
            s.result
    e = err.value
    assert e.code == "journal-end" and e.journal == "unknown" and e.outcome.get() == "Snow."
    assert e.event == {"writer": 1, "seq": store.read(s.call_id)[-1]["seq"]}
    assert e.settle() == "kept"                                           # it was kept; the answer was lost
    shown = []
    with pytest.raises(functai.JournalError):
        for x in s.events():
            shown.append(x.kind)
    assert shown[0] == "started" and "done" not in shown and "failed" not in shown   # readers never saw an end
    [rec] = records(tmp_path)
    assert rec["error"] is None and rec["outputs"] == {"result": "Snow."} and rec["journal"] == "unknown"

    refused = FlakyStore(lambda e: "refused" if e["kind"] == "done" else None)
    with functai.configure(journal=functai.Journal(refused, required=True)):
        with pytest.raises(functai.JournalError) as err:
            haiku("snow")
    assert err.value.journal == "refused" and err.value.outcome.get() == "Snow."


def test_a_required_journal_set_only_inside_a_tree_is_refused(fake):
    store = functai.MemoryStore()

    @module
    def support(message: str) -> str:
        with functai.configure(journal=functai.Journal(store, required=True)):
            return haiku(message)

    fake(XML.format("Snow."))
    with pytest.raises(functai.JournalError) as err:
        support("snow")
    assert err.value.code == "journal-scope" and store.trees() == []


def test_a_stateful_function_records_the_calls_it_was_shown(fake, tmp_path):
    chat = ai(stateful=True, log_calls=tmp_path)(haiku.__wrapped__)
    fake(responder=lambda req: XML.format("Snow."))
    chat("snow")
    chat("rain")
    first, second = records(tmp_path)
    assert first["saw"] == [] and second["saw"] == [{"call": first["id"], "steps": True}]
    calllog.check_kept(second["id"], [first, second])                      # the log keeps what showing it needs
    assert calllog.saw(second["id"], [first, second]) == second["saw"]


class Frame:
    def __repr__(self):
        return "<Frame>"


def test_a_value_with_no_json_form_is_named_described(fake, tmp_path):
    @module(log_calls=tmp_path)
    def summarize(frame) -> str:
        return "two rows"

    summarize(Frame())
    [rec] = records(tmp_path)
    assert_valid(CALL, rec)
    assert rec["inputs"] == {"frame": {"$type": "Frame", "$repr": "<Frame>"}}
    assert rec["described"] == {"inputs": ["frame"], "outputs": []}
