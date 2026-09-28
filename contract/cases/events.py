"""The cases in events/: stream events as data, written from the rules in
../streaming.md. Three kinds, told apart by "kind":

- "replay": {"events": [...], "expect": {"views": [...], "finished"},
  "resume": [{"after": N, "seqs": [...]}]}. Reading the events in order,
  the view after each one (sections "Events", law 3, and "Replaying a
  stream"): for each call started so far, {"ended": null | "done" |
  "failed", "fields": {name: text so far}} (a field shown as sizes has
  the number of code points so far instead of text). ``finished``: the
  watched call ended. ``resume``: the seqs a reader that has events up to
  ``after`` gets.
- "stored": {"events": [whole events], "kept": {call id: {field: bool}},
  "expect": {"events": [stored events]}}. The stored form of each event
  ("The stored form"), given which fields of each call its log_content
  keeps (in its interface's order; how a setting becomes this map is
  content/'s job).
- "store": {"appends": [{"event", "expect": "kept" | "duplicate" |
  "event-conflict" | "event-gap" | "event-after-end" | "event-start"}],
  "expect": {"streams": {stream id: {"seqs": [...], "finished": bool}}}}.
  Appending each event in turn to one store ("The rules a store keeps").

Every event here passes schema/event.schema.json (make.py checks).
"""

import copy

from common import canonical

V = "sha256:" + "1" * 64
SIG = "sha256:" + "5" * 64
FIELD_KINDS = ("text", "thinking")


def cid(n):
    return f"01926c00-{n:04x}-7000-8000-000000000000"


def program(name, kind="ai", answer="result"):
    return {"name": name, "kind": kind, "module": "shop", "version": V, "signature": SIG, "answer": answer}


class Writer:
    """The events of one stream, numbered as they are made."""

    def __init__(self, watched: int, *, first_seq: int = 1):
        self.stream = cid(watched)
        self.seq = first_seq - 1
        self.names = {}
        self.events = []

    def ev(self, kind, n, **fields):
        self.seq += 1
        e = {"functai_event": 2, "kind": kind, "stream": self.stream, "seq": self.seq,
             "at": f"2026-09-28T10:00:{self.seq // 100:02d}.{(self.seq % 100) * 10000:06d}Z",
             "call": cid(n), "function": self.names[n], **fields}
        self.events.append(e)
        return e

    def started(self, n, name, inputs, *, parent=None, root=None, kind="ai", answer="result", saw=()):
        self.names[n] = name
        return self.ev("started", n, parent=cid(parent) if parent else None, root=cid(root or n),
                       program=program(name, kind, answer), inputs=inputs, content=True, saw=list(saw))

    def text(self, n, field, text, request=1, answer=True):
        return self.ev("text", n, field=field, answer=answer, request=request, text=text)

    def thinking(self, n, text, request=1):
        return self.ev("thinking", n, request=request, text=text)

    def tool_call(self, n, id_, name, input_):
        return self.ev("tool_call", n, id=id_, name=name, input=input_)

    def tool_result(self, n, id_, name, output):
        return self.ev("tool_result", n, id=id_, name=name, output=output)

    def retry(self, n, reason, wait=None):
        return self.ev("retry", n, reason=reason, wait=wait)

    def done(self, n, value):
        return self.ev("done", n, value=value)

    def failed(self, n, error):
        return self.ev("failed", n, error=error)


# ------------------------------------------------------------------ the rules: replay


def replay(events: list, stream: str) -> dict:
    calls, latest, views = {}, {}, []
    finished = False
    for e in events:
        c, kind = e["call"], e["kind"]
        if kind == "started":
            calls[c] = {"ended": None, "fields": {}}
            latest[c] = 0
        elif kind in FIELD_KINDS:
            if e["request"] > latest[c]:
                latest[c] = e["request"]
                calls[c]["fields"] = {}
            if kind == "text":
                fields = calls[c]["fields"]
                if "text" in e:
                    fields[e["field"]] = fields.get(e["field"], "") + e["text"]
                else:
                    fields[e["field"]] = fields.get(e["field"], 0) + e["size"]
        elif kind in ("retry", "tool_result"):
            calls[c]["fields"] = {}
        elif kind in ("done", "failed"):
            calls[c]["ended"] = kind
            if c == stream:
                finished = True
        views.append({"calls": copy.deepcopy(calls)})
    return {"views": views, "finished": finished}


def resume(events: list, after: int) -> list:
    return [e["seq"] for e in events if e["seq"] > after]


# ------------------------------------------------------------------ the rules: the stored form


def stored(e: dict, kept: dict, answers: dict) -> dict:
    keep = kept[e["call"]]
    if all(keep.values()):
        return copy.deepcopy(e)
    out = {k: copy.deepcopy(v) for k, v in e.items()}
    kind = e["kind"]
    if kind == "started":
        out["content"] = False
        inputs = {k: v for k, v in e["inputs"].items() if keep[k]}
        del out["inputs"]
        if inputs:
            out["inputs"] = inputs
        if any(keep.values()):
            out["omitted"] = {"inputs": [k for k in e["inputs"] if not keep[k]],
                              "outputs": [k for k in keep if k not in e["inputs"] and not keep[k]]}
        return out
    if kind == "text" and keep[e["field"]]:
        return out
    if kind in FIELD_KINDS:
        del out["text"]
        out["size"] = len(e["text"])
    elif kind == "tool_call":
        del out["input"]
    elif kind == "tool_result":
        del out["output"]
        out["size"] = len(e["output"])
    elif kind == "retry":
        del out["reason"]
    elif kind == "done":
        if keep[answers[e["call"]]]:
            return out
        del out["value"]
    elif kind == "failed":
        out["error"] = {k: v for k, v in e["error"].items() if k != "message"}
    out["content"] = False
    return out


def stored_all(events: list, kept: dict) -> list:
    answers = {e["call"]: e["program"]["answer"] for e in events if e["kind"] == "started"}
    return [stored(e, kept, answers) for e in events]


# ------------------------------------------------------------------ the rules: a store


def store(appends: list) -> tuple:
    kept, results = {}, []
    for e in appends:
        s = kept.setdefault(e["stream"], [])
        seq = e["seq"]
        if seq <= len(s):
            result = "duplicate" if canonical(s[seq - 1]) == canonical(e) else "event-conflict"
        elif seq > len(s) + 1:
            result = "event-gap"
        elif s and s[-1]["call"] == e["stream"] and s[-1]["kind"] in ("done", "failed"):
            result = "event-after-end"
        elif seq == 1 and (e["kind"] != "started" or e["call"] != e["stream"]):
            result = "event-start"
        else:
            s.append(e)
            result = "kept"
        results.append(result)
    streams = {k: {"seqs": [e["seq"] for e in v],
                   "finished": bool(v) and v[-1]["call"] == k and v[-1]["kind"] in ("done", "failed")}
               for k, v in kept.items() if v}
    return results, streams


# ------------------------------------------------------------------ scenarios

UNREADABLE = "Your reply could not be read: <result> must be one of billing, shipping, product."


def one_call():
    w = Writer(1)
    w.started(1, "team", {"message": "My parcel never came."})
    w.text(1, "result", "ship")
    w.text(1, "result", "ping")
    w.done(1, "shipping")
    return w


def a_retry():
    w = Writer(1)
    w.started(1, "team", {"message": "I was charged twice."})
    w.text(1, "result", "bill")
    w.text(1, "result", "ing?")
    w.retry(1, UNREADABLE)
    w.text(1, "result", "billing", request=2)
    w.done(1, "billing")
    return w


def a_tool():
    w = Writer(1)
    w.started(1, "helper", {"question": "Where is B-2210?"})
    w.thinking(1, "Look the order up first.")
    w.text(1, "result", "Let me check.")
    w.tool_call(1, "call_1", "lookup_order", {"order": "B-2210"})
    w.tool_result(1, "call_1", "lookup_order", "In the Leeds depot, 7 days late.")
    w.text(1, "result", "It is in Leeds, ", request=2)
    w.text(1, "result", "a week late.", request=2)
    w.done(1, "It is in Leeds, a week late.")
    return w


def a_module():
    w = Writer(1)
    w.started(1, "support", {"message": "Where is B-2210? It's a week late."}, kind="module")
    w.started(2, "topic", {"message": "Where is B-2210? It's a week late."}, parent=1, root=1)
    w.started(3, "mood", {"message": "Where is B-2210? It's a week late."}, parent=1, root=1)
    w.text(2, "result", "ship")
    w.text(3, "result", "unhappy")
    w.retry(2, UNREADABLE, wait=0.5)
    w.done(3, "unhappy")
    w.text(2, "result", "shipping", request=2)
    w.done(2, "shipping")
    w.done(1, "A person will write to you today.")
    return w


def reasoning():
    w = Writer(1)
    w.started(1, "solve", {"problem": "3 ÷ 1/2"})
    w.text(1, "reasoning", "Dividing by a half ", answer=False)
    w.text(1, "reasoning", "doubles.", answer=False)
    w.text(1, "result", "6")
    w.done(1, 6)
    return w


def cancelled():
    w = Writer(1)
    w.started(1, "summarize", {"text": "A long article."})
    w.text(1, "result", "The article")
    w.failed(1, {"type": "Cancelled"})
    return w


def unfinished():
    w = Writer(1)
    w.started(1, "summarize", {"text": "A long article."})
    w.text(1, "result", "The article")
    w.text(1, "result", " says")
    return w


TRANSCRIPT = "Ana: the build is red again.\nBen: it is the flaky upload test."


def a_transcript():
    w = Writer(1)
    w.started(1, "triage_build", {"transcript": TRANSCRIPT, "question": "Is it broken for a real reason?"})
    w.thinking(1, "Ben says the upload test is flaky.")
    w.text(1, "summary", "A flaky test.", answer=False)
    w.retry(1, "Your reply could not be read: <result> must be yes or no.")
    w.text(1, "summary", "A flaky upload test.", request=2, answer=False)
    w.text(1, "result", "no", request=2)
    w.done(1, "no")
    return w


def a_failure():
    w = Writer(1)
    w.started(1, "triage_build", {"transcript": TRANSCRIPT, "question": "Is it broken for a real reason?"})
    w.tool_call(1, "call_1", "ci_log", {"job": 4412})
    w.tool_result(1, "call_1", "ci_log", "upload_test: timeout after 30 s")
    w.failed(1, {"type": "StepLimit", "message": "8 requests and still asking for tools"})
    return w


def module_with_a_private_child():
    w = Writer(1)
    w.started(1, "support", {"message": "Please read my tracking log."}, kind="module")
    w.started(2, "read_tracking", {"log": "03:12 LEEDS DEPOT ARRIVED ..."}, parent=1, root=1)
    w.text(2, "result", '{"where": "Leeds"}')
    w.done(2, {"where": "Leeds"})
    w.done(1, "Your parcel is in Leeds.")
    return w


def hidden(w: Writer, kinds=("retry", "tool_call", "tool_result", "thinking")) -> list:
    """A view that shows no retries, tool events or thinking (its seq numbers skip)."""
    return [e for e in w.events if e["kind"] not in kinds]


def replay_case(description, events, resume_after=()):
    stream = events[0]["stream"]
    return {"description": description, "kind": "replay", "events": events, "expect": replay(events, stream),
            "resume": [{"after": n, "seqs": resume(events, n)} for n in resume_after]}


def stored_case(description, w: Writer, kept):
    kept = {cid(n): k for n, k in kept.items()}
    return {"description": description, "kind": "stored", "events": w.events, "kept": kept,
            "expect": {"events": stored_all(w.events, kept)}}


def store_case(description, appends):
    results, streams = store(appends)
    return {"description": description, "kind": "store",
            "appends": [{"event": e, "expect": r} for e, r in zip(appends, results)],
            "expect": {"streams": streams}}


def cases() -> dict:
    out = {}
    out["replay-01-one-call"] = replay_case(
        "One call: its answer's pieces add up; the stream is finished when the watched call is done. A "
        "reader that has events up to N gets those after N.", one_call().events, resume_after=(0, 2, 4))
    out["replay-02-a-retry-voids-the-text"] = replay_case(
        "A retry empties the call's fields at once; the next request's pieces carry the next number.",
        a_retry().events)
    out["replay-03-tool-results-start-afresh"] = replay_case(
        "A tool result empties the call's fields: the request after it writes them afresh, without a retry. "
        "Thinking is not voided.", a_tool().events)
    out["replay-04-calls-inside-a-module"] = replay_case(
        "Calls inside a module are followed each on its own: one call's retry voids its own text only; "
        "seq numbers the one order the stream received.", a_module().events, resume_after=(5, 10))
    out["replay-05-several-fields"] = replay_case(
        "Every output is shown as its own field; only the answer's pieces say answer: true.",
        reasoning().events)
    out["replay-06-closed"] = replay_case(
        "A stream closed early ends with its call's failed Cancelled: it is finished.", cancelled().events)
    out["replay-07-unfinished"] = replay_case(
        "A kept stream whose writer stopped: the fields so far, and not finished.", unfinished().events)
    out["replay-08-a-view-without-retries"] = replay_case(
        "A view that hides retries, tool events and thinking skips seq numbers; a piece of a higher request "
        "still starts the call's fields afresh, so it shows what the whole stream shows once the new "
        "request writes.", hidden(a_retry()), resume_after=(3,))
    out["replay-09-a-view-without-tools"] = replay_case(
        "The same, for the request after tool results.", hidden(a_tool()))
    out["replay-10-a-stored-stream"] = replay_case(
        "A stored stream whose answer was kept as its size only: the field's length so far, voided by the "
        "retry as text would be.",
        stored_all(a_transcript().events, {cid(1): {"transcript": True, "question": True, "summary": True,
                                                    "result": False}}))

    whole = {"transcript": True, "question": True, "summary": True, "result": True}
    out["stored-01-whole"] = stored_case(
        "Every value kept: the stored form is the event itself.", a_transcript(), {1: whole})
    out["stored-02-no-value"] = stored_case(
        "log_content false: every event keeps its shape and its times, no value; text and thinking keep "
        "their size.", a_transcript(), {1: {k: False for k in whole}})
    out["stored-03-all-but-the-transcript"] = stored_case(
        "The transcript kept as its size only: started lists it in omitted; the outputs' pieces and the "
        "answer stay; thinking and the retry's reason go (they can quote the transcript).",
        a_transcript(), {1: {**whole, "transcript": False}})
    out["stored-04-the-answer-as-its-size"] = stored_case(
        "The answer kept as its size only: its pieces become sizes and done keeps no value; the other "
        "output's pieces stay.", a_transcript(), {1: {**whole, "result": False}})
    out["stored-05-tools-and-a-failure"] = stored_case(
        "Tool calls and results, and an error's message, go when the call's content is not whole; the error "
        "keeps its type.", a_failure(), {1: {**whole, "transcript": False}})
    out["stored-06-each-call-its-own"] = stored_case(
        "Each call follows its own log_content: a module whose values are kept, calling a helper whose "
        "input is not.", module_with_a_private_child(),
        {1: {"message": True, "result": True}, 2: {"log": False, "result": True}})

    w = a_tool()
    ev = w.events
    out["store-01-in-order"] = store_case("Events appended in seq order are kept; the stream is finished at its "
                                          "watched call's done.", ev)
    changed = copy.deepcopy(ev[2])
    changed["text"] = "Let me look."
    out["store-02-sent-again"] = store_case(
        "An event sent again after a failure changes nothing; another event at a kept seq is a second "
        "writer: refused.", ev[:4] + [copy.deepcopy(ev[2]), changed] + ev[4:])
    out["store-03-a-gap"] = store_case(
        "An event beyond the next seq is refused; the stream is kept up to the last event before the gap, "
        "and goes on when the missing event comes.", ev[:3] + [ev[4], ev[3], ev[4]])
    after = copy.deepcopy(ev[-1])
    after.update(seq=after["seq"] + 1, kind="text", field="result", answer=True, request=3, text="!")
    del after["value"]
    out["store-04-nothing-after-the-end"] = store_case(
        "Nothing is kept after the watched call's end; its last event sent again is still a duplicate.",
        ev + [after, copy.deepcopy(ev[-1])])
    m = a_module().events
    bad_first = copy.deepcopy(m[1])
    bad_first.update(seq=1, stream=cid(2))
    bad_first_wrong_call = copy.deepcopy(m[1])
    bad_first_wrong_call["seq"] = 1
    text_first = copy.deepcopy(m[3])
    text_first.update(seq=1, stream=cid(9))
    out["store-05-a-stream-starts-with-its-call"] = store_case(
        "A stream's first event is its watched call's started: another call's started, or another kind, "
        "is refused. A stream may watch a call made inside another (its started has a parent).",
        [bad_first_wrong_call, text_first, bad_first])
    other = Writer(2)
    other.started(2, "team", {"message": "Hello"})
    other.text(2, "result", "other")
    other.done(2, "other")
    mixed = [ev[0], other.events[0], ev[1], other.events[1], ev[2], other.events[2]]
    out["store-06-streams-are-numbered-apart"] = store_case(
        "Two streams in one store are numbered each on its own.", mixed)
    return out
