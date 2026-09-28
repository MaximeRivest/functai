"""The cases in events/: stream events as data, written from the rules in
../streaming.md. Four kinds, told apart by "kind":

- "replay": {"events": [...], "expect": {"views": [...], "finished"},
  "resume": [{"after": N, "seqs": [...]}]}. Reading the events in order,
  the state after each one (section "Replaying"): for each call started
  so far, {"ended": null | "done" | "failed", "fields": {name: text so
  far}}. ``finished``: the tree's outermost call ended. ``resume``: the
  seqs a reader that has events up to ``after`` gets. The events are a
  whole log, a kept log, or a view: replay reads each the same way.
- "follow": {"received": [events, in the order a reader got them],
  "expect": ["kept" | "duplicate" | "loss", ...]}. A reader following one
  form of a log ("Following a log"): an event it already has is a
  duplicate; one whose ``after`` is the last it has is kept; any other
  says events were lost.
- "kept": {"events": [whole events], "kept": {call id: {"inputs": {name:
  bool}, "outputs": {name: bool}}}, "expect": {"events": [the kept
  log]}}. The kept form of a whole log ("The kept form"), given which
  fields of each call its log_content keeps (content/'s job).
- "store": {"appends": [{"event", "expect": "kept" | "duplicate" |
  "event-conflict" | "event-gap" | "event-after-end" | "event-start" |
  "event-malformed"}], "expect": {"logs": {tree id: {"seqs": [...],
  "finished": bool}}}}. Appending each event in turn to one store ("The
  rules a store keeps").

Every event here passes schema/event.schema.json (make.py checks).
"""

import copy

from common import canonical

V = "sha256:" + "1" * 64
SIG = "sha256:" + "5" * 64
MODEL = "gpt-4.1-mini"
VOIDS = ("request", "retry")


def cid(n):
    return f"01926c00-{n:04x}-7000-8000-000000000000"


def program(name, kind="ai", answer="result"):
    p = {"name": name, "kind": kind, "module": "shop", "version": V, "signature": SIG, "interface": SIG,
         "answer": answer}
    if kind == "module":
        del p["signature"]
    return p


class Writer:
    """The whole log of one call tree, numbered as its events happen."""

    def __init__(self, tree: int, *, first_seq: int = 1, first_second: int = 0):
        self.tree = cid(tree)
        self.seq = first_seq - 1
        self.second = first_second
        self.names = {}
        self.requests = {}
        self.events = []

    def ev(self, kind, n, **fields):
        self.seq += 1
        e = {"functai_event": 2, "kind": kind, "tree": self.tree, "seq": self.seq, "after": self.seq - 1,
             "at": f"2026-09-28T10:00:{self.second + self.seq // 100:02d}.{(self.seq % 100) * 10000:06d}Z",
             "call": cid(n), "function": self.names[n], **fields}
        self.events.append(e)
        return e

    def started(self, n, name, inputs, *, parent=None, root=None, kind="ai", answer="result", saw=()):
        self.names[n] = name
        return self.ev("started", n, parent=cid(parent) if parent else None, root=cid(root or n),
                       program=program(name, kind, answer), inputs=inputs, content=True, saw=list(saw))

    def request(self, n, model=MODEL):
        self.requests[n] = self.requests.get(n, 0) + 1
        return self.ev("request", n, request=self.requests[n], model=model)

    def text(self, n, field, text, answer=True):
        return self.ev("text", n, field=field, answer=answer, text=text)

    def thinking(self, n, text):
        return self.ev("thinking", n, text=text)

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


# ------------------------------------------------------------------ the rules: replay, resume, follow


def replay(events: list) -> dict:
    calls, views = {}, []
    finished = False
    tree = events[0]["tree"] if events else None
    for e in events:
        c, kind = e["call"], e["kind"]
        if kind == "started":
            calls[c] = {"ended": None, "fields": {}}
        elif c not in calls:
            raise AssertionError(f"event {e['seq']}: its call's started is not in this form")
        elif kind in VOIDS:
            calls[c]["fields"] = {}
        elif kind == "text":
            fields = calls[c]["fields"]
            fields[e["field"]] = fields.get(e["field"], "") + e["text"]
        elif kind in ("done", "failed"):
            calls[c]["ended"] = kind
            if c == tree:
                finished = True
        views.append({"calls": copy.deepcopy(calls)})
    return {"views": views, "finished": finished}


def resume(events: list, after: int) -> list:
    return [e["seq"] for e in events if e["seq"] > after]


def follow(received: list) -> list:
    last, out = 0, []
    for e in received:
        if e["seq"] <= last:
            out.append("duplicate")
        elif e["after"] == last:
            out.append("kept")
            last = e["seq"]
        else:
            out.append("loss")
    return out


def relink(events: list) -> list:
    """A form of a log: each event's after is the seq of the event before it in this form."""
    out, last = [], 0
    for e in events:
        e = copy.deepcopy(e)
        e["after"] = last
        last = e["seq"]
        out.append(e)
    return out


# ------------------------------------------------------------------ the rules: the kept form


def kept_event(e: dict, keep: dict, info: dict):
    """The kept form of one event (None: not in the kept log)."""
    fields = {**keep["inputs"], **keep["outputs"]}
    if all(fields.values()):
        return copy.deepcopy(e)
    out = copy.deepcopy(e)
    kind = e["kind"]
    if kind == "started":
        out["content"] = False
        del out["inputs"]
        inputs = {k: v for k, v in e["inputs"].items() if keep["inputs"][k]}
        out["omitted"] = {"inputs": [k for k, v in keep["inputs"].items() if not v],
                          "outputs": [k for k, v in keep["outputs"].items() if not v]}
        if inputs:
            out["inputs"] = inputs
        order = list(e)
        order.insert(order.index("content") + 1, "omitted")
        return {k: out[k] for k in sorted(out, key=order.index)}
    if kind == "request":
        return out
    if kind == "text":
        return out if keep["outputs"][e["field"]] else None
    if kind == "thinking":
        return None
    if kind == "tool_call":
        del out["input"]
    elif kind == "tool_result":
        del out["output"]
    elif kind == "retry":
        del out["reason"]
    elif kind == "done":
        holds = [info["answer"]] if info["kind"] == "ai" else list(keep["outputs"])
        if all(keep["outputs"][k] for k in holds):
            return out
        del out["value"]
    elif kind == "failed":
        out["error"] = {k: v for k, v in e["error"].items() if k != "message"}
    out["content"] = False
    return out


def kept_log(events: list, kept: dict) -> list:
    info = {e["call"]: e["program"] for e in events if e["kind"] == "started"}
    out = [kept_event(e, kept[e["call"]], info[e["call"]]) for e in events]
    return relink([e for e in out if e is not None])


# ------------------------------------------------------------------ the rules: a store


def store(appends: list) -> tuple:
    logs, results = {}, []
    for e in appends:
        log = logs.setdefault(e["tree"], [])
        last = log[-1]["seq"] if log else 0
        have = {k["seq"]: k for k in log}
        if e["seq"] <= e["after"]:
            result = "event-malformed"
        elif e["seq"] in have:
            result = "duplicate" if canonical(have[e["seq"]]) == canonical(e) else "event-conflict"
        elif log and log[-1]["call"] == e["tree"] and log[-1]["kind"] in ("done", "failed"):
            result = "event-after-end"
        elif e["after"] > last:
            result = "event-gap"
        elif e["after"] < last or e["seq"] < last:
            result = "event-conflict"
        elif not log and (e["kind"] != "started" or e["call"] != e["tree"]):
            result = "event-start"
        else:
            log.append(e)
            result = "kept"
        results.append(result)
    out = {k: {"seqs": [e["seq"] for e in v],
               "finished": bool(v) and v[-1]["call"] == k and v[-1]["kind"] in ("done", "failed")}
           for k, v in logs.items() if v}
    return results, out


# ------------------------------------------------------------------ scenarios

UNREADABLE = "Your reply could not be read: <result> must be one of billing, shipping, product."


def one_call():
    w = Writer(1)
    w.started(1, "team", {"message": "My parcel never came."})
    w.request(1)
    w.text(1, "result", "ship")
    w.text(1, "result", "ping")
    w.done(1, "shipping")
    return w


def a_retry():
    w = Writer(1)
    w.started(1, "team", {"message": "I was charged twice."})
    w.request(1)
    w.text(1, "result", "bill")
    w.text(1, "result", "ing?")
    w.retry(1, UNREADABLE)
    w.request(1)
    w.text(1, "result", "billing")
    w.done(1, "billing")
    return w


def a_retry_then_nothing():
    w = Writer(1)
    w.started(1, "team", {"message": "I was charged twice."})
    w.request(1)
    w.text(1, "result", "billing?")
    w.retry(1, UNREADABLE, wait=2.0)
    w.request(1)
    w.failed(1, {"type": "ProviderError", "message": "503 from the provider"})
    return w


def a_tool():
    w = Writer(1)
    w.started(1, "helper", {"question": "Where is B-2210?"})
    w.request(1)
    w.thinking(1, "Look the order up first.")
    w.text(1, "result", "Let me check.")
    w.tool_call(1, "call_1", "lookup_order", {"order": "B-2210"})
    w.tool_result(1, "call_1", "lookup_order", "In the Leeds depot, 7 days late.")
    w.request(1)
    w.text(1, "result", "It is in Leeds, ")
    w.text(1, "result", "a week late.")
    w.done(1, "It is in Leeds, a week late.")
    return w


def a_tool_last():
    """Text, then tool traffic, then a reply with no text: a view that hides tools ends on hidden events."""
    w = Writer(1)
    w.started(1, "helper", {"question": "Cancel B-2210."})
    w.request(1)
    w.text(1, "result", "Cancelling.")
    w.tool_call(1, "call_1", "cancel_order", {"order": "B-2210"})
    w.tool_result(1, "call_1", "cancel_order", "cancelled")
    w.request(1)
    w.done(1, "")
    return w


def a_module():
    w = Writer(1)
    w.started(1, "support", {"message": "Where is B-2210? It's a week late."}, kind="module")
    w.started(2, "topic", {"message": "Where is B-2210? It's a week late."}, parent=1, root=1)
    w.started(3, "mood", {"message": "Where is B-2210? It's a week late."}, parent=1, root=1)
    w.request(2)
    w.request(3)
    w.text(2, "result", "ship")
    w.text(3, "result", "unhappy")
    w.retry(2, UNREADABLE, wait=0.5)
    w.request(2)
    w.done(3, "unhappy")
    w.text(2, "result", "shipping")
    w.done(2, "shipping")
    w.done(1, "A person will write to you today.")
    return w


def reasoning():
    w = Writer(1)
    w.started(1, "solve", {"problem": "3 ÷ 1/2"})
    w.request(1)
    w.text(1, "reasoning", "Dividing by a half ", answer=False)
    w.text(1, "reasoning", "doubles.", answer=False)
    w.text(1, "result", "6")
    w.done(1, 6)
    return w


def cancelled():
    w = Writer(1)
    w.started(1, "summarize", {"text": "A long article."})
    w.request(1)
    w.text(1, "result", "The article")
    w.failed(1, {"type": "Cancelled"})
    return w


def unfinished():
    w = Writer(1)
    w.started(1, "summarize", {"text": "A long article."})
    w.request(1)
    w.text(1, "result", "The article")
    w.text(1, "result", " says")
    return w


TRANSCRIPT = "Ana: the build is red again.\nBen: it is the flaky upload test."


def a_transcript():
    w = Writer(1)
    w.started(1, "triage_build", {"transcript": TRANSCRIPT, "question": "Is it broken for a real reason?"})
    w.request(1)
    w.thinking(1, "Ben says the upload test is flaky.")
    w.text(1, "summary", "A flaky test.", answer=False)
    w.retry(1, "Your reply could not be read: <result> must be yes or no.")
    w.request(1)
    w.text(1, "summary", "A flaky upload test.", answer=False)
    w.text(1, "result", "no")
    w.done(1, "no")
    return w


def a_failure():
    w = Writer(1)
    w.started(1, "triage_build", {"transcript": TRANSCRIPT, "question": "Is it broken for a real reason?"})
    w.request(1)
    w.tool_call(1, "call_1", "ci_log", {"job": 4412})
    w.tool_result(1, "call_1", "ci_log", "upload_test: timeout after 30 s")
    w.request(1)
    w.failed(1, {"type": "StepLimit", "message": "8 requests and still asking for tools: " + TRANSCRIPT[:20]})
    return w


def module_with_a_private_child():
    w = Writer(1)
    w.started(1, "support", {"message": "Please read my tracking log."}, kind="module")
    w.started(2, "read_tracking", {"log": "03:12 LEEDS DEPOT ARRIVED ..."}, parent=1, root=1)
    w.request(2)
    w.text(2, "result", '{"where": "Leeds"}')
    w.done(2, {"where": "Leeds"})
    w.done(1, "Your parcel is in Leeds.")
    return w


def module_with_two_outputs():
    w = Writer(1)
    w.started(1, "support", {"message": "My card was charged twice."}, kind="module")
    w.done(1, {"notes": "Customer 4411 has had three refunds this month.", "result": "A refund is on its way."})
    return w


def reasoning_about_a_transcript():
    w = Writer(1)
    w.started(1, "triage_build", {"transcript": TRANSCRIPT})
    w.request(1)
    w.text(1, "reasoning", "Ben says the upload test ", answer=False)
    w.text(1, "reasoning", "is flaky.", answer=False)
    w.text(1, "result", "no")
    w.done(1, "no")
    return w


def hide(events: list, kinds=("tool_call", "tool_result", "thinking")) -> list:
    """A view that shows no tool traffic or thinking, and no retry's reason."""
    out = []
    for e in events:
        if e["kind"] in kinds:
            continue
        e = copy.deepcopy(e)
        if e["kind"] == "retry":
            del e["reason"]
            e["content"] = False
        out.append(e)
    return relink(out)


def boundary(events: list) -> list:
    """A view that shows only the outermost call: its start, requests, retries (no reason), text and end."""
    tree = events[0]["tree"]
    return hide([e for e in events if e["call"] == tree])


# ------------------------------------------------------------------ cases


def same_as_whole(view: list, whole: list) -> None:
    """After each event a view shows, its state is the whole log's at that event, for what the view shows."""
    whole_states = {e["seq"]: s for e, s in zip(whole, replay(whole)["views"])}
    for e, state in zip(view, replay(view)["views"]):
        full = whole_states[e["seq"]]["calls"]
        for c, s in state["calls"].items():
            assert s["ended"] == full[c]["ended"], (e["seq"], c)
            for f, text in s["fields"].items():
                assert full[c]["fields"].get(f) == text, (e["seq"], c, f)
            shown = {x["field"] for x in view if x["call"] == c and x["kind"] == "text"}
            assert {f: t for f, t in full[c]["fields"].items() if f in shown} == s["fields"], (e["seq"], c)


def replay_case(description, events, resume_after=(), whole=None):
    if whole is not None:
        same_as_whole(events, whole)
    return {"description": description, "kind": "replay", "events": events, "expect": replay(events),
            "resume": [{"after": n, "seqs": resume(events, n)} for n in resume_after]}


def follow_case(description, received):
    return {"description": description, "kind": "follow", "received": received, "expect": follow(received)}


def kept_case(description, w: Writer, kept):
    kept = {cid(n): k for n, k in kept.items()}
    return {"description": description, "kind": "kept", "events": w.events, "kept": kept,
            "expect": {"events": kept_log(w.events, kept)}}


def store_case(description, appends):
    results, logs = store(appends)
    return {"description": description, "kind": "store",
            "appends": [{"event": e, "expect": r} for e, r in zip(appends, results)],
            "expect": {"logs": logs}}


def io(inputs: dict, outputs: dict) -> dict:
    return {"inputs": inputs, "outputs": outputs}


def cases() -> dict:
    out = {}
    out["replay-01-one-call"] = replay_case(
        "One call: its answer's pieces add up; the log is finished when the tree's outermost call is done. A "
        "reader that has events up to N gets those after N.", one_call().events, resume_after=(0, 2, 5))
    out["replay-02-a-retry-voids-the-text"] = replay_case(
        "A retry empties the call's fields at once; so does every request.", a_retry().events)
    out["replay-03-each-request-starts-afresh"] = replay_case(
        "The request after tool results empties the call's fields: the reply writes them afresh. Thinking is "
        "shown as it comes and never voided.", a_tool().events)
    out["replay-04-calls-inside-a-module"] = replay_case(
        "Calls inside a module are followed each on its own: one call's retry voids its own text only; "
        "seq numbers the one order of the tree's events.", a_module().events, resume_after=(5, 12))
    out["replay-05-several-fields"] = replay_case(
        "Every output is shown as its own field; only the answer's pieces say answer: true.",
        reasoning().events)
    out["replay-06-closed"] = replay_case(
        "A stream closed early ends with its call's failed Cancelled: the log is finished.", cancelled().events)
    out["replay-07-unfinished"] = replay_case(
        "A kept log whose writer stopped: the fields so far, and not finished.", unfinished().events)
    w = a_retry()
    out["replay-08-a-view-keeps-requests-and-retries"] = replay_case(
        "A view that hides tool traffic, thinking and retry reasons keeps every request and retry (they hold "
        "no value), so after each event it shows, its fields are the whole log's.", hide(w.events),
        resume_after=(3,), whole=w.events)
    w = a_tool()
    out["replay-09-a-view-without-tools"] = replay_case(
        "The same, for the request after tool results: seqs skip, after links each event to the one before "
        "it in the view.", hide(w.events), resume_after=(4,), whole=w.events)
    w = a_retry_then_nothing()
    out["replay-10-a-retry-that-writes-nothing"] = replay_case(
        "A retry whose request fails before writing: the view empties the text at the retry, as the whole "
        "log does, though no later piece comes.", hide(w.events), whole=w.events)
    w = a_tool_last()
    out["replay-11-a-hidden-tail"] = replay_case(
        "A view whose last shown text is followed by hidden tool traffic: the request after it still "
        "empties the field. A reader resuming after the last event it saw gets the rest of the view.",
        hide(w.events), resume_after=(3,), whole=w.events)
    w = a_module()
    out["replay-12-a-boundary"] = replay_case(
        "A view that shows only the tree's outermost call: a module's code writes no pieces, so the view "
        "shows its start and its end (what a module's answer shows while it is written is stage 3's).",
        boundary(w.events), whole=w.events)
    out["replay-13-a-kept-log"] = replay_case(
        "A kept log whose answer was dropped: its pieces are not there, the other output's are.",
        kept_log(a_transcript().events, {cid(1): io({"transcript": True, "question": True},
                                                  {"summary": True, "result": False})}))

    w = one_call()
    whole_log = w.events
    lost = [whole_log[0], whole_log[1], whole_log[3]]
    out["follow-01-a-whole-log"] = follow_case(
        "A reader of the whole log keeps each event whose after is the last it has; an event it has is a "
        "duplicate; an event whose after it lacks says events were lost (it reads again after its last).",
        lost + [whole_log[2], whole_log[3], whole_log[4], whole_log[4]])
    view = hide(a_tool().events)
    out["follow-02-a-view"] = follow_case(
        "A view's seqs skip, and nothing is lost: each event's after is the one before it in the view. A "
        "view event that does not come is a loss; reading the view again after the last kept event recovers "
        "it.", view[:3] + [view[4]] + view[3:])

    whole = io({"transcript": True, "question": True}, {"summary": True, "result": True})
    out["kept-01-whole"] = kept_case(
        "Every value kept: the kept form is the event itself.", a_transcript(), {1: whole})
    out["kept-02-no-value"] = kept_case(
        "log_content false: started keeps no input and names every field in omitted; no piece of text or "
        "thinking is kept, not even its size; requests stay (they hold no value); the retry keeps no reason, "
        "done no value.", a_transcript(),
        {1: io({"transcript": False, "question": False}, {"summary": False, "result": False})})
    out["kept-03-all-but-the-transcript"] = kept_case(
        "The transcript dropped: started names it in omitted; the outputs' pieces and the answer stay; "
        "thinking and the retry's reason go (they can quote the transcript).",
        a_transcript(), {1: io({"transcript": False, "question": True}, {"summary": True, "result": True})})
    out["kept-04-the-answer-dropped"] = kept_case(
        "The answer dropped: none of its pieces is kept and done keeps no value; the other output's pieces "
        "stay.", a_transcript(), {1: io({"transcript": True, "question": True}, {"summary": True, "result": False})})
    out["kept-05-tools-and-a-failure"] = kept_case(
        "Tool calls and results keep their id and name, not their input or output; an error keeps its type, "
        "not its message.", a_failure(),
        {1: io({"transcript": False, "question": True}, {"result": True})})
    out["kept-06-each-call-its-own"] = kept_case(
        "Each call follows its own log_content: a module whose values are kept, calling a helper whose "
        "input is not.", module_with_a_private_child(),
        {1: io({"message": True}, {"result": True}), 2: io({"log": False}, {"result": True})})
    out["kept-07-a-module-with-a-private-output"] = kept_case(
        "A module's done holds all its outputs: when one is dropped, done keeps no value (the kept outputs "
        "are in the call's record).", module_with_two_outputs(),
        {1: io({"message": True}, {"notes": False, "result": True})})
    out["kept-08-the-reasoning-functai-added"] = kept_case(
        "The reasoning FunctAI adds is a field of the call: dropped with the transcript (content's rule), so "
        "none of its pieces is kept.", reasoning_about_a_transcript(),
        {1: io({"transcript": False}, {"reasoning": False, "result": True})})

    w = a_tool()
    ev = kept_log(w.events, {cid(1): io({"question": True}, {"result": True})})
    out["store-01-in-order"] = store_case(
        "Events appended in order are kept; the log is finished at its outermost call's done.", ev)
    changed = copy.deepcopy(ev[3])
    changed["text"] = "Let me look."
    out["store-02-sent-again"] = store_case(
        "An event sent again after a failure changes nothing; another event at a kept seq is a second "
        "writer: refused.", ev[:5] + [copy.deepcopy(ev[3]), changed] + ev[5:])
    out["store-03-a-gap"] = store_case(
        "An event whose after is beyond the last kept is refused; the log goes on when the missing event "
        "comes.", ev[:3] + [ev[4], ev[3], ev[4]])
    after = copy.deepcopy(ev[-1])
    after.update(seq=after["seq"] + 1, after=after["seq"], kind="text", field="result", answer=True, text="!")
    del after["value"]
    out["store-04-nothing-after-the-end"] = store_case(
        "Nothing is kept after the outermost call's end; its last event sent again is still a duplicate.",
        ev + [after, copy.deepcopy(ev[-1])])
    m = a_module().events
    child_first = copy.deepcopy(m[1])
    child_first.update(seq=1, after=0, tree=cid(2))
    wrong_call = copy.deepcopy(m[1])
    wrong_call.update(seq=1, after=0)
    text_first = copy.deepcopy(m[5])
    text_first.update(seq=1, after=0, tree=cid(9))
    out["store-05-a-log-starts-with-its-tree"] = store_case(
        "A log's first event is its outermost call's started: a started of another call, or another kind, "
        "is refused.", [wrong_call, text_first, child_first])
    other = Writer(2)
    other.started(2, "team", {"message": "Hello"})
    other.request(2)
    other.text(2, "result", "other")
    other.done(2, "other")
    mixed = [ev[0], other.events[0], ev[1], other.events[1], ev[2], other.events[2]]
    out["store-06-trees-are-numbered-apart"] = store_case("Two trees in one store are numbered each on its own.",
                                                          mixed)
    sparse = kept_log(a_transcript().events, {cid(1): io({"transcript": False, "question": True},
                                                          {"summary": True, "result": True})})
    out["store-07-a-kept-log-skips"] = store_case(
        "A kept log leaves out events (thinking here), so its seqs skip; each event's after says what comes "
        "before it, and the store checks that.", sparse)
    first = Writer(1)
    first.started(1, "gardener", {"request": "Merge groceries.md into todo.md"})
    first.request(1)
    first.text(1, "result", "Merging")
    stale = copy.deepcopy(first.events[-1])
    stale.update(seq=stale["seq"] + 1, after=stale["seq"], text=" now")
    second = Writer(1, first_seq=first.seq + 1, first_second=30)
    second.names = dict(first.names)
    second.requests = dict(first.requests)
    second.request(1)
    second.text(1, "result", "Merged.")
    second.done(1, "Merged.")
    handed = [first.events[0], first.events[1], first.events[2], second.events[0], stale] + second.events[1:]
    out["store-08-a-later-writer-goes-on"] = store_case(
        "A log whose writer stopped can be continued by another (after a restart): it appends after the last "
        "kept event, numbering on from there. An event of the first writer that comes late is refused.",
        handed)
    bad = copy.deepcopy(ev[1])
    bad["after"] = bad["seq"]
    out["store-09-malformed"] = store_case("An event whose seq is not greater than its after is refused.",
                                           [ev[0], bad, ev[1]])
    return out
