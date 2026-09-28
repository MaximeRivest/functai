"""The cases in events/: stream events as data, written from the rules in
../streaming.md. Six kinds, told apart by "kind".

An event is named by its writer and seq: a **position** is {"writer",
"seq"}, or null for "before the first event". An event's ``after`` is a
position: the event before it in the form being read.

- "replay": {"events": [...], "expect": {"views": [...], "finished"},
  "resume": [{"after": position, "expect": {"events": [positions]} or
  {"refuses": "event-unknown"}}]}. Reading the events in order, the
  state after each one (section "Replaying"): for each call started so
  far, {"ended": null | "done" | "failed", "fields": {name: text so
  far}}. ``finished``: the tree's outermost call ended. ``resume``: what
  a source holding these events gives a reader that has events up to
  ``after``. The events are a whole log, a kept log, or a view: replay
  reads each the same way. An event of a kind the reader does not know
  changes nothing.
- "follow": {"received": [events, in the order a reader got them],
  "recover"?: {"form": "same" | "other", "source": [events]}, "expect":
  {"results": ["kept" | "duplicate" | "stale" | "rewind" | "loss" |
  "unknown-format", ...], "state": {"calls", "finished"} per tree,
  "recover"?: {"reads": [{"after": position, "expect": ...}], "state"}}}.
  A reader following one form of a log ("Following a log"), keeping for
  each tree the last event it has: an event of an earlier writer is
  stale; one of its last event's writer at or below its seq is a
  duplicate; one whose ``after`` is its last is kept (next); one of a
  later writer whose ``after`` names an event it holds (or null) makes it
  drop what it has after that event (rewind) and take it; anything else
  says events were lost. An event of a format it does not know stops it
  (nothing after it is read). ``state``: the replay of what the reader
  holds at the end. ``recover``: the reader then reads again from a
  source holding ``source`` ("Resuming"): in the same form, after its
  last event, and from the beginning when the source does not have it;
  in another form, from the beginning at once. ``reads`` are its reads
  in order, ``state`` the replay of what it holds after them.
- "kept": {"events": [whole events], "kept": {call id: {"inputs": {name:
  bool}, "outputs": {name: bool}}}, "expect": {"events": [the kept
  log]}}. The kept form of a whole log ("The kept form"), given which
  fields of each call its log_content keeps (content/'s job).
- "store": {"steps": [{"append": event, "expect": "kept" | "duplicate" |
  "event-conflict" | "event-gap" | "event-after-end" | "event-start" |
  "event-malformed"} or {"claim": tree id, "expect": {"writer", "after":
  position} or {"refuses": "event-after-end" | "event-unknown"}}],
  "reads": [{"tree", "after": position, "expect": {"events": [positions]}
  | {"refuses": "event-unknown"}}], "expect": {"logs": {tree id:
  {"events": [positions], "writer": n, "finished": bool}}}}. Each step in
  turn against one store (appending an event, or a later writer claiming
  a log), then the reads ("The rules a store keeps").
- "journal": {"mode": "required" | "best-effort", "retries": n, "events":
  [the call's whole log, as it goes when nothing fails], "script": [what
  each append attempt meets, in order: "ok" | "lost" | "down" |
  "conflict"; "ok" once the script ends; and, between attempts, what
  another writer does: "claimed" | "ended"], "expect": {"log": [the
  events the writer made], "trace": [{"seq", "transport", "answer"} or
  {"other", "writer"}], "shown": [seqs given to readers], "kept":
  {"events": [positions], "finished"}, "caller": {...}, "record":
  {"error", "journal"?}, "settled"?}}. A writer keeping a call's log in a
  journal that fails ("Keeping a log while it is written"). ``ok``: the
  append reaches the journal and its answer comes back; ``lost``: it
  reaches the journal (which applies it) and the answer is lost;
  ``down``: it does not reach the journal; ``conflict``: the journal
  answers event-conflict (another writer has the log) and keeps nothing.
  ``claimed``: another writer claims the log (it is fenced); ``ended``:
  another writer claims the log and ends it (failed, Cancelled). The
  writer sends each event at most 1 + retries times in a row before it
  gives up on it for now. ``settled``, when the caller was told
  "unknown": what reading the journal for the end the writer attempted
  says: "kept", "not-kept" (the log is unfinished) or "another-end"
  (another writer ended the log).

- "receivers": {"scenarios": [{"layers": [{"where": "own" | "block" |
  "configure", "observers"?: [names], "journal"?: {"name", "mode":
  "required" | "best-effort"} or null}, ...] (closest first), "expect":
  {"observers": [names], "journal": {...} or null} or {"refuses":
  "journal-policy"}}]}. Which observers and which journal a tree gets
  from the layers around its outermost call ("Keeping a log while it is
  written"): observers add up, outermost first; the closest journal
  setting decides, except that no closer layer may replace, weaken or
  remove a required journal.

Every event here passes schema/event.schema.json (make.py checks), except
those of a format no reader knows and appends refused event-malformed.
"""

import copy

from common import canonical

V = "sha256:" + "1" * 64
SIG = "sha256:" + "5" * 64
MODEL = "gpt-4.1-mini"
VOIDS = ("request", "retry")
KINDS = ("started", "request", "text", "thinking", "tool_call", "tool_result", "retry", "done", "failed")
ENVELOPE = ("functai_event", "kind", "tree", "writer", "seq", "after", "at", "call", "function")
KEYS = {"started": ("parent", "root", "program", "inputs", "content", "omitted", "saw"),
        "request": ("request", "model"), "text": ("field", "answer", "text"), "thinking": ("text",),
        "tool_call": ("id", "name", "input", "content"), "tool_result": ("id", "name", "output", "content"),
        "retry": ("reason", "wait", "content"), "done": ("value", "content"), "failed": ("error", "content")}


def cid(n):
    return f"01926c00-{n:04x}-7000-8000-000000000000"


def at(writer: int, seq: int) -> dict:
    """A position: the event a writer numbered seq."""
    return {"writer": writer, "seq": seq}


def pos(e: dict) -> dict:
    """An event's own position: its writer and seq (with its tree, its name in every form)."""
    return at(e["writer"], e["seq"])


def program(name, kind="ai", answer="result"):
    p = {"name": name, "kind": kind, "module": "shop", "version": V, "signature": SIG, "interface": SIG,
         "answer": answer}
    if kind == "module":
        del p["signature"]
    return p


class Writer:
    """The whole log of one call tree, numbered as its events happen. ``writer``: which writer of the log
    numbers them (1 for the first; a writer that continues a log after another has the number its claim
    gave it). Each event's ``after`` is the position of the event before it."""

    def __init__(self, tree: int, *, writer: int = 1, first_second: int = 0):
        self.tree = cid(tree)
        self.writer = writer
        self.seq = 0
        self.last = None
        self.second = first_second
        self.names = {}
        self.requests = {}
        self.events = []

    def ev(self, kind, n, **fields):
        self.seq += 1
        e = {"functai_event": 2, "kind": kind, "tree": self.tree, "writer": self.writer, "seq": self.seq,
             "after": self.last,
             "at": f"2026-09-28T10:00:{self.second + self.seq // 100:02d}.{(self.seq % 100) * 10000:06d}Z",
             "call": cid(n), "function": self.names[n], **fields}
        self.last = pos(e)
        self.events.append(e)
        return e

    def continuing(self, kept: list, claim: dict) -> "Writer":
        """This writer goes on after a kept log another writer left, with what its claim gave it: its writer
        number, and the last kept event, after which it numbers ("Continuing a log"). Each call's request
        count comes from the kept request events. Nothing is taken from the events' writer numbers: a claim
        may have given numbers no event carries."""
        assert "writer" in claim, claim
        assert kept and pos(kept[-1]) == claim["after"], (kept[-1:], claim)
        self.writer = claim["writer"]
        self.seq = claim["after"]["seq"]
        self.last = claim["after"]
        for e in kept:
            if e["kind"] == "started":
                self.names[int(e["call"].split("-")[1], 16)] = e["function"]
            if e["kind"] == "request":
                n = int(e["call"].split("-")[1], 16)
                self.requests[n] = max(self.requests.get(n, 0), e["request"])
        self.events = []
        return self

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
        if kind not in KINDS:
            pass                                    # a kind this reader does not know: its place, nothing else
        elif kind == "started":
            calls[c] = {"ended": None, "fields": {}}
        elif c not in calls:
            raise AssertionError(f"event {pos(e)}: its call's started is not in this form")
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


def state_of(events: list) -> dict:
    r = replay(events)
    return {"calls": r["views"][-1]["calls"] if r["views"] else {}, "finished": r["finished"]}


def resume(events: list, after) -> dict:
    """What a source holding these events (one form of one log) gives a reader that has events up to
    ``after`` ("Resuming"): the events after it, or event-unknown when it does not have that event."""
    if after is None:
        return {"events": [pos(e) for e in events]}
    at_ = [i for i, e in enumerate(events) if pos(e) == after]
    if not at_:
        return {"refuses": "event-unknown"}
    return {"events": [pos(e) for e in events[at_[0] + 1:]]}


def follow(received: list, recover=None) -> dict:
    """A reader following one form of logs, one state per tree ("Following a log"). Each event's ``after``
    is compared as a position (writer and seq), never as a seq alone. ``recover``: then, for the one tree
    followed, reading again from a source ("Resuming")."""
    trees, out = {}, []
    for e in received:
        if e.get("functai_event") != 2:
            out.append("unknown-format")
            break
        t = trees.setdefault(e["tree"], {"last": None, "held": []})
        last = t["last"]
        writer = last["writer"] if last else 0
        held = [pos(x) for x in t["held"]]
        if e["writer"] < writer:
            out.append("stale")
            continue
        if e["writer"] == writer and e["seq"] <= last["seq"]:
            out.append("duplicate")
            continue
        if e["after"] == last:
            out.append("kept")
        elif e["writer"] > writer and (e["after"] is None or e["after"] in held):
            t["held"] = t["held"][:held.index(e["after"]) + 1] if e["after"] is not None else []
            out.append("rewind")
        else:
            out.append("loss")
            continue
        t["held"].append(e)
        t["last"] = pos(e)
    result = {"results": out, "state": {tree: state_of(t["held"]) for tree, t in trees.items()}}
    if recover is not None:
        assert len(trees) == 1
        t = next(iter(trees.values()))
        source, reads = recover["source"], []
        answer = None
        if recover["form"] == "same":
            answer = resume(source, t["last"])
            reads.append({"after": t["last"], "expect": answer})
        if answer is None or "refuses" in answer:
            answer = resume(source, None)
            reads.append({"after": None, "expect": answer})
            held = []
        else:
            held = t["held"]
        got = {canonical(p): x for p, x in ((pos(x), x) for x in source)}
        held = held + [got[canonical(p)] for p in answer["events"]]
        result["recover"] = {"reads": reads, "state": state_of(held)}
    return result


def relink(events: list, last=None) -> list:
    """A form of a log: each event's after is the position of the event before it in this form (``last``
    before the first: a form that goes on after events it does not list)."""
    out = []
    for e in events:
        e = copy.deepcopy(e)
        e["after"] = last
        last = pos(e)
        out.append(e)
    return out


# ------------------------------------------------------------------ the rules: the kept form

ERROR_KEYS = ("type", "message", "code")
PROGRAM_KEYS = ("name", "kind", "module", "version", "signature", "interface", "answer", "saved", "file", "line")
SAW_KEYS = {"call", "steps", "without", "slot", "saw_of"}


def known(e: dict) -> dict:
    """What a form maker keeps of an event of a kind it knows: the keys it knows, and, inside the protocol
    objects it holds (error, program, saw's entries), the members it knows (it cannot know whether another
    holds a value). A saw entry it does not know becomes {}: an entry no reader knows, holding nothing, so a
    reader still cannot know what was shown."""
    out = {k: copy.deepcopy(v) for k, v in e.items() if k in ENVELOPE or k in KEYS[e["kind"]]}
    if isinstance(out.get("error"), dict):
        out["error"] = {k: v for k, v in out["error"].items() if k in ERROR_KEYS}
    if isinstance(out.get("program"), dict):
        out["program"] = {k: v for k, v in out["program"].items() if k in PROGRAM_KEYS}
    if isinstance(out.get("saw"), list):
        out["saw"] = [x if isinstance(x, dict) and set(x) <= SAW_KEYS else {} for x in out["saw"]]
    return out


def kept_event(e: dict, keep: dict, info: dict):
    """The kept form of one event (None: not in the kept log)."""
    if e["kind"] not in KINDS:
        return None
    fields = {**keep["inputs"], **keep["outputs"]}
    out = known(e)
    if all(fields.values()):
        return out
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
        out["error"] = {k: v for k, v in out["error"].items() if k in ("type", "code")}
    out["content"] = False
    return out


def kept_log(events: list, kept: dict) -> list:
    info = {e["call"]: e["program"] for e in events if e["kind"] == "started"}
    out = [kept_event(e, kept[e["call"]], info.get(e["call"])) for e in events]
    return relink([e for e in out if e is not None])


# ------------------------------------------------------------------ the rules: a store


class Store:
    """Anything that keeps logs for others to read ("The rules a store keeps"). Each claim and each append
    is one step, in one order with every other claim and append of the log (here: one method call)."""

    def __init__(self):
        self.logs, self.writers = {}, {}

    def finished(self, tree: str) -> bool:
        log = self.logs.get(tree, [])
        return bool(log) and log[-1]["call"] == tree and log[-1]["kind"] in ("done", "failed")

    def claim(self, tree: str) -> dict:
        """A writer that continues a log asks for its writer number: one more than any given for it; the
        store names the last kept event."""
        log = self.logs.get(tree, [])
        if not log:
            return {"refuses": "event-unknown"}
        if self.finished(tree):
            return {"refuses": "event-after-end"}
        self.writers[tree] = self.writers.get(tree, 1) + 1
        return {"writer": self.writers[tree], "after": pos(log[-1])}

    def append(self, e: dict) -> str:
        log = self.logs.setdefault(e["tree"], [])
        after = e["after"]
        if after is not None and (e["seq"] <= after["seq"] or after["writer"] > e["writer"]):
            return "event-malformed"
        if log and e["writer"] != self.writers.get(e["tree"], 1):
            return "event-conflict"                  # fenced, or a number the store never gave: before all else
        have = {k["seq"]: k for k in log}
        if e["seq"] in have:
            return "duplicate" if canonical(have[e["seq"]]) == canonical(e) else "event-conflict"
        if self.finished(e["tree"]):
            return "event-after-end"
        last = pos(log[-1]) if log else None
        if after != last:
            return "event-gap" if (after["seq"] if after else 0) > (last["seq"] if last else 0) else "event-conflict"
        if not log and (e["kind"] != "started" or e["call"] != e["tree"] or e["writer"] != 1):
            return "event-start"
        log.append(copy.deepcopy(e))
        return "kept"

    def read(self, tree: str, after) -> dict:
        return resume(self.logs.get(tree, []), after)

    def summary(self) -> dict:
        return {k: {"events": [pos(e) for e in v], "writer": self.writers.get(k, 1), "finished": self.finished(k)}
                for k, v in self.logs.items() if v}


def store(steps: list, reads=()) -> tuple:
    """Each step appends an event, or claims a log for a later writer."""
    s = Store()
    results = [s.append(x) if x.get("kind") else s.claim(x["claim"]) for x in steps]
    return results, s.summary(), [{**r, "expect": s.read(r["tree"], r["after"])} for r in reads]


def settle(s: Store, tree: str, end: dict) -> str:
    """What reading the journal says of an end a writer could not confirm ("A required journal that does not
    confirm"): the log holds that event (its writer and seq), or not, and then whether another writer ended
    the log."""
    if any(pos(e) == pos(end) for e in s.logs.get(tree, [])):
        return "kept"
    return "another-end" if s.finished(tree) else "not-kept"


# ------------------------------------------------------------------ the rules: receivers across layers


def receivers(layers: list) -> dict:
    """The observers and the journal a tree gets from the layers around its outermost call, closest first."""
    observers = [o for layer in reversed(layers) for o in layer.get("observers", [])]
    setting = [layer["journal"] for layer in layers if "journal" in layer]
    for i, j in enumerate(setting):
        if j is not None and j["mode"] == "required" and any(k != j for k in setting[:i]):
            return {"refuses": "journal-policy"}
    return {"observers": observers, "journal": setting[0] if setting else None}


# ------------------------------------------------------------------ the rules: a writer and its journal


def journal(events: list, script: list, *, retries: int = 2, required: bool = True) -> dict:
    """A writer keeping one call's log (an AI function's, the tree's only call) in a journal.

    Required: the call waits at three barriers, until every event up to it is confirmed: its started
    (before anything runs), each tool_call (before the tool runs), its terminal event (before it returns).
    An event is confirmed when the journal answers kept or duplicate; refused when it answers anything
    else (the writer then appends nothing more); unanswered after 1 + retries sends in a row. A barrier
    not passed stops the call: its outcome is the error JournalError (journal-barrier), and its terminal
    event says so. The terminal event is shown to readers only once confirmed; when it is not, the caller
    gets JournalError (journal-end) holding the outcome and naming the event, and the record says whether
    the journal refused it or did not answer. Best effort: no barrier; the call ends with its outcome, and
    every event is shown as it is made. Between attempts another writer may claim the log ("claimed"), or
    claim it and end it ("ended")."""
    tree = events[0]["tree"]
    transport = iter(script)
    s = Store()
    trace, shown, made = [], [], []
    pending, refused = [], False

    def other(t):
        claim = s.claim(tree)
        trace.append({"other": t, "writer": claim.get("writer")})
        if t == "ended" and "writer" in claim:
            w = Writer(1, first_second=50).continuing(s.logs[tree], claim)
            assert s.append(w.failed(1, {"type": "Cancelled"})) == "kept"

    def attempt():
        t = next(transport, "ok")
        while t in ("claimed", "ended"):
            other(t)
            t = next(transport, "ok")
        return t

    def confirm() -> str:
        nonlocal refused
        while pending:
            e, answer = pending[0], None
            for _ in range(1 + retries):
                t = attempt()
                if t == "down":
                    trace.append({"seq": e["seq"], "transport": t, "answer": None})
                    continue
                if t == "conflict":
                    trace.append({"seq": e["seq"], "transport": t, "answer": "event-conflict"})
                    refused = True
                    return "refused"
                result = s.append(e)
                trace.append({"seq": e["seq"], "transport": t, "answer": None if t == "lost" else result})
                if t == "lost":
                    continue
                answer = result
                break
            if answer is None:
                return "unanswered"
            if answer not in ("kept", "duplicate"):
                refused = True
                return "refused"
            pending.pop(0)
        return "confirmed"

    outcome = None
    for i, e in enumerate(events):
        if e["call"] == tree and e["kind"] in ("done", "failed"):
            outcome = e
            break
        made.append(e)
        shown.append(e["seq"])
        if not refused:
            pending.append(e)
            confirm()
        barrier = required and (i == 0 or e["kind"] == "tool_call")
        if barrier and (refused or pending):
            w = Writer(0)
            w.tree, w.writer, w.seq, w.last, w.names = tree, e["writer"], e["seq"], pos(e), {1: e["function"]}
            outcome = w.failed(1, {"type": "JournalError", "code": "journal-barrier",
                                   "message": f"the journal did not keep event {e['seq']}"})
            break
    made.append(outcome)
    status = "refused"
    if not refused:
        pending.append(outcome)
        status = confirm()
    if not required or status == "confirmed":
        shown.append(outcome["seq"])
    error = None if outcome["kind"] == "done" else {k: v for k, v in outcome["error"].items() if k != "message"}
    ended = {"done": outcome["value"]} if outcome["kind"] == "done" else {"failed": error}
    record = {"error": error}
    settled = None
    if required and status != "confirmed":
        journal_ = "refused" if status == "refused" else "unknown"
        caller = {"raises": {"type": "JournalError", "code": "journal-end", "journal": journal_,
                             "event": pos(outcome), "outcome": ended}}
        record["journal"] = journal_
        if journal_ == "unknown":
            settled = settle(s, tree, outcome)
    else:
        caller = {"returns": outcome["value"]} if outcome["kind"] == "done" else {"raises": error}
    kept = s.summary().get(tree, {"events": [], "finished": False})
    out = {"log": made, "trace": trace, "shown": shown,
           "kept": {"events": kept["events"], "finished": kept["finished"]}, "caller": caller, "record": record}
    if settled is not None:
        out["settled"] = settled
    return out


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
            "resume": [{"after": n, "expect": resume(events, n)} for n in resume_after]}


def follow_case(description, received, recover=None):
    case = {"description": description, "kind": "follow", "received": received}
    if recover is not None:
        case["recover"] = recover
    return case | {"expect": follow(received, recover)}


def kept_case(description, w: Writer, kept):
    kept = {cid(n): k for n, k in kept.items()}
    return {"description": description, "kind": "kept", "events": w.events, "kept": kept,
            "expect": {"events": kept_log(w.events, kept)}}


def store_case(description, steps, reads=()):
    results, logs, reads = store(steps, reads)
    return {"description": description, "kind": "store",
            "steps": [({"append": x} if x.get("kind") else x) | {"expect": r} for x, r in zip(steps, results)],
            "reads": reads, "expect": {"logs": logs}}


def claim(tree: int) -> dict:
    return {"claim": cid(tree)}


def journal_case(description, events, script, *, retries=2, required=True):
    return {"description": description, "kind": "journal", "mode": "required" if required else "best-effort",
            "retries": retries, "events": events, "script": script,
            "expect": journal(events, script, retries=retries, required=required)}


def io(inputs: dict, outputs: dict) -> dict:
    return {"inputs": inputs, "outputs": outputs}


def claimed(kept: list) -> dict:
    """What a store holding a kept log answers the first writer that claims it."""
    s = Store()
    for e in kept:
        assert s.append(e) == "kept", e
    return s.claim(kept[0]["tree"])


def a_later_kind(events: list, index: int, **extra) -> list:
    """A log in which a later writer's event of a kind this contract does not name comes before ``index``."""
    e = copy.deepcopy(events[index - 1])
    for k in list(e):
        if k not in ENVELOPE:
            del e[k]
    e.update(kind="approval", asked="May I cancel order B-2210?", **extra)
    out = copy.deepcopy(events[:index]) + [e] + copy.deepcopy(events[index:])
    for i, x in enumerate(out):
        x["seq"] = i + 1
    return relink(out)


def handed_over():
    """A writer whose last events were never kept (a dropped field's pieces), and a later writer that goes on
    from what the journal kept: vignette 4's turn resumed after a restart."""
    first = Writer(1)
    first.started(1, "gardener", {"request": "Merge groceries.md into todo.md"})
    first.request(1)
    first.text(1, "result", "Merging ")
    first.text(1, "result", "the two ")
    first.text(1, "result", "notes")
    kept = kept_log(first.events, {cid(1): io({"request": True}, {"result": False})})
    second = Writer(1, first_second=30).continuing(kept, claimed(kept))
    second.request(1)
    second.text(1, "result", "Done.")
    second.done(1, "Done.")
    return first, kept, second


def kept_continued(kept: list, events: list, keep: dict) -> list:
    """The kept form of a later writer's events, going on after the kept log it claimed."""
    info = {e["call"]: e["program"] for e in kept if e["kind"] == "started"}
    out = [kept_event(e, keep[e["call"]], info.get(e["call"])) for e in events]
    return relink([e for e in out if e is not None], pos(kept[-1]))


def hand_overs() -> dict:
    """Readers that miss events where a later writer took over. Each: (description, what the reader
    received, how it recovers, the state the store's log replays to)."""
    out = {}
    everything = {cid(1): io({"request": True}, {"result": True})}

    a = Writer(1)
    a.started(1, "gardener", {"request": "Merge groceries.md into todo.md"})
    a.request(1)
    a.text(1, "result", "Merging the two")                  # (1, 3): never kept
    kept = a.events[:2]
    b = Writer(1, first_second=30).continuing(kept, claimed(kept))
    b.request(1)                                            # (2, 3), after (1, 2): the reader misses it
    b.text(1, "result", "Merged.")                          # (2, 4), after (2, 3)
    b.done(1, "Merged.")
    stored = kept + b.events
    out["follow-09-a-hand-over-and-one-missed-event"] = (
        "An observer's feed (the kept form; every value is kept). The reader holds writer 1's seq 3, which "
        "was never kept; writer 2 took over after seq 2 and the reader misses its first event (writer 2, seq "
        "3). Writer 2's seq 4 names writer 2's seq 3 as its after: not the reader's last (writer 1, seq 3), "
        "and not an event it holds, so events were lost (a seq alone would call it next, and the reader would "
        "end with writer 1's text before writer 2's). It reads its form again after its last event: the store "
        "does not have it (event-unknown), so it starts again from the beginning, and ends where the store "
        "ends.", a.events + b.events[1:], {"form": "same", "source": stored}, state_of(stored))

    first, kept, second = handed_over()
    stored = kept + kept_continued(kept, second.events, {cid(1): io({"request": True}, {"result": False})})
    out["follow-10-a-hand-over-and-missed-events"] = (
        "A reader of the whole log holds writer 1's seqs 1 to 5 (pieces of a field the log does not keep); "
        "writer 2 took over after seq 2, and the reader misses its seqs 3 and 4. Writer 2's done names "
        "writer 2's seq 4: the reader holds writer 1's seq 4, another event, so this is a loss, not a rewind. "
        "The store gives another form (the kept form): the reader starts again from the beginning in it.",
        first.events + second.events[2:], {"form": "other", "source": stored}, state_of(stored))

    a = Writer(1)
    a.started(1, "gardener", {"request": "Merge groceries.md into todo.md"})
    a.request(1)
    a.text(1, "result", "OLD")
    a.text(1, "result", " TAIL")
    kept1 = a.events[:2]
    b = Writer(1, first_second=20).continuing(kept1, claimed(kept1))
    b.request(1)
    b.text(1, "result", "NEW")
    kept2 = kept1 + b.events
    s = Store()
    for e in kept1:
        s.append(e)
    s.claim(cid(1))
    for e in b.events:
        assert s.append(e) == "kept"
    c = Writer(1, first_second=40).continuing(kept2, s.claim(cid(1)))
    c.failed(1, {"type": "Cancelled"})
    stored = kept2 + c.events
    out["follow-11-two-hand-overs"] = (
        "Two hand-overs, and the reader (an observer's feed, every value kept) saw nothing of writer 2. It "
        "holds writer 1's seqs 1 to 4 (3 and 4 never kept); writer 3's failed names writer 2's seq 4. Seq 4 is "
        "the reader's last seq, but not its last event: a loss. It resumes from the store, which does not "
        "have writer 1's seq 4, and starts again. A later writer number alone never says that what the "
        "reader holds comes before the event (writer 2's events are missing between).",
        a.events + c.events, {"form": "same", "source": stored}, state_of(stored))

    more = copy.deepcopy(a)
    more.text(1, "result", " MORE")
    more.text(1, "result", " AGAIN")
    out["follow-12-two-hand-overs-where-a-seq-would-rewind"] = (
        "The same, when the reader holds writer 1's seqs 1 to 6: writer 3's failed names writer 2's seq 4, "
        "below the reader's last seq. The reader holds a seq 4, but writer 1's: it does not rewind to it (a "
        "seq alone would, and keep writer 1's text), since the event named is not one it holds. A loss.",
        more.events + c.events, {"form": "same", "source": stored}, state_of(stored))

    a = Writer(1)
    a.started(1, "gardener", {"request": "Merge groceries.md into todo.md"})
    a.request(1)
    a.text(1, "result", "Merging ")                         # (1, 3): never kept
    a.text(1, "result", "the notes")                        # (1, 4): never kept
    a.thinking(1, "Check for duplicates.")                  # (1, 5): the view hides it
    kept = a.events[:2]
    b = Writer(1, first_second=30).continuing(kept, claimed(kept))
    b.thinking(1, "The worker restarted.")                  # (2, 3): the view hides it
    b.request(1)                                            # (2, 4): the reader misses it
    b.text(1, "result", "Merged.")
    b.done(1, "Merged.")
    mine = hide(a.events)
    theirs = hide(kept + b.events)
    stored = kept + b.events
    out["follow-13-a-view-across-a-hand-over"] = (
        "A view that hides thinking (made from the kept form, every value kept). Writer 2's first event is "
        "hidden, so the view first shows writer 2 at its seq 4, after writer 1's seq 2; the reader misses it. "
        "Writer 2's seq 5 names writer 2's seq 4, while the reader's last is writer 1's seq 4: a loss (a seq "
        "alone would call it next). The store makes the same view from what it keeps, does not have writer "
        "1's seq 4, and the reader starts again.",
        mine + theirs[3:], {"form": "same", "source": hide(stored)}, state_of(hide(stored)))
    return out


def cases() -> dict:
    out = {}
    out["replay-01-one-call"] = replay_case(
        "One call: its answer's pieces add up; the log is finished when the tree's outermost call is done. A "
        "reader that has events up to one (named by its writer and seq) gets those after it; a position the log does "
        "not have (writer 2's seq 3) is event-unknown.", one_call().events,
        resume_after=(None, at(1, 2), at(1, 5), at(2, 3)))
    out["replay-02-a-retry-voids-the-text"] = replay_case(
        "A retry empties the call's fields at once; so does every request.", a_retry().events)
    out["replay-03-each-request-starts-afresh"] = replay_case(
        "The request after tool results empties the call's fields: the reply writes them afresh. Thinking is "
        "shown as it comes and never voided.", a_tool().events)
    out["replay-04-calls-inside-a-module"] = replay_case(
        "Calls inside a module are followed each on its own: one call's retry voids its own text only; "
        "seq numbers the one order of the tree's events.", a_module().events, resume_after=(at(1, 5), at(1, 12)))
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
        resume_after=(at(1, 3),), whole=w.events)
    w = a_tool()
    out["replay-09-a-view-without-tools"] = replay_case(
        "The same, for the request after tool results: seqs skip, after links each event to the one before "
        "it in the view.", hide(w.events), resume_after=(at(1, 4),), whole=w.events)
    w = a_retry_then_nothing()
    out["replay-10-a-retry-that-writes-nothing"] = replay_case(
        "A retry whose request fails before writing: the view empties the text at the retry, as the whole "
        "log does, though no later piece comes.", hide(w.events), whole=w.events)
    w = a_tool_last()
    out["replay-11-a-hidden-tail"] = replay_case(
        "A view whose last shown text is followed by hidden tool traffic: the request after it still "
        "empties the field. A reader resuming after the last event it saw gets the rest of the view.",
        hide(w.events), resume_after=(at(1, 3),), whole=w.events)
    w = a_module()
    out["replay-12-a-boundary"] = replay_case(
        "A view that shows only the tree's outermost call: a module's code writes no pieces, so the view "
        "shows its start and its end (what a module's answer shows while it is written is stage 3's).",
        boundary(w.events), whole=w.events)
    out["replay-13-a-kept-log"] = replay_case(
        "A kept log whose answer was dropped: its pieces are not there, the other output's are.",
        kept_log(a_transcript().events, {cid(1): io({"transcript": True, "question": True},
                                                  {"summary": True, "result": False})}))
    out["replay-14-a-kind-this-reader-does-not-know"] = replay_case(
        "An event of a kind this reader does not know (a later stage's approval) changes nothing: the state "
        "after it is the state before it.", a_later_kind(a_tool().events, 5))

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
    first, kept, second = handed_over()
    out["follow-03-a-later-writer-goes-on"] = follow_case(
        "A reader of the whole log had the first writer's pieces of a field the journal does not keep; that "
        "writer stopped, and a later one went on from the last kept event (seq 2), numbering its events from 3 "
        "again with writer 2. Its first event's after (2) is below the reader's last (5): the reader drops "
        "what it has after 2 and takes it (rewind), then follows the new writer to its end.",
        first.events + second.events)
    a = one_call().events[:3]
    b = Writer(1, first_second=30).continuing(a[:2], claimed(a[:2]))
    b.request(1)
    b.text(1, "result", "shipping")
    b.done(1, "shipping")
    out["follow-04-a-tail-never-kept"] = follow_case(
        "The same when every value is kept but the journal had not acknowledged the last event (seq 3) when "
        "its writer stopped: the later writer's seq 3 is another event. An event of the earlier writer that "
        "comes after the later writer's is stale: dropped.", a + b.events[:1] + [a[2]] + b.events[1:])
    m = a_module().events
    inner = relink([e for e in m if e["call"] == cid(2)])
    out["follow-05-a-stream-inside-a-tree"] = follow_case(
        "A stream opened on a call inside a tree (law 7) shows the tree's seqs; its first event's after is 0, "
        "each after skips the events of calls outside it.", inner)
    out["follow-06-a-kind-this-reader-does-not-know"] = follow_case(
        "A reader takes the place of an event of a kind it does not know (the next event's after names it), "
        "and changes nothing else.", a_later_kind(a_tool().events, 5))
    later = copy.deepcopy(whole_log[4])
    later["functai_event"] = 3
    out["follow-07-a-format-this-reader-does-not-know"] = follow_case(
        "An event of a format this reader does not know: it cannot know what the event's numbers mean, so it "
        "stops following there.", whole_log[:4] + [later])
    two = Writer(2)
    two.started(2, "team", {"message": "Hello"})
    two.request(2)
    two.done(2, "hi")
    mixed = [whole_log[0], two.events[0], whole_log[1], two.events[1], whole_log[2], two.events[2]]
    out["follow-08-each-tree-its-own"] = follow_case(
        "A reader of several logs (an observer's) keeps a last event for each tree.", mixed)

    # A reader that misses events at a hand-over: both third reviews' traces. Under a rule that compared a
    # bare seq, each of these ended with a finished state the store does not hold.
    for name, (description, received, recover, store_state) in hand_overs().items():
        case = follow_case(description, received, recover)
        assert "loss" in case["expect"]["results"] and "rewind" not in case["expect"]["results"], name
        assert case["expect"]["recover"]["state"] == store_state, name
        out[name] = case

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
    w.events = a_later_kind(w.events, 5)
    w.events[2]["tokens"] = 7
    out["kept-09-what-the-maker-does-not-know"] = kept_case(
        "Every value kept, but an event of a kind the form's maker does not know (approval) and a key it does "
        "not know (tokens) on an event it knows: neither is kept, since either may hold a value.", w,
        {1: io({"question": True}, {"result": True})})

    w = Writer(1)
    w.started(1, "triage_build", {"transcript": TRANSCRIPT}, saw=[{"call": cid(7)}, {"summary": "Ana is upset."}])
    w.events[0]["program"]["owner"] = "Ana"
    w.request(1)
    w.failed(1, {"type": "ProviderError", "message": "503", "code": "provider-down",
                 "detail": {"body": TRANSCRIPT[:20]}})
    out["kept-10-inside-what-the-maker-knows"] = kept_case(
        "Every value kept, but objects this contract defines carry members the maker does not know: the error "
        "a detail, the program an owner, saw an entry of a kind it does not know. The error and the program "
        "keep only their known members; the unknown saw entry becomes {} (still an entry no reader knows, so "
        "what the call saw stays not known, but nothing of it is passed on).", w,
        {1: io({"transcript": True}, {"result": True})})
    k = out["kept-10-inside-what-the-maker-knows"]["expect"]["events"]
    assert k[0]["saw"] == [{"call": cid(7)}, {}] and "owner" not in k[0]["program"] and "detail" not in k[-1]["error"]

    w = a_tool()
    ev = kept_log(w.events, {cid(1): io({"question": True}, {"result": True})})
    out["store-01-in-order"] = store_case(
        "Events appended in order are kept; the log is finished at its outermost call's done. A reader "
        "gets the events after the one it names.", ev,
        reads=[{"tree": cid(1), "after": None}, {"tree": cid(1), "after": at(1, 4)}])
    changed = copy.deepcopy(ev[3])
    changed["text"] = "Let me look."
    out["store-02-sent-again"] = store_case(
        "An event sent again after a failure changes nothing; another event at a kept seq is a second "
        "writer: refused.", ev[:5] + [copy.deepcopy(ev[3]), changed] + ev[5:])
    out["store-03-a-gap"] = store_case(
        "An event whose after is beyond the last kept is refused; the log goes on when the missing event "
        "comes.", ev[:3] + [ev[4], ev[3], ev[4]])
    after = copy.deepcopy(ev[-1])
    after.update(seq=after["seq"] + 1, after=pos(ev[-1]), kind="text", field="result", answer=True, text="!")
    del after["value"]
    out["store-04-nothing-after-the-end"] = store_case(
        "Nothing is kept after the outermost call's end; its last event sent again is still a duplicate.",
        ev + [after, copy.deepcopy(ev[-1])])
    m = a_module().events
    child_first = copy.deepcopy(m[1])
    child_first.update(seq=1, after=None, tree=cid(2))
    wrong_call = copy.deepcopy(m[1])
    wrong_call.update(seq=1, after=None)
    text_first = copy.deepcopy(m[5])
    text_first.update(seq=1, after=None, tree=cid(9))
    later_first = copy.deepcopy(m[0])
    later_first.update(tree=cid(8), call=cid(8), root=cid(8), writer=2)
    out["store-05-a-log-starts-with-its-tree"] = store_case(
        "A log's first event is its outermost call's started, from writer 1: a started of another call, "
        "another kind, or another writer is refused.", [wrong_call, text_first, later_first, child_first])
    other = Writer(2)
    other.started(2, "team", {"message": "Hello"})
    other.request(2)
    other.text(2, "result", "other")
    other.done(2, "other")
    mixed = [ev[0], other.events[0], ev[1], other.events[1], ev[2], other.events[2]]
    out["store-06-trees-are-numbered-apart"] = store_case("Two trees in one store are numbered each on its own.",
                                                          mixed)
    whole_transcript = a_transcript().events
    sparse = kept_log(whole_transcript, {cid(1): io({"transcript": False, "question": True},
                                                     {"summary": True, "result": True})})
    out["store-07-a-kept-log-skips"] = store_case(
        "A kept log leaves out events (thinking here), so its seqs skip; each event's after says what comes "
        "before it, and the store checks that. A reader that names an event the store does not have (seq 3, "
        "the thinking a live reader of the whole log saw) is told so (event-unknown): it starts again from "
        "the beginning, in the form the store keeps; it never waits for a chain that cannot come.", sparse,
        reads=[{"tree": cid(1), "after": at(1, 3)}, {"tree": cid(1), "after": at(1, 2)},
               {"tree": cid(1), "after": None}])
    first, kept, second = handed_over()
    early = copy.deepcopy(first.events[2])
    late = copy.deepcopy(first.events[4])
    late.update(after=at(1, 3))
    out["store-08-a-later-writer-goes-on"] = store_case(
        "A log whose writer stopped is continued by another (after a restart). The later writer claims the log: "
        "the store gives it writer number 2 and names the last kept event (seq 2), and from then on refuses "
        "every event of writer 1 (it is fenced), even one that would fit the chain (seq 3 after 2, sent late). "
        "The later writer numbers on from the last kept event: its seq 3 is not the first writer's seq 3 (a "
        "piece never kept). A reader that names the first writer's seq 3 is told the store does not have it.",
        kept + [claim(1), early] + second.events[:1] + [late] + second.events[1:],
        reads=[{"tree": cid(1), "after": at(1, 3)}, {"tree": cid(1), "after": at(2, 3)},
               {"tree": cid(1), "after": at(1, 2)}])
    bad = copy.deepcopy(ev[1])
    bad["after"] = pos(bad)
    later_after = copy.deepcopy(ev[1])
    later_after["after"] = at(2, 1)
    out["store-09-malformed"] = store_case(
        "An event whose seq is not greater than its after's is refused, and so is one whose after names an "
        "event of a later writer than its own (no form puts a later writer's event before an earlier one's).",
        [ev[0], bad, later_after, ev[1]])
    s = Store()
    for e in kept:
        s.append(e)
    two, three = s.claim(cid(1)), s.claim(cid(1))
    late_two = Writer(1, first_second=35).continuing(kept, two)
    late_two.request(1)
    race = Writer(1, first_second=40).continuing(kept, three)
    race.request(1)
    race.done(1, "Merged.")
    out["store-10-each-claim-fences-the-last"] = store_case(
        "Two writers claim one log (a lease, stage 2's, decides who may): each claim gives a new number and "
        "fences every earlier one, so two writers never hold one number. Writer 2 claimed and wrote nothing "
        "before writer 3 claimed (a claim may give a number no event ever carries): writer 3 numbers from what "
        "its own claim named, and writer 2's event is refused. A finished log cannot be claimed; nor can a log "
        "the store does not have.",
        kept + [claim(1), claim(1)] + race.events[:1] + late_two.events + race.events[1:] + [claim(1), claim(7)])
    w = a_tool()
    ev = kept_log(w.events, {cid(1): io({"question": True}, {"result": True})})
    tool_call = next(e for e in ev if e["kind"] == "tool_call")
    upto = ev[:ev.index(tool_call) + 1]
    out["store-11-a-fenced-writer-sends-again"] = store_case(
        "A writer's tool call is kept but its answer is lost; another writer claims the log; the first sends "
        "the tool call again. The store checks the writer before anything else: the resend is refused "
        "(event-conflict), not answered duplicate, so a fenced writer never passes a barrier. It learns what "
        "was kept by reading, which is never fenced.",
        upto + [claim(1), copy.deepcopy(tool_call)],
        reads=[{"tree": cid(1), "after": tool_call["after"]}])
    host = {"name": "chattering-store", "mode": "required"}
    folder = {"name": "notes-folder", "mode": "best-effort"}
    scenarios = [
        [{"where": "own", "observers": ["debug-print"]}, {"where": "block", "observers": ["page"]},
         {"where": "configure", "observers": ["telemetry"]}],
        [{"where": "own", "journal": folder}, {"where": "configure", "journal": {"name": "tmp", "mode": "best-effort"}}],
        [{"where": "own", "journal": None}, {"where": "configure", "journal": folder}],
        [{"where": "own", "journal": folder}, {"where": "configure", "journal": host}],
        [{"where": "own", "journal": None}, {"where": "configure", "journal": host}],
        [{"where": "own", "journal": {**host, "mode": "best-effort"}}, {"where": "configure", "journal": host}],
        [{"where": "own", "journal": host}, {"where": "block", "journal": host}, {"where": "configure"}],
        [{"where": "own", "journal": {**folder, "mode": "required"}}, {"where": "configure", "journal": folder}],
    ]
    out["receivers-01-layers"] = {
        "description": "Observers add up over every layer (a program's own is given events beside the host's). "
                       "The closest journal setting decides, null meaning none, except that a closer layer "
                       "cannot replace, weaken or remove a required journal set farther out: the tree's "
                       "outermost call is refused journal-policy before it runs. A closer layer may name the "
                       "same required journal, or make a best-effort one required.",
        "kind": "receivers",
        "scenarios": [{"layers": x, "expect": receivers(x)} for x in scenarios]}
    got = [x["expect"] for x in out["receivers-01-layers"]["scenarios"]]
    assert got[0]["observers"] == ["telemetry", "page", "debug-print"] and got[2]["journal"] is None
    assert [g.get("refuses") for g in got[3:6]] == ["journal-policy"] * 3 and got[6]["journal"] == host
    t = a_tool().events
    out["journal-01-every-event-kept"] = journal_case(
        "A required journal that keeps and acknowledges each event: the call waits at its start, before the "
        "tool runs and at its end, returns its value, and its done is shown once kept.", t, [])
    out["journal-02-the-end-refused"] = journal_case(
        "The journal refuses the call's done (another writer has the log): it is not kept. The outcome is "
        "still the call's value: the caller gets JournalError (journal-end) holding it, with journal "
        "\"refused\"; the record keeps the value and says journal \"refused\"; no reader is shown the done.",
        t, ["ok"] * 9 + ["conflict"])
    out["journal-03-the-end-kept-its-answer-lost"] = journal_case(
        "The journal keeps the done, but its answer is lost, and the resends get none: the writer cannot "
        "know whether it was kept. The caller gets JournalError (journal-end, journal \"unknown\") holding "
        "the value; the record keeps the value and says journal \"unknown\" (never a failure: the call did "
        "not fail). The store says finished, done: the three agree, and the store settles it.",
        t, ["ok"] * 9 + ["lost", "down", "down"])
    out["journal-04-the-end-not-kept-no-answer"] = journal_case(
        "The done never reaches the journal: to the writer this looks like journal-03 (no answer), so the "
        "caller and the record say the same; the store says unfinished.", t, ["ok"] * 9 + ["down"] * 3)
    out["journal-05-an-answer-lost-then-a-duplicate"] = journal_case(
        "The done's first answer is lost; sent again, the journal answers duplicate: it is kept, and the "
        "call returns its value.", t, ["ok"] * 9 + ["lost", "ok"])
    f = a_retry_then_nothing().events
    out["journal-06-a-failed-call-its-end-unanswered"] = journal_case(
        "A call that failed (the provider's error) and whose failed is kept but its answer lost: the "
        "outcome is that error; the caller gets JournalError (journal-end, \"unknown\") holding it, and "
        "the record keeps the provider's error as the call's error, with journal \"unknown\".",
        f, ["ok"] * 5 + ["lost", "down", "down"])
    out["journal-07-the-start-not-kept"] = journal_case(
        "The journal does not answer the call's started: nothing runs. The outcome is JournalError "
        "(journal-barrier); its failed is not kept either, so the caller gets JournalError (journal-end, "
        "\"unknown\") holding it.", t, ["down"] * 6)
    out["journal-08-a-tool-that-does-not-run"] = journal_case(
        "The journal does not answer the tool call: the tool does not run. The outcome is JournalError "
        "(journal-barrier). The journal comes back: the tool call and the failed are kept, and the caller "
        "gets that outcome as it is.", t, ["ok"] * 4 + ["down"] * 3)
    out["journal-09-refused-in-the-middle"] = journal_case(
        "The journal refuses an event in the middle (another writer took the log): the writer appends "
        "nothing more, the call stops at the next barrier (the tool), and its end cannot be kept: journal "
        "\"refused\".", t, ["ok"] * 2 + ["conflict"])
    out["journal-10-best-effort"] = journal_case(
        "A best-effort journal that does not answer the done: the call does not wait, returns its value, "
        "and every event is shown as it is made; the log stays unfinished.",
        t, ["ok"] * 9 + ["down"] * 3, required=False)
    out["journal-11-best-effort-its-end-kept-its-answer-lost"] = journal_case(
        "A best-effort journal keeps the done but its answer is lost: the call returns its value, and the "
        "log is finished. What the writer confirmed (up to seq 9) is all it knows; the store may hold more, "
        "the end included.", t, ["ok"] * 9 + ["lost", "down", "down"], required=False)
    out["journal-12-a-fenced-writer-does-not-run-the-tool"] = journal_case(
        "The tool call is kept, its answer lost; before the writer sends it again, another writer claims the "
        "log. The resend is refused (event-conflict: the store checks the writer first), so the barrier is "
        "not passed and the tool does not run: the outcome is JournalError (journal-barrier), which cannot be "
        "kept either (journal \"refused\").", t, ["ok"] * 4 + ["lost", "claimed", "ok"])
    out["journal-13-another-writer-ends-the-log"] = journal_case(
        "The done never reaches the journal; meanwhile another writer claims the log and ends it (Cancelled). "
        "The caller gets JournalError (journal-end, \"unknown\") naming the event it attempted (writer 1, seq "
        "10). Reading the journal settles it: the log is finished, but not with that event (seq 10 is writer "
        "2's): another writer ended it, and the caller's outcome was not kept. A finished log is not enough.",
        t, ["ok"] * 9 + ["down", "ended", "down", "down"])
    assert out["journal-13-another-writer-ends-the-log"]["expect"]["settled"] == "another-end"
    assert out["journal-03-the-end-kept-its-answer-lost"]["expect"]["settled"] == "kept"
    assert out["journal-04-the-end-not-kept-no-answer"]["expect"]["settled"] == "not-kept"
    assert out["journal-11-best-effort-its-end-kept-its-answer-lost"]["expect"]["kept"]["finished"]
    j = out["journal-12-a-fenced-writer-does-not-run-the-tool"]["expect"]
    assert j["log"][-1]["error"]["code"] == "journal-barrier" and j["log"][-2]["kind"] == "tool_call"
    return out
