"""Cases for stages 1.2 to 5, written from the rules (never from an
implementation's output):

    replies/         ../replies.md: the reply cache's key
    conversations/   ../conversations.md: a turn's state, the head, the parent a new turn takes, what a
                     turn is shown (its saw entries, earlier() rows)
    tools/           ../tools.md: which tool calls a rule asks about; what the model is told of a refusal
    views/           ../streaming.md, "Views": the outside view of a log
    context/         ../calls.md, "Rows that keep their context": a rated call's earlier turns

Each kind of case says what a harness gives the implementation and what it
must answer (cases/README.md).
"""

import copy

from common import canonical, sha

T0 = "2026-09-30T10:00:00.000000Z"
NOW = "2026-09-30T10:10:00.000000Z"
FUTURE = "2026-09-30T10:10:30.000000Z"
PAST = "2026-09-30T10:09:00.000000Z"
V1 = "sha256:" + "a" * 64
V2 = "sha256:" + "b" * 64
SIG = "sha256:" + "c" * 64


def uid(n: int) -> str:
    return f"01926d00-{n:04x}-7000-8000-000000000000"


# ------------------------------------------------------------------ replies


def reply_cases() -> dict:
    request = {"model": "gpt-4.1-mini", "messages": [{"role": "user", "parts": [{"type": "text", "text": "Chile"}]}]}
    out = {}
    for i, (n, why) in enumerate([(0, "the first answer to a request"),
                                  (2, "the third independent answer to the same request")], 1):
        out[f"{i:02d}-{'first' if n == 0 else 'a-replicate'}"] = {
            "description": f"The key of {why}: sha256 of the canonical JSON of functai_reply, the request, "
                           f"replicate.",
            "kind": "key", "request": request, "replicate": n,
            "expect": {"key": sha({"functai_reply": 1, "request": request, "replicate": n})}}
    return out


# ------------------------------------------------------------------ conversations: the rules


def R(kind: str, seq: int, **kw) -> dict:
    return {"functai_conversation": 1, "kind": kind, "seq": seq, "at": T0, **kw}


def program(seq: int, version: str = V1) -> dict:
    return R("program", seq, version=version, name="tutor", program_kind="ai", module="__main__",
             signature=SIG, answer="result",
             interface={"description": "Tutor.", "inputs": [{"name": "message", "shape": {"type": "string"}}],
                        "outputs": [{"name": "result", "shape": {"type": "string"}}]},
             fields=[{"name": "message", "direction": "input", "purpose": "plain", "shape": {"type": "string"}},
                     {"name": "result", "direction": "output", "purpose": "plain", "shape": {"type": "string"}}])


def state_of(records: list, turn: str, now: str) -> str:
    """conversations.md, *A turn's state*."""
    ended = next((r for r in records if r["kind"] == "ended" and r["turn"] == turn), None)
    if ended is not None:
        return ended["state"]
    leases = [r for r in records if r["kind"] == "lease" and r["turn"] == turn]
    waits = [r for r in records if r["kind"] == "waiting" and r["turn"] == turn]
    lease = leases[-1] if leases else None
    if waits and waits[-1]["seq"] > (lease["seq"] if lease else 0):
        return "waiting"
    return "running" if lease is not None and lease["until"] >= now else "interrupted"


def turns_of(records: list) -> dict:
    return {r["turn"]: r for r in records if r["kind"] == "turn"}


def head_of(records: list):
    head = None
    turns = turns_of(records)
    for r in records:
        if r["kind"] == "turn" or (r["kind"] == "head" and r["turn"] in turns):
            head = r["turn"]
    return head


def done_on(records: list, turn, now: str):
    turns = turns_of(records)
    while turn is not None and turn in turns:
        if state_of(records, turn, now) == "done":
            return turn
        turn = turns[turn]["parent"]
    return None


def parent_for(records: list, now: str):
    """conversations.md, *Where a new turn goes* (sends "queue", a conversation opened by id)."""
    head = head_of(records)
    if head is None:
        return {"parent": None}
    st = state_of(records, head, now)
    if st == "waiting":
        return {"refuses": "conversation-busy"}
    if st == "running":
        return {"waits": head}
    return {"parent": done_on(records, head, now)}


def unanswered(records: list, turn: str) -> list:
    waits = [r for r in records if r["kind"] == "waiting" and r["turn"] == turn]
    if not waits:
        return []
    w = waits[-1]
    answered = {(r["site"], r["invocation"]) for r in records
                if r["kind"] == "approval" and r["turn"] == turn and r["seq"] > w["seq"]}
    return [a["invocation"] for a in w["approvals"] if (a["site"], a["invocation"]) not in answered]


def unfinished(records: list, turn: str) -> list:
    started = {}
    for r in records:
        if r["kind"] == "tool" and r["turn"] == turn:
            k = (r["site"], r["invocation"])
            if r["state"] == "started":
                started[k] = r["invocation"]
            else:
                started.pop(k, None)
    return list(started.values())


def branch(records: list, turn) -> list:
    turns = turns_of(records)
    out = []
    while turn is not None and turn in turns:
        out.append(turn)
        turn = turns[turn]["parent"]
    return out[::-1]


def saw_expanded(saws: dict, call: str) -> list:
    out = []
    for i, e in enumerate(saws[call]):
        if "saw_of" in e:
            out += saw_expanded(saws, e["saw_of"])
        else:
            out.append(e)
    return out


def context(records: list, parent, rule: dict, now: str) -> dict:
    """conversations.md, *What a turn is shown*: the done turns of the branch up to ``parent``, the last
    ``rule.last`` of them (all when null), each without ``rule.without``; its saw entries (whole turns of
    the same program signature with their steps; ``saw_of`` the parent when it saw exactly the rest); the
    rows ``earlier()`` gives."""
    turns = turns_of(records)
    ended = {r["turn"]: r for r in records if r["kind"] == "ended"}
    programs = {r["version"]: r for r in records if r["kind"] == "program"}
    done = [t for t in branch(records, parent) if state_of(records, t, now) == "done"]
    if rule.get("last") is not None:
        done = done[-rule["last"]:] if rule["last"] > 0 else []
    without = set(rule.get("without") or [])
    entries, rows = [], []
    for t in done:
        rec, end = turns[t], ended[t]
        fields = set(rec["inputs"]) | set(end["outputs"])
        gone = sorted(fields & without)
        same = programs[rec["program"]].get("signature") == SIG
        e = {"call": t}
        if gone:
            e["without"] = gone
        elif same and end.get("lmcc", {}).get("steps"):
            e["steps"] = True
        entries.append(e)
        rows.append({k: v for k, v in {**rec["inputs"], **end["outputs"]}.items() if k not in without})
    saws = {t: ended[t].get("saw", []) for t in ended}
    if len(entries) >= 2 and parent is not None and entries[-1]["call"] == parent and parent in saws:
        if saw_expanded(saws, parent) == entries[:-1]:
            entries = [{"saw_of": parent}, entries[-1]]
    return {"saw": entries, "rows": rows}


def lmcc_turn(message: str, result: str) -> dict:
    return {"signature": "sha256:" + "d" * 64, "inputs": {"message": message},
            "steps": [{"kind": "model", "outputs": {"result": result}}], "outputs": {"result": result}}


def ended(seq: int, turn: str, result: str, message: str, saw: list) -> dict:
    return R("ended", seq, turn=turn, state="done", outputs={"result": result}, value=result,
             lmcc=lmcc_turn(message, result), saw=saw, attempt=1)


def a_conversation() -> list:
    """Three done turns in a row, then a branch from the first."""
    t1, t2, t3, t4 = uid(1), uid(2), uid(3), uid(4)
    return [program(1),
            R("turn", 2, turn=t1, parent=None, program=V1, inputs={"message": "Hi, I'm Alex."}),
            R("lease", 3, turn=t1, holder="h:1:a", until=FUTURE, attempt=1),
            ended(4, t1, "Welcome, Alex!", "Hi, I'm Alex.", []),
            R("turn", 5, turn=t2, parent=t1, program=V1, inputs={"message": "What is 1/2 + 1/3?"}),
            R("lease", 6, turn=t2, holder="h:1:a", until=FUTURE, attempt=1),
            ended(7, t2, "First, the bottoms.", "What is 1/2 + 1/3?", [{"call": t1, "steps": True}]),
            R("turn", 8, turn=t3, parent=t2, program=V1, inputs={"message": "Is it 2/5?"}),
            R("lease", 9, turn=t3, holder="h:1:a", until=FUTURE, attempt=1),
            ended(10, t3, "Not quite.", "Is it 2/5?",
                  [{"saw_of": t2}, {"call": t2, "steps": True}]),
            R("turn", 11, turn=t4, parent=t2, program=V1, inputs={"message": "Is it 5/6?"}),
            R("lease", 12, turn=t4, holder="h:1:a", until=FUTURE, attempt=1),
            ended(13, t4, "Yes!", "Is it 5/6?", [{"saw_of": t2}, {"call": t2, "steps": True}])]


def conversation_cases() -> dict:
    out = {}
    base = a_conversation()
    t1, t2, t3, t4 = uid(1), uid(2), uid(3), uid(4)

    def state_case(name, why, records):
        turns = list(turns_of(records))
        out[name] = {"description": why, "kind": "state", "records": records, "now": NOW,
                     "expect": {"states": {t: state_of(records, t, NOW) for t in turns},
                                "head": head_of(records), "next": parent_for(records, NOW),
                                "waiting": {t: unanswered(records, t) for t in turns if unanswered(records, t)},
                                "unfinished": {t: unfinished(records, t) for t in turns if unfinished(records, t)}}}

    state_case("state-01-done-turns-and-a-branch",
               "Every turn ended done; the head is the latest turn (a branch from the second); a new turn "
               "continues from it.", base)
    running = base + [R("turn", 14, turn=uid(5), parent=t4, program=V1, inputs={"message": "And 3/4?"}),
                      R("lease", 15, turn=uid(5), holder="h:1:a", until=FUTURE, attempt=1)]
    state_case("state-02-a-running-turn-is-waited-for",
               "The head runs (its lease holds): a new turn sent now waits for it (queue).", running)
    dead = base + [R("turn", 14, turn=uid(5), parent=t4, program=V1, inputs={"message": "And 3/4?"}),
                   R("lease", 15, turn=uid(5), holder="h:1:a", until=PAST, attempt=1)]
    state_case("state-03-a-lease-that-ran-out-is-interrupted",
               "The head's lease ran out and it never ended: interrupted. A turn that did not end done is never "
               "a parent: a new turn continues from its nearest done ancestor.", dead)
    failed = base + [R("turn", 14, turn=uid(5), parent=t4, program=V1, inputs={"message": "x"}),
                     R("lease", 15, turn=uid(5), holder="h:1:a", until=FUTURE, attempt=1),
                     R("ended", 16, turn=uid(5), state="failed", error={"type": "RateLimitError"}, attempt=1)]
    state_case("state-04-a-failed-turn-is-not-a-parent", "The head failed: a new turn continues before it.", failed)
    a = {"call": uid(6), "invocation": 2, "id": "c2", "name": "write_note", "input": {"name": "b.md"},
         "effects": "changes", "path": "gardener/write_note", "site": "gardener#1"}
    waiting = base + [R("turn", 14, turn=uid(6), parent=t4, program=V1, inputs={"message": "tidy"}),
                      R("lease", 15, turn=uid(6), holder="h:1:a", until=FUTURE, attempt=1),
                      R("tool", 16, turn=uid(6), site="gardener#1", invocation=1, id="c1", name="read_note",
                        state="done", output="text of a.md", attempt=1),
                      R("waiting", 17, turn=uid(6), approvals=[a], attempt=1)]
    state_case("state-05-a-waiting-turn",
               "A turn waits for a person's answer (after its lease was last renewed): a new turn is refused "
               "(conversation-busy) until it is answered.", waiting)
    answered = waiting + [R("approval", 18, turn=uid(6), site="gardener#1", invocation=2, path="gardener/write_note",
                            verdict="yes", by="maxime", reason=None)]
    state_case("state-06-an-answered-approval-waits-no-more",
               "Its approval answered, the turn still waits until a process resumes it (a lease after the "
               "waiting record); nothing is left unanswered.", answered)
    resumed = answered + [R("lease", 19, turn=uid(6), holder="h:2:b", until=FUTURE, attempt=2)]
    state_case("state-07-a-resumed-turn-runs", "A lease after the waiting record: the turn runs again.", resumed)
    crashed = base + [R("turn", 14, turn=uid(7), parent=t4, program=V1, inputs={"message": "tidy"}),
                      R("lease", 15, turn=uid(7), holder="h:1:a", until=PAST, attempt=1),
                      R("tool", 16, turn=uid(7), site="gardener#1", invocation=1, id="c1", name="read_note",
                        state="done", output="text", attempt=1),
                      R("tool", 17, turn=uid(7), site="gardener#1", invocation=2, id="c2", name="write_note",
                        input={"name": "b.md"}, effects="changes", state="started", attempt=1)]
    state_case("state-08-a-tool-that-may-have-run",
               "The process stopped while a tool that changes things ran: the turn is interrupted, and that "
               "tool is unfinished (it may have run).", crashed)
    moved = base + [R("head", 14, turn=t1)]
    state_case("state-09-a-head-record", "A head record makes an earlier turn the head.", moved)

    def ctx_case(name, why, records, parent, rule):
        out[name] = {"description": why, "kind": "context", "records": records, "now": NOW, "parent": parent,
                     "rule": rule, "expect": context(records, parent, rule, NOW)}

    ctx_case("context-01-every-earlier-turn",
             "Every earlier turn of the branch, whole with its steps; the parent saw exactly the rest, so saw_of.",
             base, t3, {"last": None})
    ctx_case("context-02-a-branch-sees-its-own-path", "A turn after the branch: never the other path.",
             base, t4, {"last": None})
    ctx_case("context-03-the-last-turns", "last_turns(1): only the parent.", base, t3, {"last": 1})
    ctx_case("context-04-without-a-field",
             "An input left out of every earlier turn: the entries name it, and show no steps.",
             base, t3, {"last": None, "without": ["message"]})
    ctx_case("context-05-the-first-turn", "The first turn sees nothing.", base, None, {"last": None})
    return out


# ------------------------------------------------------------------ tools


def asks(rule, a: dict) -> bool:
    """tools.md, *Which tool calls a rule asks about*."""
    if rule is None:
        return False
    if rule in ("changes", "function"):
        return a["effects"] != "reads"
    if rule == "all":
        return True
    return any(e == a["name"] or e == a["path"] or a["path"].endswith("/" + e) for e in rule)


def tool_cases() -> dict:
    approvals = [{"name": "refund", "path": "support/answer/refund", "effects": "changes"},
                 {"name": "order_status", "path": "support/answer/order_status", "effects": "reads"},
                 {"name": "send_email", "path": "support/send_email", "effects": None}]
    rules = [None, "changes", "all", "function", ["refund"], ["answer/refund"], ["support/answer/order_status"],
             ["email"]]
    out = {}
    for i, rule in enumerate(rules, 1):
        label = "none" if rule is None else rule if isinstance(rule, str) else "list-" + "-".join(
            r.replace("/", "-") for r in rule)
        out[f"asks-{i:02d}-{label}"] = {
            "description": "Which tool calls this rule asks a person about (a function is asked what "
                           "\"changes\" asks; a tool that says nothing counts as \"changes\"; a list names tools or "
                           "approval paths, or path endings).",
            "kind": "asks", "rule": rule, "approvals": approvals,
            "expect": {"asks": [asks(rule, a) for a in approvals]}}
    for i, reason in enumerate([None, "not that file"], 1):
        out[f"denial-{i:02d}"] = {"description": "What the model is shown when a person refuses a tool call.",
                                  "kind": "denial", "reason": reason,
                                  "expect": {"output": "The person did not allow this call."
                                             + (f" Reason: {reason}" if reason else "")}}
    return out


# ------------------------------------------------------------------ views


def E(kind: str, seq: int, call: str, function: str, **kw) -> dict:
    return {"functai_event": 2, "kind": kind, "tree": uid(100), "writer": 1, "seq": seq,
            "after": {"writer": 1, "seq": seq - 1} if seq > 1 else None, "at": T0, "call": call,
            "function": function, **kw}


PROGRAM_KEYS = ("name", "kind", "module", "version", "signature", "interface", "answer")


def outside(events: list, answer_from) -> list:
    """streaming.md, *Views*: the outside view."""
    root = events[0]["call"]
    root_fn = events[0]["function"]
    prog = events[0]["program"]
    kind, answer = prog["kind"], prog["answer"]
    out, forwarded, asked_caller, requests = [], {}, set(), 0
    last = None

    def link(e):
        nonlocal last
        e["after"] = last
        last = {"writer": e["writer"], "seq": e["seq"]}
        out.append(e)

    for ev in events:
        e = copy.deepcopy(ev)
        k, c = e["kind"], e["call"]
        if k in ("approval", "approved"):
            if k == "approval" and e.get("to") != "caller":
                continue
            if k == "approved" and (c, e["invocation"]) not in asked_caller:
                continue
            if k == "approval":
                asked_caller.add((c, e["invocation"]))
            e["call"], e["function"] = root, root_fn
            link(e)
            continue
        if c == root:
            if k == "started":
                e["program"] = {x: v for x, v in e["program"].items() if x in PROGRAM_KEYS}
                e.pop("invocation", None)
                link(e)
            elif k == "request" and kind == "ai":
                requests = max(requests, e["request"])
                link(e)
            elif k == "retry" and kind == "ai":
                e.pop("reason", None)
                e["content"] = False
                link(e)
            elif k == "text" and e["answer"]:
                link(e)
            elif k == "done":
                link(e)
            elif k == "failed":
                e["error"] = {x: v for x, v in e["error"].items() if x in ("type", "code")}
                e["content"] = False
                link(e)
            continue
        if kind == "ai" or answer_from is None:
            continue
        if k == "started":
            forwarded[c] = e["function"] == answer_from
        if not forwarded.get(c):
            continue
        if k in ("started", "request", "retry"):
            requests += 1
            link({"functai_event": 2, "tree": e["tree"], "writer": e["writer"], "seq": e["seq"], "at": e["at"],
                  "kind": "request", "call": root, "function": root_fn, "request": requests, "model": None})
        elif k == "text" and e["answer"]:
            e["call"], e["function"], e["field"] = root, root_fn, answer
            link(e)
    return out


def view_cases() -> dict:
    m, top, ans = uid(100), uid(101), uid(102)
    iface = {"description": "Support.", "inputs": [{"name": "message", "shape": {"type": "string"}}],
             "outputs": [{"name": "result", "shape": {"type": "string"}}]}
    mprog = {"name": "support", "kind": "module", "module": "shop", "version": V1, "interface": SIG,
             "answer": "result", "file": "/srv/shop/support.py", "line": 12}
    aprog = {"name": "answer", "kind": "ai", "module": "shop", "version": V2, "signature": SIG, "interface": SIG,
             "answer": "result"}
    started = dict(parent=None, root=m, program=mprog, inputs={"message": "Where is B-2210?"}, content=True, saw=[])
    events = [
        E("started", 1, m, "support", **started),
        E("started", 2, top, "topic", parent=m, root=m, program={**aprog, "name": "topic"},
          inputs={"message": "Where is B-2210?"}, content=True, saw=[]),
        E("request", 3, top, "topic", request=1, model="gpt-4.1-mini"),
        E("thinking", 4, top, "topic", text="shipping, surely"),
        E("text", 5, top, "topic", field="result", answer=True, text="shipping"),
        E("done", 6, top, "topic", value="shipping"),
        E("started", 7, ans, "answer", parent=m, root=m, program=aprog,
          inputs={"message": "Where is B-2210?", "topic": "shipping"}, content=True, saw=[]),
        E("request", 8, ans, "answer", request=1, model="gpt-4.1-mini"),
        E("tool_call", 9, ans, "answer", id="c1", name="refund", input={"order": "B-2210"}, invocation=1),
        E("approval", 10, ans, "answer", id="c1", invocation=1, name="refund", input={"order": "B-2210"},
          effects="changes", path="support/answer/refund", to="caller"),
        E("approved", 11, ans, "answer", id="c1", invocation=1, verdict="yes", by="ana", reason=None),
        E("tool_result", 12, ans, "answer", id="c1", name="refund", output="refunded", invocation=1),
        E("request", 13, ans, "answer", request=2, model="gpt-4.1-mini"),
        E("text", 14, ans, "answer", field="result", answer=True, text="Refunded: "),
        E("text", 15, ans, "answer", field="result", answer=True, text="it is in Leeds."),
        E("done", 16, ans, "answer", value="Refunded: it is in Leeds."),
        E("done", 17, m, "support", value="Refunded: it is in Leeds.")]
    ai_events = [
        E("started", 1, m, "answer", parent=None, root=m, program=aprog, inputs={"message": "hi"}, content=True,
          saw=[]),
        E("request", 2, m, "answer", request=1, model="gpt-4.1-mini"),
        E("text", 3, m, "answer", field="reasoning", answer=False, text="they greet"),
        E("text", 4, m, "answer", field="result", answer=True, text="Hel"),
        E("retry", 5, m, "answer", reason="the reply could not be read (it quoted the SECRET log)", wait=None),
        E("request", 6, m, "answer", request=2, model="gpt-4.1-mini"),
        E("text", 7, m, "answer", field="result", answer=True, text="Hello!"),
        E("failed", 8, m, "answer", error={"type": "ToolError", "message": "could not read: SECRET", "code": "x-1"})]
    out = {}
    for name, why, evs, af in [
            ("outside-01-a-module", "A module's boundary: its start (the program named without its file), its "
             "end, and an approval addressed to the caller. Its helpers' answers, thinking and tool calls are "
             "never shown.", events, None),
            ("outside-02-a-module-s-answer-as-it-is-written", "answer_from=answer: that helper's answer text, "
             "re-addressed to the module; each of its requests a request of the module.", events, "answer"),
            ("outside-03-an-ai-function", "An AI function's boundary: its answer's text (not its reasoning), "
             "requests and retries without their reason, and a failure's type and code, no message.",
             ai_events, None)]:
        out[name] = {"description": why, "kind": "view", "view": "outside", "answer_from": af, "events": evs,
                     "expect": {"events": outside(evs, af)}}
    return out


# ------------------------------------------------------------------ rows that keep their context


def call_rec(i: int, *, parent=None, root=None, name="tutor", kind="ai", inputs=None, outputs=None, saw=None,
             steps=None, conversation=None, content=True) -> dict:
    rec = {"functai_call": 2, "id": uid(200 + i), "parent": parent, "root": root or uid(200 + i),
           "program": {"name": name, "kind": kind, "module": "__main__", "version": V1, "interface": SIG,
                       "answer": "result", **({"signature": SIG} if kind == "ai" else {})},
           "started": f"2026-09-30T10:00:{i:02d}.000000Z", "seconds": 0.1, "content": content,
           "inputs": inputs or {}, "outputs": outputs, "sizes": {"inputs": {k: 1 for k in (inputs or {})},
                                                                 "outputs": {k: 1 for k in (outputs or {})}},
           "error": None, "model": "gpt-4.1-mini", "usage": {}, "exchanges": [], "saw": saw or [],
           "caller": {}, "process": {"host": "h", "pid": 1, "language": "python", "runtime": "3.13",
                                     "functai": "1.2.0"}}
    if steps is not None:
        rec["steps"] = steps
    if conversation is not None:
        rec["conversation"] = conversation
    if not content:
        rec["omitted"] = {"inputs": list(inputs or {}), "outputs": list(outputs or {})}
        rec.pop("inputs")
        rec.pop("outputs")
    return rec


def turn_of(rec: dict, entry: dict):
    """calls.md, *Rows that keep their context*: the turn an entry stands for, as data."""
    left = set(entry.get("without") or [])
    t = {"inputs": {k: v for k, v in rec["inputs"].items() if k not in left},
         "outputs": {k: v for k, v in (rec["outputs"] or {}).items() if k not in left}}
    if entry.get("steps"):
        if "steps" not in rec:
            return None
        t["steps"] = rec["steps"]
        t["signature"] = rec["program"]["signature"]
    return t


def context_rows(records: list, call: str):
    by_id = {r["id"]: r for r in records}
    rec = by_id[call]

    def expand(c):
        out = []
        for e in by_id[c]["saw"]:
            out += expand(e["saw_of"]) if "saw_of" in e else [e]
        return out

    earlier = []
    for e in expand(call):
        shown = by_id.get(e["call"])
        if shown is None:
            return {"refuses": "missing-call"}
        if shown.get("content") is not True:
            return {"refuses": "not-kept"}
        t = turn_of(shown, e)
        if t is None:
            return {"refuses": "not-kept"}
        earlier.append(t)
    return {"earlier": earlier, "conversation": (rec.get("conversation") or {}).get("id")}


def context_cases() -> dict:
    step = [{"kind": "model", "outputs": {"result": "Welcome, Alex!"}}]
    a = call_rec(1, inputs={"message": "Hi, I'm Alex."}, outputs={"result": "Welcome, Alex!"}, steps=step,
                 conversation={"id": "alex", "turn": uid(201), "parent": None})
    b = call_rec(2, inputs={"message": "What is my name?"}, outputs={"result": "Alex."},
                 saw=[{"call": uid(201), "steps": True}],
                 conversation={"id": "alex", "turn": uid(202), "parent": uid(201)})
    c = call_rec(3, inputs={"message": "And again?"}, outputs={"result": "Alex!"},
                 saw=[{"saw_of": uid(202)}, {"call": uid(202)}],
                 conversation={"id": "alex", "turn": uid(203), "parent": uid(202)})
    no_steps = {**a}
    no_steps.pop("steps")
    hidden = call_rec(1, inputs={"message": "Hi"}, outputs={"result": "Welcome"}, content=False)
    out = {}
    for name, why, records, call in [
            ("context-01-a-turn-and-what-it-saw", "A rated turn's earlier turn, with its steps.", [a, b], uid(202)),
            ("context-02-saw-of", "saw_of expanded: every earlier turn, in order.", [a, b, c], uid(203)),
            ("context-03-steps-not-kept", "Shown with its steps, and its record keeps none: not-kept.",
             [no_steps, b], uid(202)),
            ("context-04-values-not-kept", "Shown, and its record keeps no values: not-kept.",
             [hidden, b], uid(202)),
            ("context-05-a-call-the-log-lacks", "The call it saw is not in the log: missing-call.", [b], uid(202))]:
        out[name] = {"description": why, "kind": "context", "records": records, "call": call,
                     "expect": context_rows(records, call)}
    return out


def cases() -> dict:
    return {"replies": reply_cases(), "conversations": conversation_cases(), "tools": tool_cases(),
            "views": view_cases(), "context": context_cases()}
