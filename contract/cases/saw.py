"""The cases in saw/: the calls a call was given as context, read from the
call log, written from ../calls.md, section "Saw". Two kinds, told apart
by "kind":

- "read": {"records": [call records], "queries": [{"call": id, "expect":
  {"saw": [entries]} or {"unknown": code, "call": id}, "keeps": {"ok":
  true} or {"refuses": code, "call": id}}]}. ``saw`` is the call's entries
  with every ``saw_of`` replaced, recursively, by the entries of the call
  it names; ``unknown`` is why they cannot be known (not-recorded,
  missing-call, unknown-key, saw-cycle), and whose record says so.
  ``keeps`` is whether the log keeps what showing the call its context
  again needs: every call it saw is in the log, not truncated, with the
  values it was shown as data (missing-call, not-kept), and, with
  steps, each exchange's request hash and reply; an entry no call can
  have been shown refuses turn-invalid. It is not the showing itself:
  turning a record's exchanges into steps is stage 5's.
- "shown": {"turn": an lmcc turn (kernel §3a), the one a call's record
  stands for, "entry": a saw entry naming that call, "expect": {"slot",
  "turn"} or {"refuses": "turn-invalid"}}. The turn the entry says was
  shown, and the slot it was placed in.

An implementation passes a case when reading each query from the records,
or showing the turn, gives ``expect``.
"""

import copy

ENTRY_KEYS = {"call", "steps", "without", "slot"}


# ------------------------------------------------------------------ the rules: reading


class Unknown(Exception):
    def __init__(self, code, call):
        super().__init__(code)
        self.code, self.call = code, call


def expand(records: dict, call: str, following=()) -> list:
    rec = records.get(call)
    if rec is None or "saw" not in rec:
        raise Unknown("not-recorded" if not following else "missing-call", call)
    out = []
    for i, entry in enumerate(rec["saw"]):
        if "saw_of" in entry:
            if i != 0 or set(entry) != {"saw_of"}:
                raise Unknown("unknown-key", call)
            target = entry["saw_of"]
            if target in following or target == call:
                raise Unknown("saw-cycle", target)
            out += expand(records, target, (*following, call))
        else:
            if "call" not in entry or set(entry) - ENTRY_KEYS:
                raise Unknown("unknown-key", call)
            out.append(copy.deepcopy(entry))
    return out


CALLS = "calls"


def has_values(rec: dict, entry: dict) -> bool:
    """Whether a call's record keeps what the entry says it was shown: the values of the fields it was shown
    with, as data (not descriptions of values with no JSON form); with steps, every exchange's request hash
    and, when a reply came, the reply."""
    if rec.get("truncated"):
        return False
    left_out = set(entry.get("without", []))
    shown = set(rec["sizes"]["inputs"]) | set(rec["sizes"]["outputs"])
    shown -= left_out
    described = rec.get("described", {"inputs": [], "outputs": []})
    if shown & (set(described["inputs"]) | set(described["outputs"])):
        return False
    if entry.get("steps"):
        return rec["content"] is True and all(
            "request_hash" in ex and ("response" in ex or ex.get("finish") is None) for ex in rec["exchanges"])
    if rec["content"] is True:
        return True
    if "omitted" not in rec:                    # format 1, or no value kept
        return False
    return not ((set(rec["omitted"]["inputs"]) | set(rec["omitted"]["outputs"])) - left_out)


def read(records: list, call: str) -> dict:
    by_id = {r["id"]: r for r in records}
    try:
        entries = expand(by_id, call)
    except Unknown as why:
        return {"expect": {"unknown": why.code, "call": why.call},
                "keeps": {"refuses": why.code, "call": why.call}}
    for entry in entries:
        if entry.get("steps") and CALLS in entry.get("without", []):
            return {"expect": {"saw": entries}, "keeps": {"refuses": "turn-invalid", "call": entry["call"]}}
        rec = by_id.get(entry["call"])
        if rec is None:
            return {"expect": {"saw": entries}, "keeps": {"refuses": "missing-call", "call": entry["call"]}}
        if not has_values(rec, entry):
            return {"expect": {"saw": entries}, "keeps": {"refuses": "not-kept", "call": entry["call"]}}
    return {"expect": {"saw": entries}, "keeps": {"ok": True}}


# ------------------------------------------------------------------ the rules: the turn shown


def shown(turn: dict, entry: dict) -> dict:
    left_out = set(entry.get("without", []))
    calls_fields = {s["calls_field"] for s in turn["steps"] if s["kind"] == "model" and s.get("calls_field")}
    if entry.get("steps") and left_out & calls_fields:
        return {"refuses": "turn-invalid"}      # its tool steps would answer no call (lmcc kernel §3a)

    def strip(values):
        return None if values is None else {k: v for k, v in values.items() if k not in left_out}
    out = {"signature": turn["signature"], "inputs": strip(turn["inputs"]), "steps": []}
    if entry.get("steps"):
        for step in turn["steps"]:
            step = copy.deepcopy(step)
            if step["kind"] == "model":
                if left_out & set(step["outputs"]):
                    step.pop("message", None)
                step["outputs"] = strip(step["outputs"])
            out["steps"].append(step)
    if "outputs" in turn:
        out["outputs"] = strip(turn["outputs"])
    return {"slot": entry.get("slot", "turns"), "turn": out}


# ------------------------------------------------------------------ records


def cid(n):
    return f"01926b00-{n:04x}-7000-8000-000000000000"


def rec(n, saw, *, name="tutor", kind="ai", parent=None, root=None, content=True, inputs=None, outputs=None,
        omitted=None, fmt=2, described=None, exchanges=(), truncated=False):
    inputs = inputs if inputs is not None else {"message": f"message {n}"}
    outputs = outputs if outputs is not None else {"result": f"reply {n}"}
    program = {"name": name, "kind": kind, "module": "school", "version": "sha256:" + "1" * 64,
               "signature": "sha256:" + "5" * 64, "answer": "result"}
    if fmt == 2:
        program["interface"] = "sha256:" + "5" * 64
        if kind == "module":
            del program["signature"]
    r = {"functai_call": fmt, "id": cid(n), "parent": cid(parent) if parent else None,
         "root": cid(root if root else n), "program": program,
         "started": f"2026-09-28T10:{n:02d}:00.000000Z", "seconds": 1.0,
         "content": content and omitted is None}
    if omitted is not None:
        r["omitted"] = omitted
    elif not content and fmt == 2:
        r["omitted"] = {"inputs": list(inputs), "outputs": list(outputs)}
    if content:
        dropped = set(omitted["inputs"] + omitted["outputs"]) if omitted else set()
        kept_in = {k: v for k, v in inputs.items() if k not in dropped}
        kept_out = {k: v for k, v in outputs.items() if k not in dropped}
        if kept_in:
            r["inputs"] = kept_in
        if kept_out:
            r["outputs"] = kept_out
    r["sizes"] = {"inputs": {k: len(str(v)) + 2 for k, v in inputs.items()},
                  "outputs": {k: len(str(v)) + 2 for k, v in outputs.items()}}
    if described:
        r["described"] = described
    r.update(error=None, model="gpt-4.1-mini" if kind == "ai" else None, usage={}, confidence=None,
             exchanges=[exchange(x, content=r["content"]) for x in exchanges], caller={"kind": "conversation"},
             process={"host": "lambda", "pid": 1, "user": "maxime", "language": "python", "runtime": "3.13.1",
                      "functai": "1.2.0"})
    if saw is not None:
        r["saw"] = saw
    if truncated:
        r["truncated"] = True
    return r


def exchange(how: str, *, content=True) -> dict:
    """An exchange: "replied" (a reply came), "no reply" (a failed attempt with no reply), "reply not kept" (a
    reply came; the record lost it)."""
    ex = {"model": "gpt-4.1-mini", "provider": "openai", "started": "2026-09-28T10:00:00.100000Z", "seconds": 0.5,
          "cached": False, "finish": None if how == "no reply" else "stop"}
    if how == "no reply":
        ex["error"] = {"type": "ProviderError", **({"message": "503"} if content else {})}
    if content:
        ex["request"] = {"messages": []}
        ex["request_hash"] = "sha256:" + "a" * 64
        if how == "replied":
            ex["response"] = {"message": {"role": "assistant", "parts": []}}
    return ex


def call(n, **kw):
    return {"call": cid(n), **kw}


def of(n):
    return {"saw_of": cid(n)}


def case(description, records, queries):
    return {"description": description, "kind": "read", "records": records,
            "queries": [{"call": cid(q), **read(records, cid(q))} for q in queries]}


def shown_case(description, turn, entry):
    return {"description": description, "kind": "shown", "turn": turn, "entry": entry, "expect": shown(turn, entry)}


PHOTO = {"$media": "image/jpeg", "sha256": "9c1e" + "0" * 60}
TURN = {"signature": "sha256:" + "5" * 64,
        "inputs": {"message": "Where is B-2210?", "photo": PHOTO},
        "steps": [{"kind": "model", "outputs": {"reasoning": "Look the order up first.",
                                                "tool_calls": [{"id": "call_1", "name": "order_status",
                                                                "input": {"order": "B-2210"}}]},
                   "message": {"role": "assistant", "parts": [
                       {"type": "text", "text": "<reasoning>Look the order up first.</reasoning>"},
                       {"type": "tool_call", "id": "call_1", "name": "order_status",
                        "input": {"order": "B-2210"}}]},
                   "request": "sha256:" + "a" * 64, "calls_field": "tool_calls"},
                  {"kind": "tool", "id": "call_1", "name": "order_status",
                   "output": [{"type": "text", "text": "In the Leeds depot, 7 days late."}]},
                  {"kind": "model", "outputs": {"result": "It is in Leeds, a week late."},
                   "message": {"role": "assistant", "parts": [{"type": "text", "text": "It is in Leeds, a week late."}]},
                   "request": "sha256:" + "b" * 64}],
        "outputs": {"result": "It is in Leeds, a week late."}}


def cases() -> dict:
    conversation = [rec(1, []), rec(2, [of(1), call(1)]), rec(3, [of(2), call(2)]), rec(4, [of(3), call(3)])]
    photo = {"message": "What is wrong with my plant?", "photo": PHOTO}
    return {
        "01-nothing": case("saw [] means the call was given no earlier call.", [rec(1, [])], [1]),
        "02-not-recorded": case(
            "A record without saw (format 1, or a writer that does not record it) says nothing: what it saw is "
            "not known, never taken to be nothing.", [rec(1, None, fmt=1)], [1]),
        "03-every-earlier-turn": case(
            "A conversation where each turn sees every earlier one writes what the previous turn saw, then "
            "that turn: each record stays two entries long, and reading one gives every earlier turn in "
            "order.", conversation, [1, 2, 3, 4]),
        "04-a-window": case(
            "A turn that sees only the last two turns lists them.",
            conversation + [rec(5, [call(3), call(4)])], [5]),
        "05-shown-differently": case(
            "How each call was shown travels with it: a photo left out after its first answer, a helper's "
            "steps shown, a slot other than turns; saw_of copies entries exactly as the named call has them.",
            [rec(1, [], inputs=photo),
             rec(2, [call(1)]),
             rec(3, [call(1, without=["photo"]), call(2, steps=True, slot="helper")]),
             rec(4, [of(3), call(3)])], [2, 3, 4]),
        "06-a-missing-call": case(
            "saw_of naming a call the log does not have, or one whose record has no saw, cannot be read: "
            "what the call saw is not known (a row asked again from it would be a guess).",
            [rec(2, [of(1), call(1)]), rec(3, None, fmt=1), rec(4, [of(3), call(3)])], [2, 4]),
        "07-a-way-of-showing-this-reader-does-not-know": case(
            "An entry with a key this reader does not know, or of a kind it does not know (no call), was shown "
            "in a way it cannot reproduce: not known. A saw_of with another key is one too.",
            [rec(1, []), rec(2, [call(1, summarized=True)]), rec(3, [{"summary": "Alex struggles with fractions."}]),
             rec(4, [{"saw_of": cid(1), "last": 3}])], [2, 3, 4]),
        "08-a-cycle": case(
            "saw_of that comes back to a call already followed is a broken log: not known.",
            [rec(1, [of(2)]), rec(2, [of(1), call(1)])], [1, 2]),
        "09-helpers-inside-turns": case(
            "A helper that remembers its own calls in the conversation: its call in the second turn saw its "
            "call in the first; the module's turns saw each other. Ids are kept when values are not "
            "(content false), so what was seen is known, but a turn whose values were not kept cannot be shown "
            "again (not-kept).",
            [rec(1, [], name="support", kind="module", content=False),
             rec(2, [], name="answer", parent=1, root=1),
             rec(3, [], name="topic", parent=1, root=1),
             rec(4, [of(1), call(1)], name="support", kind="module"),
             rec(5, [of(2), call(2)], name="answer", parent=4, root=4, content=False),
             rec(6, [], name="topic", parent=4, root=4)], [4, 5, 6]),
        "10-knowing-is-not-replaying": case(
            "A call named directly but absent from the log: what was seen is known (its id), but it cannot be "
            "shown again (missing-call). Reading saw lists ids; replaying needs their records.",
            [rec(2, [call(1)]), rec(3, [call(2), call(9)])], [2, 3]),
        "11-what-must-be-kept": case(
            "Showing a call again needs the values it was shown with: a photo left out when it was shown need "
            "not be kept; one shown must be; steps need the whole record (its replies).",
            [rec(1, [], inputs=photo, omitted={"inputs": ["photo"], "outputs": []}),
             rec(2, [call(1, without=["photo"])]),
             rec(3, [call(1)]),
             rec(4, [], inputs=photo),
             rec(5, [call(1, steps=True, without=["photo"])]),
             rec(6, [call(4, steps=True)])], [2, 3, 5, 6]),
        "12-shown-with-steps-without-a-field": shown_case(
            "A call shown with its steps and without its reasoning: the field goes from the inputs, the outputs "
            "and every model step; a step whose outputs held it loses its recorded message (it would show the "
            "field) and is written from its values; tool steps stay.",
            TURN, call(1, steps=True, without=["reasoning", "photo"])),
        "13-shown-without-steps": shown_case(
            "A call shown without steps is its inputs and outputs only, in the slot its entry names (turns when "
            "none).", TURN, call(1, without=["photo"])),
        "14-a-context-that-changed": case(
            "A call whose context changed between its requests (an agent that compacted its history) writes, "
            "last, an entry no reader knows today: what it saw is not known (unknown-key), rather than a list "
            "that is true of its first request only.",
            [rec(1, []), rec(2, [call(1), {"context": "changed"}])], [2]),
        "15-what-showing-again-needs": case(
            "Showing a call again needs its values as data: an input written as a description (a value with no "
            "JSON form) is not the value (not-kept), unless the entry left it out. With steps, every exchange "
            "keeps its request hash and, when a reply came, its reply (a failed attempt with no reply needs "
            "none); a truncated record keeps too little. With steps, an entry that leaves out the tool calls "
            "is one no call can have been shown: its tool steps would answer nothing (turn-invalid).",
            [rec(1, [], inputs={"message": "Hi", "frame": {"$type": "DataFrame", "$repr": "   a\n0  1"}},
                 described={"inputs": ["frame"], "outputs": []}),
             rec(2, [call(1)]),
             rec(3, [call(1, without=["frame"])]),
             rec(4, [], exchanges=["no reply", "replied"]),
             rec(5, [call(4, steps=True)]),
             rec(6, [], exchanges=["reply not kept"]),
             rec(7, [call(6, steps=True)]),
             rec(8, [], truncated=True),
             rec(9, [call(8)]),
             rec(10, [call(4, steps=True, without=["calls"])])], [2, 3, 5, 7, 9, 10]),
        "16-shown-without-the-tool-calls": shown_case(
            "With steps, leaving out the field that holds a step's tool calls would leave tool steps that "
            "answer no call: refused (turn-invalid, lmcc's word), never shown.",
            TURN, call(1, steps=True, without=["tool_calls"])),
    }
