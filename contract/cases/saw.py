"""The cases in saw/: the calls a call was given as context, read from the
call log, written from ../calls.md, section "Saw".

A case file: {"description", "records": [call records], "queries":
[{"call": id, "expect": {"saw": [entries]} or {"unknown": code}}]}.
``saw`` is the call's entries with every ``saw_of`` replaced, recursively,
by the entries of the call it names; ``unknown`` is why they cannot be
known (not-recorded, missing-call, unknown-key, saw-cycle). An
implementation passes a case when reading each queried call's ``saw``
from the records gives ``expect``.
"""

import copy

ENTRY_KEYS = {"call", "steps", "without"}


# ------------------------------------------------------------------ the rules


class Unknown(Exception):
    pass


def expand(records: dict, call: str, following=()) -> list:
    rec = records.get(call)
    if rec is None or "saw" not in rec:
        raise Unknown("not-recorded" if not following else "missing-call")
    out = []
    for i, entry in enumerate(rec["saw"]):
        if "saw_of" in entry:
            if i != 0 or set(entry) != {"saw_of"}:
                raise Unknown("unknown-key")
            target = entry["saw_of"]
            if target in following or target == call:
                raise Unknown("saw-cycle")
            out += expand(records, target, (*following, call))
        else:
            if set(entry) - ENTRY_KEYS:
                raise Unknown("unknown-key")
            out.append(copy.deepcopy(entry))
    return out


def answer(records: list, call: str) -> dict:
    by_id = {r["id"]: r for r in records}
    try:
        return {"saw": expand(by_id, call)}
    except Unknown as why:
        return {"unknown": str(why)}


# ------------------------------------------------------------------ records


def cid(n):
    return f"01926b00-{n:04x}-7000-8000-000000000000"


def rec(n, saw, *, name="tutor", kind="ai", parent=None, root=None, content=True, inputs=None, outputs=None):
    inputs = inputs if inputs is not None else {"message": f"message {n}"}
    outputs = outputs if outputs is not None else {"result": f"reply {n}"}
    r = {"functai_call": 1, "id": cid(n), "parent": cid(parent) if parent else None,
         "root": cid(root if root else n),
         "program": {"name": name, "kind": kind, "module": "school", "version": "sha256:" + "1" * 64,
                     "signature": "sha256:" + "5" * 64, "answer": "result"},
         "started": f"2026-09-28T10:{n:02d}:00.000000Z", "seconds": 1.0, "content": content}
    if content:
        r["inputs"], r["outputs"] = inputs, outputs
    r["sizes"] = {"inputs": {k: len(str(v)) + 2 for k, v in inputs.items()},
                  "outputs": {k: len(str(v)) + 2 for k, v in outputs.items()}}
    r.update(error=None, model="gpt-4.1-mini" if kind == "ai" else None, usage={}, confidence=None,
             exchanges=[], caller={"kind": "conversation"},
             process={"host": "lambda", "pid": 1, "user": "maxime", "language": "python", "runtime": "3.13.1",
                      "functai": "1.2.0"})
    if saw is not None:
        r["saw"] = saw
    return r


def call(n, **kw):
    return {"call": cid(n), **kw}


def of(n):
    return {"saw_of": cid(n)}


def case(description, records, queries):
    return {"description": description, "records": records,
            "queries": [{"call": cid(q), "expect": answer(records, cid(q))} for q in queries]}


def cases() -> dict:
    conversation = [rec(1, []), rec(2, [of(1), call(1)]), rec(3, [of(2), call(2)]), rec(4, [of(3), call(3)])]
    return {
        "01-nothing": case("saw [] means the call was given no earlier call.", [rec(1, [])], [1]),
        "02-not-recorded": case(
            "A record without saw (written before 2026-09-28, or by a writer that does not record it) says "
            "nothing: what it saw is not known, never taken to be nothing.", [rec(1, None)], [1]),
        "03-every-earlier-turn": case(
            "A conversation where each turn sees every earlier one writes what the previous turn saw, then "
            "that turn: each record stays two entries long, and reading one gives every earlier turn in "
            "order.", conversation, [1, 2, 3, 4]),
        "04-a-window": case(
            "A turn that sees only the last two turns lists them.",
            conversation + [rec(5, [call(3), call(4)])], [5]),
        "05-shown-differently": case(
            "How each call was shown travels with it: a photo left out after its first answer, a helper's "
            "steps shown; saw_of copies entries exactly as the named call has them.",
            [rec(1, [], inputs={"message": "What is wrong with my plant?", "photo": {"$media": "image/jpeg"}}),
             rec(2, [call(1)]),
             rec(3, [call(1, without=["photo"]), call(2, steps=True)]),
             rec(4, [of(3), call(3)])], [2, 3, 4]),
        "06-a-missing-call": case(
            "saw_of naming a call the log does not have, or one whose record has no saw, cannot be read: "
            "what the call saw is not known (a row asked again from it would be a guess).",
            [rec(2, [of(1), call(1)]), rec(3, None), rec(4, [of(3), call(3)])], [2, 4]),
        "07-a-way-of-showing-this-reader-does-not-know": case(
            "An entry with a key this reader does not know was shown in a way it cannot reproduce (a later "
            "writer's): not known.",
            [rec(1, []), rec(2, [call(1, summarized=True)])], [2]),
        "08-a-cycle": case(
            "saw_of that comes back to a call already followed is a broken log: not known.",
            [rec(1, [of(2)]), rec(2, [of(1), call(1)])], [1, 2]),
        "09-helpers-inside-turns": case(
            "A helper that remembers its own calls in the conversation: its call in the second turn saw its "
            "call in the first; the module's turns saw each other. Ids are kept when values are not "
            "(content false).",
            [rec(1, [], name="support", kind="module"),
             rec(2, [], name="answer", parent=1, root=1),
             rec(3, [], name="topic", parent=1, root=1),
             rec(4, [of(1), call(1)], name="support", kind="module", content=False),
             rec(5, [of(2), call(2)], name="answer", parent=4, root=4, content=False),
             rec(6, [], name="topic", parent=4, root=4)], [4, 5, 6]),
    }
