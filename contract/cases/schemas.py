"""The contract's JSON Schemas, for make.py to check every record, event,
manifest and interface it writes into a case (schema/)."""

import json
import re
from pathlib import Path

import jsonschema
from referencing import Registry, Resource

SCHEMA = Path(__file__).resolve().parent.parent / "schema"
TEXT = {p.stem.removesuffix(".schema"): json.loads(p.read_text()) for p in sorted(SCHEMA.glob("*.schema.json"))}


def as_ecma(node):
    """The schemas as ECMA-262 reads their patterns, which JSON Schema follows: a pattern's ``$`` is the end of
    the text. Python's ``re`` also lets ``$`` match before a final newline, so a checker in Python would accept
    ``"message\\n"`` as a name where every other language refuses it; ``\\Z`` is ECMA's ``$`` in Python. The
    schemas use ``$`` only as the last anchor (checked here)."""
    if isinstance(node, dict):
        out = {k: as_ecma(v) for k, v in node.items()}
        if isinstance(node.get("pattern"), str):
            p = node["pattern"]
            assert re.fullmatch(r"[^$]*\$\)*", p), f"a pattern with $ other than as its last anchor: {p}"
            out["pattern"] = re.sub(r"\$(?=\)*$)", r"\\Z", p)
        return out
    if isinstance(node, list):
        return [as_ecma(v) for v in node]
    return node


DOCS = {k: as_ecma(v) for k, v in TEXT.items()}
REGISTRY = Registry().with_resources([(d["$id"], Resource.from_contents(d)) for d in DOCS.values()])


def validator(name: str, pointer: str = "") -> jsonschema.Draft202012Validator:
    schema = DOCS[name]
    if pointer:
        schema = {"$ref": schema["$id"] + "#" + pointer}
    return jsonschema.Draft202012Validator(schema, registry=REGISTRY)


CALL, RATING, EVENT, SAVED = validator("call"), validator("rating"), validator("event"), validator("saved")
INTERFACE = validator("interface")
CONVERSATION = validator("conversation")
SAW_ENTRY = validator("call", "/$defs/saw_entry")


def check(v: jsonschema.Draft202012Validator, value, where: str) -> None:
    errors = sorted(v.iter_errors(value), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        raise AssertionError(f"{where}: {e.message} at {list(e.path)}")


def record(rec: dict, where: str) -> None:
    if "functai_call" in rec:
        check(CALL, rec, where)
    elif "functai_rating" in rec:
        check(RATING, rec, where)
    else:
        raise AssertionError(f"{where}: neither a call nor a rating")


def refusals() -> None:
    """What the schemas must refuse: each value leaks something, or says what a format does not."""
    import copy

    import content
    import events

    rec = content.record()
    part = content.cases()["03-all-but-one-input"]["expect"]["record"]
    bad = []

    r = copy.deepcopy(part)
    r["exchanges"][0]["error"]["message"] = "the transcript, quoted"
    bad.append(("a failed attempt's message in a record that keeps some values", CALL, r))
    r = copy.deepcopy(part)
    del r["omitted"]
    bad.append(("content false without omitted (format 2)", CALL, r))
    r = copy.deepcopy(rec)
    r["omitted"] = {"inputs": [], "outputs": []}
    bad.append(("omitted beside content true", CALL, r))
    r = copy.deepcopy(part)
    r["exchanges"][1]["request_hash"] = rec["exchanges"][1]["request_hash"]
    bad.append(("a request hash in a record that keeps some values", CALL, r))
    r = copy.deepcopy(part)
    r["functai_call"] = 1
    del r["omitted"], r["saw"]
    r["program"].pop("interface")
    for ex in r["exchanges"]:
        ex.pop("request_hash", None)
    bad.append(("a format-1 record keeping some values (format 1 means none when content is false)", CALL, r))
    r = copy.deepcopy(rec)
    del r["saw"]
    bad.append(("a format-2 record without saw", CALL, r))
    r = copy.deepcopy(rec)
    del r["program"]["signature"]
    bad.append(("an AI function's record without program.signature", CALL, r))

    w = events.module_with_two_outputs()
    e = copy.deepcopy(w.events[-1])
    e["content"] = False
    bad.append(("done with a value, marked as not whole", EVENT, e))
    w = events.a_transcript()
    e = copy.deepcopy(next(x for x in w.events if x["kind"] == "text"))
    del e["text"]
    e.update(size=13, content=False)
    bad.append(("a piece kept as its size", EVENT, e))
    e = copy.deepcopy(next(x for x in w.events if x["kind"] == "thinking"))
    e["content"] = False
    bad.append(("thinking marked as not whole (thinking is kept whole or not at all)", EVENT, e))
    e = copy.deepcopy(next(x for x in events.a_tool().events if x["kind"] == "tool_call"))
    e["id"] = None
    bad.append(("a tool call without an id", EVENT, e))
    e = copy.deepcopy(w.events[0])
    e["content"] = False
    bad.append(("started not whole without omitted", EVENT, e))

    e = copy.deepcopy(events.a_tool().events[0])
    del e["writer"]
    bad.append(("an event with no writer number", EVENT, e))
    e = copy.deepcopy(events.a_tool().events[1])
    e["after"] = 1
    bad.append(("an after that is a seq alone, not a position", EVENT, e))
    e = copy.deepcopy(events.a_tool().events[1])
    e["after"] = {"seq": 1}
    bad.append(("an after with no writer", EVENT, e))
    e = copy.deepcopy(events.a_tool().events[0])
    e["after"] = 0
    bad.append(("a first event whose after is 0, not null", EVENT, e))
    r = copy.deepcopy(rec)
    r["journal"] = "failed"
    bad.append(("a journal outcome that is neither refused nor unknown", CALL, r))

    iface = {"description": "", "inputs": [{"name": "frame", "shape": {"type": "array"}, "opaque": True}],
             "outputs": [{"name": "result", "shape": {}}]}
    bad.append(("an opaque field with a shape", INTERFACE, iface))
    iface = {"description": "", "inputs": [], "outputs": [{"name": "result", "shape": {}, "optional": True}]}
    bad.append(("an optional output", INTERFACE, iface))
    iface = {"description": "", "inputs": [{"name": "q", "shape": {}, "label": "private"}],
             "outputs": [{"name": "result", "shape": {}}]}
    bad.append(("an interface key this contract does not name (refused, not ignored)", INTERFACE, iface))
    iface = {"description": "", "inputs": [{"name": "q\n", "shape": {}}], "outputs": [{"name": "result", "shape": {}}]}
    bad.append(("a field name with a newline after it (a pattern is matched whole, as ECMA-262 reads $)", INTERFACE,
                iface))
    e = copy.deepcopy(events.a_tool().events[1])
    e["kind"] = "request\n"
    bad.append(("an event kind with a newline after it", EVENT, e))
    r = copy.deepcopy(rec)
    r["id"] += "\n"
    bad.append(("a call id with a newline after it", CALL, r))

    base = {"functai_rating": 1, "id": "01926a90-0001-7000-8000-000000000000",
            "call": "01926a8e-0001-7000-8000-000000000000", "at": "2026-09-26T10:05:00.000000Z", "verdict": "right"}
    bad.append(("a rating that names a person and an account", RATING, {**base, "by": "ana", "account": "maxime"}))
    bad.append(("a rating that names neither a person nor an account", RATING, base))
    # format 2 (measures.md)
    two = {"functai_rating": 2, "id": "01926a90-0002-7000-8000-000000000000", "at": "2026-11-05T10:05:00.000000Z",
           "call": "01926a8e-0001-7000-8000-000000000000", "criterion": "real_problem", "kind": "label",
           "answers": {"yes": 1, "no": 0, "unsure": None}, "value": "yes", "by": "megan"}
    pair = {"calls": ["01926a8e-0001-7000-8000-000000000000", "01926a8e-0002-7000-8000-000000000000"]}
    choice = {k: v for k, v in two.items() if k not in ("call", "answers")} | pair | {"kind": "choice", "criterion": "preferred",
                                                                              "value": pair["calls"][1]}
    judge = {"name": "professor", "setup": "sha256:" + "d4" * 32}
    bad += [("a rating of one call and of two", RATING, {**two, **pair}),
            ("a pick with an item", RATING, {**choice, "item": "/insights/0"}),
            ("a pick of one call", RATING, {**{k: v for k, v in choice.items() if k != "calls"}, "call": two["call"]}),
            ("a pick of a third answer", RATING, {**choice, "value": "first"}),
            ("a label without its answers", RATING, {k: v for k, v in two.items() if k != "answers"}),
            ("an answer that counts for more than 1", RATING, {**two, "answers": {"yes": 2, "no": 0}}),
            ("a scale on a label", RATING, {**two, "scale": [0, 10]}),
            ("a person and a judge", RATING, {**two, "judge": judge}),
            ("a judge known by neither setup nor code", RATING,
             {**{k: v for k, v in two.items() if k != "by"}, "judge": {"name": "professor"}}),
            ("a correction of a right verdict", RATING,
             {**{k: v for k, v in two.items() if k != "answers"}, "criterion": "correct", "kind": "verdict",
              "value": "right", "answer": "x"}),
            ("an item that is not a JSON Pointer", RATING, {**two, "item": "insights/3"}),
            ("a call with a case that is not a hash", CALL, {**rec, "case": "site 3"}),
            ("a call asked a zeroth time", CALL, {**rec, "case": "sha256:" + "0" * 64, "repeat": 0})]

    for why, v, value in bad:
        assert not v.is_valid(value), f"the schema accepts {why}"

    later = copy.deepcopy(events.a_tool().events[1])
    later.update(kind="handoff", asked="May a person take over?")
    del later["request"], later["model"]
    good = [("a rating made under an account", RATING, {**base, "account": "maxime"}),
            ("a rating by a person", RATING, {**base, "by": "ana"}),
            ("a blind rating of format 1", RATING, {**base, "by": "ana", "blind": True}),
            ("a person's label of an item", RATING, {**two, "item": "/insights/3", "blind": True, "sample": "draw-1"}),
            ("a judge's pick", RATING, {**{k: v for k, v in choice.items() if k != "by"}, "judge": judge}),
            ("a tie", RATING, {**choice, "value": "tie"}),
            ("a withdrawn pick", RATING, {**choice, "value": None}),
            ("code's score on a scale", RATING,
             {**{k: v for k, v in two.items() if k not in ("by", "answers")}, "kind": "score", "scale": [1, 5], "value": 4,
              "judge": {"name": "quotes_found", "code": "sha256:" + "f6" * 32}}),
            ("a call that says its case", CALL, {**rec, "case": "sha256:" + "0" * 64, "repeat": 2}),
            ("a record naming the lmcc and lm15 that made it", CALL,
             {**rec, "process": {**rec["process"], "lmcc": "0.8.5", "lm15": "1.0.1"}}),
            ("a saw entry of a kind a later writer added", SAW_ENTRY, {"summary": "Alex struggles."}),
            ("an event of a kind a later stage adds", EVENT, later),
            ("a saw entry with a key a later writer added", SAW_ENTRY,
             {"call": rec["id"], "children": "first-layer"})]
    for why, v, value in good:
        assert v.is_valid(value), f"the schema refuses {why}"
