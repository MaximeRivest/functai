"""The cases in functions/: a definition of an AI function, and the
signature, sample input, request, version and signature id it must give.

Everything FunctAI decides is written here from the rules in
../functions.md and ../calls.md: the instructions, the fields, the
layout (../layouts/), the probe facts (../models.json), the worked
examples, the sample input, the hashes. Turning a signature, a layout and
values into a request is lmcc's rule, pinned byte for byte by lmcc's own
corpus, so this script asks lmcc (the Python kernel with its standard
pack) for that step only. It never imports FunctAI.

A case file: {"description", "definition": {name, description, inputs
(each {name, shape, desc?, optional?, default_code?}), outputs, settings,
state, tools?}, "expect": {"signature", "sample",
"request", "request_hash", "version", "signature_id"}}. ``signature`` is
lmcc's plain-data form without host type names (compare with ``type``
left out).
"""

import copy
import json
from pathlib import Path

import lmcc
import lmcc_std

from common import no_defaults, sha

CONTRACT = Path(__file__).resolve().parent.parent
PROBE = {k: v for k, v in json.loads((CONTRACT / "models.json").read_text())["probe"].items() if k != "about"}
LAYOUTS = {name: json.loads((CONTRACT / "layouts" / f"{name}.json").read_text()) for name in ("xml", "chat", "json")}
WHITE = "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008" \
        "\u2009\u200a\u2028\u2029\u202f\u205f\u3000"

TOOL_LIST = {"type": "array", "items": {"type": "object", "properties": {
    "name": {"type": "string"}, "description": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "parameters": {"anyOf": [{"type": "object"}, {"type": "null"}]}},
    "required": ["name", "description", "parameters"]}}
CALL_LIST = {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "name": {"type": "string"}, "input": {"type": "object"}},
    "required": ["id", "name", "input"]}}


def registry() -> lmcc.Registry:
    r = lmcc.Registry()
    lmcc_std.install(r)
    return r


# ------------------------------------------------------------------ the rules


def instructions(d: dict) -> str:
    state = d.get("state") or {}
    if state.get("instructions") is not None:
        return state["instructions"].strip(WHITE)
    head = []
    if d["settings"].get("include_fn_name_in_instructions", True) and d["name"]:
        head.append(f"Function: {d['name']}")
    if d["description"].strip(WHITE):
        head.append(d["description"].strip(WHITE))
    head = "\n\n".join(head).strip(WHITE)
    lines = []
    described = [f for f in d["inputs"] if f.get("desc")]
    if described:
        lines += ["Parameter guidance:", *[f"- {f['name']}: {f['desc']}" for f in described], ""]
    described = [f for f in d["outputs"] if f.get("desc")]
    if described:
        lines += ["Output guidance:", *[f"- {f['name']}: {f['desc']}" for f in described], ""]
    guidance = "\n".join(lines).strip(WHITE)
    if not guidance:
        return head
    return head + ("\n\n" if head else "") + guidance


def without_default(shape: dict) -> dict:
    """An input's shape as the signature has it: its default is how a call is bound, not data."""
    return {k: v for k, v in shape.items() if k != "default"}


def fields(d: dict) -> list:
    """The signature's fields, with the host type names the two tool fields need."""
    out = [{"name": f["name"], "direction": "input", "shape": without_default(f["shape"]), "purpose": "plain",
            **({"desc": f["desc"]} if f.get("desc") else {})} for f in d["inputs"]]
    tools = bool(d.get("tools"))
    if tools:
        out.append({"name": "tools", "direction": "input", "shape": TOOL_LIST, "purpose": "tools",
                    "type": "list[Tool]"})
    names = {f["name"] for f in d["inputs"]} | {f["name"] for f in d["outputs"]}
    if d["settings"].get("module") == "cot" and "reasoning" not in names:
        out.append({"name": "reasoning", "direction": "output", "shape": {"type": "string"},
                    "purpose": "reasoning"})
    if tools:
        out.append({"name": "calls", "direction": "output", "shape": CALL_LIST, "purpose": "tools.calls",
                    "type": "list[ToolCall]"})
    out += [{"name": f["name"], "direction": "output", "shape": f["shape"], "purpose": "plain"}
            for f in d["outputs"]]
    return out


def sample(shape: dict):
    if "enum" in shape:
        return shape["enum"][0]
    if "anyOf" in shape:
        options = [s for s in shape["anyOf"] if s.get("type") != "null"]
        return sample(options[0]) if options else None
    kind = shape.get("type")
    if isinstance(kind, list):                        # a list of types: the first non-null one (calls.md)
        kind = next((k for k in kind if k != "null"), "null")
    return {"string": "example text", "integer": 3, "number": 2.5, "boolean": True, "array": [],
            "object": {}, "null": None}.get(kind, "example text")


def defaults(d: dict) -> dict:
    """calls.md, Versions, Defaults: each input's default by its logic. ``default_code`` on an input says the
    default is written as that expression (its value today is the shape's default); otherwise it counts by
    its value."""
    out = {}
    for f in d["inputs"]:
        if "default_code" in f:
            out[f["name"]] = {"code": f["default_code"]}
        elif "default" in f["shape"]:
            out[f["name"]] = {"value": f["shape"]["default"]}
    return out


def layout(d: dict, reg: lmcc.Registry) -> lmcc.Adapter:
    name = (d["settings"].get("adapter") or "xml").lower().replace("-", "").replace(" ", "")
    name = {"default": "xml", "tags": "xml"}.get(name, name)
    return lmcc.load(copy.deepcopy(LAYOUTS[name]), registry=reg)


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def examples(d: dict, plan: lmcc.Plan) -> list:
    sig = plan.signature
    plain_in = {f.name for f in sig.inputs if f.purpose == "plain"}
    text_in = {f.name for f in sig.inputs if f.shape.get("type") == "string" and "enum" not in f.shape}
    kept_out = {f.name for f in sig.outputs if f.purpose in ("plain", "reasoning")}
    turns = []
    for demo in (d.get("state") or {}).get("demos", []):
        ins = {k: as_text(v) if k in text_in and v is not None else v
               for k, v in demo["inputs"].items() if k in plain_in}
        outs = {k: v for k, v in demo["outputs"].items() if k in kept_out}
        if not outs:
            continue
        try:
            turns.append(plan.example(ins, outs))
        except lmcc.Refusal:
            continue
    return turns


def expect(d: dict) -> dict:
    reg = registry()
    sig_dict = {"instructions": instructions(d), "fields": fields(d)}
    signature = lmcc.signature_from_dict(copy.deepcopy(sig_dict))
    plan = layout(d, reg).bind(signature, dict(PROBE), registry=reg)
    values = d["_sample"] if "_sample" in d else {f["name"]: sample(f["shape"]) for f in d["inputs"]}
    text_in = {f["name"] for f in d["inputs"] if f["shape"].get("type") == "string" and "enum" not in f["shape"]}
    turn_values = {k: as_text(v) if k in text_in and v is not None else v for k, v in values.items()}
    if d.get("tools"):
        from lmcc_std.tools import Tool
        turn_values["tools"] = [Tool(**t) for t in d["tools"]]
    request = plan.render(plan.turn(turn_values), turns=examples(d, plan)).request("probe")
    request = json.loads(json.dumps(request))
    request_hash = sha(request)
    identity = [{"direction": f["direction"], "name": f["name"], "purpose": f.get("purpose", "plain"),
                 "shape": no_defaults(f["shape"]), "type": ""} for f in sig_dict["fields"]]
    version = {"request": request_hash}
    if defaults(d):
        version["defaults"] = defaults(d)
    return {"signature": {"instructions": sig_dict["instructions"],
                          "fields": [{k: v for k, v in f.items() if k != "type"} for f in sig_dict["fields"]]},
            "sample": values, "request": request, "request_hash": request_hash,
            "version": sha(version), "signature_id": sha(identity)}


# ------------------------------------------------------------------ the definitions

S, I, N, B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
MOOD = {"enum": ["happy", "unhappy", "mixed"], "type": "string"}


def definition(name, description, inputs, outputs, *, settings=None, state=None, tools=None):
    d = {"name": name, "description": description,
         "inputs": [dict(name=n, shape=s, **({"desc": x} if x else {}), **(more[0] if more else {}))
                    for n, s, x, *more in inputs],
         "outputs": [dict(name=n, shape=s, **({"desc": x} if x else {})) for n, s, x in outputs],
         "settings": settings or {}, "state": state or {"instructions": None, "demos": []}}
    if tools:
        d["tools"] = tools
    return d


DEFINITIONS = {
    "01-an-answer-from-a-list": (
        "One text input, an answer from a fixed list, the default layout (tags).",
        definition("mood", "How does the customer feel about what they bought?",
                   [("review", S, None)], [("result", MOOD, None)])),
    "02-several-outputs": (
        "Two outputs: the last is the answer. An output's desc goes into the instruction's "
        "Output guidance, not into its field.",
        definition("triage", "Read the ticket.",
                   [("ticket", S, None)],
                   [("summary", S, "one sentence, no names"), ("result", I, "minutes to fix")])),
    "03-described-inputs": (
        "An input's desc is its field's desc and a line of Parameter guidance.",
        definition("capital", "The country's capital city.",
                   [("country", S, "an English country name"), ("year", I, None)], [("result", S, None)])),
    "04-worked-examples": (
        "Demos become example turns in order; outputs the signature lacks are dropped; a demo left "
        "with no output is skipped; an object given to a text input is written as JSON indented by two spaces.",
        definition("mood", "How does the customer feel about what they bought?",
                   [("review", S, None)], [("result", MOOD, None)],
                   state={"instructions": None, "demos": [
                       {"inputs": {"review": "Broke in a day."}, "outputs": {"result": "unhappy"}},
                       {"inputs": {"review": "Great, but late."}, "outputs": {"result": "mixed", "stars": 3}},
                       {"inputs": {"review": "Nothing to say."}, "outputs": {"stars": 3}},
                       {"inputs": {"review": {"text": "Love it"}}, "outputs": {"result": "happy"}}]})),
    "05-dspy-sections": (
        "The chat layout: DSPy's [[ ## name ## ]] sections.",
        definition("capital", "The country's capital city.", [("country", S, None)], [("result", S, None)],
                   settings={"adapter": "Chat"})),
    "06-one-json-object": (
        "The json layout: one object the provider enforces (the probe facts have native structured output).",
        definition("person", "Who is described?", [("text", S, None)],
                   [("result", {"type": "object", "properties": {"name": S, "age": I},
                                "required": ["name", "age"]}, None)],
                   settings={"adapter": "json"})),
    "07-reasoning-first": (
        "module cot adds a reasoning output first; without native reasoning it is written in the reply.",
        definition("solve", "Solve the word problem.", [("problem", S, None)], [("result", N, None)],
                   settings={"module": "cot"})),
    "08-an-improved-instruction": (
        "state.instructions replaces the whole instruction (name, description, guidance), trimmed.",
        definition("capital", "The country's capital city.", [("country", S, "a country")], [("result", S, None)],
                   state={"instructions": "\n  Give the capital city of the country, in English.\n\n",
                          "demos": []})),
    "09-no-name": (
        "Without the function's name and without a description, the instruction is the guidance alone.",
        definition("extract", "", [("text", S, None)], [("result", S, "the date, as YYYY-MM-DD")],
                   settings={"include_fn_name_in_instructions": False})),
    "10-every-sample": (
        "The sample input, one value per shape: the first choice, the first non-null option, "
        "text, 3, 2.5, true, [], {}.",
        definition("describe", "Describe.",
                   [("a", S, None), ("b", I, None), ("c", N, None), ("d", B, None),
                    ("e", {"type": "array", "items": S}, None),
                    ("f", {"anyOf": [I, {"type": "null"}]}, None),
                    ("g", {"type": "object", "additionalProperties": I}, None),
                    ("h", {"type": "object", "properties": {"name": S}, "required": ["name"]}, None),
                    ("i", {"enum": ["red", "blue"], "type": "string"}, None)],
                   [("result", S, None)])),
    "11-a-tool": (
        "With tools: a tools input and a calls output; the probe facts call tools natively.",
        definition("helper", "Help with the order.", [("question", S, None)], [("result", S, None)],
                   tools=[{"name": "lookup_order", "description": "Look up an order.",
                           "parameters": {"type": "object", "properties": {"order": S}, "required": ["order"]}}])),
    "12-an-optional-input": (
        "An input a caller may leave out: optional, its default in its shape. The signature's field has the "
        "shape without the default (a default is how a call is bound, not what the model is told).",
        definition("reply", "Answer the customer.",
                   [("message", S, None), ("tone", {"type": "string", "default": "kind"}, None, {"optional": True})],
                   [("result", S, None)])),
    "13-a-list-of-types": (
        "A shape whose type is a list: the sample value is the first non-null type's (calls.md, Versions).",
        definition("label", "Label a note.",
                   [("note", {"type": ["string", "null"]}, None), ("count", {"type": ["null", "integer"]}, None)],
                   [("result", S, None)])),
    "14-a-default-by-its-code": (
        "A default counts in the version by its logic: default_code says it is written as an expression "
        "(today()), and the version holds that text, not the value it gave when the function was defined "
        "(the shape's default). The interface keeps the value.",
        definition("plan", "Plan the day.",
                   [("notes", S, None), ("day", {"type": "string", "default": "2026-09-30"}, None,
                                         {"optional": True, "default_code": "today()"}),
                    ("tone", {"type": "string", "default": "kind"}, None, {"optional": True})],
                   [("result", S, None)])),
    "15-the-same-code-another-day": (
        "The function of case 14 defined on another day: its default gives another value, and its version is "
        "the same.",
        definition("plan", "Plan the day.",
                   [("notes", S, None), ("day", {"type": "string", "default": "2026-10-01"}, None,
                                         {"optional": True, "default_code": "today()"}),
                    ("tone", {"type": "string", "default": "kind"}, None, {"optional": True})],
                   [("result", S, None)])),
    "16-another-default-value": (
        "Case 14 with tone's default changed from kind to formal: a new version.",
        definition("plan", "Plan the day.",
                   [("notes", S, None), ("day", {"type": "string", "default": "2026-09-30"}, None,
                                         {"optional": True, "default_code": "today()"}),
                    ("tone", {"type": "string", "default": "formal"}, None, {"optional": True})],
                   [("result", S, None)])),
    "17-a-default-inside-a-record": (
        "A default inside a record's shape (a member's, as pydantic writes one) is sent to the model as part "
        "of the shape, so it is in the request, but it is not in the signature id (calls.md, program.signature: "
        "every default is left out).",
        definition("summarize", "Summarize a visit.",
                   [("visit", {"type": "object", "properties": {"note": S, "day": {"type": "string",
                                                                                   "default": "2026-09-30"}},
                               "required": ["note"]}, None)],
                   [("result", S, None)])),
}


def cases() -> dict:
    out = {name: {"description": text, "definition": d, "expect": expect(d)}
           for name, (text, d) in DEFINITIONS.items()}
    v = {k: c["expect"]["version"] for k, c in out.items()}
    assert v["14-a-default-by-its-code"] == v["15-the-same-code-another-day"] != v["16-another-default-value"]
    return out
