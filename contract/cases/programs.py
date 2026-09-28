"""The cases in programs/: a program's interface, its signature, which
interfaces are refused, and how values are checked against it, written
from the rules in ../programs.md.

Four kinds of case, told apart by "program":

- "ai": {"definition": a functions.md definition, "expect": {"interface",
  "signature", "signature_id"}}. The interface an AI function has, from its
  definition; ``signature`` the interface's signature (the call log's
  ``program.interface``); ``signature_id`` its ``program.signature`` (from
  functions.py), equal to ``signature`` when the function has neither
  reasoning nor tools.
- "module": {"interface": what the module declares, "expect":
  {"signature"}, "checks": [...]}. Each check gives the module ``inputs``
  (as JSON) and expects {"inputs": the inputs its code gets} or {"refuses":
  "interface-input", "field"}; or gives what the code ``returned`` and
  expects {"outputs": the call's outputs} or {"refuses":
  "interface-output", "field"}. An object that is exactly {"$type",
  "$repr"} stands for a value with no JSON form (the harness gives the
  program a native value of its own for it).
- "definitions": {"interfaces": [{"interface", "expect": {"signature"} or
  {"refuses": "interface-malformed", "field": name or null}}]}. Defining a
  module with each interface (or reading it from a saved folder).
- "same-data": {"interfaces": [...], "expect": {"signatures": [...]},
  "checks": [{"inputs", "expect": [one result per interface]}]}. The
  signature says what data looks like, not which calls are accepted.

"Fits" is JSON Schema, draft 2020-12, as programs.md says; this script uses
a validator of that standard (jsonschema), never FunctAI.
"""

import copy
import re

import jsonschema

import functions
from common import sha

S, I, N, B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
NULL = {"type": "null"}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ------------------------------------------------------------------ the rules


def signature(interface: dict) -> str:
    """lmcc's signature fingerprint of the interface's fields (kernel §3a), every
    field plain and untyped: what its data looks like."""
    fields = [{"direction": "input", "name": f["name"], "purpose": "plain", "shape": f["shape"], "type": ""}
              for f in interface["inputs"]]
    fields += [{"direction": "output", "name": f["name"], "purpose": "plain", "shape": f["shape"], "type": ""}
               for f in interface["outputs"]]
    return sha(fields)


def of_definition(d: dict) -> dict:
    """An AI function's interface: its definition's description, inputs and outputs."""
    def field(f):
        return {"name": f["name"], "shape": f["shape"], **({"desc": f["desc"]} if f.get("desc") else {})}
    return {"description": d["description"], "inputs": [field(f) for f in d["inputs"]],
            "outputs": [field(f) for f in d["outputs"]]}


def no_json(value) -> bool:
    """A case's stand-in for a value with no JSON form."""
    return isinstance(value, dict) and set(value) == {"$type", "$repr"}


def fits(value, field: dict) -> bool:
    if field.get("opaque"):
        return True
    if no_json(value):
        return False
    return jsonschema.Draft202012Validator(field["shape"]).is_valid(value)


def malformed(interface: dict):
    """The first reason an interface is refused when it is defined or read, or None."""
    def refuse(field):
        return {"refuses": "interface-malformed", "field": field}
    if not isinstance(interface.get("outputs"), list) or not interface["outputs"]:
        return refuse(None)
    seen = set()
    for direction in ("inputs", "outputs"):
        for f in interface[direction]:
            name = f.get("name")
            if not isinstance(name, str) or not NAME.match(name) or name in seen:
                return refuse(name)
            seen.add(name)
            shape = f.get("shape")
            if not isinstance(shape, dict):
                return refuse(name)
            try:
                jsonschema.Draft202012Validator.check_schema(shape)
            except jsonschema.SchemaError:
                return refuse(name)
            if direction == "outputs" and "optional" in f:
                return refuse(name)
            if f.get("opaque") and shape != {}:
                return refuse(name)
            if "default" in shape and not jsonschema.Draft202012Validator(shape).is_valid(shape["default"]):
                return refuse(name)
    return None


def check_inputs(interface: dict, given: dict) -> dict:
    names = [f["name"] for f in interface["inputs"]]
    for name in given:
        if name not in names:
            return {"refuses": "interface-input", "field": name}
    out = {}
    for f in interface["inputs"]:
        if f["name"] in given:
            value = given[f["name"]]
            if not fits(value, f):
                return {"refuses": "interface-input", "field": f["name"]}
            out[f["name"]] = value
        elif not f.get("optional"):
            return {"refuses": "interface-input", "field": f["name"]}
        elif "default" in f["shape"]:
            out[f["name"]] = copy.deepcopy(f["shape"]["default"])
        # else: left out, the program's own default applies; the input stays absent
    return {"inputs": out}


def check_returned(interface: dict, returned) -> dict:
    outputs = interface["outputs"]
    if len(outputs) == 1:
        values = {outputs[0]["name"]: returned}
    else:
        if not isinstance(returned, dict) or no_json(returned):
            return {"refuses": "interface-output", "field": outputs[0]["name"]}
        for f in outputs:
            if f["name"] not in returned:
                return {"refuses": "interface-output", "field": f["name"]}
        names = [f["name"] for f in outputs]
        for key in returned:
            if key not in names:
                return {"refuses": "interface-output", "field": key}
        values = {f["name"]: returned[f["name"]] for f in outputs}
    for f in outputs:
        if not fits(values[f["name"]], f):
            return {"refuses": "interface-output", "field": f["name"]}
    return {"outputs": values}


# ------------------------------------------------------------------ the cases


def interface(description, inputs, outputs):
    def one(n, s, extra):
        return {"name": n, "shape": s, **extra}
    return {"description": description, "inputs": [one(*f) for f in inputs], "outputs": [one(*f) for f in outputs]}


SUPPORT = interface("Answer a customer's message.", [("message", S, {})], [("result", S, {})])
TONE = interface("Answer a customer's message, in a tone.",
                 [("message", S, {"desc": "what the customer wrote"}),
                  ("tone", {"type": "string", "default": "kind"}, {"optional": True}),
                  ("order", {"anyOf": [S, NULL]}, {"optional": True}),
                  ("since", S, {"optional": True})],
                 [("result", S, {})])
TRIAGE = interface("Sort a ticket and say how urgent it is.",
                   [("ticket", S, {})],
                   [("team", {"enum": ["billing", "shipping", "product"], "type": "string"}, {}),
                    ("minutes", I, {"desc": "to fix"}),
                    ("result", {"type": "object", "properties": {"reply": S, "escalate": B},
                                "required": ["reply", "escalate"]}, {})])
OPAQUE = interface("A pipeline over a data frame.",
                   [("frame", {}, {"type": "pd.DataFrame", "opaque": True}),
                    ("args", {"type": "array", "default": []}, {"optional": True})],
                   [("result", {}, {"opaque": True})])
ANY_JSON = interface("Store any JSON value under a key.",
                     [("key", S, {}), ("value", {}, {"type": "JSON"})],
                     [("result", {}, {})])
TRACKING = interface("Where a parcel is.",
                     [("order", S, {})],
                     [("result", {"type": "object", "properties": {"where": S, "late_days": I},
                                  "required": ["where", "late_days"]}, {"type": "Tracking"})])
DATAFRAME = {"$type": "DataFrame", "$repr": "   a\n0  1"}


def module_case(description, iface, inputs=(), returned=()):
    assert malformed(iface) is None
    checks = [{"inputs": i, "expect": check_inputs(iface, i)} for i in inputs]
    checks += [{"returned": r, "expect": check_returned(iface, r)} for r in returned]
    return {"description": description, "program": "module", "interface": iface,
            "expect": {"signature": signature(iface)}, "checks": checks}


def ai_case(description, key):
    d = functions.DEFINITIONS[key][1]
    iface = of_definition(d)
    return {"description": description, "program": "ai", "definition": d,
            "expect": {"interface": iface, "signature": signature(iface),
                       "signature_id": functions.expect(d)["signature_id"]}}


def definitions_case(description, interfaces):
    out = []
    for iface in interfaces:
        refused = malformed(iface)
        out.append({"interface": iface, "expect": refused or {"signature": signature(iface)}})
    return {"description": description, "program": "definitions", "interfaces": out}


def field_with(iface, where, i, **change):
    iface = copy.deepcopy(iface)
    iface[where][i].update(change)
    return iface


def cases() -> dict:
    nullable = {"anyOf": [S, NULL]}
    required = interface("Greet someone.", [("name", nullable, {})], [("result", S, {})])
    optional = field_with(required, "inputs", 0, optional=True)
    defaulted = field_with(required, "inputs", 0, optional=True, shape={"anyOf": [S, NULL], "default": "friend"})
    out = {
        "01-an-ai-function-is-its-definition": ai_case(
            "An AI function's interface is its definition's inputs and outputs; with neither reasoning nor "
            "tools its signature is its program.signature.", "01-an-answer-from-a-list"),
        "02-several-outputs-keep-their-words": ai_case(
            "Outputs keep their desc in the interface (they are not in the signature's fields); the signature "
            "leaves descriptions out, so it is still program.signature.", "02-several-outputs"),
        "03-reasoning-is-not-given-back": ai_case(
            "module cot adds a reasoning output to the lmcc signature, not to the interface: program.signature "
            "and the interface's signature differ.", "07-reasoning-first"),
        "04-tools-are-not-given": ai_case(
            "Tools add a tools input and a calls output to the lmcc signature, not to the interface.", "11-a-tool"),
        "05-a-module-checks-its-inputs": module_case(
            "A module's inputs are checked before its code runs: every one given, none it does not have, "
            "each fitting its shape.", SUPPORT,
            inputs=[{"message": "Where is my parcel?"}, {}, {"message": 3},
                    {"message": "Hi", "urgent": True}, {"message": None}],
            returned=["It left the depot today.", 42, None]),
        "06-optional-inputs": module_case(
            "An optional input left out takes its shape's default; without one it stays left out (never null: "
            "the program's own default applies, and the record has no value for it). A given value is checked, "
            "null included.", TONE,
            inputs=[{"message": "Hi"}, {"message": "Hi", "tone": "brief"}, {"message": "Hi", "order": "B-2210"},
                    {"message": "Hi", "order": None}, {"message": "Hi", "tone": None}, {"tone": "brief"}]),
        "07-several-outputs": module_case(
            "A module that declares several outputs returns one record with each by name, and nothing else; "
            "the last is the answer; each must be there and fit.", TRIAGE,
            returned=[{"team": "billing", "minutes": 5, "result": {"reply": "Refunded.", "escalate": False}},
                      {"team": "billing", "minutes": 5, "result": {"reply": "Refunded.", "escalate": False},
                       "note": "a key it does not declare"},
                      {"team": "legal", "minutes": 5, "result": {"reply": "?", "escalate": True}},
                      {"team": "billing", "result": {"reply": "Refunded.", "escalate": False}},
                      {"team": "billing", "minutes": 5.5, "result": {"reply": "Refunded.", "escalate": False}},
                      {"team": "billing", "minutes": 5.0, "result": {"reply": "Refunded.", "escalate": False}},
                      "billing"]),
        "08-opaque-fields": module_case(
            "An opaque field (a type with no JSON form, such as a data frame) takes any value and is never "
            "checked; only the names are. Its values cannot cross a boundary that needs data.", OPAQUE,
            inputs=[{"frame": DATAFRAME}, {"frame": [1, 2], "args": [3]}, {"args": []}],
            returned=[DATAFRAME, None]),
        "09-any-json-is-not-opaque": module_case(
            "A shape of {} that is not opaque is JSON Schema's: any JSON value. A value with no JSON form does "
            "not fit it.", ANY_JSON,
            inputs=[{"key": "k", "value": {"a": [1, None]}}, {"key": "k", "value": None},
                    {"key": "k", "value": DATAFRAME}],
            returned=[[1, "two"], DATAFRAME]),
        "10-a-record-is-one-output": module_case(
            "A return type that is a record is one output of object shape (Python: -> Tracking, a TypedDict "
            "or dataclass); several outputs are declared as several.", TRACKING,
            returned=[{"where": "Leeds depot", "late_days": 7}, {"where": "Leeds depot"}]),
        "11-interfaces-that-are-refused": definitions_case(
            "An interface is refused when it is defined or read (interface-malformed), naming the first field "
            "at fault, inputs then outputs, in order; null when it has no output. A name is an ASCII identifier "
            "used once across inputs and outputs; a shape is valid JSON Schema; a default fits its shape; "
            "an opaque field's shape is {}; only inputs are optional.",
            [SUPPORT,
             interface("No output.", [("message", S, {})], []),
             field_with(SUPPORT, "outputs", 0, name="message"),
             field_with(SUPPORT, "inputs", 0, name="the message"),
             field_with(SUPPORT, "inputs", 0, shape={"type": "text"}),
             field_with(SUPPORT, "inputs", 0, shape={"type": "string", "default": 3}, optional=True),
             field_with(SUPPORT, "inputs", 0, opaque=True),
             field_with(SUPPORT, "outputs", 0, optional=True),
             interface("Two faults: the first is named.", [("a", {"type": "strin"}, {}), ("a", S, {})],
                       [("result", S, {})])]),
        "12-the-signature-is-the-data": {
            "description": "Required and optional inputs of one shape have one signature: the signature says "
                           "what recorded data looks like, not which calls are accepted (compare interfaces for "
                           "that). A default is part of the shape, so it is part of the signature.",
            "program": "same-data", "interfaces": [required, optional, defaulted],
            "expect": {"signatures": [signature(required), signature(optional), signature(defaulted)]},
            "checks": [{"inputs": i, "expect": [check_inputs(x, i) for x in (required, optional, defaulted)]}
                       for i in ({}, {"name": None}, {"name": "Ana"})]},
    }
    same = [out["01-an-ai-function-is-its-definition"], out["02-several-outputs-keep-their-words"]]
    assert all(c["expect"]["signature"] == c["expect"]["signature_id"] for c in same)
    assert all(c["expect"]["signature"] != c["expect"]["signature_id"]
               for c in (out["03-reasoning-is-not-given-back"], out["04-tools-are-not-given"]))
    sigs = out["12-the-signature-is-the-data"]["expect"]["signatures"]
    assert sigs[0] == sigs[1] != sigs[2]
    refused = [x["expect"].get("refuses") for x in out["11-interfaces-that-are-refused"]["interfaces"]]
    assert refused[0] is None and all(refused[1:]), refused
    return out
