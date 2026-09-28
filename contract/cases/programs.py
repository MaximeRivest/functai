"""The cases in programs/: a program's interface, its signature, which
interfaces are refused, and how values are checked against it, written
from the rules in ../programs.md.

Four kinds of case, told apart by "program":

- "ai": {"definition": a functions.md definition, "expect": {"interface",
  "signature", "signature_id"} or {"refuses": "interface-malformed",
  "field"}, "binds": [...]}. The interface an AI function has, from its
  definition; ``signature`` the interface's signature (the call log's
  ``program.interface``); ``signature_id`` its ``program.signature``
  (from functions.py), equal to ``signature`` when the function has
  neither reasoning nor tools. ``refuses``: defining the function is
  refused (its interface, by programs.md's rules, after lmcc accepted
  its signature). Each bind gives the function ``inputs`` and expects
  {"inputs": the values it is called with}: an optional input left out
  takes its default.
- "module": {"interface": what the module declares, "expect":
  {"signature"}, "checks": [...]}. Each check gives the module ``inputs``
  (as JSON) and expects {"inputs": the inputs its code gets} or {"refuses":
  "interface-input", "field"}; or gives what the code ``returned`` and
  expects {"outputs": the call's outputs} or {"refuses":
  "interface-output", "field"}. An object that is exactly {"$type",
  "$repr"} stands for a value with no JSON form (the harness gives the
  program a native value of its own for it).
- "definitions": {"interfaces": [{"interface", "ai"?: true, "expect":
  {"signature"} or {"refuses": "interface-malformed", "field": name or
  null}}]}. Defining a module with each interface, or reading it from a
  saved folder (with ``ai``: an AI function's interface, when the
  function is defined or its node read, whose shapes may carry keywords
  the vocabulary does not list).
- "same-data": {"interfaces": [...], "expect": {"signatures": [...]},
  "checks": [{"inputs", "expect": [one result per interface]}]}. The
  signature says what data looks like, not which calls are accepted.

"Fits" is programs.md's: the keywords it lists, read as it says, for
every interface (an AI function's other keywords are never read). This
script checks each value by those rules, and asserts that a validator of
JSON Schema draft 2020-12 (jsonschema) agrees wherever a shape has only
listed keywords, and that every shape the vocabulary accepts is a valid
2020-12 schema, so the vocabulary means what the standard means by it.
"""

import copy
import re

import jsonschema

import functions
import schemas
from common import canonical, sha

S, I, N, B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
NULL = {"type": "null"}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")             # matched whole (fullmatch): nothing after it
TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")
ANNOTATIONS = {"title": str, "description": str, "format": str, "$comment": str, "deprecated": bool,
               "readOnly": bool, "writeOnly": bool, "examples": list, "default": object}
DESCENDS = ("items", "prefixItems", "properties", "additionalProperties")
ASSERTIONS = {"type", "enum", "const", "anyOf", "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
              "properties", "required", "additionalProperties", "minLength", "maxLength", "minimum", "maximum",
              "exclusiveMinimum", "exclusiveMaximum", "$ref", "$defs"}
REF = re.compile(r"#/\$defs/([A-Za-z0-9_.-]+)")         # matched whole: no escapes, no newline after it
INPUT_KEYS = {"name", "shape", "desc", "type", "opaque", "optional"}
OUTPUT_KEYS = {"name", "shape", "desc", "type", "opaque"}


# ------------------------------------------------------------------ the rules


def data_shape(shape: dict) -> dict:
    """A field's shape without its own ``default``: what its data looks like."""
    return {k: v for k, v in shape.items() if k != "default"}


def signature(interface: dict) -> str:
    """lmcc's signature fingerprint (kernel §3a) of the interface's fields, every field plain and untyped, each
    shape without its default: what its data looks like."""
    fields = [{"direction": "input", "name": f["name"], "purpose": "plain", "shape": data_shape(f["shape"]),
               "type": ""} for f in interface["inputs"]]
    fields += [{"direction": "output", "name": f["name"], "purpose": "plain", "shape": data_shape(f["shape"]),
                "type": ""} for f in interface["outputs"]]
    return sha(fields)


def of_definition(d: dict) -> dict:
    """An AI function's interface: its definition's description, inputs (with optional) and outputs."""
    def field(f):
        return {"name": f["name"], "shape": f["shape"], **({"desc": f["desc"]} if f.get("desc") else {}),
                **({"optional": True} if f.get("optional") else {})}
    return {"description": d["description"], "inputs": [field(f) for f in d["inputs"]],
            "outputs": [field(f) for f in d["outputs"]]}


def no_json(value) -> bool:
    """A case's stand-in for a value with no JSON form."""
    return isinstance(value, dict) and set(value) == {"$type", "$repr"}


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def well_formed(shape, root, *, carry: bool = False) -> bool:
    """Whether a shape uses the vocabulary programs.md lists, each keyword (annotations included) with a
    value of its kind. ``carry``: an AI function's shape, whose other keywords are lmcc's, carried and never
    read here; a module's may have no other keyword."""
    if not isinstance(shape, dict):
        return False
    for k, v in shape.items():
        if k in ANNOTATIONS:
            if not isinstance(v, ANNOTATIONS[k]):
                return False
            continue
        if k not in ASSERTIONS:
            if carry:
                continue
            return False
        if k == "type":
            names = v if isinstance(v, list) else [v]
            if not names or any(n not in TYPES for n in names) or len(set(map(str, names))) != len(names):
                return False
        elif k == "enum" and not (isinstance(v, list) and v):
            return False
        elif k in ("anyOf", "prefixItems") and not (
                isinstance(v, list) and v and all(well_formed(x, root, carry=carry) for x in v)):
            return False
        elif k == "items" and not well_formed(v, root, carry=carry):
            return False
        elif k in ("properties", "$defs") and not (
                isinstance(v, dict) and all(well_formed(x, root, carry=carry) for x in v.values())):
            return False
        elif k == "additionalProperties" and not (isinstance(v, bool) or well_formed(v, root, carry=carry)):
            return False
        elif k == "required" and not (isinstance(v, list) and all(isinstance(x, str) for x in v)
                                      and len(set(v)) == len(v)):
            return False
        elif k in ("minItems", "maxItems", "minLength", "maxLength") and not (
                json_type(v) == "integer" and v >= 0):
            return False
        elif k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum") and not is_number(v):
            return False
        elif k == "uniqueItems" and not isinstance(v, bool):
            return False
        elif k == "$ref":
            m = REF.fullmatch(v) if isinstance(v, str) else None
            if not m or m.group(1) not in (root.get("$defs") or {}):
                return False
    return True


def same_value_refs(shape: dict) -> set:
    """The $defs a shape checks the same value against: its $ref, and those of its anyOf's shapes, without
    passing into an item or a member."""
    out = set()
    if "$ref" in shape:
        out.add(REF.fullmatch(shape["$ref"]).group(1))
    for x in shape.get("anyOf", []):
        out |= same_value_refs(x)
    return out


def loops(root: dict) -> bool:
    """Whether a $defs entry reaches itself through $ref and anyOf alone: checking a value against it would
    never end (a record that holds a list of itself descends into a smaller value, and is not a loop)."""
    defs = root.get("$defs") or {}
    graph = {n: same_value_refs(d) for n, d in defs.items()}

    def reaches(start, seen):
        for n in graph.get(start, ()):
            if n == seen[0] or (n not in seen and reaches(n, seen + (n,))):
                return True
        return False
    return any(reaches(n, (n,)) for n in graph)


def json_type(v) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "integer" if float(v).is_integer() else "number"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if is_number(v):
        return "integer" if float(v).is_integer() else "number"
    return {str: "string", list: "array", dict: "object"}[type(v)]


def fits_shape(v, shape: dict, root: dict) -> bool:
    """programs.md, "Checking values": the value (JSON) against a shape of the vocabulary."""
    t = json_type(v)
    if "$ref" in shape and not fits_shape(v, root["$defs"][REF.fullmatch(shape["$ref"]).group(1)], root):
        return False
    if "type" in shape:
        names = shape["type"] if isinstance(shape["type"], list) else [shape["type"]]
        if not (t in names or (t == "integer" and "number" in names)):
            return False
    if "enum" in shape and canonical(v) not in {canonical(x) for x in shape["enum"]}:
        return False
    if "const" in shape and canonical(v) != canonical(shape["const"]):
        return False
    if "anyOf" in shape and not any(fits_shape(v, s, root) for s in shape["anyOf"]):
        return False
    if t == "string":
        if len(v) < shape.get("minLength", 0) or ("maxLength" in shape and len(v) > shape["maxLength"]):
            return False
    if t in ("integer", "number"):
        if "minimum" in shape and v < shape["minimum"] or "maximum" in shape and v > shape["maximum"]:
            return False
        if "exclusiveMinimum" in shape and v <= shape["exclusiveMinimum"]:
            return False
        if "exclusiveMaximum" in shape and v >= shape["exclusiveMaximum"]:
            return False
    if t == "array":
        prefix = shape.get("prefixItems", [])
        if any(not fits_shape(x, s, root) for x, s in zip(v, prefix)):
            return False
        if "items" in shape and any(not fits_shape(x, shape["items"], root) for x in v[len(prefix):]):
            return False
        if len(v) < shape.get("minItems", 0) or ("maxItems" in shape and len(v) > shape["maxItems"]):
            return False
        if shape.get("uniqueItems") and len({canonical(x) for x in v}) != len(v):
            return False
    if t == "object":
        props = shape.get("properties", {})
        if any(k not in v for k in shape.get("required", [])):
            return False
        for k, x in v.items():
            if k in props:
                if not fits_shape(x, props[k], root):
                    return False
            elif "additionalProperties" in shape:
                extra = shape["additionalProperties"]
                if extra is False or (isinstance(extra, dict) and not fits_shape(x, extra, root)):
                    return False
    return True


def vocabulary_only(shape) -> bool:
    return well_formed(shape, shape)


def fits(value, field: dict) -> bool:
    """programs.md: the listed keywords decide; another keyword (in an AI function's shape, lmcc's) is never
    read. Where a shape has only listed keywords, a draft 2020-12 validator must agree."""
    if field.get("opaque"):
        return True
    if no_json(value):
        return False
    shape = data_shape(field["shape"])
    got = fits_shape(value, shape, shape)
    if vocabulary_only(shape):
        assert got == jsonschema.Draft202012Validator(shape).is_valid(value), (value, shape)
    return got


def malformed(interface, *, ai: bool = False):
    """The first reason an interface is refused when it is defined or read, or None. Its form (the keys it
    may have, and their kinds) and its meaning are checked together, field by field: the refusal names the
    first field at fault, inputs then outputs, in order; null when the fault is not a field's. ``ai``: an AI
    function's interface, when it is defined (after lmcc's own check, signature-malformed, which asks only
    that each shape is an object) or its node read; its shapes may carry lmcc's other keywords. Only a
    field's own default must fit; a default inside a shape is a word, never checked."""
    def refuse(field):
        return {"refuses": "interface-malformed", "field": field}
    if not isinstance(interface, dict) or set(interface) - {"description", "inputs", "outputs"} \
            or not isinstance(interface.get("description"), str) or not isinstance(interface.get("inputs"), list) \
            or not isinstance(interface.get("outputs"), list) or not interface["outputs"]:
        return refuse(None)
    seen = set()
    for direction in ("inputs", "outputs"):
        for f in interface[direction]:
            name = f.get("name") if isinstance(f, dict) else None
            at_fault = refuse(name if isinstance(name, str) else None)
            if not isinstance(f, dict) or set(f) - (INPUT_KEYS if direction == "inputs" else OUTPUT_KEYS):
                return at_fault
            if not isinstance(name, str) or not NAME.fullmatch(name) or name in seen:
                return at_fault
            seen.add(name)
            if any(k in f and not isinstance(f[k], str) for k in ("desc", "type")):
                return at_fault
            if any(k in f and f[k] is not True for k in ("opaque", "optional")):
                return at_fault
            shape = f.get("shape")
            if not isinstance(shape, dict):
                return at_fault
            if not well_formed(shape, shape, carry=ai) or loops(shape):
                return at_fault
            if ai and f.get("optional") and "default" not in shape:
                return at_fault                     # a model is sent every input: an optional one needs a default
            if f.get("opaque") and shape != {}:
                return at_fault
            if "default" in shape:
                ds = data_shape(shape)
                if not fits_shape(shape["default"], ds, ds):
                    return at_fault
    return None


def check_inputs(interface: dict, given: dict) -> dict:
    names = [f["name"] for f in interface["inputs"]]
    unknown = sorted(k for k in given if k not in names)
    if unknown:
        return {"refuses": "interface-input", "field": unknown[0]}
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
        names = [f["name"] for f in outputs]
        unknown = sorted(k for k in returned if k not in names)
        if unknown:
            return {"refuses": "interface-output", "field": unknown[0]}
        values = returned
    for f in outputs:
        if f["name"] not in values or not fits(values[f["name"]], f):
            return {"refuses": "interface-output", "field": f["name"]}
    return {"outputs": {f["name"]: values[f["name"]] for f in outputs}}


def rerun(case: dict) -> None:
    """A case as written (read back from its file), run through the rules above as a harness would."""
    kind = case["program"]
    if kind == "ai":
        iface = of_definition(case["definition"])
        refused = malformed(iface, ai=True)
        assert (refused or {"interface": iface, "signature": signature(iface)}) == \
            {k: v for k, v in case["expect"].items() if k != "signature_id"}, case["description"]
        for b in case["binds"]:
            assert check_inputs(iface, b["inputs"]) == b["expect"], case["description"]
    elif kind == "module":
        iface = case["interface"]
        assert malformed(iface) is None and signature(iface) == case["expect"]["signature"]
        for c in case["checks"]:
            got = check_inputs(iface, c["inputs"]) if "inputs" in c else check_returned(iface, c["returned"])
            assert got == c["expect"], (case["description"], c)
    elif kind == "definitions":
        for x in case["interfaces"]:
            refused = malformed(x["interface"], ai=x.get("ai", False))
            assert (refused or {"signature": signature(x["interface"])}) == x["expect"], x
    elif kind == "same-data":
        assert [signature(x) for x in case["interfaces"]] == case["expect"]["signatures"]
        for c in case["checks"]:
            assert [check_inputs(x, c["inputs"]) for x in case["interfaces"]] == c["expect"]


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
SHAPES = interface("Tag an order.",
                   [("code", {"type": "string", "minLength": 1, "maxLength": 1, "title": "Code"}, {}),
                    ("tags", {"type": "array", "items": {}, "uniqueItems": True}, {}),
                    ("order", {"$ref": "#/$defs/Order",
                               "$defs": {"Order": {"type": "object", "properties": {"id": I}, "required": ["id"]}}},
                     {}),
                    ("day", {"type": "string", "format": "date", "default": "2026-09-28"}, {"optional": True})],
                   [("result", S, {})])


NODE = {"$ref": "#/$defs/Node",
        "$defs": {"Node": {"type": "object", "properties": {"name": S, "children": {"type": "array",
                                                                                   "items": {"$ref": "#/$defs/Node"}}},
                           "required": ["name", "children"]}}}
AI_PATTERN = interface("Tag a ticket.", [("code", {"type": "string", "pattern": "^[a-z]+$", "default": "ABC"},
                                          {"optional": True})],
                       [("result", {"oneOf": [S, NULL]}, {})])


NESTED = interface("Count an order's parcels.",
                   [("order", {"type": "object", "properties": {"id": I, "count": {"type": "integer", "default": "none"}},
                               "required": ["id"]}, {})],
                   [("result", I, {})])


def module_case(description, iface, inputs=(), returned=()):
    assert malformed(iface) is None
    checks = [{"inputs": i, "expect": check_inputs(iface, i)} for i in inputs]
    checks += [{"returned": r, "expect": check_returned(iface, r)} for r in returned]
    return {"description": description, "program": "module", "interface": iface,
            "expect": {"signature": signature(iface)}, "checks": checks}


def ai_case(description, key, binds=()):
    d = functions.DEFINITIONS[key][1]
    iface = of_definition(d)
    assert malformed(iface, ai=True) is None
    return {"description": description, "program": "ai", "definition": d,
            "expect": {"interface": iface, "signature": signature(iface),
                       "signature_id": functions.expect(d)["signature_id"]},
            "binds": [{"inputs": i, "expect": check_inputs(iface, i)} for i in binds]}


def ai_refused_case(description, d):
    """Defining an AI function whose signature lmcc accepts, and whose interface programs.md refuses."""
    import lmcc
    lmcc.signature_from_dict({"instructions": functions.instructions(d), "fields": functions.fields(d)})
    refused = malformed(of_definition(d), ai=True)
    assert refused is not None
    return {"description": description, "program": "ai", "definition": d, "expect": refused, "binds": []}


def definitions_case(description, interfaces, ai=()):
    """``ai``: the indexes of the interfaces read as an AI node's."""
    out = []
    for i, iface in enumerate(interfaces):
        refused = malformed(iface, ai=i in ai)
        if refused is None:
            assert schemas.INTERFACE.is_valid(iface), iface
            for f in iface["inputs"] + iface["outputs"]:
                if vocabulary_only(f["shape"]):
                    jsonschema.Draft202012Validator.check_schema(f["shape"])
        entry = {"interface": iface}
        if i in ai:
            entry["ai"] = True
        out.append(entry | {"expect": refused or {"signature": signature(iface)}})
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
    today = interface("Plan the day.", [("since", {"type": "string", "default": "2026-09-28"}, {"optional": True})],
                      [("result", S, {})])
    tomorrow = field_with(today, "inputs", 0, shape={"type": "string", "default": "2026-09-29"})
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
            "the last is the answer; each must be there and fit. An integer is a number with no fraction: 5.0 "
            "fits, 5.5 does not.", TRIAGE,
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
            "checked; only the names are. Its values cannot cross a boundary that needs data. How a value is "
            "written in the log does not depend on the field: [1, 2] is written as JSON, a data frame as a "
            "description (calls.md, Values).", OPAQUE,
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
            "An interface is refused when it is defined or read (interface-malformed), its form and its meaning "
            "checked together, field by field: the refusal names the first field at fault, inputs then outputs, "
            "in order; null when the fault is not a field's (no output; a key the interface does not have). A "
            "field has only the keys programs.md names (a later key is refused, not ignored); a name is an ASCII "
            "identifier used once across inputs and outputs; a shape uses only the listed keywords; a default "
            "fits its shape; an opaque field's shape is {}; only inputs are optional, and optional and opaque are "
            "true when present.",
            [SUPPORT,
             interface("No output.", [("message", S, {})], []),
             field_with(SUPPORT, "outputs", 0, name="message"),
             field_with(SUPPORT, "inputs", 0, name="the message"),
             field_with(SUPPORT, "inputs", 0, shape={"type": "text"}),
             field_with(SUPPORT, "inputs", 0, shape={"type": "string", "default": 3}, optional=True),
             field_with(SUPPORT, "inputs", 0, opaque=True),
             field_with(SUPPORT, "outputs", 0, optional=True),
             interface("Two faults: the first is named.", [("a", {"type": "strin"}, {}), ("a", S, {})],
                       [("result", S, {})]),
             field_with(SUPPORT, "inputs", 0, optional=False),
             field_with(SUPPORT, "inputs", 0, label="private"),
             {**SUPPORT, "version": 2},
             field_with(SUPPORT, "inputs", 0, shape={"type": "string", "pattern": "^B-[0-9]+$"}),
             field_with(SUPPORT, "outputs", 0, shape={"$ref": "#/$defs/Reply"}),
             interface("A form fault after a meaning fault: the first field is named.",
                       [("a", {"type": "strin"}, {}), ("b", S, {"optional": False})], [("result", S, {})]),
             interface("A record by reference, as pydantic writes one.",
                       [("order", {"$ref": "#/$defs/Order", "$defs": {"Order": {
                           "type": "object", "title": "Order", "properties": {"id": I}, "required": ["id"]}}}, {})],
                       [("result", S, {})])]),
        "12-the-signature-is-the-data": {
            "description": "Required and optional inputs of one shape have one signature: the signature says "
                           "what recorded data looks like, not which calls are accepted (compare interfaces for "
                           "that). A default is behaviour, not data: it is left out of the signature, so a "
                           "default computed when the program is loaded (today's date) does not split its "
                           "records.",
            "program": "same-data", "interfaces": [required, optional, defaulted, today, tomorrow],
            "expect": {"signatures": [signature(x) for x in (required, optional, defaulted, today, tomorrow)]},
            "checks": [{"inputs": i, "expect": [check_inputs(x, i) for x in (required, optional, defaulted, today,
                                                                             tomorrow)]}
                       for i in ({}, {"name": None}, {"name": "Ana"})]},
        "13-an-optional-input-of-an-ai-function": ai_case(
            "An AI function's input the language lets a caller leave out is optional, and its default is in its "
            "shape (a model is sent every input: an optional input with no default is refused). Left out, the "
            "input takes the default; the signature leaves the default out, so it is still program.signature.",
            "12-an-optional-input", binds=[{"message": "Hi"}, {"message": "Hi", "tone": "brief"}]),
        "14-what-fits": module_case(
            "Fits is checked with the keywords programs.md lists, the same in every language: lengths count "
            "code points; items are unique by canonical JSON (1 and 1.0 are one value); a record by reference "
            "to $defs; words such as title, format and default are never checked.", SHAPES,
            inputs=[{"code": "é", "tags": ["a", "b"], "order": {"id": 1}, "day": "not a date"},
                    {"code": "", "tags": ["a"], "order": {"id": 1}},
                    {"code": "é", "tags": [1, 1.0], "order": {"id": 1}},
                    {"code": "éé", "tags": [], "order": {"id": 1}},
                    {"code": "é", "tags": [], "order": {"id": "1"}}]),
        "15-which-name-is-named": module_case(
            "When several names are at fault, the first in code-point order is named (a map has no order every "
            "language keeps); unknown names before anything else, then each field in the interface's order.",
            TRIAGE,
            inputs=[{"ticket": "?", "zeta": 1, "Alpha": 2, "alpha": 3}],
            returned=[{"team": "billing", "result": {"reply": "Refunded.", "escalate": False}, "z": 1, "b": 2},
                      {"result": {"reply": "Refunded.", "escalate": False}, "team": 3}]),
        "16-shapes-that-are-refused": definitions_case(
            "A shape's words are checked for their kind too (title is text, examples a list), prefixItems and "
            "anyOf hold at least one shape, a count is an integer (2.0 is one: every language reads it so), and "
            "additionalProperties is a shape, true or false. A $ref that comes back to itself through $ref and "
            "anyOf alone, never passing into an item or a member, is refused: checking a value against it would "
            "never end. A record that holds a list of itself descends into a smaller value, and is accepted.",
            [field_with(SUPPORT, "inputs", 0, shape={"type": "string", "title": 7}),
             field_with(SUPPORT, "inputs", 0, shape={"type": "array", "prefixItems": []}),
             field_with(SUPPORT, "inputs", 0, shape={"type": "string", "examples": "hi"}),
             field_with(SUPPORT, "inputs", 0, shape={"$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
                                                     "$ref": "#/$defs/Loop"}),
             field_with(SUPPORT, "inputs", 0, shape={"$defs": {"A": {"anyOf": [S, {"$ref": "#/$defs/B"}]},
                                                               "B": {"$ref": "#/$defs/A"}},
                                                     "$ref": "#/$defs/A"}),
             field_with(SUPPORT, "inputs", 0, shape={"$defs": {"Loop": {"$ref": "#/$defs/Loop"}}, "type": "string"}),
             field_with(SUPPORT, "inputs", 0, shape={"type": "object", "additionalProperties": True}),
             field_with(SUPPORT, "inputs", 0, shape={"type": "array", "items": S, "minItems": 2.0}),
             field_with(SUPPORT, "inputs", 0, shape=NODE)]),
        "17-a-record-that-holds-itself": module_case(
            "A shape by reference to itself, through an item: checking descends into the value, and ends.",
            interface("Count a tree's leaves.", [("tree", NODE, {})], [("result", I, {})]),
            inputs=[{"tree": {"name": "root", "children": [{"name": "a", "children": []}]}},
                    {"tree": {"name": "root", "children": [{"name": "a", "children": [1]}]}}]),
        "18-an-ai-functions-shapes": definitions_case(
            "An AI node's interface, read from a saved folder: its shapes are lmcc's, and may carry keywords "
            "the vocabulary does not list (pattern, oneOf), which lmcc passes on and FunctAI never reads. Its "
            "default fits by the listed keywords alone: a default that a pattern would refuse is accepted, the "
            "same in every language. A listed keyword with a value of the wrong kind is refused as for a "
            "module. The same interface is refused as a module's (pattern is not in the vocabulary).",
            [AI_PATTERN, field_with(AI_PATTERN, "inputs", 0, shape={"type": "string", "minLength": "3",
                                                                     "default": "abc"}),
             AI_PATTERN], ai=(0, 1)),
        "19-an-ai-function-refused-when-defined": ai_refused_case(
            "An AI function's interface is checked by the rules above when the function is defined, after lmcc "
            "accepted its signature (lmcc asks only that each shape is an object): an optional input whose "
            "default does not fit its shape (Python: pydantic's Field(ge=10) with a default of 5, which pydantic "
            "does not check) is refused then, not later by every language that reads the saved folder. The "
            "other faults of an AI function's interface (programs/11, 16, 18 with ai) are refused at definition "
            "too: what one language lets a function be defined and saved with, every language can load.",
            functions.definition("count", "Count the items worth keeping.",
                                 [("items", {"type": "array", "items": S}, None),
                                  ("at_least", {"type": "integer", "minimum": 10, "default": 5}, None,
                                   {"optional": True})],
                                 [("result", I, None)])),
        "20-names-and-references-are-matched-whole": definitions_case(
            "A field's name and a $ref are matched whole, the same in every language: nothing may follow them, "
            "not even a newline (a regular expression's $ may match before one). A $ref is #/$defs/ and a name "
            "of ASCII letters, digits, _, . and - (every name pydantic writes fits), read as it is written: no JSON "
            "Pointer escape (~1) and no percent-encoding (%20), so a $defs entry whose name has other characters "
            "cannot be referred to.",
            [field_with(SUPPORT, "inputs", 0, shape={"$defs": {"Node": S}, "$ref": "#/$defs/Node\n"}),
             field_with(SUPPORT, "inputs", 0, shape={"$defs": {"Foo/Bar": S}, "$ref": "#/$defs/Foo~1Bar"}),
             field_with(SUPPORT, "inputs", 0, shape={"$defs": {"Foo Bar": S}, "$ref": "#/$defs/Foo%20Bar"}),
             field_with(SUPPORT, "inputs", 0, name="message\n"),
             field_with(SUPPORT, "inputs", 0, shape={"$defs": {"shop.Order-2_v": S}, "$ref": "#/$defs/shop.Order-2_v"}),
             field_with(SUPPORT, "inputs", 0, shape={"$defs": {"Foo Bar": S}, "type": "string"})]),
        "21-a-default-inside-a-shape": module_case(
            "Only a field's own default is the value it takes, and must fit. A default inside its shape (a "
            "member's, as pydantic writes one for a model's defaulted field) is a word, like title: never "
            "checked, whatever it holds, and never filled in (the value is checked as given; the program's own "
            "types may fill it). It stays in the signature: it is part of the shape, as it is of lmcc's signature.",
            NESTED, inputs=[{"order": {"id": 1}}, {"order": {"id": 1, "count": 2}}, {"order": {"id": 1, "count": "none"}}]),
    }
    same = [out["01-an-ai-function-is-its-definition"], out["02-several-outputs-keep-their-words"],
            out["13-an-optional-input-of-an-ai-function"]]
    assert all(c["expect"]["signature"] == c["expect"]["signature_id"] for c in same)
    assert all(c["expect"]["signature"] != c["expect"]["signature_id"]
               for c in (out["03-reasoning-is-not-given-back"], out["04-tools-are-not-given"]))
    sigs = out["12-the-signature-is-the-data"]["expect"]["signatures"]
    assert sigs[0] == sigs[1] == sigs[2] and sigs[3] == sigs[4]
    whole = [x["expect"].get("refuses") for x in out["20-names-and-references-are-matched-whole"]["interfaces"]]
    assert whole == ["interface-malformed"] * 4 + [None, None], whole
    nested = out["21-a-default-inside-a-shape"]
    assert [c["expect"] for c in nested["checks"]][0] == {"inputs": {"order": {"id": 1}}}
    plain = copy.deepcopy(NESTED)
    del plain["inputs"][0]["shape"]["properties"]["count"]["default"]
    assert signature(plain) != nested["expect"]["signature"]
    refused = [x["expect"].get("refuses") for x in out["11-interfaces-that-are-refused"]["interfaces"]]
    assert refused[0] is None and refused[-1] is None and all(refused[1:-1]), refused
    return out
