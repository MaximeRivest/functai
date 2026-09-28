"""The cases in programs/: a program's interface, its id, and how values are
checked against it, written from the rules in ../programs.md.

Two kinds of case, told apart by "program":

- "ai": {"definition": a functions.md definition, "expect": {"interface",
  "id", "signature_id"}}. The interface an AI function has, from its
  definition; ``id`` the interface's id; ``signature_id`` its
  program.signature (from functions.py), equal to ``id`` when the function
  has neither reasoning nor tools.
- "module": {"interface": what the module declares, "expect": {"id"},
  "checks": [...]}. Each check gives the module ``inputs`` (as JSON) and
  expects {"inputs": the inputs its code gets} or {"refuses":
  "interface-input", "field"}; or gives what the code ``returned`` and
  expects {"outputs": the call's outputs} or {"refuses":
  "interface-output", "field"}. A value that fits is kept as given.

"Fits" is JSON Schema, draft 2020-12, as programs.md says; this script uses
a validator of that standard (jsonschema), never FunctAI.
"""

import copy

import jsonschema

import functions
from common import sha

S, I, N, B = {"type": "string"}, {"type": "integer"}, {"type": "number"}, {"type": "boolean"}
NULL = {"type": "null"}


# ------------------------------------------------------------------ the rules


def interface_id(interface: dict) -> str:
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


def fits(value, shape: dict) -> bool:
    return jsonschema.Draft202012Validator(shape).is_valid(value)


def check_inputs(interface: dict, given: dict) -> dict:
    names = [f["name"] for f in interface["inputs"]]
    for name in given:
        if name not in names:
            return {"refuses": "interface-input", "field": name}
    out = {}
    for f in interface["inputs"]:
        if f["name"] in given:
            value = given[f["name"]]
        elif f.get("optional"):
            value = copy.deepcopy(f["shape"].get("default"))
        else:
            return {"refuses": "interface-input", "field": f["name"]}
        if not fits(value, f["shape"]):
            return {"refuses": "interface-input", "field": f["name"]}
        out[f["name"]] = value
    return {"inputs": out}


def check_returned(interface: dict, returned) -> dict:
    outputs = interface["outputs"]
    if len(outputs) == 1:
        values = {outputs[0]["name"]: returned}
    else:
        if not isinstance(returned, dict):
            return {"refuses": "interface-output", "field": outputs[0]["name"]}
        values = {}
        for f in outputs:
            if f["name"] not in returned:
                return {"refuses": "interface-output", "field": f["name"]}
            values[f["name"]] = returned[f["name"]]
    for f in outputs:
        if not fits(values[f["name"]], f["shape"]):
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
                  ("order", {"anyOf": [S, NULL]}, {"optional": True})],
                 [("result", S, {})])
TRIAGE = interface("Sort a ticket and say how urgent it is.",
                   [("ticket", S, {})],
                   [("team", {"enum": ["billing", "shipping", "product"], "type": "string"}, {}),
                    ("minutes", I, {"desc": "to fix"}),
                    ("result", {"type": "object", "properties": {"reply": S, "escalate": B},
                                "required": ["reply", "escalate"]}, {})])
ANY = interface("A pipeline over a data frame.",
                [("frame", {}, {"type": "pd.DataFrame"}), ("args", {"type": "array", "default": []},
                                                            {"optional": True})],
                [("result", {}, {})])


def module_case(description, iface, inputs=(), returned=()):
    checks = [{"inputs": i, "expect": check_inputs(iface, i)} for i in inputs]
    checks += [{"returned": r, "expect": check_returned(iface, r)} for r in returned]
    return {"description": description, "program": "module", "interface": iface,
            "expect": {"id": interface_id(iface)}, "checks": checks}


def ai_case(description, key):
    d = functions.DEFINITIONS[key][1]
    iface = of_definition(d)
    return {"description": description, "program": "ai", "definition": d,
            "expect": {"interface": iface, "id": interface_id(iface),
                       "signature_id": functions.expect(d)["signature_id"]}}


def cases() -> dict:
    out = {
        "01-an-ai-function-is-its-definition": ai_case(
            "An AI function's interface is its definition's inputs and outputs; with neither reasoning nor "
            "tools its id is its program.signature.", "01-an-answer-from-a-list"),
        "02-several-outputs-keep-their-words": ai_case(
            "Outputs keep their desc in the interface (they are not in the signature's fields); the id "
            "leaves descriptions out, so it is still the signature's.", "02-several-outputs"),
        "03-reasoning-is-not-given-back": ai_case(
            "module cot adds a reasoning output to the signature, not to the interface: the ids differ.",
            "07-reasoning-first"),
        "04-tools-are-not-given": ai_case(
            "Tools add a tools input and a calls output to the signature, not to the interface.", "11-a-tool"),
        "05-a-module-checks-its-inputs": module_case(
            "A module's inputs are checked before its code runs: every one given, none it does not have, "
            "each fitting its shape.", SUPPORT,
            inputs=[{"message": "Where is my parcel?"}, {}, {"message": 3},
                    {"message": "Hi", "urgent": True}, {"message": None}],
            returned=["It left the depot today.", 42, None]),
        "06-optional-inputs": module_case(
            "An optional input left out takes its shape's default, or null without one; a given value "
            "is kept even when it is the default; desc and optional are not part of the id.", TONE,
            inputs=[{"message": "Hi"}, {"message": "Hi", "tone": "brief"}, {"message": "Hi", "order": "B-2210"},
                    {"message": "Hi", "tone": None}, {"tone": "brief"}]),
        "07-several-outputs": module_case(
            "A module with several outputs returns one record with each by name; the last is the answer; "
            "each output must be there and fit.", TRIAGE,
            returned=[{"team": "billing", "minutes": 5, "result": {"reply": "Refunded.", "escalate": False}},
                      {"team": "billing", "minutes": 5, "result": {"reply": "Refunded.", "escalate": False},
                       "note": "extra keys are not outputs"},
                      {"team": "legal", "minutes": 5, "result": {"reply": "?", "escalate": True}},
                      {"team": "billing", "result": {"reply": "Refunded.", "escalate": False}},
                      {"team": "billing", "minutes": 5.5, "result": {"reply": "Refunded.", "escalate": False}},
                      {"team": "billing", "minutes": 5.0, "result": {"reply": "Refunded.", "escalate": False}},
                      "billing"]),
        "08-no-types-no-refusals": module_case(
            "A shape of {} (no JSON type the language knows) accepts every value: such a module is never "
            "refused, and only its names are checked.", ANY,
            inputs=[{"frame": {"$type": "DataFrame", "$repr": "   a\n0  1"}}, {"frame": [1, 2], "args": [3]},
                    {"args": []}],
            returned=[{"$type": "DataFrame", "$repr": "   a\n0  1"}, None]),
    }
    same = [d for d in (out["01-an-ai-function-is-its-definition"], out["02-several-outputs-keep-their-words"])]
    assert all(c["expect"]["id"] == c["expect"]["signature_id"] for c in same)
    assert all(c["expect"]["id"] != c["expect"]["signature_id"]
               for c in (out["03-reasoning-is-not-given-back"], out["04-tools-are-not-given"]))
    return out
