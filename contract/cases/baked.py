"""The cases in baked/: what a generative student is trained on and called
with, from ../baked.md, "The examples".

For a definition (as in functions/), the bake's options (``layout``,
``fixed``, ``derived``, ``reasoning``) and rows, a case gives the
student's signature (the function's, without the inputs left out), the
hashes kept for fixed and derived inputs, and each row's training
conversation: the chat messages of the call, the reply last. Turning a
signature, a layout and values into a request is lmcc's rule (borrowed,
as functions.py borrows it); everything else is written here from the
rules. It never imports FunctAI.

A case file: {"description", "definition", "bake": {"fixed", "derived",
"reasoning"}, "rows": [{"inputs", "outputs"}], "expect": {"signature",
"fixed", "derived", "examples": [{"messages"}]}}.
"""

import copy
import json

import lmcc

from common import sha
from functions import LAYOUTS, S, as_text, definition, fields, instructions, registry

CAPABILITIES = {"instruct": True}       # what a student declares: no native transports


def student_layout(d: dict, reg) -> lmcc.Adapter:
    name = (d["settings"].get("adapter") or "xml").lower()
    name = {"default": "xml", "tags": "xml"}.get(name, name)
    return lmcc.load({**copy.deepcopy(LAYOUTS[name]), "replay": "values"}, registry=reg)


def chat(request: dict) -> list:
    """An lm15 request as chat messages: the system text first, a developer
    message as a system one, each message's text parts joined."""
    out = []
    if request.get("system"):
        sys_ = request["system"]
        out.append({"role": "system", "content": sys_ if isinstance(sys_, str) else
                    "".join(p["text"] for p in sys_)})
    for m in request["messages"]:
        out.append({"role": "system" if m["role"] == "developer" else m["role"],
                    "content": "".join(p["text"] for p in m["parts"])})
    return out


def expect(d: dict, bake: dict, rows: list) -> dict:
    reg = registry()
    fixed, derived = bake.get("fixed") or {}, bake.get("derived") or {}
    left_out = set(fixed) | set(derived)
    all_fields = fields(d)
    kept = [f for f in all_fields if not (f["direction"] == "input" and f["name"] in left_out)]
    if not bake.get("reasoning"):
        kept = [f for f in kept if f.get("purpose") != "reasoning"]
    sig_dict = {"instructions": instructions(d), "fields": kept}
    signature = lmcc.signature_from_dict(copy.deepcopy(sig_dict))
    plan = student_layout(d, reg).bind(signature, dict(CAPABILITIES), registry=reg)
    text_in = {f["name"] for f in d["inputs"] if f["shape"].get("type") == "string" and "enum" not in f["shape"]}

    def prepared(name, v):
        return as_text(v) if name in text_in and v is not None else v
    table = {}
    for name, src in derived.items():
        table[name] = {"from": src, "values": {sha(prepared(src, r["inputs"][src])): sha(prepared(name,
                                                                                           r["inputs"][name]))
                                               for r in rows}}
    examples = []
    for r in rows:
        values = {k: prepared(k, v) for k, v in r["inputs"].items() if k not in left_out}
        prompt = chat(json.loads(json.dumps(plan.render(plan.turn(values)).request("student"))))
        example = plan.example(values, r["outputs"])
        both = chat(json.loads(json.dumps(plan.render(plan.turn(values), turns=[example]).request("student"))))
        reply = [m for m in both if m["role"] == "assistant"][-1]
        examples.append({"messages": prompt + [reply]})
    return {"signature": sig_dict, "fixed": {k: sha(prepared(k, v)) for k, v in fixed.items()},
            "derived": table, "examples": examples}


GUIDE = "Write like a friendly librarian: short sentences, no jargon, never more than three of them."

CASES = {
    "01-a-reply-to-learn": (
        "The student reads the call as the layout writes it, with no worked examples, and learns the "
        "layout's reply for the row's answer (written from values).",
        definition("capital", "The country's capital city.", [("country", S, None)], [("result", S, None)]),
        {}, [{"inputs": {"country": "Peru"}, "outputs": {"result": "Lima"}},
             {"inputs": {"country": "Kenya"}, "outputs": {"result": "Nairobi"}}]),
    "02-a-fixed-input": (
        "A fixed input is left out of the student's signature and of every message; its value's hash is kept.",
        definition("reply", "Answer the customer.", [("message", S, None), ("guide", S, None)],
                   [("result", S, None)]),
        {"fixed": {"guide": GUIDE}},
        [{"inputs": {"message": "Where is my order?", "guide": GUIDE}, "outputs": {"result": "On its way."}}]),
    "03-a-derived-input": (
        "A derived input is left out too; the table maps each source value's hash to its value's hash.",
        definition("rewrite", "Rewrite the section.",
                   [("section", S, None), ("text", S, None), ("guidance", S, None)], [("result", S, None)]),
        {"derived": {"guidance": "section"}},
        [{"inputs": {"section": "methods", "text": "We did X.", "guidance": "Keep every number."},
          "outputs": {"result": "We did X, carefully."}},
         {"inputs": {"section": "results", "text": "Y rose.", "guidance": "Keep every table."},
          "outputs": {"result": "Y went up."}}]),
    "04-two-outputs-in-sections": (
        "The chat layout (DSPy's sections), two outputs in one reply.",
        definition("triage", "Read the ticket.", [("ticket", S, None)],
                   [("summary", S, "one sentence"), ("result", {"type": "integer"}, "minutes to fix")],
                   settings={"adapter": "Chat"}),
        {}, [{"inputs": {"ticket": "The login page is down."},
              "outputs": {"summary": "Login is down.", "result": 30}}]),
    "05-reasoning-kept": (
        "With reasoning kept (module cot, bake reasoning=True), the reasoning output is written in the "
        "reply the student learns.",
        definition("solve", "Solve the word problem.", [("problem", S, None)], [("result", {"type": "number"}, None)],
                   settings={"module": "cot"}),
        {"reasoning": True}, [{"inputs": {"problem": "Two apples and three more?"},
                               "outputs": {"reasoning": "2 + 3 = 5.", "result": 5}}]),
}


def cases() -> dict:
    return {name: {"description": text, "definition": d, "bake": bake, "rows": rows, "expect": expect(d, bake, rows)}
            for name, (text, d, bake, rows) in CASES.items()}
