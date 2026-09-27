"""The cases in saved/: a manifest (functai.json) and what a loader in
another language must do with it, written from ../saved.md. The
fingerprints are the requests functions.py derives (the same rules).

A case file: {"description", "manifest", "node": key or null (the
entry), "expect": {"refuses": code} or {"loads": {"name", "module",
"version", "signature_id", "requests"}}}.
"""

import copy

import functions
from common import sha

LANG = "python"


def ai_node(d: dict, *, module: str = "shop", probes_from_demos: bool = True) -> dict:
    """The "ai" entry of a node, as ../saved.md says, for a definition."""
    e = functions.expect(d)
    sig = {"instructions": e["signature"]["instructions"],
           "fields": [{**f, "type": None, "desc": f.get("desc")} for f in e["signature"]["fields"]]}
    probes = [e["sample"]]
    requests = [e["request_hash"]]
    for demo in (d["state"].get("demos") or [])[:3] if probes_from_demos else []:
        if demo["inputs"] not in probes:
            probes.append(demo["inputs"])
            requests.append(functions.expect(with_sample(d, demo["inputs"]))["request_hash"])
    settings = {"lm": "gpt-4.1-mini", "module": d["settings"].get("module", "predict"),
                "include_fn_name_in_instructions": d["settings"].get("include_fn_name_in_instructions", True)}
    if d["settings"].get("adapter"):
        settings["adapter"] = d["settings"]["adapter"]
    return {"settings": settings, "config": {"temperature": 0}, "template": None, "tools": [], "teacher": None,
            "state": copy.deepcopy(d["state"]), "requires": [], "signature": sig, "probes": probes,
            "fingerprints": {"signature": sha([{"direction": f["direction"], "name": f["name"],
                                                "purpose": f.get("purpose", "plain"), "shape": f["shape"],
                                                "type": ""} for f in sig["fields"]]),
                             "requests": requests},
            "body": None, "version": e["version"]}


def with_sample(d: dict, inputs: dict) -> dict:
    """A definition whose sample input is ``inputs`` (to render a demo's inputs as a probe)."""
    d = copy.deepcopy(d)
    d["_sample"] = inputs
    return d


def manifest(nodes: dict, entry: str) -> dict:
    return {"functai_saved": 1, "language": LANG, "entry": entry, "created": "2026-09-27T12:00:00+00:00",
            "python": "3.13.1", "modules": {"shop": "code/shop.py"}, "nodes": nodes,
            "requirements": ["functai==1.1.0"], "allowed": [], "warnings": [], "models": {}, "data_files": {},
            "hashes": {}}


def node(name: str, d: dict, **kw) -> dict:
    return {"kind": "ai", "module": "shop", "name": name, "ai": ai_node(d, **kw)}


MOOD = functions.DEFINITIONS["01-an-answer-from-a-list"][1]
EXAMPLES = functions.DEFINITIONS["04-worked-examples"][1]
IMPROVED = functions.DEFINITIONS["08-an-improved-instruction"][1]
JSON = functions.DEFINITIONS["06-one-json-object"][1]


def loads(key: str, m: dict) -> dict:
    n = m["nodes"][key]
    identity = [{"direction": f["direction"], "name": f["name"], "purpose": f.get("purpose") or "plain",
                 "shape": f["shape"], "type": ""} for f in n["ai"]["signature"]["fields"]]
    return {"loads": {"name": n["name"], "module": n["module"], "version": n["ai"]["version"],
                      "signature_id": sha(identity),
                      "requests": n["ai"]["fingerprints"]["requests"]}}


def cases() -> dict:
    out = {}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    out["01-a-function-the-model-writes-whole"] = {
        "description": "An AI function with no code of its own loads, and sends what was saved.",
        "manifest": m, "node": None, "expect": loads("shop:mood", m)}

    m = manifest({"shop:mood": node("mood", EXAMPLES)}, "shop:mood")
    out["02-worked-examples-and-their-probes"] = {
        "description": "Demos load with it; each of the first three demos' inputs is a probe it must render "
                       "the same way.",
        "manifest": m, "node": None, "expect": loads("shop:mood", m)}

    m = manifest({"shop:capital": node("capital", IMPROVED), "shop:person": node("person", JSON)}, "shop:capital")
    out["03-a-node-by-key"] = {
        "description": "Any AI node loads by its key, not only the entry; an improved instruction and the json "
                       "layout come with it.",
        "manifest": m, "node": "shop:person", "expect": loads("shop:person", m)}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    m["nodes"]["shop:mood"]["ai"]["body"] = {"code": "sha256:" + "c" * 64}
    out["04-code-of-its-own"] = {
        "description": "Code runs beside the model (return round(_ai, 2)): only the saving language can run it.",
        "manifest": m, "node": None, "expect": {"refuses": "saved-code"}}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    del m["nodes"]["shop:mood"]["ai"]["body"], m["nodes"]["shop:mood"]["ai"]["version"], m["language"]
    out["05-written-before-body"] = {
        "description": "A folder written before `body` existed: read as code, so it refuses.",
        "manifest": m, "node": None, "expect": {"refuses": "saved-code"}}

    m = manifest({"shop:mood": node("mood", MOOD),
                  "shop:pipeline": {"kind": "module", "module": "shop", "name": "pipeline",
                                    "module_program": {"call_defaults": {}, "requires": []}}}, "shop:pipeline")
    out["06-a-module"] = {
        "description": "A module is code: it refuses. Its AI functions still load by key.",
        "manifest": m, "node": None, "expect": {"refuses": "saved-not-ai"}}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    m["nodes"]["shop:mood"]["ai"]["tools"] = ["shop:lookup_order"]
    out["07-tools"] = {"description": "A tool is code: it refuses.",
                       "manifest": m, "node": None, "expect": {"refuses": "saved-tools"}}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    m["nodes"]["shop:mood"]["ai"]["settings"]["lm"] = {"baked": "mood-small"}
    out["08-a-baked-model"] = {"description": "A baked model is not a model string: it refuses.",
                               "manifest": m, "node": None, "expect": {"refuses": "saved-model"}}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    m["nodes"]["shop:mood"]["ai"]["fingerprints"]["requests"][0] = "sha256:" + "0" * 64
    out["09-sends-something-else"] = {
        "description": "What it would send differs from what was saved: it refuses rather than run another "
                       "program under this one's name.",
        "manifest": m, "node": None, "expect": {"refuses": "saved-differs"}}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    m["functai_saved"] = 2
    out["10-a-later-format"] = {"description": "A format this loader does not know: it refuses.",
                                "manifest": m, "node": None, "expect": {"refuses": "saved-format"}}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    m["nodes"]["shop:mood"]["ai"]["version"] = "sha256:" + "1" * 64
    out["11-another-version"] = {
        "description": "The saved version is not the one its request gives: it refuses.",
        "manifest": m, "node": None, "expect": {"refuses": "saved-differs"}}
    return out
