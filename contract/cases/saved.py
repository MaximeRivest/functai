"""The cases in saved/: a manifest (functai.json) and what a loader in
another language must do with it, written from ../saved.md. The
fingerprints are the requests functions.py derives (the same rules).

A case file: {"description", "manifest", "node": key or null (the
entry), "expect": {"refuses": code} or {"loads": {"name", "module",
"version", "signature_id", "requests"}, "sends"?: [{"inputs",
"request_hash"}]}, and "describe": {"interface"} or {"refuses": code}}.
``describe`` is what describing the node without loading it gives
(saved.md, "Describing without loading"; ../programs.md): the node's
``interface`` (checked), or, for an AI node written before nodes had one,
the interface its signature gives. ``sends``: the loaded function called
with ``inputs`` (an optional one left out) sends the request whose hash
is ``request_hash``, rendered under the probe facts.
"""

import copy

import functions
import programs
import schemas
from common import no_defaults, sha

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


def node(name: str, d: dict, *, interface: bool = True, **kw) -> dict:
    """An AI node; ``interface``: written since 2026-09-28, with the node's interface."""
    n = {"kind": "ai", "module": "shop", "name": name}
    if interface:
        n["interface"] = programs.of_definition(d)
    code = {f["name"]: {"code": f["default_code"]} for f in d["inputs"] if "default_code" in f}
    if code:
        n["defaults"] = code                   # saved.md: each input's default that counts by its code
    n["ai"] = ai_node(d, **kw)
    return n


def loaded_version(n: dict) -> str:
    """The version a loaded AI function has (calls.md, Versions): its first probe's request, and each
    input's default by the node's defaults (its code) or else the interface's value."""
    version = {"request": n["ai"]["fingerprints"]["requests"][0]}
    defaults = {}
    for f in (n.get("interface") or {}).get("inputs", []):
        if f["name"] in (n.get("defaults") or {}):
            defaults[f["name"]] = n["defaults"][f["name"]]
        elif "default" in f["shape"]:
            defaults[f["name"]] = {"value": f["shape"]["default"]}
    if defaults:
        version["defaults"] = defaults
    if n["ai"].get("body"):
        return n["ai"]["version"]              # code of its own: the source hash is the saving language's
    return sha(version)


MOOD = functions.DEFINITIONS["01-an-answer-from-a-list"][1]
EXAMPLES = functions.DEFINITIONS["04-worked-examples"][1]
IMPROVED = functions.DEFINITIONS["08-an-improved-instruction"][1]
JSON = functions.DEFINITIONS["06-one-json-object"][1]


def plain_fields(n: dict) -> list:
    return [f for f in n["ai"]["signature"]["fields"] if (f.get("purpose") or "plain") == "plain"]


def differs(n: dict) -> bool:
    """An AI node's interface must describe the data its signature takes and gives."""
    fields = plain_fields(n)
    ids = programs.signature({"inputs": [f for f in fields if f["direction"] == "input"],
                              "outputs": [f for f in fields if f["direction"] == "output"]})
    return programs.signature(n["interface"]) != ids


def describe(m: dict, key) -> dict:
    """What describing a node without loading it gives (saved.md, programs.md)."""
    if m.get("functai_saved") != 1:
        return {"refuses": "saved-format"}
    if not schemas.SAVED.is_valid(m):
        return {"refuses": "saved-malformed"}
    n = m["nodes"][key or m["entry"]]
    if n["kind"] not in ("ai", "module"):
        return {"refuses": "saved-not-ai"}
    if "interface" in n:
        refused = programs.malformed(n["interface"], ai=n["kind"] == "ai")
        if refused:
            return {"refuses": refused["refuses"]}
        if n["kind"] == "ai" and differs(n):
            return {"refuses": "saved-differs"}
        return {"interface": n["interface"]}
    if n["kind"] == "module":
        return {"refuses": "saved-no-interface"}
    sig = n["ai"]["signature"]

    def field(f):
        return {"name": f["name"], "shape": f["shape"], **({"desc": f["desc"]} if f.get("desc") else {}),
                **({"type": f["type"]} if isinstance(f.get("type"), str) else {})}
    plain = plain_fields(n)
    return {"interface": {"description": sig["instructions"],
                          "inputs": [field(f) for f in plain if f["direction"] == "input"],
                          "outputs": [field(f) for f in plain if f["direction"] == "output"]}}


def loads(key: str, m: dict) -> dict:
    n = m["nodes"][key]
    if "interface" in n:
        if programs.malformed(n["interface"], ai=True):
            return {"refuses": "interface-malformed"}
        if differs(n):
            return {"refuses": "saved-differs"}
    identity = [{"direction": f["direction"], "name": f["name"], "purpose": f.get("purpose") or "plain",
                 "shape": no_defaults(f["shape"]), "type": ""} for f in n["ai"]["signature"]["fields"]]
    if "version" in n["ai"] and loaded_version(n) != n["ai"]["version"]:
        return {"refuses": "saved-differs"}
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

    m = manifest({"shop:mood": node("mood", MOOD, interface=False)}, "shop:mood")
    del m["nodes"]["shop:mood"]["ai"]["body"], m["nodes"]["shop:mood"]["ai"]["version"], m["language"]
    out["05-written-before-body"] = {
        "description": "A folder written before `body` existed: read as code, so it refuses. Written before "
                       "nodes had an interface too: describing it reads the interface from the signature, with "
                       "the instruction as its description (all an old folder has).",
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

    support = {"description": "Answer a customer's message.",
               "inputs": [{"name": "message", "shape": {"type": "string"}, "type": "str"},
                          {"name": "tone", "shape": {"type": "string", "default": "kind"}, "type": "str",
                           "optional": True}],
               "outputs": [{"name": "result", "shape": {"type": "string"}, "type": "str"}]}
    m = manifest({"shop:mood": node("mood", MOOD),
                  "shop:support": {"kind": "module", "module": "shop", "name": "support", "interface": support,
                                   "module_program": {"call_defaults": {}, "requires": []}}}, "shop:support")
    out["12-a-module-and-its-interface"] = {
        "description": "A module node written since 2026-09-28 has its interface: another language still "
                       "refuses to load it (it is code), but describes it without running anything.",
        "manifest": m, "node": None, "expect": {"refuses": "saved-not-ai"}}

    m = manifest({"shop:mood": node("mood", EXAMPLES)}, "shop:mood")
    m["nodes"]["shop:mood"]["interface"]["inputs"][0]["desc"] = "what the customer wrote"
    out["13-an-ai-function-and-its-interface"] = {
        "description": "An AI node written since 2026-09-28 has its interface: describing it gives the "
                       "function's description and its fields' words, not its instruction; loading it checks "
                       "that the interface's signature is its signature's plain fields' (words may differ).",
        "manifest": m, "node": None, "expect": loads("shop:mood", m)}

    m = manifest({"shop:mood": node("mood", MOOD)}, "shop:mood")
    m["nodes"]["shop:mood"]["interface"]["outputs"][0]["shape"] = {"type": "string"}
    out["14-an-interface-that-says-otherwise"] = {
        "description": "An AI node whose interface does not describe what its signature takes and gives is "
                       "refused, loading and describing: a server would otherwise accept or promise other data.",
        "manifest": m, "node": None, "expect": loads("shop:mood", m)}

    bad = {"description": "Two fields named alike.",
           "inputs": [{"name": "message", "shape": {"type": "string"}}],
           "outputs": [{"name": "message", "shape": {"type": "string"}}]}
    m = manifest({"shop:mood": node("mood", MOOD),
                  "shop:support": {"kind": "module", "module": "shop", "name": "support", "interface": bad,
                                   "module_program": {"call_defaults": {}, "requires": []}}}, "shop:support")
    out["15-a-malformed-interface"] = {
        "description": "An interface the schema accepts can still be refused (programs.md): describing the "
                       "node says interface-malformed.",
        "manifest": m, "node": None, "expect": {"refuses": "saved-not-ai"}}

    reply = functions.DEFINITIONS["12-an-optional-input"][1]
    m = manifest({"shop:reply": node("reply", reply)}, "shop:reply")
    out["16-an-optional-input"] = {
        "description": "An AI function with an input a caller may leave out: its interface says optional, with "
                       "the default in the shape; describing shows it; loading takes it from the interface, so "
                       "the loaded function called without the input sends the default, as the saving language "
                       "does (a folder without an interface has no optional input).",
        "manifest": m, "node": None,
        "expect": {**loads("shop:reply", m),
                   "sends": [{"inputs": i, "request_hash": functions.expect(with_sample(reply, {**b}))["request_hash"]}
                             for i, b in (({"message": "Hi"}, {"message": "Hi", "tone": "kind"}),
                                          ({"message": "Hi", "tone": "brief"}, {"message": "Hi", "tone": "brief"}))]}}

    plan = functions.DEFINITIONS["14-a-default-by-its-code"][1]
    m = manifest({"shop:plan": node("plan", plan)}, "shop:plan")
    out["17-a-default-by-its-code"] = {
        "description": "A node whose input's default is written as code (today()) keeps its text in the node's "
                       "defaults: the loaded function's version counts it by that text, and equals the saved "
                       "one; a default that counts by its value (tone) is the interface's.",
        "manifest": m, "node": None, "expect": loads("shop:plan", m)}
    assert out["17-a-default-by-its-code"]["expect"]["loads"]["version"] == functions.expect(plan)["version"]
    m = manifest({"shop:plan": node("plan", plan)}, "shop:plan")
    del m["nodes"]["shop:plan"]["defaults"]
    out["18-a-default-whose-code-was-not-kept"] = {
        "description": "The same node without its defaults: the loaded function would count today() by the "
                       "value it gave when saved, a version other than the saved one: it refuses.",
        "manifest": m, "node": None, "expect": loads("shop:plan", m)}
    assert out["18-a-default-whose-code-was-not-kept"]["expect"] == {"refuses": "saved-differs"}

    for case in out.values():
        case["expect"]["describe"] = describe(case["manifest"], case["node"])
    assert "refuses" not in out["01-a-function-the-model-writes-whole"]["expect"]["describe"]
    assert out["14-an-interface-that-says-otherwise"]["expect"]["describe"] == {"refuses": "saved-differs"}
    return out
