"""The cases in files/: an AI function reading files (an input whose type
is ``Artifacts``, the result of Attachments' ``att()``), written from
../functions.md, *Pictures, sounds, documents and files*.

What the model sees is Attachments' ``parts()``, which is Attachments'
rule, pinned byte for byte by its own contract
(``attachments/contract/presenting.md``). This script asks Attachments'
Python package for that step only, as functions.py asks lmcc for the
layout, and writes everything FunctAI decides from the rules: the
binding, the empty value, the refusals. It never imports FunctAI.

A case file: {"description", "definition" (functions.md's, an input
carrying "type": "Artifacts"), "expect" (as functions/: "signature",
"sample", "request", "request_hash", "version", "signature_id"),
"sends": [{"inputs", "expect": {"request"} or {"refuses":
"interface-input", "field"}}], "manifest", "node"}. ``inputs`` hold the
artifacts in Attachments' wire form. ``manifest``: the function as a
saved folder another language wrote; loading its ``node`` gives the
same version, and calling it sends the same requests.

A language that does not read files passes each case by refusing the
function ``files-unavailable`` (when it is defined, when it is loaded,
or at the latest when its version or a call is rendered), and sending
nothing.
"""

import copy

import functions
import saved
from common import sha

TYPE = "Artifacts"
#: functions.md: the input's shape, Attachments' wire form, a list of artifacts.
SHAPE = {"type": "array", "items": {"type": "object"}}
EMPTY = [{"type": "text", "text": "(no content)"}]


def host_type():
    from attachments import Artifacts
    return Artifacts


def artifacts(value):
    """The value as Attachments' ``Artifacts`` (its wire form read back), or a TypeError naming why not."""
    from attachments import Artifacts
    if not isinstance(value, list):
        raise TypeError(f"not a list of artifacts: {type(value).__name__}")
    return Artifacts.from_wire(value)


def write(value) -> list:
    """What the model sees at the input's place (functions.md): Attachments' parts, or ``(no content)``."""
    return artifacts(value).parts() or copy.deepcopy(EMPTY)


def bind(registry) -> None:
    """The binding every implementation makes, in lmcc's registry for this script."""
    from attachments import Artifacts
    registry.format(Artifacts, shape=SHAPE, write=write, writes="parts", direction="in",
                    to_json=Artifacts.to_wire, from_json=Artifacts.from_wire)


def refusal(value):
    """functions.md: a value that is not the wire form, or holds a file that failed, is refused at binding."""
    try:
        a = artifacts(value)
    except (TypeError, ValueError, KeyError):
        return True
    return bool(a.errors)


# ------------------------------------------------------------------ the artifacts

PNG, PNG_2 = functions.PNG, functions.PNG_2
NOTES = "Revenue rose 12% in the third quarter.\nCosts were flat."
PAGE_1, PAGE_2 = "Q3 report\n\nRevenue by region: see the chart.", "Outlook\n\nWe expect growth to slow."
REPORT = PAGE_1 + "\n\n" + PAGE_2


def artifact(text, source, *, kind="text", images=(), segments=None, error=None):
    meta = {"source": source, "kind": kind}
    if segments is not None:
        meta["segments"] = segments
    if error is not None:
        meta["error"] = error
    return {"text": text, "images": list(images), "audio": [], "video": [], "meta": meta}


TEXT = artifact(NOTES, "notes.md")
PDF = artifact(REPORT, "q3.pdf", kind="pdf",
               images=[{"name": "q3-page-1.png", "mimetype": "image/png", "bytes_b64": PNG, "page": 1},
                       {"name": "q3-page-2.png", "mimetype": "image/png", "bytes_b64": PNG_2, "page": 2}],
               segments=[{"kind": "page", "label": "page 1", "start": 0, "end": len(PAGE_1), "page": 1},
                         {"kind": "page", "label": "page 2", "start": len(PAGE_1) + 2, "end": len(REPORT),
                          "page": 2}])
PHOTO = artifact("", "shelf.png", kind="image",
                 images=[{"name": "shelf.png", "mimetype": "image/png", "bytes_b64": PNG}])
FAILED = artifact("", "scan.pdf", kind="pdf",
                  error={"code": "missing-dependency", "message": "PDF reading needs pypdf: pip install 'attachments[pdf]'"})


# ------------------------------------------------------------------ the definitions

S = {"type": "string"}


def files_input(name, desc=None):
    return (name, SHAPE, desc, {"type": TYPE})


DEFINITIONS = {
    "01-a-report": (
        "An Artifacts input in the default layout (tags). Its sample is [] (an array), sent as one text part "
        "(no content). Sent: one text file; a folder of a text file and a two-page PDF, each page's picture "
        "after that page's text, the file names shown, as Attachments' parts() gives them; nothing at all. "
        "Refused at binding, before any request: a file att() could not read (alone, or among good ones), "
        "and values Attachments' from_wire refuses (not a list; text that is not text). A wire value with members "
        "left out is Attachments' to fill (its contract, cases/wire.json): no source is shown as (unknown).",
        functions.definition("summarize", "The report in three sentences.",
                             [files_input("report", "what to summarize"), ("audience", S, None)],
                             [("result", S, None)]),
        [{"report": [TEXT], "audience": "the board"},
         {"report": [TEXT, PDF], "audience": "the board"},
         {"report": [], "audience": "the board"},
         {"report": [FAILED], "audience": "the board"},
         {"report": [TEXT, FAILED], "audience": "the board"},
         {"report": "q3.pdf", "audience": "the board"},
         {"report": [{"text": 3}], "audience": "the board"},
         {"report": [{"text": "A note with no meta."}], "audience": "the board"}]),
    "02-a-picture-in-the-chat-layout": (
        "An Artifacts input holding a picture, in the chat layout ([[ ## name ## ]] sections): the picture "
        "is an image part inside the input's section, after the file's name.",
        functions.definition("count_items", "How many items are on the shelf?", [files_input("photo")],
                             [("result", {"type": "integer"}, None)], settings={"adapter": "Chat"}),
        [{"photo": [PHOTO]}]),
}


def node_for(d: dict) -> dict:
    """The saved node: as saved.py writes one, with the input's type kept in its interface and signature
    (functions.md: the type name of files is what tells them from any list of objects)."""
    n = saved.node(d["name"], d)
    for f in n["interface"]["inputs"]:
        src = next(x for x in d["inputs"] if x["name"] == f["name"])
        if src.get("type"):
            f["type"] = src["type"]
    for f in n["ai"]["signature"]["fields"]:
        src = next((x for x in d["inputs"] if x["name"] == f["name"] and f["direction"] == "input"), None)
        if src is not None and src.get("type"):
            f["type"] = src["type"]
    return n


def cases() -> dict:
    out = {}
    for name, (text, d, inputs) in DEFINITIONS.items():
        _, render = functions.probe(d)
        sends = []
        for x in inputs:
            bad = [k for k, v in x.items() if any(f["name"] == k and f.get("type") == TYPE for f in d["inputs"])
                   and refusal(v)]
            sends.append({"inputs": x, "expect": {"refuses": "interface-input", "field": bad[0]} if bad
                          else {"request": render(x)}})
        key = f"shop:{d['name']}"
        manifest = saved.manifest({key: node_for(d)}, key)
        e = functions.expect(d)
        assert saved.loaded_version(manifest["nodes"][key]) == e["version"], name
        out[name] = {"description": text, "definition": d, "expect": e, "sends": sends,
                     "manifest": manifest, "node": key}
    # the rules these cases show, checked as they are written
    first = out["01-a-report"]
    assert first["expect"]["sample"]["report"] == []
    parts = [p for m in first["expect"]["request"]["messages"] for p in m["parts"]]
    assert any(p.get("text", "").find("(no content)") >= 0 for p in parts)
    folder = first["sends"][1]["expect"]["request"]
    shown = [p["type"] for m in folder["messages"] for p in m["parts"]]
    assert shown.count("image") == 2 and PNG not in str([p.get("text") for m in folder["messages"] for p in m["parts"]])
    assert [s["expect"].get("refuses") for s in first["sends"]][3:] == ["interface-input"] * 4 + [None]
    assert sha(first["expect"]["request"]) == first["expect"]["request_hash"]
    return out
