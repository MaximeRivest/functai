"""Writes every case in this folder from the rules, never from an
implementation's output:

    rated/       ../calls.md, "Rows with known answers"      (rated.py)
    functions/   ../functions.md and ../calls.md, "Versions"  (functions.py)
    files/       ../functions.md, "Pictures, sounds, documents and files"  (files.py)
    scores/      ../scores.md                                (scores.py)
    saved/       ../saved.md                                 (saved.py)
    programs/    ../programs.md                              (programs.py)
    content/     ../calls.md, "Content"                      (content.py)
    saw/         ../calls.md, "Saw"                          (saw.py)
    events/      ../streaming.md                             (events.py)
    replies/, usage/, conversations/, tools/, views/, context/
                 ../replies.md, ../calls.md "Token counts", ../conversations.md, ../tools.md,
                 ../streaming.md "Views", ../calls.md "Rows that keep their context"   (stages.py)
    plugins/     ../plugins.md, "Order"                      (plugins.py)
    baked/       ../baked.md, "The examples"                 (baked.py)
    agents/      ../tools.md "What a tool returns", "A result that comes later"; ../streaming.md "The kept
                 form"; ../plugins.md "The step hook", "limits", "keep_pictures"   (agents.py)
    measures/    ../measures.md: case keys, splits, draws, ratings of format 2, values, the statistics,
                 scoreboards   (measures.py, from the rules in measure_rules.py)

Every record, event, manifest and interface a case holds is checked
against ../schema as it is written, and the schemas are checked to refuse
what must never be written (schemas.py).

    python contract/cases/make.py      (python/.venv has lmcc for functions/)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agents  # noqa: E402
import baked  # noqa: E402
import content  # noqa: E402
import events  # noqa: E402
import files  # noqa: E402
import partial  # noqa: E402
import setup  # noqa: E402
import functions  # noqa: E402
import measures  # noqa: E402
import plugins  # noqa: E402
import programs  # noqa: E402
import rated  # noqa: E402
import saved  # noqa: E402
import saw  # noqa: E402
import schemas  # noqa: E402
import scores  # noqa: E402
import stages  # noqa: E402


KNOWN = {"functai_call": (1, 2), "functai_rating": (1, 2)}


def check(folder: str, cases: dict) -> None:
    """Each case's data passes the contract's schemas."""
    for name, case in cases.items():
        where = f"{folder}/{name}"
        if folder == "measures":
            for i, rec in enumerate(measures.records_of(case)):
                if any(k in rec and rec[k] not in v for k, v in KNOWN.items()):   # a format no reader knows
                    assert not schemas.CALL.is_valid(rec) and not schemas.RATING.is_valid(rec), where
                    continue
                schemas.record(rec, f"{where} records[{i}]")
        elif folder == "rated" or (folder == "saw" and case["kind"] == "read"):
            for i, rec in enumerate(case["records"]):
                if any(k in rec and rec[k] not in v for k, v in KNOWN.items()):   # a format no reader knows
                    assert not schemas.CALL.is_valid(rec) and not schemas.RATING.is_valid(rec), where
                    continue
                schemas.record(rec, f"{where} records[{i}]")
        elif folder == "saw":
            schemas.check(schemas.SAW_ENTRY, case["entry"], f"{where} entry")
        elif folder == "content":
            schemas.check(schemas.CALL, case["record"], f"{where} record")
            if "record" in case["expect"]:
                schemas.check(schemas.CALL, case["expect"]["record"], f"{where} expect.record")
        elif folder == "files":
            schemas.check(schemas.SAVED, case["manifest"], f"{where} manifest")
        elif folder == "saved":
            if case["expect"].get("refuses") == "saved-format":        # a format this schema does not know
                assert not schemas.SAVED.is_valid(case["manifest"]), f"{where}: the schema accepts it"
            else:
                schemas.check(schemas.SAVED, case["manifest"], f"{where} manifest")
        elif folder == "programs":
            if case["program"] == "ai":
                ifaces = [case["expect"]["interface"]] if "interface" in case["expect"] else []
            elif case["program"] == "definitions":
                ifaces = [x["interface"] for x in case["interfaces"] if "signature" in x["expect"]]
            elif case["program"] == "same-data":
                ifaces = case["interfaces"]
            elif case["program"] == "notes":
                ifaces = case["chain"]
            else:
                ifaces = [case["interface"]]
            for iface in ifaces:
                schemas.check(schemas.INTERFACE, iface, f"{where} interface")
        elif folder == "conversations" or (folder == "agents" and case["kind"] == "state"):
            for i, rec in enumerate(case["records"]):
                schemas.check(schemas.CONVERSATION, rec, f"{where} records[{i}]")
        elif folder == "views":
            for i, e in enumerate(case["events"] + case["expect"]["events"]):
                schemas.check(schemas.EVENT, e, f"{where} event {i}")
        elif folder == "context":
            for i, rec in enumerate(case["records"]):
                schemas.record(rec, f"{where} records[{i}]")
        elif folder == "agents" and case["kind"] == "kept":
            for i, e in enumerate(case["events"] + case["expect"]["events"]):
                schemas.check(schemas.EVENT, e, f"{where} event {i}")
        elif folder == "events":
            kind = case["kind"]
            if kind == "receivers":
                continue
            evs = ([e for x in case["steps"] for e in ([x["append"]] if "append" in x else x.get("batch", []))]
                   if kind == "store" else
                   list(case["received"]) if kind == "follow" else list(case["events"]))   # a copy: never the case's
            if kind == "kept":
                evs += case["expect"]["events"]
            if kind == "journal":
                evs += case["expect"]["log"]
            if kind == "follow" and "recover" in case:
                evs += case["recover"]["source"]
            malformed = [x["append"] for x in case.get("steps", []) if x.get("expect") == "event-malformed"]
            for i, e in enumerate(evs):
                if any(e is m for m in malformed):
                    continue                                    # seq not above after: the schema cannot say
                if e.get("functai_event") not in (None, 2):     # a format no reader knows
                    assert not schemas.EVENT.is_valid(e), where
                    continue
                schemas.check(schemas.EVENT, e, f"{where} event {i}")


def write(folder: str, cases: dict) -> int:
    check(folder, cases)
    out = HERE / folder
    out.mkdir(exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for name, case in cases.items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(case, indent=1, ensure_ascii=False) + "\n")
        written = json.loads(path.read_text())          # what a harness reads, run through the rules again
        if folder == "events":
            assert events.rerun(written) == events.as_written(written), f"{folder}/{name}: its data gives another result"
        elif folder == "programs":
            programs.rerun(written)
        elif folder == "partial":
            for c in written["checks"]:
                assert partial.so_far(written["shape"], c["text"]) == c["expect"], f"{folder}/{name}: {c}"
        elif folder == "setup":
            doc = setup.document(written["definition"], written["settings"], written["plugins"])
            assert {"document": doc, "setup": setup.sha(doc)} == written["expect"], f"{folder}/{name}"
    return len(cases)


if __name__ == "__main__":
    schemas.refusals()
    n = write("rated", rated.CASES) + write("functions", functions.cases()) + write("files", files.cases()) + write("scores", scores.cases()) \
        + write("saved", saved.cases()) + write("programs", programs.cases()) + write("content", content.cases()) \
        + write("saw", saw.cases()) + write("events", events.cases())
    for folder, made in stages.cases().items():
        n += write(folder, made)
    n += write("plugins", plugins.cases())
    n += write("baked", baked.cases())
    n += write("agents", agents.cases())
    n += write("partial", partial.cases())
    n += write("setup", setup.cases())
    n += write("measures", measures.cases())
    print(f"{n} cases written")
