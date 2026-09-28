"""Writes every case in this folder from the rules, never from an
implementation's output:

    rated/       ../calls.md, "Rows with known answers"      (rated.py)
    functions/   ../functions.md and ../calls.md, "Versions"  (functions.py)
    scores/      ../scores.md                                (scores.py)
    saved/       ../saved.md                                 (saved.py)
    programs/    ../programs.md                              (programs.py)
    content/     ../calls.md, "Content"                      (content.py)
    saw/         ../calls.md, "Saw"                          (saw.py)
    events/      ../streaming.md                             (events.py)

Every record, event, manifest and interface a case holds is checked
against ../schema as it is written (schemas.py).

    python contract/cases/make.py      (python/.venv has lmcc for functions/)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import content  # noqa: E402
import events  # noqa: E402
import functions  # noqa: E402
import programs  # noqa: E402
import rated  # noqa: E402
import saved  # noqa: E402
import saw  # noqa: E402
import schemas  # noqa: E402
import scores  # noqa: E402


def check(folder: str, cases: dict) -> None:
    """Each case's data passes the contract's schemas."""
    for name, case in cases.items():
        where = f"{folder}/{name}"
        if folder in ("rated", "saw"):
            for i, rec in enumerate(case["records"]):
                schemas.record(rec, f"{where} records[{i}]")
        elif folder == "content":
            schemas.check(schemas.CALL, case["record"], f"{where} record")
            if "record" in case["expect"]:
                schemas.check(schemas.CALL, case["expect"]["record"], f"{where} expect.record")
        elif folder == "saved":
            if case["expect"].get("refuses") == "saved-format":        # a format this schema does not know
                assert not schemas.SAVED.is_valid(case["manifest"]), f"{where}: the schema accepts it"
            else:
                schemas.check(schemas.SAVED, case["manifest"], f"{where} manifest")
        elif folder == "programs":
            iface = case.get("interface") or case["expect"]["interface"]
            schemas.check(schemas.INTERFACE, iface, f"{where} interface")
        elif folder == "events":
            evs = [a["event"] for a in case["appends"]] if case["kind"] == "store" else list(case["events"])
            if case["kind"] == "stored":
                evs += case["expect"]["events"]
            for i, e in enumerate(evs):
                schemas.check(schemas.EVENT, e, f"{where} event {i}")


def write(folder: str, cases: dict) -> int:
    check(folder, cases)
    out = HERE / folder
    out.mkdir(exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for name, case in cases.items():
        (out / f"{name}.json").write_text(json.dumps(case, indent=1, ensure_ascii=False) + "\n")
    return len(cases)


if __name__ == "__main__":
    n = write("rated", rated.CASES) + write("functions", functions.cases()) + write("scores", scores.cases()) \
        + write("saved", saved.cases()) + write("programs", programs.cases()) + write("content", content.cases()) \
        + write("saw", saw.cases()) + write("events", events.cases())
    print(f"{n} cases written")
