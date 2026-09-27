"""Writes every case in this folder from the rules, never from an
implementation's output:

    rated/       ../calls.md, "Rows with known answers"      (rated.py)
    functions/   ../functions.md and ../calls.md, "Versions"  (functions.py)
    scores/      ../scores.md                                (scores.py)
    saved/       ../saved.md                                 (saved.py)

    python contract/cases/make.py      (python/.venv has lmcc for functions/)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import functions  # noqa: E402
import rated  # noqa: E402
import saved  # noqa: E402
import scores  # noqa: E402


def write(folder: str, cases: dict) -> int:
    out = HERE / folder
    out.mkdir(exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for name, case in cases.items():
        (out / f"{name}.json").write_text(json.dumps(case, indent=1, ensure_ascii=False) + "\n")
    return len(cases)


if __name__ == "__main__":
    n = write("rated", rated.CASES) + write("functions", functions.cases()) + write("scores", scores.cases()) \
        + write("saved", saved.cases())
    print(f"{n} cases written")
