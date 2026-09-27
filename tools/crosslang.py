"""Python, TypeScript and R against each other, offline (a fake model in each).

    python/.venv/bin/python tools/crosslang.py

The contract's cases check each language alone. This checks the promises
between them, on real output of both:

1. Functions Python saves load in TypeScript and in R: the same version,
   and the same request, byte for byte, for the same input. One with code
   of its own is refused.
2. The same function written natively in TypeScript and in R has Python's
   version and signature.
3. One call log, three writers: each language logs calls of the same
   function and rates one into one folder. Every record passes the
   contract's schemas, and every language's `rated` gives the same rows,
   including the others' calls and ratings.

R runs with `r/.lib` (r/check installs it) and Rscript on PATH.

`../check` runs it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "tests"))

import jsonschema  # noqa: E402
from conftest import FakeRouter  # noqa: E402

import functai  # noqa: E402
from functai import calllog  # noqa: E402

SHOP = '''
from typing import Literal
import dataclasses
from functai import ai, _ai

@ai(lm="gpt-4.1-mini", temperature=0)
def mood(review: str) -> Literal["happy", "unhappy", "mixed"]:
    """How does the customer feel about what they bought?"""

@ai(lm="gpt-4.1-mini")
def mood_taught(review: str) -> Literal["happy", "unhappy", "mixed"]:
    """How does the customer feel about what they bought?"""

mood_taught.demos = [{"review": "Broke in a day.", "result": "unhappy"}, {"review": "Love it!", "result": "happy"}]
mood_taught.instructions = "Say how the customer feels, in one word."

@dataclasses.dataclass
class Person:
    name: str
    age: int

@ai(lm="gpt-4.1-mini", adapter="json")
def person(text: str) -> Person:
    """Who is described?"""

@ai(lm="claude-haiku-4-5", module="cot")
def solve(problem: str,  # a word problem, in English
          ) -> float:
    """Solve the word problem."""

@ai(lm="gpt-4.1-mini")
def rounded(x: float) -> float:
    """Double it."""
    return round(_ai, 2)
'''

NAMES = ["mood", "mood_taught", "person", "solve", "rounded"]
SAMPLE = {"mood": {"review": "Broke."}, "mood_taught": {"review": "Broke."}, "person": {"text": "Ana, 31."},
          "solve": {"problem": "Two plus two?"}, "rounded": {"x": 1.5}}


def schemas():
    """Validators for the contract's schemas (the rating schema refers to the call schema)."""
    from referencing import Registry, Resource
    contract = ROOT / "contract" / "schema"
    docs = {name: json.loads((contract / f"{name}.schema.json").read_text()) for name in ("call", "rating", "saved")}
    registry = Registry().with_resources([(d["$id"], Resource.from_contents(d)) for d in docs.values()])
    return {name: jsonschema.Draft202012Validator(d, registry=registry) for name, d in docs.items()}


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="functai-crosslang-"))
    (work / "shop.py").write_text(textwrap.dedent(SHOP))
    sys.path.insert(0, str(work))
    import shop
    log = work / "log"
    os.environ.pop("FUNCTAI_LOG_CALLS", None)

    # 1. Python saves; what TypeScript must reproduce
    from lm15.serde import request_to_dict
    expected = {}
    (work / "saved").mkdir()
    for name in NAMES:
        fn = getattr(shop, name)
        functai.save(fn, work / "saved" / name)
        schemas()["saved"].validate(json.loads((work / "saved" / name / "functai.json").read_text()))
        expected[name] = {"version": fn.version, "signature": calllog.signature_id(fn.signature),
                          "request": request_to_dict(fn.render(**SAMPLE[name])), "inputs": SAMPLE[name]}
    (work / "python.json").write_text(json.dumps(expected))

    # 3a. Python logs two calls of mood and rates one
    router = FakeRouter(responder=lambda req: "<result>\nunhappy\n</result>")
    with functai.configure(lm="gpt-4.1-mini", client=router, log_calls=str(log)):
        p1 = shop.mood("I was charged twice.", all=True)
        shop.mood("Late, but fine.", all=True)
        functai.rate(p1, "right", by="ana")

    # 2, 1 and 3b in TypeScript
    node = subprocess.run(
        ["node", "--conditions=functai-source", "--conditions=lmcc-source", "tools/crosslang.ts", str(work)],
        cwd=ROOT / "ts", capture_output=True, text=True)
    print(node.stdout, end="")
    if node.returncode != 0:
        print(node.stderr, file=sys.stderr)
        return 1

    # 1, 2 and 3b in R
    r = subprocess.run(["Rscript", "r/tools/crosslang.R", str(work)], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "R_LIBS": str(ROOT / "r" / ".lib")})
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 1

    # 3c. every record passes the schemas; Python's rated sees TypeScript's and R's calls and ratings
    s = schemas()
    calls_, ratings = calllog.read(log)
    languages = {c["process"]["language"] for c in calls_}
    assert languages == {"python", "typescript", "r"}, languages
    for c in calls_:
        s["call"].validate(c)
    for r in ratings:
        s["rating"].validate(r)
    rows, _left = calllog.rated_rows(calls_, ratings, name="mood", module="shop",
                                     signature=calllog.signature_id(shop.mood.signature))
    with functai.configure(log_calls=str(log)):
        assert len(functai.rated(shop.mood).collect().to_dicts()) == len(rows)
    ts_rows = json.loads((work / "typescript-rated.json").read_text())
    r_rows = json.loads((work / "r-rated.json").read_text())
    assert len(rows) == 3, rows                                   # one rated in each language
    assert {r["rated_by"] for r in rows} == {"ana", "ben", "cleo"}
    assert rows == r_rows, (rows, r_rows)
    assert rows[:2] == ts_rows, (rows, ts_rows)                   # TypeScript read the log before R wrote to it
    print(f"  ok    one log, three languages: {len(calls_)} calls and {len(ratings)} ratings pass the schemas; "
          f"Python's, TypeScript's and R's rated give the same rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
