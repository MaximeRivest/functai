"""Grade the `export` task: training conversations for a small model.

The right file holds, for every row, exactly the chat the library will send the
trained model (the style guide left out, since the student learns it) with the
row's answer as the reply. The grader rebuilds that with the library's public
API and compares message for message: FunctAI's `functai.bake.examples(...,
fixed=...)`; DSPy's `ChatAdapter().format_finetune_data(...)` for
`Simplify.delete("style_guide")`, the call the DSPy task names.

Checks
- files:    train.jsonl, one JSON object per line with a `messages` list of {role, content}
- rows:     one conversation per row of rows.csv (24)
- exact:    share of the reference conversations found, identical
- no_guide: the style guide's text is in no message
- free:     no model call was made (every row already has its answer)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import subprocess

from common import check, recorded, summarize

HERE = Path(__file__).parent


def reference(work: Path, env: dict) -> list[list[tuple[str, str]]]:
    if env.get("BENCH_LIBRARY") == "dspy":      # DSPy's own training format, computed in its environment
        out = subprocess.run([str(HERE.parent.parent / ".venv-dspy" / "bin" / "python"), str(HERE / "dspy_reference.py")],
                             capture_output=True, text=True, check=True).stdout
        return [[(m["role"], m["content"]) for m in conv] for conv in json.loads(out)]
    return functai_reference()


def functai_reference() -> list[list[tuple[str, str]]]:
    import functai.bake
    from dpyr import read
    spec = importlib.util.spec_from_file_location("simplify_ref", HERE / "files" / "simplify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    guide = (HERE / "files" / "style_guide.md").read_text()
    ex = functai.bake.examples(module.simplify, read(str(HERE / "files" / "rows.csv")), fixed={"style_guide": guide},
                               log=False)
    return [[(m["role"], m["content"]) for m in r["messages"]] for r in ex.rows]


def grade(work: Path, env: dict, python: str, out: Path) -> dict:
    c = {}
    path = work / "train.jsonl"
    convs = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                convs.append([(m["role"], m["content"]) for m in json.loads(line)["messages"]])
        ok = bool(convs)
    except Exception as e:  # noqa: BLE001
        ok, c_detail = False, f"{type(e).__name__}: {e}"
    c["files"] = check(ok, detail="train.jsonl with a messages list per line" if ok else c_detail)
    if not ok:
        for k in ("rows", "exact", "no_guide", "free"):
            c[k] = check(False, 0.0, "no usable output")
        return summarize(c, required=list(c))

    ref = reference(work, env)
    c["rows"] = check(len(convs) == len(ref), None, f"{len(convs)} conversations for {len(ref)} rows")
    have = {tuple(x) for x in convs}
    exact = sum(tuple(r) in have for r in ref) / len(ref)
    detail = "all identical"
    if exact < 1 and convs:
        r0 = ref[0]
        got = min(convs, key=lambda x: sum(a != b for a, b in zip(x, r0)) + abs(len(x) - len(r0)))
        diff = [f"{a[0]}: expected {a[1][:160]!r} got {b[1][:160]!r}" for a, b in zip(r0, got) if a != b]
        detail = "first difference: " + (diff[0] if diff else f"{len(got)} messages instead of {len(r0)}")
    c["exact"] = check(exact >= 0.95, round(exact, 2), detail)
    guide_lines = [l.strip("- ").strip() for l in (HERE / "files" / "style_guide.md").read_text().splitlines()
                   if l.startswith("- ")]
    leaked = sum(any(g[:30] in m for _, m in conv for g in guide_lines) for conv in convs)
    c["no_guide"] = check(leaked == 0, None, f"style guide text in {leaked} conversations")
    sent = recorded(env.get("BENCH_RECORD"))
    c["free"] = check(not sent, None, f"{len(sent)} requests reached the provider")
    return summarize(c, required=list(c))

