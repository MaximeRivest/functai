"""Benchmark B: each library used well. Runs the hand-written solution of every task
in each library (tasks/<task>/reference/ for FunctAI, tasks/<task>/reference.dspy/
for DSPy) in a fresh workspace, through the recording proxy, and grades it like an
agent's work. Compares correctness, the requests and tokens the task sent, time,
and the solution's length.

    .venv/bin/python references.py                         # every task, both libraries, 3 times
    .venv/bin/python references.py --tasks forms --libraries dspy --reps 1    # check one grader

Real model calls (gpt-4.1-mini): a few cents per round. Results: runs/references-<stamp>/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import run

HERE = Path(__file__).resolve().parent
LIBS = {"functai": ("reference", "functai"), "dspy": ("reference.dspy", "dspy")}


def lines_of_code(path: Path) -> int:
    """Lines of code: not blank, not comments, not docstrings (a rough size of the solution)."""
    import ast
    source = path.read_text()
    skip = set()
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            skip.update(range(body[0].lineno, body[0].end_lineno + 1))
    return sum(1 for i, line in enumerate(source.splitlines(), 1)
               if i not in skip and line.strip() and not line.strip().startswith("#"))


def one(task: str, library: str, rep: int, root: Path) -> dict:
    folder, arm_name = LIBS[library]
    arm = run.ARMS[arm_name]
    solution = next((HERE / "tasks" / task / folder).glob("*.py"))
    base = root / task / f"{library}-{rep}"
    work = base / "work"
    run.workspace(task, arm, work)
    shutil.copy(solution, work / solution.name)
    recording = run.Recording(base)
    try:
        env = run.run_env(base, arm, recording)
        started = time.time()
        p = subprocess.run([str(HERE / arm["venv"] / "bin" / "python"), solution.name], cwd=work, env=env,
                           capture_output=True, text=True, timeout=1200)
        seconds = round(time.time() - started, 1)
        (base / "solution.log").write_text(p.stdout[-20000:] + "\n--- stderr\n" + p.stderr[-20000:])
        from common import record_summary
        sent = record_summary(base / "requests.jsonl")
        graded = run.grade(task, work, env, base / "grading", str(HERE / arm["venv"] / "bin" / "python"))
    finally:
        recording.close()
    return {"task": task, "library": library, "rep": rep, "exit": p.returncode, "seconds": seconds,
            "lines": lines_of_code(solution), "requests": sent, "grade": graded}


def render(rows: list[dict]) -> str:
    med = lambda xs: statistics.median(xs) if xs else None
    out = ["| task | library | runs | success | score | requests | input tokens | output tokens | seconds | lines of code |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for task in sorted({r["task"] for r in rows}):
        for lib in LIBS:
            rs = [r for r in rows if r["task"] == task and r["library"] == lib]
            if not rs:
                continue
            out.append(f"| {task} | {lib} | {len(rs)} | {sum(r['grade'].get('success', False) for r in rs)}/{len(rs)} | "
                       f"{med([r['grade'].get('score', 0) for r in rs]):.2f} | {med([r['requests']['requests'] for r in rs]):g} | "
                       f"{med([r['requests']['input_tokens'] for r in rs]):,.0f} | {med([r['requests']['output_tokens'] for r in rs]):,.0f} | "
                       f"{med([r['seconds'] for r in rs]):.1f} | {rs[0]['lines']} |")
    bad = [f"- {r['task']} / {r['library']} #{r['rep']}: " + "; ".join(f"{k} ({c['detail'] or c['value']})"
           for k, c in r["grade"].get("checks", {}).items() if not c["pass"]) + (f" {r['grade'].get('grader_error')}" if r["grade"].get("grader_error") else "")
           for r in rows if not r["grade"].get("success")]
    return "\n".join(out + (["", "Failed checks:", *bad] if bad else [])) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=run.TASKS, choices=run.TASKS)
    ap.add_argument("--libraries", nargs="*", default=list(LIBS), choices=list(LIBS))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = run.WORK_ROOT / "references" / stamp
    results = HERE / "runs" / f"references-{stamp}"
    results.mkdir(parents=True)
    rows = []
    for rep in range(1, args.reps + 1):
        for task in args.tasks:
            for lib in args.libraries:
                r = one(task, lib, rep, root)
                rows.append(r)
                print(f"{task:8} {lib:8} #{rep}  {'✓' if r['grade'].get('success') else '✗'} score {r['grade'].get('score')}"
                      f"  {r['requests']['requests']} requests  {r['requests']['input_tokens']:,}+{r['requests']['output_tokens']:,} tokens"
                      f"  {r['seconds']}s", flush=True)
    (results / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    (results / "report.md").write_text(render(rows))
    print(render(rows))


if __name__ == "__main__":
    main()
