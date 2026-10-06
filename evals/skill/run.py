"""Run coding agents on the tasks, with and without the functai skill, and grade them.

    .venv/bin/python run.py                                   # every task, both conditions, once
    .venv/bin/python run.py --tasks forms photos --reps 3 --parallel 4
    .venv/bin/python run.py --model openai-codex/gpt-6-luna --thinking low
    .venv/bin/python report.py runs/<stamp>                   # the comparison table (run.py prints it too)
    .venv/bin/python run.py --collect runs/<stamp>            # grade an earlier run again from its workspaces

Each run gets a fresh workspace OUTSIDE this repository (~/.cache/functai-skill-eval/,
or $FUNCTAI_SKILL_EVAL_WORK) holding the task's files and a `.venv` link to this
folder's environment. Pi runs headless with nothing personal loaded: no
extensions, skills, AGENTS.md or prompt templates; the `with` condition adds
only a copy of skills/functai. Both get the same task text and the same rule:
do not open this evaluation or the skill in the repository (a run that does is
marked contaminated). The functai disk cache and call log are per run, so no
run reuses another's paid replies.

What is recorded per run (runs/<stamp>/<task>/<condition>-<rep>/result.json):
the grade (checks, score, success), the agent's wall time, tokens and
list-price cost (from Pi's session), its tool calls and failed tool calls,
whether it read the skill, and the functai model calls the task made.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SKILL = REPO / "skills" / "functai"
WORK_ROOT = Path(os.environ.get("FUNCTAI_SKILL_EVAL_WORK", Path.home() / ".cache" / "functai-skill-eval"))
KEYS_FILE = Path(os.environ.get("FUNCTAI_SKILL_EVAL_KEYS", Path.home() / "Projects" / "lm15-dev" / ".env"))
TASKS = sorted(p.name for p in (HERE / "tasks").iterdir() if (p / "task.md").exists())

RULES = (
    "You are working in the current folder for a user who will read your final message. "
    "Do not open, list or search anything under ~/Projects/functai/evals or ~/Projects/functai/skills. "
    "Finish the task completely, then reply with a short summary."
)
FORBIDDEN = re.compile(r"functai/(evals|skills)\b")


def keys() -> dict:
    """Provider keys from a dotenv-style file (`export NAME=value`), never printed."""
    out = {}
    if KEYS_FILE.exists():
        for line in KEYS_FILE.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Z0-9_]+)=(.*)$", line)
            if m:
                out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def session_metrics(session_dir: Path) -> dict:
    files = sorted(session_dir.glob("*.jsonl"))
    m = dict(cost=0.0, input_tokens=0, output_tokens=0, cache_read_tokens=0, turns=0, tool_calls=0,
             tool_errors=0, read_skill=False, contaminated=False, first=None, last=None, final="")
    for f in files:
        for line in f.read_text().splitlines():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = o.get("timestamp")
            if ts and o.get("type") == "message":
                m["first"] = m["first"] or ts
                m["last"] = ts
            if o.get("type") != "message":
                continue
            msg = o["message"]
            if msg.get("role") == "assistant":
                m["turns"] += 1
                u = msg.get("usage") or {}
                m["cost"] += (u.get("cost") or {}).get("total", 0) or 0
                m["input_tokens"] += u.get("input", 0) or 0
                m["output_tokens"] += u.get("output", 0) or 0
                m["cache_read_tokens"] += u.get("cacheRead", 0) or 0
                for c in msg.get("content") or []:
                    if c.get("type") == "toolCall":
                        m["tool_calls"] += 1
                        args = json.dumps(c.get("arguments"))
                        if "SKILL.md" in args or "recipes.md" in args:
                            m["read_skill"] = True
                        if FORBIDDEN.search(args):
                            m["contaminated"] = True
                    elif c.get("type") == "text":
                        m["final"] = c.get("text", "")
            elif msg.get("role") == "toolResult":
                text = " ".join(c.get("text", "") for c in msg.get("content") or [] if isinstance(c, dict))
                if msg.get("isError") or re.search(r"Traceback \(most recent|Command exited with code [1-9]", text):
                    m["tool_errors"] += 1
    m["cost"] = round(m["cost"], 4)
    if m["first"] and m["last"]:
        from datetime import datetime
        t = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
        m["session_seconds"] = round(t(m["last"]) - t(m["first"]), 1)
    m["final"] = m["final"][-1500:]
    return m


def grade(task: str, work: Path, env: dict, out: Path) -> dict:
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location(f"grade_{task}", HERE / "tasks" / task / "grade.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.grade(work, env, str(HERE / ".venv" / "bin" / "python"), out)
    except Exception as e:  # noqa: BLE001  a grader crash is a result, not a stop
        return {"checks": {}, "score": 0.0, "success": False, "grader_error": f"{type(e).__name__}: {e}"}


def one(task: str, condition: str, rep: int, run_root: Path, results: Path, args) -> dict:
    base = run_root / task / f"{condition}-{rep}"
    work = base / "work"
    shutil.copytree(HERE / "tasks" / task / "files", work)
    (work / ".venv").symlink_to(HERE / ".venv")
    env = {**os.environ, **keys(), "XDG_CACHE_HOME": str(base / "xdg"), "FUNCTAI_LOG_CALLS": str(base / "calls"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("VIRTUAL_ENV", None)
    provider, model = args.model.split("/", 1)
    cmd = ["pi", "-p", "--mode", "json", "--no-extensions", "--no-skills", "--no-context-files",
           "--no-prompt-templates", "--no-themes", "--session-dir", str(base / "session"),
           "--provider", provider, "--model", model, "--thinking", args.thinking,
           "--append-system-prompt", RULES]
    if condition == "with":
        cmd += ["--skill", str(run_root / "skill" / "functai")]
    cmd.append((HERE / "tasks" / task / "task.md").read_text())
    started = time.time()
    try:
        p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=args.timeout)
        status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
        (base / "pi.stderr").write_text(p.stderr[-20000:])
    except subprocess.TimeoutExpired:
        status = "timeout"
    seconds = round(time.time() - started, 1)
    metrics = session_metrics(base / "session")
    from common import all_records, summary_of
    task_calls = summary_of(all_records(env, work))
    graded = grade(task, work, env, base / "grading")
    result = {"task": task, "condition": condition, "rep": rep, "status": status, "seconds": seconds,
              "agent": {k: v for k, v in metrics.items() if k not in ("first", "last")},
              "functai_calls": task_calls, "grade": graded, "work": str(work)}
    out = results / task / f"{condition}-{rep}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    for f in (base / "session").glob("*.jsonl"):
        shutil.copy(f, out / "session.jsonl")
    print(f"{task:8} {condition:7} #{rep}  {'✓' if graded.get('success') else '✗'} score {graded.get('score')}"
          f"  {seconds:6.0f}s  ${metrics['cost']:.2f}  {metrics['tool_calls']} tools ({metrics['tool_errors']} failed)"
          f"{'  CONTAMINATED' if metrics['contaminated'] else ''}  {status}", flush=True)
    return result


def collect(results: Path):
    """Grade every run of an earlier evaluation again, from its workspace (after fixing a
    grader, or when a run's bookkeeping was lost), and rewrite its result.json."""
    stamp = results.name
    for base in sorted((WORK_ROOT / "runs" / stamp).glob("*/*-*")):
        task, (condition, rep) = base.parent.name, base.name.rsplit("-", 1)
        path = results / task / base.name / "result.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        work = base / "work"
        env = {**os.environ, **keys(), "XDG_CACHE_HOME": str(base / "xdg"), "FUNCTAI_LOG_CALLS": str(base / "calls"),
               "PYTHONDONTWRITEBYTECODE": "1"}
        shutil.rmtree(base / "grading", ignore_errors=True)
        metrics = session_metrics(base / "session")
        from common import all_records, summary_of
        result = {"task": task, "condition": condition, "rep": int(rep), "status": old.get("status", "collected"),
                  "seconds": old.get("seconds"), "agent": {k: v for k, v in metrics.items() if k not in ("first", "last")},
                  "functai_calls": summary_of(all_records(env, work)), "grade": grade(task, work, env, base / "grading"),
                  "work": str(work)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1, default=str))
        for f in (base / "session").glob("*.jsonl"):
            shutil.copy(f, path.parent / "session.jsonl")
        print(f"{task:8} {condition:7} #{rep}  {'✓' if result['grade'].get('success') else '✗'} "
              f"score {result['grade'].get('score')}", flush=True)
    import report
    (results / "report.md").write_text(report.render(results))
    print(report.render(results))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--collect":
        return collect(Path(sys.argv[2]).resolve())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    ap.add_argument("--conditions", nargs="*", default=["without", "with"], choices=["without", "with"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--model", default="openai-codex/gpt-6-astra", help="provider/model as Pi names it")
    ap.add_argument("--thinking", default="medium")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per agent run")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = WORK_ROOT / "runs" / stamp
    results = HERE / "runs" / stamp
    results.mkdir(parents=True)
    shutil.copytree(SKILL, run_root / "skill" / "functai")
    (results / "settings.json").write_text(json.dumps({**vars(args), "stamp": stamp,
        "skill_sha": subprocess.run(["git", "-C", str(REPO), "hash-object", str(SKILL / "SKILL.md")],
                                    capture_output=True, text=True).stdout.strip()}, indent=1))
    jobs = [(t, c, r) for r in range(1, args.reps + 1) for t in args.tasks for c in args.conditions]
    print(f"{len(jobs)} runs → {results}", flush=True)
    def safe(job):
        try:
            return one(*job, run_root, results, args)
        except Exception as e:  # noqa: BLE001  one broken run must not stop the others
            print(f"{job}: {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(safe, jobs))
    import report
    print(report.render(results))
    (results / "report.md").write_text(report.render(results))


if __name__ == "__main__":
    main()
