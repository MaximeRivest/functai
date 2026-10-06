"""Grade a hand-written solution for one task, without any agent: checks that a
task's grader gives a known-good solution full marks (and a known-bad one less).

    .venv/bin/python try_reference.py forms tasks/forms/reference/extract.py
    .venv/bin/python try_reference.py measure tasks/measure/reference/score.py

The solution is copied into a fresh workspace with the task's files, run once
(real model calls: a few cents), then graded.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FUNCTAI_SKILL_EVAL_WORK", Path.home() / ".cache" / "functai-skill-eval"))


def load_grader(task: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"grade_{task}", HERE / "tasks" / task / "grade.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(task: str, solution: str):
    work = ROOT / "reference" / task / Path(solution).stem
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(HERE / "tasks" / task / "files", work)
    shutil.copy(solution, work / Path(solution).name)
    env = dict(os.environ, XDG_CACHE_HOME=str(work / ".xdg"), FUNCTAI_LOG_CALLS=str(work / ".calls"))
    python = str(HERE / ".venv" / "bin" / "python")
    r = subprocess.run([python, Path(solution).name], cwd=work, env=env, capture_output=True, text=True)
    print("solution exit", r.returncode, r.stderr[-800:])
    result = load_grader(task).grade(work, env, python, work / ".grade")
    for name, c in result["checks"].items():
        print(f"  {'✓' if c['pass'] else '✗'} {name:18} {c['value'] if c['value'] is not None else '':<6} {c['detail']}")
    print(f"score {result['score']}  success {result['success']}")
    return result


if __name__ == "__main__":
    main(*sys.argv[1:3])
