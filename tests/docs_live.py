"""The documentation runs, against real models (costs cents). Not run by pytest.

    set -a; source ~/Projects/lm15-dev/.env; set +a
    .venv/bin/python tests/docs_live.py            # the README's code blocks, in order
    .venv/bin/python tests/docs_live.py --render   # and re-render the examples and the website
    .venv/bin/python tests/docs_live.py --render --refresh   # the website re-runs every page

README: every ```python block runs, top to bottom, in one namespace, in a
scratch folder, except those right after a `<!-- skip: reason -->` line
(sign-ins, other people's keys, multi-file projects).

The examples are Quarto documents (examples/*/main.qmd); rendering runs
every cell and writes the Markdown next to them (README.md). The website
(docs/, a Quarto project) renders after them, since it shows those
READMEs: every page's code runs, and the outputs are kept in
docs/_freeze/ (committed), so a page runs again only when its source
changed (--refresh: all of them). A failing cell fails the render.
Quarto runs cells on the Python named by $QUARTO_PYTHON (default: this
one), which needs the docs group: `uv sync --group docs`.
"""

from __future__ import annotations

import linecache
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def readme_blocks(path: Path) -> list[tuple[int, str, str | None]]:
    """(line, code, skip reason or None) for each python block."""
    text = path.read_text()
    out = []
    for m in re.finditer(r"(?:<!-- skip: (.*?) -->\n)?```python\n(.*?)```", text, re.S):
        line = text[: m.start(2)].count("\n")
        out.append((line, m.group(2), m.group(1)))
    return out


def run_readme() -> int:
    failures = 0
    ns: dict = {"__name__": "__main__"}
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        for line, code, skip in readme_blocks(ROOT / "README.md"):
            if skip:
                print(f"  skip  README.md:{line}  ({skip})")
                continue
            name = f"<README.md:{line}>"
            # register the source, as notebooks do: functai reads a function's body
            linecache.cache[name] = (len(code), None, code.splitlines(True), name)
            try:
                exec(compile(code, name, "exec", dont_inherit=True), ns)
                print(f"  ok    README.md:{line}")
            except Exception:  # noqa: BLE001 — report every failing block
                failures += 1
                print(f"  FAIL  README.md:{line}\n{traceback.format_exc(limit=3)}")
        os.chdir(ROOT)
    return failures


def render_all(refresh: bool = False) -> int:
    quarto = os.environ.get("QUARTO", "quarto")
    env = {**os.environ, "QUARTO_PYTHON": os.environ.get("QUARTO_PYTHON", sys.executable)}
    failures = 0
    tracked = subprocess.run(["git", "ls-files", "examples/*/main.qmd"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    new = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "examples/*/main.qmd"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()
    for qmd in sorted(ROOT / f for f in [*tracked, *new]):   # not ignored scratch folders
        done = subprocess.run([quarto, "render", qmd.name], cwd=qmd.parent, env=env,
                              capture_output=True, text=True)
        ok = done.returncode == 0
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {qmd.relative_to(ROOT)}")
        if not ok:
            print(done.stderr[-2000:])
    site = subprocess.run([quarto, "render", *(["--cache-refresh"] if refresh else [])], cwd=ROOT / "docs",
                          env=env, capture_output=True, text=True, check=False)
    failures += site.returncode != 0
    print(f"  {'ok  ' if site.returncode == 0 else 'FAIL'}  docs/ (the website, in docs/_site)")
    if site.returncode != 0:
        print(site.stderr[-3000:])
    return failures


if __name__ == "__main__":
    failures = run_readme()
    if "--render" in sys.argv:
        failures += render_all(refresh="--refresh" in sys.argv)
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
