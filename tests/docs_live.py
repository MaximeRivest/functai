"""The documentation runs, against real models (costs cents). Not run by pytest.

    set -a; source ~/Projects/lm15-dev/.env; set +a
    .venv/bin/python tests/docs_live.py            # the README's code blocks, in order
    .venv/bin/python tests/docs_live.py --render   # and every docs page, run again on rat

README: every ```python block runs, top to bottom, in one namespace, in a
scratch folder, except those right after a `<!-- skip: reason -->` line.

The docs are Markdown notebooks (MRMD, the format Chattering writes):
`--render` runs each page of docs/ on a fresh rat kernel and writes the
new outputs into it (tools/docs.py run), so a failing cell fails it. The
examples (examples/*/README.md) are notebooks too, run one at a time on
purpose: `python tools/docs.py run examples/modules/README.md`.
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


def render_all() -> int:
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "docs.py"), "run"], cwd=ROOT, check=False)
    return done.returncode


if __name__ == "__main__":
    failures = run_readme()
    if "--render" in sys.argv:
        failures += render_all()
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
