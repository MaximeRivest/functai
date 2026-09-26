"""The documentation: Markdown notebooks (MRMD), run on rat, published with Zensical.

Every page in docs/ is a notebook: ```python cells, each followed by its
output in an ```output fence, the format Chattering (MRMD) writes when you
press Run. Open any page in Chattering and run it; or run them all here:

    python tools/docs.py generate          # reference/, examples/, news.md from the code and the repo
    python tools/docs.py run [PAGE ...]    # run pages (default: all) and write their outputs
    python tools/docs.py site              # the website, in site/ (runs nothing)

`run` needs model keys in the environment (the examples call real models,
for cents). Each page runs top to bottom on its own fresh kernel,
`py-functai-docs` (the project's .venv; registered on first use), so no
page depends on what another left behind, and the project's shared kernel
is never touched. A cell that fails stops the page and fails the run.

Code shown but not run (a sign-in, a GPU job) is fenced ```{.python .no-run}:
MRMD runs only fences whose info string is a bare language name.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
TOOLS = ROOT / "tools"
REPO = "https://github.com/maximerivest/functai/blob/master"
KERNEL = "py-functai-docs"
# how tables print in the docs kernel: whole texts, every column, no shape line
KERNEL_ENV = {"POLARS_FMT_STR_LEN": "90", "POLARS_FMT_MAX_COLS": "12", "POLARS_TABLE_WIDTH": "160",
              "POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION": "1"}

# ------------------------------------------------------------------ notebooks: MRMD's result format

FENCE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")
RUNNABLE = re.compile(r"^(python|py|python3)\s*$")          # MRMD: a bare language word
OWNED_IMAGE = re.compile(r"^!\[plot(?:-\d+)?\]\(([^)\s]*_assets/[^)\s]*)\)\s*$")


def blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    """Fenced blocks as (first line, last line, info string), fences inside fences ignored."""
    out, i = [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        ticks, info = m.group(2), m.group(3).strip()
        j = i + 1
        while j < len(lines):
            c = FENCE.match(lines[j])
            if c and c.group(2)[0] == ticks[0] and len(c.group(2)) >= len(ticks) and not c.group(3).strip():
                break
            j += 1
        out.append((i, j, info))
        i = j + 1
    return out


def fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def format_result(text: str) -> list[str]:
    body = text.rstrip()
    if not body:
        return []
    ticks = fence_for(body)
    return [ticks + "output", *body.split("\n"), ticks]


def cells(text: str) -> list[tuple[int, int, str]]:
    """The runnable cells: (first line, last line, code)."""
    lines = text.split("\n")
    return [(a, b, "\n".join(lines[a + 1:b])) for a, b, info in blocks(lines) if RUNNABLE.match(info)]


def write_results(text: str, results: list[str | None]) -> str:
    """The notebook with each cell's result under it (replacing an older one).
    ``results[i]`` None leaves cell i as it is (it did not run)."""
    lines = text.split("\n")
    found = blocks(lines)
    runnable = [(a, b) for a, b, info in found if RUNNABLE.match(info)]
    outputs = {a: (a, b) for a, b, info in found if info.split()[:1] and
               info.split()[0].lower().split(":")[0] == "output"}
    for (a, b), result in reversed(list(zip(runnable, results))):
        if result is None:
            continue
        # what follows the cell: blank lines, then maybe its old result
        k = b + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        end = b + 1
        if k in outputs:
            end = outputs[k][1] + 1
            while True:                           # and the plots a run left after it
                n = end
                while n < len(lines) and not lines[n].strip():
                    n += 1
                if n < len(lines) and OWNED_IMAGE.match(lines[n]):
                    end = n + 1
                else:
                    break
        new = format_result(result)
        lines[b + 1:end] = ([""] + new) if new else []
    return "\n".join(lines)


# ------------------------------------------------------------------ running on rat


def rat(*args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["rat", *args], cwd=ROOT, capture_output=True, text=True, check=check,
                          env={**os.environ, **(env or {})})


def ensure_kernel() -> None:
    if KERNEL not in rat("status", check=False).stdout:
        rat("add", KERNEL, str(ROOT), "--venv", str(ROOT / ".venv"),
            *[a for k, v in KERNEL_ENV.items() for a in ("--env", f"{k}={v}")])


def run_cell(code: str) -> tuple[str, str]:
    """("ok" | "failed" | "rat", text). "failed": the cell raised (its traceback
    is its output, as in Chattering). "rat": rat itself could not run it; that
    is never written into a page."""
    done = rat("run", "--timeout", "900s", KERNEL, "--events", code, check=False, env=KERNEL_ENV)
    for line in reversed(done.stdout.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "result":
            r = event.get("result") or {}
            if event.get("success"):
                return "ok", r.get("output") or ""
            out = (r.get("output") or "").rstrip()
            return "failed", (out + "\n" if out else "") + _trim_traceback(r.get("error") or "")
        if event.get("event") == "error":
            return "rat", event.get("message", "rat could not run the cell")
    return "rat", (done.stderr or done.stdout or "no result from rat").strip()


def _trim_traceback(error: str) -> str:
    """The kernel's own frames out of a traceback: it starts at the cell."""
    lines = error.splitlines()
    for i, line in enumerate(lines):
        if '"<rat-cell-' in line:
            return "\n".join(["Traceback (most recent call last):", *lines[i:]])
    return error


def run_page(page: Path) -> bool:
    text = page.read_text()
    todo = cells(text)
    if not todo:
        return True
    rat("restart", KERNEL, env=KERNEL_ENV)
    results: list[str | None] = []
    status = "ok"
    for _a, _b, code in todo:
        if status != "ok":
            results.append(None)
            continue
        status, out = run_cell(code)
        results.append(None if status == "rat" else out)
        if status != "ok":
            print(f"  FAIL  {page.relative_to(ROOT)}: {'rat: ' if status == 'rat' else ''}{out[-1500:]}")
    page.write_text(write_results(text, results))
    if status == "ok":
        print(f"  ok    {page.relative_to(ROOT)} ({len(todo)} cells)")
    return status == "ok"


def pages(args: list[str]) -> list[Path]:
    if args:
        return [Path(a).resolve() for a in args]
    # docs/examples/ are copies of examples/*/README.md: run those on purpose (some need a GPU, the web)
    return sorted(p for p in DOCS.rglob("*.md")
                  if not {"_assets", "examples"} & set(p.relative_to(DOCS).parts))


# ------------------------------------------------------------------ generated pages

EXAMPLES = [  # in the order of the gallery
    ("typing_and_extraction", "Types in, types out: lists, dicts, enums, literals, dataclasses, pydantic models."),
    ("docments_flexiclass", "Comments are prompts: on parameters, the return line, class fields and outputs."),
    ("claide_code", "A terminal assistant: a shell tool with an allow-list, and memory."),
    ("local_simple_rag_agent", "An agent that reads the web, a fact checker, and the same agent on a local model."),
    ("graph_rag", "A knowledge graph built chunk by chunk with pydantic models, then queried."),
    ("modules", "A multi-hop fact checker as one @module: evaluated, optimized, run on a table."),
    ("optimizing_translator", "English to Québécois French: an AI judge as the metric, InstructionSearch, before and after."),
    ("tracking_and_osb", "Observability: phistory, token usage, logged evaluation runs, the reply cache."),
]

REFERENCE_SETUP = "```python\nimport functai\nfrom functai import *\n```\n\n"


def outputs_by_code(text: str) -> dict[str, str]:
    """Each cell's code → the output written under it (for cells that have one)."""
    lines = text.split("\n")
    found = blocks(lines)
    out = {}
    for i, (a, b, info) in enumerate(found):
        if not RUNNABLE.match(info):
            continue
        k = b + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        nxt = next((x for x in found[i + 1:i + 2] if x[0] == k and x[2].split(":")[0] == "output"), None)
        out["\n".join(lines[a + 1:b])] = "\n".join(lines[nxt[0] + 1:nxt[1]]) if nxt else ""
    return out


def keep_outputs(new: str, old: str) -> str:
    """``new`` with the outputs ``old`` had for the cells whose code did not change."""
    known = outputs_by_code(old)
    return write_results(new, [known.get(code) for _a, _b, code in cells(new)])


def generate_reference() -> None:
    """reference/*.md from the docstrings (quartodoc writes the Markdown).
    Code blocks are shown, not run, except in an Examples section, where they
    are cells (a block starting with `# not run: <why>` stays shown only)."""
    out = DOCS / "reference"
    before = {p.name: p.read_text() for p in out.glob("*.md")}
    for old in out.glob("*.md"):
        old.unlink()
    subprocess.run([sys.executable, "-m", "quartodoc", "build", "--config", str(TOOLS / "reference.yml")],
                   cwd=TOOLS, check=True, capture_output=True)
    (TOOLS / "objects.json").unlink(missing_ok=True)
    for qmd in out.glob("*.qmd"):
        name = "ai-sentinel" if qmd.stem == "_ai" else qmd.stem
        text = qmd.read_text().replace(".qmd#", ".md#").replace(".qmd)", ".md)").replace("](_ai.md", "](ai-sentinel.md")
        qmd.unlink()
        text = re.sub(r"```python\n", "```{.python .no-run}\n", text)
        m = re.search(r"^(#+) Examples[^\n]*\n", text, re.M)
        if m:
            level = len(m.group(1))
            end = re.search(rf"^#{{1,{level}}} ", text[m.end():], re.M)
            stop = m.end() + end.start() if end else len(text)
            section = re.sub(r"```\{\.python \.no-run\}\n(?!# not run)", "```python\n", text[m.end():stop])
            text = text[:m.end()] + "\n" + REFERENCE_SETUP + section.lstrip("\n") + text[stop:]
        ids = set(re.findall(r"\{ #([^ }]+)", text))
        text = re.sub(r"\[([^\]]+)\]\(#([^)]+)\)", lambda m: m.group(0) if m.group(2) in ids else f"`{m.group(1)}`", text)
        text = see_also_links(text, {p.stem for p in out.glob("*.qmd")} | {q.stem for q in out.glob("*.md")})
        if name == "index":
            text = text.replace("# Reference {.doc .doc-index}", "# Reference")
        (out / f"{name}.md").write_text(keep_outputs(text, before.get(f"{name}.md", "")))


def see_also_links(text: str, pages: set[str]) -> str:
    def repl(m: re.Match) -> str:
        items = []
        for line in m.group(2).strip().splitlines():
            if not line.strip():
                continue
            name, _, why = line.partition(" : ")
            head = name.split(".")[0]
            target = f"{name}.md" if name in pages else f"{head}.md#functai.{name}" if head in pages else None
            items.append(f"- [`{name}`]({target}): {why}" if target else f"- `{name}`: {why}")
        return m.group(1) + "\n" + "\n".join(items) + "\n\n"
    return re.sub(r"(^#+ See Also[^\n]*\n)(.*?)(?=^#)", repl, text, flags=re.MULTILINE | re.DOTALL)


def generate_examples() -> None:
    out = DOCS / "examples"
    out.mkdir(exist_ok=True)
    rows = []
    for name, blurb in EXAMPLES:
        text = re.sub(r"\A---\n.*?\n---\n+", "", (ROOT / "examples" / name / "README.md").read_text(), flags=re.S)
        title = re.match(r"# (.+)\n", text).group(1)
        body = text[text.index("\n") + 1:]
        (out / f"{name}.md").write_text(
            f"# {title}\n\n*{blurb}*\n\n<!-- generated by docs/build.py from examples/{name}/README.md "
            f"(the notebook): edit that one -->\n" + body)
        rows.append(f"- **[{title}]({name}.md)**: {blurb}")
    (out / "index.md").write_text(
        "# Examples\n\nEach example solves one real problem from start to finish, with the real replies "
        "of real models. They assume you have read [Get started](../get-started.md).\n\n" + "\n".join(rows) + "\n")


def generate_news() -> None:
    text = (ROOT / "CHANGELOG.md").read_text()
    (DOCS / "news.md").write_text(re.sub(r"^# Changelog\n", "# News\n", text))


# ------------------------------------------------------------------ main


def main(argv: list[str]) -> int:
    what, rest = (argv[0], argv[1:]) if argv else ("help", [])
    if what == "generate":
        generate_reference()
        generate_examples()
        generate_news()
        return 0
    if what == "run":
        ensure_kernel()
        failures = sum(not run_page(p) for p in pages(rest))
        print(f"\n{failures} page(s) failed")
        return 1 if failures else 0
    if what == "site":
        return subprocess.run([str(Path(sys.executable).parent / "zensical"), "build", "--clean"], cwd=ROOT).returncode
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
