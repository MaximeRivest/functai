"""The comparison table for one evaluation run: with vs without the skill.

    .venv/bin/python report.py runs/<stamp>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*/*/result.json"))]


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(v, kind=""):
    if v is None:
        return "–"
    if kind == "$":
        return f"${v:.2f}"
    if kind == "%":
        return f"{v:.0%}"
    if kind == "s":
        return f"{v / 60:.1f} min"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def render(folder: Path) -> str:
    folder = Path(folder)
    rows = load(folder)
    settings = json.loads((folder / "settings.json").read_text()) if (folder / "settings.json").exists() else {}
    lines = [f"# Skill evaluation {folder.name}", "",
             f"Agent: {settings.get('model', '?')} (thinking {settings.get('thinking', '?')}); "
             f"{len(rows)} runs. Medians per task and condition; time is the agent session's duration; cost is the agent's list-price cost "
             "as Pi computes it (subscription runs are not billed per token).", "",
             "| task | condition | runs | success | score | time | agent cost | agent tokens | tool calls | failed tools | task's model calls | read skill | contaminated |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tasks = sorted({r["task"] for r in rows})
    for t in tasks:
        for cond in ("without", "with"):
            rs = [r for r in rows if r["task"] == t and r["condition"] == cond]
            if not rs:
                continue
            a = [r["agent"] for r in rs]
            lines.append("| " + " | ".join([
                t, cond, str(len(rs)),
                fmt(sum(r["grade"].get("success", False) for r in rs) / len(rs), "%"),
                fmt(med([r["grade"].get("score") for r in rs])),
                fmt(med([r["agent"].get("session_seconds") or r["seconds"] for r in rs]), "s"),
                fmt(med([x["cost"] for x in a]), "$"),
                fmt(med([x["input_tokens"] + x["output_tokens"] + x["cache_read_tokens"] for x in a])),
                fmt(med([x["tool_calls"] for x in a])),
                fmt(med([x["tool_errors"] for x in a])),
                fmt(med([r["functai_calls"]["uncached"] for r in rs])),
                f"{sum(x['read_skill'] for x in a)}/{len(a)}",
                f"{sum(x['contaminated'] for x in a)}/{len(a)}",
            ]) + " |")
    lines += ["", "## Failed checks", ""]
    for r in rows:
        bad = {k: c for k, c in r["grade"].get("checks", {}).items() if not c["pass"]}
        if bad or r["grade"].get("grader_error"):
            lines.append(f"- **{r['task']} / {r['condition']} #{r['rep']}**: "
                         + "; ".join(f"{k} ({c['detail'] or c['value']})" for k, c in bad.items())
                         + (f" grader error: {r['grade']['grader_error']}" if r["grade"].get("grader_error") else ""))
    lines += ["", "## Totals", ""]
    for cond in ("without", "with"):
        rs = [r for r in rows if r["condition"] == cond]
        if rs:
            lines.append(f"- **{cond}**: {sum(r['grade'].get('success', False) for r in rs)}/{len(rs)} succeeded, "
                         f"{sum((r['agent'].get('session_seconds') or r['seconds'] or 0) for r in rs) / 60:.0f} agent-minutes, "
                         f"${sum(r['agent']['cost'] for r in rs):.2f} list price, "
                         f"{sum(r['agent']['tool_calls'] for r in rs)} tool calls")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
