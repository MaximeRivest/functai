"""The one-page summary: what the pipeline does, how often it is right, where it goes
wrong, what it costs, where data goes, what was tested, and a few examples step by
step. One self-contained HTML file: no server, no account, no script; it opens in
any browser and prints to PDF. Built only from the records (no model writes it)."""

from __future__ import annotations

import datetime
import html
import json
from typing import Any, Dict, List, Optional, Tuple

from .data import wilson


def fmt_cents(c: Optional[float]) -> str:
    from .running import fmt_cents as f
    return f(c)


def provider(model: Optional[str]) -> str:
    m = (model or "").lower()
    if ":" in m:
        return m.split(":", 1)[0]
    for prefix, who in (("claude", "Anthropic"), ("gpt", "OpenAI"), ("o1", "OpenAI"), ("o3", "OpenAI"),
                        ("o4", "OpenAI"), ("gemini", "Google"), ("grok", "xAI"), ("mistral", "Mistral"),
                        ("deepseek", "DeepSeek")):
        if m.startswith(prefix):
            return who
    return model or "an unknown provider"


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def score_words(end: Dict[str, Any]) -> Tuple[str, bool]:
    """How often it is right, in words; and whether checking more is the next step."""
    k, n = end["right"], end["checked"]
    if n == 0:
        return "Nobody has checked its answers yet.", True
    if n < 15:
        return (f"{k} of {n} checked {'answers were' if n != 1 else 'answer was'} right. That is too few for a "
                f"reliable score: checking about {max(10, 15 - n)} more would give one."), True
    lo, hi = wilson(k, n)
    return f"{k} of {n} checked answers were right: the true rate is likely between {_pct(lo)} and {_pct(hi)}.", False


def weak_spot(snap: Dict[str, Any], blame: Optional[Dict[str, Any]]) -> str:
    if blame and blame.get("counts"):
        top, k = next(iter(blame["counts"].items()))
        return (f"Mostly in {top}: when it was given the right answer, {k} of {blame['wrong']} wrong results "
                f"came out right.")
    worst = sorted(((v["checked"] - v["right"], name) for name, v in snap["scores"]["steps"].items()), reverse=True)
    if worst and worst[0][0] > 0:
        return f"Most often in {worst[0][1]} ({worst[0][0]} answers judged wrong)."
    if snap["scores"]["end"]["wrong"]:
        return "Some results are wrong, but no step has been judged yet, so where is not known."
    return "No wrong result has been found yet."


def facts(snap: Dict[str, Any], *, blame: Optional[Dict[str, Any]] = None, monthly: Optional[int] = None
          ) -> List[Tuple[str, str]]:
    prog = snap["program"]
    end = snap["scores"]["end"]
    steps = snap["steps"]
    what = (prog["description"] + " " if prog["description"] else "") + "Steps: " + "; ".join(
        f"{name} ({(s.get('instruction') or '').splitlines()[0] if s.get('instruction') else 'no instruction'})"
        for name, s in steps.items()) + "."
    acc = snap["money"].get("accounts") or []
    paid = [a for a in acc if not a.get("local")]
    subs = [a for a in paid if a.get("subscription")]
    who = sorted({a["company"] for a in paid})
    pm = snap["money"]["per_message_cents"]
    monthly = monthly or snap["settings"]["monthly"]
    if paid and len(subs) == len(paid):
        cost = f"No cost per use: it runs on {' and '.join('your ' + a['company'] + ' subscription' for a in subs)}, within its limits."
    elif pm is not None:
        cost = (f"About {fmt_cents(pm)} per message, paid to {' and '.join(who) or 'nobody'}. At {monthly:,} messages a "
                f"month: about {fmt_cents(pm * monthly)}.")
    else:
        cost = "Not known yet: nothing has run."
    if snap["money"]["unknown_prices"]:
        cost += f" (No price is known for {', '.join(snap['money']['unknown_prices'])}.)"
    judges = sorted({(c.get("judgement") or {}).get("by") for r in snap["rows"]
                     for c in [r.get("end") or {}, *(r.get("steps") or {}).values()]
                     if (c.get("judgement") or {}).get("source") == "person"} - {None})
    ran = sum(1 for r in snap["rows"] if r.get("current"))
    tested = (f"Version {prog['version']}, on {ran} example{'s' if ran != 1 else ''}"
              f"{' (' + str(end['failed']) + ' failed)' if end['failed'] else ''}; "
              + (f"{end['checked']} checked by {', '.join(judges) or 'people'}." if end["checked"]
                 else "nobody has checked the answers yet."))
    return [
        ("What it does", what),
        ("How often it's right", score_words(end)[0]),
        ("Where it goes wrong", weak_spot(snap, blame)),
        ("What it costs", cost),
        ("Where data goes", ("To " + " and ".join(who) + ", to get each step's answer. What they keep depends on "
                             "their rules and the account used.") if who else "Nowhere: every step runs on this computer."),
        ("What was tested", tested),
    ]


def map_svg(graph: Dict[str, Any]) -> str:
    """The map as an SVG drawing (boxes, wires, conditions), for the page."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    w, h = graph["width"], graph["height"]
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="system-ui,sans-serif">']
    for e in graph["edges"]:
        a, b = nodes[e["from"]], nodes[e["to"]]
        x1, y1, x2, y2 = a["x"] + a["w"], a["y"] + a["h"] / 2, b["x"], b["y"] + b["h"] / 2
        m = (x1 + x2) / 2
        out.append(f'<path d="M{x1},{y1} C{m},{y1} {m},{y2} {x2},{y2}" fill="none" stroke="#c9c7bf" stroke-width="2"/>')
    fill = {"input": "#f1f0eb", "const": "#f1f0eb", "value": "#faf9f6", "step": "#fff", "output": "#fff"}
    for n in graph["nodes"]:
        dash = ' stroke-dasharray="4 3"' if n["kind"] == "value" else ""
        out.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="12" '
                   f'fill="{fill[n["kind"]]}" stroke="#d8d6cf"{dash}/>')
        title = "end result" if n["kind"] == "output" else n["label"]
        out.append(f'<text x="{n["x"] + 12}" y="{n["y"] + 24}" font-size="14" font-weight="700">'
                   f'{html.escape(str(title))}</text>')
        sub = n.get("signature") or n.get("sub") or (f'if {n["when"]}' if n.get("when") else "")
        if n["kind"] == "step" and n.get("when"):
            sub2 = f'if {n["when"]}'
            out.append(f'<text x="{n["x"] + 12}" y="{n["y"] + 64}" font-size="11" fill="#8a8990">'
                       f'{html.escape(sub2[:38])}</text>')
        if sub:
            out.append(f'<text x="{n["x"] + 12}" y="{n["y"] + 44}" font-size="11" fill="#77767d">'
                       f'{html.escape(str(sub)[:40])}</text>')
    out.append("</svg>")
    return "".join(out)


def _example_html(snap: Dict[str, Any], row: Dict[str, Any]) -> str:
    def val(v: Any) -> str:
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        text = text if len(text) <= 400 else text[:399] + "…"
        return html.escape(text) if text != "" else "<i>(nothing)</i>"
    parts = [f'<div class="ex"><div class="q">Example {row["n"]}</div>']
    parts.append("<p><b>In:</b> " + "; ".join(f"{html.escape(k)}: {val(v)}" for k, v in row["inputs"].items())
                 + "</p><ol>")
    for name, c in row.get("steps", {}).items():
        j = c.get("judgement") or {}
        mark = {"right": ' <span class="r">✔ right</span>', "wrong": ' <span class="w">✘ wrong</span>'}.get(
            j.get("verdict"), "")
        should = (f'<br><span class="s">should have been: {val(j["answer"])}</span>'
                  if j.get("verdict") == "wrong" and j.get("answer") is not None else "")
        said = f"failed: {html.escape(c['error'])}" if c.get("error") else val(c.get("value"))
        parts.append(f"<li><b>{html.escape(name)}</b> said {said}{mark}{should}</li>")
    end = row.get("end") or {}
    state = row.get("state")
    parts.append(f'</ol><p><b>End result:</b> {val(end.get("value")) if not end.get("error") else "failed"} — '
                 f'<b class="{ {"right": "r", "wrong": "w"}.get(state, "") }">{html.escape(str(state))}</b></p></div>')
    return "".join(parts)


def summary_html(snap: Dict[str, Any], *, blame: Optional[Dict[str, Any]] = None, monthly: Optional[int] = None,
                 title: Optional[str] = None) -> str:
    prog = snap["program"]
    name = title or prog["name"].replace("_", " ").capitalize()
    rows = [r for r in snap["rows"] if r.get("current")]
    picks: List[Dict[str, Any]] = []
    for state in ("right", "wrong", "failed", "not checked"):
        r = next((r for r in rows if r["state"] == state and r not in picks), None)
        if r is not None and len(picks) < 3:
            picks.append(r)
    table = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
                    for k, v in facts(snap, blame=blame, monthly=monthly))
    made = datetime.date.today().isoformat()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)}: summary</title>
<style>
body{{font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:860px;margin:40px auto;padding:0 22px;color:#1c1c1f}}
h1{{margin:0 0 4px}} .meta{{color:#6f6e75;margin:0 0 24px}}
table{{border-collapse:collapse;width:100%}} th{{text-align:left;color:#6f6e75;font-weight:600;padding:8px 16px 8px 0;vertical-align:top;width:180px}} td{{padding:8px 0}}
tr+tr th,tr+tr td{{border-top:1px solid #eee}}
.ex{{border:1px solid #e7e5df;border-radius:12px;padding:12px 16px;margin:12px 0}} .q{{color:#6f6e75;font-size:13px}}
.r{{color:#1e8a4e}} .w{{color:#c93b3b}} .s{{color:#1e8a4e;font-size:14px}} ol{{margin:6px 0}}
.foot{{color:#6f6e75;font-size:13px;margin-top:28px}}
@media print{{body{{margin:0}} .ex{{break-inside:avoid}}}}
</style></head><body>
<h1>{html.escape(name)}</h1>
<p class="meta">Version {prog['version']} · summary made {made} · from <code>{html.escape(prog['file'])}</code></p>
<table>{table}</table>
<h2>How it works</h2>
{map_svg(snap['graph'])}
<h2>Examples, step by step</h2>
{''.join(_example_html(snap, r) for r in picks) or '<p>Nothing has run yet.</p>'}
<p class="foot">Made with FunctAI Studio from the recorded runs; no AI wrote this page. The pipeline is ordinary
Python ({html.escape(prog['file'])}); prices are list prices of {html.escape(snap['money']['prices_date'])}.</p>
</body></html>
"""
