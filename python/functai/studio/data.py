"""What happened: the project's call log read into rows, steps, judgements, scores and costs.

Everything here is computed from files: the call log in ``.functai/calls`` (runs
and ratings, FunctAI's contract format), the examples file and the code. Delete
nothing and lose nothing; delete ``.functai/studio`` and only settings go.

Judgements, and where they come from (``source``):

- ``person``: someone judged this very call (a rating);
- ``carried``: someone judged the same step's same answer to the same input in
  another run (another version, say), or gave the right answer and this answer
  is exactly it, or (for answers that can be compared exactly: a choice, a
  number, yes/no) is not it. Text answers that differ are never judged for
  anyone: they stay "not checked";
- ``step``: no one judged the end result, but a step on its way was judged
  wrong, so the end is counted wrong "because of" that step.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .. import calllog
from ..calllog import canonical
from .graph import build_map
from .project import Project, StudioError, example_key

PRICES = json.loads((Path(__file__).parent / "prices.json").read_text())


# ------------------------------------------------------------------ money

def _bare(model: Optional[str]) -> str:
    m = (model or "").split(":", 1)[-1] if ":" in (model or "") else (model or "")
    return m.split("/", 1)[-1]


SUBSCRIPTION_PREFIXES = ("claude:", "claude-code:", "chatgpt:", "codex:", "openai-codex:", "copilot:",
                         "github-copilot:", "kimi:", "kimi-code:")
LOCAL_PREFIXES = ("ollama:", "lmstudio:", "llamacpp:", "vllm:", "local:")


def price_of(model: Optional[str]) -> Optional[Dict[str, float]]:
    """List price per million tokens; a subscription or a model on this computer costs nothing per use."""
    m = model or ""
    if m.startswith(SUBSCRIPTION_PREFIXES) or m.startswith(LOCAL_PREFIXES):
        return {"input": 0.0, "output": 0.0, "free": True}
    if m.startswith("openrouter:"):
        listed = m.split(":", 1)[1]
        return next((p for p in PRICES["models"].values() if p.get("listed_as") == listed), None)
    return PRICES["models"].get(_bare(model))


def exchange_cents(ex: Dict[str, Any], *, as_if_new: bool = False) -> Tuple[float, Optional[str]]:
    """What one model request cost, in cents, and the model when its price is unknown.
    A reply from the cache cost nothing (unless ``as_if_new``: what it would cost asked again),
    nor one paid by a subscription or answered on this computer."""
    if ex.get("cached") and not as_if_new:
        return 0.0, None
    from .connect import LOCAL_ROUTES, SUBSCRIPTION_ROUTES
    if ex.get("provider") in SUBSCRIPTION_ROUTES or ex.get("provider") in LOCAL_ROUTES:
        return 0.0, None
    usage = ex.get("usage") or {}
    p = price_of(ex.get("model"))
    if p is None:
        return 0.0, ex.get("model") if (usage.get("input_tokens") or usage.get("output_tokens")) else None
    cents = ((usage.get("input_tokens") or 0) * p["input"] + (usage.get("output_tokens") or 0) * p["output"]) / 1e6 * 100
    return cents, None


def call_cents(call: Dict[str, Any], *, as_if_new: bool = False) -> Tuple[float, Set[str]]:
    total, unknown = 0.0, set()
    for ex in call.get("exchanges") or []:
        c, u = exchange_cents(ex, as_if_new=as_if_new)
        total += c
        if u:
            unknown.add(u)
    return total, unknown


def wilson(k: int, n: int) -> Tuple[float, float]:
    if not n:
        return 0.0, 1.0
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - m) / d), min(1.0, (c + m) / d)


# ------------------------------------------------------------------ reading the log

def read_log(project: Project) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not project.calls_dir.exists():
        return [], []
    return calllog.read(project.calls_dir)


def _answer(call: Dict[str, Any]) -> Any:
    name = (call.get("program") or {}).get("answer") or "result"
    outs = call.get("outputs")
    if not isinstance(outs, dict):
        return None
    return outs.get(name)


def _key(value: Any) -> str:
    try:
        return canonical(value)
    except Exception:  # noqa: BLE001
        return repr(value)


def _exact(shape: Dict[str, Any]) -> bool:
    """Answers of this shape can be compared exactly (a choice, a number, yes/no)."""
    return bool(shape.get("enum")) or shape.get("type") in ("integer", "number", "boolean")


class Judgements:
    """Who said what about which answers: per call, and per (step, input, answer)."""

    def __init__(self, calls: Iterable[Dict[str, Any]], ratings: Iterable[Dict[str, Any]]):
        self.current = calllog.current_ratings(ratings)
        self.right: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.wrong: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.answer: Dict[Tuple[str, str], Any] = {}
        for c in calls:
            rs = self.current.get(c.get("id"))
            if not rs or c.get("outputs") is None:
                continue
            r = rs[-1]
            k = (c["program"]["name"], _key(c.get("inputs")))
            out = _key(_answer(c))
            if r["verdict"] == "right":
                self.right[k].add(out)
                self.answer.setdefault(k, _answer(c))
            else:
                self.wrong[k].add(out)
                if r.get("answer") is not None:
                    self.answer[k] = r["answer"]

    def of(self, call: Dict[str, Any], *, exact: bool) -> Optional[Dict[str, Any]]:
        rs = self.current.get(call.get("id"))
        if rs:
            r = rs[-1]
            return {"verdict": r["verdict"], "source": "person", "by": r.get("by") or "this computer's account",
                    "answer": r.get("answer"), "note": r.get("note"), "at": r.get("at")}
        if call.get("outputs") is None:
            return None
        k = (call["program"]["name"], _key(call.get("inputs")))
        out = _key(_answer(call))
        if out in self.right.get(k, ()):
            return {"verdict": "right", "source": "carried", "why": "the same answer was judged right before"}
        if out in self.wrong.get(k, ()):
            return {"verdict": "wrong", "source": "carried", "why": "the same answer was judged wrong before",
                    "answer": self.answer.get(k)}
        if k in self.answer:
            if _key(self.answer[k]) == out:
                return {"verdict": "right", "source": "carried", "why": "it is the answer someone said was right"}
            if exact:
                return {"verdict": "wrong", "source": "carried", "answer": self.answer[k],
                        "why": "someone said the right answer is " + json.dumps(self.answer[k], ensure_ascii=False)}
        return None

    def known_answer(self, name: str, inputs: Any) -> Tuple[bool, Any]:
        k = (name, _key(inputs))
        return (k in self.answer), self.answer.get(k)


# ------------------------------------------------------------------ the snapshot

def _is_ours(call: Dict[str, Any], project: Project) -> bool:
    p = call.get("program") or {}
    if p.get("name") != project.name:
        return False
    return p.get("module") == project.module_name or p.get("file") == str(project.file)


def _studio(call: Dict[str, Any]) -> Dict[str, Any]:
    return (call.get("caller") or {}).get("studio") or {}


def _who(call: Dict[str, Any]) -> str:
    caller = call.get("caller") or {}
    st = caller.get("studio") or {}
    if st.get("from") == "dashboard":
        return "you, in the dashboard"
    kind = caller.get("kind")
    if kind == "agent":
        return "a coding agent"
    if kind == "notebook":
        return "a notebook"
    if st:
        return "the command line"
    return {"script": "a script", "api": "the served program", "test": "a test"}.get(kind, "a program")


def git_changed_lines(project: Project) -> Tuple[Set[int], bool]:
    """Lines of the file changed since the last commit (when the folder is a git repository)."""
    try:
        out = subprocess.run(["git", "diff", "-U0", "HEAD", "--", project.file.name], cwd=project.root,
                             capture_output=True, text=True, timeout=5)
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", project.file.name], cwd=project.root,
                                 capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return set(), False
    if tracked.returncode != 0:
        return set(), False
    lines: Set[int] = set()
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            plus = line.split("+", 1)[1].split(" ", 1)[0]
            start, _, count = plus.partition(",")
            n = int(count) if count else 1
            lines |= set(range(int(start), int(start) + n))
    return lines, True


# ------------------------------------------------------------------ versions kept, to go back to

def version_file(project: Project, version: str) -> Path:
    return project.studio_dir / "versions" / (version.split(":", 1)[-1][:24] + ".py")


def keep_version_code(project: Project, version: str) -> None:
    """Keep the file's text the first time a version is seen, so a person can go back to it.
    (A model change is not a new version: the first text of a version is the one kept.)"""
    path = version_file(project, version)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(project.file.read_text(encoding="utf-8"), encoding="utf-8")


# ------------------------------------------------------------------ who pays

def accounts(project: Project, steps: Dict[str, Any]) -> List[Dict[str, Any]]:
    """For each connection the steps go through: which steps, whether it can run here, who pays
    (never a key: only whether one is connected)."""
    from . import connect
    default = project.default_lm()
    ready = connect.usable()
    by: Dict[str, Dict[str, Any]] = {}
    for name, step in steps.items():
        model = getattr(step, "lm", None) or default
        if not isinstance(model, str):
            model = getattr(model, "model", None) or str(model)
        st = connect.model_status(model, ready)
        key = st["provider"] or model
        entry = by.setdefault(key, {"company": st["label"], "provider": st["provider"], "steps": [],
                                    "found": st["ready"], "how": st.get("how"), "pays": st.get("pays"),
                                    "local": st.get("local", False), "subscription": st.get("subscription", False)})
        entry["steps"].append(name)
    return list(by.values())


def snapshot(project: Project, *, examples_file: Optional[Path] = None) -> Dict[str, Any]:
    """Everything the dashboard and the commands show, as plain data."""
    program = project.load()
    settings = project.settings()
    default_lm = project.default_lm()
    info = calllog.program_info(program)
    current = info["version"]
    steps = project.steps()
    graph = build_map(project)
    calls, ratings = read_log(project)
    by_id = {c["id"]: c for c in calls}
    kids: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in calls:
        if c.get("parent"):
            kids[c["parent"]].append(c)
    roots = sorted((c for c in calls if not c.get("parent") and _is_ours(c, project)), key=lambda c: c["started"])

    # versions, named by first appearance
    order: List[str] = []
    for r in roots:
        v = r["program"].get("version")
        if v and v not in order:
            order.append(v)
    if current not in order:
        order.append(current)
    label = {v: i + 1 for i, v in enumerate(order)}

    judged = Judgements(calls, ratings)
    step_shapes = {}
    for name, s in steps.items():
        outs = (getattr(s, "interface", {}) or {}).get("outputs") or [{}]
        step_shapes[name] = outs[-1].get("shape") or {}
    end_shape = ((program.interface.get("outputs") or [{}])[-1]).get("shape") or {}

    examples = project.examples(examples_file)
    by_key = {e["key"]: e for e in examples}

    def main_runs(rs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in rs if not _studio(r).get("trial")]

    latest: Dict[str, Dict[str, Any]] = {}
    older: Dict[str, Dict[str, Any]] = {}
    for r in main_runs(roots):
        k = _studio(r).get("key") or (example_key(r["inputs"]) if isinstance(r.get("inputs"), dict) else None)
        if not k:
            continue
        if r["program"].get("version") == current:
            latest[k] = r
        else:
            older[k] = r

    def step_cells(root: Dict[str, Any]) -> Dict[str, Any]:
        cells: Dict[str, Any] = {}
        for c in kids.get(root["id"], []):
            name = c["program"]["name"]
            if name not in steps:
                continue
            j = judged.of(c, exact=_exact(step_shapes.get(name, {})))
            cents, _ = call_cents(c)
            for g in kids.get(c["id"], []):
                cents += call_cents(g)[0]
            cell = {"call": c["id"], "value": _answer(c), "error": (c.get("error") or {}).get("message"),
                    "error_type": (c.get("error") or {}).get("type"), "model": c.get("model") or
                    next((e.get("model") for e in c.get("exchanges") or []), None),
                    "seconds": c.get("seconds"), "cents": cents, "judgement": j,
                    "cached": bool(c.get("exchanges")) and all(e.get("cached") for e in c["exchanges"])}
            if name in cells:
                cells[name].setdefault("more", 0)
                cells[name]["more"] += 1
            else:
                cells[name] = cell
        return cells

    def tree_cents(root: Dict[str, Any], *, as_if_new: bool = False) -> Tuple[float, Set[str]]:
        total, unknown = call_cents(root, as_if_new=as_if_new)
        stack = list(kids.get(root["id"], []))
        while stack:
            c = stack.pop()
            t, u = call_cents(c, as_if_new=as_if_new)
            total += t
            unknown |= u
            stack.extend(kids.get(c["id"], []))
        return total, unknown

    def make_row(n: Any, key: str, inputs: Any, root: Optional[Dict[str, Any]], *, source: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {"n": n, "key": key, "inputs": inputs, "held": project.held_back(key),
                               "source": source, "ran": root is not None}
        if root is None:
            row.update(state="not run", steps={}, end=None)
            return row
        v = root["program"].get("version")
        cells = step_cells(root)
        err = root.get("error")
        end_j = judged.of(root, exact=_exact(end_shape)) if not err else None
        if end_j is None and not err:
            bad = next((name for name, c in cells.items() if (c.get("judgement") or {}).get("verdict") == "wrong"),
                       None)
            if bad:
                end_j = {"verdict": "wrong", "source": "step", "because": bad,
                         "why": f"{bad} was judged wrong on the way"}
        failed = bool(err) or any(c.get("error") for c in cells.values())
        state = "failed" if failed else (end_j or {}).get("verdict") or "not checked"
        cents, unknown = tree_cents(root)
        new_cents, _ = tree_cents(root, as_if_new=True)
        row.update(call=root["id"], version=label.get(v), current=(v == current), started=root["started"],
                   seconds=root.get("seconds"), state=state, steps=cells,
                   end={"value": _answer(root), "error": (err or {}).get("message"),
                        "error_type": (err or {}).get("type"), "judgement": end_j},
                   cents=cents, cents_new=new_cents, unknown_prices=sorted(unknown), who=_who(root))
        return row

    rows: List[Dict[str, Any]] = []
    for e in examples:
        root = latest.get(e["key"]) or older.get(e["key"])
        rows.append(make_row(e["n"], e["key"], e["inputs"], root, source="example"))
    # runs of this program that are not examples: real use, an agent's tries
    extra = [r for k, r in latest.items() if k not in by_key]
    for i, r in enumerate(sorted(extra, key=lambda r: r["started"])[-200:], 1):
        rows.append(make_row(f"u{i}", _studio(r).get("key") or example_key(r["inputs"]), r.get("inputs"), r,
                             source="use"))

    # each version: its score on its own latest answers, and whether its code is kept to go back to
    keep_version_code(project, current)
    by_version: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in main_runs(roots):
        k = _studio(r).get("key") or (example_key(r["inputs"]) if isinstance(r.get("inputs"), dict) else None)
        if k and k in by_key:
            by_version[r["program"].get("version")][k] = r
    versions = []
    for v in order:
        tally: Dict[str, int] = defaultdict(int)
        for k, r in by_version.get(v, {}).items():
            tally[make_row(0, k, r.get("inputs"), r, source="example")["state"]] += 1
        first = next((r["started"] for r in roots if r["program"].get("version") == v), None)
        versions.append({"label": label[v], "hash": v, "current": v == current, "first_run": first,
                         "ran": sum(tally.values()), "right": tally["right"], "wrong": tally["wrong"],
                         "failed": tally["failed"], "unchecked": tally["not checked"],
                         "code_kept": version_file(project, v).exists()})

    # scores, over the current version's rows
    cur_rows = [r for r in rows if r.get("current")]
    count = defaultdict(int)
    for r in cur_rows:
        count[r["state"]] += 1
    checked = count["right"] + count["wrong"]
    lo, hi = wilson(count["right"], checked)
    step_scores = {}
    for name in steps:
        k = n = 0
        for r in cur_rows:
            j = (r["steps"].get(name) or {}).get("judgement")
            if j and j.get("verdict") in ("right", "wrong"):
                n += 1
                k += j["verdict"] == "right"
        slo, shi = wilson(k, n)
        step_scores[name] = {"right": k, "checked": n, "low": slo, "high": shi}

    # money, by what it was spent on
    spent_by: Dict[str, float] = defaultdict(float)
    for c in calls:
        st = _studio(c)
        kind = ("comparisons" if st.get("run") == "compare" else "trials of other models" if st.get("trial")
                else "runs of the examples" if st else "use outside Studio")
        spent_by[kind] += call_cents(c)[0]
    for sub, kind in (("whatif", "what-ifs (finding the step at fault)"), ("compare", "comparisons")):
        folder = project.studio_dir / sub
        if folder.exists():
            spent_by[kind] += sum(call_cents(c)[0] for c in calllog.read(folder)[0])
    spent = sum(spent_by.values())
    per = [r["cents_new"] for r in cur_rows if r.get("ran") and not r.get("unknown_prices")]
    per_message = sum(per) / len(per) if per else None
    unknown = sorted({m for r in rows for m in r.get("unknown_prices", [])})

    # who did what, newest first
    activity: List[Dict[str, Any]] = []
    seen_v: Set[str] = set()
    runs: Dict[str, Dict[str, Any]] = {}
    for r in roots:
        v = r["program"].get("version")
        if v not in seen_v:
            seen_v.add(v)
            activity.append({"at": r["started"], "what": f"Version {label.get(v)} first ran", "who": _who(r)})
        st = _studio(r)
        rid = st.get("run") or r["id"]
        a = runs.setdefault(rid, {"at": r["started"], "n": 0, "who": _who(r), "trial": st.get("trial")})
        a["n"] += 1
    for a in runs.values():
        what = f"ran {a['n']} example{'s' if a['n'] != 1 else ''}"
        if a.get("trial"):
            what += f" with {a['trial'].get('lm')} for {a['trial'].get('step')} (a trial)"
        activity.append({"at": a["at"], "what": what, "who": a["who"]})
    names = set(steps) | {project.name}
    for r in ratings:
        c = by_id.get(r.get("call"))
        if c and c["program"]["name"] in names and r.get("verdict"):
            activity.append({"at": r.get("at"), "who": r.get("by") or "this computer's account",
                             "what": f"judged {c['program']['name']} {r['verdict']}"
                                     + (f" (via {r['via']})" if r.get("via") else "")})
    activity.sort(key=lambda a: a.get("at") or "", reverse=True)

    changed, tracked = git_changed_lines(project)
    from . import connect
    connections_now = connect.connections()
    catalogue_now = connect.catalogue({c["provider"]: c for c in connections_now if c["status"] == "ready"})
    step_info = {}
    for n in graph["nodes"]:
        if n["kind"] == "step":
            lm = n.get("own_lm")
            step_info[n["id"]] = {"instruction": n.get("instruction"), "signature": n.get("signature"),
                                  "gives": n.get("gives"), "takes": n.get("takes"),
                                  "choices": n.get("choices"), "own_lm": lm, "lm": lm or default_lm,
                                  "def_line": n.get("def_line"), "is_module": n.get("is_module")}
    return {
        "program": {"name": project.name, "file": project.file.name, "path": str(project.file),
                    "description": (program.interface.get("description") or "").strip(),
                    "inputs": project.input_names(), "answer": graph.get("answer"),
                    "version": label[current], "version_hash": current,
                    "versions": versions},
        "graph": graph,
        "steps": step_info,
        "rows": rows,
        "examples_file": str(project.examples_path().relative_to(project.root)),
        "scores": {"end": {"right": count["right"], "wrong": count["wrong"], "failed": count["failed"],
                           "unchecked": count["not checked"], "not_run": sum(1 for r in rows if not r["ran"]),
                           "checked": checked, "low": lo, "high": hi, "rows": len(cur_rows)},
                   "steps": step_scores},
        "money": {"spent_cents": spent, "spent_by": dict(spent_by), "accounts": accounts(project, steps),

                  "per_message_cents": per_message, "unknown_prices": unknown,
                  "prices_date": PRICES["date"], "prices_source": PRICES["source"]},
        "settings": {**{k: settings[k] for k in ("lm", "budget_dollars", "cap_dollars", "monthly", "held_back")},
                     "default_lm": default_lm},
        "connections": connections_now,
        "catalogue": catalogue_now,
        "person": project.person(),
        "models": {"suggested": PRICES["suggested"], "priced": sorted(PRICES["models"]),
                   "prices": PRICES["models"]},
        "activity": activity[:40],
        "code": {"text": project.file.read_text(encoding="utf-8"), "changed": sorted(changed),
                 "tracked": tracked},
    }


# ------------------------------------------------------------------ one example, everything the models saw

def _parts_text(parts: Any) -> Tuple[str, str]:
    """(text, reasoning) of a message's parts; pictures and files as a short placeholder."""
    text, reasoning = [], []
    for p in parts or []:
        kind = p.get("type")
        if kind == "text":
            text.append(p.get("text") or "")
        elif kind == "thinking":
            if p.get("text"):
                reasoning.append(p["text"])
        elif kind in ("image", "audio", "video", "document", "file"):
            text.append(f"[{kind}{': ' + p['media_type'] if p.get('media_type') else ''}]")
        elif kind in ("tool_call", "tool_result"):
            text.append(f"[{kind}: {json.dumps(p.get('input') or p.get('output') or p.get('name'), ensure_ascii=False)[:300]}]")
    return "".join(text), "\n".join(reasoning)


def _call_view(c: Dict[str, Any], judged: "Judgements", exact: bool) -> Dict[str, Any]:
    exchanges = c.get("exchanges") or []
    last = next((e for e in reversed(exchanges) if e.get("response")), exchanges[-1] if exchanges else {})
    req = last.get("request") or {}
    sent = {"system": req.get("system") or "",
            "messages": [{"role": m.get("role"), "text": _parts_text(m.get("parts"))[0]} for m in req.get("messages") or []]}
    reply_text, reasoning = _parts_text(((last.get("response") or {}).get("message") or {}).get("parts"))
    cents, _ = call_cents(c)
    return {"call": c["id"], "name": c["program"]["name"], "answer": (c.get("program") or {}).get("answer") or "result",
            "inputs": c.get("inputs") or {}, "outputs": c.get("outputs"), "value": _answer(c),
            "error": (c.get("error") or {}).get("message"), "model": c.get("model") or last.get("model"),
            "seconds": c.get("seconds"), "cents": cents, "usage": c.get("usage") or {},
            "cached": bool(exchanges) and all(e.get("cached") for e in exchanges), "attempts": len(exchanges),
            "judgement": judged.of(c, exact=exact) if c.get("outputs") is not None else None,
            "sent": sent if exchanges else None, "reply": {"text": reply_text, "reasoning": reasoning} if exchanges else None}


def example_detail(project: Project, n: Any) -> Dict[str, Any]:
    """Everything about one example's latest run: what came in, and for each step what it was given,
    what was sent to the model, what came back, what it answered, and people's judgements."""
    snap = snapshot(project)
    row = next((r for r in snap["rows"] if str(r["n"]) == str(n)), None)
    if row is None:
        raise StudioError(f"no example #{n}")
    out = {"n": row["n"], "inputs": row["inputs"], "state": row["state"], "ran": row.get("ran"),
           "held": row.get("held"), "current": row.get("current"), "version": row.get("version"), "steps": [],
           "end": None}
    if not row.get("ran"):
        return out
    calls, ratings = read_log(project)
    judged = Judgements(calls, ratings)
    kids: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in calls:
        if c.get("parent"):
            kids[c["parent"]].append(c)
    root = next(c for c in calls if c["id"] == row["call"])
    program = project.load()
    end_shape = ((program.interface.get("outputs") or [{}])[-1]).get("shape") or {}
    shapes = {}
    for name, s in project.steps().items():
        outs = (getattr(s, "interface", {}) or {}).get("outputs") or [{}]
        shapes[name] = outs[-1].get("shape") or {}
    steps: List[Dict[str, Any]] = []
    stack = sorted(kids.get(root["id"], []), key=lambda c: c["started"])
    while stack:
        c = stack.pop(0)
        v = _call_view(c, judged, _exact(shapes.get(c["program"]["name"], {})))
        v["instruction"] = (snap["steps"].get(v["name"]) or {}).get("instruction")
        v["choices"] = (snap["steps"].get(v["name"]) or {}).get("choices")
        v["depth"] = 0 if c.get("parent") == root["id"] else 1
        steps.append(v)
        stack = sorted(kids.get(c["id"], []), key=lambda c: c["started"]) + stack
    out["steps"] = steps
    end = _call_view(root, judged, _exact(end_shape))
    end["judgement"] = row["end"]["judgement"]
    end["choices"] = end_shape.get("enum")
    out["end"] = end
    return out
