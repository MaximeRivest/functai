"""Running things: examples, trials of another model for one step, what-if
experiments, finding the weak step, and comparing two versions.

Every run is an ordinary FunctAI call whose ``caller`` says which example it
answered (``{"studio": {"run", "example", "key"}}``), written to the project's
call log; replies already asked are reused from the project's reply cache, free.
What-if runs and the "before" side of a comparison are kept in their own
folders under ``.functai/studio`` so they never count as the pipeline's answers.
"""

from __future__ import annotations

import contextvars
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import functai

from .. import calllog
from ..config import effective
from .data import Judgements, _exact, _key, price_of, read_log, snapshot
from .project import Project, StudioError

_GIVEN_LOCK = threading.Lock()


class OverBudget(StudioError):
    def __init__(self, estimate: Dict[str, Any], budget_cents: float):
        self.estimate = estimate
        self.budget_cents = budget_cents
        super().__init__(f"this would cost about {fmt_cents(estimate['cents'])}, over the budget of "
                         f"{fmt_cents(budget_cents)}; run again with --budget {max(0.01, estimate['cents'] / 100 * 1.5):.2f} "
                         f"(dollars) to allow it")


def fmt_cents(c: Optional[float]) -> str:
    if c is None:
        return "unknown"
    if c == 0:
        return "0¢"
    if c < 0.1:
        return f"{c:.3f}".rstrip("0").rstrip(".") + "¢"
    if c < 1:
        return f"{c:.2f}".rstrip("0").rstrip(".") + "¢"
    if c < 100:
        return f"{c:.1f}".rstrip("0").rstrip(".") + "¢"
    return f"${c / 100:,.2f}"


def models_plugin(overrides: Dict[str, str]) -> Optional[functai.Plugin]:
    """A plugin that asks a step's question of another model (recorded on each call it changes)."""
    if not overrides:
        return None
    p = functai.Plugin("studio-models", version="1")

    @p.before_call
    def pick(call):  # noqa: ANN001
        lm = overrides.get(call.function)
        return functai.Change(lm=lm) if lm else None
    return p


def _settings_for(project: Project, *, folder: Path, caller: Dict[str, Any],
                  overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    s: Dict[str, Any] = {"log_calls": str(folder), "caller": {"studio": caller},
                         "cache_replies": str(project.replies_path)}
    if not effective().get("lm"):
        s["lm"] = project.default_lm()
    plug = models_plugin(overrides or {})
    if plug is not None:
        s["plugins"] = [plug]
    return s


# ------------------------------------------------------------------ estimates

def estimate(project: Project, examples: List[Dict[str, Any]], overrides: Optional[Dict[str, str]] = None,
             *, threads: int = 4) -> Dict[str, Any]:
    """About how many model requests, how much money and how long, before running.
    From earlier runs of each step when there are some, else a rough guess from the text sizes."""
    overrides = overrides or {}
    steps = project.steps()
    calls, _ = read_log(project)
    default_lm = effective().get("lm") or project.default_lm()
    per_step: Dict[str, Tuple[float, float, float]] = {}
    for name in steps:
        mine = [c for c in calls if c["program"]["name"] == name and c.get("usage")]
        if mine:
            tin = sum(c["usage"].get("input_tokens", 0) for c in mine) / len(mine)
            tout = sum(c["usage"].get("output_tokens", 0) for c in mine) / len(mine)
            sec = sum(c.get("seconds") or 0 for c in mine) / len(mine)
            per_step[name] = (tin, tout, sec)
    basis = "earlier runs" if len(per_step) == len(steps) else "a rough guess (no runs yet)"
    cents = 0.0
    seconds = 0.0
    unknown = set()
    for ex in examples:
        size = sum(len(str(v)) for v in ex["inputs"].values())
        for name, step in steps.items():
            lm = overrides.get(name) or getattr(step, "lm", None) or default_lm
            tin, tout, sec = per_step.get(name, ((len((step.__doc__ or "")) + size + 900) / 4, 60, 2.0))
            p = price_of(lm)
            if p is None:
                unknown.add(lm)
            else:
                cents += (tin * p["input"] + tout * p["output"]) / 1e6 * 100
            seconds += sec
    return {"examples": len(examples), "requests": len(examples) * len(steps), "cents": cents,
            "seconds": seconds / max(1, threads), "basis": basis, "unknown_prices": sorted(unknown),
            "note": "at most: a step that does not run (a branch not taken) costs nothing, and replies already "
                    "asked come back free from the cache"}


def step_costs(project: Project, overrides: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """About what one call of each step costs (cents; None when its model has no known price), from
    the tokens its earlier calls used, else a rough guess. Steps in the order the code calls them."""
    overrides = overrides or {}
    steps = project.steps()
    calls, _ = read_log(project)
    default_lm = effective().get("lm") or project.default_lm()
    examples = project.examples()
    size = (sum(sum(len(str(v)) for v in e["inputs"].values()) for e in examples) / len(examples)) if examples else 200
    out: Dict[str, Dict[str, Any]] = {}
    for name, step in steps.items():
        lm = overrides.get(name) or getattr(step, "lm", None) or default_lm
        mine = [c for c in calls if c["program"]["name"] == name and c.get("usage")]
        if mine:
            tin = sum(c["usage"].get("input_tokens", 0) for c in mine) / len(mine)
            tout = sum(c["usage"].get("output_tokens", 0) for c in mine) / len(mine)
        else:
            tin, tout = (len(step.__doc__ or "") + size + 900) / 4, 60
        p = price_of(lm)
        out[name] = {"lm": lm, "cents": None if p is None else (tin * p["input"] + tout * p["output"]) / 1e6 * 100}
    return out


def _sum(costs: Dict[str, Dict[str, Any]], names: List[str], times: int = 1) -> Dict[str, Any]:
    cents, unknown = 0.0, set()
    for n in names:
        c = costs[n]["cents"]
        if c is None:
            unknown.add(costs[n]["lm"])
        else:
            cents += c * times
    return {"cents": cents, "requests": len(names) * times, "unknown_prices": sorted(unknown)}


def _add(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {"cents": a["cents"] + b["cents"], "requests": a["requests"] + b["requests"],
            "unknown_prices": sorted(set(a["unknown_prices"]) | set(b["unknown_prices"]))}


_NOTHING_TO_RUN = {"cents": 0.0, "requests": 0, "unknown_prices": []}


def estimate_why(project: Project, row: Dict[str, Any], costs: Optional[Dict[str, Dict[str, Any]]] = None
                 ) -> Dict[str, Any]:
    """What ``why`` costs on one example: the steps after each step judged wrong with its right answer
    (all of them, at most: a given answer may take another path)."""
    costs = costs or step_costs(project)
    order = list(costs)
    total = dict(_NOTHING_TO_RUN)
    for i, name in enumerate(order):
        j = ((row.get("steps") or {}).get(name) or {}).get("judgement") or {}
        if j.get("verdict") == "wrong" and j.get("answer") is not None:
            total = _add(total, _sum(costs, order[i + 1:]))
    return total


def estimate_blame(project: Project, snap: Dict[str, Any]) -> Dict[str, Any]:
    costs = step_costs(project)
    total = dict(_NOTHING_TO_RUN)
    for r in snap["rows"]:
        if r.get("current") and r["state"] == "wrong":
            total = _add(total, estimate_why(project, r, costs))
    return total


def _compare_pool(project: Project, use_all: bool) -> List[Dict[str, Any]]:
    calls, ratings = read_log(project)
    judged = Judgements(calls, ratings)
    known = [e for e in project.examples() if judged.known_answer(project.name, e["inputs"])[0]
             or judged.right.get((project.name, _key(e["inputs"]))) or judged.wrong.get((project.name, _key(e["inputs"])))]
    return known if use_all else [e for e in known if project.held_back(e["key"])]


def estimate_compare(project: Project, snap: Dict[str, Any], *, before: Optional[str] = None,
                     overrides: Optional[Dict[str, str]] = None, use_all: bool = False) -> Dict[str, Any]:
    """What ``compare`` costs, at most: the side that runs the current code is free for examples it
    already answered (the cache); a step on another model, and everything after it, is asked again."""
    project.load()
    pool = _compare_pool(project, use_all)
    now = step_costs(project)
    order = list(now)
    ran = {r["key"] for r in snap["rows"] if r.get("current")}
    total = dict(_NOTHING_TO_RUN)
    for e in pool:
        if before or e["key"] not in ran:
            total = _add(total, _sum(now, order))                 # the side that is not already cached
        if overrides:
            first = min(order.index(s) for s in overrides if s in order) if any(s in order for s in overrides) else 0
            total = _add(total, _sum(step_costs(project, overrides), order[first:]))
    total["examples"] = len(pool)
    return total


def check_budget(est: Dict[str, Any], budget_dollars: Optional[float]) -> None:
    if budget_dollars is not None and est["cents"] > budget_dollars * 100:
        raise OverBudget(est, budget_dollars * 100)


# ------------------------------------------------------------------ running examples

def run_examples(project: Project, examples: List[Dict[str, Any]], *, overrides: Optional[Dict[str, str]] = None,
                 threads: int = 4, origin: str = "command line",
                 progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """Run each example through the program; failures are recorded, never raised."""
    program = project.load()
    project.ensure_dirs()
    run_id = uuid.uuid4().hex[:12]
    trial = None
    if overrides:
        trial = {"step": ",".join(sorted(overrides)), "lm": ",".join(overrides[k] for k in sorted(overrides))}
    done = [0]
    lock = threading.Lock()
    results: List[Dict[str, Any]] = [None] * len(examples)  # type: ignore[list-item]
    started = time.time()

    def one(i: int, ex: Dict[str, Any]) -> None:
        caller = {"run": run_id, "example": ex["n"], "key": ex["key"], "from": origin}
        if trial:
            caller["trial"] = trial
        try:
            with functai.configure(**_settings_for(project, folder=project.calls_dir, caller=caller,
                                                   overrides=overrides)):
                value = program(**ex["inputs"])
            results[i] = {"n": ex["n"], "ok": True, "value": value}
        except Exception as exc:  # noqa: BLE001 — a failed example is a result, not a crash
            results[i] = {"n": ex["n"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        with lock:
            done[0] += 1
            if progress:
                progress(done[0], len(examples))

    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futs = [pool.submit(contextvars.copy_context().run, one, i, ex) for i, ex in enumerate(examples)]
        for f in futs:
            f.result()
    return {"run": run_id, "examples": len(examples), "failed": sum(1 for r in results if not r["ok"]),
            "seconds": time.time() - started, "results": results, "trial": trial}


# ------------------------------------------------------------------ what if a step had said something else

def given(project: Project, inputs: Dict[str, Any], answers: Dict[str, Any], *,
          program: Any = None, folder: Optional[Path] = None) -> Dict[str, Any]:
    """Run the program with some steps' answers given instead of asked, the rest for real.
    Returns what every step said (given or asked) and the end result. Recorded in
    ``.functai/studio/whatif``, never in the project's call log."""
    program = program or project.load()
    if not hasattr(program, "_globals"):
        raise StudioError("a what-if needs a pipeline of several steps (@module)")
    g = program._globals
    missing = [n for n in answers if n not in g]
    if missing:
        raise StudioError(f"no step named {', '.join(missing)}")
    seen: Dict[str, Any] = {}

    def give(name: str, value: Any):
        def stand_in(*a: Any, **k: Any) -> Any:
            seen[name] = value
            return value
        return stand_in

    def watch(name: str, fn: Any):
        def watched(*a: Any, **k: Any) -> Any:
            out = fn(*a, **k)
            seen.setdefault(name, out)
            return out
        return watched

    folder = folder or (project.studio_dir / "whatif")
    folder.mkdir(parents=True, exist_ok=True)
    with _GIVEN_LOCK:
        originals = {n: g[n] for n in project.steps() if n in g}
        try:
            for n, fn in originals.items():
                g[n] = give(n, answers[n]) if n in answers else watch(n, fn)
            with functai.configure(**_settings_for(project, folder=folder,
                                                   caller={"whatif": {k: v for k, v in answers.items()}})):
                try:
                    end, error = program(**inputs), None
                except Exception as exc:  # noqa: BLE001
                    end, error = None, f"{type(exc).__name__}: {exc}"
        finally:
            g.update(originals)
            # FunctAI remembers which code a pipeline reaches, re-reading it only when a name it bound
            # changes; the stand-ins bound none, so forget what was read while they stood in
            calllog._module_graphs.pop(program, None)
    return {"given": answers, "steps": seen, "end": end, "error": error}


def _whatif_path(project: Project) -> Path:
    return project.studio_dir / "whatif-judgements.jsonl"


def judge_whatif(project: Project, inputs: Any, end: Any, verdict: str, *, by: str) -> None:
    """A person's judgement of an end result a what-if produced. Kept in Studio's own folder:
    it judges a what could have been, never a call of the pipeline."""
    import datetime
    import json
    if verdict not in ("right", "wrong"):
        raise StudioError("a what-if result is judged right or wrong")
    project.ensure_dirs()
    line = {"inputs": _key(inputs), "end": _key(end), "verdict": verdict, "by": by,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with _whatif_path(project).open("a") as f:
        f.write(json.dumps(line) + "\n")


def _whatif_verdicts(project: Project) -> Dict[Tuple[str, str], str]:
    import json
    out: Dict[Tuple[str, str], str] = {}
    try:
        for line in _whatif_path(project).read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[(d["inputs"], d["end"])] = d["verdict"]
    except OSError:
        pass
    return out


def _end_verdict(project: Project, judged: Judgements, inputs: Any, value: Any) -> Optional[str]:
    """right / wrong when people's judgements settle it, else None (someone has to look)."""
    program = project.load()
    name = project.name
    k = (name, _key(inputs))
    out = _key(value)
    w = _whatif_verdicts(project).get((k[1], out))
    if w:
        return w
    if out in judged.right.get(k, ()):
        return "right"
    if out in judged.wrong.get(k, ()):
        return "wrong"
    known, answer = judged.known_answer(name, inputs)
    if known:
        if _key(answer) == out:
            return "right"
        shape = ((program.interface.get("outputs") or [{}])[-1]).get("shape") or {}
        if _exact(shape):
            return "wrong"
    return None


def why(project: Project, n: Any, *, snap: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """For one example whose end is wrong: give each step judged wrong its right answer,
    in the code's order, and see whether the end comes out right."""
    snap = snap or snapshot(project)
    row = next((r for r in snap["rows"] if str(r["n"]) == str(n)), None)
    if row is None:
        raise StudioError(f"no example #{n}")
    if not row.get("ran"):
        return {"n": n, "verdict": "not run", "say": f"#{n} has not run yet: functai studio run first"}
    order = [nd["id"] for nd in snap["graph"]["nodes"] if nd["kind"] == "step"]
    calls, ratings = read_log(project)
    judged = Judgements(calls, ratings)
    tried = []
    for name in order:
        cell = row["steps"].get(name)
        j = (cell or {}).get("judgement") or {}
        if j.get("verdict") != "wrong":
            continue
        answer = j.get("answer")
        if answer is None:
            tried.append({"step": name, "result": "needs the right answer",
                          "say": f"“{name}” was judged wrong on #{n}, but nobody said what it should have said: "
                                 f"give its right answer (functai studio rate #{n}/{name} wrong --answer …) to try a what-if"})
            continue
        out = given(project, row["inputs"], {name: answer})
        verdict = _end_verdict(project, judged, row["inputs"], out["end"]) if not out["error"] else "failed"
        tried.append({"step": name, "given": answer, "end": out["end"], "steps": out["steps"],
                      "error": out["error"], "result": verdict or "someone has to look"})
        if verdict == "right":
            break
    if not tried:
        if row["state"] == "right":
            return {"n": n, "verdict": "right", "tried": [], "say": f"#{n} came out right."}
        return {"n": n, "verdict": row["state"], "tried": [],
                "say": f"#{n} is {row['state']}, and no step on its way was judged wrong: judge the steps "
                       f"(functai studio show {project.file.name} #{n}) to find which one is at fault"}
    blamed = next((t["step"] for t in tried if t["result"] == "right"), None)
    if blamed:
        say = (f"#{n}: when “{blamed}” was given the right answer ({_short(tried[-1]['given'])}), the end came out "
               f"right. So “{blamed}” is at fault here.")
    else:
        parts = []
        for t in tried:
            if "given" not in t:
                parts.append(t["say"])
            elif t["result"] in ("wrong", "failed"):
                parts.append(f"when “{t['step']}” was given its right answer, the end was still {t['result']}, so "
                             f"another step is at fault too")
            else:
                parts.append(f"when “{t['step']}” was given its right answer, the end became a new answer nobody "
                             f"has judged yet: a person has to say whether it is right")
        say = f"#{n}: " + "; ".join(parts)
    return {"n": n, "verdict": row["state"], "tried": tried, "blamed": blamed, "say": say}


def blame(project: Project, progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """``why`` over every wrong example: which step to fix first."""
    snap = snapshot(project)
    wrong = [r for r in snap["rows"] if r.get("current") and r["state"] == "wrong"]
    counts: Dict[str, int] = {}
    results = []
    for i, r in enumerate(wrong, 1):
        res = why(project, r["n"], snap=snap)
        results.append(res)
        if res.get("blamed"):
            counts[res["blamed"]] = counts.get(res["blamed"], 0) + 1
        if progress:
            progress(i, len(wrong))
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    look = [res["n"] for res in results if any(t.get("result") == "someone has to look" for t in res["tried"])]
    stuck = [res["n"] for res in results if not res["tried"]
             or all(t.get("result") == "needs the right answer" for t in res["tried"])]
    s_ = "s" if len(wrong) != 1 else ""
    if not wrong:
        say = "no example is judged wrong in the current version"
    elif ranked:
        top, k = ranked[0]
        say = f"Fix “{top}” first: given its right answer, {k} of the {len(wrong)} wrong example{s_} came out right"
    else:
        say = f"{len(wrong)} wrong example{s_}, and none came out right yet in a what-if"
    if look:
        say += (f". The what-if gave new answers nobody has judged for {', '.join('#' + str(x) for x in look)}: "
                f"judge them (functai studio why … shows them) and run blame again")
    if stuck:
        say += (f". For {', '.join('#' + str(x) for x in stuck)} no step was judged wrong with what it should "
                f"have said, so no what-if could run")
    return {"wrong": len(wrong), "counts": dict(ranked), "results": results, "say": say, "to_judge": look,
            "stuck": stuck}


# ------------------------------------------------------------------ comparing two versions

def load_before(project: Project, before: str) -> Project:
    """The program as it was: ``HEAD`` (or any git revision), or another file."""
    path = Path(before)
    if not path.exists():
        try:
            rel = project.file.relative_to(_git_root(project))
        except (ValueError, StudioError):
            raise StudioError(f"{before}: not a file, and {project.file.name} is not in a git repository") from None
        out = subprocess.run(["git", "show", f"{before}:{rel.as_posix()}"], cwd=project.root,
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise StudioError(f"git has no {project.file.name} at {before}: {out.stderr.strip()}")
        folder = project.studio_dir / "compare"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{project.module_name}__before.py"
        path.write_text(out.stdout)
    other = Project(path, project.name, module_name=f"{project.module_name}__before", root=project.root)
    other.file = path.resolve()
    other.load()
    return other


def _git_root(project: Project) -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=project.root, capture_output=True, text=True)
    if out.returncode != 0:
        raise StudioError("not a git repository")
    return Path(out.stdout.strip())


def compare(project: Project, *, before: Optional[str] = None, overrides: Optional[Dict[str, str]] = None,
            threads: int = 4, use_all: bool = False) -> Dict[str, Any]:
    """The current code against an earlier one (``before``), or against itself with another
    model for some steps (``overrides``), paired, on the judged examples kept aside."""
    from ..evaluation import _paired_t
    if not before and not overrides:
        raise StudioError("compare with what? --before HEAD (or a file), or --lm step=model")
    project.load()
    calls, ratings = read_log(project)
    judged = Judgements(calls, ratings)
    pool = _compare_pool(project, use_all)
    if not pool:
        return {"n": 0, "say": ("no judged example is kept aside yet: judge the end result of more examples "
                                "(about 4 in 10 are kept aside), or use --all to compare on every judged one, "
                                "knowing whoever wrote the change may have seen them")}
    if before:
        old = load_before(project, before)
        a = [given_or_run(old, e, folder=project.studio_dir / "compare") for e in pool]
        a_label = f"before ({before})"
    else:
        a = [given_or_run(project, e, folder=project.calls_dir) for e in pool]
        a_label = "now"
    b = [given_or_run(project, e, folder=project.calls_dir, overrides=overrides) for e in pool]
    b_label = "with " + ", ".join(f"{k}={v}" for k, v in (overrides or {}).items()) if overrides else "now"
    rows = []
    for e, x, y in zip(pool, a, b, strict=True):
        vx = _end_verdict(project, judged, e["inputs"], x) if x is not _FAILED else "failed"
        vy = _end_verdict(project, judged, e["inputs"], y) if y is not _FAILED else "failed"
        rows.append({"n": e["n"], "before": vx, "after": vy,
                     "before_value": None if x is _FAILED else x, "after_value": None if y is _FAILED else y})
    settled = [r for r in rows if r["before"] in ("right", "wrong", "failed") and r["after"] in ("right", "wrong", "failed")]
    d = [(r["after"] == "right") - (r["before"] == "right") for r in settled]
    better = sum(v > 0 for v in d)
    worse = sum(v < 0 for v in d)
    br = sum(r["before"] == "right" for r in settled)
    ar = sum(r["after"] == "right" for r in settled)
    unsettled = len(rows) - len(settled)
    low = high = None
    if len(d) >= 2 and any(d):
        _, low, high = _paired_t(d)
    where = "judged examples kept aside" if not use_all else "judged examples (whoever wrote the change may have seen them)"
    say = f"Tested on {len(settled)} {where}: {br} right {a_label}, {ar} right {b_label}."
    if better or worse:
        say += f" {better} got better, {worse} got worse."
    if len(settled) < 10:
        say += f" {len(settled)} is too few to be sure: judge more examples for a reliable answer."
    elif low is not None and low > 0:
        say += " That is a real improvement, not luck."
    elif high is not None and high < 0:
        say += " That is really worse, not luck."
    else:
        say += " The difference could be luck."
    if unsettled:
        say += (f" {unsettled} more gave answers nobody has judged yet (text that changed): a person judges them "
                f"in the dashboard's comparison, then compare again.")
    return {"n": len(rows), "settled": len(settled), "before_right": br, "after_right": ar, "better": better,
            "worse": worse, "low": low, "high": high, "rows": rows, "say": say, "held_back": not use_all}


class _Failed:
    pass


_FAILED = _Failed()


def given_or_run(project: Project, ex: Dict[str, Any], *, folder: Path,
                 overrides: Optional[Dict[str, str]] = None) -> Any:
    program = project.load()
    caller = {"run": "compare", "example": ex["n"], "key": ex["key"], "from": "compare"}
    if overrides:
        caller["trial"] = {"step": ",".join(sorted(overrides)), "lm": ",".join(overrides[k] for k in sorted(overrides))}
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with functai.configure(**_settings_for(project, folder=folder, caller=caller, overrides=overrides)):
            return program(**ex["inputs"])
    except Exception:  # noqa: BLE001
        return _FAILED


def _short(value: Any, n: int = 60) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"

