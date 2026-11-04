"""``functai studio``: the command line, for coding agents and people alike.

Never asks a question (whatever it needs is a flag); text by default, ``--json``
for programs; long values shortened with their length (``--full`` for all);
every command that asks a model says what it will cost first and stops above the
budget (exit 3). Exit codes: 0 done, 2 something is wrong (the message says what
to do), 3 over budget.

    functai studio                         # the dashboard: connect an AI, make or open a pipeline
    functai studio example club            # a ready project to try
    functai studio map                     # the steps, read from the code
    functai studio run --estimate          # what running the examples would cost
    functai studio run                     # run them
    functai studio show [#5[/kind]]        # what each step said
    functai studio rate #5/kind wrong --answer question --by maxime
    functai studio why #5                  # which step is at fault there
    functai studio blame                   # which step to fix first, over all wrong examples
    functai studio compare --before HEAD   # the change, measured on judged examples kept aside
    functai studio compare --lm kind=claude-sonnet-5-5
    functai studio versions                # each version's score
    functai studio page                    # the one-page summary to share
    functai studio doctor                  # models, keys, the project
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .project import DEFAULT_LM, Project, StudioError, find_project

EXIT_ERROR, EXIT_BUDGET = 2, 3


def _out(args: argparse.Namespace, data: Any, text: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    else:
        print(text)


def _short(value: Any, full: bool = False, n: int = 90) -> str:
    if value is None:
        return "—"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if text == "":
        return '"" (nothing)'
    if full:
        return text
    flat = " ".join(text.split())
    words = len(text.split())
    return flat if len(flat) <= n else f"{flat[: n - 1]}… ({words} words)"


def _split_target(rest: List[str]) -> Tuple[Optional[str], List[str]]:
    if rest and (rest[0].endswith(".py") or ".py:" in rest[0] or Path(rest[0]).is_dir()):
        return rest[0], rest[1:]
    return None, rest


def _overrides(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        step, sep, model = item.partition("=")
        if not sep or not step or not model:
            raise StudioError(f"--lm takes step=model, e.g. --lm kind={DEFAULT_LM} (got {item!r})")
        out[step] = model
    return out


def _project(target: Optional[str]) -> Project:
    p = find_project(target)
    p.load()
    os.chdir(p.root)
    return p


def _addr(text: str) -> Tuple[str, Optional[str]]:
    t = text.lstrip("#")
    n, _, step = t.partition("/")
    if not n:
        raise StudioError(f"which example? write #5 or #5/kind (got {text!r})")
    return n, (step or None)


def _row(snap: Dict[str, Any], n: str) -> Dict[str, Any]:
    row = next((r for r in snap["rows"] if str(r["n"]) == n), None)
    if row is None:
        raise StudioError(f"no example #{n} (there are {len(snap['rows'])}: functai studio show lists them)")
    return row


def _verdict_text(j: Optional[Dict[str, Any]]) -> str:
    if not j:
        return "not checked"
    who = {"person": f"by {j.get('by')}", "carried": "(" + (j.get("why") or "judged before") + ")",
           "step": f"(because {j.get('because')} was judged wrong)"}.get(j.get("source"), "")
    return f"{j['verdict']} {who}".strip()


# ------------------------------------------------------------------ commands

def cmd_map(args: argparse.Namespace, rest: List[str]) -> int:
    from .graph import build_map
    p = _project(_split_target(rest)[0])
    g = build_map(p)
    lines = [f"{p.name} ({p.file.name}): " + ", ".join(f"{n['label']}" for n in g["nodes"] if n["kind"] == "input")
             + " → " + g["answer"]]
    nodes = {n["id"]: n for n in g["nodes"]}
    for n in sorted((n for n in g["nodes"] if n["kind"] in ("step", "value")), key=lambda n: n.get("line") or 0):
        ins = [nodes[e["from"]]["label"] for e in g["edges"] if e["to"] == n["id"]]
        to = ["end result" if e["to"] == "out" else nodes[e["to"]]["label"] for e in g["edges"] if e["from"] == n["id"]]
        when = f"  if {n['when']}" if n.get("when") else ""
        if n["kind"] == "step":
            lines.append(f"  {n['label']}: {n.get('signature', '')}{when}{'  (in a loop)' if n.get('loop') else ''}")
        else:
            lines.append(f"  fixed value {n['label']}{when}")
        lines.append(f"      from {', '.join(ins) or '—'}   to {', '.join(to) or '—'}")
    for note in g["notes"]:
        lines.append(f"  note: {note}")
    _out(args, g, "\n".join(lines))
    return 0


def cmd_run(args: argparse.Namespace, rest: List[str]) -> int:
    from . import running
    on = Path(args.on).resolve() if args.on else None
    p = _project(_split_target(rest)[0])
    examples = p.examples(on)
    if not examples:
        raise StudioError(f"no examples: write {p.examples_path().relative_to(p.root)} (a CSV with a column per "
                          f"input: {', '.join(p.input_names())}), or give --on FILE")
    if args.only:
        want = {s.strip().lstrip("#") for s in args.only.split(",")}
        examples = [e for e in examples if str(e["n"]) in want]
    if args.limit:
        examples = examples[: args.limit]
    ov = _overrides(args.lm)
    est = running.estimate(p, examples, ov, threads=args.threads)
    est_text = (f"{len(examples)} example{'s' if len(examples) != 1 else ''}, up to {est['requests']} model requests, "
                f"about {running.fmt_cents(est['cents'])}, about {est['seconds']:.0f} s (from {est['basis']}; "
                f"{est['note']})")
    if est["unknown_prices"]:
        est_text += f". No price known for {', '.join(est['unknown_prices'])}"
    if args.estimate:
        _out(args, est, est_text)
        return 0
    budget = args.budget if args.budget is not None else p.settings()["budget_dollars"]
    try:
        running.check_budget(est, budget)
    except running.OverBudget as exc:
        _out(args, {"over_budget": True, "estimate": est, "budget_dollars": budget}, f"Not run: {exc}")
        return EXIT_BUDGET
    if not args.json:
        print(f"Running {est_text}.", file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if not args.json:
            print(f"\r  {done}/{total}", end="", file=sys.stderr, flush=True)

    res = running.run_examples(p, examples, overrides=ov, threads=args.threads, progress=progress)
    if not args.json:
        print(file=sys.stderr)
    from .data import snapshot
    snap = snapshot(p)
    end = snap["scores"]["end"]
    rows = [r for r in snap["rows"] if r.get("source") == "example" and r["n"] in {e["n"] for e in examples}]
    spent = sum(r.get("cents") or 0 for r in rows)
    lines = [f"Ran {res['examples']} example{'s' if res['examples'] != 1 else ''} in {res['seconds']:.1f} s"
             f"{' with ' + ', '.join(f'{k}={v}' for k, v in ov.items()) if ov else ''}: "
             f"{res['failed']} failed. Cost of these answers: {running.fmt_cents(spent)} (replies reused from the cache are free)."]
    if ov:
        lines.append("This was a trial: the code did not change. Compare it: functai studio compare --lm "
                     + " ".join(f"{k}={v}" for k, v in ov.items()))
    for r in rows:
        if r["state"] == "failed":
            err = r["end"].get("error") or next((c.get("error") for c in r["steps"].values() if c.get("error")), "")
            lines.append(f"  #{r['n']} failed: {_short(err)}")
    lines.append(f"Version {snap['program']['version']}: {end['right']} right, {end['wrong']} wrong, {end['failed']} "
                 f"failed, {end['unchecked']} not checked yet.")
    if end["unchecked"]:
        lines.append("Only a person can say which answers are right: open the dashboard (functai studio open) "
                     "and judge them, or relay what the person said with functai studio rate.")
    _out(args, {"run": res, "scores": end, "cents": spent}, "\n".join(lines))
    return 0


def cmd_show(args: argparse.Namespace, rest: List[str]) -> int:
    from .data import snapshot
    target, rest = _split_target(rest)
    p = _project(target)
    snap = snapshot(p)
    if rest:
        n, step = _addr(rest[0])
        row = _row(snap, n)
        if args.json:
            _out(args, row if not step else row["steps"].get(step), "")
            return 0
        lines = [f"#{row['n']}{' (kept aside to test changes)' if row.get('held') else ''}: {row['state']}"]
        lines += [f"  in   {k}: {_short(v, args.full)}" for k, v in row["inputs"].items()]
        for name, c in row.get("steps", {}).items():
            if step and name != step:
                continue
            said = f"failed: {c['error']}" if c.get("error") else _short(c.get("value"), args.full)
            lines.append(f"  {name}: {said}")
            lines.append(f"      {_verdict_text(c.get('judgement'))} · {c.get('model')} · "
                         f"{(c.get('seconds') or 0):.1f} s{' · from the cache' if c.get('cached') else ''}")
            j = c.get("judgement") or {}
            if j.get("verdict") == "wrong" and j.get("answer") is not None:
                lines.append(f"      should have said: {_short(j['answer'], args.full)}")
        if not step:
            end = row.get("end") or {}
            lines.append(f"  end result: {_short(end.get('value'), args.full) if not end.get('error') else 'failed: ' + end['error']}")
            lines.append(f"      {_verdict_text(end.get('judgement'))}")
        if not row.get("ran"):
            lines.append("  not run yet: functai studio run")
        print("\n".join(lines))
        return 0
    rows = snap["rows"]
    if args.wrong:
        rows = [r for r in rows if r["state"] == "wrong"]
    if args.failed:
        rows = [r for r in rows if r["state"] == "failed"]
    if args.unchecked:
        rows = [r for r in rows if r["state"] == "not checked"]
    if args.json:
        _out(args, rows, "")
        return 0
    end = snap["scores"]["end"]
    lines = [f"{snap['program']['name']} version {snap['program']['version']}: {end['right']} right, {end['wrong']} "
             f"wrong, {end['failed']} failed, {end['unchecked']} not checked, {end['not_run']} not run"]
    for r in rows:
        flags = (" (older version)" if r.get("ran") and not r.get("current") else "") + (" ·aside" if r.get("held") else "")
        first = next(iter(r["inputs"].values()), "")
        lines.append(f"#{r['n']:<4} {r['state']:<12} {_short(first, n=48):<50} → "
                     f"{_short((r.get('end') or {}).get('value'), n=48)}{flags}")
    print("\n".join(lines))
    return 0


def cmd_rate(args: argparse.Namespace, rest: List[str]) -> int:
    import functai
    from .data import snapshot
    target, rest = _split_target(rest)
    if len(rest) != 2 or rest[1] not in ("right", "wrong", "withdraw"):
        raise StudioError("functai studio rate #5[/step] right|wrong|withdraw [--answer VALUE] --by NAME")
    if not args.by:
        raise StudioError("--by NAME is needed: only people judge, and a judgement names who made it. If a person "
                          "told you this in a conversation, give their name; never rate on your own opinion")
    p = _project(target)
    snap = snapshot(p)
    n, step = _addr(rest[0])
    row = _row(snap, n)
    if not row.get("ran"):
        raise StudioError(f"#{n} has not run yet: functai studio run --only {n}")
    call = (row["steps"].get(step) or {}).get("call") if step else row.get("call")
    if not call:
        raise StudioError(f"#{n} has no answer from {step} (steps that ran: {', '.join(row['steps']) or 'none'})")
    answer: Any = None
    if args.answer is not None:
        try:
            answer = json.loads(args.answer)
        except ValueError:
            answer = args.answer
    verdict = None if rest[1] == "withdraw" else rest[1]
    kw: Dict[str, Any] = {"by": args.by, "folder": p.calls_dir}
    if args.note:
        kw["note"] = args.note
    if answer is not None:
        kw["answer"] = answer
    functai.rate(call, verdict, **kw)
    what = f"#{n}{'/' + step if step else ''}"
    _out(args, {"call": call, "verdict": verdict, "answer": answer, "by": args.by},
         f"Recorded: {what} {verdict or 'withdrawn'} by {args.by}"
         + (f", should have said {_short(answer)}" if answer is not None else ""))
    return 0


def _guard(args: argparse.Namespace, p: Project, est: Dict[str, Any], what: str) -> bool:
    """Say the cost; stop above the budget. True when it may go on."""
    from . import running
    budget = args.budget if args.budget is not None else p.settings()["budget_dollars"]
    if est["cents"] > budget * 100:
        _out(args, {"over_budget": True, "estimate": est, "budget_dollars": budget},
             f"Not run: {what} would cost about {running.fmt_cents(est['cents'])} ({est['requests']} model "
             f"requests at most), over the budget of ${budget:.2f}; add --budget DOLLARS to allow it")
        return False
    if not args.json and est["requests"]:
        print(f"{what}: up to {est['requests']} model requests, about {running.fmt_cents(est['cents'])}.",
              file=sys.stderr)
    return True


def cmd_why(args: argparse.Namespace, rest: List[str]) -> int:
    from . import running
    from .data import snapshot
    target, rest = _split_target(rest)
    if not rest:
        raise StudioError("functai studio why #5")
    p = _project(target)
    n = _addr(rest[0])[0]
    if not _guard(args, p, running.estimate_why(p, _row(snapshot(p), n)), "Finding the step at fault"):
        return EXIT_BUDGET
    res = running.why(p, n)
    lines = [res["say"]]
    for t in res.get("tried", []):
        if "given" in t:
            lines.append(f"  given {t['step']} = {_short(t['given'], args.full)}")
            lines.append(f"    the end became: {_short(t.get('end'), args.full) if not t.get('error') else 'failed: ' + t['error']}"
                         f"  ({t['result']})")
    if any(t.get("result") == "someone has to look" for t in res.get("tried", [])):
        lines.append("  A person judges that new end result in the dashboard (Which step is at fault?).")
    _out(args, res, "\n".join(lines))
    return 0


def cmd_blame(args: argparse.Namespace, rest: List[str]) -> int:
    from . import running
    from .data import snapshot
    p = _project(_split_target(rest)[0])
    if not _guard(args, p, running.estimate_blame(p, snapshot(p)), "Finding the step to fix first"):
        return EXIT_BUDGET
    res = running.blame(p)
    (p.studio_dir / "blame.json").write_text(json.dumps(res, default=str, indent=1))
    lines = [res["say"]]
    for step, k in res["counts"].items():
        lines.append(f"  {step}: {k} of {res['wrong']}")
    _out(args, res, "\n".join(lines))
    return 0


def cmd_compare(args: argparse.Namespace, rest: List[str]) -> int:
    from . import running
    from .data import snapshot
    p = _project(_split_target(rest)[0])
    ov = _overrides(args.lm)
    est = running.estimate_compare(p, snapshot(p), before=args.before, overrides=ov, use_all=args.all)
    if not _guard(args, p, est, "This comparison"):
        return EXIT_BUDGET
    res = running.compare(p, before=args.before, overrides=ov, threads=args.threads, use_all=args.all)
    lines = [res["say"]]
    for r in res.get("rows", []):
        if r["before"] != r["after"]:
            lines.append(f"  #{r['n']}: {r['before']} → {r['after']}")
    _out(args, res, "\n".join(lines))
    return 0


def cmd_versions(args: argparse.Namespace, rest: List[str]) -> int:
    from .data import snapshot
    p = _project(_split_target(rest)[0])
    vs = snapshot(p)["program"]["versions"]
    lines = []
    for v in reversed(vs):
        score = (f"{v['right']} right, {v['wrong']} wrong, {v['failed']} failed, {v['unchecked']} not checked "
                 f"of {v['ran']} run" if v["ran"] else "not run")
        lines.append(f"Version {v['label']}{' (now)' if v['current'] else ''}: {score}"
                     f"{'' if v['code_kept'] else ' · code not kept'}")
    lines.append("Going back to a version: the dashboard's version list, or git.")
    _out(args, vs, "\n".join(lines))
    return 0


def cmd_page(args: argparse.Namespace, rest: List[str]) -> int:
    from .data import snapshot
    from .page import summary_html
    out = Path(args.output).resolve() if args.output else None
    p = _project(_split_target(rest)[0])
    blame = None
    try:
        blame = json.loads((p.studio_dir / "blame.json").read_text())
    except (OSError, ValueError):
        pass
    html = summary_html(snapshot(p), blame=blame)
    out = out or (p.root / f"{p.name}-summary.html")
    out.write_text(html, encoding="utf-8")
    _out(args, {"file": str(out.resolve())}, f"Wrote {out} (one file: opens in any browser, prints to PDF).")
    return 0


def cmd_open(args: argparse.Namespace, rest: List[str]) -> int:
    from .server import serve
    target = _split_target(rest)[0]
    folder = Path.cwd()
    project = None
    try:
        project = find_project(target)
        project.load()
    except StudioError:
        if target:
            raise
        print("Starting Studio here, with no pipeline open yet: the page below connects an AI, makes a pipeline, "
              "or opens one from this folder.", file=sys.stderr)
    serve(project, host=args.host, port=args.port, folder=folder, open_browser=not args.no_browser)
    return 0


def cmd_doctor(args: argparse.Namespace, rest: List[str]) -> int:
    import functai
    from .data import PRICES
    keys = {"Anthropic": "ANTHROPIC_API_KEY", "OpenAI": "OPENAI_API_KEY", "Google": "GEMINI_API_KEY",
            "xAI": "XAI_API_KEY", "OpenRouter": "OPENROUTER_API_KEY"}
    lines = [f"functai {functai.__version__}, Python {sys.version.split()[0]}"]
    found = [who for who, var in keys.items() if os.environ.get(var)]
    lines.append("model keys found: " + (", ".join(found) or "none (set ANTHROPIC_API_KEY, OPENAI_API_KEY, …)"))
    data: Dict[str, Any] = {"keys": found}
    try:
        p = _project(_split_target(rest)[0])
        s = p.settings()
        ex = p.examples()
        lines.append(f"project: {p.file} → {p.name}; {len(ex)} examples in {p.examples_path().relative_to(p.root)}")
        lines.append(f"default model for steps that name none: {s['lm']} (prices of {PRICES['date']})")
        lines.append(f"runs and judgements: {p.calls_dir}")
        data.update(project=str(p.file), program=p.name, examples=len(ex), lm=p.default_lm())
    except StudioError as exc:
        lines.append(f"project: {exc}")
    if args.live:
        try:
            p = find_project(_split_target(rest)[0])
            lm = p.default_lm()
        except StudioError:
            lm = DEFAULT_LM

        @functai.ai
        def echo(word: str) -> str:
            """Repeat the word."""
        try:
            with functai.configure(lm=lm):
                echo("ok")
            lines.append(f"live check: {lm} answered")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"live check: {lm} failed: {type(exc).__name__}: {exc}")
    _out(args, data, "\n".join(lines))
    return 0


EXAMPLES = Path(__file__).parent / "examples"


def cmd_example(args: argparse.Namespace, rest: List[str]) -> int:
    name = rest[0] if rest else "club"
    src = EXAMPLES / name
    if not src.is_dir():
        raise StudioError(f"no example {name!r}; there are: {', '.join(p.name for p in EXAMPLES.iterdir() if p.is_dir())}")
    dest = Path(rest[1] if len(rest) > 1 else f"{name}-inbox" if name == "club" else name).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise StudioError(f"{dest} exists and is not empty")
    shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".functai"))
    _out(args, {"folder": str(dest)},
         f"Made {dest}.\nNext: cd {dest.name} && functai studio run && functai studio open")
    return 0


COMMANDS = {"map": cmd_map, "run": cmd_run, "show": cmd_show, "rate": cmd_rate, "why": cmd_why, "blame": cmd_blame,
            "compare": cmd_compare, "versions": cmd_versions, "page": cmd_page, "open": cmd_open, "doctor": cmd_doctor, "example": cmd_example}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="functai studio",
                                 description="FunctAI Studio: see, judge, fix and share a pipeline. Never asks a "
                                             "question; --json on any command; exit 3 when a run would cost more "
                                             "than the budget.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n\n")[2:]))
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("rest", nargs="*", help="[FILE.py[:name]] and what the command takes")
        sp.add_argument("--json", action="store_true", help="answer as JSON")
        return sp

    add("map", "the steps and how they connect, read from the code")
    r = add("run", "run the examples (cost first)")
    r.add_argument("--on", help="another examples file (CSV or JSON lines)")
    r.add_argument("--only", help="these examples: 1,4,7")
    r.add_argument("--limit", type=int, help="the first N examples")
    r.add_argument("--lm", action="append", metavar="STEP=MODEL", help="ask a step another model (a trial)")
    r.add_argument("--estimate", action="store_true", help="only say what it would cost")
    r.add_argument("--budget", type=float, metavar="DOLLARS", help="stop when it would cost more (default: the project's)")
    r.add_argument("--threads", type=int, default=4)
    s = add("show", "what each step said, row by row")
    s.add_argument("--wrong", action="store_true")
    s.add_argument("--failed", action="store_true")
    s.add_argument("--unchecked", action="store_true")
    s.add_argument("--full", action="store_true", help="whole values, not shortened")
    t = add("rate", "record a person's judgement: #5[/step] right|wrong|withdraw")
    t.add_argument("--answer", help="what it should have said (JSON or text)")
    t.add_argument("--by", help="the person who judged (required)")
    t.add_argument("--note")
    w = add("why", "which step is at fault for one wrong example (#5)")
    w.add_argument("--full", action="store_true")
    w.add_argument("--budget", type=float, metavar="DOLLARS")
    bl = add("blame", "which step to fix first, over every wrong example")
    bl.add_argument("--budget", type=float, metavar="DOLLARS")
    c = add("compare", "measure a change on the judged examples kept aside")
    c.add_argument("--before", help="HEAD (or another git revision), or a file")
    c.add_argument("--lm", action="append", metavar="STEP=MODEL", help="compare against another model for a step")
    c.add_argument("--all", action="store_true", help="use every judged example, not only those kept aside")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--budget", type=float, metavar="DOLLARS")
    add("versions", "each version's score, newest first")
    pg = add("page", "write the one-page summary (HTML)")
    pg.add_argument("-o", "--output")
    o = add("open", "start the dashboard")
    o.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (this computer); another address needs the key it prints")
    o.add_argument("--port", type=int, default=7781)
    o.add_argument("--no-browser", action="store_true")
    d = add("doctor", "models, keys and the project")
    d.add_argument("--live", action="store_true", help="also ask the default model one tiny question (a fraction of a cent)")
    add("example", "make a ready project to try: functai studio example club [FOLDER]")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("--") and argv[0] not in ("--help", "-h"):
        argv = ["open", *argv]          # `functai studio` alone opens the dashboard
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args, list(args.rest))
    except StudioError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"functai studio: {exc}", file=sys.stderr)
        return EXIT_ERROR
