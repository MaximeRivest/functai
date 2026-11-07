"""The command line: ``functai serve``, ``functai describe``, ``functai verify``, ``functai skill``.

    functai serve team/ --lm gpt-4.1-mini --keys keys.txt --port 8080
    functai serve support.py:support --store conversations/
    functai describe team/
    python -m functai verify team/          # what verify() runs in a fresh environment
    functai skill                           # where the coding-agent skill is, and which agents have it
    functai skill --install                 # copy it to every agent found here (Claude Code, Pi, Codex)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, List, Optional


def _program(target: str, *, trust: bool) -> Any:
    """A saved folder, or ``file.py:name`` / ``package.module:name``."""
    path = Path(target)
    if path.is_dir():
        from .saved import load
        return load(path, trust=trust)
    if ":" not in target:
        raise SystemExit(f"functai: {target} is neither a saved folder nor file.py:name")
    where, _, name = target.rpartition(":")
    if where.endswith(".py"):
        spec = importlib.util.spec_from_file_location(Path(where).stem, where)
        if spec is None or spec.loader is None:
            raise SystemExit(f"functai: cannot read {where}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(where)
    try:
        return getattr(mod, name)
    except AttributeError:
        raise SystemExit(f"functai: {where} has no {name}") from None


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "verify":
        from .saved import main as verify_main
        return verify_main(argv)
    parser = argparse.ArgumentParser(prog="functai", description="FunctAI programs, from the command line.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="serve a program over HTTP")
    s.add_argument("program", help="a saved folder, or file.py:name, or package.module:name")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--keys", help="a file of keys, one per line (needed beyond 127.0.0.1)")
    s.add_argument("--lm", help="the model every call uses")
    s.add_argument("--store", help="a folder where conversations are kept (default: memory)")
    s.add_argument("--approvals", choices=("owner", "caller"), default="owner")
    s.add_argument("--view", choices=("outside", "kept", "full"), default="outside",
                   help="what callers see: the program's boundary (default), what your log keeps, or everything "
                        "(thinking, tools, the model's stream, error messages: only for callers you trust)")
    s.add_argument("--trust", action="store_true", help="load a saved folder that runs its author's code")
    d = sub.add_parser("describe", help="what a saved program takes and gives, without loading it")
    d.add_argument("folder")
    v = sub.add_parser("verify", help="load a saved folder here and compare its fingerprints")
    v.add_argument("folder")
    sd = sub.add_parser("setup-diff", help="what differs between what two saved folders ran on, step by step")
    sd.add_argument("before")
    sd.add_argument("after")
    sd.add_argument("--json", action="store_true", help="the differences as JSON")
    k = sub.add_parser("skill", help="the skill that teaches coding agents to use functai: where it is, and install it")
    k.add_argument("--install", nargs="*", metavar="DIR",
                   help="copy it into these skills folders (default: every agent's found here)")
    args = parser.parse_args(argv)
    if args.command == "skill":
        return _skill(args.install)
    if args.command == "describe":
        from .saved import describe
        print(json.dumps(describe(args.folder), indent=2, ensure_ascii=False))
        return 0
    if args.command == "verify":
        from .saved import main as verify_main
        return verify_main(["verify", args.folder])
    if args.command == "setup-diff":
        from .setup import setup_diff
        diff = setup_diff(args.before, args.after)
        if args.json:
            print(json.dumps(diff, indent=2, ensure_ascii=False))
        else:
            for d in diff:
                print(f"{d['program']}: {d['field']}: {json.dumps(d['before'])} -> {json.dumps(d['after'])}")
            if not diff:
                print("the same setup")
        return 0
    from .serving import Service
    program = _program(args.program, trust=args.trust)
    service = Service(program, keys=args.keys, store=args.store, lm=args.lm, approvals=args.approvals,
                      view=args.view)
    shows = "" if args.view == "outside" else f"; callers see the {args.view} view"
    print(f"serving {program.__name__} at http://{args.host}:{args.port} (the API at /openapi.json{shows})",
          file=sys.stderr)
    service.serve(args.host, args.port)
    return 0


SKILL = Path(__file__).resolve().parent / "skill"

# where coding agents look for skills, when the agent is installed here (its home folder exists)
AGENT_SKILLS = {"Claude Code": "~/.claude/skills", "Pi": "~/.pi/agent/skills", "Codex": "~/.codex/skills"}


def _skill(install: Optional[List[str]]) -> int:
    """Say where the skill is and which agents have it; with ``--install``, copy it.
    A copy is replaced only when it is functai's (its SKILL.md says ``name: functai``);
    a link (a checkout's, kept current by itself) is left as it is."""
    import shutil

    from . import __version__
    targets: List[Path] = []
    if install:
        targets = [Path(d).expanduser() for d in install]
    else:
        for agent, folder in AGENT_SKILLS.items():
            path = Path(folder).expanduser()
            if path.parent.exists():
                targets.append(path)
    if install is None:
        print(f"the functai skill (functai {__version__}): {SKILL}")
        for agent, folder in AGENT_SKILLS.items():
            path = Path(folder).expanduser()
            if not path.parent.exists():
                continue
            here = path / "functai"
            state = ("linked" if here.is_symlink() else "installed" if (here / "SKILL.md").exists()
                     else "not installed")
            print(f"  {agent:12} {here}: {state}")
        print("install or update it with: functai skill --install [DIR ...]")
        return 0
    if not targets:
        print("functai: no coding agent found here (Claude Code, Pi, Codex); name a skills folder: "
              "functai skill --install DIR", file=sys.stderr)
        return 1
    for folder in targets:
        here = folder / "functai"
        if here.is_symlink():
            print(f"{here}: a link ({here.resolve()}), left as it is")
            continue
        if here.exists():
            if "name: functai" not in (here / "SKILL.md").read_text(encoding="utf-8", errors="replace") \
                    if (here / "SKILL.md").exists() else True:
                print(f"{here}: holds something else than functai's skill, left as it is", file=sys.stderr)
                continue
            shutil.rmtree(here)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SKILL, here, ignore=shutil.ignore_patterns("__pycache__"))
        print(f"{here}: installed (functai {__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
