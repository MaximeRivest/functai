"""The command line: ``functai serve``, ``functai describe``, ``functai verify``.

    functai serve team/ --lm gpt-4.1-mini --keys keys.txt --port 8080
    functai serve support.py:support --store conversations/
    functai describe team/
    python -m functai verify team/          # what verify() runs in a fresh environment
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
    s.add_argument("--trust", action="store_true", help="load a saved folder that runs its author's code")
    d = sub.add_parser("describe", help="what a saved program takes and gives, without loading it")
    d.add_argument("folder")
    v = sub.add_parser("verify", help="load a saved folder here and compare its fingerprints")
    v.add_argument("folder")
    args = parser.parse_args(argv)
    if args.command == "describe":
        from .saved import describe
        print(json.dumps(describe(args.folder), indent=2, ensure_ascii=False))
        return 0
    if args.command == "verify":
        from .saved import main as verify_main
        return verify_main(["verify", args.folder])
    from .serving import Service
    program = _program(args.program, trust=args.trust)
    service = Service(program, keys=args.keys, store=args.store, lm=args.lm, approvals=args.approvals)
    print(f"serving {program.__name__} at http://{args.host}:{args.port} (the API at /openapi.json)",
          file=sys.stderr)
    service.serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
