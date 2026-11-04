"""A Studio project: one program file, its examples, and the folders Studio keeps.

``find_project("club.py")``, ``find_project("club.py:club_inbox")`` or
``find_project()`` (the one program file in the current folder) gives a
``Project``. Loading runs the file the way ``functai serve file.py:name`` does:
as a module named after the file, with the file's folder as the working folder,
so its relative paths (``open("club_faq.md")``) work.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import importlib.util
import json
import os
import secrets
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..calllog import canonical
from ..core import FunctAIFunc
from ..module import FunctAIModule

# The model a step uses when neither its code nor the person's settings name one.
# Checked against the providers' own model lists on 2026-11-02: the newest small,
# fast model that answered through FunctAI here (OpenAI's gpt-6-luna costs the same).
DEFAULT_LM = "claude-haiku-5-5"

_LOAD_LOCK = threading.RLock()


class StudioError(Exception):
    """Something Studio cannot do, said so that a person or an agent can act on it."""


def _is_program(obj: Any) -> bool:
    return isinstance(obj, (FunctAIModule, FunctAIFunc))


def _candidate_files(folder: Path) -> List[Path]:
    out = []
    for p in sorted(folder.glob("*.py")):
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "functai" in text and ("@module" in text or "@ai" in text or "functai.ai" in text):
            out.append(p)
    return out


def find_project(target: Optional[str] = None, *, folder: Optional[Path] = None) -> "Project":
    """The project a command works on. ``target``: ``file.py`` or ``file.py:name``;
    None: the only file in ``folder`` (default: the current one) that defines a
    FunctAI program."""
    folder = Path(folder or os.getcwd()).resolve()
    name = None
    if target:
        where, sep, rest = target.rpartition(":")
        if sep and where.endswith(".py"):
            target, name = where, rest
        path = (folder / target).resolve()
        if path.is_dir():
            return find_project(None, folder=path)
        if not path.exists():
            raise StudioError(f"{target}: no such file")
        return Project(path, name)
    files = _candidate_files(folder)
    if not files:
        raise StudioError(f"no FunctAI program in {folder}: name the file (functai studio open club.py), "
                          f"or start from the example (functai studio example club)")
    if len(files) > 1:
        raise StudioError("several files define FunctAI programs here: " + ", ".join(p.name for p in files)
                          + ". Name one: functai studio open " + files[0].name)
    return Project(files[0], None)


class Project:
    """One program file and what Studio keeps beside it."""

    def __init__(self, file: Path, name: Optional[str] = None, *, module_name: Optional[str] = None,
                 root: Optional[Path] = None):
        self.file = Path(file).resolve()
        self.root = Path(root).resolve() if root else self.file.parent
        self.module_name = module_name or self.file.stem
        self.requested = name
        self.name: Optional[str] = name
        self.module: Any = None
        self.program: Any = None
        self.load_error: Optional[str] = None
        self._mtime: Optional[float] = None
        self._steps: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- folders

    @property
    def functai_dir(self) -> Path:
        return self.root / ".functai"

    @property
    def calls_dir(self) -> Path:
        return self.functai_dir / "calls"

    @property
    def studio_dir(self) -> Path:
        return self.functai_dir / "studio"

    @property
    def replies_path(self) -> Path:
        return self.studio_dir / "replies.sqlite"

    def ensure_dirs(self) -> None:
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.studio_dir.mkdir(parents=True, exist_ok=True)
        ignore = self.functai_dir / ".gitignore"
        if not ignore.exists():
            # runs hold what people typed; they stay out of git unless someone decides otherwise
            ignore.write_text("# FunctAI Studio: runs, judgements and caches of this project.\n*\n")

    # ---------------------------------------------------------------- settings

    @property
    def settings_path(self) -> Path:
        return self.studio_dir / "settings.json"

    def settings(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        try:
            data = json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            pass
        changed = False
        if "salt" not in data:
            data["salt"] = secrets.token_hex(8)
            changed = True
        data.setdefault("lm", DEFAULT_LM)
        data.setdefault("budget_dollars", 0.50)
        data.setdefault("cap_dollars", 1.00)
        data.setdefault("monthly", 300)
        data.setdefault("held_back", 0.4)
        if changed:
            self.save_settings(data)
        return data

    def save_settings(self, data: Dict[str, Any]) -> None:
        self.ensure_dirs()
        tmp = self.settings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(self.settings_path)

    def update_settings(self, **changes: Any) -> Dict[str, Any]:
        data = self.settings()
        data.update(changes)
        self.save_settings(data)
        return data

    def default_lm(self) -> str:
        """The model for steps whose code names none: the person's choice (Models) when it can run
        here, else the first small model of an account connected on this computer."""
        from .connect import default_model
        return default_model(self.settings().get("lm")) or self.settings().get("lm") or DEFAULT_LM

    def person(self) -> str:
        """Who judges in the dashboard: the settings' name, else this computer's account name."""
        return self.settings().get("by") or getpass.getuser()

    # ---------------------------------------------------------------- loading

    def stale(self) -> bool:
        try:
            return self.file.stat().st_mtime != self._mtime
        except OSError:
            return True

    def load(self, *, force: bool = False) -> Any:
        """Import the file (again when it changed) and pick the program. Raises
        StudioError with the reason when the file cannot be loaded."""
        with _LOAD_LOCK:
            if self.program is not None and not force and not self.stale():
                return self.program
            self._mtime = self.file.stat().st_mtime
            old_cwd = os.getcwd()
            sys.modules.pop(self.module_name, None)
            if str(self.root) not in sys.path:
                sys.path.insert(0, str(self.root))
            try:
                os.chdir(self.root)
                spec = importlib.util.spec_from_file_location(self.module_name, self.file)
                if spec is None or spec.loader is None:
                    raise StudioError(f"cannot read {self.file.name}")
                mod = importlib.util.module_from_spec(spec)
                sys.modules[self.module_name] = mod
                spec.loader.exec_module(mod)
            except StudioError:
                raise
            except BaseException as exc:  # noqa: BLE001 — the person's code: say what went wrong, keep going
                self.load_error = f"{type(exc).__name__}: {exc}"
                self.program = None
                raise StudioError(f"{self.file.name} did not load: {self.load_error}") from exc
            finally:
                os.chdir(old_cwd)
            self.module = mod
            self._steps = None
            self.program = self._pick(mod)
            self.name = self.program.__name__
            self.load_error = None
            return self.program

    def _pick(self, mod: Any) -> Any:
        own = {k: v for k, v in vars(mod).items() if _is_program(v) and _defined_in(v, mod)}
        if self.requested:
            if self.requested not in own:
                raise StudioError(f"{self.file.name} has no FunctAI program named {self.requested}; "
                                  f"it has: {', '.join(own) or 'none'}")
            return own[self.requested]
        modules = {k: v for k, v in own.items() if isinstance(v, FunctAIModule)}
        if modules:
            used = set()
            for m in modules.values():
                used |= {f for f in _called_names(m) if f in modules}
            top = [k for k in modules if k not in used]
            if len(top) == 1:
                return modules[top[0]]
            raise StudioError(f"{self.file.name} has several pipelines ({', '.join(top)}): name one, "
                              f"e.g. functai studio open {self.file.name}:{top[0]}")
        if len(own) == 1:
            return next(iter(own.values()))
        if not own:
            raise StudioError(f"{self.file.name} defines no FunctAI program (@ai or @module)")
        raise StudioError(f"{self.file.name} has several AI functions and no @module joining them "
                          f"({', '.join(own)}): name one, e.g. {self.file.name}:{next(iter(own))}")

    # ---------------------------------------------------------------- the program

    def steps(self) -> Dict[str, Any]:
        """The AI functions (and inner modules) the program calls, by name, in the code's order;
        a program that is one AI function is its own single step."""
        program = self.load()
        if self._steps is not None:
            return dict(self._steps)
        if isinstance(program, FunctAIFunc):
            self._steps = {program.__name__: program}
            return dict(self._steps)
        out: Dict[str, Any] = {}
        g = getattr(program, "_globals", {}) or {}
        for name in _called_names(program):
            obj = g.get(name)
            if _is_program(obj):
                out[name] = obj
        for name, fn in program.named_ai_functions().items():
            out.setdefault(name, fn)
        # reading which functions a pipeline reaches walks every installed package (about a second):
        # remembered until the file changes (load() reads it again)
        self._steps = out
        return dict(out)

    def interface(self) -> Dict[str, Any]:
        return self.load().interface

    def input_names(self) -> List[str]:
        return [f["name"] for f in self.interface().get("inputs", [])]

    # ---------------------------------------------------------------- examples

    def examples_path(self) -> Path:
        name = self.name or self.requested or self.module_name
        folder = self.root / "examples"
        for ext in (".csv", ".jsonl"):
            p = folder / f"{name}{ext}"
            if p.exists():
                return p
        return folder / f"{name}.csv"

    def examples(self, path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """The examples: each ``{"n", "key", "inputs"}``; ``n`` counts from 1 in the file's order."""
        self.load()
        path = Path(path) if path else self.examples_path()
        if not path.exists():
            return []
        fields = self.interface().get("inputs", [])
        names = [f["name"] for f in fields]
        rows: List[Dict[str, Any]] = []
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        out = []
        for i, row in enumerate(rows, 1):
            missing = [f["name"] for f in fields if f["name"] not in row and not f.get("optional")]
            if missing:
                raise StudioError(f"{path.name}, row {i}: no column {', '.join(missing)} "
                                  f"(the program takes {', '.join(names)})")
            inputs = {f["name"]: _coerce(row[f["name"]], f.get("shape") or {}) for f in fields if f["name"] in row}
            out.append({"n": i, "key": example_key(inputs), "inputs": inputs})
        return out

    def held_back(self, key: str) -> bool:
        """Whether an example is kept aside from whoever writes changes, to test them on.
        Decided by the example's own key and this project's salt: stable, and no list to keep."""
        s = self.settings()
        h = hashlib.sha256((s["salt"] + key).encode()).digest()
        return h[0] / 256 < float(s.get("held_back", 0.4))


def example_key(inputs: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical(inputs).encode()).hexdigest()


def _coerce(value: Any, shape: Dict[str, Any]) -> Any:
    """A CSV cell is text: give it the type the input asks for."""
    if not isinstance(value, str):
        return value
    t = shape.get("type")
    try:
        if t == "integer":
            return int(value)
        if t == "number":
            return float(value)
        if t == "boolean":
            return value.strip().lower() in ("1", "true", "yes", "y")
        if t in ("array", "object"):
            return json.loads(value)
    except ValueError:
        return value
    return value


def _defined_in(obj: Any, mod: Any) -> bool:
    fn = getattr(obj, "_fn", None) or getattr(obj, "__wrapped__", None) or obj
    return getattr(fn, "__module__", None) == mod.__name__


def _called_names(program: Any) -> List[str]:
    """Names called in a module's body, in the order they appear."""
    import ast
    import inspect
    import textwrap
    fn = getattr(program, "_fn", None)
    if fn is None:
        return []
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in names:
            names.append(node.func.id)
    # ast.walk is breadth-first: put them in source order
    order: List[Tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            order.append((node.lineno, node.col_offset, node.func.id))
    seen: List[str] = []
    for _, _, n in sorted(order):
        if n not in seen:
            seen.append(n)
    return seen
