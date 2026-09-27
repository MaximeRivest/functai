"""What a functai program depends on, found by reading its code.

``functai.check(program)`` follows every name the program's code reaches (the
bodies of AI functions and ``@module`` functions, their tools, the plain
functions and classes they use, the types in their signatures, the decorator
arguments and default values) and sorts each one:

- another AI function or ``@module``: a node, followed the same way
- a function or class from your code (a notebook, a script, a module that is
  not an installed package): a node; its source is saved with the program
- a module or name from an installed package: a pinned requirement
- the standard library and builtins: nothing to record
- a constant (numbers, strings, lists and dicts of them, Enum members,
  compiled patterns, lmcc adapters): saved by value
- anything else, or a global the code changes: a problem, with its fix

It reads code, never runs it. What it cannot see (names looked up at run time
with ``getattr``/``importlib``/``eval``, functions passed in as arguments, data
files) is reported where it shows, and ``functai.verify`` catches the rest by
loading the saved program in a fresh environment.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import dis
import enum
import importlib.metadata as md
import inspect
import json
import re
import sys
import textwrap
import types
import typing
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import lmcc

from .core import FunctAIFunc, _AISentinel
from .docments import class_source as _class_source
from .module import FunctAIModule

# ------------------------------------------------------------------ problems


@dataclasses.dataclass(frozen=True)
class Problem:
    """One thing that keeps a program from being saved cleanly.

    ``code`` is stable (match on it, or pass it to ``save(allow=...)``);
    ``where`` names the offender as ``module:name``; ``fix`` says what to do."""
    code: str
    where: str
    message: str
    fix: str
    severity: str = "error"          # "error" blocks save; "warning" is reported

    def __str__(self) -> str:
        mark = "✗" if self.severity == "error" else "!"
        return f"{mark} {self.code}  {self.where}: {self.message}\n    fix: {self.fix}"


class Refused(Exception):
    """``save`` found errors; ``.report`` has them all."""

    def __init__(self, report: "Report"):
        self.report = report
        errors = "\n".join(str(p) for p in report.errors)
        super().__init__(f"the program cannot be saved cleanly:\n{errors}\n"
                         f"(save(..., allow=[code, ...]) records a deliberate exception)")


# ------------------------------------------------------------------ where modules come from

_STDLIB = frozenset(sys.stdlib_module_names) | {"builtins", "__future__"}
_RUNTIME = ("functai", "lmcc", "lmcc_std", "lmcc_lm15", "lm15")


def _distributions() -> Dict[str, List[str]]:
    return md.packages_distributions()


class Origins:
    """Where each module comes from: ``local`` (saved as code), ``stdlib``, or
    ``package`` (a requirement, with its distribution)."""

    def __init__(self, include: Iterable[str] = ()):
        self.include = tuple(include)
        self._dists = _distributions()

    def of(self, module: str) -> str:
        top = module.split(".")[0]
        if any(module == p or module.startswith(p + ".") for p in self.include):
            return "local"
        if top in _STDLIB:
            return "stdlib"
        if self.distribution(module):
            return "package"
        return "local"

    def distribution(self, module: str) -> Optional[str]:
        top = module.split(".")[0]
        names = self._dists.get(top)
        if names:
            return names[0]
        # Editable installs (hatchling, uv) leave no file list for the lookup above;
        # a distribution named like the module is the one that provides it.
        try:
            return md.distribution(top.replace("_", "-")).metadata["Name"]
        except md.PackageNotFoundError:
            try:
                return md.distribution(top).metadata["Name"]
            except md.PackageNotFoundError:
                return None


# ------------------------------------------------------------------ the graph


@dataclasses.dataclass
class Binding:
    """A name a saved module file must define, and how.

    kind: ``def`` (a node defined in this file, ``node`` its key), ``local``
    (a name from another saved file: ``module``, ``attr``), ``localmod`` (another
    saved file as a module), ``import`` (``stmt``), ``value`` (``expr``, the
    Python expression that rebuilds it, and ``needs`` the names it uses)."""
    name: str
    kind: str
    obj_id: int
    node: Optional[str] = None
    module: Optional[str] = None
    attr: Optional[str] = None
    stmt: Optional[str] = None
    expr: Optional[str] = None
    needs: Tuple[str, ...] = ()
    deftime: bool = False


@dataclasses.dataclass
class Node:
    key: str                          # "module:name"
    kind: str                         # "ai" | "module" | "function" | "class"
    module: str
    name: str
    obj: Any = dataclasses.field(repr=False, default=None)
    source: str = ""                  # dedented; decorators stripped for ai/module
    deftime_names: Set[str] = dataclasses.field(default_factory=set)
    uses: List[str] = dataclasses.field(default_factory=list)     # node keys, first-use order
    tools: List[Any] = dataclasses.field(default_factory=list)    # node keys or import refs or tool data
    teacher: Optional[str] = None
    escalate: Optional[str] = None                                # node key of an escalate_to AI function
    baked: Dict[str, Any] = dataclasses.field(default_factory=dict)   # setting ("lm", "escalate_to") → Baked
    requires: Tuple[str, ...] = ()
    future_annotations: bool = False
    files: List[str] = dataclasses.field(default_factory=list)    # functai.file("...") arguments


@dataclasses.dataclass
class Requirement:
    distribution: str
    version: str
    spec: str                         # the line for requirements.txt
    local_path: Optional[str] = None  # installed from a folder on this machine
    reason: str = ""


class Report:
    """What ``functai.check`` found: the graph, the requirements, the problems."""

    def __init__(self, entry: str, nodes: Dict[str, Node], bindings: Dict[str, Dict[str, Binding]],
                 requirements: Dict[str, Requirement], problems: List[Problem], packages: Dict[str, str],
                 declared: List[str]):
        self.entry = entry
        self.nodes = nodes
        self.bindings = bindings              # saved module → name → Binding
        self.requirements = requirements      # distribution → Requirement
        self.problems = problems
        self.packages = packages              # module → distribution, for every package reached
        self.declared = declared              # requirements declared by hand (requires=)

    @property
    def errors(self) -> List[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> List[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    @property
    def local_modules(self) -> List[str]:
        return list(self.bindings)

    # ---- the tree

    def _label(self, node: Node) -> str:
        if node.kind in ("ai",):
            try:
                spec = node.obj._spec()
            except Exception as exc:  # noqa: BLE001 — the report still shows the rest
                return f"AI function (signature error: {exc})"
            ins = ", ".join(f"{f.name}: {f.type or 'Any'}" for f in spec.signature.inputs if f.purpose == "plain")
            main = next(f for f in spec.signature.outputs if f.name == spec.main)
            on = node.baked.get("lm")
            where = f" on {on.model} ({on.student}, {on.size() / 1e6:,.0f} MB)" if on is not None else ""
            return f"AI function ({ins} → {main.type or 'Any'}){where}"
        if node.kind == "module":
            return "@module"
        return node.kind

    def tree(self) -> str:
        lines: List[str] = []
        shown: Set[str] = set()

        def leafs(node: Node) -> List[str]:
            out = []
            b = self.bindings.get(node.module, {})
            for name in sorted(_names_of(node)):
                x = b.get(name)
                if x is None or x.kind in ("def", "local", "localmod"):
                    continue
                if x.kind == "import":
                    mod = x.stmt.split()[1]
                    dist = self.packages.get(mod) or self.packages.get(mod.split(".")[0])
                    if dist in _RUNTIME_DISTS:
                        continue
                    req = self.requirements.get(dist) if dist else None
                    out.append(f"{name}  ({req.spec if req else 'stdlib'})")
                elif x.kind == "value":
                    text = x.expr if len(x.expr or "") <= 40 else (x.expr or "")[:37] + "..."
                    out.append(f"{name} = {text}")
            return out

        def walk(key: str, prefix: str, last: bool, role: str = "") -> None:
            node = self.nodes[key]
            branch = "" if not prefix and not lines else ("└── " if last else "├── ")
            label = f"{role}{node.name}  {self._label(node)}  [{node.module}]"
            if key in shown:
                lines.append(f"{prefix}{branch}{role}{node.name}  (see above)")
                return
            shown.add(key)
            lines.append(f"{prefix}{branch}{label}")
            child_prefix = prefix + ("" if not branch else ("    " if last else "│   "))
            children: List[Tuple[str, str]] = [(k, "tool ") for k in node.tools if isinstance(k, str) and k in self.nodes]
            children += [(k, "") for k in node.uses if k in self.nodes and k not in {c for c, _ in children}]
            extras = leafs(node) + [f"file {f}" for f in node.files]
            total = len(children) + len(extras)
            for i, (k, role_) in enumerate(children):
                walk(k, child_prefix, i == total - 1, role_)
            for j, text in enumerate(extras):
                lines.append(f"{child_prefix}{'└── ' if len(children) + j == total - 1 else '├── '}{text}")

        walk(self.entry, "", True)
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = [self.tree(), ""]
        if self.requirements:
            parts.append("requirements: " + ", ".join(r.spec for r in self.requirements.values()))
        if self.problems:
            parts.append("")
            parts += [str(p) for p in self.problems]
        else:
            parts.append("no problems: ready to save")
        return "\n".join(parts)

    __str__ = __repr__


_RUNTIME_DISTS = ("functai", "lmcc", "lm15")


def _names_of(node: Node) -> Set[str]:
    return getattr(node, "_names", set())


# ------------------------------------------------------------------ reading code


_MUTATING_METHODS = frozenset({"append", "extend", "insert", "remove", "pop", "clear", "update", "setdefault",
                               "popitem", "add", "discard", "sort", "reverse", "__setitem__", "__delitem__",
                               "difference_update", "intersection_update", "symmetric_difference_update"})
_DYNAMIC_CALLS = frozenset({"eval", "exec", "__import__", "globals", "vars"})


def _code_objects(code: types.CodeType) -> Iterable[types.CodeType]:
    stack = [code]
    while stack:
        c = stack.pop()
        yield c
        stack += [k for k in c.co_consts if isinstance(k, types.CodeType)]


@dataclasses.dataclass
class CodeFacts:
    """What a function's bytecode says about the names it uses at call time."""
    globals_read: List[str]
    globals_written: Set[str]
    attr_chains: Dict[str, Set[Tuple[str, ...]]]     # global name → attribute chains read on it
    imports: Set[str]


def _code_facts(code: types.CodeType) -> CodeFacts:
    read: List[str] = []
    written: Set[str] = set()
    chains: Dict[str, Set[Tuple[str, ...]]] = {}
    imports: Set[str] = set()
    for c in _code_objects(code):
        ins = list(dis.get_instructions(c))
        for i, x in enumerate(ins):
            if x.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                name = x.argval
                if name not in read:
                    read.append(name)
                chain: List[str] = []
                j = i + 1
                while j < len(ins) and ins[j].opname in ("LOAD_ATTR", "LOAD_METHOD"):
                    chain.append(ins[j].argval)
                    j += 1
                if chain:
                    chains.setdefault(name, set()).add(tuple(chain))
            elif x.opname in ("STORE_GLOBAL", "DELETE_GLOBAL"):
                written.add(x.argval)
            elif x.opname == "IMPORT_NAME":
                imports.add(x.argval)
    return CodeFacts(read, written, chains, imports)


def _parse_def(source: str) -> Optional[ast.AST]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node
    return None


def _load_names(nodes: Iterable[Optional[ast.AST]]) -> Set[str]:
    out: Set[str] = set()
    for n in nodes:
        if n is None:
            continue
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                out.add(x.id)
    return out


def _string_annotation_names(nodes: Iterable[Optional[ast.AST]]) -> Set[str]:
    """Names inside string annotations (``x: "Row"``), which functai evaluates."""
    out: Set[str] = set()
    for n in nodes:
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            try:
                out |= _load_names([ast.parse(n.value, mode="eval")])
            except SyntaxError:
                pass
    return out


def _deftime_names(fn_node: ast.AST) -> Set[str]:
    """Names evaluated when a def runs (decorators, annotations, defaults) plus
    the annotations inside the body (functai evaluates ``x: T = _ai`` lines)."""
    parts: List[Optional[ast.AST]] = []
    if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = fn_node.args
        parts += list(fn_node.decorator_list) + [fn_node.returns]
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if arg is not None:
                parts.append(arg.annotation)
        parts += list(a.defaults) + [d for d in a.kw_defaults if d is not None]
        for x in ast.walk(fn_node):
            if isinstance(x, ast.AnnAssign):
                parts.append(x.annotation)
    elif isinstance(fn_node, ast.ClassDef):
        parts += list(fn_node.decorator_list) + list(fn_node.bases) + [k.value for k in fn_node.keywords]
        for stmt in fn_node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts += list(stmt.decorator_list) + [stmt.returns]
                a = stmt.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                    if arg is not None:
                        parts.append(arg.annotation)
                parts += list(a.defaults) + [d for d in a.kw_defaults if d is not None]
            else:
                parts.append(stmt)
    return _load_names(parts) | _string_annotation_names(parts)


def _mutations(fn_node: ast.AST, local_names: Set[str]) -> Dict[str, str]:
    """Globals the code changes in place: ``{name: what it does}``."""
    out: Dict[str, str] = {}

    def base(expr: ast.AST) -> Optional[str]:
        while isinstance(expr, (ast.Subscript, ast.Attribute)):
            expr = expr.value
        return expr.id if isinstance(expr, ast.Name) and expr.id not in local_names else None

    for x in ast.walk(fn_node):
        if isinstance(x, ast.Global):
            for n in x.names:
                out.setdefault(n, f"declares `global {n}` and rebinds it")
        targets: List[ast.AST] = []
        if isinstance(x, ast.Assign):
            targets = x.targets
        elif isinstance(x, (ast.AugAssign, ast.AnnAssign)) and x.target is not None:
            targets = [x.target]
        elif isinstance(x, ast.Delete):
            targets = x.targets
        for t in targets:
            if isinstance(t, (ast.Subscript, ast.Attribute)):
                n = base(t)
                if n:
                    out.setdefault(n, f"writes into it ({ast.unparse(t)} = ...)")
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr in _MUTATING_METHODS:
            n = base(x.func.value)
            if n:
                out.setdefault(n, f"changes it ({ast.unparse(x.func)}(...))")
    return out


def _dynamic_uses(fn_node: ast.AST, resolve: Callable[[str], Any]) -> List[str]:
    found: List[str] = []
    for x in ast.walk(fn_node):
        if not isinstance(x, ast.Call):
            continue
        f = x.func
        if isinstance(f, ast.Name) and f.id in _DYNAMIC_CALLS:
            found.append(f"{f.id}(...)")
        elif isinstance(f, ast.Name) and f.id == "getattr" and x.args and isinstance(x.args[0], ast.Name) \
                and isinstance(resolve(x.args[0].id), types.ModuleType) and not (
                    len(x.args) > 1 and isinstance(x.args[1], ast.Constant)):
            found.append(f"getattr({x.args[0].id}, ...)")
        elif isinstance(f, ast.Attribute) and f.attr == "import_module":
            found.append("importlib.import_module(...)")
    return found


def _file_calls(fn_node: ast.AST, resolve: Callable[[str], Any]) -> Tuple[List[str], int]:
    """Constant arguments of ``functai.file(...)`` calls, and how many were not constant."""
    from .saved import file as _file_fn
    consts: List[str] = []
    dynamic = 0
    for x in ast.walk(fn_node):
        if not isinstance(x, ast.Call):
            continue
        f = x.func
        target = None
        if isinstance(f, ast.Attribute) and f.attr == "file" and isinstance(f.value, ast.Name):
            mod = resolve(f.value.id)
            target = getattr(mod, "file", None) if isinstance(mod, types.ModuleType) else None
        elif isinstance(f, ast.Name):
            target = resolve(f.id)
        if target is not _file_fn:
            continue
        if x.args and isinstance(x.args[0], ast.Constant) and isinstance(x.args[0].value, str):
            consts.append(x.args[0].value)
        else:
            dynamic += 1
    return consts, dynamic


def _strip_decorators(source: str) -> str:
    node = _parse_def(source)
    if node is None or not getattr(node, "decorator_list", None):
        return source
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1:])


_MISSING = object()



# ------------------------------------------------------------------ values


def _literal(value: Any, depth: int = 0) -> Optional[str]:
    """Python source that rebuilds a plain value exactly, or None."""
    if depth > 50:
        return None
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return repr(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return f"float({str(value)!r})"
        return repr(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_literal(v, depth + 1) for v in value]
        if any(i is None for i in items):
            return None
        if isinstance(value, list):
            return "[" + ", ".join(items) + "]"
        if isinstance(value, tuple):
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
        body = "{" + ", ".join(sorted(items)) + "}" if items else ""
        if isinstance(value, frozenset):
            return f"frozenset({body})"
        return body or "set()"
    if isinstance(value, dict):
        pairs = []
        for k, v in value.items():
            ks, vs = _literal(k, depth + 1), _literal(v, depth + 1)
            if ks is None or vs is None:
                return None
            pairs.append(f"{ks}: {vs}")
        return "{" + ", ".join(pairs) + "}"
    return None


# ------------------------------------------------------------------ the analysis


class Analysis:
    def __init__(self, include: Iterable[str] = (), *, discover: bool = False):
        self.discover = discover          # find every AI function, wherever it is defined
        self.origins = Origins(include)
        self.nodes: Dict[str, Node] = {}
        self.bindings: Dict[str, Dict[str, Binding]] = {}
        self.problems: List[Problem] = []
        self.packages: Dict[str, str] = {}          # module → distribution
        self.mutated: Dict[Tuple[str, str], Tuple[str, str]] = {}   # (module, name) → (where, what)
        self.declared: List[str] = []
        self._by_id: Dict[int, str] = {}

    # ---- problems

    def problem(self, code: str, where: str, message: str, fix: str, severity: str = "error") -> None:
        p = Problem(code, where, message, fix, severity)
        if p not in self.problems:
            self.problems.append(p)

    # ---- modules

    def _package(self, module: str) -> None:
        dist = self.origins.distribution(module)
        if dist:
            self.packages[module] = dist
            self.packages.setdefault(module.split(".")[0], dist)

    def _bind(self, module: str, b: Binding, where: str) -> str:
        """Record that ``module``'s file defines ``b.name``; a different object
        already bound to that name is a conflict."""
        table = self.bindings.setdefault(module, {})
        old = table.get(b.name)
        if old is not None:
            if old.obj_id != b.obj_id:
                self.problem("name-conflict", f"{module}:{b.name}",
                             f"two different things are saved under the name {b.name!r} in {module}",
                             "give them distinct names (functions defined inside other functions are "
                             "saved at the top of their module)")
            elif b.deftime and not old.deftime:
                old.deftime = True
            return b.name
        table[b.name] = b
        return b.name

    # ---- entry points

    def program(self, obj: Any) -> str:
        if isinstance(obj, (FunctAIFunc, FunctAIModule)) or inspect.isfunction(obj):
            key = self.code(obj, where="entry")
            if key is None:
                raise TypeError(f"functai.check needs a program defined in your code, not {obj!r}")
            return key
        raise TypeError(f"functai.check takes an @ai function, an @module, or a Python function; "
                        f"not {type(obj).__name__}")

    # ---- code nodes

    def _defining(self, obj: Any) -> Tuple[Any, str, str]:
        fn = obj._fn if isinstance(obj, (FunctAIFunc, FunctAIModule)) else obj
        return fn, getattr(fn, "__module__", None) or "__main__", getattr(fn, "__name__", "")

    def code(self, obj: Any, *, where: str) -> Optional[str]:
        """The node key of a function/class/AI function/module from local code."""
        if id(obj) in self._by_id:
            return self._by_id[id(obj)]
        fn, module, name = self._defining(obj)
        key = f"{module}:{name}"
        if isinstance(obj, FunctAIFunc):
            kind = "ai"
        elif isinstance(obj, FunctAIModule):
            kind = "module"
        elif inspect.isclass(obj):
            kind = "class"
        else:
            kind = "function"
        if name == "<lambda>":
            self.problem("lambda", where, "a lambda has no source functai can save on its own",
                         "define it with def")
            return None
        existing = self.nodes.get(key)
        if existing is not None and existing.obj is not obj:
            self.problem("name-conflict", key, f"two different {kind}s are both named {name!r} in {module}",
                         "give them distinct names (functions defined inside other functions are saved at "
                         "the top of their module)")
            return key
        node = Node(key, kind, module, name, obj)
        self.nodes[key] = node
        self._by_id[id(obj)] = key
        try:
            source = textwrap.dedent(_class_source(fn) if inspect.isclass(fn) else inspect.getsource(fn))
        except (OSError, TypeError):
            self.problem("no-source", key, f"the source of {name!r} cannot be read",
                         "define it in a file or a notebook cell (not with exec), so its source can be saved")
            return key
        if kind in ("ai", "module"):
            source = _strip_decorators(source)
            if getattr(fn, "__wrapped__", None) is not None:
                self.problem("wrapped-function", key, "the function is wrapped by another decorator under @ai",
                             "put @ai (or @module) directly on the function")
        node.source = source
        tree = _parse_def(source)
        if tree is None:
            self.problem("no-source", key, "the source cannot be parsed as one definition",
                         "define it as a plain def or class")
            return key
        node.future_annotations = kind != "class" and bool(
            getattr(getattr(fn, "__code__", None), "co_flags", 0) & 0x1000000)
        if kind == "class":
            self._class(node, fn, tree)
        else:
            self._function(node, fn, tree)
        if kind == "ai":
            self._ai(node, obj)
        elif kind == "module":
            node.requires = tuple(getattr(obj, "_requires", ()) or ())
        return key

    def _namespace(self, fn: Any) -> Dict[str, Any]:
        g = getattr(fn, "__globals__", None)
        if g is None:
            mod = sys.modules.get(getattr(fn, "__module__", "") or "")
            g = vars(mod) if mod is not None else {}
        return g

    def _function(self, node: Node, fn: Any, tree: ast.AST) -> None:
        ns = self._namespace(fn)
        closure: Dict[str, Any] = {}
        code = fn.__code__
        for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
            try:
                closure[name] = cell.cell_contents
            except ValueError:
                self.problem("unresolved-name", node.key, f"it closes over {name!r}, which is not set",
                             "pass it as an argument or make it a module-level name")
        facts = _code_facts(code)
        locals_ = set(code.co_varnames) | set(code.co_cellvars)
        deftime = _deftime_names(tree)
        # Nested defs have their own locals; names bound anywhere inside are not globals.
        for c in _code_objects(code):
            locals_ |= set(c.co_varnames) | set(c.co_cellvars) | set(c.co_freevars)

        def resolve(name: str) -> Any:
            if name in closure:
                return closure[name]
            if name in ns:
                return ns[name]
            return getattr(builtins, name, _MISSING)

        node._scope = lambda n: n in closure or n in ns   # type: ignore[attr-defined]
        node._deftime = deftime                             # type: ignore[attr-defined]
        names = list(dict.fromkeys(list(closure) + facts.globals_read + sorted(deftime)))
        for name in facts.globals_written:
            self.mutated.setdefault((node.module, name), (node.key, f"rebinds it (global {name})"))
        for name, what in _mutations(tree, locals_ - set(closure)).items():
            if name in ns or name in closure:
                self.mutated.setdefault((node.module, name), (node.key, what))
        for use in _dynamic_uses(tree, resolve):
            self.problem("dynamic-lookup", node.key, f"it uses {use}, which functai cannot follow",
                         "prefer plain names and imports; functai.verify will show anything still missing",
                         "warning")
        for mod in sorted(facts.imports):
            if self.origins.of(mod) == "local" and mod != node.module:
                self.problem("local-import-inside", node.key,
                             f"it imports your module {mod!r} inside the function",
                             f"import it at the top of {node.module}, so functai can follow and save it")
            elif self.origins.of(mod) == "package":
                self._package(mod)
        files, dynamic = _file_calls(tree, resolve)
        node.files = files
        if dynamic:
            self.problem("dynamic-file", node.key, "functai.file(...) is called with a path that is not a "
                         "literal string, so the file cannot be saved", "write the path as a string literal",
                         "warning")
        self._uses(node, names, resolve, deftime, facts.attr_chains)

    def _class(self, node: Node, cls: type, tree: ast.ClassDef) -> None:
        mod = sys.modules.get(cls.__module__)
        ns = vars(mod) if mod is not None else {}
        deftime = _deftime_names(tree)
        names = sorted(deftime)
        call_names: List[str] = []
        chains: Dict[str, Set[Tuple[str, ...]]] = {}
        method_names = [s.name for s in tree.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        body_locals = {t.id for s in tree.body for t in ast.walk(s) if isinstance(t, ast.Name)
                       and isinstance(t.ctx, ast.Store)}
        for m in method_names:
            raw = cls.__dict__.get(m)
            f = getattr(raw, "__func__", raw)
            if isinstance(raw, property):
                f = raw.fget
            if not inspect.isfunction(f):
                continue
            facts = _code_facts(f.__code__)
            call_names += facts.globals_read
            for k, v in facts.attr_chains.items():
                chains.setdefault(k, set()).update(v)
            for name in facts.globals_written:
                self.mutated.setdefault((node.module, name), (node.key, f"rebinds it (global {name})"))
            sub = next(s for s in tree.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == m)
            local_names = set(f.__code__.co_varnames)
            for name, what in _mutations(sub, local_names).items():
                if name in ns:
                    self.mutated.setdefault((node.module, name), (node.key, what))

        def resolve(name: str) -> Any:
            if name in ns:
                return ns[name]
            return getattr(builtins, name, _MISSING)

        node._scope = lambda n: n in ns   # type: ignore[attr-defined]
        node._deftime = deftime            # type: ignore[attr-defined]
        all_names = list(dict.fromkeys(names + call_names))
        self._uses(node, [n for n in all_names if n not in body_locals or n in ns], resolve, deftime, chains)

    def _uses(self, node: Node, names: List[str], resolve: Callable[[str], Any], deftime: Set[str],
              chains: Dict[str, Set[Tuple[str, ...]]]) -> None:
        node._names = set()   # type: ignore[attr-defined]
        for name in names:
            value = resolve(name)
            if value is _MISSING:
                if name in deftime and name not in _DUNDER_OK:
                    self.problem("unresolved-name", node.key,
                                 f"{name!r} is not defined where the function is (a name from the enclosing "
                                 f"function?)", "make it a module-level name, or a literal")
                elif name not in deftime and name not in _DUNDER_OK:
                    self.problem("unresolved-name", node.key, f"it uses {name!r}, which is not defined",
                                 "define it, or remove the use", "warning")
                continue
            if not node._scope(name):     # type: ignore[attr-defined]  # a builtin
                continue
            node._names.add(name)   # type: ignore[attr-defined]
            self.value(node.module, name, value, where=node.key, deftime=name in deftime, user=node,
                       chains=chains.get(name, set()))

    # ---- AI functions

    def _ai(self, node: Node, fn: FunctAIFunc) -> None:
        node.requires = tuple(getattr(fn, "_requires", ()) or ())
        for t in fn._tools:
            if isinstance(t, FunctAIFunc):
                k = self.code(t, where=f"{node.key} tools")
                if k:
                    node.tools.append(k)
                    if k not in node.uses:
                        node.uses.append(k)
            elif inspect.isfunction(t) and self.origins.of(t.__module__) == "local":
                k = self.code(t, where=f"{node.key} tools")
                if k:
                    node.tools.append(k)
                    if k not in node.uses:
                        node.uses.append(k)
            elif inspect.isfunction(t) or inspect.isbuiltin(t):
                path = self._import_path(t)
                if path is None:
                    self.problem("unsaveable-value", node.key, f"the tool {t!r} cannot be imported by name",
                                 "wrap it in a function of your own")
                else:
                    node.tools.append({"import": f"{path[0]}:{path[1]}"})
                    self._package(path[0])
            elif hasattr(t, "name") and hasattr(t, "parameters"):
                node.tools.append({"tool": {"name": t.name, "description": t.description,
                                            "parameters": t.parameters}})
            else:
                self.problem("unsaveable-value", node.key, f"the tool {t!r} is not a function",
                             "use a plain def as the tool (a bound method or a partial carries hidden state)")
        teacher = fn._settings.get("teacher")
        if isinstance(teacher, FunctAIFunc):
            node.teacher = self.code(teacher, where=f"{node.key} teacher")
            if node.teacher and node.teacher not in node.uses:
                node.uses.append(node.teacher)
        spec = fn._spec()
        for f in spec.signature.inputs:
            if f.purpose == "plain" and f.name in fn._sig.parameters and \
                    fn._sig.parameters[f.name].annotation is inspect.Parameter.empty:
                self.problem("untyped-input", node.key, f"the input {f.name!r} has no type (it is sent as text)",
                             f"annotate it, e.g. {f.name}: str")
        if not _typed_output(fn, spec):
            self.problem("untyped-output", node.key, f"the output {spec.main!r} has no type",
                         "annotate the return type, e.g. -> str")
        for k, v in fn._settings.items():
            if k == "adapter" and isinstance(v, lmcc.Adapter):
                try:
                    v.dump()
                except lmcc.Refusal as exc:
                    self.problem("unsaveable-value", node.key, f"its adapter cannot be written as data: {exc.hint}",
                                 "ship formats written in code with lmcc.ship, or name registered formats")
            elif k in ("lm", "escalate_to") and _is_baked(v):
                node.baked[k] = v
                try:
                    type(v)(v.path)              # its files are what was baked
                except Exception as exc:  # noqa: BLE001 — reported as the problem it is
                    self.problem("unsaveable-value", node.key, f"its baked model cannot be saved: {exc}",
                                 "bake it again, or restore its folder")
            elif k == "escalate_to" and isinstance(v, FunctAIFunc):
                target = self.code(v, where=f"{node.key} escalate_to")
                if target:
                    node.escalate = target
                    if target not in node.uses:
                        node.uses.append(target)
            elif k in ("client",):
                self.problem("connection-not-saved", node.key, "its client= connection is not saved",
                             "the loading machine's logins, keys or client= are used", "warning")
            elif k == "api_key":
                self.problem("secret-not-saved", node.key, "its api_key is not saved (secrets never are)",
                             "the loading machine's environment or functai.login() supplies it", "warning")
            elif k == "auth" and v not in (None, True, False):
                self.problem("connection-not-saved", node.key, "its auth= credentials file is not saved",
                             "the loading machine's own logins are used", "warning")
            elif k == "lm" and not isinstance(v, str):
                self.problem("connection-not-saved", node.key,
                             f"lm is a {type(v).__name__}; only its model name is saved",
                             "the loading machine's logins or keys reach that model", "warning")
            elif k in ("optimizer",) and v is not None:
                self.problem("setting-not-saved", node.key, f"the default {k}= is code and is not saved",
                             "pass optimizer= to .opt() where the program is optimized", "warning")
            elif callable(v) and not isinstance(v, (FunctAIFunc, lmcc.Adapter)) and k not in ("client",):
                self.problem("unsaveable-value", node.key, f"the setting {k}= is a {type(v).__name__}",
                             f"use a string or a number for {k}=")
        try:
            state_to_json(fn._state)
        except lmcc.Refusal as exc:
            self.problem("unsaveable-value", node.key, f"its demos cannot be written as data: {exc.hint}",
                         "use plain values (text, numbers, lists, dicts, dataclasses) in demos")
        self.declared += list(node.requires)

    # ---- values

    def value(self, module: str, name: str, value: Any, *, where: str, deftime: bool = False,
              user: Optional[Node] = None, chains: Set[Tuple[str, ...]] = frozenset()) -> None:
        """Bind ``name`` in ``module``'s saved file to ``value``."""
        oid = id(value)
        # functai's own sentinel and helpers, lmcc, lm15: imports
        if isinstance(value, types.ModuleType):
            self._module_value(module, name, value, where, deftime, user, chains)
            return
        if isinstance(value, (FunctAIFunc, FunctAIModule)) or inspect.isfunction(value) or inspect.isclass(value):
            fn, defmod, defname = self._defining(value)
            origin = self.origins.of(defmod)
            if origin == "local" or (self.discover and isinstance(value, (FunctAIFunc, FunctAIModule))):
                key = self.code(value, where=where)
                if key is None:
                    return
                if user is not None and key not in user.uses and key != user.key:
                    user.uses.append(key)
                if defmod == module:
                    self._bind(module, Binding(name, "def" if name == defname else "value", oid, node=key,
                                               expr=None if name == defname else defname,
                                               needs=() if name == defname else (defname,), deftime=deftime), where)
                    if name != defname:
                        self._bind(module, Binding(defname, "def", oid, node=key, deftime=deftime), where)
                else:
                    self._bind(module, Binding(name, "local", oid, module=defmod, attr=defname, deftime=deftime),
                               where)
                return
            self._import_binding(module, name, value, where, deftime)
            return
        if isinstance(value, _AISentinel):
            self._bind(module, Binding(name, "import", oid, stmt=_from_import("functai", "_ai", name)), where)
            self._package("functai")
            return
        expr, needs = self._expression(module, value, where, user)
        if expr is not None:
            self._bind(module, Binding(name, "value", oid, expr=expr, needs=tuple(needs), deftime=deftime), where)
            if len(expr) > 100_000:
                self.problem("large-constant", f"{module}:{name}", f"{name} is {len(expr):,} characters of source",
                             "load large data from a file with functai.file(...)", "warning")
            return
        if self._import_binding(module, name, value, where, deftime, quiet=True):
            return
        self.problem("unsaveable-value", f"{module}:{name}",
                     f"{name} is a {type(value).__name__} object, which cannot be written as code or data "
                     f"(used by {where})",
                     "create it inside the function (or a tool), or pass it in as an argument")

    def _module_value(self, module: str, name: str, mod: types.ModuleType, where: str, deftime: bool,
                      user: Optional[Node], chains: Set[Tuple[str, ...]]) -> None:
        mname = mod.__name__
        origin = self.origins.of(mname)
        if origin != "local":
            if origin == "package":
                self._package(mname)
            stmt = f"import {mname}" if name == mname else f"import {mname} as {name}"
            self._bind(module, Binding(name, "import", id(mod), stmt=stmt, deftime=deftime), where)
            return
        # a module of your own: save the members the code reaches through it
        self._bind(module, Binding(name, "localmod", id(mod), module=mname, deftime=deftime), where)
        if not chains:
            self.problem("dynamic-lookup", where, f"it uses your module {mname!r} as a whole",
                         f"use its members by name (from {mname} import ...), so functai can save them",
                         "warning")
        for chain in chains:
            target: Any = mod
            owner = mname
            for attr in chain:
                if not isinstance(target, types.ModuleType):
                    break
                value = getattr(target, attr, _MISSING)
                if value is _MISSING:
                    self.problem("unresolved-name", where, f"{owner}.{attr} does not exist",
                                 "fix the name", "warning")
                    break
                self.value(target.__name__, attr, value, where=where, user=user)
                target, owner = value, getattr(value, "__name__", attr)

    def _import_binding(self, module: str, name: str, value: Any, where: str, deftime: bool,
                        quiet: bool = False) -> bool:
        path = self._import_path(value)
        if path is None:
            if not quiet:
                self.problem("unsaveable-value", f"{module}:{name}",
                             f"{name} ({type(value).__name__}) comes from {getattr(value, '__module__', '?')} "
                             f"but cannot be imported by name", "import it from its public module")
            return False
        self._package(path[0])
        self._bind(module, Binding(name, "import", id(value), stmt=_from_import(path[0], path[1], name),
                                   deftime=deftime), where)
        return True

    @staticmethod
    def _import_path(value: Any) -> Optional[Tuple[str, str]]:
        """(module, attribute) that imports ``value``: the shortest public path."""
        candidates: List[Tuple[str, Optional[str]]] = []
        mod = getattr(value, "__module__", None)
        qual = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
        if isinstance(mod, str):
            parts = mod.split(".")
            for i in range(1, len(parts) + 1):
                candidates.append((".".join(parts[:i]), qual))
        tmod = type(value).__module__
        if isinstance(tmod, str) and tmod != "builtins":
            parts = tmod.split(".")
            for i in range(1, len(parts) + 1):
                candidates.append((".".join(parts[:i]), None))
        for modname, attr in candidates:
            m = sys.modules.get(modname)
            if m is None:
                continue
            if attr and "." not in attr and getattr(m, attr, None) is value:
                if not attr.startswith("_") or modname.split(".")[0] in ("functai",):
                    return modname, attr
            if attr is None:
                for k, v in vars(m).items():
                    if v is value and not k.startswith("__"):
                        return modname, k
        return None

    def _expression(self, module: str, value: Any, where: str, user: Optional[Node]) -> Tuple[Optional[str], List[str]]:
        """Source that rebuilds a value, and the names it needs in ``module``."""
        lit = _literal(value)
        if lit is not None:
            return lit, []
        if isinstance(value, enum.Enum):
            cls = type(value)
            cname = self._need(module, cls, where, user)
            return (f"{cname}.{value.name}", [cname]) if cname else (None, [])
        if isinstance(value, re.Pattern):
            default = re.compile(value.pattern).flags if isinstance(value.pattern, str) else 0
            flags = "" if value.flags == default else f", {int(value.flags)}"
            return f"__import__('re').compile({value.pattern!r}{flags})", []
        if isinstance(value, Path):
            self.problem("path-value", where, f"it uses the path {str(value)!r}; the file there is not saved",
                         "read files through functai.file('...'), which saves them with the program", "warning")
            return f"__import__('pathlib').{type(value).__name__}({str(value)!r})", []
        if isinstance(value, lmcc.Adapter):
            try:
                data = value.dump()
            except lmcc.Refusal:
                return None, []
            text = _literal(json.loads(json.dumps(data)))
            return (f"__import__('lmcc').load({text}, registry=__import__('lmcc').default_registry)", []) \
                if text else (None, [])
        t = self._type_expression(module, value, where, user)
        if t is not None:
            return t
        return None, []

    def _need(self, module: str, obj: Any, where: str, user: Optional[Node]) -> Optional[str]:
        """A name in ``module`` bound to ``obj`` (a class or function), binding it if needed."""
        if getattr(builtins, getattr(obj, "__name__", ""), None) is obj:
            return obj.__name__
        name = getattr(obj, "__name__", None)
        if not name:
            return None
        table = self.bindings.get(module, {})
        if name in table and table[name].obj_id != id(obj):
            name = f"_{name}_{abs(id(obj)) % 10_000}"
        self.value(module, name, obj, where=where, deftime=True, user=user)
        return name

    def _type_expression(self, module: str, t: Any, where: str, user: Optional[Node]) -> Optional[Tuple[str, List[str]]]:
        origin, args = typing.get_origin(t), typing.get_args(t)
        if origin is None:
            return None
        needs: List[str] = []

        def expr(x: Any) -> Optional[str]:
            if x is type(None) or x is None:
                return "None"
            if x is Ellipsis:
                return "..."
            if isinstance(x, (type, types.FunctionType)) and typing.get_origin(x) is None:
                n = self._need(module, x, where, user)
                if n:
                    needs.append(n)
                return n
            lit = _literal(x)
            if lit is not None and not isinstance(x, type):
                return lit
            sub = self._type_expression(module, x, where, user)
            if sub is None:
                return None
            needs.extend(sub[1])
            return sub[0]

        if origin is typing.Literal:
            items = [_literal(a) for a in args]
            return (f"__import__('typing').Literal[{', '.join(items)}]", []) if None not in items else None
        if origin is typing.Annotated:
            base = expr(args[0])
            extras = [_literal(a) for a in args[1:]]
            if base is None or None in extras:
                return None
            return f"__import__('typing').Annotated[{base}, {', '.join(extras)}]", needs
        if origin is typing.Union or isinstance(t, types.UnionType):
            parts = [expr(a) for a in args]
            return (f"__import__('typing').Union[{', '.join(parts)}]", needs) if None not in parts else None
        head = expr(origin) if isinstance(origin, type) else None
        if head is None:
            return None
        parts = [expr(a) for a in args]
        if None in parts:
            return None
        return f"{head}[{', '.join(parts)}]", needs

    # ---- after the walk

    def finish(self) -> None:
        for (module, name), (where, what) in self.mutated.items():
            b = self.bindings.get(module, {}).get(name)
            if b is not None and b.kind == "value":
                self.problem("hidden-state", where,
                             f"the global {name!r} is state the program changes: it {what}",
                             "pass it in as an argument, return it, or keep it in an object you give a tool; "
                             "save(allow=['hidden-state']) saves its current value as a constant")
            elif b is not None and b.kind in ("def", "local"):
                self.problem("hidden-state", where, f"it replaces {name!r} at run time ({what})",
                             "do not rebind functions or classes while the program runs")


_DUNDER_OK = frozenset({"__name__", "__file__", "__doc__", "__class__", "__builtins__", "__spec__",
                        "__loader__", "__package__", "__annotations__", "__module__", "__qualname__"})


def _from_import(module: str, attr: str, name: str) -> str:
    return f"from {module} import {attr}" + ("" if name == attr else f" as {name}")


def _is_baked(obj: Any) -> bool:
    return getattr(type(obj), "__functai_baked__", False) is True


def _typed_output(fn: FunctAIFunc, spec) -> bool:
    ret = fn._sig.return_annotation
    if ret is not inspect.Parameter.empty:
        return True
    from .signature import _collect_ast_outputs
    for n, t, _d in _collect_ast_outputs(fn._fn):
        if n == spec.main and t is not None:
            return True
    return False


def state_to_json(state: Any) -> Dict[str, Any]:
    """An AI function's instruction and demos as JSON (lmcc turns as their own JSON)."""
    to_json = lmcc.turn.to_json
    demos = []
    for d in state.demos:
        if isinstance(d, lmcc.Turn):
            demos.append(d.to_dict())
        elif isinstance(d, dict) and "signature" in d:
            demos.append(to_json(d, where="demo"))
        else:
            demos.append({"inputs": to_json(d["inputs"], where="demo inputs"),
                          "outputs": to_json(d["outputs"], where="demo outputs")})
    return {"instructions": state.instructions, "demos": demos}


# ------------------------------------------------------------------ requirements


def _direct_url(dist: md.Distribution) -> Optional[dict]:
    try:
        text = dist.read_text("direct_url.json")
    except Exception:
        return None
    return json.loads(text) if text else None


def requirement_of(name: str, reason: str = "") -> Optional[Requirement]:
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return None
    version = dist.version
    canonical = dist.metadata["Name"] or name
    url = _direct_url(dist)
    if url and "vcs_info" in url:
        vcs = url["vcs_info"]
        spec = f"{canonical} @ {vcs.get('vcs', 'git')}+{url['url']}@{vcs.get('commit_id', '')}"
        return Requirement(canonical, version, spec, None, reason)
    if url and url.get("url", "").startswith("file://"):
        return Requirement(canonical, version, f"{canonical} @ {url['url']}", url["url"][len("file://"):], reason)
    if "+" in version:            # a local build (torch 2.10.0+cu128): the index's build of that version
        return Requirement(canonical, version, f"{canonical}=={version.split('+')[0]}", None, reason)
    return Requirement(canonical, version, f"{canonical}=={version}", None, reason)


BAKED_RUNTIME = ("torch", "transformers", "safetensors")


def _requirement_names(dist: md.Distribution) -> List[str]:
    """Distributions ``dist`` requires, leaving out extras."""
    out = []
    for line in dist.requires or ():
        head, _, marker = line.partition(";")
        if "extra" in marker:
            continue
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", head)
        if m:
            out.append(m.group(1))
    return out


def lock(requirements: Iterable[Requirement]) -> List[Requirement]:
    """Every distribution the requirements pull in, as installed here, in a
    stable order. Requirements whose markers exclude this machine are not
    installed here and so are left out; that is the one thing the lock cannot
    know."""
    seen: Dict[str, Requirement] = {}
    queue = [r.distribution for r in requirements]
    while queue:
        name = queue.pop(0)
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in seen:
            continue
        req = requirement_of(name)
        if req is None:
            continue
        seen[key] = req
        queue += _requirement_names(md.distribution(name))
    return [seen[k] for k in sorted(seen)]


# ------------------------------------------------------------------ check


def check(program: Any, *, include: Iterable[str] = (), requires: Iterable[str] = ()) -> Report:
    '''List everything a program depends on, and what would stop a clean save.

    Follows every name the code reaches: AI functions and modules (also
    through helper functions), their tools, your own functions and classes
    (in files or notebook cells), the types in the signatures, constants,
    and data files read through ``functai.file``. Reads code only: runs
    nothing and calls no model.

    Parameters
    ----------
    program : AI function, module, or function
        The program's entry point.
    include : list of str
        Modules or package prefixes to save as code even though they are
        installed (your own project, installed in editable mode).
    requires : list of str
        Requirements to add by hand (``"numpy>=2"``), for what the code
        reaches in ways that reading it can't see.

    Returns
    -------
    Report
        Displays as a tree of dependencies, the requirements, and each
        problem with its fix. ``report.ok`` is True when nothing stops a save.

    See Also
    --------
    save : save the program, once ``check`` is clean.

    Examples
    --------
    ```python
    ORDERS = {"A-1042": "stuck at carrier"}

    def lookup_order(order_id: str) -> str:
        """The order's shipping status."""
        return ORDERS.get(order_id, "no such order")

    @ai(tools=[lookup_order])
    def reply(message: str) -> str:
        """A short reply to the customer. Check the order first."""
        ...

    check(reply)
    ```
    '''
    a = Analysis(include)
    entry = a.program(program)
    a.finish()
    if isinstance(program, (FunctAIModule,)) or inspect.isfunction(program):
        fn = program._fn if isinstance(program, FunctAIModule) else program
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.annotation is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                a.problem("untyped-input", entry, f"the input {p.name!r} has no type",
                          f"annotate it, e.g. {p.name}: str")
        if sig.return_annotation is inspect.Signature.empty:
            a.problem("untyped-output", entry, "the program's result has no type",
                      "annotate the return type, e.g. -> list[str]")
    # requirements: the runtime, every package reached, and what was declared
    reqs: Dict[str, Requirement] = {}
    baked_models = [b for n in a.nodes.values() for b in n.baked.values()]
    runtime = list(BAKED_RUNTIME) if baked_models else []
    for dist in runtime:
        if requirement_of(dist) is None:
            a.problem("missing-package", "requirements", f"a baked model needs {dist}, which is not installed here",
                      'pip install "functai[bake]"')
    for dist in ["functai"] + runtime + sorted(set(a.packages.values())):
        r = requirement_of(dist, "reached by the code" if dist != "functai" else "runtime")
        if r is not None:
            reqs[r.distribution] = r
    declared = list(dict.fromkeys(list(requires) + a.declared))
    for spec in declared:
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
        if m and requirement_of(m.group(1)) is None:
            a.problem("missing-package", "requires", f"{spec!r} is declared but not installed here",
                      "install it, so the program can be verified against it", "warning")
    builds = [r for r in reqs.values() if "+" in r.version]
    if builds:
        a.problem("local-build", "requirements",
                  "; ".join(f"{r.distribution} {r.version} is a local build: the program asks for {r.spec}"
                            for r in builds),
                  "install the matching build where the program runs (e.g. torch from the PyTorch index for "
                  "your CUDA)", "warning")
    local = [r for r in lock(reqs.values()) if r.local_path]
    if local:
        a.problem("local-install", "requirements",
                  "installed from folders on this machine: " + ", ".join(f"{r.distribution} ({r.local_path})"
                                                                         for r in local),
                  "the saved program loads where those folders exist; publish them, or install released "
                  "versions, to load it anywhere", "warning")
    report = Report(entry, a.nodes, a.bindings, reqs, a.problems, a.packages, declared)
    return report


__all__ = ["check", "Report", "Problem", "Refused", "Requirement", "lock"]
