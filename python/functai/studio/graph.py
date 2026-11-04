"""The map of a pipeline, read from its code (never by running it).

For a ``@module``, the body is read statement by statement: which step's answer
reaches which step (through variables, conditions and expressions), under which
condition (``if kind == "spam"``), what is returned, and fixed values that stand
in for a step on some paths (``"Love it!" if ... else answer(...)``). A program
that is one AI function is a one-step map. Where the code cannot be followed
(a function passed in, ``getattr``), ``flows`` is false and the steps are
listed in the order the code calls them, without wires between them.

The result is plain data with a deterministic layout (the same code always
draws the same picture)::

    {"flows": true, "nodes": [{"id", "kind", "label", "x", "y", "w", "h", ...}],
     "edges": [{"from", "to"}], "width", "height", "notes": [...]}

Node kinds: ``input``, ``step`` (an AI function or an inner module), ``const``
(a value from the file, such as an FAQ text), ``value`` (a fixed value some path
uses or returns), ``output`` (the pipeline's answer).
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import FunctAIFunc

SIZES = {"input": (180, 92), "step": (210, 116), "const": (160, 56), "value": (190, 62), "output": (140, 96)}
COL, ROW, PAD = 255, 122, 8


def _short(text: str, n: int = 40) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _negate(text: str) -> str:
    return f"not ({text})"


class _Walk:
    def __init__(self, steps: Dict[str, Any], inputs: List[str], glob: Dict[str, Any], line0: int):
        self.steps = steps
        self.inputs = inputs
        self.glob = glob
        self.line0 = line0
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Set[Tuple[str, str]] = set()
        self.env: Dict[str, Set[str]] = {}
        self.order: List[str] = []
        self.values = 0
        self.unfollowed: List[str] = []
        for name in inputs:
            self._node(f"in:{name}", kind="input", label=name)

    # ---- nodes and edges

    def _node(self, nid: str, **fields: Any) -> Dict[str, Any]:
        if nid not in self.nodes:
            self.nodes[nid] = {"id": nid, **fields}
            self.order.append(nid)
        return self.nodes[nid]

    def _edge(self, a: str, b: str) -> None:
        if a != b:
            self.edges.add((a, b))

    def _controlled(self, nid: str, conds: List[Tuple[str, Set[str]]]) -> None:
        """A condition is a connection too: what it reads decides whether this runs."""
        for _, ps in conds:
            for p in ps:
                if self.nodes.get(p, {}).get("kind") == "step":
                    self._edge(p, nid)

    def _line(self, e: ast.AST) -> int:
        return self.line0 + getattr(e, "lineno", 1) - 1

    def cond_text(self, test: ast.AST) -> str:
        """The condition as written, with a variable that holds one step's answer named after that step."""
        t = copy.deepcopy(test)
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                prods = self.env.get(n.id)
                if prods and len(prods) == 1:
                    only = next(iter(prods))
                    if self.nodes.get(only, {}).get("kind") == "step":
                        n.id = only
        try:
            return ast.unparse(t)
        except Exception:  # noqa: BLE001
            return "a condition"

    # ---- statements

    def body(self, stmts: List[ast.stmt], conds: List[str], loop: bool) -> None:
        for s in stmts:
            self.stmt(s, conds, loop)

    def bind(self, target: ast.AST, prods: Set[str]) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = set(prods)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                self.bind(t, prods)
        elif isinstance(target, ast.Starred):
            self.bind(target.value, prods)

    def stmt(self, s: ast.stmt, conds: List[str], loop: bool) -> None:
        if isinstance(s, ast.Assign):
            prods = self.value(s.value, conds, loop)
            for t in s.targets:
                self.bind(t, prods)
        elif isinstance(s, ast.AnnAssign) and s.value is not None:
            self.bind(s.target, self.value(s.value, conds, loop))
        elif isinstance(s, ast.AugAssign):
            prods = self.value(s.value, conds, loop)
            if isinstance(s.target, ast.Name):
                self.env[s.target.id] = self.env.get(s.target.id, set()) | prods
        elif isinstance(s, ast.If):
            t = self.cond_text(s.test)
            ps = self.expr(s.test, conds, loop)
            self.body(s.body, conds + [(t, ps)], loop)
            self.body(s.orelse, conds + [(_negate(t), ps)], loop)
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            self.bind(s.target, self.expr(s.iter, conds, loop))
            self.body(s.body, conds, True)
            self.body(s.orelse, conds, loop)
        elif isinstance(s, ast.While):
            self.expr(s.test, conds, True)
            self.body(s.body, conds, True)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for item in s.items:
                self.expr(item.context_expr, conds, loop)
            self.body(s.body, conds, loop)
        elif isinstance(s, ast.Try):
            self.body(s.body, conds, loop)
            for h in s.handlers:
                self.body(h.body, conds, loop)
            self.body(s.orelse, conds, loop)
            self.body(s.finalbody, conds, loop)
        elif isinstance(s, ast.Return):
            if s.value is None:
                return
            for p in self.value(s.value, conds, loop, returns=True):
                self._edge(p, "out")
        elif isinstance(s, ast.Expr):
            self.expr(s.value, conds, loop)
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.unfollowed.append(f"a function defined inside the pipeline ({s.name}, line {self._line(s)})")

    # ---- expressions

    def value(self, e: ast.AST, conds: List[str], loop: bool, *, returns: bool = False) -> Set[str]:
        """An expression whose result goes somewhere: a fixed value becomes a node of its own."""
        if isinstance(e, ast.IfExp):
            t = self.cond_text(e.test)
            ps = self.expr(e.test, conds, loop)
            before = set(self.nodes)
            a = self.value(e.body, conds + [(t, ps)], loop, returns=returns)
            mid = set(self.nodes)
            b = self.value(e.orelse, conds + [(_negate(t), ps)], loop, returns=returns)
            new_a = mid - before
            new_b = set(self.nodes) - mid
            steps_a = sorted(n for n in new_a | (a & set(self.steps)) if n in self.steps)
            steps_b = sorted(n for n in new_b | (b & set(self.steps)) if n in self.steps)
            for v in a:
                if self.nodes.get(v, {}).get("kind") == "value":
                    self.nodes[v]["alt"] = sorted(set(self.nodes[v].get("alt", [])) | set(steps_b))
            for v in b:
                if self.nodes.get(v, {}).get("kind") == "value":
                    self.nodes[v]["alt"] = sorted(set(self.nodes[v].get("alt", [])) | set(steps_a))
            return a | b
        if isinstance(e, ast.Constant) or (isinstance(e, ast.JoinedStr) and not any(
                isinstance(v, ast.FormattedValue) for v in e.values)):
            value = e.value if isinstance(e, ast.Constant) else ast.unparse(e)
            self.values += 1
            nid = f"val:{self.values}"
            self._node(nid, kind="value", label=_short(repr(value) if not isinstance(value, str) else
                                                        f'"{value}"', 34),
                       value=value, when=conds[-1][0] if conds else None, returns=returns, line=self._line(e),
                       col=getattr(e, "col_offset", None), editable=isinstance(e, ast.Constant), alt=[])
            self._controlled(nid, conds)
            return {nid}
        return self.expr(e, conds, loop)

    def expr(self, e: Optional[ast.AST], conds: List[str], loop: bool) -> Set[str]:
        if e is None:
            return set()
        if isinstance(e, ast.Name):
            if e.id in self.env:
                return set(self.env[e.id])
            if e.id in self.inputs:
                return {f"in:{e.id}"}
            if e.id in self.glob:
                v = self.glob[e.id]
                if isinstance(v, (str, int, float, list, tuple, dict)) and not isinstance(v, bool):
                    sub = (f"text · {len(v.splitlines()) or 1} lines" if isinstance(v, str) and "\n" in v
                           else f"text · {len(v.split())} words" if isinstance(v, str)
                           else type(v).__name__)
                    self._node(f"const:{e.id}", kind="const", label=e.id, sub=sub)
                    return {f"const:{e.id}"}
            return set()
        if isinstance(e, ast.Call):
            prods: Set[str] = set()
            for a in list(e.args) + [k.value for k in e.keywords]:
                prods |= self.expr(a.value if isinstance(a, ast.Starred) else a, conds, loop)
            fname = e.func.id if isinstance(e.func, ast.Name) else None
            if fname and fname in self.steps:
                node = self._node(fname, kind="step", label=fname, line=self._line(e),
                                  when=conds[-1][0] if conds else None, loop=loop)
                self._controlled(fname, conds)
                if loop:
                    node["loop"] = True
                for p in prods:
                    self._edge(p, fname)
                return {fname}
            if isinstance(e.func, ast.Attribute):
                prods |= self.expr(e.func.value, conds, loop)
                if e.func.attr in ("map", "vectorize") and isinstance(e.func.value, ast.Name) \
                        and e.func.value.id in self.steps:
                    node = self._node(e.func.value.id, kind="step", label=e.func.value.id, line=self._line(e),
                                      when=conds[-1][0] if conds else None, loop=True)
                    self._controlled(e.func.value.id, conds)
                    node["loop"] = True
                    for p in prods - {e.func.value.id}:
                        self._edge(p, e.func.value.id)
                    return {e.func.value.id}
            elif fname in ("getattr", "eval", "exec"):
                self.unfollowed.append(f"{fname}(…) on line {self._line(e)}")
            return prods
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            out: Set[str] = set()
            for g in e.generators:
                it = self.expr(g.iter, conds, loop)
                self.bind(g.target, it)
                out |= it
            elts = [e.key, e.value] if isinstance(e, ast.DictComp) else [e.elt]
            for el in elts:
                out |= self.expr(el, conds, True)
            return out
        if isinstance(e, ast.Lambda):
            return set()
        out = set()
        for child in ast.iter_child_nodes(e):
            if isinstance(child, ast.expr):
                out |= self.expr(child, conds, loop)
        return out


def _layout(nodes: List[Dict[str, Any]], edges: List[Dict[str, str]]) -> Tuple[int, int]:
    """Columns by the longest path from the inputs; rows in source order; output last."""
    preds: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    succs: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        preds[e["to"]].append(e["from"])
        succs[e["from"]].append(e["to"])
    layer: Dict[str, int] = {}
    visiting: Set[str] = set()

    def depth(nid: str) -> int:
        if nid in layer:
            return layer[nid]
        if nid in visiting:          # a loop in the data: stop here
            return 0
        visiting.add(nid)
        d = 0 if not preds[nid] else 1 + max(depth(p) for p in preds[nid])
        visiting.discard(nid)
        layer[nid] = d
        return d

    kinds = {n["id"]: n["kind"] for n in nodes}
    for n in nodes:
        if n["kind"] in ("step", "input"):
            depth(n["id"])
    # fixed values and file values sit just before what uses them
    for n in nodes:
        if n["kind"] in ("const", "value") and n["id"] not in layer:
            uses = [depth(s) for s in succs[n["id"]] if kinds[s] != "output"]
            layer[n["id"]] = max(0, min(uses) - 1) if uses else (max(layer.values(), default=0))
    last = max([layer[n["id"]] for n in nodes if n["kind"] != "output"], default=0) + 1
    for n in nodes:
        if n["kind"] == "output":
            layer[n["id"]] = last
    # returned fixed values: in the column before the output
    for n in nodes:
        if n["kind"] == "value" and n.get("returns"):
            layer[n["id"]] = last - 1 if last > 0 else 0
    columns: Dict[int, List[Dict[str, Any]]] = {}
    rank = {"input": 0, "const": 1, "value": 2, "step": 1, "output": 3}
    for n in nodes:
        columns.setdefault(layer[n["id"]], []).append(n)
    tallest = max((sum(SIZES[n["kind"]][1] + 26 for n in col) for col in columns.values()), default=0)
    for c, col in columns.items():
        col.sort(key=lambda n: (n.get("line") or 0, rank[n["kind"]]))
        total = sum(SIZES[n["kind"]][1] + 26 for n in col)
        y = PAD + (tallest - total) / 2
        for n in col:
            w, h = SIZES[n["kind"]]
            n.update(x=PAD + c * COL, y=round(y), w=w, h=h, column=c)
            y += h + 26
    width = PAD * 2 + max(columns) * COL + max(SIZES[n["kind"]][0] for n in nodes) if nodes else 0
    return int(width), int(tallest + PAD * 2)


def build_map(project: Any) -> Dict[str, Any]:
    """The map of the project's program."""
    program = project.load()
    steps = project.steps()
    iface = program.interface
    inputs = [f["name"] for f in iface.get("inputs", [])]
    answer = (iface.get("outputs") or [{"name": "result"}])[-1]["name"]
    notes: List[str] = []
    flows = True
    if isinstance(program, FunctAIFunc):
        name = program.__name__
        nodes = [{"id": f"in:{i}", "kind": "input", "label": i} for i in inputs]
        nodes.append({"id": name, "kind": "step", "label": name, "line": _line_of(program)})
        edges = [{"from": f"in:{i}", "to": name} for i in inputs] + [{"from": name, "to": "out"}]
    else:
        fn = program._fn
        try:
            lines, line0 = inspect.getsourcelines(fn)
            tree = ast.parse(textwrap.dedent("".join(lines)))
            fdef = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        except (OSError, TypeError, SyntaxError, StopIteration) as exc:
            fdef, line0 = None, 1
            notes.append(f"the code could not be read ({type(exc).__name__}); steps are shown in call order")
        w = _Walk(steps, inputs, getattr(program, "_globals", {}) or {}, line0 + _decorator_lines(fdef))
        if fdef is not None:
            body = fdef.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]                      # the docstring
            w.body(body, [], False)
        for name in steps:
            if name not in w.nodes:
                flows = False
                w._node(name, kind="step", label=name, line=_line_of(steps[name]))
        if w.unfollowed:
            notes.append("the code does things the map cannot follow: " + "; ".join(w.unfollowed))
        # fixed values nobody uses are not part of any path
        used = {a for a, _ in w.edges} | {b for _, b in w.edges}
        nodes = [n for nid, n in w.nodes.items() if n["kind"] not in ("value", "const") or nid in used]
        edges = [{"from": a, "to": b} for a, b in sorted(w.edges) if b == "out" or b in w.nodes]
        if not any(e["to"] == "out" for e in edges):
            flows = False
        if not flows:
            notes.append("some steps' connections could not be read from the code: they are shown in the "
                         "order the code calls them")
    nodes.append({"id": "out", "kind": "output", "label": answer})
    ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["from"] in ids and e["to"] in ids]
    for n in nodes:
        if n["kind"] == "step":
            n.update(_step_facts(steps[n["id"]]))
    width, height = _layout(nodes, edges)
    return {"flows": flows, "nodes": nodes, "edges": edges, "width": width, "height": height, "notes": notes,
            "answer": answer}


def _decorator_lines(fdef: Optional[ast.AST]) -> int:
    """inspect gives the first line of the decorators; ast numbers lines from there too."""
    return 0


def _line_of(obj: Any) -> Optional[int]:
    fn = getattr(obj, "_fn", None) or getattr(obj, "fn", None) or getattr(obj, "__wrapped__", None)
    try:
        return inspect.getsourcelines(fn)[1] if fn is not None else None
    except (OSError, TypeError):
        return None


def _step_facts(step: Any) -> Dict[str, Any]:
    """What a step's card says: its instruction, its answers, its model."""
    try:
        iface = step.interface
    except Exception:  # noqa: BLE001
        iface = {}
    outs = iface.get("outputs") or []
    main = outs[-1] if outs else {}
    choices = (main.get("shape") or {}).get("enum")
    facts: Dict[str, Any] = {
        "instruction": (iface.get("description") or "").strip(),
        "takes": [f["name"] for f in iface.get("inputs", [])],
        "gives": main.get("type") or (main.get("shape") or {}).get("type"),
        "choices": choices,
        "own_lm": getattr(step, "lm", None) if isinstance(step, FunctAIFunc) else None,
        "is_module": not isinstance(step, FunctAIFunc),
        "def_line": _line_of(step),
    }
    sig = ", ".join(facts["takes"]) + " → " + (" · ".join(map(str, choices)) if choices else str(facts["gives"] or "?"))
    facts["signature"] = sig
    return facts
