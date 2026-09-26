"""From a Python function to an lmcc signature.

The parameters are the inputs. The outputs are what the body declares with
``_ai`` (``reasoning: str = _ai["think first"]``) plus the primary output
(``result`` for ``return _ai``, or the variable the body returns). The
docstring, inline comments and ``_ai[...]`` descriptions become the
instruction and the fields' descriptions. Types become JSON-Schema shapes.

This module only reads code; it never calls a model.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import textwrap
import typing
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import lmcc
from lmcc import core as lmcc_core

from .docments import (_class_field_docments, _harvest_ai_output_inline_comments,
                       _harvest_inline_param_and_return_comments, flexiclass)

MAIN_OUTPUT_DEFAULT_NAME = "result"
RESERVED_PARAMS = frozenset({"_prediction", "all"})

# ──────────────────────────────────────────────────────────────────────────────
# Type-hint helpers
# ──────────────────────────────────────────────────────────────────────────────
def _is_type_hint_like(tp: Any) -> bool:
    """Return True if tp looks like a usable type hint (builtins, typing generics, PEP 585 generics)."""
    if tp is None or tp is inspect._empty:
        return False
    try:
        if isinstance(tp, type):
            return True
    except Exception:
        pass
    try:
        # typing.List[str], list[str], Union, Annotated, etc.
        if typing.get_origin(tp) is not None:
            return True
    except Exception:
        pass
    # Fall back: many typing constructs live under typing.*
    mod = getattr(tp, "__module__", "")
    return mod.startswith("typing")

def _raw_return_annotation(fn: Any) -> Any:
    """Return the raw function return annotation if present, else None (do not coerce)."""
    sig = inspect.signature(fn)
    hints = _safe_get_type_hints(fn)
    if "return" in hints:
        return hints["return"]
    return sig.return_annotation if sig.return_annotation is not inspect._empty else None

def _strip_annotated_optional(tp: Any) -> Any:
    """Remove Annotated[...] and Optional[...] (Union[..., None]) wrappers for comparison."""
    try:
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)
        # Annotated[T, ...] -> T
        if origin is typing.Annotated and args:
            return _strip_annotated_optional(args[0])
        # Optional[T] -> T ; Union[T, None] -> T
        if origin is typing.Union and args:
            core = [a for a in args if a is not type(None)]  # noqa: E721
            if len(core) == 1:
                return _strip_annotated_optional(core[0])
    except Exception:
        pass
    return tp

def _is_any(tp: Any) -> bool:
    return tp is Any or str(tp) == "typing.Any"

def _types_compatible(a: Any, b: Any) -> bool:
    """Conservatively decide if two hints are compatible."""
    if a is None or b is None:
        return True
    if _is_any(a) or _is_any(b):
        return True
    a = _strip_annotated_optional(a)
    b = _strip_annotated_optional(b)
    oa, aa = typing.get_origin(a), typing.get_args(a)
    ob, ab = typing.get_origin(b), typing.get_args(b)
    # Plain types
    if oa is None and ob is None:
        return a == b
    # Generics must share origin and have pairwise compatible args
    if oa != ob:
        return False
    if len(aa) != len(ab):
        return False
    return all(_types_compatible(x, y) for x, y in zip(aa, ab))

def _hint_str(tp: Any) -> str:
    try:
        return getattr(tp, "__name__", str(tp))
    except Exception:
        return str(tp)

def _compose_system_doc(fn: Any, *, include_fn_name: bool) -> str:
    # history are removed. Docstring is the instruction; optional function name.
    parts = []
    if include_fn_name and getattr(fn, "__name__", None):
        parts.append(f"Function: {fn.__name__}")
    base = (fn.__doc__ or "").strip()
    if base:
        parts.append(base)
    return "\n\n".join([p for p in parts if p]).strip()

# ──────────────────────────────────────────────────────────────────────────────
# AST-based collection of declared outputs (x: T = _ai["desc"])
# ──────────────────────────────────────────────────────────────────────────────

def _eval_annotation(expr: ast.AST, env: Dict[str, Any]) -> Any:
    try:
        code = compile(ast.Expression(expr), filename="<ann>", mode="eval")
        return eval(code, env, {})
    except Exception:
        return str

def _extract_desc_from_subscript(node: ast.Subscript) -> str:
    try:
        sl = node.slice
        if isinstance(sl, ast.Index):
            sl = sl.value  # py3.8 compat
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return str(sl.value)
        if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
            second = sl.elts[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str):
                return str(second.value)
    except Exception:
        pass
    return ""

def _collect_ast_outputs(fn: Any) -> List[Tuple[str, Any, str]]:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []

    # Find our function node
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return []

    outputs_ordered: List[Tuple[str, Any, str]] = []
    env = dict(fn.__globals__)
    env.setdefault("typing", typing)

    for node in ast.walk(fn_node):
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                continue
            name = node.target.id
            val = node.value
            if val is None:
                continue
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "_ai"
            if not (is_ai or is_ai_sub):
                continue
            typ = _eval_annotation(node.annotation, env) if node.annotation is not None else str
            desc = _extract_desc_from_subscript(val) if is_ai_sub else ""
            if not any(n == name for n, _, _ in outputs_ordered):
                outputs_ordered.append((name, typ, desc))
        elif isinstance(node, ast.Assign):
            if not node.targets:
                continue
            val = node.value
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "_ai"
            if not (is_ai or is_ai_sub):
                continue
            desc = _extract_desc_from_subscript(val) if is_ai_sub else ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    name = tgt.id
                    if not any(n == name for n, _, _ in outputs_ordered):
                        outputs_ordered.append((name, None, desc))
    return outputs_ordered

class _ReturnInfo(TypedDict, total=False):
    mode: str  # 'name' | 'sentinel' | 'ellipsis' | 'empty' | 'other'
    name: Optional[str]

def _collect_return_info(fn: Any) -> _ReturnInfo:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return {"mode": "other", "name": None}
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return {"mode": "other", "name": None}
    ret: _ReturnInfo = {"mode": "other", "name": None}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return):
            val = node.value
            if val is None:
                ret = {"mode": "empty", "name": None}
            elif isinstance(val, ast.Name):
                if val.id == "_ai":
                    ret = {"mode": "sentinel", "name": None}
                else:
                    ret = {"mode": "name", "name": val.id}
            elif isinstance(val, ast.Constant) and val.value is Ellipsis:
                ret = {"mode": "ellipsis", "name": None}
            else:
                ret = {"mode": "other", "name": None}
    return ret

def _extract_return_names(fn: Any) -> List[str]:
    """Best-effort: extract variable names referenced in `return (...)` or
    `return [...]` constructs. Used to map bare `_ai` placeholders to concrete
    output field names by position.

    Example: for `return (id, email)`, returns ["id", "email"].
    """
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return []
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return []
    names: List[str] = []
    last_ret: Optional[ast.Return] = None
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return):
            last_ret = node
    if last_ret is None or last_ret.value is None:
        return []
    val = last_ret.value
    elts: List[ast.AST] = []
    if isinstance(val, (ast.Tuple, ast.List)):
        elts = list(val.elts)
    elif isinstance(val, ast.Name):
        return [val.id]
    else:
        return []
    for e in elts:
        if isinstance(e, ast.Name):
            names.append(e.id)
    return names

def _safe_get_type_hints(fn: Any) -> Dict[str, Any]:
    """Best-effort type_hints that won't error on unknown/forward-ref annotations.
    Falls back to raw __annotations__ if evaluation fails.
    """
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:
        anns = getattr(fn, "__annotations__", {}) or {}
        return dict(anns)

def _return_label_from_ast(fn: Any) -> Optional[str]:
    """Extract a textual label from the return annotation (e.g., -> "french")."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        # Fallback: inspect raw annotations
        try:
            anns = getattr(fn, "__annotations__", {}) or {}
            ret = anns.get("return")
            if isinstance(ret, str):
                return ret
        except Exception:
            pass
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            ann = node.returns
            if isinstance(ann, ast.Name):
                return ann.id
            if isinstance(ann, ast.Attribute):
                # attr chain like lang.French -> "French" or "lang.French"
                parts: List[str] = []
                cur = ann
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                return ".".join(reversed(parts)) if parts else None
            if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
                return str(ann.value)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Types → JSON-Schema shapes (what lmcc needs) and back to Python values
# ──────────────────────────────────────────────────────────────────────────────

_UNION_TYPES: Tuple[Any, ...] = (typing.Union,)
try:  # PEP 604 unions (int | None)
    import types as _pytypes
    _UNION_TYPES = (typing.Union, _pytypes.UnionType)
except AttributeError:  # pragma: no cover - Python < 3.10
    pass

_SEQUENCES = {list, typing.List, typing.Sequence, typing.MutableSequence}
try:
    import collections.abc as _abc
    _SEQUENCES |= {_abc.Sequence, _abc.MutableSequence, _abc.Iterable}
    _MAPPINGS = {dict, typing.Dict, typing.Mapping, _abc.Mapping, _abc.MutableMapping}
except Exception:  # pragma: no cover
    _MAPPINGS = {dict, typing.Dict, typing.Mapping}


def unannotate(ann: Any) -> Tuple[Any, Optional[str]]:
    """``Annotated[T, "a description", ...]`` → ``(T, "a description")``."""
    desc = None
    while typing.get_origin(ann) is typing.Annotated:
        ann, *extras = typing.get_args(ann)
        for extra in extras:
            if isinstance(extra, str) and desc is None:
                desc = extra
    return ann, desc


def _is_pydantic_model(tp: Any) -> bool:
    return isinstance(tp, type) and callable(getattr(tp, "model_json_schema", None)) \
        and callable(getattr(tp, "model_validate", None))


def shape_of(ann: Any, registry: Any = None, *, where: str = "?") -> dict:
    """A Python annotation as the JSON-Schema shape lmcc lowers it to.

    Beyond what lmcc maps itself (scalars, Literal, Enum, list, dict, Optional,
    dataclasses), functai maps ``Any`` (any JSON), tuples, sets, TypedDicts,
    pydantic models, and plain classes with annotations (made dataclasses with
    ``flexiclass``). A type bound with ``lmcc.format`` keeps its bound shape.
    Anything else refuses by name (lmcc ``unmapped-type``), never a silent ``str()``."""
    ann, _ = unannotate(ann)
    if ann is typing.Any or ann is object:
        return {}
    if ann is None or ann is type(None):
        return {"type": "null"}
    if ann in (str, int, float, bool):
        return lmcc_core.annotation_to_shape(ann, registry, field_name=where)
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    if origin is typing.Literal:
        return lmcc_core.annotation_to_shape(ann, registry, field_name=where)
    if origin in _UNION_TYPES:
        return {"anyOf": [shape_of(a, registry, where=where) for a in args]}
    if ann is tuple or origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": shape_of(args[0], registry, where=f"{where}[]")}
        return {"type": "array", "prefixItems": [shape_of(a, registry, where=f"{where}[{i}]")
                                                 for i, a in enumerate(args)],
                "minItems": len(args), "maxItems": len(args)}
    if ann in (set, frozenset) or origin in (set, frozenset, typing.Set, typing.FrozenSet):
        items = shape_of(args[0], registry, where=f"{where}[]") if args else {}
        return {"type": "array", "items": items, "uniqueItems": True}
    if ann is list or origin in _SEQUENCES:
        return {"type": "array", "items": shape_of(args[0], registry, where=f"{where}[]")} if args \
            else {"type": "array"}
    if ann is dict or origin in _MAPPINGS:
        if len(args) == 2 and args[1] not in (typing.Any, object):
            return {"type": "object", "additionalProperties": shape_of(args[1], registry, where=f"{where}{{}}")}
        return {"type": "object"}
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return lmcc_core.annotation_to_shape(ann, registry, field_name=where)
    if registry is not None:
        bound = registry.shape_of(ann)
        if bound is not None:
            return bound
    if _is_pydantic_model(ann):
        return ann.model_json_schema()
    if isinstance(ann, type) and typing.is_typeddict(ann):
        hints = typing.get_type_hints(ann)
        required = sorted(getattr(ann, "__required_keys__", hints))
        return {"type": "object", "properties": {k: shape_of(t, registry, where=f"{where}.{k}")
                                                 for k, t in hints.items()},
                "required": [k for k in hints if k in required]}
    if isinstance(ann, type) and not dataclasses.is_dataclass(ann) and getattr(ann, "__annotations__", None) \
            and ann.__module__ != "builtins":
        flexiclass(ann)
    if isinstance(ann, type) and dataclasses.is_dataclass(ann):
        hints = typing.get_type_hints(ann, include_extras=True)
        fields = [f for f in dataclasses.fields(ann) if f.init]
        return {"type": "object",
                "properties": {f.name: shape_of(hints.get(f.name, typing.Any), registry, where=f"{where}.{f.name}")
                               for f in fields},
                "required": [f.name for f in fields]}
    return lmcc_core.annotation_to_shape(ann, registry, field_name=where)


def coerce(ann: Any, value: Any) -> Any:
    """What JSON cannot carry, back to the declared Python type: tuples and sets
    (lists on the wire), inside containers and Optionals. Everything else
    (dataclasses, pydantic, Enums) the reading format already built."""
    ann, _ = unannotate(ann)
    if value is None or ann is None:
        return value
    if ann is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)                     # JSON wrote 999 for a float field
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    if origin in _UNION_TYPES:
        real = [a for a in args if a is not type(None)]
        return coerce(real[0], value) if len(real) == 1 else value
    if isinstance(ann, type) and dataclasses.is_dataclass(ann) and isinstance(value, ann):
        hints = typing.get_type_hints(ann)
        for f in dataclasses.fields(value):
            current = getattr(value, f.name)
            fixed = coerce(hints.get(f.name), current)
            if fixed is not current:
                object.__setattr__(value, f.name, fixed)
        return value
    if (ann is tuple or origin is tuple) and isinstance(value, list):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(coerce(args[0], v) for v in value)
        if args and len(args) == len(value):
            return tuple(coerce(a, v) for a, v in zip(args, value))
        return tuple(value)
    if (ann in (set, frozenset) or origin in (set, frozenset, typing.Set, typing.FrozenSet)) \
            and isinstance(value, (list, tuple, set)):
        kind = frozenset if (ann is frozenset or origin in (frozenset, typing.FrozenSet)) else set
        return kind(coerce(args[0], v) if args else v for v in value)
    if (ann is list or origin in _SEQUENCES) and isinstance(value, list) and args:
        return [coerce(args[0], v) for v in value]
    if (ann is dict or origin in _MAPPINGS) and isinstance(value, dict) and len(args) == 2:
        return {k: coerce(args[1], v) for k, v in value.items()}
    return value


def _field(name: str, direction: str, ann: Any, registry: Any, *, desc: Optional[str] = None,
           purpose: str = "plain", output: bool = False) -> lmcc_core.Field:
    base, adesc = unannotate(ann)
    return lmcc_core.Field(name, direction, shape_of(base, registry, where=name),
                           type=lmcc_core.typename(base), purpose=purpose,
                           desc=None if output else ((desc or adesc) or None), annotation=base)


# ──────────────────────────────────────────────────────────────────────────────
# Instruction text
# ──────────────────────────────────────────────────────────────────────────────

def _instruction_appendix(fn: Any, *, outputs: List[Tuple[str, str]], main_output_type: Any) -> str:
    """Guidance harvested from the code: parameter comments, ``_ai`` descriptions
    and comments, the return comment, and the documented fields of a returned
    class. Only what was written down; undocumented names add nothing."""
    parts: List[str] = []
    param_docs, ret_cmt = _harvest_inline_param_and_return_comments(fn)
    documented = [(k, v) for k, v in param_docs.items() if v]
    if documented:
        parts += ["Parameter guidance:", *[f"- {k}: {v}" for k, v in documented], ""]
    described = [(n, d) for n, d in outputs if d]
    if described:
        parts += ["Output guidance:", *[f"- {n}: {d}" for n, d in described], ""]
    if ret_cmt:
        parts += [f"Return guidance: {ret_cmt}", ""]
    if isinstance(main_output_type, type) and getattr(main_output_type, "__annotations__", None):
        src_docs = _class_field_docments(main_output_type) or {}
        meta_docs: Dict[str, Optional[str]] = {}
        if dataclasses.is_dataclass(main_output_type):
            meta_docs = {f.name: (f.metadata.get("doc") if f.metadata else None)
                         for f in dataclasses.fields(main_output_type)}
        cls_name = getattr(main_output_type, "__name__", "Object")
        lines = [f"- {cls_name}.{k}: {src_docs.get(k) or meta_docs.get(k)}"
                 for k in main_output_type.__annotations__ if src_docs.get(k) or meta_docs.get(k)]
        if lines:
            parts += [f"{cls_name} fields:", *lines, ""]
    return "\n".join(parts).strip()


# ──────────────────────────────────────────────────────────────────────────────
# The spec: a function lowered to an lmcc signature, plus what functai needs
# to hand values back
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Spec:
    signature: lmcc.SignatureCore
    main: str                          # the primary output (what `return _ai` returns)
    outputs: Tuple[str, ...]           # the function's own outputs, declaration order, main last
    annotations: Dict[str, Any]        # output name → declared Python type
    params: Tuple[str, ...]
    reasoning: bool = False
    tools: bool = False


def build_spec(fn: Any, *, instructions: Optional[str] = None, include_fn_name: bool = True,
               reasoning: bool = False, tools: bool = False, registry: Any = None) -> Spec:
    sig = inspect.signature(fn)
    hints = _safe_get_type_hints(fn)
    param_docs, ret_cmt = _harvest_inline_param_and_return_comments(fn)

    # ---- inputs
    inputs: List[lmcc_core.Field] = []
    for pname, p in sig.parameters.items():
        if pname in RESERVED_PARAMS:
            raise ValueError(f"{fn.__name__}: parameter name {pname!r} is reserved by functai.")
        if tools and pname in ("tools", "calls"):
            raise ValueError(f"{fn.__name__}: with tools, the parameter name {pname!r} is reserved "
                             f"(it carries the tool definitions and calls).")
        ann = hints.get(pname, p.annotation if p.annotation is not inspect._empty else str)
        if not _is_type_hint_like(unannotate(ann)[0]):
            ann = str
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            ann = typing.List[ann]  # type: ignore[valid-type]
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            ann = typing.Dict[str, ann]  # type: ignore[valid-type]
        inputs.append(_field(pname, "input", ann, registry, desc=param_docs.get(pname)))
    input_names = {f.name for f in inputs}

    # ---- outputs declared in the body, and the primary one
    ast_outputs = _collect_ast_outputs(fn)
    ret_label = _return_label_from_ast(fn)
    ret_info = _collect_return_info(fn)
    order_names = [n for n, _, _ in ast_outputs]
    mode = ret_info.get("mode")
    if mode == "name" and ret_info.get("name") in order_names:
        main_name = typing.cast(str, ret_info.get("name"))
    elif mode in {"sentinel", "ellipsis"}:
        main_name = MAIN_OUTPUT_DEFAULT_NAME
    elif order_names:
        main_name = order_names[-1]
    elif ret_label and str(ret_label).isidentifier() and not _is_type_hint_like(_raw_return_annotation(fn)):
        main_name = str(ret_label)
    else:
        main_name = MAIN_OUTPUT_DEFAULT_NAME

    ai_inline = _harvest_ai_output_inline_comments(fn)
    ast_map: Dict[str, Tuple[Any, str]] = {}
    for n, t, d in ast_outputs:
        extra = ai_inline.get(n)
        ast_map[n] = (t, f"{d} — {extra}" if (d and extra) else (d or extra or ""))

    fn_ret_raw = _raw_return_annotation(fn)
    fn_ret_hint = fn_ret_raw if _is_type_hint_like(fn_ret_raw) else None
    if isinstance(fn_ret_hint, typing.ForwardRef):
        fn_ret_hint = None
    main_desc = ""
    if main_name in ast_map:
        t0, d0 = ast_map[main_name]
        explicit_var_t = t0 if _is_type_hint_like(t0) else None
        if mode == "name" and ret_info.get("name") == main_name:
            if explicit_var_t is not None and fn_ret_hint is not None and not _types_compatible(explicit_var_t, fn_ret_hint):
                raise TypeError(
                    f"Type mismatch for main output '{main_name}': "
                    f"function return is {_hint_str(fn_ret_hint)} but '{main_name}' is annotated as {_hint_str(explicit_var_t)}. "
                    f"Fix one of: (a) annotate '{main_name}: {_hint_str(fn_ret_hint)} = _ai', "
                    f"(b) remove the variable annotation to inherit the function return type, "
                    f"(c) change the function return annotation.")
            main_typ, main_desc = (explicit_var_t or fn_ret_hint or str), d0
        elif mode in {"sentinel", "ellipsis"}:
            main_typ = fn_ret_hint or str
        else:
            main_typ, main_desc = (explicit_var_t or fn_ret_hint or str), d0
    else:
        main_typ = fn_ret_hint or str
    if not main_desc and ret_cmt:
        main_desc = ret_cmt

    if main_name in input_names:           # an input already has the name: fall back
        for candidate in (MAIN_OUTPUT_DEFAULT_NAME, "output", "answer"):
            if candidate not in input_names and candidate not in ast_map:
                main_name = candidate
                break

    extras = [(n, ast_map[n][0] if _is_type_hint_like(ast_map[n][0]) else str, ast_map[n][1])
              for n in order_names if n != main_name]
    outputs: List[lmcc_core.Field] = []
    if reasoning and "reasoning" not in ast_map and main_name != "reasoning" and "reasoning" not in input_names:
        outputs.append(lmcc_core.Field("reasoning", "output", {"type": "string"}, type="str",
                                       purpose="reasoning", annotation=str))
    if tools:
        from lmcc_std.tools import Tool, ToolCall
        inputs.append(lmcc_core.Field("tools", "input", lmcc_core.annotation_to_shape(list[Tool], registry, field_name="tools"),
                                      type=lmcc_core.typename(list[Tool]), purpose="tools", annotation=list[Tool]))
        outputs.append(lmcc_core.Field("calls", "output",
                                       lmcc_core.annotation_to_shape(list[ToolCall], registry, field_name="calls"),
                                       type=lmcc_core.typename(list[ToolCall]), purpose="tools.calls",
                                       annotation=list[ToolCall]))
    # Output descriptions go into the instruction ("Output guidance"), not the
    # fields: lmcc shows a field's desc *instead of* its format hint in the reply
    # pattern, and a structured output must keep its JSON hint.
    for n, t, _d in extras:
        outputs.append(_field(n, "output", t, registry, output=True))
    outputs.append(_field(main_name, "output", main_typ, registry, output=True))

    # ---- instruction
    if instructions is not None:
        text = instructions.strip()
    else:
        base = _compose_system_doc(fn, include_fn_name=include_fn_name)
        appendix = _instruction_appendix(fn, outputs=[(n, d) for n, _t, d in extras] + [(main_name, main_desc)],
                                         main_output_type=unannotate(main_typ)[0])
        text = base if not appendix else (base + ("\n\n" if base else "") + appendix)

    signature = lmcc_core._validated(lmcc_core.SignatureCore(text, inputs + outputs))
    own = tuple([n for n, _t, _d in extras] + [main_name])
    annotations = {f.name: f.annotation for f in outputs}
    return Spec(signature=signature, main=main_name, outputs=own, annotations=annotations,
                params=tuple(sig.parameters), reasoning=any(f.purpose == "reasoning" for f in outputs),
                tools=tools)


def describe_signature(spec: Spec, name: str) -> str:
    """A one-line summary: ``Signature: summarize | Inputs: text:str | Outputs: result*``."""
    sig = spec.signature
    doc = sig.instructions
    parts = [f"Signature: {name}"]
    if doc:
        parts.append("Doc: " + (doc[:120] + ("…" if len(doc) > 120 else "")))
    ins = [f"{f.name}:{f.type or 'Any'}" for f in sig.inputs if f.purpose == "plain"]
    if ins:
        parts.append("Inputs: " + ", ".join(ins))
    outs = [f"{f.name}{'*' if f.name == spec.main else ''}" for f in sig.outputs if f.purpose != "tools.calls"]
    if outs:
        parts.append("Outputs: " + ", ".join(outs))
    return " | ".join(parts)
