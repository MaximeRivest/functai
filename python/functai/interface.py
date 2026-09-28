"""A program's interface: the inputs it takes and the outputs it gives
(contract/programs.md).

Every program has one. An AI function's comes from its definition; a
module's is derived from its Python function (or declared as data):

    @module
    def support(message: str, tone: str = "kind") -> str:
        ...

    support.interface
    # {"description": "", "inputs": [{"name": "message", "shape": {"type": "string"}, "type": "str"},
    #   {"name": "tone", "shape": {"type": "string", "default": "kind"}, "type": "str", "optional": True}],
    #  "outputs": [{"name": "result", "shape": {"type": "string"}, "type": "str"}]}

An interface is checked when its program is defined (``check``), and a
module checks every call against it (``bind_inputs``, ``check_outputs``).
"Fits" reads only the keywords programs.md lists, the same in every
language: no JSON Schema validator decides it. The interface's
``signature`` says what its data looks like; the call log records it as
``program.interface``.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import re
import typing
from collections.abc import Mapping
from typing import Annotated, Any, Dict, List, Optional, Tuple

import lmcc

from .errors import InterfaceError


class _JSONMark:
    def __repr__(self) -> str:
        return "functai.JSON"


_JSON_MARK = _JSONMark()

JSON = Annotated[Any, _JSON_MARK]
"""Any JSON value: annotate a module's input or output with it to say "data,
of any shape" (shape ``{}``). A plain ``Any``, ``object`` or no annotation
says the value may have no JSON form at all (an opaque field)."""

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
REF = re.compile(r"#/\$defs/([A-Za-z0-9_.-]+)")
TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")
WORDS: Dict[str, Any] = {"title": str, "description": str, "format": str, "$comment": str, "deprecated": bool,
                         "readOnly": bool, "writeOnly": bool, "examples": list, "default": object}
LISTED = frozenset({"type", "enum", "const", "anyOf", "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
                    "properties", "required", "additionalProperties", "minLength", "maxLength", "minimum", "maximum",
                    "exclusiveMinimum", "exclusiveMaximum", "$ref", "$defs"})
COUNTS = ("minItems", "maxItems", "minLength", "maxLength")
BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
INPUT_KEYS = frozenset({"name", "shape", "desc", "type", "opaque", "optional"})
OUTPUT_KEYS = frozenset({"name", "shape", "desc", "type", "opaque"})
_NOTHING = object()


# ------------------------------------------------------------------ JSON values


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def json_kind(v: Any) -> str:
    """The JSON type of a JSON value; an integer is a number with no fraction (``5.0`` is one)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if _is_number(v):
        return "integer" if (isinstance(v, int) or (math.isfinite(v) and float(v).is_integer())) else "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, (list, tuple)):
        return "array"
    if isinstance(v, Mapping):
        return "object"
    raise TypeError(f"{type(v).__name__} is not a JSON value")


def json_form(value: Any) -> Tuple[bool, Any]:
    """``(True, the value as JSON)``, or ``(False, None)`` for a value with no
    JSON form (a data frame, a file handle)."""
    try:
        return True, lmcc.turn.to_json(value, where="value")
    except lmcc.Refusal:
        return False, None
    except (TypeError, ValueError):
        return False, None


def _canonical(value: Any) -> str:
    from .calllog import canonical
    return canonical(value)


# ------------------------------------------------------------------ the vocabulary


def _shape_fault(shape: Any, root: Dict[str, Any], carry: bool) -> Optional[str]:
    """Why a shape breaks the vocabulary (a keyword not listed, in a module's
    shape; a listed keyword or a word with a value of the wrong kind; a
    reference that is not ``#/$defs/<name>`` of the shape's own), or None.
    ``carry``: an AI function's shape, whose other keywords are lmcc's."""
    if not isinstance(shape, dict):
        return "a shape is a JSON object"
    for k, v in shape.items():
        if k in WORDS:
            kind = WORDS[k]
            if kind is not object and not isinstance(v, kind):
                return f"{k!r} must be {'text' if kind is str else 'true or false' if kind is bool else 'a list'}"
            continue
        if k not in LISTED:
            if carry:
                continue
            return (f"{k!r} is not a keyword every language checks alike (programs.md lists them); declare the "
                    f"field opaque, or use a type the vocabulary can say")
        if k == "type":
            names = v if isinstance(v, list) else [v]
            if not names or not all(isinstance(n, str) and n in TYPES for n in names) or len(set(names)) != len(names):
                return f"'type' must be one of {list(TYPES)} or a list of them, each once"
        elif k == "enum":
            if not (isinstance(v, list) and v):
                return "'enum' must be a list of at least one value"
        elif k in ("anyOf", "prefixItems"):
            if not (isinstance(v, list) and v):
                return f"{k!r} must be a list of at least one shape"
            for x in v:
                fault = _shape_fault(x, root, carry)
                if fault:
                    return fault
        elif k == "items":
            fault = _shape_fault(v, root, carry)
            if fault:
                return fault
        elif k in ("properties", "$defs"):
            if not isinstance(v, dict):
                return f"{k!r} must map names to shapes"
            for x in v.values():
                fault = _shape_fault(x, root, carry)
                if fault:
                    return fault
        elif k == "additionalProperties":
            if not isinstance(v, bool):
                fault = _shape_fault(v, root, carry)
                if fault:
                    return fault
        elif k == "required":
            if not (isinstance(v, list) and all(isinstance(x, str) for x in v) and len(set(v)) == len(v)):
                return "'required' must be a list of names, each once"
        elif k in COUNTS:
            if not (_is_number(v) and json_kind(v) == "integer" and v >= 0):
                return f"{k!r} must be a whole number of at least 0"
        elif k in BOUNDS:
            if not _is_number(v):
                return f"{k!r} must be a number"
        elif k == "uniqueItems":
            if not isinstance(v, bool):
                return "'uniqueItems' must be true or false"
        elif k == "$ref":
            m = REF.fullmatch(v) if isinstance(v, str) else None
            if m is None:
                return f"'$ref' {v!r} must be #/$defs/<name>, the name of ASCII letters, digits, _, . and -"
            if m.group(1) not in (root.get("$defs") or {}):
                return f"'$ref' {v!r} names no entry of the shape's own $defs"
    return None


def _same_value_refs(shape: Dict[str, Any]) -> set:
    """The $defs a shape checks the very same value against: through $ref and anyOf."""
    out = set()
    ref = shape.get("$ref")
    if isinstance(ref, str):
        m = REF.fullmatch(ref)
        if m:
            out.add(m.group(1))
    for x in shape.get("anyOf") or []:
        if isinstance(x, dict):
            out |= _same_value_refs(x)
    return out


def _loops(root: Dict[str, Any]) -> bool:
    """A $defs entry that reaches itself through $ref and anyOf alone: checking would never end."""
    defs = root.get("$defs") or {}
    graph = {n: _same_value_refs(d) for n, d in defs.items() if isinstance(d, dict)}
    for start in graph:
        seen, todo = set(), list(graph[start])
        while todo:
            n = todo.pop()
            if n == start:
                return True
            if n not in seen:
                seen.add(n)
                todo.extend(graph.get(n, ()))
    return False


def _fits_shape(v: Any, shape: Dict[str, Any], root: Dict[str, Any]) -> bool:
    """A JSON value against a shape of the vocabulary (keywords not listed are never read)."""
    kind = json_kind(v)
    ref = shape.get("$ref")
    if isinstance(ref, str):
        target = (root.get("$defs") or {})[REF.fullmatch(ref).group(1)]   # type: ignore[union-attr]
        if not _fits_shape(v, target, root):
            return False
    if "type" in shape:
        names = shape["type"] if isinstance(shape["type"], list) else [shape["type"]]
        if kind not in names and not (kind == "integer" and "number" in names):
            return False
    if "enum" in shape and _canonical(v) not in {_canonical(x) for x in shape["enum"]}:
        return False
    if "const" in shape and _canonical(v) != _canonical(shape["const"]):
        return False
    if "anyOf" in shape and not any(_fits_shape(v, s, root) for s in shape["anyOf"]):
        return False
    if kind == "string":
        if len(v) < shape.get("minLength", 0) or ("maxLength" in shape and len(v) > shape["maxLength"]):
            return False
    elif kind in ("integer", "number"):
        if ("minimum" in shape and v < shape["minimum"]) or ("maximum" in shape and v > shape["maximum"]):
            return False
        if ("exclusiveMinimum" in shape and v <= shape["exclusiveMinimum"]) \
                or ("exclusiveMaximum" in shape and v >= shape["exclusiveMaximum"]):
            return False
    elif kind == "array":
        prefix = shape.get("prefixItems") or []
        if any(not _fits_shape(x, s, root) for x, s in zip(v, prefix)):
            return False
        if "items" in shape and any(not _fits_shape(x, shape["items"], root) for x in list(v)[len(prefix):]):
            return False
        if len(v) < shape.get("minItems", 0) or ("maxItems" in shape and len(v) > shape["maxItems"]):
            return False
        if shape.get("uniqueItems") and len({_canonical(x) for x in v}) != len(v):
            return False
    elif kind == "object":
        props = shape.get("properties") or {}
        if any(k not in v for k in shape.get("required") or []):
            return False
        extra = shape.get("additionalProperties", True)
        for k, x in v.items():
            if k in props:
                if not _fits_shape(x, props[k], root):
                    return False
            elif extra is False or (isinstance(extra, dict) and not _fits_shape(x, extra, root)):
                return False
    return True


def data_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    """A field's shape without its own ``default``: what its data looks like."""
    return {k: v for k, v in shape.items() if k != "default"}


def fits(value: Any, field: Mapping[str, Any]) -> bool:
    """Whether a value fits a field: an opaque field takes anything; any other
    takes a value with a JSON form that fits its shape (read by the listed
    keywords alone)."""
    if field.get("opaque"):
        return True
    ok, data = json_form(value)
    if not ok:
        return False
    shape = data_shape(field["shape"])
    return _fits_shape(data, shape, shape)


def fits_json(data: Any, field: Mapping[str, Any]) -> bool:
    """``fits`` for a value already in its JSON form."""
    if field.get("opaque"):
        return True
    shape = data_shape(field["shape"])
    return _fits_shape(data, shape, shape)


# ------------------------------------------------------------------ interfaces refused


def _refuse(field: Optional[str], why: str, program: str = "") -> InterfaceError:
    where = f"{program}: " if program else ""
    what = f"field {field!r}" if field is not None else "the interface"
    return InterfaceError("interface-malformed", field, f"{where}{what} is refused: {why}")


def check(interface: Any, *, ai: bool = False, program: str = "") -> None:
    """Refuse an interface that breaks programs.md's rules (``InterfaceError``,
    code ``interface-malformed``), naming the first field at fault, inputs
    then outputs. ``ai``: an AI function's, whose shapes may carry keywords
    the vocabulary does not list (lmcc's), which are never read here."""
    if not isinstance(interface, Mapping):
        raise _refuse(None, "an interface is a JSON object", program)
    extra = sorted(set(interface) - {"description", "inputs", "outputs"})
    if extra:
        raise _refuse(None, f"it has keys no interface has ({', '.join(extra)}): an interface is closed", program)
    if not isinstance(interface.get("description"), str):
        raise _refuse(None, "its description is text", program)
    if not isinstance(interface.get("inputs"), list):
        raise _refuse(None, "its inputs are a list", program)
    if not isinstance(interface.get("outputs"), list) or not interface["outputs"]:
        raise _refuse(None, "it has at least one output", program)
    seen: set = set()
    for direction in ("inputs", "outputs"):
        allowed = INPUT_KEYS if direction == "inputs" else OUTPUT_KEYS
        for f in interface[direction]:
            name = f.get("name") if isinstance(f, Mapping) else None
            at = name if isinstance(name, str) else None
            if not isinstance(f, Mapping):
                raise _refuse(None, f"each of its {direction} is a JSON object", program)
            unknown = sorted(set(f) - allowed)
            if unknown:
                raise _refuse(at, f"it has keys a field of {direction} does not have ({', '.join(unknown)})", program)
            if not isinstance(name, str) or not NAME.fullmatch(name):
                raise _refuse(at, "a name is an ASCII identifier (a letter or _, then letters, digits or _)", program)
            if name in seen:
                raise _refuse(at, "the name is used twice among the inputs and outputs", program)
            seen.add(name)
            for k in ("desc", "type"):
                if k in f and not isinstance(f[k], str):
                    raise _refuse(at, f"its {k} is text", program)
            for k in ("opaque", "optional"):
                if k in f and f[k] is not True:
                    raise _refuse(at, f"{k}, when present, is true", program)
            shape = f.get("shape")
            if not isinstance(shape, dict):
                raise _refuse(at, "its shape is a JSON object", program)
            fault = _shape_fault(shape, shape, ai)
            if fault:
                raise _refuse(at, fault, program)
            if _loops(shape):
                raise _refuse(at, "a $defs entry refers back to itself without passing into a value", program)
            if ai and f.get("optional") and "default" not in shape:
                raise _refuse(at, "an AI function's optional input needs a default in its shape (a model is "
                                  "sent every input)", program)
            if f.get("opaque") and shape != {}:
                raise _refuse(at, "an opaque field's shape is {}", program)
            if "default" in shape and not fits_json(shape["default"], {"shape": shape}):
                raise _refuse(at, f"its default {shape['default']!r} does not fit its shape", program)
    return None


def signature(interface: Mapping[str, Any]) -> str:
    """What the interface's data looks like: lmcc's signature fingerprint of
    its fields, each plain and untyped, each shape without its own default
    (the call log's ``program.interface``)."""
    from .calllog import canonical, _sha
    fields = [{"direction": d[:-1], "name": f["name"], "purpose": "plain", "shape": data_shape(f["shape"]),
               "type": ""} for d in ("inputs", "outputs") for f in interface[d]]
    return _sha(canonical(fields))


# ------------------------------------------------------------------ checking a call


def bind_inputs(interface: Mapping[str, Any], given: Mapping[str, Any], *, program: str = "",
                has_default: Any = ()) -> Dict[str, Any]:
    """The inputs a module's code gets, checked (programs.md, *Checking values*).

    ``given``: the inputs the call gives, by name (native values or JSON).
    An optional input left out takes its shape's default, unless the code
    has a default of its own for it (``has_default``, names), which then
    applies; with neither, it stays left out. Raises ``InterfaceError``
    (``interface-input``) naming the first input at fault."""
    where = f"{program}: " if program else ""
    names = [f["name"] for f in interface["inputs"]]
    unknown = sorted(k for k in given if k not in names)
    if unknown:
        raise InterfaceError("interface-input", unknown[0],
                             f"{where}no input named {unknown[0]!r} (inputs: {', '.join(names) or 'none'})")
    out: Dict[str, Any] = {}
    for f in interface["inputs"]:
        name = f["name"]
        if name in given:
            value = given[name]
            if not fits(value, f):
                ok, _ = json_form(value)
                shown = repr(value) if len(repr(value)) <= 80 else repr(value)[:77] + "..."
                why = (f"{shown} has no JSON form, and the input is not opaque" if not ok
                       else f"{shown} does not fit {_canonical(data_shape(f['shape']))}")
                raise InterfaceError("interface-input", name, f"{where}input {name!r}: {why}")
            out[name] = value
        elif not f.get("optional"):
            raise InterfaceError("interface-input", name, f"{where}input {name!r} is required")
        elif name in has_default:
            continue
        elif "default" in f["shape"]:
            out[name] = copy.deepcopy(f["shape"]["default"])
    return out


def recorded_inputs(interface: Mapping[str, Any], given: Mapping[str, Any]) -> Dict[str, Any]:
    """The inputs a call's record holds: those given, and for an optional
    input left out, its shape's default (absent when it has none)."""
    out: Dict[str, Any] = {}
    for f in interface["inputs"]:
        if f["name"] in given:
            out[f["name"]] = given[f["name"]]
        elif "default" in f["shape"]:
            out[f["name"]] = copy.deepcopy(f["shape"]["default"])
    for k, v in given.items():
        out.setdefault(k, v)
    return out


def outputs_of(interface: Mapping[str, Any], returned: Any) -> Tuple[bool, Dict[str, Any]]:
    """What the code returned, by output: ``(True, {name: value})``, or
    ``(False, {})`` when several outputs are declared and it is not a record."""
    outputs = interface["outputs"]
    if len(outputs) == 1:
        return True, {outputs[0]["name"]: returned}
    if isinstance(returned, Mapping):
        return True, dict(returned)
    if dataclasses.is_dataclass(returned) and not isinstance(returned, type):
        return True, {f.name: getattr(returned, f.name) for f in dataclasses.fields(returned)}
    dump = getattr(returned, "model_dump", None)
    if callable(dump) and not isinstance(returned, type):
        return True, {k: getattr(returned, k) for k in dump()}
    return False, {}


def check_outputs(interface: Mapping[str, Any], returned: Any, *, program: str = "") -> Dict[str, Any]:
    """The call's outputs by name, checked; raises ``InterfaceError``
    (``interface-output``) naming the first output at fault."""
    where = f"{program}: " if program else ""
    outputs = interface["outputs"]
    names = [f["name"] for f in outputs]
    ok, values = outputs_of(interface, returned)
    if not ok:
        raise InterfaceError("interface-output", names[0],
                             f"{where}it declares several outputs ({', '.join(names)}), so it returns a dict of "
                             f"them by name, not {type(returned).__name__}")
    unknown = sorted(k for k in values if k not in names)
    if unknown:
        raise InterfaceError("interface-output", unknown[0],
                             f"{where}it returned {unknown[0]!r}, which is not one of its outputs "
                             f"({', '.join(names)})")
    for f in outputs:
        name = f["name"]
        if name not in values:
            raise InterfaceError("interface-output", name, f"{where}output {name!r} is missing")
        if not fits(values[name], f):
            value = values[name]
            ok, _ = json_form(value)
            shown = repr(value) if len(repr(value)) <= 80 else repr(value)[:77] + "..."
            why = (f"{shown} has no JSON form, and the output is not opaque" if not ok
                   else f"{shown} does not fit {_canonical(data_shape(f['shape']))}")
            raise InterfaceError("interface-output", name, f"{where}output {name!r}: {why}")
    return {f["name"]: values[f["name"]] for f in outputs}


# ------------------------------------------------------------------ interfaces from Python


def _type_name(ann: Any) -> Optional[str]:
    if ann is inspect.Parameter.empty or ann is None:
        return None
    try:
        from lmcc import core as lmcc_core
        name = lmcc_core.typename(ann)
        if name:
            return name
    except Exception:  # noqa: BLE001 — a name for people only
        pass
    return getattr(ann, "__qualname__", None) or str(ann)


def _is_json_mark(ann: Any) -> bool:
    return typing.get_origin(ann) is Annotated and any(x is _JSON_MARK for x in typing.get_args(ann)[1:])


def _accepts_null(shape: Dict[str, Any]) -> bool:
    if shape == {} or set(shape) <= set(WORDS):
        return True
    t = shape.get("type")
    if t == "null" or (isinstance(t, list) and "null" in t):
        return True
    return any(isinstance(x, dict) and _accepts_null(x) for x in shape.get("anyOf") or [])


def nullable(shape: Dict[str, Any]) -> Dict[str, Any]:
    """A shape that also takes null (``T = None`` in Python is ``Optional[T]``)."""
    if _accepts_null(shape):
        return shape
    return {"anyOf": [shape, {"type": "null"}]}


def field_of(name: str, ann: Any, *, desc: Optional[str] = None, where: str = "") -> Dict[str, Any]:
    """One field of a module's interface from a Python annotation: its shape,
    or opaque for ``Any``, ``object``, no annotation, or a type with no JSON
    form; ``functai.JSON`` is any JSON value."""
    from .signature import shape_of, unannotate
    out: Dict[str, Any] = {"name": name}
    if ann is inspect.Parameter.empty:
        out.update(shape={}, opaque=True)
    elif _is_json_mark(ann):
        out.update(shape={}, type="JSON")
    else:
        base, adesc = unannotate(ann)
        desc = desc or adesc
        if base is Any or base is object:
            out.update(shape={}, opaque=True)
            if base is object:
                out["type"] = "object"
        else:
            try:
                shape = shape_of(ann, _registry(), where=where or name)
            except lmcc.Refusal:                     # a type with no JSON form: opaque
                out.update(shape={}, opaque=True)
            else:
                out["shape"] = shape
            name_ = _type_name(base)
            if name_:
                out["type"] = name_
    if desc:
        out["desc"] = desc
    return out


def _registry() -> Any:
    from .adapters import REGISTRY
    return REGISTRY


def _hints(fn: Any) -> Dict[str, Any]:
    """The function's annotations, resolved (a name not defined yet raises NameError)."""
    return typing.get_type_hints(fn, include_extras=True)


def _with_default(field: Dict[str, Any], default: Any) -> Dict[str, Any]:
    field["optional"] = True
    if field.get("opaque"):
        return field
    shape = dict(field["shape"])
    if default is None:
        shape = dict(nullable(shape))
    ok, data = json_form(default)
    if ok:
        shape["default"] = data
    field["shape"] = shape
    return field


def of_function(fn: Any, *, outputs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A module's interface, derived from its Python function (programs.md,
    *How each program has one*): each parameter an input (with a default:
    optional, its default in the shape when it has a JSON form; ``None``
    makes the shape nullable; ``*args`` a list, ``**kwargs`` an object, both
    optional), the return annotation one output named ``result`` (or
    ``outputs``: several, by name, each a type)."""
    from .docments import _harvest_inline_param_and_return_comments
    hints = _hints(fn)
    try:
        comments, returns = _harvest_inline_param_and_return_comments(fn)
    except Exception:  # noqa: BLE001 — words only
        comments, returns = {}, None
    inputs: List[Dict[str, Any]] = []
    for pname, p in inspect.signature(fn).parameters.items():
        ann = hints.get(pname, inspect.Parameter.empty)
        desc = comments.get(pname) or None
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            shape: Dict[str, Any] = {"type": "array"}
            if ann is not inspect.Parameter.empty:
                shape["items"] = field_of(pname, ann).get("shape", {})
            f = {"name": pname, "shape": {**shape, "default": []}, "optional": True}
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            shape = {"type": "object"}
            if ann is not inspect.Parameter.empty:
                inner = field_of(pname, ann)
                if not inner.get("opaque"):
                    shape["additionalProperties"] = inner["shape"]
            f = {"name": pname, "shape": {**shape, "default": {}}, "optional": True}
        else:
            f = field_of(pname, ann, desc=desc)
            if p.default is not inspect.Parameter.empty:
                f = _with_default(f, p.default)
        if desc and "desc" not in f:
            f["desc"] = desc
        inputs.append(_ordered(f))
    if outputs is not None:
        outs = [_ordered(field_of(n, t)) for n, t in outputs.items()]
    else:
        ret = hints.get("return", inspect.Parameter.empty)
        out = field_of("result", ret, desc=returns)
        if ret is type(None):
            out["shape"] = {"type": "null"}
        outs = [_ordered(out)]
    return {"description": inspect.cleandoc(fn.__doc__ or ""), "inputs": inputs, "outputs": outs}


_ORDER = ("name", "shape", "desc", "type", "opaque", "optional")


def _ordered(f: Dict[str, Any]) -> Dict[str, Any]:
    return {k: f[k] for k in _ORDER if k in f}


def of_ai(fn: Any) -> Dict[str, Any]:
    """An AI function's interface: its definition's inputs (optional when the
    Python parameter has a default, which goes into the shape) and its own
    outputs, without the fields FunctAI adds (reasoning, tools)."""
    spec = fn._spec()
    params = fn._sig.parameters
    inputs = []
    for f in spec.signature.inputs:
        if f.purpose != "plain":
            continue
        field: Dict[str, Any] = {"name": f.name, "shape": copy.deepcopy(f.shape)}
        if f.desc:
            field["desc"] = f.desc
        if f.type:
            field["type"] = f.type
        p = params.get(f.name)
        if p is not None and p.default is not inspect.Parameter.empty:
            field["optional"] = True
            ok, data = json_form(p.default)
            if ok:
                field["shape"]["default"] = data
        inputs.append(field)
    outputs = []
    descs = getattr(spec, "descs", {}) or {}
    by_name = {f.name: f for f in spec.signature.outputs}
    for name in spec.outputs:
        f = by_name[name]
        field = {"name": name, "shape": copy.deepcopy(f.shape)}
        if descs.get(name):
            field["desc"] = descs[name]
        if f.type:
            field["type"] = f.type
        outputs.append(field)
    return {"description": inspect.cleandoc(fn.__wrapped__.__doc__ or ""), "inputs": inputs, "outputs": outputs}


__all__ = ["JSON", "check", "signature", "fits", "bind_inputs", "check_outputs", "of_function", "of_ai",
           "InterfaceError"]
