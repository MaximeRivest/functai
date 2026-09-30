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
import itertools
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
        # a record (properties, no additionalProperties) is closed: programs.md, "Checking values"
        extra = shape.get("additionalProperties", False if "properties" in shape else True)
        for k, x in v.items():
            if k in props:
                if not _fits_shape(x, props[k], root):
                    return False
            elif extra is False or (isinstance(extra, dict) and not _fits_shape(x, extra, root)):
                return False
    return True


_SUBSHAPES = ("items", "additionalProperties", "not")
_SHAPE_LISTS = ("anyOf", "prefixItems", "oneOf", "allOf")
_SHAPE_MAPS = ("properties", "$defs")


def data_shape(shape: Any) -> Any:
    """A shape without any ``default`` keyword, its own or one inside it: what
    its data looks like (programs.md, the signature). A member named
    ``default`` stays, and so does data (``enum``, ``const``, ``examples``)."""
    if not isinstance(shape, dict):
        return shape
    out: Dict[str, Any] = {}
    for k, v in shape.items():
        if k == "default":
            continue
        if k in _SUBSHAPES and isinstance(v, dict):
            out[k] = data_shape(v)
        elif k in _SHAPE_LISTS and isinstance(v, list):
            out[k] = [data_shape(x) for x in v]
        elif k in _SHAPE_MAPS and isinstance(v, dict):
            out[k] = {n: data_shape(x) for n, x in v.items()}
        else:
            out[k] = v
    return out


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


# ------------------------------------------------------------------ binding (programs.md, "Binding a call's inputs")


_NUMBER_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_INTEGER_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)")
_REFUSED = object()


def is_missing(value: Any) -> bool:
    """Python's missing values: ``None``, a float not-a-number (pandas' gap
    in a numeric column), pandas' ``NA`` and ``NaT``."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return type(value).__name__ in ("NAType", "NaTType")


_DEFAULT_TEXT = re.compile(r"<[^<>]* at 0x[0-9a-fA-F]+>", re.S)


def _own_text(value: Any) -> Optional[str]:
    """A value's text when its type gives it one (a data frame, a date), else
    None: text that is only Python's default for any object
    (``<Thing object at 0x…>``) says nothing about the value."""
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 — a text that raises is no text
        return None
    return None if _DEFAULT_TEXT.fullmatch(text) else text


def _text_of(v: Any) -> str:
    """JSON as text (programs.md, Binding): a number as canonical JSON, a
    boolean ``true``/``false``, a list or object indented by two spaces."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if _is_number(v):
        return _canonical(v)
    return _indented(v)


def _indented(v: Any, level: int = 0) -> str:
    pad, inner = "  " * level, "  " * (level + 1)
    if isinstance(v, (list, tuple)):
        return "[]" if not v else "[\n" + ",\n".join(inner + _indented(x, level + 1) for x in v) + "\n" + pad + "]"
    if isinstance(v, Mapping):
        return "{}" if not v else "{\n" + ",\n".join(
            inner + _canonical(str(k)) + ": " + _indented(x, level + 1) for k, x in v.items()) + "\n" + pad + "}"
    return _canonical(v)


def _number_from_text(s: str) -> Any:
    s = s.strip(" \t\n\r")
    if not _NUMBER_TEXT.fullmatch(s):
        return None
    if _INTEGER_TEXT.fullmatch(s):
        return int(s)
    x = float(s)
    if not math.isfinite(x):
        return None
    return int(x) if x.is_integer() and abs(x) < 2 ** 53 else x


class _NoJSON:
    """A value with no JSON form, carried through binding (only text may take it)."""
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


def _convert(v: Any, kind: str) -> Any:
    if kind == "null":
        return None if v is None else _REFUSED
    if v is None:
        return _REFUSED
    if isinstance(v, _NoJSON):
        if kind == "string":
            text = _own_text(v.value)
            return text if text is not None else _REFUSED
        return _REFUSED
    if kind == "string":
        return v if isinstance(v, str) else _text_of(v)
    if kind == "boolean":
        return v if isinstance(v, bool) else _REFUSED
    if kind in ("integer", "number"):
        if isinstance(v, bool):
            return _REFUSED
        x = v if _is_number(v) else _number_from_text(v) if isinstance(v, str) else None
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return _REFUSED
        if kind == "integer":
            if not float(x).is_integer():
                return _REFUSED
            return int(x) if isinstance(x, float) else x
        return x
    if kind == "array":
        return list(v) if isinstance(v, (list, tuple)) else _REFUSED
    if kind == "object":
        return dict(v) if isinstance(v, Mapping) else _REFUSED
    return _REFUSED


def _fits_bound(v: Any, shape: Dict[str, Any], root: Dict[str, Any]) -> bool:
    if isinstance(v, _NoJSON):
        return False
    try:
        return _fits_shape(v, shape, root)
    except TypeError:
        return False


def _bind_shape(v: Any, shape: Dict[str, Any], root: Dict[str, Any]) -> Any:
    ref = shape.get("$ref")
    if isinstance(ref, str) and REF.fullmatch(ref):
        v = _bind_shape(v, (root.get("$defs") or {})[REF.fullmatch(ref).group(1)], root)   # type: ignore[union-attr]
    if isinstance(shape.get("anyOf"), list) and shape["anyOf"]:
        if v is None:
            return None
        for option in shape["anyOf"]:
            b = _bind_shape(v, option, root)
            if _fits_bound(b, option, root):
                v = b
                break
        else:
            return _bind_shape(v, shape["anyOf"][0], root)
    if "type" in shape:
        names = shape["type"] if isinstance(shape["type"], list) else [shape["type"]]
        if v is None:
            return None
        for name in names:
            if name == "null":
                continue
            c = _convert(v, name)
            if c is _REFUSED:
                continue
            c = _descend(c, shape, root)
            if _fits_bound(c, {**shape, "type": name}, root):
                return c
        return v
    return _descend(v, shape, root)


def _descend(v: Any, shape: Dict[str, Any], root: Dict[str, Any]) -> Any:
    if isinstance(v, (list, tuple)) and ("items" in shape or "prefixItems" in shape):
        prefix = shape.get("prefixItems") or []
        return [_bind_shape(x, prefix[i], root) if i < len(prefix) else
                _bind_shape(x, shape["items"], root) if isinstance(shape.get("items"), dict) else x
                for i, x in enumerate(v)]
    if isinstance(v, Mapping) and ("properties" in shape or "additionalProperties" in shape):
        props = shape.get("properties") or {}
        extra = shape.get("additionalProperties")
        out: Dict[str, Any] = {}
        for k, x in v.items():
            if k in props:
                out[k] = _bind_shape(x, props[k], root)
            elif isinstance(extra, dict):
                out[k] = _bind_shape(x, extra, root)
            elif extra is True or (extra is None and "properties" not in shape):
                out[k] = x
            # a record (closed) drops a member it does not name
        return out
    return v


def bind(value: Any, field: Mapping[str, Any]) -> Tuple[bool, Any]:
    """``(True, the bound value)`` or ``(False, value)`` when it does not bind
    and fit (programs.md, *Binding a call's inputs*). An opaque field takes
    anything. A value that fits as it is comes back as it is (a dataclass
    stays one); one that binds to something else comes back as the bound
    JSON (``"5"`` for an integer is ``5``)."""
    if field.get("opaque"):
        return True, value
    shape = {k: v for k, v in field["shape"].items() if k != "default"}
    ok, data = json_form(value)
    start: Any = None if is_missing(value) else data if ok else _NoJSON(value)
    bound = _bind_shape(start, shape, shape)
    if isinstance(bound, _NoJSON) or not _fits_bound(bound, shape, shape):
        return False, value
    if ok and _same_json(bound, data):
        return True, value                   # it fitted as it was: the program gets its own value
    return True, bound


def _same_json(a: Any, b: Any) -> bool:
    """Equal as JSON, and of the same kinds all the way down (``5.0`` is not
    ``5`` here: binding made one from the other)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if _is_number(a) and _is_number(b):
        return type(a) is type(b) and a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return list(a) == list(b) and all(_same_json(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _missing_to_optional(f: Mapping[str, Any], value: Any) -> bool:
    """``null`` given to an optional input that null does not fit is that input left out."""
    if not f.get("optional") or not is_missing(value) or f.get("opaque"):
        return False
    shape = {k: v for k, v in f["shape"].items() if k != "default"}
    return not _fits_bound(None, shape, shape)


def refusal_message(program: str, field: Mapping[str, Any], value: Any, direction: str = "input",
                    quote: bool = True) -> str:
    """programs.md, *The message*: the program, the field, what it wants, and
    the value (canonical JSON, or a description's ``$repr``, cut after 80 code
    points) unless its field is dropped from the log."""
    where = f"{program}: " if program else ""
    wants = "any value" if field.get("opaque") else _canonical(data_shape(field["shape"]))
    name = field["name"]
    if not quote:
        return f"{where}{direction} {name!r} does not bind to {wants} (its value is not shown: the log drops it)"
    ok, data = json_form(value)
    if ok:
        text = "null" if is_missing(value) else _canonical(data)
    else:
        from .calllog import safe_repr
        text = safe_repr(value)
    if len(text) > 80:
        text = text[:80] + "\u2026"
    why = "has no JSON form, and does not bind to" if not ok else "does not bind to"
    return f"{where}{direction} {name!r}: {text} {why} {wants}"


def bind_call(interface: Mapping[str, Any], given: Mapping[str, Any], *, program: str = "",
              has_default: Any = (), dropped: Any = ()) -> Dict[str, Any]:
    """Every input bound and checked (programs.md, *Binding a call's inputs*,
    *Checking values*): what a program's code, or a model, is given.

    ``given``: the inputs the call gives, by name. An optional input left out,
    or given a missing value null does not fit, takes its shape's default,
    unless the code has a default of its own for it (``has_default``), which
    then applies; with neither, it stays left out. ``dropped``: the fields a
    ``log_content`` layer drops, whose values a message never quotes. Raises
    ``InterfaceError`` (``interface-input``) naming the first input at fault."""
    where = f"{program}: " if program else ""
    names = [f["name"] for f in interface["inputs"]]
    unknown = sorted(k for k in given if k not in names)
    if unknown:
        raise InterfaceError("interface-input", unknown[0],
                             f"{where}no input named {unknown[0]!r} (inputs: {', '.join(names) or 'none'})")
    out: Dict[str, Any] = {}
    for f in interface["inputs"]:
        name = f["name"]
        if name in given and not _missing_to_optional(f, given[name]):
            ok, value = bind(given[name], f)
            if not ok:
                raise InterfaceError("interface-input", name,
                                     refusal_message(program, f, given[name],
                                                     quote="*" not in dropped and name not in dropped))
            out[name] = value
        elif name in given or f.get("optional"):
            if name in has_default:
                continue
            if "default" in f["shape"]:
                out[name] = copy.deepcopy(f["shape"]["default"])
        else:
            raise InterfaceError("interface-input", name, f"{where}input {name!r} is required")
    return out


def bound_for_record(interface: Mapping[str, Any], given: Mapping[str, Any]) -> Dict[str, Any]:
    """The inputs a call's record holds: each bound where it binds, as given
    where it does not (the call is then refused, and its record says so),
    and an optional input left out with its shape's default. A name the
    interface lacks is never recorded."""
    out: Dict[str, Any] = {}
    for f in interface["inputs"]:
        name = f["name"]
        if name in given and not _missing_to_optional(f, given[name]):
            ok, value = bind(given[name], f)
            out[name] = value
        elif "default" in f["shape"]:
            out[name] = copy.deepcopy(f["shape"]["default"])
    return out


def bind_inputs(interface: Mapping[str, Any], given: Mapping[str, Any], *, program: str = "",
                has_default: Any = (), dropped: Any = ()) -> Dict[str, Any]:
    """The inputs a module's code gets: ``bind_call``."""
    return bind_call(interface, given, program=program, has_default=has_default, dropped=dropped)


def _shown(value: Any) -> str:
    """A value as an error message quotes it: its repr, cut to 80 characters
    (a repr that raises is described, never raised)."""
    from .calllog import safe_repr
    text = safe_repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def recorded_inputs(interface: Mapping[str, Any], given: Mapping[str, Any]) -> Dict[str, Any]:
    """The inputs a call's record holds, named as the interface names them
    (``bound_for_record``). A value given under a name the interface does
    not have is refused (``bind_inputs``) and never recorded."""
    return bound_for_record(interface, given)


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


def check_outputs(interface: Mapping[str, Any], returned: Any, *, program: str = "",
                  dropped: Any = ()) -> Dict[str, Any]:
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
            raise InterfaceError("interface-output", name,
                                 refusal_message(program, f, values[name], "output",
                                                 quote="*" not in dropped and name not in dropped)
                                 .replace("does not bind to", "does not fit"))
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


def _hints(fn: Any, localns: Optional[Mapping[str, Any]] = None, *, program: str = "",
           outputs: Tuple[str, ...] = ("result",)) -> Dict[str, Any]:
    """The function's annotations, resolved in its module's names and
    ``localns`` (where it was defined). A name not defined yet refuses
    ``interface-malformed``, naming the first field whose annotation uses it:
    an interface is checked when its program is defined, and one that cannot
    be read then cannot be checked."""
    try:
        return typing.get_type_hints(fn, localns=dict(localns) if localns else None, include_extras=True)
    except NameError:
        pass
    where = f"{program}: " if program else ""
    globalns = getattr(fn, "__globals__", {})
    for name, ann in list(getattr(fn, "__annotations__", {}).items()):
        def probe() -> None: ...
        probe.__annotations__ = {"x": ann}
        try:
            typing.get_type_hints(probe, globalns=globalns, localns=dict(localns) if localns else None,
                                  include_extras=True)
        except NameError as exc:
            field = outputs[-1] if name == "return" else name
            missing = getattr(exc, "name", None) or str(exc)
            raise InterfaceError("interface-malformed", field,
                                 f"{where}the annotation of {field!r} names {missing!r}, which is not defined where "
                                 f"the module is defined: define it first (an interface is checked when its "
                                 f"program is defined)") from None
    return typing.get_type_hints(fn, localns=dict(localns) if localns else None, include_extras=True)


def _with_default(field: Dict[str, Any], default: Any) -> Dict[str, Any]:
    field["optional"] = True
    if field.get("opaque"):
        return field
    shape = dict(field["shape"])
    if default is None:
        shape = dict(nullable(shape))
    ok, data = json_form(default)
    if ok:
        # bound as a given value is (programs.md, Binding): "5" for an integer is 5; one that does not bind
        # stays as it is, and check() refuses it (interface-malformed)
        bound_ok, bound = bind(default, {"name": field["name"], "shape": shape})
        if bound_ok and not is_missing(default):
            ok2, data2 = json_form(bound)
            shape["default"] = data2 if ok2 else data
        else:
            shape["default"] = data
    field["shape"] = shape
    return field


def of_function(fn: Any, *, outputs: Optional[Mapping[str, Any]] = None,
                output_fields: Optional[List[Dict[str, Any]]] = None,
                localns: Optional[Mapping[str, Any]] = None, program: str = "") -> Dict[str, Any]:
    """A module's interface, derived from its Python function (programs.md,
    *How each program has one*): each parameter an input (with a default:
    optional, its default in the shape when it has a JSON form; ``None``
    makes the shape nullable; ``*args`` a list, ``**kwargs`` an object, both
    optional), the return annotation one output named ``result`` (or
    ``outputs``: several, by name, each a type; or ``output_fields``, the
    outputs as data). ``localns``: the names where it was defined."""
    from .docments import _harvest_inline_param_and_return_comments
    out_names = tuple(outputs) if outputs else tuple(f["name"] for f in output_fields) if output_fields else ("result",)
    hints = _hints(fn, localns, program=program, outputs=out_names)
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
    if output_fields is not None:
        outs = [copy.deepcopy(dict(f)) for f in output_fields]
    elif outputs is not None:
        outs = [_ordered(field_of(n, t)) for n, t in outputs.items()]
    else:
        ret = hints.get("return", inspect.Parameter.empty)
        out = field_of("result", ret, desc=returns)
        if ret is type(None):
            out["shape"] = {"type": "null"}
        outs = [_ordered(out)]
    return {"description": inspect.cleandoc(fn.__doc__ or ""), "inputs": inputs, "outputs": outs}


_JSON_TYPES: Dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool,
                                "null": type(None)}


def python_type(shape: Any, root: Optional[Mapping[str, Any]] = None, _seen: Tuple[str, ...] = ()) -> Any:
    """The Python type of the values a shape admits, as they come from data
    (a record is a ``dict``, a list a ``list``): ``str``, ``int``, ``float``,
    ``bool``, ``None``, ``Literal[...]`` for an ``enum`` or ``const`` of text,
    whole numbers, booleans or null, ``list[T]``, ``dict[str, T]``, a
    ``Union`` for ``anyOf`` or several types; ``Any`` for anything else (an
    opaque ``{}``, a reference that loops). For introspection: a function
    loaded from data states its inputs and answer with these."""
    if not isinstance(shape, Mapping):
        return Any
    root = shape if root is None else root
    ref = shape.get("$ref")
    if isinstance(ref, str):
        m = REF.fullmatch(ref)
        target = (root.get("$defs") or {}).get(m.group(1)) if m else None
        if target is None or ref in _seen:
            return Any
        return python_type(target, root, (*_seen, ref))
    scalar = (str, int, bool, type(None))
    if "const" in shape:
        c = shape["const"]
        return typing.Literal[c] if isinstance(c, scalar) and not isinstance(c, float) else Any
    if isinstance(shape.get("enum"), list) and shape["enum"] and all(
            isinstance(v, scalar) and not isinstance(v, float) for v in shape["enum"]):
        return typing.Literal[tuple(shape["enum"])]
    if isinstance(shape.get("anyOf"), list) and shape["anyOf"]:
        return _union([python_type(s, root, _seen) for s in shape["anyOf"]])
    kind = shape.get("type")
    if isinstance(kind, list) and kind:
        return _union([python_type({**shape, "type": k}, root, _seen) for k in kind])
    if kind == "array":
        return List[python_type(shape.get("items", {}), root, _seen)]  # type: ignore[misc]
    if kind == "object":
        extra = shape.get("additionalProperties")
        return Dict[str, python_type(extra, root, _seen) if isinstance(extra, Mapping) else Any]  # type: ignore[misc]
    return _JSON_TYPES.get(kind, Any) if isinstance(kind, str) else Any


def _union(types: List[Any]) -> Any:
    if any(t is Any for t in types):
        return Any
    distinct = list(dict.fromkeys(types))
    return distinct[0] if len(distinct) == 1 else typing.Union[tuple(distinct)]


def python_signature(interface: Mapping[str, Any]) -> Tuple[inspect.Signature, bool]:
    """``(signature, exact)``: an interface's inputs and answer as a Python
    signature, each input annotated with ``python_type`` of its shape and an
    optional one with its default.

    ``exact`` when the signature binds exactly as the interface does (every
    name a Python parameter name, no required input after an optional one).
    Otherwise Python cannot write it, and the signature is the closest it
    can: the inputs from the first one out of Python's order on are
    keyword-only (a call may still give them by position, in the
    interface's order), and names Python reserves (``class``) are given
    through ``**`` (``fn(**{"class": ...})``), named ``inputs`` or, when an
    input has that name, the first of ``inputs_1``, ``inputs_2``, ... none has
    (only a name for display: the interface binds the call)."""
    import keyword
    params: List[inspect.Parameter] = []
    reserved: List[str] = []
    names = [f["name"] for f in interface.get("inputs") or []]
    positional, seen_optional = True, False
    for f in interface.get("inputs") or []:
        name, shape = f["name"], f.get("shape") or {}
        if not name.isidentifier() or keyword.iskeyword(name):
            reserved.append(name)
            positional = False
            continue
        optional = bool(f.get("optional"))
        if positional and seen_optional and not optional:
            positional = False
        seen_optional = seen_optional or optional
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD if positional else inspect.Parameter.KEYWORD_ONLY
        default = copy.deepcopy(shape["default"]) if optional and "default" in shape else inspect.Parameter.empty
        params.append(inspect.Parameter(name, kind, default=default,
                                        annotation=python_type(shape)))
    if reserved:
        taken = set(names)
        rest = next(n for n in itertools.chain(["inputs"], (f"inputs_{i}" for i in itertools.count(1)))
                    if n not in taken)
        params.append(inspect.Parameter(rest, inspect.Parameter.VAR_KEYWORD))
    outputs = interface.get("outputs") or []
    answer = python_type(outputs[-1].get("shape") or {}) if outputs else Any
    exact = positional and not reserved
    return inspect.Signature(params, return_annotation=answer), exact


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
                # bound as a given value is (programs.md, Binding): "5" for an integer is 5; one that does
                # not bind stays as it is, and check() refuses it (interface-malformed)
                bound_ok, bound = bind(p.default, {"name": f.name, "shape": field["shape"]})
                ok2, data2 = json_form(bound) if bound_ok else (False, None)
                field["shape"]["default"] = data2 if ok2 else data
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
           "python_type", "python_signature", "InterfaceError"]


# ------------------------------------------------------------------ defaults by their logic (calls.md, Versions)


def _is_value_expr(node: Any) -> bool:
    """A default written as a constant (a literal, a container of them, a signed
    number) or a name (``DEFAULT_TONE``, ``Tone.KIND``): it counts by its value."""
    import ast
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) \
            and isinstance(node.operand, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_value_expr(x) for x in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_value_expr(k) for k in node.keys) and all(_is_value_expr(v)
                                                                                  for v in node.values)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _is_value_expr(node.value) if isinstance(node.value, (ast.Name, ast.Attribute)) else False
    return False


_default_code_cache: Dict[Any, Dict[str, str]] = {}
_warned_no_source: set = set()


def default_code(fn: Any) -> Dict[str, str]:
    """``{parameter: text}`` for each parameter whose default is written as an
    expression that is neither a constant nor a name (``today()``,
    ``3 * 60``): ``ast.unparse`` of it (calls.md, *Versions*, *Defaults*).
    Read from the function's source once; where the source cannot be read (a
    function typed at a bare prompt), every default counts by its value, with
    one warning per function."""
    import ast
    import textwrap
    import warnings
    fn = inspect.unwrap(fn)
    code_obj = getattr(fn, "__code__", None)
    key = code_obj if code_obj is not None else id(fn)
    if key in _default_code_cache:
        return _default_code_cache[key]
    out: Dict[str, str] = {}
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == fn.__name__)
    except (OSError, TypeError, SyntaxError, StopIteration, IndentationError):
        has_defaults = any(p.default is not inspect.Parameter.empty
                           for p in inspect.signature(fn).parameters.values())
        if has_defaults and key not in _warned_no_source:
            _warned_no_source.add(key)
            warnings.warn(f"[functai] {getattr(fn, '__name__', fn)}: its source cannot be read, so its defaults "
                          f"count in its version by their values, not by what is written", stacklevel=3)
        _default_code_cache[key] = out
        return out
    a = node.args
    positional = [*a.posonlyargs, *a.args]
    for arg, default in zip(positional[len(positional) - len(a.defaults):], a.defaults):
        if not _is_value_expr(default):
            out[arg.arg] = ast.unparse(default)
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        if default is not None and not _is_value_expr(default):
            out[arg.arg] = ast.unparse(default)
    _default_code_cache[key] = out
    return out


def defaults_document(interface: Mapping[str, Any], code: Mapping[str, str]) -> Dict[str, Any]:
    """``D`` of a version (calls.md, *Versions*, *Defaults*): each input that has
    a default, in the interface's order, ``{"code": text}`` when it is
    written as an expression, else ``{"value": its JSON}``."""
    out: Dict[str, Any] = {}
    for f in interface["inputs"]:
        name = f["name"]
        if name in code:
            out[name] = {"code": code[name]}
        elif "default" in (f.get("shape") or {}):
            out[name] = {"value": f["shape"]["default"]}
    return out
