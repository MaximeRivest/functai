"""The contract's JSON Schemas (``functai/schema/``, a copy of
``contract/schema/`` that a test keeps equal), checked at run time.

    from functai import schemas
    schemas.problem("event", event)     # None, or where and why it fails
    schemas.valid("saved", manifest)

Where the contract says a value must pass a schema before anything else is
done with it (a store refusing ``event-malformed``, a loader refusing
``saved-malformed``), FunctAI checks the whole schema, not a part of it.

This is a JSON Schema draft 2020-12 validator for the keywords the
contract's schemas use, and no more: a schema holding a keyword it does not
implement is refused when it is read (``SchemaError``), so a schema the
contract grows can never be checked partly and pass what it refuses. It
reads patterns as ECMA-262 does (a pattern's ``$`` is the end of the text),
integers as numbers with no fraction (``1.0`` is one), ``true`` as no
number, and compares ``const``, ``enum`` and ``uniqueItems`` as canonical
JSON, as the contract does. The test suite runs it beside the reference
validator (``jsonschema``) on every case and on mutations of them.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin

NAMES = ("call", "event", "interface", "rating", "saved")
FOLDER = Path(__file__).with_name("schema")

# The keywords this validator applies, and those it reads as words only.
APPLIED = frozenset({
    "type", "enum", "const", "required", "properties", "additionalProperties", "propertyNames", "maxProperties",
    "minProperties", "items", "prefixItems", "minItems", "maxItems", "uniqueItems", "minLength", "maxLength",
    "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "anyOf", "oneOf", "allOf", "not",
    "if", "then", "else", "$ref", "$defs"})
WORDS = frozenset({"$schema", "$id", "$comment", "title", "description", "default", "examples", "deprecated",
                   "readOnly", "writeOnly", "format"})
_SUBSCHEMA = ("additionalProperties", "propertyNames", "items", "not", "if", "then", "else")
_SUBSCHEMAS = ("prefixItems", "anyOf", "oneOf", "allOf")
_MAPS = ("properties", "$defs")


class SchemaError(Exception):
    """A schema this validator cannot check whole (a keyword it does not implement)."""


# ------------------------------------------------------------------ reading the schemas


def _ecma(pattern: str) -> "re.Pattern[str]":
    """A schema pattern as ECMA-262 reads it: ``$`` at the end is the end of the
    text (Python's ``$`` also matches before a final newline)."""
    if "$" in pattern and not re.fullmatch(r"[^$]*\$\)*", pattern):
        raise SchemaError(f"a pattern whose $ is not its last anchor cannot be read alike: {pattern!r}")
    return re.compile(re.sub(r"\$(?=\)*$)", r"\\Z", pattern))


def _check_keywords(node: Any, where: str) -> None:
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise SchemaError(f"{where}: a schema is an object or a boolean")
    unknown = sorted(set(node) - APPLIED - WORDS)
    if unknown:
        raise SchemaError(f"{where}: keywords this validator does not implement: {unknown}")
    for k in _SUBSCHEMA:
        if k in node:
            _check_keywords(node[k], f"{where}/{k}")
    for k in _SUBSCHEMAS:
        for i, x in enumerate(node.get(k) or []):
            _check_keywords(x, f"{where}/{k}/{i}")
    for k in _MAPS:
        for name, x in (node.get(k) or {}).items():
            _check_keywords(x, f"{where}/{k}/{name}")
    if "pattern" in node:
        _ecma(node["pattern"])


@lru_cache(maxsize=None)
def _documents() -> Dict[str, Dict[str, Any]]:
    """Every schema by its ``$id``, each checked for keywords this validator lacks."""
    docs: Dict[str, Dict[str, Any]] = {}
    for name in NAMES:
        doc = json.loads((FOLDER / f"{name}.schema.json").read_text(encoding="utf-8"))
        _check_keywords(doc, f"{name}.schema.json")
        docs[doc["$id"]] = doc
    return docs


def _by_name(name: str) -> Dict[str, Any]:
    for doc in _documents().values():
        if doc["$id"].endswith(f"/{name}.schema.json"):
            return doc
    raise KeyError(f"no schema named {name!r} (one of {', '.join(NAMES)})")


def _resolve(base: str, ref: str) -> Tuple[str, Any]:
    """(the base of the schema a $ref names, that schema)."""
    uri, fragment = urldefrag(urljoin(base, ref))
    doc = _documents().get(uri)
    if doc is None:
        raise SchemaError(f"$ref {ref!r} names no schema this package holds")
    node: Any = doc
    if fragment:
        if not fragment.startswith("/"):
            raise SchemaError(f"$ref {ref!r}: only JSON Pointers are read")
        for token in fragment[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            node = node[token]
    return uri, node


@lru_cache(maxsize=None)
def _pattern(p: str) -> "re.Pattern[str]":
    return _ecma(p)


# ------------------------------------------------------------------ checking values


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_type(v: Any, t: str) -> bool:
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return _is_number(v) and (isinstance(v, int) or (math.isfinite(v) and float(v).is_integer()))
    if t == "number":
        return _is_number(v)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, (list, tuple))
    if t == "object":
        return isinstance(v, Mapping)
    raise SchemaError(f"unknown type {t!r}")


def _same(a: Any, b: Any) -> bool:
    from .calllog import canonical
    try:
        return canonical(a) == canonical(b)
    except (TypeError, ValueError):
        return False


def _errors(v: Any, schema: Any, base: str, path: str) -> Iterator[str]:
    """Why ``v`` fails ``schema`` (each a sentence with where), none when it passes."""
    if schema is True:
        return
    if schema is False:
        yield f"{path or '/'}: nothing is allowed here"
        return
    if "$ref" in schema:
        rbase, target = _resolve(base, schema["$ref"])
        yield from _errors(v, target, rbase, path)
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(v, t) for t in types):
            yield f"{path or '/'}: not of type {' or '.join(types)}"
            return
    if "const" in schema and not _same(v, schema["const"]):
        yield f"{path or '/'}: not {json.dumps(schema['const'])}"
    if "enum" in schema and not any(_same(v, x) for x in schema["enum"]):
        yield f"{path or '/'}: not one of {json.dumps(schema['enum'])}"
    if isinstance(v, str):
        n = len(v)
        if n < schema.get("minLength", 0):
            yield f"{path or '/'}: shorter than {schema['minLength']}"
        if "maxLength" in schema and n > schema["maxLength"]:
            yield f"{path or '/'}: longer than {schema['maxLength']}"
        if "pattern" in schema and not _pattern(schema["pattern"]).search(v):
            yield f"{path or '/'}: does not match {schema['pattern']}"
    if _is_number(v):
        if "minimum" in schema and v < schema["minimum"]:
            yield f"{path or '/'}: less than {schema['minimum']}"
        if "maximum" in schema and v > schema["maximum"]:
            yield f"{path or '/'}: more than {schema['maximum']}"
        if "exclusiveMinimum" in schema and v <= schema["exclusiveMinimum"]:
            yield f"{path or '/'}: not more than {schema['exclusiveMinimum']}"
        if "exclusiveMaximum" in schema and v >= schema["exclusiveMaximum"]:
            yield f"{path or '/'}: not less than {schema['exclusiveMaximum']}"
    if isinstance(v, (list, tuple)):
        prefix = schema.get("prefixItems") or []
        for i, (x, s) in enumerate(zip(v, prefix)):
            yield from _errors(x, s, base, f"{path}/{i}")
        if "items" in schema:
            for i in range(len(prefix), len(v)):
                yield from _errors(v[i], schema["items"], base, f"{path}/{i}")
        if len(v) < schema.get("minItems", 0):
            yield f"{path or '/'}: fewer than {schema['minItems']} items"
        if "maxItems" in schema and len(v) > schema["maxItems"]:
            yield f"{path or '/'}: more than {schema['maxItems']} items"
        if schema.get("uniqueItems"):
            from .calllog import canonical
            try:
                if len({canonical(x) for x in v}) != len(v):
                    yield f"{path or '/'}: items are not unique"
            except (TypeError, ValueError):
                yield f"{path or '/'}: not JSON"
    if isinstance(v, Mapping):
        for k in schema.get("required") or ():
            if k not in v:
                yield f"{path or '/'}: {k!r} is required"
        props = schema.get("properties") or {}
        for k, x in v.items():
            if not isinstance(k, str):
                yield f"{path or '/'}: a key that is not text"
                continue
            if k in props:
                yield from _errors(x, props[k], base, f"{path}/{k}")
            elif "additionalProperties" in schema:
                yield from _errors(x, schema["additionalProperties"], base, f"{path}/{k}")
            if "propertyNames" in schema:
                yield from _errors(k, schema["propertyNames"], base, f"{path}/{k} (its name)")
        if "maxProperties" in schema and len(v) > schema["maxProperties"]:
            yield f"{path or '/'}: more than {schema['maxProperties']} keys"
        if len(v) < schema.get("minProperties", 0):
            yield f"{path or '/'}: fewer than {schema['minProperties']} keys"
    for s in schema.get("allOf") or ():
        yield from _errors(v, s, base, path)
    if "anyOf" in schema and not any(_passes(v, s, base) for s in schema["anyOf"]):
        yield f"{path or '/'}: fits none of anyOf"
    if "oneOf" in schema:
        n = sum(1 for s in schema["oneOf"] if _passes(v, s, base))
        if n != 1:
            yield f"{path or '/'}: fits {n} of oneOf, not exactly one"
    if "not" in schema and _passes(v, schema["not"], base):
        yield f"{path or '/'}: fits what 'not' refuses"
    if "if" in schema:
        if _passes(v, schema["if"], base):
            if "then" in schema:
                yield from _errors(v, schema["then"], base, path)
        elif "else" in schema:
            yield from _errors(v, schema["else"], base, path)


def _passes(v: Any, schema: Any, base: str) -> bool:
    return next(_errors(v, schema, base, ""), None) is None


def problem(name: str, value: Any) -> Optional[str]:
    """Why ``value`` does not pass the contract's schema ``name`` (``"event"``,
    ``"saved"``, ``"interface"``, ``"call"``, ``"rating"``), or None when it does."""
    doc = _by_name(name)
    return next(_errors(value, doc, doc["$id"], ""), None)


def valid(name: str, value: Any) -> bool:
    return problem(name, value) is None


def problems(name: str, value: Any, limit: int = 20) -> List[str]:
    """Up to ``limit`` reasons ``value`` fails schema ``name``."""
    doc = _by_name(name)
    out: List[str] = []
    for p in _errors(value, doc, doc["$id"], ""):
        out.append(p)
        if len(out) >= limit:
            break
    return out


__all__ = ["problem", "problems", "valid", "SchemaError", "NAMES"]
