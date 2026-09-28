"""The contract's JSON Schemas, for make.py to check every record, event,
manifest and interface it writes into a case (schema/)."""

import json
from pathlib import Path

import jsonschema
from referencing import Registry, Resource

SCHEMA = Path(__file__).resolve().parent.parent / "schema"
DOCS = {p.stem.removesuffix(".schema"): json.loads(p.read_text()) for p in sorted(SCHEMA.glob("*.schema.json"))}
REGISTRY = Registry().with_resources([(d["$id"], Resource.from_contents(d)) for d in DOCS.values()])


def validator(name: str, pointer: str = "") -> jsonschema.Draft202012Validator:
    schema = DOCS[name]
    if pointer:
        schema = {"$ref": schema["$id"] + "#" + pointer}
    return jsonschema.Draft202012Validator(schema, registry=REGISTRY)


CALL, RATING, EVENT, SAVED = validator("call"), validator("rating"), validator("event"), validator("saved")
INTERFACE = validator("saved", "/$defs/interface")


def check(v: jsonschema.Draft202012Validator, value, where: str) -> None:
    errors = sorted(v.iter_errors(value), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        raise AssertionError(f"{where}: {e.message} at {list(e.path)}")


def record(rec: dict, where: str) -> None:
    if "functai_call" in rec:
        check(CALL, rec, where)
    elif "functai_rating" in rec:
        check(RATING, rec, where)
    else:
        raise AssertionError(f"{where}: neither a call nor a rating")
