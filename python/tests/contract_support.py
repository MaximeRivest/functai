"""What the contract harnesses share: the case files, the schemas read as
JSON Schema reads them, and native values for the cases' stand-ins.

The schemas' patterns are read as ECMA-262 reads them (the standard JSON
Schema follows): a pattern's ``$`` is the end of the text. Python's ``re``
lets ``$`` match before a final newline too, so a stock validator would
accept ``"message\\n"`` as a name where every other language refuses it
(design/08, *Found while doing this*)."""

import json
import re
import tempfile
from pathlib import Path

import functai

import jsonschema
from referencing import Registry, Resource

CONTRACT = Path(__file__).resolve().parents[2] / "contract"


def load(path: Path):
    return json.loads(path.read_text())


def case_files(folder: str, prefix: str = ""):
    return sorted((CONTRACT / "cases" / folder).glob(f"{prefix}*.json"))


def _ecma(node):
    if isinstance(node, dict):
        out = {k: _ecma(v) for k, v in node.items()}
        p = node.get("pattern")
        if isinstance(p, str):
            assert re.fullmatch(r"[^$]*\$\)*", p), f"a pattern whose $ is not its last anchor: {p}"
            out["pattern"] = re.sub(r"\$(?=\)*$)", r"\\Z", p)
        return out
    if isinstance(node, list):
        return [_ecma(v) for v in node]
    return node


_DOCS = {p.name.removesuffix(".schema.json"): _ecma(load(p)) for p in (CONTRACT / "schema").glob("*.schema.json")}
_REGISTRY = Registry().with_resources([(d["$id"], Resource.from_contents(d)) for d in _DOCS.values()])


def validator(name: str) -> jsonschema.Draft202012Validator:
    jsonschema.Draft202012Validator.check_schema(_DOCS[name])
    return jsonschema.Draft202012Validator(_DOCS[name], registry=_REGISTRY)


def assert_valid(v: jsonschema.Draft202012Validator, value, where: str = "") -> None:
    errors = sorted(v.iter_errors(value), key=lambda e: list(e.path))
    assert not errors, f"{where}: {errors[0].message} at {list(errors[0].path)}"


# ------------------------------------------------------------------ values with no JSON form


def is_stand_in(value) -> bool:
    """A case's stand-in for a value with no JSON form: exactly {"$type", "$repr"}, and "$text" when its type
    gives it a text of its own (a data frame, a date)."""
    return isinstance(value, dict) and set(value) in ({"$type", "$repr"}, {"$type", "$repr", "$text"})


_classes = {}


def native(value):
    """The case's JSON with each stand-in replaced by a native value of ours
    with no JSON form, whose type name and repr are the stand-in's."""
    if is_stand_in(value):
        cls = _classes.get(value["$type"])
        if cls is None:
            def rep(self):
                return self._repr

            def text(self):                   # its own text, or only the default for any object
                return self._text if self._text is not None else f"<{type(self).__name__} object at 0x7f>"

            cls = _classes[value["$type"]] = type(value["$type"], (), {"__repr__": rep, "__str__": text,
                                                                        "__slots__": ("_repr", "_text")})
        obj = cls()
        obj._repr = value["$repr"]
        obj._text = value.get("$text")
        return obj
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native(v) for v in value]
    return value


def as_json(value):
    """Back to the case's JSON: a native value with no JSON form is its stand-in."""
    if type(value).__name__ in _classes and isinstance(value, _classes[type(value).__name__]):
        out = {"$type": type(value).__name__, "$repr": value._repr}
        if value._text is not None:
            out["$text"] = value._text
        return out
    if isinstance(value, dict):
        return {k: as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def without_type(interface: dict) -> dict:
    """An interface with the host type names left out: ``type`` is for people,
    never compared across languages (programs.md)."""
    def field(f):
        return {k: v for k, v in f.items() if k != "type"}
    return {**interface, "inputs": [field(f) for f in interface["inputs"]],
            "outputs": [field(f) for f in interface["outputs"]]}


# ------------------------------------------------------------------ a definition as Python code (functions.md)


_BOUNDS = {"minimum": "ge", "maximum": "le", "exclusiveMinimum": "gt", "exclusiveMaximum": "lt",
           "minLength": "min_length", "maxLength": "max_length", "minItems": "min_length", "maxItems": "max_length"}


def annotation(shape: dict, name: str, classes: list) -> str:
    """Python type for a JSON Schema shape (records become dataclasses in
    ``classes``; bounds are pydantic ``Field`` constraints, as a Python user
    writes them)."""
    bounds = {_BOUNDS[k]: v for k, v in shape.items() if k in _BOUNDS}
    if bounds:
        plain = {k: v for k, v in shape.items() if k not in _BOUNDS}
        args = ", ".join(f"{k}={v!r}" for k, v in bounds.items())
        return f"Annotated[{annotation(plain, name, classes)}, Field({args})]"
    if "enum" in shape:
        return "Literal[" + ", ".join(repr(v) for v in shape["enum"]) + "]"
    if "anyOf" in shape:
        [inner] = [s for s in shape["anyOf"] if s.get("type") != "null"]
        return f"Optional[{annotation(inner, name, classes)}]"
    t = shape.get("type")
    if t == "array":
        return f"list[{annotation(shape['items'], name, classes)}]"
    if t == "object" and "properties" in shape:
        cls = f"Rec_{name}"
        lines = [f"@dataclasses.dataclass\nclass {cls}:"]
        lines += [f"    {k}: {annotation(v, name + '_' + k, classes)}" for k, v in shape["properties"].items()]
        classes.append("\n".join(lines))
        return cls
    if t == "object":
        return f"dict[str, {annotation(shape['additionalProperties'], name, classes)}]"
    return {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}[t]


def writable(d: dict) -> bool:
    """Whether Python annotations can write the definition's shapes: a list of
    types (``{"type": ["string", "null"]}``) has no annotation of its own
    (``Optional[str]`` is ``anyOf``); Python meets one in a folder another
    language saved."""
    def plain(shape):
        inner_default = any("default" in x for x in (shape.get("properties") or {}).values())
        return not isinstance(shape.get("type"), list) and not inner_default and all(
            plain(x) for k in ("items", "additionalProperties") if isinstance(shape.get(k), dict)
            for x in [shape[k]]) and all(plain(x) for x in (shape.get("properties") or {}).values()) \
            and all(plain(x) for x in shape.get("anyOf") or [])
    return all(plain(f["shape"]) for f in [*d["inputs"], *d["outputs"]])


def loaded_function(d: dict):
    """The definition as a folder another language saved (the contract's rules make its node), loaded."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("contract_rules_saved", CONTRACT / "cases" / "saved.py")
    import sys
    sys.path.insert(0, str(CONTRACT / "cases"))
    try:
        rules = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules)
    finally:
        sys.path.remove(str(CONTRACT / "cases"))
    key = f"shop:{d['name']}"
    manifest = rules.manifest({key: rules.node(d["name"], d)}, key)
    manifest["language"] = "typescript"
    return functai.saved.from_manifest(manifest)


def python_function(d: dict):
    """The definition written the way a Python user writes it, then decorated."""
    classes: list = []
    params = []
    helpers = []
    for f in d["inputs"]:
        default = f" = {f['shape']['default']!r}" if f.get("optional") else ""
        if "default_code" in f:
            # a default written as an expression: a function of that name gives the shape's default today
            called = f["default_code"].split("(")[0]
            helpers.append(f"def {called}():\n    return {f['shape']['default']!r}\n")
            default = f" = {f['default_code']}"
        params.append(f"    {f['name']}: {annotation(f['shape'], f['name'], classes)}{default},"
                      + (f"  # {f['desc']}" if f.get("desc") else ""))
    *extras, main = d["outputs"]
    body = [f'    """{d["description"]}"""'] if d["description"] else []
    for f in extras:
        body.append(f"    {f['name']}: {annotation(f['shape'], f['name'], classes)} = "
                    f"_ai[{f.get('desc', '')!r}]")
    if main.get("desc"):
        body.append(f"    {main['name']}: {annotation(main['shape'], main['name'], classes)} = _ai[{main['desc']!r}]")
    if not body:
        body = ["    ..."]
    returns = annotation(main["shape"], "result", classes)
    source = "\n\n".join(classes + helpers) + "\n\n" + f"def {d['name']}(\n" + "\n".join(params) + \
        f"\n) -> {returns}:\n" + "\n".join(body) + "\n"
    namespace = {"__name__": "contract_case"}
    # a real file, so inspect finds the source (comments are guidance)
    path = Path(tempfile.mkdtemp()) / "contract_case.py"
    path.write_text("import dataclasses\nfrom typing import Annotated, Literal, Optional\nfrom pydantic import Field\n"
                    "from functai import _ai\n\n" + source)
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    fn = namespace[d["name"]]
    settings = {k: v for k, v in d["settings"].items()}
    if d.get("tools"):
        from lmcc_std.tools import Tool
        settings["tools"] = [Tool(**t) for t in d["tools"]]
    program = functai.ai(**settings)(fn) if settings else functai.ai(fn)
    program.load_state(d["state"])
    return program
