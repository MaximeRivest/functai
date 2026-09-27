"""The contract every language shares (../contract), held by the Python
implementation: AI functions (functions.md: signature, request, version),
the layouts and the model table as data, scores (scores.md), saved
manifests (saved.md). The call log's cases are in test_call_log.py."""

import dataclasses
import json
import tempfile
import textwrap
from pathlib import Path

import jsonschema
import pytest

import functai
from functai import adapters, calllog, evaluation, models
from functai.saved import probe_plan, probe_request, request_fingerprint

CONTRACT = Path(__file__).resolve().parents[2] / "contract"


def load(path: Path):
    return json.loads(path.read_text())


def case_files(folder: str):
    return sorted((CONTRACT / "cases" / folder).glob("*.json"))


# ------------------------------------------------------------------ a definition as Python code


def annotation(shape: dict, name: str, classes: list) -> str:
    """Python type for a JSON Schema shape (records become dataclasses in ``classes``)."""
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


def python_function(d: dict):
    """The definition written the way a Python user writes it, then decorated."""
    classes: list = []
    params = []
    for f in d["inputs"]:
        params.append(f"    {f['name']}: {annotation(f['shape'], f['name'], classes)},"
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
    source = "\n\n".join(classes) + "\n\n" + f"def {d['name']}(\n" + "\n".join(params) + \
        f"\n) -> {returns}:\n" + "\n".join(body) + "\n"
    namespace = {"__name__": "contract_case"}
    # a real file, so inspect finds the source (comments are guidance)
    path = Path(tempfile.mkdtemp()) / "contract_case.py"
    path.write_text("import dataclasses\nfrom typing import Literal, Optional\nfrom functai import _ai\n\n" + source)
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    fn = namespace[d["name"]]
    settings = {k: v for k, v in d["settings"].items()}
    if d.get("tools"):
        from lmcc_std.tools import Tool
        settings["tools"] = [Tool(**t) for t in d["tools"]]
    program = functai.ai(**settings)(fn) if settings else functai.ai(fn)
    program.load_state(d["state"])
    return program


def without_type(signature) -> dict:
    data = json.loads(json.dumps(__import__("lmcc").signature_to_dict(signature)))
    fields = []
    for f in data["fields"]:
        f = {k: v for k, v in f.items() if k != "type" and v is not None}
        f.setdefault("purpose", "plain")
        fields.append(f)
    return {"instructions": data["instructions"], "fields": fields}


@pytest.mark.parametrize("path", case_files("functions"), ids=lambda p: p.stem)
def test_function_case(path):
    case = load(path)
    d, expect = case["definition"], case["expect"]
    fn = python_function(d)
    spec, settings = fn._spec(), fn._effective()
    got = without_type(spec.signature)
    want = {"instructions": expect["signature"]["instructions"],
            "fields": [{**f, "purpose": f.get("purpose", "plain")} for f in expect["signature"]["fields"]]}
    assert got == want
    assert functai.saved._sample_inputs(spec) == expect["sample"]
    plan, past = probe_plan(fn, spec, settings)
    request = probe_request(fn, spec, plan, past, expect["sample"])
    assert json.loads(json.dumps(request)) == expect["request"]
    assert request_fingerprint(request) == expect["request_hash"]
    assert fn.version == expect["version"]
    assert calllog.signature_id(spec.signature) == expect["signature_id"]


def test_the_contract_has_function_cases():
    assert len(case_files("functions")) >= 10


# ------------------------------------------------------------------ data every language reads


@pytest.mark.parametrize("name", ["xml", "chat", "json"])
def test_the_layouts_are_the_contracts(name):
    mine = adapters.resolve_adapter(name).dump()
    theirs = load(CONTRACT / "layouts" / f"{name}.json")
    mine["versions"].pop("kernel"), theirs["versions"].pop("kernel")
    assert mine == theirs


def test_the_model_table_is_the_contracts():
    table = load(CONTRACT / "models.json")
    assert set(table["native"]["providers"]) == models.NATIVE_PROVIDERS
    assert {k: tuple(v) for k, v in table["native"]["reasoning_prefixes"].items()} == models._REASONING_PREFIXES
    assert {k: v for k, v in table["speaks_as"].items() if k != "about"} == models.SUBSCRIPTION_AS
    assert set(table["chat_completions"]["providers"]) == models.CHAT_COMPLETIONS
    assert set(table["native_tool_hosts"]["providers"]) == models.NATIVE_TOOL_HOSTS
    assert set(table["judgment_only"]["providers"]) == models.JUDGMENT_ONLY
    assert {k: v for k, v in table["probe"].items() if k != "about"} == functai.saved.PROBE_CAPABILITIES
    for provider in table["native"]["providers"]:
        caps = models.capabilities(provider, "some-model")
        assert caps["stop_sequences"] == (provider not in table["native"]["no_stop_sequences"])
    assert {k: tuple(v) for k, v in table["fixed_sampling"].items() if k != "about"} == models._FIXED_SAMPLING
    assert models.capabilities("anthropic", "claude-3-5-haiku")["assistant_prefill"] is True
    assert models.capabilities("openai", "gpt-4.1")["assistant_prefill"] is False



def test_a_sampling_the_model_does_not_take_is_left_out_once(recwarn):
    from types import SimpleNamespace
    models._warned.clear()
    luna = SimpleNamespace(provider="openai", model="gpt-6-luna")
    assert models.adjust({"temperature": 1}, luna) == {"temperature": 1}      # 1 is what it samples at
    out = models.adjust({"temperature": 0, "top_p": 0.5}, luna)
    assert out["temperature"] is None and out["top_p"] is None and out["_dropped"] == ("temperature", "top_p")
    models.adjust({"temperature": 0, "top_p": 0.5}, luna)
    assert len([w for w in recwarn if "does not take" in str(w.message)]) == 1
    sonnet = SimpleNamespace(provider="claude-code", model="claude-sonnet-5")
    assert models.adjust({"temperature": 0}, sonnet)["temperature"] is None
    haiku = SimpleNamespace(provider="anthropic", model="claude-haiku-4-5")
    assert models.adjust({"temperature": 0}, haiku) == {"temperature": 0}


# ------------------------------------------------------------------ scores


@pytest.mark.parametrize("path", case_files("scores"), ids=lambda p: p.stem)
def test_score_case(path):
    case = load(path)
    if case["kind"] == "interval":
        mean, low, high = evaluation.interval([float(v) for v in case["values"]])
        for got, want in zip((mean, low, high), (case["expect"][k] for k in ("mean", "low", "high"))):
            assert (got is None and want is None) or abs(got - want) < 1e-12
        return
    answers, prediction = case["answers"], case["prediction"]
    assert evaluation.exact_match(answers, prediction) == case["expect"]["exact_match"]
    keys = [k for k in prediction if k in answers]
    if len(keys) > 1:
        for k in keys:
            assert evaluation.exact_match({k: answers[k]}, {k: prediction[k]}) == case["expect"][f"{k}_match"]


# ------------------------------------------------------------------ saved manifests


@pytest.fixture(scope="module")
def saved_schema():
    schema = load(CONTRACT / "schema" / "saved.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


@pytest.mark.parametrize("path", case_files("saved"), ids=lambda p: p.stem)
def test_saved_case_manifests_pass_the_schema(path, saved_schema):
    case = load(path)
    if case["expect"].get("refuses") == "saved-format":
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(case["manifest"], saved_schema)
    else:
        jsonschema.validate(case["manifest"], saved_schema)


def test_a_python_save_writes_what_another_language_loads(tmp_path, saved_schema, monkeypatch):
    """What save writes passes the schema and says its language, its body and its version."""
    src = tmp_path / "shop.py"
    src.write_text(textwrap.dedent('''
        from typing import Literal
        from functai import ai, _ai

        @ai(lm="gpt-4.1-mini", temperature=0)
        def mood(review: str) -> Literal["happy", "unhappy", "mixed"]:
            """How does the customer feel about what they bought?"""

        @ai(lm="gpt-4.1-mini")
        def rounded(x: float) -> float:
            """Double it."""
            return round(_ai, 2)
        '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    import shop
    functai.save(shop.mood, tmp_path / "mood")
    functai.save(shop.rounded, tmp_path / "rounded")
    m = load(tmp_path / "mood" / "functai.json")
    jsonschema.validate(m, saved_schema)
    node = m["nodes"]["shop:mood"]["ai"]
    assert m["language"] == "python" and node["body"] is None and node["version"] == shop.mood.version
    r = load(tmp_path / "rounded" / "functai.json")["nodes"]["shop:rounded"]["ai"]
    assert r["body"] == {"code": calllog.code_hash(shop.rounded.__wrapped__)}
    _ = dataclasses
