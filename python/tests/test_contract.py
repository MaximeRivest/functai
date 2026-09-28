"""The contract every language shares (../contract), held by the Python
implementation: AI functions (functions.md: signature, request, version),
the layouts and the model table as data, scores (scores.md), saved
manifests (saved.md). The call log's cases are in test_call_log.py."""

import dataclasses
import json
import textwrap

import jsonschema
import pytest

import functai
from functai import adapters, calllog, evaluation, models
from functai.saved import probe_plan, probe_request, request_fingerprint
from conftest import FakeRouter
from contract_support import CONTRACT, case_files, load, python_function, validator


# ------------------------------------------------------------------ AI functions (functions.md)


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
    """The manifest's schema, with the interface schema it refers to, its
    patterns read as ECMA-262 reads them (contract_support)."""
    return validator("saved")


@pytest.mark.parametrize("path", case_files("saved"), ids=lambda p: p.stem)
def test_saved_case_manifests_pass_the_schema(path, saved_schema):
    case = load(path)
    if case["expect"].get("refuses") == "saved-format":
        with pytest.raises(jsonschema.ValidationError):
            saved_schema.validate(case["manifest"])
    else:
        saved_schema.validate(case["manifest"])


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
    saved_schema.validate(m)
    node = m["nodes"]["shop:mood"]["ai"]
    assert m["language"] == "python" and node["body"] is None and node["version"] == shop.mood.version
    r = load(tmp_path / "rounded" / "functai.json")["nodes"]["shop:rounded"]["ai"]
    assert r["body"] == {"code": calllog.code_hash(shop.rounded.__wrapped__)}
    _ = dataclasses


# ------------------------------------------------------------------ saved folders another language reads (saved.md)


def loaded(case):
    """What loading the case's node gives, in the case's words."""
    from functai.saved import from_manifest
    try:
        fn = from_manifest(case["manifest"], node=case["node"])
    except functai.LoadRefused as err:
        return {"refuses": err.code}, None
    spec, settings = fn._spec(), fn._effective()
    plan, past = probe_plan(fn, spec, settings)
    node = case["manifest"]["nodes"][case["node"] or case["manifest"]["entry"]]
    requests = []
    for probe in node["ai"]["probes"]:
        try:
            requests.append(request_fingerprint(probe_request(fn, spec, plan, past, probe)))
        except __import__("lmcc").Refusal as exc:
            requests.append(f"refused:{exc.code}")
    return {"loads": {"name": fn.__name__, "module": calllog.program_info(fn)["module"], "version": fn.version,
                      "signature_id": calllog.signature_id(fn.signature), "requests": requests}}, fn


@pytest.mark.parametrize("path", case_files("saved"), ids=lambda p: p.stem)
def test_saved_case_loads(path, monkeypatch, tmp_path):
    case = load(path)
    expect = {k: v for k, v in case["expect"].items() if k not in ("describe", "sends")}
    got, fn = loaded(case)
    assert got == expect
    if not case["expect"].get("sends"):
        return
    # The loaded function called for real (a fake model answers) under the probe facts (the capabilities and the
    # provider fingerprints are rendered with): the request its call rendered, as the probe hash names it.
    import lmcc
    import lmcc_lm15
    rendered = []
    real = lmcc_lm15.request

    def spy(render, **kwargs):
        rendered.append(render)
        return real(render, **kwargs)

    monkeypatch.setattr(lmcc_lm15, "request", spy)
    router = FakeRouter(responder=lambda request: "<result>\nok\n</result>")
    with functai.configure(client=router, lm="probe:model", capabilities=functai.saved.PROBE_CAPABILITIES,
                           log_calls=tmp_path):
        for send in case["expect"]["sends"]:
            fn(**send["inputs"])                  # an optional input left out: its default is sent
            assert request_fingerprint(rendered[-1].request("probe")) == send["request_hash"], send
    records = [json.loads(line) for f in tmp_path.rglob("*.jsonl") for line in f.read_text().splitlines()]
    hashes = [r["exchanges"][0]["request_hash"] for r in sorted(records, key=lambda r: r["id"])]
    assert hashes == [lmcc.turn.sha256(r.request()) for r in rendered]      # what the record says was sent


@pytest.mark.parametrize("path", case_files("saved"), ids=lambda p: p.stem)
def test_saved_case_describes(path):
    case = load(path)
    try:
        got = {"interface": functai.describe(case["manifest"], node=case["node"])}
    except functai.LoadRefused as err:
        got = {"refuses": err.code}
    assert got == case["expect"]["describe"]


def test_a_folder_of_another_language_loads_from_its_data(tmp_path, fake):
    """functai.load reads a folder another language wrote without running any
    code (no trust needed), and the function it gives calls a model."""
    case = load(CONTRACT / "cases" / "saved" / "16-an-optional-input.json")
    manifest = {**case["manifest"], "language": "typescript"}
    (tmp_path / "functai.json").write_text(json.dumps(manifest))
    reply = functai.load(tmp_path)
    assert reply.interface == manifest["nodes"]["shop:reply"]["interface"]
    router = fake("<result>\nHello!\n</result>")
    assert reply("Hi") == "Hello!"
    assert "<tone>\nkind\n</tone>" in router.user()               # the default, from the interface
    assert functai.describe(tmp_path)["inputs"][1]["optional"] is True
