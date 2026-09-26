"""Changing a function's model, connection and layout: at definition, afterwards, or on a copy."""

import lm15
import lmcc
import pytest

import functai
from conftest import FakeRouter
from functai import _ai, ai, system, turns, user

XML = "<result>\nok\n</result>"
QA = lmcc.adapter(messages=[
    lmcc.system("{instruction}\n{% for f in outputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"),
    lmcc.turns(), lmcc.user("Q: {x}")])


def last_user(r):
    return r.requests[-1].messages[-1].parts[0].text


def make(**kw):
    @ai(**kw)
    def f(x: str) -> str:
        """Echo."""
        return _ai
    return f


def test_a_copy_can_switch_from_a_template_to_an_adapter_and_back(fake):
    r = fake(responder=lambda req: XML if "<result>" in (req.system or "") else "plain")
    f = make(template=[system("{instruction}"), user("T: {x}")])
    assert f("a") == "plain" and last_user(r) == "T: a"

    g = f.using(adapter=QA)                          # the adapter replaces the template (was ignored)
    assert g("b") == "ok" and last_user(r) == "Q: b"
    assert f.template is not None and g.template is None and f.adapter is None

    h = g.using(template=[system("{instruction}"), user("U: {x}")])
    assert h("c") == "plain" and last_user(r) == "U: c" and h.adapter is None

    d = f.using(template=None)                       # no template: the default layout
    assert d("d") == "ok" and "<x>\nd\n</x>" in last_user(r)
    f("e")
    assert last_user(r) == "T: e"                    # the original is untouched


def test_a_saved_adapter_artifact_works_everywhere(fake):
    fake(XML)
    f = make(adapter=QA.dump())
    assert f("a") == "ok"


def test_setters_switch_layouts_and_refuse_bad_values_before_changing_anything(fake):
    r = fake(responder=lambda req: XML if "<result>" in (req.system or "") else "plain")
    f = make()
    f.template = [system("{instruction}"), user("T: {x}")]
    f("a")
    assert last_user(r) == "T: a"
    with pytest.raises(lmcc.Refusal):
        f.template = [system("{instruction"), user("{x}")]
    assert f.template[1] == user("T: {x}")          # unchanged
    with pytest.raises(ValueError, match="unknown adapter"):
        f.adapter = "yaml"
    assert f.template is not None                   # unchanged
    f.adapter = QA
    f("b")
    assert last_user(r) == "Q: b" and f.template is None
    with pytest.raises(TypeError, match="not both"):
        f.using(adapter="chat", template=[user("{x}")])


def test_using_none_means_inherit(fake):
    r = fake(responder=lambda req: XML)
    functai.configure(temperature=0.3)
    f = make(temperature=0.9)
    f("a")
    assert r.requests[-1].config.temperature == 0.9
    f.using(temperature=None)("b")
    assert r.requests[-1].config.temperature == 0.3


class OneProvider:
    """Stands in for an lm15 provider LM (OpenAILM, ClaudeCodeLM, ...)."""

    def __init__(self, provider="openai"):
        self.provider = provider
        self.inner = FakeRouter(responder=lambda req: XML, provider=provider)

    def complete(self, request):
        return self.inner.complete(request)


def test_a_provider_lm_is_a_connection_given_as_client():
    conn = OneProvider("openai")
    f = make(lm="gpt-4.1-mini", client=conn)
    assert f("a") == "ok"
    assert conn.inner.requests[-1].model == "gpt-4.1-mini"
    g = make(lm="openai:gpt-4.1", client=conn)       # its own prefix is dropped: the wire name goes out
    g("b")
    assert conn.inner.requests[-1].model == "gpt-4.1"
    claude = OneProvider("claude-code")
    make(lm="claude:claude-haiku-4-5", client=claude)("c")
    assert claude.inner.requests[-1].model == "claude-haiku-4-5"
    with pytest.raises(ValueError, match="client is a 'openai' connection"):
        make(lm="anthropic:claude-haiku-4-5", client=conn)("d")


def test_a_provider_lm_given_as_the_model_says_what_to_do():
    lm = lm15.OpenAILM(api_key="sk-test")
    with pytest.raises(TypeError, match="Pass it as client= and the model name as lm="):
        make(lm=lm)
    f = make()
    with pytest.raises(TypeError, match="client="):
        f.lm = lm
    with pytest.raises(TypeError, match="client="):
        functai.configure(lm=lm)
    with pytest.raises(TypeError, match="client must be"):
        make(client="openai")
    with pytest.raises(TypeError, match="since functai 1.0"):
        functai.configure(router=object())


def test_the_real_lm15_objects_are_recognized():
    from functai import models
    models.check_client(lm15.OpenAILM(api_key="sk-test"))
    models.check_client(lm15.LMRouter())
    with pytest.raises(TypeError, match="asynchronous"):
        models.check_client(lm15.AsyncOpenAILM(api_key="sk-test"))
    client, model, route = models.resolve({"lm": "gpt-4.1", "client": lm15.OpenAILM(api_key="sk-test")})
    assert (model, route.provider, route.model) == ("gpt-4.1", "openai", "gpt-4.1")


class Selection:
    def __init__(self, provider, model):
        self.provider, self.model, self.routed = provider, model, f"{provider}:{model}"


class Bound:
    """Stands in for lm15's BoundClient: one login, one model."""

    def __init__(self, provider, model):
        self.selection = Selection(provider, model)
        self.provider, self.model = provider, model
        self.inner = FakeRouter(responder=lambda req: XML, provider=provider)

    def complete(self, request):
        assert request.model == self.selection.routed
        return self.inner.complete(request)


def test_a_bound_client_is_a_model_with_its_connection(fake):
    fake()
    b = Bound("github-copilot", "gpt-4.1")
    f = make(lm=b)
    assert f("a") == "ok"
    assert f.plan().capabilities["native_function_calling"] is True
    with pytest.raises(TypeError, match="drop client="):
        make(lm=b, client=OneProvider())                 # both in one place: a contradiction
    functai.configure(client=OneProvider("openai"))      # a wider client does not apply to it
    assert f("b") == "ok" and len(b.inner.requests) == 2
    with pytest.raises(TypeError, match="pass it as lm="):
        make(client=b)
