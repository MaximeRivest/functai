"""One call: the function is the prompt, `_ai` is the model's answer."""

import dataclasses
import enum
import threading
from typing import List, Literal, Optional, Tuple

import lmcc
import pytest

import functai
from functai import _ai, ai, configure


def test_a_function_calls_the_model_and_returns_its_type(fake):
    @ai
    def summarize(text: str, focus: str = "key points") -> str:
        """Summarize the text in one concise sentence."""
        return _ai

    r = fake("<result>\nShort.\n</result>")
    assert summarize("long text") == "Short."
    req = r.requests[0]
    assert req.model == "gpt-4.1-mini"
    assert "Summarize the text in one concise sentence." in req.system
    assert "<result>" in req.system
    assert "<text>\nlong text\n</text>" in r.user()
    assert "<focus>\nkey points\n</focus>" in r.user()


def test_empty_body_ellipsis_and_docstring_only_all_call_the_model(fake):
    @ai
    def a(x: str) -> int:
        """Count."""
        ...

    @ai
    def b(x: str) -> int:
        """Count."""

    fake("<result>\n3\n</result>", "<result>\n4\n</result>")
    assert a("x") == 3
    assert b("x") == 4


def test_post_processing_with_the_sentinel(fake):
    @ai
    def score(text: str) -> float:
        """A score between 0 and 1."""
        s = _ai
        return max(0.0, min(1.0, float(s)))

    fake("<s>\n1.7\n</s>")                    # the output is named after the variable
    assert score("great") == 1.0


def test_declared_outputs_come_first_and_all_returns_everything(fake):
    @ai
    def solve(question: str) -> float:
        """Solve the math problem."""
        reasoning: str = _ai["Step-by-step thinking."]
        return _ai

    r = fake("<reasoning>\n120/2\n</reasoning>\n<result>\n60\n</result>")
    pred = solve("speed?", all=True)
    assert pred.reasoning == "120/2" and pred.result == 60.0 and isinstance(pred.result, float)
    assert dict(pred) == {"reasoning": "120/2", "result": 60.0}
    assert r.system().index("<reasoning>") < r.system().index("<result>")
    assert "- reasoning: Step-by-step thinking." in r.system()
    assert pred.turn.outputs == {"reasoning": "120/2", "result": 60.0}
    assert pred.usage["total_tokens"] == 5


def test_several_outputs_returned_as_a_tuple(fake):
    @ai
    def critique_and_improve(text: str) -> Tuple[str, str, int]:
        """Critique the text and improve it."""
        critique: str = _ai["Constructive criticism."]
        improved_text: str = _ai["The improved version."]
        return critique, improved_text, 1

    fake("<critique>\nToo terse.\n</critique>\n<improved_text>\nPlease fix it.\n</improved_text>")
    assert critique_and_improve("fix it") == ("Too terse.", "Please fix it.", 1)


def test_bare_sentinels_in_a_returned_tuple_map_by_name(fake):
    @ai
    def split(text: str):
        """Extract an id and an email."""
        id: str = _ai
        email: str = _ai
        return (id, email)

    fake("<id>\n42\n</id>\n<email>\na@b.c\n</email>")
    assert split("user 42 a@b.c") == ("42", "a@b.c")


def test_returning_a_named_output_uses_its_annotation(fake):
    @ai
    def keywords(article: str) -> list[str]:
        """Five key terms."""
        keywords: list[str] = _ai
        return [k.lower() for k in keywords]

    fake('<keywords>\n["LLM", "Python"]\n</keywords>')
    assert keywords("...") == ["llm", "python"]


def test_format_specs_on_the_sentinel(fake):
    @ai
    def price(text: str) -> float:
        """The price."""
        p = _ai
        return f"${p:.2f}"

    fake("<p>\n9.5\n</p>")
    assert price("nine fifty") == "$9.50"


def test_unannotated_parameters_are_text_and_objects_are_written_as_text(fake):
    @ai
    def judge(example, prediction) -> float:
        """High if the prediction matches the example."""
        return _ai

    r = fake("<result>\n0.9\n</result>")
    assert judge(functai.Example(q="a", result="b"), {"result": "b"}) == 0.9
    assert "Example(q='a', result='b')" in r.user()
    assert '"result": "b"' in r.user()


# ------------------------------------------------------------------ types


@dataclasses.dataclass
class ProductInfo:
    name: str
    price: float
    features: List[str]
    in_stock: bool


class Priority(enum.Enum):
    LOW = "low"
    HIGH = "high"


def test_a_dataclass_return_is_one_structured_value(fake):
    @ai
    def extract(description: str) -> ProductInfo:
        """Extract product information."""
        return _ai

    r = fake('<result>\n{"name": "iPhone", "price": 999, "features": ["5G"], "in_stock": true}\n</result>')
    p = extract("iPhone $999")
    assert p == ProductInfo("iPhone", 999.0, ["5G"], True) and isinstance(p.price, float)
    assert "JSON" in r.system()


def test_a_structured_input_is_written_as_json(fake):
    @ai
    def valid(invoice: ProductInfo) -> bool:
        """Is it complete?"""
        return _ai

    r = fake("<result>\ntrue\n</result>")
    assert valid(ProductInfo("x", 1.0, [], True)) is True
    assert '"price": 1' in r.user()


def test_enums_and_literals_restrict_the_answer(fake):
    @ai
    def priority(issue: str) -> Priority:
        """Classify the priority."""
        return _ai

    @ai
    def categorize(text: str) -> Literal["sport", "fashion"]:
        ...

    r = fake("<result>\nhigh\n</result>", "<result>\nsport\n</result>")
    assert priority("db down") is Priority.HIGH
    assert "low, high" in r.system(0)
    assert categorize("vibe coding is a sport") == "sport"


def test_pydantic_models_optional_and_tuples(fake):
    pydantic = pytest.importorskip("pydantic")

    class Analysis(pydantic.BaseModel):
        sentiment: str
        themes: list[str]

    @ai
    def analyze(text: str) -> Analysis:
        """Analyze."""
        return _ai

    @ai
    def maybe(text: str) -> Optional[int]:
        """A number, if any."""
        return _ai

    @ai
    def pair(text: str) -> Tuple[str, int]:
        """Name and age."""
        return _ai

    fake('<result>\n{"sentiment": "positive", "themes": ["design"]}\n</result>',
         "<result>\nnull\n</result>", '<result>\n["Ann", 41]\n</result>')
    a = analyze("love it")
    assert isinstance(a, Analysis) and a.themes == ["design"]
    assert maybe("none here") is None
    assert pair("Ann, 41") == ("Ann", 41)


def test_plain_classes_become_dataclasses(fake):
    class Person:
        name: str  # full name
        age: int

    @ai
    def person(text: str) -> Person:
        """Extract the person."""
        return _ai

    r = fake('<result>\n{"name": "Ann", "age": 41}\n</result>')
    p = person("Ann is 41")
    assert (p.name, p.age) == ("Ann", 41)
    assert "Person.name: full name" in r.system()


def test_docments_become_guidance(fake):
    @ai
    def f(
        text: str,  # the raw input
    ) -> str:  # one sentence
        """Summarize."""
        return _ai

    r = fake("<result>\nok\n</result>")
    f("x")
    assert "- text: the raw input" in r.system()
    assert "Return guidance: one sentence" in r.system()


# ------------------------------------------------------------------ settings


def test_the_cascade_function_beats_block_beats_global(fake):
    r = fake(responder=lambda req: "<result>\nok\n</result>")

    @ai
    def g(x: str) -> str:
        """G."""

    @ai(temperature=0.1)
    def h(x: str) -> str:
        """H."""

    configure(temperature=0.5)
    g("a")
    assert r.requests[-1].config.temperature == 0.5
    with configure(temperature=0.0, lm="claude-haiku-4-5"):
        g("b")
        assert r.requests[-1].config.temperature == 0.0
        assert r.requests[-1].model == "claude-haiku-4-5"
        h("c")
        assert r.requests[-1].config.temperature == 0.1
    g("d")
    assert r.requests[-1].config.temperature == 0.5 and r.requests[-1].model == "gpt-4.1-mini"
    assert functai.settings.temperature == 0.5
    g.using(lm="gpt-4.1")("e")
    assert r.requests[-1].model == "gpt-4.1"


def test_a_block_is_scoped_to_its_thread_but_configure_is_global(fake):
    r = fake(responder=lambda req: "<result>\nok\n</result>")

    @ai
    def g(x: str) -> str:
        """G."""

    seen = {}
    with configure(lm="gpt-4.1"):
        t = threading.Thread(target=lambda: seen.setdefault("lm", functai.settings.lm))
        t.start(); t.join()
    assert seen["lm"] == "gpt-4.1-mini"


def test_unknown_settings_are_refused():
    with pytest.raises(TypeError, match="unknown setting"):
        configure(temprature=0.1)
    with pytest.raises(TypeError, match="unknown setting"):
        @ai(temprature=0.1)
        def f(x: str) -> str: ...


def test_no_model_is_a_clear_error():
    @ai
    def f(x: str) -> str: ...
    with pytest.raises(RuntimeError, match="no model configured"):
        f("x")


def test_litellm_style_model_strings_route_through_lm15():
    from functai import models
    router, model, route = models.resolve({"lm": "anthropic/claude-sonnet-4-5"})
    assert model == "anthropic:claude-sonnet-4-5" and route.provider == "anthropic"
    router, model, route = models.resolve({"lm": "groq/openai/gpt-oss-120b"})
    assert model == "groq:openai/gpt-oss-120b" and route.provider == "groq"


def test_lm15_config_fields_reach_the_request(fake):
    r = fake("<result>\nok\n</result>")

    @ai(max_tokens=50, seed=7)
    def f(x: str) -> str: ...
    f("x")
    assert r.requests[0].config.max_tokens == 50 and r.requests[0].config.seed == 7


# ------------------------------------------------------------------ robustness


def test_an_unreadable_reply_is_asked_again_once(fake):
    @ai
    def f(x: str) -> int:
        """A number."""

    r = fake("I think it is 3", "<result>\n3\n</result>")
    pred = f("x", all=True)
    assert pred.result == 3 and pred.attempts == 2
    assert "could not be read" in r.user(1)


def test_retries_zero_raises_the_refusal(fake):
    @ai(retries=0)
    def f(x: str) -> int: ...
    fake("no idea")
    with pytest.raises(lmcc.Refusal) as err:
        f("x")
    assert err.value.code.startswith("parse-")


def test_misspelled_tags_are_repaired_and_reported(fake):
    @ai
    def f(x: str) -> str: ...
    fake("**<Result>**\nok\n</RESULT>")
    pred = f("x", all=True)
    assert pred.result == "ok" and [r["saw"] for r in pred.repairs] == ["**<Result>**", "</RESULT>"]


def test_the_reply_cache_is_off_unless_turned_on(fake):
    @ai
    def f(x: str) -> str: ...
    r = fake(responder=lambda req: "<result>\nok\n</result>")
    f("x"); f("x")
    assert len(r.requests) == 2                      # off by default: every call reaches the model
    with configure(cache_replies=True):
        f("x"); f("x")
    assert len(r.requests) == 3                      # the second identical call was answered from the cache
    assert "(from cache)" in functai.phistory(1)


def test_transient_errors_are_retried(fake, monkeypatch):
    import lm15
    monkeypatch.setattr("time.sleep", lambda s: None)

    @ai
    def f(x: str) -> str: ...
    r = fake(lm15.RateLimitError("slow down"), "<result>\nok\n</result>")
    assert f("x") == "ok" and len(r.requests) == 2


def test_phistory_shows_the_exchange(fake):
    @ai
    def f(x: str) -> str:
        """Echo."""
    fake("<result>\nhello\n</result>")
    f("hi")
    text = functai.phistory()
    assert "System message:" in text and "User message:" in text and "hello" in text
    assert "f → gpt-4.1-mini" in text


def test_render_shows_the_request_without_sending(fake):
    @ai
    def f(x: str) -> str:
        """Echo."""
    r = fake()
    req = f.render("hi")
    assert "<x>\nhi\n</x>" in req.messages[0].parts[0].text and not r.requests
