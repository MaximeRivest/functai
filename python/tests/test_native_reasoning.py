"""A reasoning output on a model that thinks natively (contract/functions.md, *Reasoning*): the request asks
for the thinking and its summary, and the function's own `reasoning` setting chooses the effort."""

import lm15
import pytest

import functai
from functai import ai, models

XML = "<result>\n{}\n</result>"


@ai(module="cot")
def solve(puzzle: str) -> int:
    """Solve the puzzle."""


@pytest.fixture
def thinker(fake):
    return fake(responder=lambda req: [lm15.ThinkingPart("Twelve pens leave 14 for two notebooks."), lm15.TextPart(XML.format(12))],
                provider="anthropic", lm="anthropic:claude-opus-5-5")


def sent(router):
    return router.requests[-1].config.reasoning


def test_claude_5_thinks_natively():
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-fable-5-1",
                  "claude-mythos-5", "claude-opus-4-8", "claude-sonnet-4-6"):
        caps = models.capabilities("anthropic", model)
        assert caps["native_reasoning"] and not caps["assistant_prefill"], model


def test_the_thinking_and_its_summary_are_asked_for(thinker):
    p = solve.predict(puzzle="x")
    assert p.reasoning == "Twelve pens leave 14 for two notebooks." and p.result == 12
    assert sent(thinker) == lm15.Reasoning(effort="medium", summary="auto")


@pytest.mark.parametrize("setting, wanted", [
    ("high", lm15.Reasoning(effort="high", summary="auto")),
    ({"effort": "low"}, lm15.Reasoning(effort="low", summary="auto")),
    (lm15.Reasoning(effort="low", summary="detailed"), lm15.Reasoning(effort="low", summary="detailed")),
    (lm15.Reasoning(effort="high", thinking_budget=4096), lm15.Reasoning(effort="high", thinking_budget=4096, summary="auto")),
    ("off", lm15.Reasoning(effort="off")),
])
def test_the_functions_own_reasoning_setting_chooses_the_effort(thinker, setting, wanted):
    solve.using(reasoning=setting).predict(puzzle="x")              # never a conflict with the layout's medium
    assert sent(thinker) == wanted
    assert solve.using(reasoning=setting).render(puzzle="x").config.reasoning == wanted   # what render shows is sent


def test_reasoning_shorthands_are_checked():
    assert functai.config.check({"reasoning": "low"}, "t")["reasoning"] == lm15.Reasoning(effort="low")
    with pytest.raises(TypeError, match="effort, thinking_budget and summary"):
        functai.config.check({"reasoning": {"effort": "low", "budget": 9}}, "t")


def test_a_model_that_writes_its_reasoning_is_not_asked_for_thinking(fake):
    r = fake(XML.format(12).replace("<result>", "<reasoning>\nBecause.\n</reasoning>\n<result>"), lm="gpt-4.1-mini")
    p = solve.using(reasoning="high").predict(puzzle="x")
    assert p.reasoning == "Because." and sent(r) == lm15.Reasoning(effort="high")    # the setting goes as given
