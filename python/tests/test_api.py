"""The Python API itself: predict, async, and improvement that returns copies. Offline."""

import asyncio
from typing import Literal

import pytest

import functai
from functai import _ai, ai


@ai
def team(message: str) -> Literal["shipping", "billing"]:
    """Which team should answer this customer message?"""
    ...


def reply(req):
    text = "".join(p.text for p in req.messages[-1].parts)
    return "<result>\n" + ("billing" if "charged" in text else "shipping") + "\n</result>"


def test_predict_gives_everything_and_a_parameter_may_be_called_all(fake):
    fake(responder=reply)
    p = team.predict("I was charged twice.")
    assert p.result == "billing" and p.call_id

    @ai
    def pick(all: list[str]) -> str:                    # `all` was reserved; it is an input like any other
        """Pick one."""
        ...

    fake("<result>\nb\n</result>")
    assert pick(all=["a", "b"]) == "b"


def test_acall_and_apredict_run_in_async_code_with_the_callers_settings(fake):
    r = fake(responder=reply)

    async def main():
        with functai.configure(temperature=0.25):          # a scoped setting reaches the worker thread
            answers = await asyncio.gather(*(team.acall(m) for m in ["charged twice", "parcel late", "charged"]))
            p = await team.apredict("charged again")
        return answers, p

    answers, p = asyncio.run(main())
    assert answers == ["billing", "shipping", "billing"] and p.result == "billing"
    assert {req.config.temperature for req in r.requests} == {0.25}


def test_wrong_arguments_to_acall_fail_in_the_caller():
    with pytest.raises(TypeError):
        asyncio.run(team.acall("a", "b"))


def test_an_async_def_ai_function_is_awaited(fake):
    fake("<reasoning>\nIt is about money.\n</reasoning>\n<result>\nbilling\n</result>")

    @ai
    async def team_async(message: str) -> Literal["shipping", "billing"]:
        """Which team should answer this customer message?"""
        reasoning: str = _ai     # a line on why
        return _ai

    assert asyncio.run(team_async("charged twice")) == "billing"
    assert "reasoning" in team_async._spec().outputs


def test_an_async_def_with_code_of_its_own_is_refused():
    with pytest.raises(TypeError, match="acall"):
        @ai
        async def shout(text: str) -> str:
            """Repeat it."""
            return _ai.upper()


ROWS = [{"message": "charged twice", "category": "billing"}, {"message": "parcel late", "category": "shipping"},
        {"message": "charged for shipping", "category": "billing"}, {"message": "box crushed", "category": "shipping"}]


def test_improving_returns_a_copy_and_leaves_the_function_as_it_was(fake):
    fake(responder=reply)
    taught = functai.labeled_few_shot(team, ROWS, k=2, expected="category")
    assert len(taught.demos) == 2 and team.demos == []
    assert taught.version != team.version
    assert taught.optimization_runs()[-1]["optimizer"] == "LabeledFewShot" and team.optimization_runs() == []
    boot = functai.bootstrap_few_shot(team, ROWS, expected="category", max_bootstrapped=2, max_labeled=2)
    assert boot.demos and team.demos == []
