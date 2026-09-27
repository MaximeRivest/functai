"""What a type checker must see in functai's API. Not run: tests/test_types.py
checks it with basedpyright. A `# pyright: ignore[...]` line is an error the
checker must report (an unneeded ignore fails the check); `assert_type` pins a type."""

from typing import Literal, assert_type

import functai
from dpyr import col
from functai import Prediction, _ai, ai


@ai
def team(message: str, urgent: bool = False) -> Literal["shipping", "billing"]:
    """Which team should answer this customer message?"""
    ...


@ai
def solve(problem: str) -> float:
    """Solve the word problem."""
    reasoning: str = _ai     # step by step
    return _ai


@ai(lm="gpt-6-luna", temperature=0)
def headline(article: str) -> str:
    """A headline of at most eight words."""
    ...


@ai
async def team_async(message: str) -> Literal["shipping", "billing"]:
    """Which team should answer this customer message?"""
    ...


def calls() -> None:
    assert_type(team("I was charged twice."), Literal["shipping", "billing"])
    assert_type(team(message="x", urgent=True), Literal["shipping", "billing"])
    assert_type(solve("2 + 2"), float)
    assert_type(headline("..."), str)
    assert_type(team.predict("x"), Prediction)
    assert_type(team.using(lm="gpt-5.4-nano")("x"), Literal["shipping", "billing"])    # a copy keeps the types

    team(123)                          # pyright: ignore[reportCallIssue, reportArgumentType]
    team("a", "b", "c")                # pyright: ignore[reportArgumentType]
    team(mesage="typo")                # pyright: ignore[reportArgumentType]
    team.predict(123)                  # pyright: ignore[reportArgumentType]
    team.nonsense                      # pyright: ignore[reportAttributeAccessIssue]


def columns() -> None:
    team(col.message)                  # a column expression: one call per row
    team(message=col.message)


async def in_async_code() -> None:
    assert_type(await team.acall("x"), Literal["shipping", "billing"])
    assert_type(await team.apredict("x"), Prediction)
    assert_type(await team_async("x"), Literal["shipping", "billing"])
    await team.acall(1)                # pyright: ignore[reportArgumentType]


def improving(rows: list[dict[str, str]]) -> None:
    assert_type(functai.labeled_few_shot(team, rows, k=4)("x"), Literal["shipping", "billing"])   # improved copies keep the types
    assert_type(functai.gepa(team, rows, teacher="gpt-6-sol").predict("x"), Prediction)
    assert_type(team.opt(rows)("x", urgent=True), Literal["shipping", "billing"])
    functai.gepa(team, rows)(123)      # pyright: ignore[reportCallIssue, reportArgumentType]
