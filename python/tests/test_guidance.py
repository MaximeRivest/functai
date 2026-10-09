# ruff: noqa: F821, F841  (JSON-Schema annotations hold strings linters read as names; `x = _ai` declares an output)
"""Words written in the code reach the model: comments on the choices of a
Literal or an Enum, after a closing bracket, on the fields of nested records,
a record's docstring, Annotated text (functai/guidance.py). Each test reads
the instruction a function would send; nothing is sent."""
import enum
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

import functai
from functai import _ai, ai

functai.configure(lm="gpt-4.1-mini")


def instruction(fn) -> str:
    return fn._spec().signature.instructions


# ---------------------------------------------------------------- records


@dataclass
class Analysis:
    """One statistical test whose result the paper reports."""
    method: Literal[
        "t_test",      # comparing two groups
        "anova",       # three or more groups
        "other",
    ]  # the test, as the codebook names it
    pseudoreplicated: bool  # repeated measurements of one unit treated as independent


@dataclass
class Claim:
    relation: Literal["effect",        # one thing changes another
                      "description"]   # something has a value,
                                       # a composition, a distribution
    scope: Literal[
        "sample",
        "general"]  # how far the claim reaches


@ai
def claim(text: str) -> Claim:
    """The claim."""
    ...


def test_on_the_closing_line_a_comment_is_the_last_choices_when_every_choice_has_one():
    text = instruction(claim)
    assert "- Claim.relation: one of:\n  - effect: one thing changes another\n  - description: something has a value, a composition, a distribution" in text
    assert "- Claim.scope: how far the claim reaches" in text and "  - general:" not in text


@dataclass
class Study:
    setting: Literal["field", "lab"]  # field: outdoors; lab: indoors
    units: Annotated[Optional[int], "independent units per group; None when not stated"]
    analyses: list[Analysis]
    best: Optional[Analysis]


@ai
def read_study(methods: str) -> Study:
    """Fill in the form."""
    ...


def test_nested_records_their_choices_and_docstrings_reach_the_model():
    text = instruction(read_study)
    assert "Study fields:\n- Study.setting: field: outdoors; lab: indoors\n" in text
    assert "- Study.units: independent units per group; None when not stated" in text
    assert "Analysis: One statistical test whose result the paper reports." in text
    assert ("Analysis fields:\n"
            "- Analysis.method: the test, as the codebook names it\n"
            "  - t_test: comparing two groups\n"
            "  - anova: three or more groups\n"
            "- Analysis.pseudoreplicated: repeated measurements of one unit treated as independent") in text
    assert text.count("Analysis fields:") == 1                     # list[Analysis] and Optional[Analysis]: once


class Setting(enum.Enum):
    """Where the study took place."""
    FIELD = "field"   # outdoors, in nature
    LAB = "lab"       # indoors, controlled


@dataclass
class Place:
    setting: Setting
    site: str


@ai
def place(methods: str) -> Place:
    """Where was it?"""
    ...


def test_an_enums_members_and_docstring_describe_its_choices():
    assert ("- Place.setting: Where the study took place.\n"
            "  - field: outdoors, in nature\n"
            "  - lab: indoors, controlled") in instruction(place)


@dataclass
class Paper:
    """A paper as the corpus holds it."""
    title: str
    year: int  # the year of publication, not of the data


@ai
def summarize(paper: Paper) -> str:
    """One sentence about the paper."""
    ...


def test_records_a_function_reads_are_described_too():
    text = instruction(summarize)
    assert "Paper: A paper as the corpus holds it.\nPaper fields:\n- Paper.year: the year of publication" in text


class Obs(BaseModel):
    """One value."""
    value: str = Field(description="as printed")
    unit: str  # as printed; empty when none


class Rows(BaseModel):
    observations: list[Obs]


@ai
def rows(text: str) -> Rows:
    """Every observation."""
    ...


def test_a_pydantic_models_described_fields_are_left_to_its_schema():
    text = instruction(rows)
    assert "- Obs.unit: as printed; empty when none" in text       # a comment, which the schema lacks
    assert "Obs.value" not in text and "Obs: One value." not in text   # in the JSON schema already


# ---------------------------------------------------------------- parameters, outputs, the answer


@ai
def route(
    message: str,  # the customer's words
    channel: Literal[
        "email",   # written at leisure, often long
        "chat",    # short, often several messages
    ],
) -> Literal[
    "shipping",    # where the parcel is, damage in transit
    "billing",     # charges, refunds
]:
    """Which team answers?"""
    urgency: Literal[
        "low",
        "high",    # the customer is losing money or time now
    ] = _ai  # how soon to answer
    return _ai


def test_choices_of_parameters_outputs_and_the_answer_are_described():
    text = instruction(route)
    assert ("- channel: one of:\n  - email: written at leisure, often long\n  - chat: short, often several messages") in text
    assert "- message: the customer's words" in text
    assert "- urgency: how soon to answer\n  - high: the customer is losing money or time now" in text
    assert "- result: one of:\n  - shipping: where the parcel is, damage in transit\n  - billing: charges, refunds" in text


def test_described_choices_are_written_as_every_language_writes_them():
    # contract/functions.md, "A choice's answers, described one by one": route is case 18's definition, written in
    # Python's way (R's choice(), TypeScript's t.choice and Julia's OneOf give it the same version)
    case = json.loads((Path(__file__).resolve().parents[2] / "contract" / "cases" / "functions"
                       / "18-choices-described-one-by-one.json").read_text())
    assert route.using(lm="gpt-4.1-mini").version == case["expect"]["version"]


# ---------------------------------------------------------------- what stays the same


@dataclass
class Design:
    n_units: int | None  # how many independent units; None when not stated
    quotes: list[str]  # pieces copied exactly from the paper


@ai
def design(methods: str) -> Design:
    """Describe how the study was designed. Never guess."""
    note: str = _ai  # one sentence on anything unusual
    return _ai


def test_a_prompt_that_was_already_complete_is_unchanged():
    assert instruction(design) == (
        "Function: design\n\nDescribe how the study was designed. Never guess.\n\n"
        "Output guidance:\n- note: one sentence on anything unusual\n\n"
        "Design fields:\n- Design.n_units: how many independent units; None when not stated\n"
        "- Design.quotes: pieces copied exactly from the paper")


# ---------------------------------------------------------------- no source, no comments


def test_a_function_defined_without_a_source_says_its_comments_are_lost():
    code = textwrap.dedent('''
        import inspect, warnings, functai
        from functai import ai
        functai.configure(lm="gpt-4.1-mini")
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            @ai
            def f(x: str) -> str:  # a comment Python may not keep
                """Echo."""
                ...
            f._spec(); f.using(temperature=0)._spec()
        try:
            kept = bool(inspect.getsource(f._fn))
        except OSError:
            kept = False
        print(int(kept), sum("Python keeps no source" in str(w.message) for w in seen))
    ''')
    run = lambda *how, **kw: subprocess.run([sys.executable, *how], capture_output=True, text=True, check=True,
                                            **kw).stdout.split()
    assert run("-", input=code) == ["0", "1"]                                    # standard input: lost, said once
    assert run("-c", f"exec({code!r})") == ["0", "1"]                            # exec: lost, said once
    kept, said = run("-c", code)                                                 # -c: kept from Python 3.13 on
    assert said == ("0" if kept == "1" else "1")


@ai
def first(x: str,  # the words, on the def line
          y: int) -> str:
    """Echo."""
    ...


def test_a_comment_after_the_first_parameter_on_the_def_line_reaches_the_model():
    assert "- x: the words, on the def line" in instruction(first)


# ---------------------------------------------------------------- shapes written as JSON Schema


@ai
def colour(picture: {"media": "image"},
           size: {"type": "string", "enum": ["small", "large"]}) -> str:
    """The main colour."""
    ...


def test_a_shape_written_as_json_schema_is_kept_and_a_picture_is_sent_as_one():
    shapes = {f["name"]: f["shape"] for f in colour.interface["inputs"]}
    assert shapes == {"picture": {"media": "image"}, "size": {"type": "string", "enum": ["small", "large"]}}
    request = colour.render({"media_type": "image/png", "data": "iVBORw0KGgo="}, "small")
    parts = [p for m in request.messages for p in m.parts]
    assert [p.type for p in parts] == ["text", "image", "text"]          # an image part, not base64 in the text
    assert "iVBORw0KGgo" not in "".join(p.text for p in parts if p.type == "text")


# ---------------------------------------------------------------- the skill for coding agents


def test_the_skill_ships_in_the_package_and_installs_where_an_agent_looks(tmp_path, capsys):
    from functai.__main__ import SKILL, main
    assert (SKILL / "SKILL.md").read_text().startswith("---\nname: functai\n")
    assert main(["skill", "--install", str(tmp_path / "skills")]) == 0
    assert sorted(p.name for p in (tmp_path / "skills" / "functai").iterdir()) == sorted(
        p.name for p in SKILL.iterdir() if p.name != "__pycache__")
    (tmp_path / "other" / "functai").mkdir(parents=True)
    (tmp_path / "other" / "functai" / "SKILL.md").write_text("---\nname: someone-else\n---\n")
    assert main(["skill", "--install", str(tmp_path / "other")]) == 0
    assert "someone-else" in (tmp_path / "other" / "functai" / "SKILL.md").read_text()   # not ours: left alone
    assert "holds something else" in capsys.readouterr().err


@dataclass
class Price:
    amount: float  # in dollars  # type: ignore[assignment]
    currency: str  # noqa: E501


@ai
def price(text: str,  # the listing  # noqa: E501
          ) -> Price:  # pyright: ignore
    """The price."""
    ...


def test_tool_pragmas_are_not_words_for_the_model():
    text = instruction(price)
    assert text.endswith("Price fields:\n- Price.amount: in dollars") and "- text: the listing" in text
    assert "noqa" not in text and "ignore" not in text and "Price.currency" not in text
