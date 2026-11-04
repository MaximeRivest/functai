"""Studio's own AI helpers, written with FunctAI: drafting a pipeline's plan from a sentence,
and writing example inputs. They write data, never code, and never right answers: a plan
is checked by the person and turned into code by ``newpipe``; examples are inputs only,
judged later by people.

They use the model Studio would use for a new step (the first small model of a connected
account), and are not recorded in any pipeline's call log: they are Studio's, not yours.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Literal, Optional

import functai
from functai import ai

from .project import StudioError


@dataclasses.dataclass
class InputField:
    name: str            # a short lowercase name, like message or notes
    about: str           # what it is, in a few words


@dataclasses.dataclass
class Fact:
    name: str            # a short lowercase name, like faq or price_list
    about: str           # what the person must provide, in a few words, like "the club's FAQ"


@dataclasses.dataclass
class StepPlan:
    name: str                                     # a short lowercase verb or noun, like sort, answer, reply
    instruction: str                              # what the step is asked, in plain words, as you would ask a careful assistant
    takes: List[str]                              # names of the pipeline's inputs, facts or earlier steps it reads
    answer: Literal["text", "choices", "number", "whole number", "yes/no", "list of texts"]
    choices: List[str]                            # only when answer is choices: the few answers allowed


@dataclasses.dataclass
class Plan:
    name: str                       # a short lowercase name for the pipeline, words joined by _
    description: str                # one sentence: what it does, for whom
    inputs: List[InputField]        # what each use gives it, usually one
    facts: List[Fact]               # facts the person has and the AI cannot know (an FAQ, prices, rules); empty if none
    steps: List[StepPlan]           # in order; the last step's answer is the result
    examples: List[Dict[str, str]]  # 8 varied, realistic inputs (typical, hard, odd), keyed by input name; never answers


@ai
def plan_pipeline(request: str) -> Plan:
    """Plan a small AI pipeline that does what the person asks, as a few clear steps.

    Prefer 2 or 3 steps, each doing one thing a careful person could check: sort or classify
    with choices, extract, decide, then write. Never invent facts the person has and you do
    not (their FAQ, prices, rules, names): list them as facts, and steps that need them take
    them. Example inputs must be realistic and varied, and must never include right answers."""


@ai
def write_examples(pipeline: str, inputs: List[InputField], already: List[str], count: int) -> List[Dict[str, str]]:
    """Write `count` new, varied, realistic example inputs for this pipeline, keyed by input name.

    Mix typical cases, hard ones, and odd ones (very short, very long, off topic, another
    language). Never repeat the ones already there. Inputs only: never the right answers."""


def _model(project: Any = None) -> str:
    from .connect import default_model
    m = project.default_lm() if project is not None else default_model(None)
    if not m:
        raise StudioError("connect an AI account first (Accounts): the helper needs a model to write with")
    return m


def draft_plan(request: str, project: Any = None) -> Dict[str, Any]:
    if not (request or "").strip():
        raise StudioError("say what you want it to do, in a sentence or two")
    lm = _model(project)
    with functai.configure(lm=lm, log_calls=False):
        p = plan_pipeline(request)
    out = dataclasses.asdict(p)
    out["model"] = lm
    return out


def draft_examples(project: Any, count: int = 10) -> List[Dict[str, str]]:
    program = project.load()
    iface = program.interface
    fields = [InputField(f["name"], f.get("desc") or f["name"]) for f in iface.get("inputs", [])]
    have = [", ".join(f"{k}: {v}" for k, v in e["inputs"].items())[:200] for e in project.examples()][:40]
    about = (iface.get("description") or project.name) + " Steps: " + "; ".join(
        f"{n}: {(getattr(s, '__doc__', '') or '').strip().splitlines()[0] if getattr(s, '__doc__', '') else ''}"
        for n, s in project.steps().items())
    with functai.configure(lm=_model(project), log_calls=False):
        rows = write_examples(about, fields, have, max(1, min(50, int(count))))
    names = [f.name for f in fields]
    out = []
    for r in rows or []:
        if isinstance(r, dict):
            row = {n: str(r.get(n, "")) for n in names}
            if any(v.strip() for v in row.values()):
                out.append(row)
    return out


def estimate_cents(project: Any = None, *, kind: str = "plan") -> Optional[float]:
    """About what one helper call costs (cents), from the model's list price."""
    from .data import price_of
    try:
        p = price_of(_model(project))
    except StudioError:
        return None
    if p is None:
        return None
    tin, tout = (900, 1200) if kind == "plan" else (700, 900)
    return (tin * p["input"] + tout * p["output"]) / 1e6 * 100
