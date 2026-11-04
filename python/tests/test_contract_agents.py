"""The contract's cases for long agents and pipelines (contract/cases/agents),
run through this implementation."""

import pytest
from lm15.serde import part_from_dict

from functai import builtins, conversations, engine, eventlog
from functai.plugins import Spent, StepResult

from contract_support import case_files, load


def cases(kind):
    return case_files("agents", f"{kind}-")


def native(returned):
    """What a tool written in Python returns for the case's data: a str, lm15's part, or a list of them."""
    if isinstance(returned, dict):
        return part_from_dict(returned)
    if isinstance(returned, list):
        return [native(x) for x in returned]
    return returned


@pytest.mark.parametrize("path", cases("result"), ids=lambda p: p.stem)
def test_what_a_tool_returns(path):
    case = load(path)
    result = engine.result_of(native(case["returned"]))
    assert result == case["expect"]["result"]
    event = {"output": engine.result_text(result)}
    if not isinstance(result, str):
        event["parts"] = result
    assert event == case["expect"]["event"]


@pytest.mark.parametrize("path", cases("state"), ids=lambda p: p.stem)
def test_a_turn_that_waits_for_later_results(path, monkeypatch):
    case = load(path)
    monkeypatch.setattr(conversations, "_now", lambda: conversations._parse(case["now"]))
    log = conversations._Log()
    log.apply(case["records"])
    expect = case["expect"]
    assert {t: st.state() for t, st in log.turns.items()} == expect["states"]
    assert {t: [x["key"] for x in st.undelivered(log.delivered)] for t, st in log.turns.items()
            if st.undelivered(log.delivered)} == expect["later"]
    assert {t: [x["invocation"] for x in st.unfinished()] for t, st in log.turns.items() if st.unfinished()} \
        == expect["unfinished"]
    assert {k: r["output"] for k, r in log.delivered.items()} == expect["delivered"]
    resumes = {t: True if not st.unanswered() and not st.undelivered(log.delivered) else {"refuses": "turn-state"}
               for t, st in log.turns.items() if st.state() == "waiting"}
    assert resumes == expect["resumes"]


@pytest.mark.parametrize("path", cases("kept"), ids=lambda p: p.stem)
def test_the_kept_form_of_notes_laters_and_pictures(path):
    case = load(path)
    assert eventlog.kept_form(case["events"], case["kept"]) == case["expect"]["events"]


class _Event:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.parametrize("path", cases("limits"), ids=lambda p: p.stem)
def test_limits(path):
    case = load(path)
    lim = dict(case["limits"])
    plugin = builtins.limits(**lim)
    spent = Spent([tuple(x) for x in case["spent"]["exchanges"]], case["spent"]["seconds"])
    at = case["at"]
    if at["hook"] == "before_call":
        [handler] = plugin.handlers["before_call"]
        got = handler(_Event(spent=spent, lm=at["lm"]))
    else:
        [handler] = plugin.handlers["step"]
        got = handler(_Event(spent=spent, step=at["step"], max_steps=at["max_steps"], tools=at["tools"],
                             last_step=at["step"] >= at["max_steps"]))
    assert (got.fields() if got is not None else None) == case["expect"]["change"]


@pytest.mark.parametrize("path", cases("spent"), ids=lambda p: p.stem)
def test_what_a_tree_spent(path):
    case = load(path)
    s = Spent([tuple(x) for x in case["exchanges"]], 0.0)
    assert s.usage == case["expect"]["usage"] and s.tokens == case["expect"]["tokens"]
    cost = s.cost(case["prices"])
    want = case["expect"]["cost"]
    assert (cost is None) == (want is None) and (want is None or abs(cost - want) < 1e-12)


@pytest.mark.parametrize("path", cases("pictures"), ids=lambda p: p.stem)
def test_keep_pictures(path):
    case = load(path)
    results = [StepResult(r["invocation"], "look", {}, r["output"]) for r in case["results"]]
    got = builtins.pictures_to_drop(results, last=case["last"], at_once=case["at_once"])
    assert {str(k): v for k, v in got.items()} == case["expect"]["replace"]
