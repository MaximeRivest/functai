"""The contract's setup/ cases (contract/calls.md, *Setups*): an AI
function's setup document and its hash, under settings, plugins and walls."""

import pytest

import functai
from contract_support import case_files, load, python_function

CASES = case_files("setup")


def builtin(spec):
    make = {"limits": functai.limits, "keep_pictures": functai.keep_pictures}[spec["builtin"]]
    return make(**spec["args"])


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_setup_case(path):
    case = load(path)
    fn = python_function(case["definition"])
    settings = dict(case["settings"])
    if case["plugins"]:
        settings["plugins"] = [builtin(p) for p in case["plugins"]]
    with functai.configure(**settings):
        assert fn.setup_document() == case["expect"]["document"]
        assert fn.setup == case["expect"]["setup"]


def test_the_contract_has_setup_cases():
    assert len(CASES) >= 12
