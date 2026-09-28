"""contract/cases/programs: a program's interface, its signature, which
interfaces are refused, and how a module's calls are checked against it
(contract/programs.md). Every kind of case runs through the real surface: an
AI function defined from Python code, a module declared with ``interface=``
and called, its record read back from the call log."""

import json

import pytest

import functai
from functai import calllog, interface, module
from contract_support import as_json, case_files, load, native, python_function, validator, assert_valid, \
    without_type

CASES = case_files("programs")
INTERFACE = validator("interface")


class _Got(Exception):
    """The code ran and saw its inputs: the check of the call's inputs is over."""


def declared(iface):
    """A module declaring ``iface``, whose code keeps what it got (its inputs)."""
    got = {}

    def body(**inputs):
        got.clear()
        got.update(inputs)
        raise _Got()

    m = module(interface=iface)(body)
    return m, got


def refusal(err: functai.InterfaceError) -> dict:
    return {"refuses": err.code, "field": err.field}


def check_inputs(m, got, inputs):
    got.clear()
    try:
        m(**native(inputs))
    except functai.InterfaceError as err:
        assert err.code == "interface-input" and not got
        return refusal(err)
    except _Got:
        pass
    return {"inputs": as_json(dict(got))}


def check_returned(iface, returned, tmp_path):
    """What the code returned, as the call's outputs (the call log's record)."""
    value = native(returned)

    def body(**_inputs):
        return value

    m = module(interface=iface, log_calls=tmp_path)(body)
    inputs = {f["name"]: _sample(f) for f in iface["inputs"] if not f.get("optional")}
    before = set(tmp_path.rglob("*.jsonl"))
    try:
        m(**inputs)
    except functai.InterfaceError as err:
        assert err.code == "interface-output"
        return refusal(err)
    records = [json.loads(line) for f in sorted(tmp_path.rglob("*.jsonl")) for line in f.read_text().splitlines()]
    assert before is not None
    rec = records[-1]
    assert rec["error"] is None
    if rec.get("described"):
        # a value with no JSON form is written as its description, and named so
        assert set(rec["described"]["outputs"]) <= set(rec["outputs"])
    return {"outputs": rec["outputs"]}


def _sample(field):
    shape = field["shape"]
    t = shape.get("type")
    return {"string": "x", "integer": 1, "number": 1.5, "boolean": True, "array": [], "object": {},
            "null": None}.get(t, "x")


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_programs_case(path, tmp_path):
    case = load(path)
    kind = case["program"]
    if kind == "ai":
        d = case["definition"]
        if "refuses" in case["expect"]:
            with pytest.raises(functai.InterfaceError) as err:
                python_function(d)
            assert refusal(err.value) == case["expect"]
            return
        fn = python_function(d)
        iface = fn.interface
        assert_valid(INTERFACE, iface, path.stem)
        assert without_type(iface) == case["expect"]["interface"]
        assert interface.signature(iface) == case["expect"]["signature"]
        assert calllog.signature_id(fn.signature) == case["expect"]["signature_id"]
        for b in case["binds"]:
            assert fn._bind_inputs((), dict(b["inputs"])) == b["expect"]["inputs"]
    elif kind == "module":
        m, got = declared(case["interface"])
        assert interface.signature(m.interface) == case["expect"]["signature"]
        assert m.interface == case["interface"]
        for check in case["checks"]:
            if "inputs" in check:
                assert check_inputs(m, got, check["inputs"]) == check["expect"], check
            else:
                assert check_returned(case["interface"], check["returned"], tmp_path) == check["expect"], check
    elif kind == "definitions":
        for x in case["interfaces"]:
            iface, expect = x["interface"], x["expect"]
            if x.get("ai"):
                # an AI function's interface, as defining the function or reading its node checks it
                try:
                    interface.check(iface, ai=True)
                except functai.InterfaceError as err:
                    got = refusal(err)
                else:
                    got = {"signature": interface.signature(iface)}
            else:
                try:
                    m, _got = declared(iface)
                except functai.InterfaceError as err:
                    got = refusal(err)
                else:
                    got = {"signature": interface.signature(m.interface)}
            assert got == expect, x
    elif kind == "same-data":
        modules = [declared(iface) for iface in case["interfaces"]]
        assert [interface.signature(m.interface) for m, _ in modules] == case["expect"]["signatures"]
        for check in case["checks"]:
            assert [check_inputs(m, got, check["inputs"]) for m, got in modules] == check["expect"], check
    else:
        raise AssertionError(f"a kind of case this harness does not know: {kind}")


def test_the_contract_has_program_cases():
    assert len(CASES) >= 21
