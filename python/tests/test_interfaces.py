"""Programs' interfaces as a Python programmer meets them (contract/programs.md):
derived from the function, checked on every call of a module, written in the
call log and in a saved folder."""

import json
import re
import textwrap
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional

import pytest
from pydantic import BaseModel, Field

import functai
from functai import _ai, ai, interface, module
from contract_support import assert_valid, validator

INTERFACE = validator("interface")
CALL = validator("call")


def records(folder):
    return [json.loads(line) for f in sorted(folder.rglob("*.jsonl")) for line in f.read_text().splitlines()]


class Frame:
    """A value with no JSON form (a data frame, say)."""

    def __repr__(self):
        return "<Frame 2x2>"


@dataclass
class Tracking:
    where: str
    late_days: int


def test_a_module_s_interface_is_derived_from_its_function():
    @module
    def pipeline(frame, text: str, options: functai.JSON, anything: Any, *extra: int, when: Optional[str] = None,
                 tone: str = "kind", **more) -> Tracking:
        """Where a parcel is."""
        return Tracking("Leeds", 7)

    iface = pipeline.interface
    assert_valid(INTERFACE, iface)
    by = {f["name"]: f for f in iface["inputs"]}
    assert by["frame"] == {"name": "frame", "shape": {}, "opaque": True}          # unannotated: any value
    assert by["anything"] == {"name": "anything", "shape": {}, "opaque": True}
    assert by["options"] == {"name": "options", "shape": {}, "type": "JSON"}      # any JSON value, checked as data
    assert by["text"]["shape"] == {"type": "string"} and "optional" not in by["text"]
    assert by["extra"] == {"name": "extra", "shape": {"type": "array", "items": {"type": "integer"}, "default": []},
                           "optional": True}
    assert by["more"] == {"name": "more", "shape": {"type": "object", "default": {}}, "optional": True}
    assert by["when"]["optional"] is True and by["when"]["shape"]["default"] is None     # None: nullable
    assert by["tone"]["shape"] == {"type": "string", "default": "kind"}
    [out] = iface["outputs"]
    assert out["name"] == "result" and out["shape"]["required"] == ["where", "late_days"]
    assert iface["description"] == "Where a parcel is."
    assert pipeline(Frame(), "x", {"a": [1]}, object(), 1, 2) == Tracking("Leeds", 7)


def test_a_module_checks_its_inputs_and_outputs_and_the_log_says_so(fake, tmp_path):
    @module(log_calls=tmp_path)
    def triage(ticket: str, minutes: int = 5) -> Literal["billing", "shipping"]:
        return "billing" if "charged" in ticket else "legal"

    assert triage("charged twice") == "billing"
    with pytest.raises(functai.InterfaceError) as err:
        triage(ticket=3)
    assert (err.value.code, err.value.field) == ("interface-input", "ticket")
    with pytest.raises(functai.InterfaceError) as err:
        triage("where is it?")
    assert (err.value.code, err.value.field) == ("interface-output", "result")
    with pytest.raises(functai.InterfaceError) as err:
        triage("x", minutes=2.5)                     # an integer is a number with no fraction
    assert err.value.field == "minutes"
    triage("charged", minutes=5.0)                   # 5.0 is one
    ok, bad_in, bad_out, _frac, _whole = records(tmp_path)
    for rec in (ok, bad_in, bad_out):
        assert_valid(CALL, rec)
    assert ok["inputs"] == {"ticket": "charged twice", "minutes": 5}           # named, the default recorded
    assert ok["outputs"] == {"result": "billing"}
    assert ok["program"]["interface"] == interface.signature(triage.interface)
    assert "signature" not in ok["program"]
    assert bad_in["error"] == {"type": "InterfaceError", "code": "interface-input",
                               "message": bad_in["error"]["message"]} and bad_in["outputs"] is None
    assert bad_out["error"]["code"] == "interface-output" and bad_out["outputs"] is None


def test_several_outputs_are_declared_and_returned_by_name():
    @module(outputs={"team": Literal["billing", "shipping"], "minutes": Annotated[int, "to fix"], "result": str})
    def triage(ticket: str):
        return {"team": "billing", "minutes": 5, "result": "Refunded."}

    iface = triage.interface
    assert [f["name"] for f in iface["outputs"]] == ["team", "minutes", "result"]
    assert iface["outputs"][1]["desc"] == "to fix"
    assert triage("x")["result"] == "Refunded."

    @module(outputs={"team": str, "result": str})
    def sloppy(ticket: str):
        return {"team": "billing", "result": "ok", "note": "not an output"}

    with pytest.raises(functai.InterfaceError, match="'note'") as err:
        sloppy("x")
    assert err.value.field == "note"


def test_a_declared_interface_is_the_whole_interface():
    iface = {"description": "Answer a customer's message.",
             "inputs": [{"name": "message", "shape": {"type": "string"}},
                        {"name": "tone", "shape": {"type": "string", "default": "kind"}, "optional": True}],
             "outputs": [{"name": "result", "shape": {"type": "string"}}]}

    @module(interface=iface)
    def support(message, tone="curt"):         # declared: the interface's default applies, not the code's
        return f"{tone}: {message}"

    assert support.interface == iface
    assert support("Hi") == "kind: Hi"                 # the code has no default: the interface's is passed
    assert support("Hi", tone="brief") == "brief: Hi"
    with pytest.raises(TypeError, match="does not take"):
        module(interface=iface)(lambda message: message)
    with pytest.raises(TypeError, match="needs 'order'"):
        module(interface=iface)(lambda message, tone, order: message)


def test_interfaces_every_language_refuses_are_refused_when_defined():
    class Order(BaseModel):
        code: str = Field(pattern="^B-[0-9]+$")

    with pytest.raises(functai.InterfaceError, match="pattern") as err:
        @module
        def track(order: Order) -> str:
            return "Leeds"
    assert (err.value.code, err.value.field) == ("interface-malformed", "order")

    @ai                                                # an AI function's shapes are lmcc's: pattern is carried
    def track_ai(order: Order) -> str:
        """Where is the order?"""
    assert "pattern" in json.dumps(track_ai.interface)

    with pytest.raises(functai.InterfaceError) as err:
        @ai
        def count(items: list[str], at_least: Annotated[int, Field(ge=10)] = 5) -> int:
            """Count the items worth keeping."""
    assert (err.value.code, err.value.field) == ("interface-malformed", "at_least")

    with pytest.raises(functai.InterfaceError, match="default") as err:
        @ai
        def since(day: str, start: str = Frame()) -> str:     # a model is sent every input: no JSON default
            """Plan."""
    assert err.value.field == "start"


def test_a_name_defined_later_is_looked_up_at_the_first_call():
    @module
    def later(order: "Late") -> str:                                      # noqa: F821
        return order.where

    global Late
    Late = Tracking
    try:
        assert later(Tracking("Leeds", 1)) == "Leeds"
        assert later.interface["inputs"][0]["shape"]["required"] == ["where", "late_days"]
    finally:
        del Late


def test_an_ai_function_s_interface_and_its_optional_inputs(fake):
    @ai
    def reply(
        message: str,  # what the customer wrote
        tone: str = "kind",
        order: Optional[str] = None,
    ) -> str:
        """Answer the customer."""
        summary: str = _ai["one sentence"]
        return _ai

    iface = reply.interface
    assert_valid(INTERFACE, iface)
    assert iface["description"] == "Answer the customer."
    assert iface["inputs"][0]["desc"] == "what the customer wrote"
    assert iface["inputs"][1] == {"name": "tone", "shape": {"type": "string", "default": "kind"}, "type": "str",
                                  "optional": True}
    assert iface["inputs"][2]["shape"]["default"] is None
    assert [f["name"] for f in iface["outputs"]] == ["summary", "result"] and iface["outputs"][0]["desc"] == \
        "one sentence"
    # the signature leaves the default out: it is the program.signature when there is neither reasoning nor tools
    from functai import calllog
    assert interface.signature(iface) == calllog.signature_id(reply.signature)
    assert interface.signature(reply.using(module="cot").interface) == interface.signature(iface)


def test_a_module_s_version_includes_its_interface():
    def make(ann):
        ns = {"module": module, "__name__": "versions"}
        exec(textwrap.dedent(f"""
            @module
            def triage(ticket: {ann}) -> str:
                return "billing"
            """), ns)
        return ns["triage"]
    assert make("str").version != make("int").version


def test_save_writes_every_program_s_interface_and_describe_reads_it(tmp_path, monkeypatch):
    src = tmp_path / "shop_desk.py"
    src.write_text(textwrap.dedent('''
        from typing import Literal
        from functai import ai, module

        @ai(lm="gpt-4.1-mini")
        def mood(review: str, tone: str = "plain") -> Literal["happy", "unhappy"]:
            """How does the customer feel?"""

        @module(interface={"description": "Sort a review.", "inputs": [{"name": "review", "shape": {"type": "string"}}],
                           "outputs": [{"name": "result", "shape": {"type": "string"}}]}, log_content={"review": False})
        def sort(review):
            return mood(review)
        '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    import shop_desk as shop
    functai.save(shop.sort, tmp_path / "sorted")
    manifest = json.loads((tmp_path / "sorted" / "functai.json").read_text())
    assert manifest["nodes"]["shop_desk:sort"]["interface"] == shop.sort.interface
    assert manifest["nodes"]["shop_desk:mood"]["interface"] == shop.mood.interface
    assert functai.describe(tmp_path / "sorted") == shop.sort.interface
    assert functai.describe(tmp_path / "sorted", node="shop_desk:mood")["inputs"][1]["optional"] is True
    loaded = functai.load(tmp_path / "sorted", trust=True)
    assert loaded.interface == shop.sort.interface and loaded._declared
    assert loaded._settings["log_content"] == {"review": False}


def test_log_content_names_are_checked_when_the_program_is_defined():
    with pytest.raises(functai.LogContentError) as err:
        @ai(log_content={"transcrpit": False})
        def summarize(transcript: str) -> str:
            """Summarize."""
    assert err.value.field == "transcrpit" and err.value.code == "log-content-field"
    with pytest.raises(functai.LogContentError, match="tools is the function's state"):
        @ai(log_content={"tools": False}, tools=[lambda: "x"])
        def helper(question: str) -> str:
            """Help."""
    with pytest.raises(functai.LogContentError):
        @module(log_content={"messages": False})
        def support(message: str) -> str:
            return message
    assert re.match(r"^[A-Za-z_]", "ok")
