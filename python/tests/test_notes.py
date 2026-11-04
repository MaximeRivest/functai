"""Notes a program declares (contract/programs.md, *Notes*): declared by a
module or by an AI function's tools, checked where they are made, shown in
the outside view when declared ``outside``, and never in the interface's
signature."""

from dataclasses import dataclass
from typing import Literal

import lm15
import pytest

import functai
from conftest import FakeRouter
from functai import interface, views

XML = "<result>\n{}\n</result>"


@dataclass
class Page:
    url: str
    title: str


@functai.tool(effects="reads", notes={"page captured": functai.Note(Page, outside=True)})
def screenshot(url: str) -> str:
    """Take a screenshot of a page."""
    functai.note("page captured", Page(url, "Home"))
    return f"captured {url}"


@functai.ai(tools=[screenshot])
def walker(url: str) -> str:
    """Walk the site and say what is wrong with it."""
    ...


@functai.module(notes={"stage": functai.Note(Literal["walking", "writing"], latest=True, outside=True,
                                             desc="where the walk is"),
                       "cost so far": float})
def study(url: str) -> str:
    functai.note("stage", "walking")
    found = walker(url)
    functai.note("cost so far", 0.01)
    functai.note("stage", "writing")
    return found


def test_an_ai_function_declares_its_tools_notes():
    assert walker.interface["notes"] == [{"name": "page captured", "outside": True, "shape": {
        "type": "object", "properties": {"url": {"type": "string"}, "title": {"type": "string"}},
        "required": ["url", "title"]}}]
    # notes are not in the signature: the same function without them has the same one
    plain = {k: v for k, v in walker.interface.items() if k != "notes"}
    assert interface.signature(walker.interface) == interface.signature(plain)


def test_a_module_declares_its_notes_in_order():
    notes = study.interface["notes"]
    assert [n["name"] for n in notes] == ["stage", "cost so far"]
    assert notes[0] == {"name": "stage", "shape": {"enum": ["walking", "writing"], "type": "string"},
                        "desc": "where the walk is", "latest": True, "outside": True}
    assert notes[1] == {"name": "cost so far", "shape": {"type": "number"}}


def test_two_tools_that_declare_one_note_differently_are_refused():
    @functai.tool(notes={"page captured": str})
    def other(url: str) -> str:
        """Another."""
        return ""
    with pytest.raises(functai.InterfaceError) as err:
        @functai.ai(tools=[screenshot, other])
        def both(url: str) -> str:
            """Both."""
            ...
    assert err.value.code == "interface-malformed" and err.value.field == "page captured"
    with pytest.raises(functai.InterfaceError):
        walker.tools = [screenshot, other]
    assert walker.tools == [screenshot]                    # refused before anything changed


def test_a_note_with_no_json_form_cannot_be_declared():
    with pytest.raises(functai.InterfaceError) as err:
        functai.module(notes={"frame": object})(lambda x: x)
    assert err.value.code == "interface-malformed" and err.value.field == "frame"
    with pytest.raises(TypeError):
        functai.module(notes={"stage": str}, interface={"description": "", "inputs": [], "outputs": [
            {"name": "result", "shape": {}}]})(lambda: None)


def test_a_note_is_checked_where_it_is_made():
    @functai.module(notes={"stage": Literal["walking", "writing"]})
    def renamed(url: str) -> str:
        functai.note("stage changed", "walking")
        return "x"

    @functai.module(notes={"stage": Literal["walking", "writing"]})
    def wrong(url: str) -> str:
        functai.note("stage", "resting")
        return "x"

    @functai.module
    def free(url: str) -> str:
        functai.note("anything at all", {"x": 1})
        return "x"
    with pytest.raises(functai.InterfaceError) as err:
        renamed("u")
    assert (err.value.code, err.value.field) == ("interface-note", "stage changed")
    with pytest.raises(functai.InterfaceError) as err:
        wrong("u")
    assert (err.value.code, err.value.field) == ("interface-note", "stage")
    assert free("u") == "x"                               # a program that declares none refuses none
    functai.note("stage", "outside any call: nothing")    # outside any call, nothing is checked


def test_the_outside_view_shows_only_the_notes_declared_outside(tmp_path):
    def reply(request):
        done = [p for m in request.messages for p in m.parts if type(p).__name__ == "ToolResultPart"]
        if not done and request.tools:
            return [lm15.ToolCallPart(id="c1", name="screenshot", input={"url": "https://a.example/"})]
        return XML.format("The fees appear late.")
    router = FakeRouter(responder=reply)
    with functai.configure(lm="gpt-4.1-mini", client=router):
        s = study.stream("https://a.example/")
        shown = list(s.events(view="outside"))
        whole = list(s.events())
    assert s.result == "The fees appear late."
    notes = [(e["name"], e.get("data"), e["function"]) for e in (x if isinstance(x, dict) else x.to_dict() for x in shown)
             if e["kind"] == "note"]
    # stage (declared outside) and page captured (made by the walker's tool, declared outside by the walker
    # but not by study, the program the caller sees): only study's own outside notes
    assert notes == [("stage", "walking", "study"), ("stage", "writing", "study")]
    every = [e if isinstance(e, dict) else e.to_dict() for e in whole]
    assert {e["name"] for e in every if e["kind"] == "note"} == {"stage", "page captured", "cost so far"}
    assert views.outside_notes(study) == ["stage"]


@functai.module(notes={"stage": functai.Note(Literal["walking", "writing"], latest=True, outside=True)})
def stages(url: str) -> str:
    functai.note("stage", "walking")
    return url


def test_a_saved_module_keeps_its_notes(tmp_path):
    functai.save(stages, tmp_path / "stages")
    described = functai.describe(tmp_path / "stages")
    assert described["notes"] == stages.interface["notes"]
