"""A function loaded from a saved folder's data (``from_manifest``) is the
function that was saved. Every operation sends what the original sends:

- its calls;
- copies made with ``using``;
- row demos;
- ``map``, ``evaluate`` and optimizers;
- ``stream``.

Its interface is the one source of its inputs' names, order, requiredness
and defaults.

Three functions are compared by the requests a fake provider receives:

- the original, defined in Python;
- the function loaded by running the saved code (``load(trust=True)``);
- the function loaded from data.

The interfaces compared are an ordinary one, one whose optional input comes
before a required one, and one whose input is named ``class`` (which only
data can say; its requests are the original's with that field renamed)."""

import copy
import importlib
import inspect
import itertools
import json
import textwrap
import typing

import lm15
import pytest

import functai
from functai import calllog, saved
from conftest import FakeRouter

XML = "<result>\n{}\n</result>"
_ids = itertools.count()

PLAIN = '''
from functai import ai


@ai
def reply(message: str, tone: str = "kind") -> str:
    """Answer the customer."""
    ...
'''

# an optional input before a required one: Python says it with a keyword-only parameter
OPTIONAL_FIRST = '''
from functai import ai


@ai
def reply(message: str = "Hi", *, tone: str) -> str:
    """Answer the customer."""
    ...
'''


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for var in (calllog.ENV_FOLDER, calllog.ENV_CONTENT, calllog.ENV_CALLER):
        monkeypatch.delenv(var, raising=False)


def _renamed(manifest, old, new, key):
    """The manifest another language would have saved with the input ``old``
    named ``new`` (its fingerprints and version recomputed)."""
    m = copy.deepcopy(manifest)
    node = m["nodes"][key]
    for f in node["interface"]["inputs"]:
        if f["name"] == old:
            f["name"] = new
    for f in node["ai"]["signature"]["fields"]:
        if f["name"] == old:
            f["name"] = new
    for probe in node["ai"]["probes"]:
        if old in probe:
            probe[new] = probe.pop(old)
    fn = saved._LoadedAI(key, node, None, node["interface"])
    node["ai"]["fingerprints"] = saved._fingerprints(fn, node["ai"]["probes"])
    node["ai"]["version"] = fn.version
    return m


class Three:
    """The original, the one loaded by running its code, the one loaded from
    data; ``name``: what each calls the first input; ``expected(requests)``:
    what the one loaded from data should send, from what the original sent."""

    def __init__(self, source, tmp_path, monkeypatch, keyword=False):
        root = tmp_path / "project"
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.syspath_prepend(str(root))
        modname = f"loadedprobe_{next(_ids)}"
        (root / f"{modname}.py").write_text(textwrap.dedent(source))
        importlib.invalidate_caches()
        self.original = importlib.import_module(modname).reply
        folder = tmp_path / "saved"
        functai.save(self.original, folder)
        self.native = functai.load(folder, trust=True)
        manifest = json.loads((folder / "functai.json").read_text())
        self.keyword = keyword
        if keyword:
            manifest = _renamed(manifest, "message", "class", manifest["entry"])
        self.data = saved.from_manifest(manifest)

    def names(self, fn):
        return "class" if self.keyword and getattr(fn, "_loaded", False) else "message"

    def expected(self, requests):
        if not self.keyword:
            return requests
        text = json.dumps(requests).replace("<message>", "<class>").replace("</message>", "</class>")
        return json.loads(text)


def sends(fn, operation, reply=XML.format("Hello!")):
    """The requests ``operation(fn)`` sends, as data, in order."""
    router = FakeRouter(responder=lambda request: reply)
    with functai.configure(client=router, lm="gpt-4.1-mini"):
        operation(fn)
    return [lm15.serde.request_to_dict(r) for r in router.requests]


def row(three, fn, values):
    """``values`` with the first input named as ``fn`` names it."""
    return {three.names(fn) if k == "message" else k: v for k, v in values.items()}


def operations(three, given, demo_row, train):
    """What a caller does with an AI function: ``given``, the calls' inputs
    (named as the original names them)."""
    def calls(fn):
        for values in given:
            fn(**row(three, fn, values))

    def with_demo(fn):
        copy_ = fn.using()
        copy_.demos = [row(three, fn, demo_row)]
        calls(copy_)

    def mapped(fn):
        rows = fn.map([row(three, fn, v) for v in given]).collect().to_dicts()
        assert [r["error"] for r in rows] == [None] * len(given)

    def evaluated(fn):
        ev = functai.evaluate(fn, [row(three, fn, v) for v in train])
        assert ev.errors == [] and ev.score == 1.0

    def labeled(fn):
        calls(fn.opt([row(three, fn, v) for v in train], optimizer=functai.LabeledFewShot(k=2)))

    def bootstrapped(fn):
        calls(fn.opt([row(three, fn, v) for v in train], optimizer=functai.BootstrapFewShot(max_labeled_demos=1)))

    def streamed(fn):
        for values in given:
            assert fn.stream(**row(three, fn, values)).result == "Hello!"

    return {
        "calls": calls,
        "using()": lambda fn: calls(fn.using()),
        "using(the same model)": lambda fn: calls(fn.using(lm="gpt-4.1-mini")),
        "a row demo": with_demo,
        "map": mapped,
        "evaluate": evaluated,
        "LabeledFewShot": labeled,
        "BootstrapFewShot": bootstrapped,
        "stream": streamed,
    }


CASES = {
    "plain": (PLAIN, False, [{"message": "HELLO"}, {"message": "Hey", "tone": "warm"}]),
    "optional first": (OPTIONAL_FIRST, False, [{"tone": "brief"}, {"message": "Hey", "tone": "warm"}]),
    "an input named class": (PLAIN, True, [{"message": "HELLO"}, {"message": "Hey", "tone": "warm"}]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_function_loaded_from_data_sends_what_the_original_sends(case, tmp_path, monkeypatch):
    source, keyword, given = CASES[case]
    three = Three(source, tmp_path, monkeypatch, keyword=keyword)
    demo_row = {"message": "DEMO-INPUT", "tone": "gentle", "result": "DEMO-ANSWER"}
    train = [{**v, "result": "Hello!"} for v in given]
    for what, operation in operations(three, given, demo_row, train).items():
        original = sends(three.original, operation)
        assert original, what
        assert sends(three.native, operation) == original, what
        assert sends(three.data, operation) == three.expected(original), what
        if what == "a row demo":
            assert "DEMO-INPUT" in json.dumps(original)


def test_by_position_a_loaded_function_binds_as_the_original(tmp_path, monkeypatch):
    three = Three(PLAIN, tmp_path, monkeypatch)
    for args in (("HELLO",), ("Hey", "warm")):
        original = sends(three.original, lambda fn: fn(*args))
        assert sends(three.native, lambda fn: fn(*args)) == original
        assert sends(three.data, lambda fn: fn(*args)) == original
    three = Three(OPTIONAL_FIRST, tmp_path / "second", monkeypatch)
    original = sends(three.original, lambda fn: fn("Hey", tone="warm"))
    assert sends(three.data, lambda fn: fn("Hey", tone="warm")) == original
    # the interface's order binds positional values (Python cannot say so of the original)
    assert sends(three.data, lambda fn: fn("Hey", "warm")) == original


def test_a_loaded_function_states_its_interface_to_python(tmp_path, monkeypatch):
    three = Three(PLAIN, tmp_path, monkeypatch)
    assert str(inspect.signature(three.data)) == str(inspect.signature(three.original)) \
        == "(message: str, tone: str = 'kind') -> str"
    assert typing.get_type_hints(three.data._fn) == {"message": str, "tone": str, "return": str}
    assert three.data.__doc__ == "Answer the customer."
    three = Three(OPTIONAL_FIRST, tmp_path / "second", monkeypatch)
    assert str(inspect.signature(three.data)) == str(inspect.signature(three.original)) \
        == "(message: str = 'Hi', *, tone: str) -> str"
    three = Three(PLAIN, tmp_path / "third", monkeypatch, keyword=True)
    assert str(inspect.signature(three.data)) == "(*, tone: str = 'kind', **inputs) -> str"


def test_what_reads_a_loaded_function_s_inputs_reads_its_interface(tmp_path, monkeypatch):
    from functai.bake.examples import row_inputs
    from functai.columns import _return_type
    three = Three(OPTIONAL_FIRST, tmp_path, monkeypatch)
    for values in ({"tone": "brief"}, {"message": "Hey", "tone": "warm", "result": "x"}):
        assert row_inputs(three.data, values) == row_inputs(three.original, values)
    assert three.data._named_inputs() == three.original._named_inputs() == [("message", False), ("tone", True)]
    assert _return_type(three.data._fn, None) is str            # a column of text, as the original's
    assert type(three.data.using()) is type(three.data)


def test_a_loaded_function_refuses_a_setting_that_would_change_its_signature(tmp_path, monkeypatch):
    three = Three(PLAIN, tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="loaded from data"):
        three.data.using(module="cot")
    with pytest.raises(TypeError, match="loaded from data"):
        three.data.module = "cot"

    def lookup(order: str) -> str:
        """An order's status."""
        return "shipped"

    with pytest.raises(TypeError, match="loaded from data"):
        three.data.tools = [lookup]
    three.data.using(module="predict")                          # what it is: nothing changes


def test_a_function_loaded_from_data_is_not_saved_as_python(tmp_path, monkeypatch):
    three = Three(PLAIN, tmp_path, monkeypatch)
    report = functai.check(three.data)
    assert [p.code for p in report.errors] == ["loaded-from-data"]
    for allow in ((), ("loaded-from-data",)):
        with pytest.raises(functai.Refused):
            functai.save(three.data, tmp_path / "again", allow=allow)
    assert not (tmp_path / "again").exists()


@pytest.mark.parametrize("interface, text, exact", [
    ({"inputs": [{"name": "a", "shape": {"type": "integer"}},
                 {"name": "b", "shape": {"type": ["string", "null"], "default": None}, "optional": True}],
      "outputs": [{"name": "result", "shape": {"type": "array", "items": {"enum": ["x", "y"]}}}]},
     "(a: int, b: Optional[str] = None) -> List[Literal['x', 'y']]", True),
    ({"inputs": [{"name": "a", "shape": {"type": "boolean", "default": True}, "optional": True},
                 {"name": "b", "shape": {"type": "object"}}, {"name": "if", "shape": {}}],
      "outputs": [{"name": "result", "shape": {"$ref": "#/$defs/n", "$defs": {"n": {"type": "number"}}}}]},
     "(a: bool = True, *, b: Dict[str, Any], **inputs) -> float", False),
])
def test_an_interface_as_a_python_signature(interface, text, exact):
    from functai.interface import python_signature
    signature, is_exact = python_signature(interface)
    assert str(signature).replace("typing.", "") == text and is_exact is exact


# inputs with every name the ``**`` of a Python signature has had; two are renamed, as another language
# could name them, to names Python reserves (``class``, ``for``)
EVERY_NAME = '''
from functai import ai


@ai
def reply(message: str, tone: str, inputs: str, fields: str, values: str, named: str, inputs_1: str) -> str:
    """Answer the customer."""
    ...
'''


def test_a_loaded_function_takes_any_names_its_interface_gives(tmp_path, monkeypatch):
    three = Three(EVERY_NAME, tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / "saved" / "functai.json").read_text())
    for old, new in (("message", "class"), ("tone", "for")):
        manifest = _renamed(manifest, old, new, manifest["entry"])
    data = saved.from_manifest(manifest)
    assert [f["name"] for f in data.interface["inputs"]] == \
        ["class", "for", "inputs", "fields", "values", "named", "inputs_1"]
    assert str(inspect.signature(data)) == \
        "(*, inputs: str, fields: str, values: str, named: str, inputs_1: str, **inputs_2) -> str"
    given = {"message": "HELLO", "tone": "warm", "inputs": "i", "fields": "f", "values": "v", "named": "n",
             "inputs_1": "i1"}
    renamed = {{"message": "class", "tone": "for"}.get(k, k): v for k, v in given.items()}

    def expected(requests):
        text = json.dumps(requests)
        for old, new in (("message", "class"), ("tone", "for")):
            text = text.replace(f"<{old}>", f"<{new}>").replace(f"</{old}>", f"</{new}>")
        return json.loads(text)

    for what, operation in {"a call": lambda fn, v: fn(**v),
                            "using()": lambda fn, v: fn.using()(**v),
                            "using(the same model)": lambda fn, v: fn.using(lm="gpt-4.1-mini")(**v)}.items():
        original = sends(three.original, lambda fn: operation(fn, given))
        assert original, what
        assert sends(data, lambda fn: operation(fn, renamed)) == expected(original), what
    # the interface binds a call: the name its ``**`` has for Python is no input
    with pytest.raises(TypeError, match="inputs_2"):
        data(**{**renamed, "inputs_2": "x"})


RECORDS = '''
from typing import Literal, Optional

from pydantic import BaseModel

from functai import ai


class Item(BaseModel):
    name: str
    qty: int


class Box(BaseModel):
    label: str
    item: Item


@ai
def reply(items: list[Item], box: Box, mood: Literal["a", "b"] = "a", note: Optional[str] = None,
          one: Optional[Item] = None) -> Item:
    """Pick one."""
    ...
'''


def test_a_loaded_function_used_as_a_tool_is_the_original_s_tool(tmp_path, monkeypatch):
    """Its tool's JSON Schema is its interface's shapes, records whole, as the
    original's annotations give them; a model's call of it, and what it
    answers, are sent as the original's are."""
    from functai import engine
    three = Three(RECORDS, tmp_path, monkeypatch)
    original = engine.tool_spec(three.original)
    assert original.parameters["properties"]["items"]["items"]["required"] == ["name", "qty"]
    assert "$defs" in original.parameters["properties"]["box"]           # a record in a record
    assert engine.tool_spec(three.native) == original
    assert engine.tool_spec(three.data) == original

    def reply(request):
        if not request.tools:                                      # the tool: the function itself
            return XML.format(json.dumps({"name": "a", "qty": 1}))
        if request.messages[-1].role == "tool":
            return XML.format("Done")
        return [lm15.ToolCallPart(id="c1", name="reply", input={
            "items": [{"name": "a", "qty": 1}], "mood": "b",
            "box": {"label": "L", "item": {"name": "b", "qty": 2}}})]

    def agent_of(fn):
        @functai.ai(tools=[fn])
        def agent(question: str) -> str:
            """Use the tool."""
            ...
        return agent

    requests = {}
    for which in ("original", "native", "data"):
        agent = agent_of(getattr(three, which))
        router = FakeRouter(responder=reply)
        with functai.configure(client=router, lm="gpt-4.1-mini"):
            assert agent("x") == "Done"
        requests[which] = [lm15.serde.request_to_dict(r) for r in router.requests]
    assert len(requests["original"]) == 3                          # the agent, the tool, the agent again
    assert requests["original"][0]["tools"][0]["parameters"] == original.parameters
    [answered] = requests["original"][2]["messages"][-1]["parts"]
    assert answered["content"] == [{"type": "text", "text": '{"name": "a", "qty": 1}'}]      # its answer, as JSON
    assert requests["native"] == requests["original"]
    assert requests["data"] == requests["original"]


COT = '''
from functai import ai


@ai(module="cot")
def reply(message: str, tone: str = "kind") -> str:
    """Answer the customer."""
    ...
'''

# chain of thought whose reasoning is an output of its own (no field FunctAI adds)
COT_OWN = '''
from functai import ai, _ai


@ai(module="cot")
def reply(message: str, tone: str = "kind") -> str:
    """Answer the customer."""
    reasoning: str = _ai
    return _ai
'''


@pytest.mark.parametrize("source", [COT, COT_OWN], ids=["cot", "cot with its own reasoning"])
def test_a_function_saved_with_its_module_loads_from_data_with_it(source, tmp_path, monkeypatch):
    three = Three(source, tmp_path, monkeypatch)
    answer = "<reasoning>\nhm\n</reasoning>\n" + XML.format("Hello!")
    for what, operation in {"a call": lambda fn: fn("HELLO"),
                            "using(the same model)": lambda fn: fn.using(lm="gpt-4.1-mini")("HELLO"),
                            "using(its own module)": lambda fn: fn.using(module="cot")("HELLO")}.items():
        original = sends(three.original, operation, answer)
        assert sends(three.native, operation, answer) == original, what
        assert sends(three.data, operation, answer) == original, what
    with pytest.raises(TypeError, match="loaded from data"):
        three.data.using(module="predict")
