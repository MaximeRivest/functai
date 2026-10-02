"""Stage 1.1 (design/09): binding a call's inputs, what an error message
quotes, defaults counted by their logic, a re-ask's request hash, every tool
call in outputs.calls, `returned` and dropped fields, a host refusing a
program's observers, notebooks known by their file, ratings under an account,
and the libraries a record names. Offline: a fake provider answers."""

import json
import sys
import textwrap
from pathlib import Path
from typing import Literal, Optional

import lm15
import pytest

import functai
from conftest import FakeRouter
from functai import ai, calllog, module


@pytest.fixture(autouse=True)
def no_env(monkeypatch, tmp_path):
    for var in (calllog.ENV_FOLDER, calllog.ENV_CONTENT, calllog.ENV_CALLER):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    calllog._warned.clear()


def records(folder):
    return [json.loads(line) for f in sorted(Path(folder).rglob("*.jsonl")) for line in f.read_text().splitlines()
            if '"functai_call"' in line]


def xml(value):
    return f"<result>\n{value}\n</result>"


# ------------------------------------------------------------------ A6: binding an AI function's inputs


def test_an_ai_function_binds_its_inputs_and_records_the_bound_values(tmp_path):
    @ai
    def label(note, count: int, tone: str = "kind") -> str:
        """Label a note."""

    router = FakeRouter(responder=lambda r: xml("ok"))
    with functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path):
        label(42, "5")                                    # a number for text, text for an integer
        label({"b": 1}, 5.0, None)                        # a missing value for an optional input: its default
        with pytest.raises(functai.InterfaceError) as err:
            label("x", "five")
    assert err.value.code == "interface-input" and err.value.field == "count"
    assert '"five"' in str(err.value)                     # the message quotes the value
    first, second, refused = records(tmp_path)
    assert first["inputs"] == {"note": "42", "count": 5, "tone": "kind"}
    assert second["inputs"] == {"note": '{\n  "b": 1\n}', "count": 5, "tone": "kind"}
    assert refused["error"]["type"] == "InterfaceError" and refused["exchanges"] == []
    assert len(router.requests) == 2                      # the refused call sent nothing
    assert "<count>\n5\n</count>" in router.requests[0].messages[-1].parts[0].text


def test_a_message_never_quotes_a_value_the_log_drops():
    @ai(log_content={"secret": False})
    def check(secret: int) -> str:
        """Check."""

    with pytest.raises(functai.InterfaceError) as err:
        check("card 4242")
    assert "4242" not in str(err.value) and "secret" in str(err.value)
    with functai.configure(log_content=False):
        @ai
        def other(n: int) -> str:
            """Other."""
        with pytest.raises(functai.InterfaceError) as err:
            other("card 4242")
    assert "4242" not in str(err.value)


def test_a_default_is_bound_when_the_function_is_defined():
    @ai
    def pick(n: int = "5") -> str:                     # noqa: RUF100 — a default that binds to 5
        """Pick."""
    assert pick.interface["inputs"][0]["shape"]["default"] == 5
    with pytest.raises(functai.InterfaceError, match="interface-malformed|does not fit"):
        @ai
        def bad(n: int = "five") -> str:
            """Bad."""


def test_a_module_s_code_gets_the_bound_values():
    got = {}

    @module
    def keep(n: int, *more: int, **named: int) -> str:
        got.update(n=n, more=more, named=named)
        return "ok"

    keep("3", "4", 5.0, extra="6")
    assert got == {"n": 3, "more": (4, 5), "named": {"extra": 6}}


def test_a_record_input_keeps_only_the_members_it_names():
    from dataclasses import dataclass

    @dataclass
    class Person:
        name: str
        age: int

    got = {}

    @module
    def greet(person: Person) -> str:
        got["person"] = person
        return "hi"

    greet({"name": "Ada", "age": "36", "city": "London"})
    assert got["person"] == {"name": "Ada", "age": 36}
    greet(Person("Ada", 36))
    assert got["person"] == Person("Ada", 36)          # a value that fits as it is stays the program's own


# ------------------------------------------------------------------ B14: defaults by their logic


def _function_from_source(tmp_path, source, name, stamp):
    path = tmp_path / f"notebook_{stamp}.py"
    path.write_text(textwrap.dedent(source))
    namespace = {"__name__": f"nb_{stamp}"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace[name]


def test_a_computed_default_counts_by_what_is_written(tmp_path):
    source = """
        from functai import ai
        def today():
            return "{day}"
        @ai
        def plan(notes: str, day: str = today(), tone: str = "{tone}") -> str:
            \"\"\"Plan the day.\"\"\"
    """
    monday = _function_from_source(tmp_path, source.format(day="2026-09-28", tone="kind"), "plan", 1)
    tuesday = _function_from_source(tmp_path, source.format(day="2026-09-29", tone="kind"), "plan", 2)
    formal = _function_from_source(tmp_path, source.format(day="2026-09-28", tone="formal"), "plan", 3)
    assert monday.interface["inputs"][1]["shape"]["default"] != tuesday.interface["inputs"][1]["shape"]["default"]
    assert monday.version == tuesday.version              # the same logic on another day
    assert monday.version != formal.version               # another default value
    assert functai.interface.signature(monday.interface) == functai.interface.signature(formal.interface)


def test_a_saved_folder_keeps_a_default_s_code(tmp_path):
    source = """
        from functai import ai
        def today():
            return "2026-09-30"
        @ai
        def plan(notes: str, day: str = today()) -> str:
            \"\"\"Plan the day.\"\"\"
    """
    fn = _function_from_source(tmp_path, source, "plan", 4)
    sys.modules["nb_4"] = type(sys)("nb_4")              # a module save can name
    folder = tmp_path / "saved"
    functai.save(fn, folder)
    manifest = json.loads((folder / "functai.json").read_text())
    node = next(n for n in manifest["nodes"].values() if n["kind"] == "ai")
    assert node["defaults"] == {"day": {"code": "today()"}}
    loaded = functai.saved.from_manifest(manifest)
    assert loaded.version == fn.version


# ------------------------------------------------------------------ A4, A5, A9: what a record holds


def test_a_re_ask_has_the_hash_of_what_it_sent(tmp_path):
    @ai(retries=1)
    def mood(text: str) -> Literal["happy", "sad"]:
        """Mood."""

    router = FakeRouter(xml("maybe"), xml("happy"))
    with functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path):
        mood("a sunny day")
    [rec] = records(tmp_path)
    first, again = rec["exchanges"]
    assert first["request_hash"] != again["request_hash"]
    assert len(router.requests[1].messages) == len(router.requests[0].messages) + 2


def test_outputs_calls_holds_every_tool_call_of_the_call(tmp_path):
    def lookup(order: str) -> str:
        """Look up an order."""
        return "in Leeds"

    @ai(tools=[lookup])
    def where(question: str) -> str:
        """Where is the parcel?"""

    call = lm15.ToolCallPart(id="c1", name="lookup", input={"order": "B-1"})
    router = FakeRouter([call], xml("In Leeds."))
    with functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path):
        where("Where is B-1?")
    [rec] = records(tmp_path)
    assert [c["name"] for c in rec["outputs"]["calls"]] == ["lookup"]


def test_what_the_code_returned_is_kept_only_when_nothing_is_dropped(tmp_path):
    from functai import _ai

    @ai(log_content={"critique": False})
    def review(text: str) -> tuple[str, str]:
        """Review."""
        critique: str = _ai["what is wrong"]
        return critique, _ai

    router = FakeRouter(responder=lambda r: "<critique>\nsecret\n</critique>\n<result>\nfine\n</result>")
    with functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path):
        review("a draft")
    [rec] = records(tmp_path)
    assert "returned" not in rec and "secret" not in json.dumps(rec)


def test_dropping_the_tool_calls_keeps_the_reasoning():
    kept = calllog.kept_fields(["question"], ["reasoning", "calls", "result"], ["reasoning", "calls"],
                               [{"calls": False}])
    assert kept == {"question": True, "reasoning": True, "calls": False, "result": True}
    kept = calllog.kept_fields(["question"], ["reasoning", "calls", "result"], ["reasoning", "calls"],
                               [{"question": False}])
    assert kept["reasoning"] is False and kept["calls"] is False


# ------------------------------------------------------------------ A8: a host refuses a program's observers


def test_a_host_refuses_a_program_s_observers():
    mine, host = [], []

    @ai(observers=[mine])
    def answer(q: str) -> str:
        """Answer."""

    router = FakeRouter(responder=lambda r: xml("yes"))
    with functai.configure(client=router, lm="gpt-4.1-mini", observers=[host], program_observers=False):
        answer("?")
    assert host and not mine
    with functai.configure(client=router, lm="gpt-4.1-mini"):
        answer("?")
    assert mine


# ------------------------------------------------------------------ F1, F2, F5


def test_a_notebook_s_program_is_known_by_the_notebook(monkeypatch):
    monkeypatch.setenv("JPY_SESSION_NAME", "/home/maxime/legal.ipynb")
    assert calllog.top_level_file("/tmp/ipykernel_123/456.py") == "/home/maxime/legal.ipynb"
    monkeypatch.delenv("JPY_SESSION_NAME")
    assert calllog.top_level_file("/tmp/ipykernel_123/456.py") is None
    assert calllog.top_level_file("/home/maxime/job.py") == "/home/maxime/job.py"


def test_a_record_names_the_libraries_that_made_it(tmp_path):
    import lmcc

    router = FakeRouter(responder=lambda r: xml("x"))

    @ai
    def echo(text: str) -> str:
        """Echo."""

    with functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path):
        echo("hi")
    [rec] = records(tmp_path)
    assert rec["process"]["lmcc"] == lmcc.__version__ and rec["process"]["lm15"]


def test_ratings_under_a_shared_account_are_all_kept(tmp_path):
    router = FakeRouter(responder=lambda r: xml("billing"))

    @ai
    def team(message: str) -> str:
        """Team."""

    with functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path):
        p = team.predict("charged twice")
        functai.rate(p, "right")                         # one person on the family account
        functai.rate(p, "wrong", answer="shipping")      # another, on the same account
        rows = functai.rated(team, any_file=True).to_pylist() if hasattr(functai.rated(team, any_file=True),
                                                                          "to_pylist") else None
    ratings = [json.loads(line) for f in tmp_path.rglob("*.jsonl") for line in f.read_text().splitlines()
               if '"functai_rating"' in line]
    assert all("by" not in r and r["account"] for r in ratings)
    found, read = calllog.read(tmp_path)
    got, _left = calllog.rated_rows(found, read, name="team")
    assert got[0]["disputed"] is True and got[0]["rated_by"] is None and got[0]["result"] == "shipping"
    del rows
