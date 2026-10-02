"""What happens when things go wrong, through the real calls: a value given
under a name the interface lacks, a call that cannot be prepared, a store that
answers something else than kept, a value whose repr raises, a receiver that
changes what it is given, a store that never answers, an observer that comes
and goes, a saved folder the schema refuses, a history edited by hand.

Each test here began as a reproduction that showed a defect; it stays so the
defect cannot come back unseen."""

import copy
import dataclasses
import gc
import json
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import weakref
from pathlib import Path

import lm15
import pytest

import functai
from functai import _ai, ai, calllog, eventlog, module, saved, schemas
from functai.errors import EventRefused
from conftest import FakeRouter
from contract_support import CONTRACT, assert_valid, case_files, load, validator

EVENT = validator("event")
CALL = validator("call")
XML = "<result>\n{}\n</result>"
HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for var in (calllog.ENV_FOLDER, calllog.ENV_CONTENT, calllog.ENV_CALLER):
        monkeypatch.delenv(var, raising=False)


def records(folder):
    return [json.loads(line) for f in sorted(Path(folder).rglob("*.jsonl")) for line in f.read_text().splitlines()]


def everything(*values) -> str:
    return json.dumps(values, ensure_ascii=False, default=str)


IFACE = {"description": "", "inputs": [{"name": "x", "shape": {"type": "string"}}],
         "outputs": [{"name": "result", "shape": {"type": "string"}}]}


# ------------------------------------------------------------------ retention fails closed


@pytest.mark.parametrize("dropping", ["false", "star", "environment", "whole"])
def test_a_value_under_a_name_the_interface_lacks_is_refused_and_kept_nowhere(dropping, tmp_path, monkeypatch):
    """A call given an input its interface does not have is refused
    (interface-input, naming it), and no record, observer or journal keeps
    what it held, whatever log_content says (it is no field: its value is
    never recorded)."""
    own = {"false": {"log_content": False}, "star": {"log_content": {"*": False, "x": True}}}.get(dropping, {})
    if dropping == "environment":
        monkeypatch.setenv(calllog.ENV_CONTENT, "0")
    seen, store = [], functai.MemoryStore()

    @module(interface=IFACE, observers=[seen], log_calls=tmp_path, journal=store, **own)
    def echo(**inputs):
        return "ok"

    with pytest.raises(functai.InterfaceError) as err:
        echo(x="ok", password="SECRET-UNKNOWN-INPUT")
    assert err.value.code == "interface-input" and err.value.field == "password"
    assert isinstance(err.value, TypeError)
    assert functai.flush(5)
    [rec] = records(tmp_path)
    assert_valid(CALL, rec)
    kept = store.read(store.trees()[0])
    assert "SECRET-UNKNOWN-INPUT" not in everything(rec, seen, kept)
    assert rec["error"]["code"] == "interface-input"
    if dropping == "whole":
        assert rec["content"] is True and rec["inputs"] == {"x": "ok"}          # the input it has, as given
        assert seen[0]["inputs"] == {"x": "ok"}


def test_a_call_that_cannot_be_prepared_keeps_nothing(fake, monkeypatch):
    """If what a record and its events need cannot be worked out, nothing of
    the call is kept or given (fail closed), and a required journal stops the
    tree at its start: the code does not run."""
    router = fake(XML.format("hunter2 was the secret"))

    @ai
    def echo(secret: str) -> str:
        """Repeat."""

    monkeypatch.setattr(calllog, "program_info", lambda program: 1 / 0)
    seen = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with functai.configure(observers=[seen]):
            assert echo("hunter2") == "hunter2 was the secret"
    assert seen == []                                                     # not even a started with every value
    assert all("hunter2" not in str(w.message) for w in caught)
    store = functai.MemoryStore()
    with functai.configure(journal=functai.Journal(store, required=True)):
        with pytest.raises(functai.JournalError) as err:
            echo("hunter2")
    assert err.value.code == "journal-end" and err.value.journal == "refused"
    assert err.value.outcome.error.code == "journal-barrier"
    assert len(router.requests) == 1 and store.trees() == []              # the second call's code never ran


# ------------------------------------------------------------------ a required journal needs an answer that says kept


@pytest.mark.parametrize("answer", ["event-conflict", None, "ok", True])
def test_a_store_answer_other_than_kept_or_duplicate_is_a_refusal(answer, fake):
    ran = []

    class Answering:
        def append(self, event):
            return answer

        def read(self, tree, after=None):
            return []

    @module(journal=functai.Journal(Answering(), required=True))
    def effect() -> str:
        ran.append(1)
        return "CODE-RAN"

    with pytest.raises(functai.JournalError) as err:
        effect()
    assert ran == []
    assert err.value.code == "journal-end" and err.value.journal == "refused"
    assert err.value.outcome.error.code == "journal-barrier"
    writer = eventlog.JournalWriter(functai.Journal(Answering(), required=True), thread=False)
    writer.send({"functai_event": 2, "kind": "started", "tree": "t", "writer": 1, "seq": 1})
    assert writer.barrier() == "refused"


def test_a_value_whose_repr_raises_is_described_and_its_end_is_kept():
    """A value with no JSON form whose repr raises is described without the
    error's message; the required journal keeps the end, and no warning quotes
    the value."""
    class BadRepr:
        def __repr__(self):
            raise ValueError("SECRET-REPR")

    store = functai.MemoryStore()
    seen = []

    @module(journal=functai.Journal(store, required=True), observers=[seen])
    def value():
        return BadRepr()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert isinstance(value(), BadRepr)
    [tree] = store.trees()
    assert store.finished(tree)
    done = store.read(tree)[-1]
    assert done["value"]["$type"].endswith("BadRepr")
    assert done["value"]["$repr"] == f"<{done['value']['$type']} object: its repr raised ValueError>"
    assert "SECRET-REPR" not in everything([str(w.message) for w in caught], store.read(tree), seen)

    @module(journal=functai.Journal(store, required=True), log_content=False)
    def hidden():
        return BadRepr()

    assert isinstance(hidden(), BadRepr)
    tree = [t for t in store.trees() if t != tree][0]
    assert store.finished(tree) and "value" not in store.read(tree)[-1]

    @module
    def takes(x: str) -> str:
        return x

    with pytest.raises(functai.InterfaceError) as err:                     # not the repr's ValueError
        takes(BadRepr())
    assert err.value.field == "x" and "SECRET" not in str(err.value)


# ------------------------------------------------------------------ receivers never share an event


def test_an_observer_that_changes_its_event_changes_no_one_else_s():
    changed = threading.Event()
    second = []
    lists = []

    def first(e):
        if e["kind"] == "started":
            e["inputs"]["x"] = "CORRUPTED"
            changed.set()

    def later(e):
        if e["kind"] == "started":
            changed.wait(2)
        second.append(e)

    class Slow(functai.MemoryStore):
        def append(self, e):
            if e["kind"] == "started":
                assert changed.wait(3)
            return super().append(e)

        def extend(self, events):
            if any(e["kind"] == "started" for e in events):
                assert changed.wait(3)
            return super().extend(events)

    def mutate_list_later():
        if lists:
            lists[0]["inputs"]["x"] = "LIST-CORRUPTED"

    store = Slow()

    @module(observers=[first, later, lists], journal=functai.Journal(store, required=True))
    def echo(x: str) -> str:
        mutate_list_later()
        return x

    assert echo("original") == "original"
    assert functai.flush(5)
    assert second[0]["inputs"] == {"x": "original"}
    assert store.read(store.trees()[0])[0]["inputs"] == {"x": "original"}


def test_a_store_that_changes_what_it_is_given_changes_no_resend():
    """Each append gets a fresh copy: a store (or a transport) that changes it
    cannot change what is sent again after no answer came."""
    sent = []

    class Mangling(functai.MemoryStore):
        extend = None

        def append(self, e):
            sent.append(copy.deepcopy(e))
            if len(sent) == 1:
                e["inputs"]["x"] = "MANGLED"
                raise TimeoutError("no answer")
            return super().append(e)

    store = Mangling()

    @module(journal=functai.Journal(store, required=True, backoff=0))
    def echo(x: str) -> str:
        return x

    assert echo("original") == "original"
    assert sent[0] == sent[1] and sent[1]["inputs"] == {"x": "original"}


# ------------------------------------------------------------------ saved folders: the whole schema, every probe


def manifest16():
    return copy.deepcopy(load(CONTRACT / "cases" / "saved" / "16-an-optional-input.json")["manifest"])


def refit(m, key="shop:reply"):
    """Fingerprints and version recomputed for a node changed by hand (what
    another language would have saved for it)."""
    node = m["nodes"][key]
    fn = saved._LoadedAI(key, node, None, node.get("interface"))
    node["ai"]["fingerprints"] = saved._fingerprints(fn, node["ai"]["probes"])
    node["ai"]["version"] = fn.version
    return m


def test_an_optional_input_before_a_required_one_loads_and_binds_from_data():
    m = manifest16()
    fields = m["nodes"]["shop:reply"]["interface"]["inputs"]
    fields[0]["optional"] = True
    fields[0]["shape"]["default"] = "Hi"
    fields[1].pop("optional")
    fields[1]["shape"].pop("default")
    refit(m)                                    # its defaults are in its version: what that language saved
    assert validator("saved").is_valid(m)
    reply = saved.from_manifest(m)
    router = FakeRouter(responder=lambda request: XML.format("Hello!"))
    with functai.configure(client=router, lm="gpt-4.1-mini"):
        assert reply(tone="brief") == "Hello!"
        assert "<message>\nHi\n</message>" in router.user() and "<tone>\nbrief\n</tone>" in router.user()
        reply("Hey", "warm")                                              # by position, in the interface's order
        assert "<message>\nHey\n</message>" in router.user() and "<tone>\nwarm\n</tone>" in router.user()
        with pytest.raises(TypeError, match="missing the input 'tone'"):
            reply("Hey")


def test_an_input_named_as_a_python_keyword_loads_and_is_given_by_name():
    m = manifest16()
    node = m["nodes"]["shop:reply"]
    node["interface"]["inputs"][0]["name"] = "class"
    node["ai"]["signature"]["fields"][0]["name"] = "class"
    for probe in node["ai"]["probes"]:
        probe["class"] = probe.pop("message")
    refit(m)
    assert validator("saved").is_valid(m)
    reply = saved.from_manifest(m)
    router = FakeRouter(responder=lambda request: XML.format("Hello!"))
    with functai.configure(client=router, lm="gpt-4.1-mini"):
        assert reply(**{"class": "Hi"}) == "Hello!"
    assert "<class>\nHi\n</class>" in router.user()


def test_a_manifest_the_schema_refuses_is_refused_before_anything_else():
    m = manifest16()
    del m["nodes"]["shop:reply"]["ai"]["fingerprints"]
    for read in (saved.from_manifest, functai.describe):
        with pytest.raises(functai.LoadRefused) as err:
            read(m)
        assert err.value.code == "saved-malformed"
    m = manifest16()
    m["nodes"]["shop:reply"]["ai"]["fingerprints"]["signature"] = "not-a-hash"
    del m["nodes"]["shop:reply"]["ai"]["probes"]
    with pytest.raises(functai.LoadRefused) as err:
        functai.describe(m)
    assert err.value.code == "saved-malformed"


@pytest.mark.parametrize("requests", [[], ["refused:x"] * 2])
def test_every_probe_has_its_fingerprint(requests):
    m = manifest16()
    data = m["nodes"]["shop:reply"]["ai"]
    data.pop("version")
    data["fingerprints"]["requests"] = requests
    data["signature"]["instructions"] = "A completely different prompt"
    with pytest.raises(functai.LoadRefused) as err:
        saved.from_manifest(m)
    assert err.value.code == "saved-malformed"


def test_a_store_refuses_an_event_the_schema_refuses():
    case = load(case_files("events", "journal-01")[0])
    e = copy.deepcopy(case["events"][0])
    e["at"], e["program"], e["content"] = "yesterday", None, False
    assert not validator("event").is_valid(e)
    with pytest.raises(EventRefused) as err:
        functai.MemoryStore().append(e)
    assert err.value.code == "event-malformed"


PORTABLE = '''
    from functai import module

    @module
    def portable_default(x: str = "hello") -> str:
        return x
    '''


def test_a_trusted_load_checks_the_interface_before_running_code(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "portable.py").write_text(textwrap.dedent(PORTABLE))
    monkeypatch.syspath_prepend(str(src))
    import portable
    folder = tmp_path / "saved"
    functai.save(portable.portable_default, folder)
    path = folder / "functai.json"
    original = json.loads(path.read_text())
    m = copy.deepcopy(original)
    m["nodes"][m["entry"]]["interface"]["inputs"][0]["shape"]["default"] = 123       # does not fit
    path.write_text(json.dumps(m))
    before = set(sys.modules)
    with pytest.raises(functai.LoadRefused) as err:
        functai.load(folder, trust=True)
    assert err.value.code == "interface-malformed"
    assert not any(name.startswith("_functai_saved_") for name in set(sys.modules) - before)   # nothing ran
    m["nodes"][m["entry"]]["interface"]["inputs"][0]["shape"]["default"] = "bye"     # fits, and is not what it does
    path.write_text(json.dumps(m))
    with pytest.raises(functai.LoadRefused, match="other data than its saved interface"):
        functai.load(folder, trust=True)
    path.write_text(json.dumps(original))
    assert functai.load(folder, trust=True)() == "hello"


TRIAGE = '''
    from functai import module

    @module(outputs={"team": str, "result": str})
    def triage(ticket: str) -> dict:
        return {"team": "support", "result": ticket}

    @module(outputs={"team": str, "result": str})
    def bare(ticket: str):
        return {"team": "support", "result": ticket}
    '''


@pytest.mark.parametrize("name", ["triage", "bare"])
def test_a_module_with_several_outputs_saves_describes_and_loads(name, tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "triaging.py").write_text(textwrap.dedent(TRIAGE))
    monkeypatch.syspath_prepend(str(src))
    import triaging
    program = getattr(triaging, name)
    folder = tmp_path / name
    functai.save(program, folder)
    assert [f["name"] for f in functai.describe(folder)["outputs"]] == ["team", "result"]
    loaded = functai.load(folder, trust=True)
    assert loaded.interface == program.interface
    assert loaded("printer on fire") == {"team": "support", "result": "printer on fire"}


def test_load_gives_the_node_asked_for(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "pair.py").write_text(textwrap.dedent('''
        from functai import module

        @module
        def helper(x: str) -> str:
            return x.upper()

        @module
        def entry(x: str) -> str:
            return helper(x)
        '''))
    monkeypatch.syspath_prepend(str(src))
    import pair
    functai.save(pair.entry, tmp_path / "saved")
    assert functai.load(tmp_path / "saved", trust=True, node="pair:helper").__name__ == "helper"
    with pytest.raises(functai.LoadRefused) as err:
        functai.load(tmp_path / "saved", trust=True, node="pair:nothing")
    assert err.value.code == "saved-malformed"


# ------------------------------------------------------------------ a module's inputs, named as its interface names them


def test_a_derived_module_names_the_input_at_fault(tmp_path):
    @module(log_calls=tmp_path)
    def m(message: str, tone: str = "kind") -> str:
        return tone

    for call, field in [(lambda: m(), "message"), (lambda: m("x", colour="red"), "colour"),
                        (lambda: m(message=None), "message"), (lambda: m("x", message="y"), "message")]:
        with pytest.raises(functai.InterfaceError) as err:
            call()
        assert err.value.code == "interface-input" and err.value.field == field
        assert isinstance(err.value, TypeError)
    with pytest.raises(functai.InterfaceError) as err:
        m("a", "b", "c")
    assert err.value.field is None
    recs = sorted(records(tmp_path), key=lambda r: r["id"])
    assert [r["error"]["code"] for r in recs] == ["interface-input"] * 5
    assert recs[1]["inputs"] == {"message": "x", "tone": "kind"}               # what it was given that it has
    assert m("hi") == "kind"

    @module
    def extra(first: str, *rest: str, **named: int) -> str:
        return first + "".join(rest) + str(sorted(named.items()))

    assert extra("a", "b", "c", n=1) == "abc[('n', 1)]"
    with pytest.raises(functai.InterfaceError) as err:
        extra("a", n="not a number")
    assert err.value.field == "named"
    with pytest.raises(functai.InterfaceError):
        m.stream()                                                           # in the caller, as fn.stream does


# ------------------------------------------------------------------ what a call saw is what it was shown


def test_what_a_call_saw_is_what_it_was_shown_even_if_history_is_reset_meanwhile(tmp_path):
    router = FakeRouter(responder=lambda request: XML.format("ok"))
    functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path)

    @ai(stateful=True)
    def chat(message: str) -> str:
        """Answer."""

    chat("FIRST-MESSAGE")

    class Gate(functai.MemoryStore):
        def __init__(self):
            super().__init__()
            self.arrived, self.allow = threading.Event(), threading.Event()

        def append(self, e):
            if e["kind"] == "started":
                self.arrived.set()
                assert self.allow.wait(5)
            return super().append(e)

        def extend(self, events):
            if any(e["kind"] == "started" for e in events):
                self.arrived.set()
                assert self.allow.wait(5)
            return super().extend(events)

    gate = Gate()
    with functai.configure(journal=functai.Journal(gate, required=True)):
        s = chat.stream("SECOND")
        assert gate.arrived.wait(5)
        assert len(router.requests) == 1                                       # nothing sent before the start is kept
        chat.reset()
        gate.allow.set()
        assert s.result == "ok"
    last = max(records(tmp_path), key=lambda r: r["id"])
    shown = "FIRST-MESSAGE" in str(router.requests[-1])
    assert bool(last["saw"]) == shown and shown                               # the record and the request agree


def test_a_turn_s_call_is_known_by_the_turn_itself():
    router = FakeRouter(responder=lambda request: XML.format("ok"))
    functai.configure(client=router, lm="gpt-4.1-mini")

    @ai(stateful=True)
    def chat(message: str) -> str:
        """Answer."""

    first, second = chat.predict("one"), chat.predict("two")
    entries, _turns = chat._saw()
    assert [e["call"] for e in entries] == [first.call_id, second.call_id]
    chat.history.pop()                                                         # the newest turn dropped by hand
    entries, _turns = chat._saw()
    assert [e["call"] for e in entries] == [first.call_id]                    # the one left keeps its own id
    chat.history.append(dataclasses.replace(chat.history[0]) if dataclasses.is_dataclass(chat.history[0])
                        else copy.copy(chat.history[0]))                      # a turn made by hand
    entries, _turns = chat._saw()
    assert entries == [{"call": first.call_id, "steps": True}, {"unrecorded": True}]
    chat.history.reverse()
    entries, _turns = chat._saw()
    assert entries == [{"unrecorded": True}, {"call": first.call_id, "steps": True}]


# ------------------------------------------------------------------ a format this reader does not know


def test_an_unknown_format_stops_replay_resume_and_following():
    events = copy.deepcopy(load(case_files("events", "journal-01")[0])["events"])
    events[1]["functai_event"] = 99

    class Source:
        writer = None

        def read(self, tree, after=None):
            return eventlog.read_after(events, after)

    follower = functai.Follower()
    follower.resume(events[0]["tree"], Source())
    assert follower.stopped and len(follower.events(events[0]["tree"])) == 1
    r = eventlog.Replay()
    for e in events:
        r.apply(e)
    assert r.stopped and not r.finished and list(r.calls) == [events[0]["call"]]
    assert len(eventlog.replay(events)) == 1
    other = functai.Follower()
    assert [other.receive(e) for e in events[:3]] == ["kept", "unknown-format", "unknown-format"]


def test_a_follower_forgets_a_tree():
    events = load(case_files("events", "journal-01")[0])["events"]
    follower = functai.Follower()
    for e in events:
        follower.receive(e)
    tree = events[0]["tree"]
    assert follower.state(tree)["finished"]
    follower.forget(tree)
    assert follower.trees() == [] and follower.events(tree) == []


# ------------------------------------------------------------------ observers come and go


def test_observers_hold_no_thread_and_no_reference_once_their_trees_end(fake):
    fake(responder=lambda request: XML.format("ok ok ok"))

    @ai
    def echo(secret: str) -> str:
        """Repeat."""

    threads_before = threading.active_count()
    refs = []
    for _ in range(30):
        got = []

        def observer(event, got=got):
            got.append(event)

        refs.append(weakref.ref(observer))
        echo.using(observers=[observer])("x")
        assert functai.flush(5)
        assert got[-1]["kind"] == "done"
        del observer
    gc.collect()
    assert eventlog._feeds == {}
    assert all(r() is None for r in refs)                                     # none is kept alive
    deadline = time.monotonic() + 5
    while threading.active_count() > threads_before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= threads_before


def test_a_list_observer_is_complete_when_the_call_returns(fake):
    fake(responder=lambda request: XML.format("ok ok ok ok ok"))

    @ai
    def echo(secret: str) -> str:
        """Repeat."""

    for _ in range(50):
        seen = []
        echo.using(observers=[seen])("x")
        assert seen and seen[-1]["kind"] == "done"


def test_a_broken_observer_is_let_go(fake):
    fake(responder=lambda request: XML.format("ok"))

    @ai
    def echo(secret: str) -> str:
        """Repeat."""

    def broken(event):
        raise RuntimeError("no")

    ref = weakref.ref(broken)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        echo.using(observers=[broken])("x")
        assert functai.flush(5)
    del broken
    gc.collect()
    assert ref() is None and eventlog._feeds == {}


def test_an_observer_set_by_configure_goes_on_in_a_forked_child(tmp_path):
    script = tmp_path / "fork_child.py"
    script.write_text(textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {str(HERE)!r})
        from conftest import FakeRouter
        import functai
        from functai import ai

        @ai
        def echo(secret: str) -> str:
            """Repeat."""

        functai.configure(lm="gpt-4.1-mini", client=FakeRouter(responder=lambda req: "<result>\\nok\\n</result>"))
        seen = []
        functai.configure(observers=[seen.append])      # a function observer: fed from a thread
        echo("parent"); functai.flush(5)
        pid = os.fork()
        if pid == 0:
            n = len(seen)
            echo("child"); functai.flush(5)
            os._exit(0 if len(seen) - n == 4 else 1)
        _, status = os.waitpid(pid, 0)
        sys.exit(os.waitstatus_to_exitcode(status))
        '''))
    if not hasattr(__import__("os"), "fork"):
        pytest.skip("no fork here")
    out = subprocess.run([sys.executable, "-W", "ignore", str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr


# ------------------------------------------------------------------ journals: scope, time, batches


def test_a_program_s_required_journal_under_the_host_s_same_journal_warns_nothing(fake):
    fake(responder=lambda request: XML.format("ok"))

    @ai
    def echo(secret: str) -> str:
        """Repeat."""

    store = functai.MemoryStore()
    functai.configure(journal=store)

    @module(journal=functai.Journal(store, required=True))
    def outer(x: str) -> str:
        return echo(x)

    @module
    def plain(x: str) -> str:
        return echo(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert outer("hi") == "ok"
        with functai.configure(journal=functai.Journal(store, required=True)):
            assert plain("hi") == "ok"
    assert [str(w.message) for w in caught if "journal" in str(w.message)] == []
    for tree in store.trees():
        assert [e["kind"] for e in store.read(tree)][-1] == "done"


def test_a_barrier_waits_at_most_its_timeout():
    release = threading.Event()

    class Hanging(functai.MemoryStore):
        extend = None

        def append(self, e):
            release.wait(10)
            return super().append(e)

    ran = []

    @module(journal=functai.Journal(Hanging(), required=True, timeout=0.3))
    def effect() -> str:
        ran.append(1)
        return "ran"

    t0 = time.monotonic()
    with pytest.raises(functai.JournalError) as err:
        effect()
    assert time.monotonic() - t0 < 3
    assert ran == [] and err.value.code == "journal-end" and err.value.journal == "unknown"
    assert err.value.outcome.error.code == "journal-barrier"
    release.set()


def test_resends_wait_longer_each_time():
    tries = []

    class Flaky(functai.MemoryStore):
        extend = None

        def append(self, e):
            tries.append(time.monotonic())
            if len(tries) <= 2:
                raise ConnectionError("blip")
            return super().append(e)

    @module(journal=functai.Journal(Flaky(), required=True, retries=2, backoff=0.1))
    def effect() -> str:
        return "ran"

    assert effect() == "ran"
    assert tries[1] - tries[0] >= 0.09 and tries[2] - tries[1] >= 0.19


def test_closing_a_stream_ends_its_wait_at_a_barrier():
    class Hanging(functai.MemoryStore):
        extend = None

        def append(self, e):
            time.sleep(0.2 if e["kind"] != "started" else 5)
            return super().append(e)

    @module(journal=functai.Journal(Hanging(), required=True, timeout=1))
    def effect() -> str:
        return "ran"

    s = effect.stream()
    time.sleep(0.2)
    t0 = time.monotonic()
    s.close()
    with pytest.raises(BaseException) as err:
        s.result
    assert isinstance(err.value, (functai.Cancelled, functai.JournalError))
    if isinstance(err.value, functai.JournalError):
        assert isinstance(err.value.outcome.error, functai.Cancelled)
    assert time.monotonic() - t0 < 3


def test_a_writer_sends_what_waits_as_one_batch(fake):
    """A store with extend is sent every event that waits as one batch."""
    fake(responder=lambda request: XML.format("a long answer, " * 40))

    class Counting(functai.MemoryStore):
        def __init__(self):
            super().__init__()
            self.appends, self.batches = 0, 0

        def append(self, e):
            self.appends += 1
            time.sleep(0.02)
            return super().append(e)

        def extend(self, events):
            self.batches += 1
            time.sleep(0.02)
            return super().extend(events)

    @ai
    def talk(topic: str) -> str:
        """Talk."""

    store = Counting()
    with functai.configure(journal=functai.Journal(store, required=True)):
        s = talk.stream("x")
        s.result
    [tree] = store.trees()
    kept = store.read(tree)
    assert store.finished(tree) and [e["seq"] for e in kept] == list(range(1, len(kept) + 1))
    assert store.appends + store.batches < len(kept)                          # fewer round trips than events
    for e in kept:
        assert_valid(EVENT, e)


# ------------------------------------------------------------------ the schemas FunctAI checks at run time


def test_the_package_carries_the_contract_s_schemas():
    for name in schemas.NAMES:
        mine = json.loads((schemas.FOLDER / f"{name}.schema.json").read_text())
        theirs = json.loads((CONTRACT / "schema" / f"{name}.schema.json").read_text())
        assert mine == theirs, name


def _values(folder_glob):
    found = {name: [] for name in schemas.NAMES}

    def walk(x):
        if isinstance(x, dict):
            if "functai_event" in x or {"kind", "tree", "seq"} <= set(x):
                found["event"].append(x)
            for key, name in (("functai_saved", "saved"), ("functai_call", "call"), ("functai_rating", "rating")):
                if key in x:
                    found[name].append(x)
            if {"description", "inputs", "outputs"} <= set(x) and isinstance(x.get("inputs"), list):
                found["interface"].append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for path in sorted((CONTRACT / "cases").glob(folder_glob)):
        walk(load(path))
    return found


def _mutations(x, depth=0):
    yield x
    if depth > 2:
        return
    if isinstance(x, dict):
        for k in list(x):
            y = dict(x)
            del y[k]
            yield y
            for bad in (None, 1, 1.5, "x", True, [], {}):
                y = dict(x)
                y[k] = bad
                yield y
            for m in list(_mutations(x[k], depth + 1))[:12]:
                y = dict(x)
                y[k] = m
                yield y
        yield {**x, "zz_extra": 1}
    elif isinstance(x, list) and x:
        yield x[1:]
        yield x + [x[0]]
        for m in list(_mutations(x[0], depth + 1))[:12]:
            yield [m] + x[1:]
    elif isinstance(x, str):
        yield x + "\n"
        yield ""
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield 0
        yield -1
        yield x + 0.5
        yield float(x)


def test_the_schema_checker_agrees_with_the_reference_validator():
    """FunctAI's own schema checker and jsonschema (patterns read as ECMA-262
    reads them) say the same of every value in the cases, and of values made
    from them by removing, retyping and changing members."""
    found = _values("**/*.json")
    checked = 0
    for name, values in found.items():
        reference = validator(name)
        for value in values[:30]:
            for m in _mutations(value):
                checked += 1
                assert reference.is_valid(m) == schemas.valid(name, m), (name, schemas.problem(name, m),
                                                                        json.dumps(m)[:300])
    assert checked > 5_000


def test_a_schema_keyword_the_checker_lacks_is_refused_when_read():
    with pytest.raises(schemas.SchemaError):
        schemas._check_keywords({"type": "object", "dependentRequired": {"a": ["b"]}}, "test")
    with pytest.raises(schemas.SchemaError):
        schemas._check_keywords({"properties": {"a": {"unevaluatedProperties": False}}}, "test")


def test_an_object_that_is_no_constraint_puts_no_keyword_in_a_shape():
    import re
    from typing import Annotated
    from functai.signature import _constraints
    assert _constraints(Annotated[str, re.compile("x")]) == {}
    from pydantic import Field
    assert _constraints(Annotated[int, Field(ge=3)]) == {"minimum": 3}
    _ = _ai
    _ = lm15


def test_a_turn_made_for_another_plan_is_recorded_as_shown(tmp_path):
    """A turn made for another signature (reasoning added since) is shown as an example of its values, without its steps: the record says so
    (``{"call"}``, not ``"steps"``), and what was sent agrees."""
    router = FakeRouter(responder=lambda request: XML.format("ok"))
    functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path)

    @ai(stateful=True)
    def chat(message: str) -> str:
        """Answer."""

    first = chat.predict("FIRST-MESSAGE")
    chat.predict("SECOND")
    second = max(records(tmp_path), key=lambda r: r["id"])
    assert second["saw"] == [{"call": first.call_id, "steps": True}]          # the same plan: whole, with steps
    chat.module = "cot"                                                      # another signature from now on
    router.responder = lambda request: "<reasoning>\nhm\n</reasoning>\n" + XML.format("ok")
    chat.predict("THIRD")
    third = max(records(tmp_path), key=lambda r: r["id"])
    assert third["saw"] == [{"call": first.call_id}, {"call": second["id"]}]  # shown as values, not as steps
    assert "FIRST-MESSAGE" in str(router.requests[-1])


# ------------------------------------------------------------------ round two: what the second reviews found


REPLY = '''
    from functai import ai

    @ai
    def reply(message: str, tone: str = "kind") -> str:
        """Answer the customer."""
        ...
    '''


@pytest.mark.parametrize("hashes", ["none", "fewer", "more"])
def test_a_python_folder_s_probes_each_have_their_fingerprint_before_its_code_runs(hashes, tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "replyprobe.py").write_text(textwrap.dedent(REPLY))
    monkeypatch.syspath_prepend(str(src))
    import replyprobe
    folder = tmp_path / "saved"
    functai.save(replyprobe.reply, folder, examples=[{"message": "a"}, {"message": "b", "tone": "c"}])
    path = folder / "functai.json"
    m = json.loads(path.read_text())
    data = m["nodes"][m["entry"]]["ai"]
    assert len(data["probes"]) >= 2
    requests = data["fingerprints"]["requests"]
    data["fingerprints"]["requests"] = {"none": [], "fewer": requests[:1], "more": requests * 3}[hashes]
    # a demo no fingerprint vouches for
    data["state"]["demos"] = [{"inputs": {"message": "UNVERIFIED", "tone": "kind"}, "outputs": {"result": "x"}}]
    path.write_text(json.dumps(m))
    before = set(sys.modules)
    with pytest.raises(functai.LoadRefused) as err:
        functai.load(folder, trust=True)
    assert err.value.code == "saved-malformed"
    assert not any(name.startswith("_functai_saved_") for name in set(sys.modules) - before)   # nothing ran
    with pytest.raises(functai.LoadRefused) as err:
        saved.from_manifest(m)
    assert err.value.code == "saved-malformed"


def _chat_with_a_turn(tmp_path):
    router = FakeRouter(responder=lambda request: XML.format("ok"))
    functai.configure(client=router, lm="gpt-4.1-mini", log_calls=tmp_path)

    @ai(stateful=True)
    def chat(message: str) -> str:
        """Answer."""

    first = chat.predict("ORIGINAL")
    return router, chat, first


@pytest.mark.parametrize("edit", ["input", "output", "step"])
def test_a_turn_changed_in_place_is_no_longer_its_call_s(edit, tmp_path):
    router, chat, first = _chat_with_a_turn(tmp_path)
    turn = chat.history[0]
    if edit == "input":
        turn.inputs["message"] = "ALTERED"
    elif edit == "output":
        turn.outputs["result"] = "ALTERED"
    else:
        turn.steps[0].outputs["result"] = "ALTERED"
    chat("second")
    last = max(records(tmp_path), key=lambda r: r["id"])
    assert last["saw"] == [{"unrecorded": True}]                   # the turn is not what the first call made
    if edit != "output":                                           # (a turn with steps is shown by its steps)
        assert "ALTERED" in str(router.requests[-1])
    assert next(r for r in records(tmp_path) if r["id"] == first.call_id)["inputs"] == {"message": "ORIGINAL"}


def test_what_a_call_is_shown_is_copied_when_it_is_prepared(tmp_path):
    router, chat, first = _chat_with_a_turn(tmp_path)

    class Gate(functai.MemoryStore):
        def __init__(self):
            super().__init__()
            self.arrived, self.allow = threading.Event(), threading.Event()

        def append(self, e):
            if e["kind"] == "started":
                self.arrived.set()
                assert self.allow.wait(5)
            return super().append(e)

    gate = Gate()
    with functai.configure(journal=functai.Journal(gate, required=True)):
        s = chat.stream("SECOND")
        assert gate.arrived.wait(5)
        chat.history[0].inputs["message"] = "ALTERED-AFTER-START"      # after the call was prepared
        gate.allow.set()
        assert s.result == "ok"
    sent = str(router.requests[-1])
    assert "ORIGINAL" in sent and "ALTERED-AFTER-START" not in sent
    started = gate.read(gate.trees()[0])[0]
    assert started["saw"] == [{"call": first.call_id, "steps": True}]
    last = max(records(tmp_path), key=lambda r: r["id"])
    assert last["saw"] == [{"call": first.call_id, "steps": True}]


class _Text(str):
    """Text as JSON, holding a host value that cannot be copied."""

    def __new__(cls, value):
        text = super().__new__(cls, value)
        text.lock = threading.Lock()
        return text


def test_a_turn_that_cannot_be_copied_is_shown_as_its_json_form(tmp_path):
    router, chat, first = _chat_with_a_turn(tmp_path)
    chat.history[0].inputs["message"] = _Text("ORIGINAL")        # the same JSON as the first call recorded
    with pytest.raises(TypeError):
        copy.deepcopy(chat.history[0])

    class Gate(functai.MemoryStore):
        def __init__(self):
            super().__init__()
            self.arrived, self.allow = threading.Event(), threading.Event()

        def append(self, e):
            if e["kind"] == "started":
                self.arrived.set()
                assert self.allow.wait(5)
            return super().append(e)

    gate = Gate()
    with functai.configure(journal=functai.Journal(gate, required=True)):
        s = chat.stream("SECOND")
        assert gate.arrived.wait(5)
        chat.history[0].inputs["message"] = "ALTERED-AFTER-START"      # after the call was prepared
        gate.allow.set()
        assert s.result == "ok"
    sent = str(router.requests[-1])
    assert "ORIGINAL" in sent and "ALTERED-AFTER-START" not in sent
    assert gate.read(gate.trees()[0])[0]["saw"] == [{"call": first.call_id, "steps": True}]


def test_a_turn_edited_while_it_is_copied_is_not_its_call_s(tmp_path, monkeypatch):
    """What is compared with the recorded turn is the copy the call is shown,
    not the live turn read a moment before or after."""
    from functai import core
    router, chat, first = _chat_with_a_turn(tmp_path)
    frozen = core._frozen

    def edited_meanwhile(turn):
        turn.inputs["message"] = "ALTERED"                           # another thread, just before the copy
        return frozen(turn)

    monkeypatch.setattr(core, "_frozen", edited_meanwhile)
    chat("second")
    last = max(records(tmp_path), key=lambda r: r["id"])
    assert last["saw"] == [{"unrecorded": True}]
    assert "ALTERED" in str(router.requests[-1])


def test_memory_stores_are_known_by_identity(fake, tmp_path):
    """A subclass need not be hashable; two equal stores are two stores, each
    given fresh locks in a forked child; a store let go leaves nothing behind."""
    fake(responder=lambda request: XML.format("ok"))

    @dataclasses.dataclass
    class Transport(functai.MemoryStore):                # unhashable: a mutable dataclass
        label: str = "test"

        def __post_init__(self):
            super().__init__()

    @module
    def echo(x: str) -> str:
        return x

    store = Transport()
    with functai.configure(journal=functai.Journal(store, required=True)):
        assert echo("ok") == "ok"
    [tree] = store.trees()
    assert [e["kind"] for e in store.read(tree)] == ["started", "done"]
    ref = weakref.ref(store)
    del store
    assert functai.flush(5)
    deadline = time.monotonic() + 5                    # its writer's thread lets go as it ends
    while ref() is not None and time.monotonic() < deadline:
        time.sleep(0.02)
        gc.collect()
    assert ref() is None and all(r() is not None for r in eventlog._memory_stores.values())

    if not hasattr(__import__("os"), "fork"):
        return
    script = tmp_path / "fork_stores.py"
    script.write_text(textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {str(HERE)!r})
        import functai

        class Same(functai.MemoryStore):                 # every one equal to every other, one hash
            def __eq__(self, other):
                return isinstance(other, Same)

            def __hash__(self):
                return 1

        a, b = Same(), Same()
        a._lock.acquire(); b._lock.acquire()             # held by the parent when it forks
        pid = os.fork()
        if pid == 0:
            os._exit(0 if not a._lock.locked() and not b._lock.locked() else 1)
        _, status = os.waitpid(pid, 0)
        print("CHILD", os.waitstatus_to_exitcode(status))
        '''))
    out = subprocess.run([sys.executable, "-W", "ignore", str(script)], capture_output=True, text=True, timeout=60)
    assert "CHILD 0" in out.stdout, out.stdout + out.stderr


def test_a_store_that_overrides_only_extend_is_sent_every_event_through_it(fake):
    fake(responder=lambda request: XML.format("word " * 40))

    class Durable(functai.MemoryStore):
        def __init__(self):
            super().__init__()
            self.durable = []

        def extend(self, events):
            time.sleep(0.005)
            self.durable.extend(e["seq"] for e in events)
            return super().extend(events)

    @ai
    def say(x: str) -> str:
        """Say something long."""

    store = Durable()
    with functai.configure(journal=functai.Journal(store, required=True)):
        say.stream("hi").result
    [tree] = store.trees()
    assert [e["seq"] for e in store.read(tree)] == store.durable


def test_a_call_made_in_a_process_forked_inside_a_call_starts_its_own_tree(tmp_path):
    """Forked while an outer call runs (its observer busy on a thread), the
    child's calls start a tree of their own there, seen by the outer call's
    observers and kept in its required journal; the outer call's tree is the
    parent's, and the child writes nothing of it, even when the outer call
    ends in the child too."""
    if not hasattr(__import__("os"), "fork"):
        pytest.skip("no fork here")
    script = tmp_path / "fork_inside.py"
    script.write_text(textwrap.dedent(f'''
        import json, os, sys, threading, warnings
        sys.path.insert(0, {str(HERE)!r})
        from conftest import FakeRouter
        import functai
        from functai import ai, module, calllog

        functai.configure(lm="gpt-4.1-mini", client=FakeRouter(responder=lambda req: "<result>\\nok\\n</result>"))
        store = functai.MemoryStore()
        seen, busy, go = [], threading.Event(), threading.Event()

        def observer(e):
            if e["kind"] == "started" and e["function"] == "outer":
                busy.set()
                go.wait(5)
            seen.append((e["kind"], e["function"]))

        @ai
        def inner(x: str) -> str:
            """Echo."""

        @module(observers=[observer], journal=functai.Journal(store, required=True, timeout=5))
        def outer() -> str:
            assert busy.wait(5)                       # the observer's thread is mid-event
            read, write = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read)
                n = len(seen)
                value = inner("child")
                ok = functai.flush(5)
                mine = calllog.current().log.tree
                trees = [t for t in store.trees() if t != mine]
                report = {{"value": value, "flush": ok, "seen": seen[n:],
                          "trees": [[e["kind"] for e in store.read(t)] for t in trees],
                          "finished": [store.finished(t) for t in trees]}}
                os.write(write, json.dumps(report).encode())
                os.close(write)
                return "child"                          # the outer call ends in the child too
            os.close(write)
            data = b""
            while True:
                chunk = os.read(read, 65536)
                if not chunk:
                    break
                data += chunk
            os.waitpid(pid, 0)
            go.set()
            print("CHILD", data.decode(), flush=True)
            return "parent"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = outer()
        if got == "child":
            outer_tree = [t for t in store.trees() if store.read(t)[0]["function"] == "outer"]
            print("CHILD-END", json.dumps({{"warned": any("ends in a process forked" in str(w.message)
                                                          for w in caught),
                                           "outer_events": [[e["kind"] for e in store.read(t)]
                                                            for t in outer_tree]}}), flush=True)
            os._exit(0)
        functai.flush(5)
        [tree] = store.trees()
        print("PARENT", json.dumps({{"events": [e["kind"] for e in store.read(tree)],
                                    "finished": store.finished(tree)}}), flush=True)
        '''))
    out = subprocess.run([sys.executable, "-W", "ignore", str(script)], capture_output=True, text=True, timeout=120)
    lines = {line.split(" ", 1)[0]: json.loads(line.split(" ", 1)[1]) for line in out.stdout.splitlines()
             if line.split(" ", 1)[0] in ("CHILD", "CHILD-END", "PARENT")}
    assert set(lines) == {"CHILD", "CHILD-END", "PARENT"}, out.stdout + out.stderr
    child = lines["CHILD"]
    assert child["value"] == "ok" and child["flush"] is True
    assert child["seen"][0] == ["started", "inner"] and child["seen"][-1] == ["done", "inner"]
    assert child["trees"] and child["trees"][0][0] == "started" and child["trees"][0][-1] == "done"
    assert child["finished"] == [True]
    assert lines["CHILD-END"] == {"warned": True, "outer_events": [["started"]]}   # the child ended nothing of it
    assert lines["PARENT"] == {"events": ["started", "done"], "finished": True}


def test_a_broken_observer_is_let_go_whatever_its_kind(fake):
    import operator
    fake(responder=lambda request: XML.format("ok"))

    @ai
    def echo(secret: str) -> str:
        """Repeat."""

    @dataclasses.dataclass
    class Recorder:                                      # callable, weakly referable, and unhashable
        events: list = dataclasses.field(default_factory=list)

        def __call__(self, event):
            self.events.append(event)
            raise RuntimeError("broken")

    class Refusing(list):                                # a list subclass: fed from a thread, by its append
        def append(self, event):
            raise RuntimeError("broken")

    refs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for make in (Recorder, Refusing) * 3:
            observer = make()
            refs.append(weakref.ref(observer))
            echo.using(observers=[observer])("x")
            assert functai.flush(5)
            del observer
        # one that cannot be weakly referred to: broken while its feed lives, then held by nothing
        getter = operator.itemgetter("no such key")
        echo.using(observers=[getter])("x")
        assert functai.flush(5)
    gc.collect()
    assert all(r() is None for r in refs)
    assert eventlog._broken == {} and eventlog._feeds == {}
    assert sys.getrefcount(getter) == 2                  # this name, and getrefcount's argument


def test_a_store_that_overrides_only_append_is_sent_every_event_through_it(fake):
    fake(responder=lambda request: XML.format("word " * 40))

    class Durable(functai.MemoryStore):
        def __init__(self):
            super().__init__()
            self.durable = []

        def append(self, e):
            time.sleep(0.005)
            self.durable.append(e["seq"])
            return super().append(e)

    @ai
    def say(x: str) -> str:
        """Say something long."""

    store = Durable()
    with functai.configure(journal=functai.Journal(store, required=True)):
        say.stream("hi").result
    [tree] = store.trees()
    assert [e["seq"] for e in store.read(tree)] == store.durable

    class Both(functai.MemoryStore):
        def append(self, e):
            return super().append(e)

        def extend(self, events):
            return super().extend(events)

    class Listish(list):
        def append(self, e):
            return "kept"

        def read(self, tree, after=None):
            return []

    class Says(Durable):
        batches = True

    assert eventlog.takes_batches(functai.MemoryStore()) and eventlog.takes_batches(Both())
    assert not eventlog.takes_batches(Durable()) and not eventlog.takes_batches(Listish())
    assert eventlog.takes_batches(Says())
    off = functai.MemoryStore()
    off.batches = False
    assert not eventlog.takes_batches(off)


def test_a_tool_whose_call_an_append_only_store_does_not_keep_does_not_run(fake):
    """streaming.md, journal-08: through a store whose own append is down for
    the tool call, the tool does not run."""
    ran = []

    def lookup_order(order: str) -> str:
        """Where an order is."""
        ran.append(order)
        return "Leeds"

    @ai(tools=[lookup_order])
    def helper(question: str) -> str:
        """Help."""

    class Down(functai.MemoryStore):
        def append(self, e):
            if e["kind"] == "tool_call":
                raise ConnectionError("down")
            return super().append(e)

    fake([lm15.TextPart("Let me check."), lm15.ToolCallPart(id="c1", name="lookup_order", input={"order": "B-2210"})],
         XML.format("It is in Leeds."))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with functai.configure(journal=functai.Journal(Down(), required=True, backoff=0, timeout=5)):
            with pytest.raises(functai.JournalError) as err:
                helper.stream("Where is B-2210?").result
    assert ran == []
    assert err.value.code == "journal-end" and err.value.outcome.error.code == "journal-barrier"


def test_a_call_after_its_tree_ended_is_given_to_no_receiver(fake):
    import contextvars
    fake(responder=lambda request: XML.format("ok"))
    seen_list, seen_fn = [], []
    store = functai.MemoryStore()

    @module
    def inner(x: str) -> str:
        return x

    done = threading.Event()

    @module(observers=[seen_list, lambda e: seen_fn.append(e["function"])], journal=store)
    def outer(x: str) -> str:
        ctx = contextvars.copy_context()

        def later():
            time.sleep(0.2)
            ctx.run(inner, x)
            done.set()

        threading.Thread(target=later).start()
        return x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert outer("a") == "a"
        assert done.wait(5) and functai.flush(5)
    assert [e["function"] for e in seen_list] == ["outer", "outer"] and seen_fn == ["outer", "outer"]
    assert [e["function"] for e in store.read(store.trees()[0])] == ["outer", "outer"]
    assert any("after its call tree ended" in str(w.message) for w in caught)


def test_a_positional_only_input_is_refused_where_it_is_defined():
    with pytest.raises(TypeError, match="positional-only"):
        @module
        def po(x: str, /) -> str:
            return x

    with pytest.raises(TypeError, match="positional-only"):
        @ai
        def po_ai(x: str, /) -> str:
            """Echo."""


def test_a_program_naming_its_host_s_journal_again_waits_as_the_host_says(fake):
    fake(responder=lambda request: XML.format("ok"))
    store = functai.MemoryStore()
    host = functai.Journal(store, required=True, timeout=0.5, retries=0)
    mine = functai.Journal(store, required=True, timeout=None, retries=5)
    layers = [("own", {"journal": mine}), ("configure", {"journal": host})]
    assert eventlog.receivers(layers).journal is host
    functai.configure(journal=host)

    @module(journal=mine)
    def effect() -> float:
        return calllog.current().log.journal.timeout

    assert effect() == 0.5
