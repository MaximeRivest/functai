"""contract/cases/events: a call tree's events as data (contract/streaming.md,
format 2): replaying and following them, their kept form, the rules a store
keeps, a writer keeping a log in a journal that fails, and which receivers a
tree gets from the layers of settings around it."""


import json

import pytest

import functai
from functai import ai, calllog, eventlog, module
from functai.errors import EventRefused, JournalError
from functai.eventlog import Follower, Journal, MemoryStore, position
from contract_support import assert_valid, case_files, load, validator

EVENT = validator("event")


def cases(kind):
    return case_files("events", kind + "-")


def positions(events):
    return [position(e) for e in events]


def answer(read):
    """A read's result as the cases say it: the positions given, or the refusal."""
    if isinstance(read, EventRefused):
        return {"refuses": read.code}
    return {"events": positions(read)}


# ------------------------------------------------------------------ replay


@pytest.mark.parametrize("path", cases("replay"), ids=lambda p: p.stem)
def test_replay_case(path):
    case = load(path)
    for e in case["events"]:
        assert_valid(EVENT, e, path.stem)
    r = eventlog.Replay()
    for e in case["events"]:
        r.apply(e)
    assert eventlog.replay(case["events"]) == case["expect"]["views"]
    assert r.finished == case["expect"]["finished"]
    for x in case["resume"]:
        try:
            got = answer(eventlog.read_after(case["events"], x["after"]))
        except EventRefused as exc:
            got = answer(exc)
        assert got == x["expect"], x


# ------------------------------------------------------------------ follow


class Source:
    """What a store (``from == "store"``) or a writer's process gives, in the reader's form."""

    def __init__(self, events, writer):
        self.events, self.writer = events, writer

    def read(self, tree, after):
        return eventlog.read_after([e for e in self.events if e["tree"] == tree], after)


@pytest.mark.parametrize("path", cases("follow"), ids=lambda p: p.stem)
def test_follow_case(path):
    case = load(path)
    recover = case.get("recover")
    reader = Follower(live=bool(recover) and recover["reader"] == "live")
    results = []
    for e in case["received"]:
        got = reader.receive(e)
        results.append(got)
        if got == "unknown-format":
            break
    expect = case["expect"]
    assert results == expect["results"]
    assert {t: reader.state(t) for t in reader.trees()} == expect["state"]
    if recover is not None:
        [tree] = reader.trees()
        source = Source(recover["source"], None if recover["from"] == "store" else recover["from"])
        reads = [{"after": after, "expect": answer(got)} for after, got in reader.resume(tree, source)]
        assert reads == expect["recover"]["reads"]
        assert reader.state(tree) == expect["recover"]["state"]


# ------------------------------------------------------------------ the kept form


@pytest.mark.parametrize("path", cases("kept"), ids=lambda p: p.stem)
def test_kept_case(path):
    case = load(path)
    kept = eventlog.kept_form(case["events"], case["kept"])
    for e in kept:
        assert_valid(EVENT, e, path.stem)
    assert kept == case["expect"]["events"]


# ------------------------------------------------------------------ stores


@pytest.mark.parametrize("path", cases("store"), ids=lambda p: p.stem)
def test_store_case(path):
    case = load(path)
    store = MemoryStore()
    for step in case["steps"]:
        try:
            if "append" in step:
                got = store.append(step["append"])
            elif "batch" in step:
                got = store.extend(step["batch"])
            else:
                got = store.claim(step["claim"])
        except EventRefused as exc:
            if "append" in step:
                got = exc.code
            elif "batch" in step:
                got = {"refuses": exc.code, "event": exc.event}
            else:
                got = {"refuses": exc.code}
        assert got == step["expect"], step
    for r in case["reads"]:
        try:
            got = answer(store.read(r["tree"], r["after"]))
        except EventRefused as exc:
            got = answer(exc)
        assert got == r["expect"], r
    logs = {t: {"events": positions(store.read(t)), "writer": store.writer_of(t), "finished": store.finished(t)}
            for t in store.trees()}
    assert logs == case["expect"]["logs"]


# ------------------------------------------------------------------ journals


class ScriptedStore:
    """A MemoryStore reached through a transport that fails as a case's script
    says: each append meets the next word ("ok", "lost", "down", "conflict");
    between appends, another writer may claim the log, or claim and end it."""

    def __init__(self, script, function):
        self.store = MemoryStore()
        self.script = iter(script)
        self.trace = []
        self.function = function
        self.tree = None

    def _word(self):
        word = next(self.script, "ok")
        while word in ("claimed", "ended"):
            self._other(word)
            word = next(self.script, "ok")
        return word

    def _other(self, word):
        try:
            claim = self.store.claim(self.tree)
        except EventRefused:
            claim = {}
        self.trace.append({"other": word, "writer": claim.get("writer")})
        if word == "ended" and claim:
            last = self.store.read(self.tree)[-1]
            self.store.append({"functai_event": 2, "kind": "failed", "tree": self.tree, "writer": claim["writer"],
                               "seq": claim["after"]["seq"] + 1, "after": claim["after"], "at": last["at"],
                               "call": self.tree, "function": self.function, "error": {"type": "Cancelled"}})

    def append(self, event):
        self.tree = event["tree"]
        word = self._word()
        if word == "down":
            self.trace.append({"seq": event["seq"], "transport": word, "answer": None})
            raise ConnectionError("the journal is down")
        if word == "conflict":
            self.trace.append({"seq": event["seq"], "transport": word, "answer": "event-conflict"})
            raise EventRefused("event-conflict")
        try:
            got = self.store.append(event)
        except EventRefused as exc:
            got, refused = exc.code, exc
        else:
            refused = None
        self.trace.append({"seq": event["seq"], "transport": word, "answer": None if word == "lost" else got})
        if word == "lost":
            raise TimeoutError("the answer was lost")
        if refused is not None:
            raise refused
        return got

    def read(self, tree, after=None):
        return self.store.read(tree, after)

    def claim(self, tree):
        return self.store.claim(tree)


def without_message(error):
    return {k: v for k, v in error.items() if k != "message"}


def scripted(case, ran):
    """The case's AI function, as a program whose code makes the case's events
    through the call's own API, in the order the engine makes them (a request,
    thinking, pieces of text, a tool call and then the barrier before the tool
    runs, a tool result, a retry), then returns the case's value or raises its
    error. Everything else is the library's, on its real path: numbering, the
    kept form, the journal's writer and its thread, the barriers, the failed
    event of a stopped call, what readers are shown, the JournalError, the
    call record."""
    events = case["events"]
    first, end = events[0], events[-1]
    iface = {"description": "", "inputs": [{"name": k, "shape": {}} for k in first["inputs"]],
             "outputs": [{"name": "result", "shape": {}}]}

    def body(**inputs):
        call = calllog.current()
        for e in events[1:-1]:
            kind = e["kind"]
            if kind == "request":
                call.request(e["model"])
            elif kind == "tool_call":
                made = call.emit("tool_call", id=e["id"], name=e["name"], input=e["input"])
                calllog.tool_barrier(made.seq if made is not None else None)
                ran.append(e["id"])                                   # the tool runs here
            elif kind == "tool_result":
                call.emit("tool_result", id=e["id"], name=e["name"], output=e["output"])
            elif kind == "text":
                call.emit("text", field=e["field"], answer=e["answer"], text=e["text"])
            elif kind == "thinking":
                call.emit("thinking", text=e["text"])
            elif kind == "retry":
                call.emit("retry", reason=e["reason"], wait=e["wait"])
            else:
                raise AssertionError(f"a kind this harness does not script: {kind}")
        if end["kind"] == "done":
            return end["value"]
        raise type(end["error"]["type"], (Exception,), {})(end["error"]["message"])

    body.__name__ = body.__qualname__ = first["function"]
    return module(interface=iface)(body)


def normalized(made, case):
    """The events the writer made, with what a real run cannot share with the
    case (ids, the clock, the program's hashes) put back to the case's."""
    first = case["events"][0]
    log = case["expect"]["log"]
    out = []
    for i, e in enumerate(made):
        e = dict(e)
        e["tree"] = e["call"] = first["tree"]
        if e["kind"] == "started":
            e["root"] = first["root"]
            e["program"] = first["program"]
        if i < len(log):
            e["at"] = log[i]["at"]
        out.append(e)
    return out


@pytest.mark.parametrize("path", cases("journal"), ids=lambda p: p.stem)
def test_journal_case(path, tmp_path, monkeypatch):
    """A real call, streamed, with the case's journal (a MemoryStore reached
    through the case's script) and the call log on: what the writer made, what
    it sent and the store answered, what readers were shown, what the store
    keeps, what the caller got, what the record says and what settling says
    all come from the library."""
    case = load(path)
    required = case["mode"] == "required"
    made = []
    real_emit = eventlog.TreeLog.emit

    def spy(self, call, kind, **fields):
        event = real_emit(self, call, kind, **fields)
        if event is not None:
            made.append(event.to_dict())
        return event

    monkeypatch.setattr(eventlog.TreeLog, "emit", spy)
    store = ScriptedStore(case["script"], case["events"][0]["function"])
    ran, seen = [], []
    program = scripted(case, ran)
    journal = Journal(store, required=required, retries=case["retries"], backoff=0)
    with functai.configure(journal=journal, observers=[seen], log_calls=tmp_path):
        s = program.stream(**case["events"][0]["inputs"])
        try:
            got = {"returns": s.result}
        except JournalError as err:
            raised = err
            got = None
        except Exception as err:  # noqa: BLE001 — the call's own error, raised as it is
            raised = err
            got = None
        shown = []
        try:
            for e in s.events():
                shown.append(e.seq)
        except BaseException:  # noqa: BLE001 — the stream ends with the call's error
            pass
    assert functai.flush(10)
    expect = case["expect"]
    tree = s.call_id
    assert normalized(made, case) == expect["log"]
    for e in made:
        assert_valid(EVENT, e, path.stem)
    assert store.trace == expect["trace"]
    assert shown == expect["shown"]
    assert [e["seq"] for e in seen] == expect["shown"]                   # an observer is shown the same
    kept = store.store
    assert {"events": positions(kept.read(tree)) if tree in kept.trees() else [],
            "finished": kept.finished(tree)} == expect["kept"]
    if got is None:
        if isinstance(raised, JournalError) and raised.code == "journal-end":
            o = raised.outcome
            outcome = {"failed": without_message(calllog._error(o.error))} if o.failed else {"done": o.value}
            got = {"raises": {"type": "JournalError", "code": "journal-end", "journal": raised.journal,
                              "event": raised.event, "outcome": outcome}}
            if raised.journal == "unknown":
                assert raised.settle() == expect["settled"]
        else:
            got = {"raises": without_message(calllog._error(raised))}
    assert got == expect["caller"]
    [rec] = [json.loads(line) for f in tmp_path.rglob("*.jsonl") for line in f.read_text().splitlines()]
    record = {"error": None if rec["error"] is None else without_message(rec["error"])}
    if "journal" in rec:
        record["journal"] = rec["journal"]
    assert record == expect["record"]
    tool_calls = [e for e in expect["log"] if e["kind"] == "tool_call"]
    ran_tool = any(e["kind"] == "tool_result" for e in expect["log"])
    assert bool(ran) == ran_tool and (not ran or len(tool_calls) == len(ran))    # a tool runs only past its barrier


# ------------------------------------------------------------------ receivers


@pytest.mark.parametrize("path", cases("receivers"), ids=lambda p: p.stem)
def test_receivers_case(path):
    """Each scenario's layers set for real: the program's own settings, a
    ``with configure(...)`` block, ``configure(...)``; which receivers the
    tree gets, and for a refused one, that calling it raises journal-policy
    before it runs, its started and failed going to every observer and to the
    host's journal."""
    case = load(path)
    for n, scenario in enumerate(case["scenarios"]):
        stores, observers = {}, {}

        def journal(spec):
            if spec is None:
                return False
            store = stores.setdefault(spec["name"], MemoryStore())
            return Journal(store, required=spec["mode"] == "required")

        def settings(layer):
            out = {}
            if "observers" in layer:
                out["observers"] = [observers.setdefault(o, []) for o in layer["observers"]]
            if "journal" in layer:
                out["journal"] = journal(layer["journal"])
            if "program_observers" in layer:
                out["program_observers"] = layer["program_observers"]
            return out

        layers = {x["where"]: settings(x) for x in scenario["layers"]}
        functai.config._GLOBAL.clear()
        if "configure" in layers:
            functai.configure(**layers["configure"])

        @ai(**layers.get("own", {}))
        def answer(message: str) -> str:
            """Answer."""

        def names_of(got):
            by_id = {id(v): k for k, v in observers.items()}
            j = None if got.journal is None else {"name": next(k for k, v in stores.items() if v is got.journal.store),
                                                  "mode": got.journal.mode}
            return [by_id[id(o)] for o in got.observers], j

        with functai.configure(**layers.get("block", {})):
            got = eventlog.receivers(calllog._layers(answer))
            obs, j = names_of(got)
            expect = scenario["expect"]
            if "refuses" in expect:
                assert got.refused is not None and got.refused.code == expect["refuses"], n
                assert {"observers": obs, "journal": j} == {"observers": expect["observers"],
                                                            "journal": expect["journal"]}, n
                with pytest.raises(JournalError) as err:
                    answer("Hi")
                assert err.value.code == "journal-policy"
                eventlog.drain()
                kinds = ["started", "failed"]
                for name in expect["observers"]:
                    assert [e["kind"] for e in observers[name]] == kinds, (n, name)
                if expect["journal"] is not None:
                    host = stores[expect["journal"]["name"]]
                    [tree] = host.trees()
                    kept = host.read(tree)
                    assert [e["kind"] for e in kept] == kinds and kept[-1]["error"]["code"] == "journal-policy"
            else:
                assert got.refused is None, n
                assert {"observers": obs, "journal": j} == expect, n
    functai.config._GLOBAL.clear()


def test_the_contract_has_event_cases():
    assert len(case_files("events")) >= 67
