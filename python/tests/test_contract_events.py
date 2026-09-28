"""contract/cases/events: a call tree's events as data (contract/streaming.md,
format 2): replaying and following them, their kept form, the rules a store
keeps, a writer keeping a log in a journal that fails, and which receivers a
tree gets from the layers of settings around it."""

import copy
import json

import pytest

import functai
from functai import ai, calllog, eventlog
from functai.errors import EventRefused, JournalError
from functai.eventlog import Follower, Journal, JournalWriter, MemoryStore, position
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


def later(at):
    """The writer's clock, one step (10 ms, as the cases' writer) after ``at``."""
    import datetime as dt
    t = dt.datetime.strptime(at, "%Y-%m-%dT%H:%M:%S.%fZ") + dt.timedelta(milliseconds=10)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def without_message(error):
    return {k: v for k, v in error.items() if k != "message"}


@pytest.mark.parametrize("path", cases("journal"), ids=lambda p: p.stem)
def test_journal_case(path):
    """The writer's side is functai's JournalWriter; the call's side (barriers,
    what readers are shown, what the caller gets) is what calllog.run does."""
    case = load(path)
    events, required = case["events"], case["mode"] == "required"
    tree = events[0]["tree"]
    store = ScriptedStore(case["script"], events[0]["function"])
    writer = JournalWriter(Journal(store, required=required, retries=case["retries"]), thread=False)
    made, shown, outcome = [], [], None
    for i, e in enumerate(events):
        if e["call"] == tree and e["kind"] in ("done", "failed"):
            outcome = e
            break
        made.append(e)
        shown.append(e["seq"])
        writer.send(e)
        if required and (i == 0 or e["kind"] == "tool_call") and writer.barrier() != "confirmed":
            stopped = calllog.barrier_error(e["seq"])
            outcome = {k: e[k] for k in ("functai_event", "tree", "writer", "call", "function")}
            outcome.update(kind="failed", seq=e["seq"] + 1, after=position(e),
                           at=later(e["at"]), error=calllog._error(stopped))
            outcome = {k: outcome[k] for k in ("functai_event", "kind", "tree", "writer", "seq", "after", "at", "call",
                                               "function", "error")}
            break
    made.append(outcome)
    status = writer.end(outcome)
    if not required or status == "confirmed":
        shown.append(outcome["seq"])
    expect = case["expect"]
    assert made == expect["log"]
    assert store.trace == expect["trace"]
    assert shown == expect["shown"]
    kept = store.store
    assert {"events": positions(kept.read(tree)) if tree in kept.trees() else [],
            "finished": kept.finished(tree)} == expect["kept"]
    failed = None if outcome["kind"] == "done" else without_message(outcome["error"])
    record = {"error": failed}
    if required and status != "confirmed":
        journal = "refused" if status == "refused" else "unknown"
        record["journal"] = journal
        done = {"done": outcome["value"]} if failed is None else {"failed": failed}
        caller = {"raises": {"type": "JournalError", "code": "journal-end", "journal": journal,
                             "event": position(outcome), "outcome": done}}
        if journal == "unknown":
            assert eventlog.settle(store, tree, position(outcome)) == expect["settled"]
    else:
        caller = {"returns": outcome["value"]} if failed is None else {"raises": failed}
    assert caller == expect["caller"]
    assert record == expect["record"]


# ------------------------------------------------------------------ receivers


def named(name):
    def observer(event):
        return None
    observer.__qualname__ = name
    return observer


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
