"""A call tree's log of events: numbered as they happen, given to readers in
the form each may see, kept in a journal, followed and resumed elsewhere
(contract/streaming.md, format 2).

    seen = []
    functai.configure(observers=[seen])                # the kept form of every event, best effort
    store = functai.MemoryStore()
    functai.configure(journal=functai.Journal(store, required=True))   # whole trees, confirmed

    follower = functai.Follower()                        # another process, reading a store
    for event in store.read(tree):
        follower.receive(event)
    follower.state(tree)                                 # what a live watcher saw

One log per call tree: every event has the tree's id (its outermost call's),
the writer that numbered it and its ``seq`` (together, its position), the
position of the event before it in the form being read (``after``) and when
it was numbered (``at``). The process running a tree gives any form of it;
a store keeps the kept form, which follows each call's ``log_content``.

What is here:

- ``TreeLog``: the writer of one tree in this process (numbering, handing
  each event to the streams, observers and journal that see it);
- the kept form (``kept_form``, ``kept_event``) and ``relink``;
- replaying (``Replay``, ``replay``) and following (``Follower``);
- the rules a store keeps (``MemoryStore``, ``read_after``) and settling an
  end a journal did not confirm (``settle``);
- journals (``Journal``, ``JournalWriter``) and which receivers a tree gets
  from the layers of settings around it (``receivers``).
"""

from __future__ import annotations

import atexit
import copy
import dataclasses
import queue
import threading
import time
import warnings
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .errors import EventRefused, JournalError

FORMAT = 2
KINDS = ("started", "request", "text", "thinking", "tool_call", "tool_result", "retry", "done", "failed")
ENVELOPE = ("functai_event", "kind", "tree", "writer", "seq", "after", "at", "call", "function")
KEYS = {"started": ("parent", "root", "program", "inputs", "content", "omitted", "saw"),
        "request": ("request", "model"), "text": ("field", "answer", "text"), "thinking": ("text",),
        "tool_call": ("id", "name", "input", "content"), "tool_result": ("id", "name", "output", "content"),
        "retry": ("reason", "wait", "content"), "done": ("value", "content"), "failed": ("error", "content")}
ERROR_KEYS = ("type", "message", "code")
PROGRAM_KEYS = ("name", "kind", "module", "version", "signature", "interface", "answer", "saved", "file", "line")
SAW_KEYS = frozenset({"call", "steps", "without", "slot", "saw_of"})


def position(event: Mapping[str, Any]) -> Dict[str, int]:
    """An event's position: the writer that numbered it and its seq."""
    return {"writer": event["writer"], "seq": event["seq"]}


def _same(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]]) -> bool:
    """Two positions (or null) are the same only when both numbers are."""
    if a is None or b is None:
        return a is None and b is None
    return a.get("writer") == b.get("writer") and a.get("seq") == b.get("seq")


def relink(events: Iterable[Mapping[str, Any]], last: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Events as one form: each ``after`` the position of the event before it
    in this form (``last`` before the first, for a form that goes on)."""
    out = []
    for e in events:
        e = dict(e)
        e["after"] = dict(last) if last is not None else None
        last = position(e)
        out.append(e)
    return out


# ------------------------------------------------------------------ replaying


class Replay:
    """What a person watching a form of a log saw, event by event
    (streaming.md, *Replaying*): for each call started, whether it ended
    (``None``, ``"done"``, ``"failed"``) and each field's text so far. An
    event of a kind this reader does not know changes nothing."""

    def __init__(self) -> None:
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.tree: Optional[str] = None
        self.finished = False

    def apply(self, event: Mapping[str, Any]) -> None:
        if self.tree is None:
            self.tree = event.get("tree")
        kind, call = event.get("kind"), event.get("call")
        if kind not in KINDS:
            return
        if kind == "started":
            self.calls[call] = {"ended": None, "fields": {}}
            return
        state = self.calls.get(call)
        if state is None:
            return                         # a call whose start this form does not show
        if kind in ("request", "retry"):
            state["fields"] = {}
        elif kind == "text":
            state["fields"][event["field"]] = state["fields"].get(event["field"], "") + event["text"]
        elif kind in ("done", "failed"):
            state["ended"] = kind
            if call == event.get("tree"):
                self.finished = True

    @property
    def state(self) -> Dict[str, Any]:
        """``{"calls": {call: {"ended", "fields"}}, "finished": bool}``, a copy."""
        return {"calls": copy.deepcopy(self.calls), "finished": self.finished}


def replay(events: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The state after each event of a form: ``{"calls": {...}}`` each."""
    r, out = Replay(), []
    for e in events:
        r.apply(e)
        out.append({"calls": copy.deepcopy(r.calls)})
    return out


def read_after(events: Sequence[Mapping[str, Any]], after: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """What a source holding these events (one form of one log) gives a reader
    that has events up to ``after``: the events after it, in order, or
    ``EventRefused("event-unknown")`` when it does not have that event."""
    if after is None:
        return copy.deepcopy([dict(e) for e in events])
    for i, e in enumerate(events):
        if _same(position(e), after):
            return copy.deepcopy([dict(x) for x in events[i + 1:]])
    raise EventRefused("event-unknown", f"this source has no event {after}")


# ------------------------------------------------------------------ the kept form


def known(event: Mapping[str, Any]) -> Dict[str, Any]:
    """What a form maker keeps of an event of a kind it knows: the keys it
    knows, and inside the objects this contract defines (an error, a program,
    saw's entries) the members it knows. A saw entry it does not know becomes
    ``{}``: still not known to a reader, and nothing of it passed on."""
    kind = event["kind"]
    out = {k: copy.deepcopy(v) for k, v in event.items() if k in ENVELOPE or k in KEYS[kind]}
    if isinstance(out.get("error"), Mapping):
        out["error"] = {k: v for k, v in out["error"].items() if k in ERROR_KEYS}
    if isinstance(out.get("program"), Mapping):
        out["program"] = {k: v for k, v in out["program"].items() if k in PROGRAM_KEYS}
    if isinstance(out.get("saw"), list):
        out["saw"] = [x if isinstance(x, Mapping) and set(x) <= SAW_KEYS else {} for x in out["saw"]]
    return out


def kept_event(event: Mapping[str, Any], keep: Mapping[str, Mapping[str, bool]],
               program: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The kept form of one event (streaming.md, *The kept form*), None when
    the kept form has no such event. ``keep``: which fields of the event's
    call its log_content keeps, ``{"inputs": {name: bool}, "outputs": {...}}``;
    ``program``: the call's program (its ``kind`` and ``answer``). The
    ``after`` is left as it is (``relink`` sets it for a form)."""
    kind = event.get("kind")
    if kind not in KINDS:
        return None
    out = known(event)
    ins, outs = keep.get("inputs", {}), keep.get("outputs", {})
    if all(ins.values()) and all(outs.values()):
        return out
    if kind == "started":
        inputs = {k: v for k, v in (event.get("inputs") or {}).items() if ins.get(k, True)}
        out.pop("inputs", None)
        rebuilt: Dict[str, Any] = {}
        for k, v in out.items():
            if k == "content":
                if inputs:
                    rebuilt["inputs"] = inputs
                rebuilt["content"] = False
                rebuilt["omitted"] = {"inputs": [n for n, v in ins.items() if not v],
                                      "outputs": [n for n, v in outs.items() if not v]}
            elif k != "omitted":
                rebuilt[k] = v
        return rebuilt
    if kind == "request":
        return out
    if kind == "text":
        return out if outs.get(event["field"], True) else None
    if kind == "thinking":
        return None
    if kind == "tool_call":
        out.pop("input", None)
    elif kind == "tool_result":
        out.pop("output", None)
    elif kind == "retry":
        out.pop("reason", None)
    elif kind == "done":
        program = program or {}
        holds = [program.get("answer") or "result"] if program.get("kind", "ai") == "ai" else list(outs)
        if all(outs.get(k, True) for k in holds):
            return out
        out.pop("value", None)
    elif kind == "failed":
        out["error"] = {k: v for k, v in (out.get("error") or {}).items() if k in ("type", "code")}
    out["content"] = False
    return out


def kept_form(events: Iterable[Mapping[str, Any]], keep: Mapping[str, Mapping[str, Mapping[str, bool]]]
              ) -> List[Dict[str, Any]]:
    """The kept form of a whole log: ``keep`` gives, for each call id, which of
    its fields its log_content keeps. Events of kinds this maker does not know
    are left out, and so are keys and members it does not know."""
    events = list(events)
    programs = {e["call"]: e.get("program") for e in events if e.get("kind") == "started"}
    out = []
    for e in events:
        k = kept_event(e, keep.get(e.get("call"), {}), programs.get(e.get("call")))
        if k is not None:
            out.append(k)
    return relink(out)


# ------------------------------------------------------------------ following


class Follower:
    """A reader that follows a form of logs live, one state per tree
    (streaming.md, *Following a log*).

    ``receive(event)`` says what it did with each: ``"kept"`` (the next
    event), ``"duplicate"``, ``"stale"`` (a writer the log has left behind),
    ``"rewind"`` (a later writer continued the log from an event this reader
    holds: it drops what it has after that event, keeps the rest, and takes
    this one), ``"loss"`` (events were lost: ``resume`` from a source) or
    ``"unknown-format"`` (it stops following). It keeps the events it holds,
    since it may have to rewind.

    ``live=True``: it follows a live form (the whole log, or a view of it that
    may show values the kept form lacks), given each event by the process of
    the event's writer; such a form it resumes in place only from the process
    that gave it every event it holds. The kept form (the default) is the same
    from every source."""

    def __init__(self, *, live: bool = False) -> None:
        self.live = live
        self._held: Dict[str, List[Dict[str, Any]]] = {}
        self._givers: Dict[str, List[Any]] = {}
        self.stopped = False

    def receive(self, event: Mapping[str, Any], *, given_by: Any = None) -> str:
        if self.stopped:
            return "unknown-format"
        if event.get("functai_event") != FORMAT:
            self.stopped = True
            return "unknown-format"
        tree = event["tree"]
        held = self._held.setdefault(tree, [])
        givers = self._givers.setdefault(tree, [])
        last = position(held[-1]) if held else None
        writer = last["writer"] if last else 0
        if event["writer"] < writer:
            return "stale"
        if last is not None and event["writer"] == writer and event["seq"] <= last["seq"]:
            return "duplicate"
        after = event.get("after")
        if _same(after, last):
            result = "kept"
        elif event["writer"] > writer and (after is None or any(_same(position(e), after) for e in held)):
            keep = 0 if after is None else next(i for i, e in enumerate(held) if _same(position(e), after)) + 1
            del held[keep:], givers[keep:]
            result = "rewind"
        else:
            return "loss"
        held.append(dict(event))
        givers.append(given_by if given_by is not None else event["writer"])
        return result

    def events(self, tree: str) -> List[Dict[str, Any]]:
        """The events it holds of a tree, in order."""
        return [dict(e) for e in self._held.get(tree, [])]

    def last(self, tree: str) -> Optional[Dict[str, int]]:
        held = self._held.get(tree)
        return position(held[-1]) if held else None

    def trees(self) -> List[str]:
        return list(self._held)

    def state(self, tree: str) -> Dict[str, Any]:
        """The replay of what it holds of a tree: ``{"calls", "finished"}``."""
        r = Replay()
        for e in self._held.get(tree, []):
            r.apply(e)
        return r.state

    def can_resume_from(self, tree: str, source: Any) -> bool:
        """Whether reading ``source`` after its last event keeps what it holds
        true: the source can give every event it holds, as it holds it
        (streaming.md, *Resuming*). ``source.writer`` is the writer whose
        process it is, or None for a store."""
        if not self.live:
            return True
        writer = getattr(source, "writer", None)
        return writer is not None and all(g == writer for g in self._givers.get(tree, []))

    def resume(self, tree: str, source: Any) -> List[Tuple[Optional[Dict[str, int]], Any]]:
        """Read a tree again from ``source`` (anything with ``read(tree, after)``):
        after its last event when it may resume in place, from the beginning
        when it may not or the source says ``event-unknown``. Returns its reads,
        each ``(after, the events read or the EventRefused)``."""
        reads: List[Tuple[Optional[Dict[str, int]], Any]] = []
        writer = getattr(source, "writer", None)
        last = self.last(tree)
        if last is not None and self.can_resume_from(tree, source):
            try:
                got = source.read(tree, last)
            except EventRefused as exc:
                reads.append((last, exc))
            else:
                reads.append((last, got))
                for e in got:
                    self._held[tree].append(dict(e))
                    self._givers[tree].append(writer)
                return reads
        got = source.read(tree, None)
        reads.append((None, got))
        self._held[tree] = [dict(e) for e in got]
        self._givers[tree] = [writer] * len(got)
        return reads


# ------------------------------------------------------------------ stores


def _finished(log: Sequence[Mapping[str, Any]], tree: str) -> bool:
    return bool(log) and log[-1].get("call") == tree and log[-1].get("kind") in ("done", "failed")


def _is_event(e: Any) -> bool:
    """The event schema's checks a store needs to refuse what is not an event of this format."""
    if not isinstance(e, Mapping) or e.get("functai_event") != FORMAT:
        return False
    for k in ENVELOPE:
        if k not in e:
            return False
    ints = all(isinstance(e.get(k), int) and not isinstance(e.get(k), bool) and e[k] >= 1 for k in ("writer", "seq"))
    after = e.get("after")
    ok_after = after is None or (isinstance(after, Mapping) and set(after) == {"writer", "seq"} and all(
        isinstance(after[k], int) and not isinstance(after[k], bool) and after[k] >= 1 for k in ("writer", "seq")))
    if not (ints and ok_after and isinstance(e.get("kind"), str) and isinstance(e.get("tree"), str)
            and isinstance(e.get("call"), str) and isinstance(e.get("function"), str) and bool(e.get("function"))
            and isinstance(e.get("at"), str)):
        return False
    return all(k in e for k in _REQUIRED.get(e["kind"], ()))


# The keys each kind this contract names must have (event.schema.json); a kind
# a later stage adds is open.
_REQUIRED = {"started": ("parent", "root", "program", "content", "saw"), "request": ("request", "model"),
             "text": ("field", "answer", "text"), "thinking": ("text",), "tool_call": ("id", "name"),
             "tool_result": ("id", "name"), "retry": ("wait",), "done": (), "failed": ("error",)}


class MemoryStore:
    """A store kept in this process's memory, by the rules every store keeps
    (streaming.md, *The rules a store keeps*): each claim and each append is
    one step per log. A journal for tests, notebooks and one process; logs are
    lost when it ends.

    ``append(event)`` and ``extend(events)`` (a batch, kept whole or not at
    all) answer ``"kept"`` or ``"duplicate"``, or raise ``EventRefused``;
    ``claim(tree)`` gives a later writer its number and the last kept event's
    position; ``read(tree, after)`` gives the kept events after a position."""

    #: None: this is a store, not the process of a writer (``Follower.resume``).
    writer = None

    def __init__(self) -> None:
        self._logs: Dict[str, List[Dict[str, Any]]] = {}
        self._writers: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ----- the rules

    def _append(self, logs: Dict[str, List[Dict[str, Any]]], writers: Dict[str, int], e: Any,
                tree: Optional[str] = None) -> str:
        if not _is_event(e) or (tree is not None and e["tree"] != tree):
            raise EventRefused("event-malformed", "not an event of format 2 of this log")
        after = e["after"]
        if after is not None and (e["seq"] <= after["seq"] or after["writer"] > e["writer"]):
            raise EventRefused("event-malformed", f"event {position(e)} comes after {after}, which cannot be")
        log = logs.setdefault(e["tree"], [])
        if log and e["writer"] != writers.get(e["tree"], 1):
            raise EventRefused("event-conflict", f"writer {e['writer']} does not have this log (writer "
                                                 f"{writers.get(e['tree'], 1)} does)")
        from .calllog import canonical
        for kept in log:
            if kept["seq"] == e["seq"]:
                if canonical(kept) == canonical(e):
                    return "duplicate"
                raise EventRefused("event-conflict", f"seq {e['seq']} is kept, and holds another event")
        if _finished(log, e["tree"]):
            raise EventRefused("event-after-end", "the log is finished")
        last = position(log[-1]) if log else None
        if not _same(after, last):
            if (after["seq"] if after else 0) > (last["seq"] if last else 0):
                raise EventRefused("event-gap", f"events are missing before {position(e)} (the last kept is {last})")
            raise EventRefused("event-conflict", f"the log went on another way (the last kept is {last})")
        if not log and (e["kind"] != "started" or e["call"] != e["tree"] or e["writer"] != 1):
            raise EventRefused("event-start", "a log starts with its outermost call's started, from writer 1")
        log.append(copy.deepcopy(dict(e)))
        return "kept"

    def append(self, event: Mapping[str, Any]) -> str:
        with self._lock:
            return self._append(self._logs, self._writers, event)

    def extend(self, events: Sequence[Mapping[str, Any]]) -> str:
        """A batch of one log's events, checked each as if appended alone, in
        order, kept whole or not at all: ``"duplicate"`` when every event is
        one, ``"kept"`` when every event is kept or a duplicate; else nothing is
        kept, and the refusal names the first event refused (``err.event``)."""
        events = list(events)
        if not events:
            return "duplicate"
        with self._lock:
            logs = {k: list(v) for k, v in self._logs.items()}
            writers = dict(self._writers)
            answers = []
            tree = events[0].get("tree") if isinstance(events[0], Mapping) else None
            for e in events:
                try:
                    answers.append(self._append(logs, writers, e, tree))
                except EventRefused as exc:
                    raise EventRefused(exc.code, str(exc), event=position(e) if _is_event(e) else None) from None
            self._logs, self._writers = logs, writers
            return "duplicate" if all(a == "duplicate" for a in answers) else "kept"

    def claim(self, tree: str) -> Dict[str, Any]:
        """A later writer claims an unfinished log: it gets a writer number, one
        more than any given for this log, and the position of the last kept
        event, after which it numbers. Every earlier writer is fenced."""
        with self._lock:
            log = self._logs.get(tree)
            if not log:
                raise EventRefused("event-unknown", f"no log {tree}")
            if _finished(log, tree):
                raise EventRefused("event-after-end", "the log is finished")
            self._writers[tree] = self._writers.get(tree, 1) + 1
            return {"writer": self._writers[tree], "after": position(log[-1])}

    def read(self, tree: str, after: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        """The kept events of a log after a position (all of them after None).
        Never fenced: a fenced writer learns what was kept by reading."""
        with self._lock:
            return read_after(self._logs.get(tree, []), after)

    # ----- looking at it

    def trees(self) -> List[str]:
        with self._lock:
            return [t for t, log in self._logs.items() if log]

    def finished(self, tree: str) -> bool:
        with self._lock:
            return _finished(self._logs.get(tree, []), tree)

    def writer_of(self, tree: str) -> int:
        """The last writer number this store gave for a log (1 before any claim)."""
        with self._lock:
            return self._writers.get(tree, 1)

    def __repr__(self) -> str:
        return f"<MemoryStore: {len(self.trees())} logs>"


def settle(store: Any, tree: Optional[str], event: Optional[Mapping[str, Any]], *, claim: bool = False) -> str:
    """What reading a journal says of an end its writer could not confirm:
    ``"kept"`` (the log holds that event), ``"another-end"`` (another writer
    ended the log) or ``"not-kept"`` (the log is unfinished; final only after
    a claim, which ``claim=True`` makes first)."""
    if claim:
        try:
            store.claim(tree)
        except EventRefused:
            pass                                 # finished, or never kept: reading says which
    try:
        log = store.read(tree, None)
    except EventRefused:
        return "not-kept"
    if any(_same(position(e), event) for e in log):
        return "kept"
    return "another-end" if _finished(log, tree or "") else "not-kept"


# ------------------------------------------------------------------ journals


class Journal:
    """Where a call tree's kept log is written while it runs: a store, and how
    surely (``configure(journal=Journal(store, required=True))``).

    Best effort (the default) never makes a call wait: if the store fails,
    FunctAI warns once and the log is kept as far as it could be. A required
    journal makes the call wait until every event so far is confirmed at
    three barriers: when it starts, before each tool runs, and before it
    returns. If the start or a tool is not confirmed, the call stops with
    ``JournalError`` (``journal-barrier``); if its end is not, the caller gets
    ``JournalError`` (``journal-end``) holding what the call did. ``retries``:
    how many times an append is sent again, in a row, when no answer came.

    A bare store is a best-effort journal: ``configure(journal=store)``.
    Two journals are the same when their store and mode are."""

    __slots__ = ("store", "required", "retries")

    def __init__(self, store: Any, *, required: bool = False, retries: int = 2):
        if isinstance(store, Journal):
            store = store.store
        for method in ("append", "read"):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"a journal's store has append(event) and read(tree, after) (a functai.MemoryStore, "
                                f"say); {type(store).__name__} has no {method}()")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError(f"retries is a whole number of at least 0, not {retries!r}")
        self.store = store
        self.required = bool(required)
        self.retries = retries

    @property
    def mode(self) -> str:
        return "required" if self.required else "best-effort"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Journal) and other.store is self.store and other.required == self.required

    def __hash__(self) -> int:
        return hash((id(self.store), self.required))

    def __repr__(self) -> str:
        return f"Journal({self.store!r}, required={self.required})"


def as_journal(value: Any) -> Optional[Journal]:
    """A ``journal`` setting as a Journal: None for ``False`` (none)."""
    if value is None or value is False:
        return None
    return value if isinstance(value, Journal) else Journal(value)


def check_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Refuse receiver settings that could only fail later; returns them normalized."""
    out: Dict[str, Any] = {}
    observers = settings.get("observers")
    if observers is not None:
        if callable(observers) or not isinstance(observers, (list, tuple)):
            raise TypeError("observers is a list: observers=[print_event, events_list]")
        for o in observers:
            if not (callable(o) or callable(getattr(o, "append", None))):
                raise TypeError(f"an observer is a function of one event, or a list (it is appended to); "
                                f"not {type(o).__name__}")
        out["observers"] = tuple(observers)
    journal = settings.get("journal")
    if journal is not None:
        if journal is True:
            raise TypeError("journal is a store, functai.Journal(store, required=True), or False (none); not True")
        out["journal"] = False if journal is False else as_journal(journal)
    return out


class JournalWriter:
    """Sends one tree's kept log to a journal's store, event by event, in its
    own thread (streaming.md, *Appending*). An event is confirmed when the
    store answers kept or duplicate, refused when it answers anything else
    (nothing more is sent), unanswered when it raises: it may have been kept,
    so it is sent again, up to ``1 + retries`` times in a row; then the writer
    tries again when the next event comes. ``barrier()`` waits until every
    event so far has had its turn and says ``"confirmed"``, ``"refused"`` or
    ``"unanswered"``."""

    def __init__(self, journal: Journal, *, name: str = "", on_problem: Optional[Callable[[str], None]] = None,
                 thread: bool = True):
        self.journal = journal
        self.store = journal.store
        self.retries = journal.retries
        self.name = name
        self.pending: List[Dict[str, Any]] = []
        self.refused: Optional[EventRefused] = None
        self.problem: Optional[str] = None
        self.on_problem = on_problem
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._threaded = thread
        self._stopped = False
        self._lock = threading.Lock()

    # ----- the call's side

    def send(self, event: Mapping[str, Any]) -> None:
        self._put(("event", dict(event)))

    def barrier(self) -> str:
        """Wait until every event sent so far has been tried; the result."""
        done = threading.Event()
        holder: List[str] = []
        self._put(("barrier", done, holder))
        done.wait()
        return holder[0]

    def end(self, event: Mapping[str, Any]) -> str:
        """Send the log's last event and wait: ``"confirmed"``, ``"refused"`` or ``"unanswered"``."""
        self.send(event)
        status = self.barrier()
        self._put(("stop",))
        return status

    # ----- the sender

    def _put(self, item: tuple) -> None:
        if not self._threaded:
            if item[0] != "stop":
                self._handle(item)
            return
        with self._lock:
            if self._stopped:                    # the log's end was sent: only barriers are answered
                if item[0] == "barrier":
                    self._handle(item)
                return
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name=f"functai-journal-{self.name or 'log'}")
                self._thread.start()
                _writers.add(self)
            self._queue.put(item)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item[0] == "stop":
                with self._lock:
                    self._stopped = True
                    _writers.discard(self)
                    while not self._queue.empty():   # a barrier asked meanwhile
                        rest = self._queue.get_nowait()
                        if rest[0] == "barrier":
                            self._handle(rest)
                return
            self._handle(item)

    def _handle(self, item: tuple) -> None:
        if item[0] == "event":
            if self.refused is None:
                self.pending.append(item[1])
                self._round()
        elif item[0] == "barrier":
            item[2].append(self.status)
            item[1].set()

    @property
    def status(self) -> str:
        if self.refused is not None:
            return "refused"
        return "unanswered" if self.pending else "confirmed"

    def _round(self) -> None:
        while self.pending and self.refused is None:
            event = self.pending[0]
            answered = False
            for _ in range(1 + self.retries):
                try:
                    self.store.append(event)
                except EventRefused as exc:
                    self.refused = exc
                    self._warn(f"the journal refused event {position(event)} ({exc.code}: {exc}); nothing more "
                               f"of this log is sent to it")
                    return
                except Exception as exc:  # noqa: BLE001 — no answer came: it may have been kept
                    self._warn(f"the journal did not answer ({type(exc).__name__}: {exc}); sending again")
                    continue
                answered = True
                break
            if not answered:
                return
            self.pending.pop(0)

    def _warn(self, message: str) -> None:
        if self.problem is None:
            self.problem = message
            if self.on_problem is not None:
                self.on_problem(message)


_writers: "set[JournalWriter]" = set()


@atexit.register
def _drain_at_exit() -> None:
    """Give the journals' senders and the observers' feeds a moment at exit."""
    drain(2.0)


def drain(timeout: float = 5.0) -> None:
    """Wait (at most ``timeout`` seconds) until every observer has been given
    the events made so far and every best-effort journal has tried to keep them."""
    deadline = time.monotonic() + timeout
    for w in list(_writers):
        done = threading.Event()
        w._put(("barrier", done, []))
        done.wait(max(0.0, deadline - time.monotonic()))
    for feed in list(_feeds.values()):
        feed.drain(max(0.0, deadline - time.monotonic()))


# ------------------------------------------------------------------ observers


class _Feed:
    """One observer's events, handed to it in order from a thread of its own,
    so a slow observer never slows a call. If it raises, FunctAI warns once
    and gives it no more events."""

    MAX = 100_000

    def __init__(self, observer: Any):
        self.observer = observer
        self.give = observer if callable(observer) else observer.append
        self.queue: "queue.Queue[Any]" = queue.Queue()
        self.broken = False
        self.dropped = False
        self.thread = threading.Thread(target=self._run, daemon=True, name="functai-observer")
        self.thread.start()

    def put(self, event: Dict[str, Any]) -> None:
        if self.broken:
            return
        if self.queue.qsize() >= self.MAX:
            if not self.dropped:
                self.dropped = True
                warnings.warn(f"[functai] an observer ({_name(self.observer)}) is too slow: events are dropped "
                              f"(it will see a loss)", stacklevel=2)
            return
        self.queue.put(event)

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if isinstance(item, threading.Event):
                    item.set()
                    continue
                if not self.broken:
                    self.give(item)
            except Exception as exc:  # noqa: BLE001 — an observer never gets in the way
                self.broken = True
                warnings.warn(f"[functai] an observer ({_name(self.observer)}) failed and is given no more events: "
                              f"{type(exc).__name__}: {exc}", stacklevel=2)
            finally:
                self.queue.task_done()

    def drain(self, timeout: float) -> None:
        done = threading.Event()
        self.queue.put(done)
        done.wait(timeout)


_feeds: Dict[int, _Feed] = {}
_feeds_lock = threading.Lock()


def _feed(observer: Any) -> _Feed:
    with _feeds_lock:
        feed = _feeds.get(id(observer))
        if feed is None or feed.observer is not observer:
            feed = _feeds[id(observer)] = _Feed(observer)
        return feed


def _name(observer: Any) -> str:
    return getattr(observer, "__qualname__", None) or type(observer).__name__


# ------------------------------------------------------------------ which receivers a tree gets


@dataclasses.dataclass
class Receivers:
    """The observers (outermost first) and the journal a call gets from the
    layers of settings around it; ``refused``: the JournalError the tree
    raises before it runs (``journal-policy``), when the layers break the
    journal policy (then ``journal`` is the host's, where its log goes)."""
    observers: List[Any]
    journal: Optional[Journal]
    refused: Optional[JournalError] = None


_ABSENT = object()


def _journal_setting(layer: Mapping[str, Any]) -> Any:
    if "journal" not in layer or layer["journal"] is None:
        return _ABSENT
    return as_journal(layer["journal"])


def _refused(layers: List[Tuple[str, Mapping[str, Any]]]) -> List[int]:
    """The layers (indexes, closest first) whose journal setting a farther layer's refuses."""
    settings = [(i, _journal_setting(layer)) for i, (_w, layer) in enumerate(layers)]
    settings = [(i, j) for i, j in settings if j is not _ABSENT]
    out = set()
    for n, (i, far) in enumerate(settings):
        for k, near in settings[:n]:
            if near == far:
                continue
            if far is not None and far.required:
                out.add(k)                        # replaces, weakens or removes a required journal
            elif far is not None and layers[k][0] == "own" and layers[i][0] != "own" and not (
                    near is not None and near.store is far.store and near.required):
                out.add(k)                        # a program replaces or removes a host's journal
    return sorted(out)


def closest_journal(layers: List[Tuple[str, Mapping[str, Any]]]) -> Optional[Journal]:
    """The journal the closest layer that sets one names (None: none, or no layer sets one)."""
    for _w, layer in layers:
        j = _journal_setting(layer)
        if j is not _ABSENT:
            return j
    return None


def receivers(layers: List[Tuple[str, Mapping[str, Any]]]) -> Receivers:
    """Which observers and journal a call tree gets from the layers around its
    outermost call, closest first (``("own" | "block" | "configure", settings)``):
    observers add up, outermost first; the closest journal setting decides
    (``False``: none), except that a program's own setting cannot replace or
    remove a journal a host layer set (it may name the same one, or make it
    required), and no closer layer can replace, weaken or remove a required
    one. A tree whose layers break that is refused ``journal-policy``: its log
    goes to every observer and to the journal of the layers farther out than
    every refused setting."""
    observers: List[Any] = []
    for _w, layer in reversed(layers):
        for o in layer.get("observers") or ():
            if not any(o is x for x in observers):
                observers.append(o)
    refused = _refused(layers)
    if refused:
        host = receivers(layers[max(refused) + 1:])
        where = ", ".join(f"{layers[i][0]}" for i in refused)
        error = JournalError("journal-policy", f"the journal setting of the {where} layer breaks the journal "
                                               f"policy: a program cannot replace or remove the journal its host "
                                               f"set, and no setting closer to a call can replace, weaken or remove "
                                               f"a required journal (the host's journal is "
                                               f"{host.journal!r}); use an observer to keep a copy of your own")
        return Receivers(observers, host.journal, error)
    for _w, layer in layers:
        j = _journal_setting(layer)
        if j is not _ABSENT:
            return Receivers(observers, j)
    return Receivers(observers, None)


# ------------------------------------------------------------------ the tree log, in this process


def _now_iso() -> str:
    from .calllog import _iso
    return _iso(time.time())


class TreeLog:
    """The writer of one call tree's log in this process: it numbers every
    event as it happens (``seq`` dense over the whole log, ``at`` never going
    back) and hands each to the streams, observers and journal that see its
    call, in the form each may see: a stream the whole log of its call and
    the calls inside it; an observer and the journal the kept form (each
    call's ``log_content``), each with its own ``after`` chain."""

    def __init__(self, tree: str, *, journal: Optional[Journal] = None, writer: int = 1):
        self.tree = tree
        self.writer = writer
        self.seq = 0
        self.last: Optional[Dict[str, int]] = None
        self.at = ""
        self.lock = threading.RLock()
        self.journal = journal
        self.sender: Optional[JournalWriter] = None
        self.links: Dict[int, Optional[Dict[str, int]]] = {}      # each sink's last position in its form
        self.warned: set = set()
        if journal is not None:
            self.sender = JournalWriter(journal, name=tree[:8], on_problem=self._journal_problem)

    def _journal_problem(self, message: str) -> None:
        if self.journal is not None and not self.journal.required:
            _warn_once(("journal", id(self.journal.store)), f"best-effort journal: {message}")

    @property
    def required(self) -> bool:
        return self.journal is not None and self.journal.required

    def emit(self, call: Any, kind: str, *, hold: bool = False, **fields: Any) -> Optional[Any]:
        """Number an event of ``call`` and hand it on (when anything sees the
        call). ``hold``: give it to the journal only; ``release`` gives it to
        readers once the journal confirmed it (a required journal's last event)."""
        with self.lock:
            self.seq += 1
            pos = {"writer": self.writer, "seq": self.seq}
            after, self.last = self.last, pos
            at = _now_iso()
            if at < self.at:
                at = self.at
            self.at = at
            if not (call.streams or call.observers or self.journal is not None):
                return None
            from . import streaming
            event = streaming.make_event(kind, call=call.id, function=call.function, tree=self.tree,
                                         writer=self.writer, seq=self.seq, after=after, at=at, **fields)
            try:
                self._deliver(call, event, journal_only=hold)
            except Exception as exc:  # noqa: BLE001 — watching a call never stands in its way
                _warn_once(("deliver", type(exc).__name__), f"an event of {call.function} could not be given to "
                                                            f"its readers: {type(exc).__name__}: {exc}")
            return event

    def _deliver(self, call: Any, event: Any, *, journal_only: bool = False, readers_only: bool = False) -> None:
        whole: Optional[Dict[str, Any]] = None
        kept: Any = _ABSENT
        if not readers_only and self.sender is not None:
            whole = event.to_dict()
            kept = kept_event(whole, call.keep, call.info)
            if kept is not None:
                kept["after"] = self.links.get(id(self.sender))
                self.links[id(self.sender)] = position(kept)
                self.sender.send(kept)
        if journal_only:
            return
        pos = {"writer": event.writer, "seq": event.seq}
        for stream in call.streams:
            key = id(stream)
            linked = event if _same(event.after, self.links.get(key)) else \
                dataclasses.replace(event, after=self.links.get(key))
            self.links[key] = pos
            stream._receive(linked, call)
        if call.observers:
            if whole is None:
                whole = event.to_dict()
            if kept is _ABSENT:
                kept = kept_event(whole, call.keep, call.info)
            if kept is not None:
                for o in call.observers:
                    key = id(o)
                    mine = dict(kept)
                    mine["after"] = self.links.get(key)
                    self.links[key] = position(mine)
                    _feed(o).put(mine)

    def release(self, call: Any, event: Any) -> None:
        """Give readers an event held back until the journal confirmed it."""
        if event is not None:
            with self.lock:
                try:
                    self._deliver(call, event, readers_only=True)
                except Exception as exc:  # noqa: BLE001
                    _warn_once(("deliver", type(exc).__name__), f"an event of {call.function} could not be given "
                                                                f"to its readers: {type(exc).__name__}: {exc}")

    def barrier(self) -> str:
        return self.sender.barrier() if self.sender is not None else "confirmed"

    def end(self) -> str:
        """After the outermost call's last event: wait for a required journal
        (a best-effort one is left to finish on its own)."""
        if self.sender is None:
            return "confirmed"
        if self.required:
            status = self.sender.barrier()
            self.sender._put(("stop",))
            return status
        self.sender._put(("stop",))
        return "confirmed"


def _warn_once(key: Any, message: str) -> None:
    from .calllog import _warn_once as warn
    warn(key, message)


__all__ = ["Replay", "replay", "read_after", "kept_form", "kept_event", "relink", "position", "Follower",
           "MemoryStore", "settle", "Journal", "JournalWriter", "receivers", "Receivers", "TreeLog", "drain"]
