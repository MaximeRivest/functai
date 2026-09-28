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
- the rules a store keeps (``Store``, ``MemoryStore``, ``read_after``) and
  settling an end a journal did not confirm (``settle``);
- journals (``Journal``, ``JournalWriter``) and which receivers a tree gets
  from the layers of settings around it (``receivers``);
- ``flush``: wait until observers and best-effort journals have been given
  what was made so far.

**Receivers never share data.** Every observer gets its own copy of each
event, and a journal's writer keeps its own copy of what it sends (each
append gets a fresh one), so no receiver can change what another one sees
or what a journal keeps.
"""

from __future__ import annotations

import atexit
import collections
import copy
import dataclasses
import os
import threading
import time
import warnings
import weakref
from collections.abc import Mapping
from typing import (Any, Callable, Deque, Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple,
                    runtime_checkable)

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

#: What a follower did with an event it received.
Received = Literal["kept", "duplicate", "stale", "rewind", "loss", "unknown-format"]
#: A store's answer to an append that kept (or already held) it.
Answer = Literal["kept", "duplicate"]
#: What a journal's writer says of every event sent so far.
Status = Literal["confirmed", "refused", "unanswered"]
#: What reading a journal says of an end its writer could not confirm.
Settled = Literal["kept", "another-end", "not-kept"]


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


def _known_format(event: Any) -> bool:
    """Whether a reader of format 2 knows what this event's numbers mean."""
    return isinstance(event, Mapping) and event.get("functai_event") == FORMAT


# ------------------------------------------------------------------ replaying


class Replay:
    """What a person watching a form of a log saw, event by event
    (streaming.md, *Replaying*): for each call started, whether it ended
    (``None``, ``"done"``, ``"failed"``) and each field's text so far. An
    event of a kind this reader does not know changes nothing; at an event of
    a format it does not know, it stops (``stopped``): what that format's
    numbers mean is not known, so nothing from there on is applied."""

    def __init__(self) -> None:
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.tree: Optional[str] = None
        self.finished = False
        self.stopped = False

    def apply(self, event: Mapping[str, Any]) -> bool:
        """Apply one event; False once stopped (this event and every later one are not applied)."""
        if self.stopped:
            return False
        if not _known_format(event):
            self.stopped = True
            return False
        if self.tree is None:
            self.tree = event.get("tree")
        kind, call = event.get("kind"), event.get("call")
        if kind not in KINDS:
            return True
        if kind == "started":
            self.calls[call] = {"ended": None, "fields": {}}
            return True
        state = self.calls.get(call)
        if state is None:
            return True                    # a call whose start this form does not show
        if kind in ("request", "retry"):
            state["fields"] = {}
        elif kind == "text":
            state["fields"][event["field"]] = state["fields"].get(event["field"], "") + event["text"]
        elif kind in ("done", "failed"):
            state["ended"] = kind
            if call == event.get("tree"):
                self.finished = True
        return True

    @property
    def state(self) -> Dict[str, Any]:
        """``{"calls": {call: {"ended", "fields"}}, "finished": bool}``, a copy."""
        return {"calls": copy.deepcopy(self.calls), "finished": self.finished}


def replay(events: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The state after each event of a form: ``{"calls": {...}}`` each. At an
    event of a format this reader does not know, it stops: the list ends
    before that event."""
    r, out = Replay(), []
    for e in events:
        if not r.apply(e):
            break
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


def whole(keep: Mapping[str, Mapping[str, bool]]) -> bool:
    """Whether a call's content is whole: every field of it kept."""
    return all(keep.get("inputs", {}).values()) and all(keep.get("outputs", {}).values())


def kept_event(event: Mapping[str, Any], keep: Mapping[str, Mapping[str, bool]],
               program: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The kept form of one event (streaming.md, *The kept form*), None when
    the kept form has no such event. ``keep``: which fields of the event's
    call its log_content keeps, ``{"inputs": {name: bool}, "outputs": {...}}``;
    a name it does not list is not kept (fail closed); ``program``: the
    call's program (its ``kind`` and ``answer``). The ``after`` is left as it
    is (``relink`` sets it for a form)."""
    kind = event.get("kind")
    if kind not in KINDS:
        return None
    out = known(event)
    ins, outs = keep.get("inputs", {}), keep.get("outputs", {})
    if whole(keep) and all(k in ins for k in (event.get("inputs") or {})):
        return out
    if kind == "started":
        inputs = {k: v for k, v in (event.get("inputs") or {}).items() if ins.get(k, False)}
        dropped = [n for n, v in ins.items() if not v]
        dropped += [n for n in (event.get("inputs") or {}) if n not in ins and n not in dropped]
        out.pop("inputs", None)
        rebuilt: Dict[str, Any] = {}
        for k, v in out.items():
            if k == "content":
                if inputs:
                    rebuilt["inputs"] = inputs
                rebuilt["content"] = False
                rebuilt["omitted"] = {"inputs": dropped, "outputs": [n for n, v in outs.items() if not v]}
            elif k != "omitted":
                rebuilt[k] = v
        return rebuilt
    if kind == "request":
        return out
    if kind == "text":
        return out if outs.get(event["field"], False) else None
    if kind == "thinking":
        return None
    if kind == "tool_call":
        out.pop("input", None)
    elif kind == "tool_result":
        out.pop("output", None)
    elif kind == "retry":
        out.pop("reason", None)
    elif kind == "done":
        if _done_value_kept(keep, program):
            return out
        out.pop("value", None)
    elif kind == "failed":
        out["error"] = {k: v for k, v in (out.get("error") or {}).items() if k in ("type", "code")}
    out["content"] = False
    return out


def _done_value_kept(keep: Mapping[str, Mapping[str, bool]], program: Optional[Mapping[str, Any]]) -> bool:
    """Whether a call's ``done`` keeps its value: every output it holds is kept
    (an AI function's answer; a module's outputs)."""
    program = program or {}
    outs = keep.get("outputs", {})
    holds = [program.get("answer") or "result"] if program.get("kind", "ai") == "ai" else list(outs)
    return all(outs.get(k, False) for k in holds)


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
    since it may have to rewind; ``forget(tree)`` lets a tree go.

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

    def _classify(self, tree: str, event: Mapping[str, Any]) -> Tuple[Received, int]:
        """What to do with an event of a known format: the result, and for a
        rewind how many held events to keep."""
        held = self._held.get(tree, [])
        last = position(held[-1]) if held else None
        writer = last["writer"] if last else 0
        if event["writer"] < writer:
            return "stale", 0
        if last is not None and event["writer"] == writer and event["seq"] <= last["seq"]:
            return "duplicate", 0
        after = event.get("after")
        if _same(after, last):
            return "kept", len(held)
        if event["writer"] > writer and (after is None or any(_same(position(e), after) for e in held)):
            keep = 0 if after is None else next(i for i, e in enumerate(held) if _same(position(e), after)) + 1
            return "rewind", keep
        return "loss", 0

    def _take(self, tree: str, event: Mapping[str, Any], keep: int, giver: Any) -> None:
        held = self._held.setdefault(tree, [])
        givers = self._givers.setdefault(tree, [])
        del held[keep:], givers[keep:]
        held.append(copy.deepcopy(dict(event)))
        givers.append(giver)

    def receive(self, event: Mapping[str, Any], *, given_by: Any = None) -> Received:
        if self.stopped:
            return "unknown-format"
        if not _known_format(event):
            self.stopped = True
            return "unknown-format"
        tree = event["tree"]
        result, keep = self._classify(tree, event)
        if result in ("kept", "rewind"):
            self._take(tree, event, keep, given_by if given_by is not None else event["writer"])
        return result

    def events(self, tree: str) -> List[Dict[str, Any]]:
        """The events it holds of a tree, in order (copies)."""
        return copy.deepcopy(self._held.get(tree, []))

    def last(self, tree: str) -> Optional[Dict[str, int]]:
        held = self._held.get(tree)
        return position(held[-1]) if held else None

    def trees(self) -> List[str]:
        return [t for t, held in self._held.items() if held]

    def forget(self, tree: str) -> None:
        """Let a tree go: its events and state (a long-lived reader forgets
        the trees it is done with)."""
        self._held.pop(tree, None)
        self._givers.pop(tree, None)

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

    def _admit(self, tree: str, events: Sequence[Mapping[str, Any]], giver: Any) -> None:
        """Events a source gave, in order: each must be of a format this reader
        knows (it stops at one that is not) and come next in the chain (a
        source that gives a broken chain is not read further)."""
        for e in events:
            if not _known_format(e):
                self.stopped = True
                return
            if e.get("tree") != tree:
                return
            result, keep = self._classify(tree, e)
            if result != "kept":
                return
            self._take(tree, e, keep, giver)

    def resume(self, tree: str, source: Any) -> List[Tuple[Optional[Dict[str, int]], Any]]:
        """Read a tree again from ``source`` (anything with ``read(tree, after)``):
        after its last event when it may resume in place, from the beginning
        when it may not or the source says ``event-unknown``. Returns its reads,
        each ``(after, the events read or the EventRefused)``. It stops at an
        event of a format it does not know, as when following."""
        reads: List[Tuple[Optional[Dict[str, int]], Any]] = []
        if self.stopped:
            return reads
        writer = getattr(source, "writer", None)
        last = self.last(tree)
        if last is not None and self.can_resume_from(tree, source):
            try:
                got = source.read(tree, last)
            except EventRefused as exc:
                reads.append((last, exc))
            else:
                reads.append((last, got))
                self._admit(tree, got, writer)
                return reads
        got = source.read(tree, None)
        reads.append((None, got))
        self._held[tree], self._givers[tree] = [], []
        self._admit(tree, got, writer)
        return reads


# ------------------------------------------------------------------ stores


@runtime_checkable
class Store(Protocol):
    """What a journal needs of a store (streaming.md, *The rules a store keeps*).

    ``append(event)`` keeps one event and answers ``"kept"`` or
    ``"duplicate"``, or raises ``EventRefused`` (``event-malformed``,
    ``event-conflict``, ``event-gap``, ``event-after-end``, ``event-start``).
    Any other answer is taken as a refusal; any other exception means no
    answer came (the event may have been kept: it is sent again).
    ``read(tree, after)`` gives the kept events after a position (all of them
    after None), or raises ``EventRefused("event-unknown")``.

    A store may also have ``extend(events)`` (a batch of one log, kept whole
    or not at all: the writer then sends what waits as one batch) and
    ``claim(tree)`` (a later writer's claim, which fences earlier writers;
    ``JournalError.settle(claim=True)`` needs it). A store times out its own
    I/O; a journal also stops waiting at its barriers after its ``timeout``."""

    def append(self, event: Mapping[str, Any]) -> str: ...

    def read(self, tree: str, after: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]: ...


def _finished(log: Sequence[Mapping[str, Any]], tree: str) -> bool:
    return bool(log) and log[-1].get("call") == tree and log[-1].get("kind") in ("done", "failed")


def _is_event(e: Any) -> bool:
    """Whether ``e`` passes the event schema, as an event of format 2 (a store
    refuses anything else ``event-malformed``)."""
    from . import schemas
    return isinstance(e, Mapping) and e.get("functai_event") == FORMAT and schemas.valid("event", e)


class MemoryStore:
    """A store kept in this process's memory, by the rules every store keeps
    (streaming.md, *The rules a store keeps*): each claim and each append is
    one step per log, and an event that does not pass the event schema is
    refused ``event-malformed``. A journal for tests, notebooks and one
    process; logs are lost when it ends.

    ``append(event)`` and ``extend(events)`` (a batch, kept whole or not at
    all) answer ``"kept"`` or ``"duplicate"``, or raise ``EventRefused``;
    ``claim(tree)`` gives a later writer its number and the last kept event's
    position; ``read(tree, after)`` gives the kept events after a position.

    A journal's writer sends what waits as one batch (``extend``) when it can,
    so a subclass that changes how appends go (a transport, failures for a
    test) overrides ``extend`` as well as ``append`` (or sets ``extend =
    None``: then every event is sent alone)."""

    #: None: this is a store, not the process of a writer (``Follower.resume``).
    writer = None

    def __init__(self) -> None:
        self._logs: Dict[str, List[Dict[str, Any]]] = {}
        self._writers: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ----- the rules

    def _append(self, logs: Dict[str, List[Dict[str, Any]]], writers: Dict[str, int], e: Any,
                tree: Optional[str] = None) -> Answer:
        if not _is_event(e) or (tree is not None and e["tree"] != tree):
            raise EventRefused("event-malformed", "not an event of format 2 of this log (it does not pass the "
                                                  "event schema)")
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

    def append(self, event: Mapping[str, Any]) -> Answer:
        with self._lock:
            return self._append(self._logs, self._writers, event)

    def extend(self, events: Sequence[Mapping[str, Any]]) -> Answer:
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
                    named = position(e) if isinstance(e, Mapping) and isinstance(e.get("writer"), int) \
                        and isinstance(e.get("seq"), int) else None
                    raise EventRefused(exc.code, str(exc), event=named) from None
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


def settle(store: Any, tree: Optional[str], event: Optional[Mapping[str, Any]], *, claim: bool = False) -> Settled:
    """What reading a journal says of an end its writer could not confirm:
    ``"kept"`` (the log holds that event), ``"another-end"`` (another writer
    ended the log) or ``"not-kept"`` (the log is unfinished; final only after
    a claim, which ``claim=True`` makes first)."""
    if claim:
        if not callable(getattr(store, "claim", None)):
            raise TypeError(f"settling for good claims the log first, and this store ({type(store).__name__}) "
                            f"has no claim()")
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
    ``JournalError`` (``journal-end``) holding what the call did.

    ``retries``: how many times an append is sent again, in a row, when no
    answer came, waiting ``backoff`` seconds before the first resend and
    twice as long before each next one. ``timeout``: the longest a barrier
    waits for an answer; past it, no answer came (``"unknown"``: the store
    may still keep what it was sent). A store should also time out its own
    I/O. See ``Store`` for what a store answers.

    A bare store is a best-effort journal: ``configure(journal=store)``.
    Two journals are the same when their store and mode are."""

    __slots__ = ("store", "required", "retries", "backoff", "timeout")

    def __init__(self, store: Any, *, required: bool = False, retries: int = 2, backoff: float = 0.05,
                 timeout: Optional[float] = 30.0):
        if isinstance(store, Journal):
            store = store.store
        for method in ("append", "read"):
            if not callable(getattr(store, method, None)):
                raise TypeError(f"a journal's store has append(event) and read(tree, after) (a functai.MemoryStore, "
                                f"say); {type(store).__name__} has no {method}()")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError(f"retries is a whole number of at least 0, not {retries!r}")
        if not isinstance(backoff, (int, float)) or isinstance(backoff, bool) or backoff < 0:
            raise ValueError(f"backoff is a number of seconds of at least 0, not {backoff!r}")
        if timeout is not None and (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                                    or timeout <= 0):
            raise ValueError(f"timeout is a number of seconds greater than 0 (or None: wait for ever), "
                             f"not {timeout!r}")
        self.store = store
        self.required = bool(required)
        self.retries = retries
        self.backoff = float(backoff)
        self.timeout = None if timeout is None else float(timeout)

    @property
    def mode(self) -> Literal["required", "best-effort"]:
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


_BATCH = 256        # events sent in one batch at most, to a store that takes batches


class JournalWriter:
    """Sends one tree's kept log to a journal's store, in order, from its own
    thread (streaming.md, *Appending*). Each event is a private copy, and each
    append is given a fresh copy of it, so neither the call's code, an
    observer nor the store can change what is sent again.

    An event is confirmed when the store answers ``"kept"`` or
    ``"duplicate"``; refused when it raises ``EventRefused`` or answers
    anything else (nothing more is sent); unanswered when it raises another
    exception (it may have been kept): it is sent again, up to ``1 +
    retries`` times in a row with a growing pause, then the writer tries
    again when the next event comes. A store with ``extend`` is sent what
    waits as one batch. ``barrier()`` waits (at most the journal's
    ``timeout``) until every event given so far has had its turn and says
    ``"confirmed"``, ``"refused"`` or ``"unanswered"``.

    ``fault(message)``: an event of this log could not be made at all (it
    will never be sent): nothing more is sent, and every later barrier says
    ``"refused"``."""

    def __init__(self, journal: Journal, *, name: str = "", on_problem: Optional[Callable[[str], None]] = None,
                 thread: bool = True):
        self.journal = journal
        self.store = journal.store
        self.retries = journal.retries
        self.name = name
        self.pending: Deque[Dict[str, Any]] = collections.deque()      # not confirmed yet, in order
        self.refused: Optional[EventRefused] = None
        self.faulted: Optional[str] = None
        self.problem: Optional[str] = None
        self.on_problem = on_problem
        self._threaded = thread
        self._cond = threading.Condition()
        self._incoming: Deque[Dict[str, Any]] = collections.deque()    # given, not yet had its turn
        self._given = 0                                                # events given so far
        self._tried = 0                                                # events that have had their turn
        self._confirmed = 0                                            # events confirmed (in order)
        self._stopping = False
        self._thread: Optional[threading.Thread] = None

    # ----- the call's side

    def send(self, event: Mapping[str, Any]) -> None:
        """Give the writer an event to keep (a copy is taken now)."""
        snapshot = copy.deepcopy(dict(event))
        if not self._threaded:
            self._given += 1
            self._incoming.append(snapshot)
            self._work_once()
            return
        with self._cond:
            if self._stopping:
                return
            self._given += 1
            self._incoming.append(snapshot)
            self._start()
            self._cond.notify_all()

    def fault(self, message: str) -> None:
        """An event of this log could not be made: the log cannot be kept whole."""
        with self._cond:
            if self.faulted is None:
                self.faulted = message
                self._warn(message)
            self._cond.notify_all()

    def barrier(self, timeout: Any = "journal", cancelled: Optional[Callable[[], bool]] = None) -> Status:
        """Wait until every event given so far has had its turn (at most
        ``timeout`` seconds: the journal's by default), and say how it went.
        ``cancelled``: asked while waiting; when it says True the wait ends
        (the caller stops the call)."""
        if timeout == "journal":
            timeout = self.journal.timeout
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            target = self._given
            while self._confirmed < target and self._tried < target and self.faulted is None \
                    and self.refused is None:
                if cancelled is not None and cancelled():
                    break
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    break
                self._cond.wait(0.05 if left is None or cancelled is not None else min(left, 0.25))
            if self.faulted is not None or self.refused is not None:
                return "refused"
            return "confirmed" if self._confirmed >= target else "unanswered"

    def stop(self) -> None:
        """No more events will come: the writer finishes what waits, then its thread ends."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if not self._threaded:
            _writers.discard(self)

    def wait(self, timeout: Optional[float]) -> bool:
        """Wait until every event given has had its turn (best effort's flush)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._tried < self._given and self.faulted is None and self.refused is None:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._cond.wait(0.05 if left is None else min(left, 0.25))
        return True

    @property
    def status(self) -> Status:
        if self.faulted is not None or self.refused is not None:
            return "refused"
        return "confirmed" if self._confirmed >= self._given else "unanswered"

    # ----- the sender

    def _start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name=f"functai-journal-{self.name or 'log'}")
            self._thread.start()
            _writers.add(self)

    def _run(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._incoming and not self._stopping:
                        self._cond.wait()
                    if not self._incoming and self._stopping:
                        return
                self._work_once()
        finally:
            with self._cond:
                self._cond.notify_all()
            _writers.discard(self)

    def _work_once(self) -> None:
        """Take every event waiting, and send what is not confirmed, in order."""
        with self._cond:
            taken = len(self._incoming)
            if self.refused is None and self.faulted is None:
                self.pending.extend(self._incoming)
            self._incoming.clear()
        try:
            if taken and self.refused is None and self.faulted is None:
                self._round()
        finally:
            with self._cond:
                self._tried += taken
                self._cond.notify_all()

    def _answer(self, got: Any, event: Mapping[str, Any]) -> bool:
        """True when ``got`` confirms. Any other answer is a refusal (recorded):
        a store that answers without saying kept or duplicate has not said it
        kept the event."""
        if isinstance(got, str) and got in ("kept", "duplicate"):
            return True
        self.refused = EventRefused(got if isinstance(got, str) and got.startswith("event-") else "event-answer",
                                    f"the journal answered {got!r} to event {position(event)}, which is neither "
                                    f"kept nor duplicate")
        self._warn(f"the journal refused event {position(event)} (it answered {got!r}); nothing more of this log is "
                   f"sent to it")
        return False

    def _round(self) -> None:
        batches = callable(getattr(self.store, "extend", None))
        while self.pending and self.refused is None and self.faulted is None:
            group = list(self.pending)[:_BATCH] if batches and len(self.pending) > 1 else [self.pending[0]]
            answered = False
            pause = self.journal.backoff
            for attempt in range(1 + self.retries):
                if attempt and pause:
                    time.sleep(pause)
                    pause = min(pause * 2, 2.0)
                try:
                    if len(group) == 1:
                        got = self.store.append(copy.deepcopy(group[0]))
                    else:
                        got = self.store.extend(copy.deepcopy(group))
                except EventRefused as exc:
                    self.refused = exc
                    named = exc.event if exc.event is not None else position(group[0])
                    self._warn(f"the journal refused event {named} ({exc.code}: {exc}); nothing more of this log "
                               f"is sent to it")
                    return
                except Exception as exc:  # noqa: BLE001 — no answer came: it may have been kept
                    self._warn(f"the journal did not answer ({type(exc).__name__}: {exc}); sending again")
                    continue
                if not self._answer(got, group[0]):
                    return
                answered = True
                break
            if not answered:
                return
            with self._cond:
                for _ in group:
                    self.pending.popleft()
                self._confirmed += len(group)
                self._cond.notify_all()

    def _warn(self, message: str) -> None:
        if self.problem is None:
            self.problem = message
            if self.on_problem is not None:
                self.on_problem(message)


_writers: "set[JournalWriter]" = set()


# ------------------------------------------------------------------ observers


def _is_list(observer: Any) -> bool:
    """An observer given events where they are made: a list (or deque), whose
    append cannot be slow and never fails."""
    return type(observer) is list or type(observer) is collections.deque


_broken_refs: "weakref.WeakSet[Any]" = weakref.WeakSet()
_broken_ids: Dict[int, Any] = {}             # observers that cannot be weakly referred to


def _is_broken(observer: Any) -> bool:
    try:
        return observer in _broken_refs
    except TypeError:
        return _broken_ids.get(id(observer)) is observer


def _mark_broken(observer: Any) -> None:
    try:
        _broken_refs.add(observer)
    except TypeError:
        _broken_ids[id(observer)] = observer


class _Feed:
    """One observer's events, handed to it in order from a thread that runs
    while events wait for it and ends when none does, so a slow observer
    never slows a call and an idle one holds no thread. A feed lives while a
    tree running in this process uses its observer, or events wait for it;
    then it is let go. If the observer raises, FunctAI warns once and gives
    it no more events."""

    MAX = 100_000

    def __init__(self, observer: Any):
        self.observer = observer
        self.give = observer if callable(observer) else observer.append
        self.queue: Deque[Dict[str, Any]] = collections.deque()
        self.cond = threading.Condition()
        self.running = False
        self.users = 0
        self.dropped = False

    def put(self, event: Dict[str, Any]) -> None:
        with self.cond:
            if _is_broken(self.observer):
                return
            if len(self.queue) >= self.MAX:
                if not self.dropped:
                    self.dropped = True
                    warnings.warn(f"[functai] an observer ({_name(self.observer)}) is too slow: events are dropped "
                                  f"(it will see a loss)", stacklevel=2)
                return
            self.queue.append(event)
            if not self.running:
                self.running = True
                threading.Thread(target=self._run, daemon=True, name="functai-observer").start()

    def _run(self) -> None:
        while True:
            with self.cond:
                if not self.queue:
                    self.running = False
                    self.cond.notify_all()
                    break
                item = self.queue.popleft()
            try:
                if not _is_broken(self.observer):
                    self.give(item)
            except Exception as exc:  # noqa: BLE001 — an observer never gets in the way
                _mark_broken(self.observer)
                with self.cond:
                    self.queue.clear()
                warnings.warn(f"[functai] an observer ({_name(self.observer)}) failed and is given no more events: "
                              f"{type(exc).__name__}: {exc}", stacklevel=2)
        _retire(self)

    def idle(self) -> bool:
        return not self.queue and not self.running

    def wait(self, timeout: Optional[float]) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.cond:
            while not self.idle():
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self.cond.wait(left)
        return True


_feeds: Dict[int, _Feed] = {}
_feeds_lock = threading.Lock()


def _acquire(observer: Any) -> _Feed:
    """The feed of an observer, for a tree that will give it events."""
    with _feeds_lock:
        feed = _feeds.get(id(observer))
        if feed is None or feed.observer is not observer:
            feed = _feeds[id(observer)] = _Feed(observer)
        feed.users += 1
        return feed


def _release(feed: _Feed) -> None:
    """A tree is done with a feed; it is let go once no tree uses it and nothing waits."""
    with _feeds_lock:
        feed.users -= 1
    _retire(feed)


def _retire(feed: _Feed) -> None:
    with _feeds_lock:
        with feed.cond:
            if feed.users <= 0 and feed.idle() and _feeds.get(id(feed.observer)) is feed:
                del _feeds[id(feed.observer)]


def _name(observer: Any) -> str:
    return getattr(observer, "__qualname__", None) or type(observer).__name__


def flush(timeout: Optional[float] = 5.0) -> bool:
    """Wait (at most ``timeout`` seconds; None: for ever) until every observer
    has been given the events made so far, and every best-effort journal has
    tried to keep them. True when all of it was done in time.

    Observers that are lists get each event as it is made, so they are
    complete when a call returns; a function observer runs in a thread of
    its own, and may still be catching up."""
    deadline = None if timeout is None else time.monotonic() + timeout

    def left() -> Optional[float]:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    done = True
    for w in list(_writers):
        done = w.wait(left()) and done
    for feed in list(_feeds.values()):
        done = feed.wait(left()) and done
    return done


def drain(timeout: float = 5.0) -> bool:
    """``flush``, by its older name."""
    return flush(timeout)


@atexit.register
def _flush_at_exit() -> None:
    """Give the journals' senders and the observers' feeds a moment at exit."""
    flush(2.0)


def _after_fork() -> None:
    """A forked child has none of its parent's threads: its feeds and writers
    start afresh (an observer set by configure goes on in the child)."""
    global _feeds_lock
    _feeds_lock = threading.Lock()
    _feeds.clear()
    _writers.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


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


def closest_journal(layers: List[Tuple[str, Mapping[str, Any]]]) -> Any:
    """The journal setting of the closest layer that sets one: a Journal,
    None (``journal=False``: none), or ``_ABSENT`` when no layer sets one."""
    for _w, layer in layers:
        j = _journal_setting(layer)
        if j is not _ABSENT:
            return j
    return _ABSENT


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
    j = closest_journal(layers)
    return Receivers(observers, None if j is _ABSENT else j)


def set_inside(layers: List[Tuple[str, Mapping[str, Any]]], at_start: "LayersAtStart") -> Any:
    """The journal setting of the layers around a call inside a tree that were
    not there when the tree started (its program's own, a block entered in
    the tree, ``configure`` changed meanwhile), closest first: a Journal,
    None, or ``_ABSENT`` when none of them sets one."""
    for where, layer in layers:
        if where == "block" and id(layer) in at_start.blocks:
            return _ABSENT                       # a host layer the tree started under, and all beyond it
        if where == "configure":
            j = _journal_setting(layer)
            return _ABSENT if j is _ABSENT or j == at_start.configure else j
        j = _journal_setting(layer)
        if j is not _ABSENT:
            return j
    return _ABSENT


@dataclasses.dataclass(frozen=True)
class LayersAtStart:
    """What of the host's layers a tree started under: its blocks (by identity)
    and configure's journal setting."""
    blocks: frozenset
    configure: Any

    @classmethod
    def of(cls, layers: List[Tuple[str, Mapping[str, Any]]]) -> "LayersAtStart":
        blocks = frozenset(id(layer) for where, layer in layers if where == "block")
        conf = next((_journal_setting(layer) for where, layer in layers if where == "configure"), _ABSENT)
        return cls(blocks, conf)


# ------------------------------------------------------------------ the tree log, in this process


def _now_iso() -> str:
    from .calllog import _iso
    return _iso(time.time())


_FAULT = object()


class TreeLog:
    """The writer of one call tree's log in this process: it numbers every
    event as it happens (``seq`` dense over the whole log, ``at`` never going
    back) and hands each to the streams, observers and journal that see its
    call, in the form each may see: a stream the whole log of its call and
    the calls inside it; an observer and the journal the kept form (each
    call's ``log_content``), each with its own ``after`` chain and its own
    copy.

    If an event cannot be put in its kept form (a fault of this process),
    the kept form stops there: the journal gets nothing more (a required one
    says ``"refused"`` at its next barrier) and observers get nothing more of
    this tree, as when a writer stops. Nothing is skipped or invented."""

    def __init__(self, tree: str, *, journal: Optional[Journal] = None, writer: int = 1,
                 at_start: Optional[LayersAtStart] = None):
        self.tree = tree
        self.writer = writer
        self.seq = 0
        self.last: Optional[Dict[str, int]] = None
        self.at = ""
        self.lock = threading.RLock()
        self.journal = journal
        self.at_start = at_start or LayersAtStart(frozenset(), _ABSENT)
        self.sender: Optional[JournalWriter] = None
        self.links: Dict[int, Optional[Dict[str, int]]] = {}      # each sink's last position in its form
        self.feeds: Dict[int, _Feed] = {}                         # the observers' feeds this tree holds
        self.kept_stopped: Optional[str] = None
        self.closed = False
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
            self._deliver(call, event, journal_only=hold)
            return event

    def _kept(self, call: Any, event: Any) -> Any:
        """The event's kept form (None: the kept form has no such event;
        ``_FAULT``: it could not be made, and the kept form stops)."""
        if self.kept_stopped is not None:
            return _FAULT
        keep = getattr(call, "keep", None)
        try:
            if keep is None:
                raise RuntimeError("which values this call's log keeps is not known")
            data = event.to_dict(content=whole(keep), keep=keep, program=call.info)
            return kept_event(data, keep, call.info)
        except Exception as exc:  # noqa: BLE001 — fail closed: the kept form stops, nothing is guessed
            self.kept_stopped = (f"an event of {call.function} could not be put in its kept form "
                                 f"({type(exc).__name__}); the tree's kept log stops there")
            _warn_once(("kept", type(exc).__name__), self.kept_stopped)
            if self.sender is not None:
                self.sender.fault(self.kept_stopped)
            return _FAULT

    def _deliver(self, call: Any, event: Any, *, journal_only: bool = False, readers_only: bool = False) -> None:
        kept: Any = _ABSENT
        if not readers_only and self.sender is not None:
            kept = self._kept(call, event)
            if kept is not None and kept is not _FAULT:
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
            try:
                stream._receive(linked, call)
            except Exception as exc:  # noqa: BLE001 — a stream's reader never stands in the call's way
                _warn_once(("stream", type(exc).__name__), f"a stream of {call.function} could not take an event "
                                                           f"({type(exc).__name__})")
        if call.observers:
            if kept is _ABSENT:
                kept = self._kept(call, event)
            if kept is None or kept is _FAULT:
                return
            for o in call.observers:
                key = id(o)
                mine = copy.deepcopy(kept)
                mine["after"] = self.links.get(key)
                self.links[key] = position(mine)
                if _is_list(o):
                    o.append(mine)
                    continue
                feed = self.feeds.get(key)
                if feed is None:
                    feed = self.feeds[key] = _acquire(o)
                feed.put(mine)

    def release(self, call: Any, event: Any) -> None:
        """Give readers an event held back until the journal confirmed it."""
        if event is not None:
            with self.lock:
                self._deliver(call, event, readers_only=True)

    def barrier(self, cancelled: Optional[Callable[[], bool]] = None) -> Status:
        return self.sender.barrier(cancelled=cancelled) if self.sender is not None else "confirmed"

    def end(self) -> Status:
        """After the outermost call's last event: wait for a required journal
        (a best-effort one is left to finish on its own)."""
        if self.sender is None:
            return "confirmed"
        if self.required:
            status = self.sender.barrier()
            self.sender.stop()
            return status
        self.sender.stop()
        return "confirmed"

    def close(self) -> None:
        """The tree is over in this process: let go of the observers' feeds."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            feeds, self.feeds = list(self.feeds.values()), {}
        for feed in feeds:
            _release(feed)


def _warn_once(key: Any, message: str) -> None:
    from .calllog import _warn_once as warn
    warn(key, message)


__all__ = ["Replay", "replay", "read_after", "kept_form", "kept_event", "relink", "position", "Follower",
           "Store", "MemoryStore", "settle", "Journal", "JournalWriter", "receivers", "Receivers", "TreeLog",
           "flush", "drain", "Received", "Answer", "Status", "Settled"]
