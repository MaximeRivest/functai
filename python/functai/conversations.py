"""Conversations: a program's calls that remember each other
(contract/conversations.md).

    chat = tutor.conversation("alex", store="tutoring/")   # opened by id; the same line tomorrow reopens it
    chat("Hi, I'm Alex.")                                   # called like its program: one turn
    chat.turns[-1].saw                                      # the earlier turns that answer was based on
    other = chat.continue_from(chat.turns[0])               # a branch: nothing is ever deleted
    chat.render("What is 1/2 + 1/3?")                       # the next request, nothing sent

A conversation is records in a store (``functai.stores``), appended and never
changed: a ``turn`` record before the call (the turn's id is its call's id,
known before the model is asked), ``ended`` after it, ``lease`` records
while it runs, and what resuming it needs (``reply``, ``tool``, ``waiting``,
``approval``). Memory belongs to the conversation, never to the function:
``tutor`` is unchanged and still callable on its own.

Inside a module's turn, helpers remember nothing unless the conversation
says so (``remembers={answer: "conversation"}``), and ``functai.earlier()``
is the conversation so far, as data.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import secrets
import socket
import threading
import time
from collections import deque
from contextvars import ContextVar
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import calllog, stores
from .errors import ConversationError, Waiting

FORMAT = 1
LEASE = 30.0                 # seconds a running turn's lease lasts
RENEW = 10.0                 # how often its process renews it
POLL = 0.25                  # how often a running turn looks for a stop from another process


def _now() -> float:
    return time.time()


def _iso(t: Optional[float] = None) -> str:
    return calllog._iso(_now() if t is None else t)


def _parse(text: Optional[str]) -> float:
    import datetime as _dt
    if not text:
        return 0.0
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _json(value: Any) -> Any:
    return calllog.to_json(value)[0]


# ------------------------------------------------------------------ what the model sees


@dataclasses.dataclass(frozen=True)
class Context:
    """Which earlier turns a turn is shown: every one (``last`` None), or the
    last ``last``; ``without``: inputs or outputs left out of every earlier
    turn (a bulky document, a photo)."""
    last: Optional[int] = None
    without: Tuple[str, ...] = ()

    def pick(self, turns: List[Any]) -> List[Any]:
        return list(turns) if self.last is None else list(turns[-self.last:]) if self.last > 0 else []

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"last": self.last}
        if self.without:
            out["without"] = list(self.without)
        return out


def all_turns(*, without: Iterable[str] = ()) -> Context:
    """Every earlier turn is shown (the default). ``without``: fields left out
    of every earlier turn."""
    return Context(None, tuple(without))


def last_turns(n: int, *, without: Iterable[str] = ()) -> Context:
    """Only the last ``n`` earlier turns are shown. ``without``: fields left
    out of every earlier turn."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"last_turns takes a whole number of turns, not {n!r}")
    return Context(n, tuple(without))


@dataclasses.dataclass(frozen=True)
class Memory:
    """What a helper remembers in a conversation: ``"conversation"`` (its own
    earlier calls on this branch, in earlier turns and this one) or ``"turn"``
    (its earlier calls in this turn only); ``steps``: with their tool steps."""
    mode: str
    steps: bool = False


def remember(mode: str = "conversation", *, steps: bool = False) -> Memory:
    """What a helper remembers: ``remember("conversation", steps=True)``."""
    if mode not in ("conversation", "turn"):
        raise ValueError(f"a helper remembers 'conversation' or 'turn', not {mode!r}")
    return Memory(mode, bool(steps))


def _memory(value: Any) -> Any:
    if isinstance(value, Memory) or value == "own":
        return value
    if value in ("conversation", "turn"):
        return Memory(value)
    raise ValueError(f"remembers maps a helper to 'conversation', 'turn', functai.remember(...), or 'own' (a "
                     f"conversation used inside); not {value!r}")


# ------------------------------------------------------------------ the records, read


@dataclasses.dataclass
class _TurnState:
    """What the records say of one turn."""
    record: Dict[str, Any]
    ended: Optional[Dict[str, Any]] = None
    lease: Optional[Dict[str, Any]] = None
    waiting: Optional[Dict[str, Any]] = None
    stops: int = 0
    answers: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    tools: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    replies: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    calls: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    children: List[str] = dataclasses.field(default_factory=list)
    attempt: int = 1

    @property
    def id(self) -> str:
        return self.record["turn"]

    @property
    def parent(self) -> Optional[str]:
        return self.record.get("parent")

    def state(self, now: Optional[float] = None) -> str:
        """``running``, ``waiting``, ``interrupted`` (its lease ran out with
        no end: its process stopped), or how it ended (``done``, ``failed``,
        ``stopped``, ``abandoned``)."""
        if self.ended is not None:
            return self.ended["state"]
        lease_seq = self.lease["seq"] if self.lease else 0
        if self.waiting is not None and self.waiting["seq"] > lease_seq:
            return "waiting"
        until = _parse(self.lease.get("until")) if self.lease else 0.0
        return "running" if until >= (now if now is not None else _now()) else "interrupted"

    def unanswered(self) -> List[Dict[str, Any]]:
        """The approvals the turn waits for that have no answer yet."""
        if self.waiting is None:
            return []
        done = {_akey(a) for a in self.answers if a["seq"] > self.waiting["seq"]}
        return [a for a in self.waiting.get("approvals", []) if _akey(a) not in done]

    def unfinished(self) -> List[Dict[str, Any]]:
        """Tools that started and have no result in the records: they may have run."""
        started: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for t in self.tools:
            k = (t.get("site"), t.get("invocation"))
            if t.get("state") == "started":
                started[k] = t
            elif t.get("state") in ("done", "given", "rerun"):
                started.pop(k, None)
        return list(started.values())


def _akey(a: Mapping[str, Any]) -> Tuple[Any, int, str]:
    """Which question an approval record answers: the asking call's site, the
    tool call's invocation, and the plugin that asked."""
    return (a.get("site"), int(a.get("invocation") or 0), a.get("plugin") or "approval")


class _Log:
    """A conversation's records, applied in order (read again incrementally)."""

    def __init__(self) -> None:
        self.n = 0
        self.turns: Dict[str, _TurnState] = {}
        self.order: List[str] = []
        self.head: Optional[str] = None
        self.request_ids: Dict[str, str] = {}
        self.programs: Dict[str, Dict[str, Any]] = {}
        self.entries: List[Dict[str, Any]] = []
        self.stops_seen = 0

    def apply(self, records: Sequence[Dict[str, Any]]) -> None:
        for r in records:
            self.n = max(self.n, int(r.get("seq") or self.n + 1))
            if r.get("functai_conversation") != FORMAT:
                continue                              # a format this reader does not know: skipped
            kind = r.get("kind")
            if kind == "program":
                self.programs.setdefault(r.get("version"), r)
                continue
            if kind == "entry":
                self.entries.append(r)
                continue
            tid = r.get("turn")
            if kind == "turn":
                if tid in self.turns:
                    continue
                st = self.turns[tid] = _TurnState(r)
                self.order.append(tid)
                if r.get("parent") in self.turns:
                    self.turns[r["parent"]].children.append(tid)
                if r.get("request_id"):
                    self.request_ids.setdefault(r["request_id"], tid)
                self.head = tid
                continue
            st = self.turns.get(tid)
            if kind == "head":
                if st is not None:
                    self.head = tid
                continue
            if st is None:
                continue
            if kind == "ended":
                st.ended = st.ended or r
            elif kind == "lease":
                st.lease = r
                st.attempt = max(st.attempt, int(r.get("attempt") or 1))
            elif kind == "waiting":
                st.waiting = r
            elif kind == "stop":
                st.stops += 1
            elif kind == "approval":
                st.answers.append(r)
            elif kind == "tool":
                st.tools.append(r)
            elif kind == "reply":
                st.replies.append(r)
            elif kind == "call":
                st.calls.append(r)

    def branch(self, turn: Optional[str]) -> List[_TurnState]:
        """The turns from the first to ``turn``, in order."""
        out: List[_TurnState] = []
        seen = set()
        while turn is not None and turn in self.turns and turn not in seen:
            seen.add(turn)
            out.append(self.turns[turn])
            turn = self.turns[turn].parent
        return out[::-1]

    def done_on(self, turn: Optional[str]) -> Optional[str]:
        """``turn`` when it ended ``done``, else its nearest ancestor that did:
        a turn that did not end ``done`` is never a parent (it has no answer
        to show)."""
        while turn is not None and turn in self.turns:
            if self.turns[turn].state() == "done":
                return turn
            turn = self.turns[turn].parent
        return None


# ------------------------------------------------------------------ programs, described


def _program_key(program: Any) -> Tuple[str, str]:
    from .saved import origin
    fn = getattr(program, "__wrapped__", None) or getattr(program, "_fn", None)
    return program.__name__, origin(getattr(fn, "__module__", None))[0]


def _fields(program: Any) -> List[Dict[str, Any]]:
    """An AI function's signature fields as data (names, directions, purposes,
    shapes without type names or defaults): what decides whether an earlier
    turn can be shown whole."""
    from .core import FunctAIFunc
    from .interface import data_shape
    if not isinstance(program, FunctAIFunc):
        iface = program.interface
        return [{"name": f["name"], "direction": d[:-1], "purpose": "plain",
                 "shape": data_shape(f["shape"])} for d in ("inputs", "outputs") for f in iface[d]]
    out = []
    for f in program._spec().signature.fields:
        if f.direction == "input" and f.purpose != "plain":
            continue
        out.append({"name": f.name, "direction": f.direction, "purpose": f.purpose,
                    "shape": data_shape(f.shape)})
    return out


def describe(program: Any) -> Dict[str, Any]:
    """A program's descriptor record: its name, kind, version, signature, whole
    interface and fields (contract/conversations.md, ``program``)."""
    info = calllog.program_info(program)
    out = {"functai_conversation": FORMAT, "kind": "program", "at": _iso(), "version": info["version"],
           "name": info["name"], "program_kind": info["kind"], "module": info["module"],
           "interface": program.interface, "fields": _fields(program), "answer": info["answer"]}
    if "signature" in info:
        out["signature"] = info["signature"]
    return out


def _check_signature(program: Any, log: _Log, turns: List[_TurnState], earlier_without: Sequence[str]) -> None:
    """Refuse a program that cannot be shown its earlier turns (decision 20):
    it now writes an output they lack, unless ``earlier_without`` names it and
    it is one FunctAI adds (reasoning, tool calls); or a field changed its
    shape or went away."""
    now = {f["name"]: f for f in _fields(program)}
    seen = set()
    for st in turns:
        version = st.record.get("program")
        if version in seen or version not in log.programs:
            continue
        seen.add(version)
        was = {f["name"]: f for f in log.programs[version].get("fields", [])}
        if was == now:
            continue
        changed = sorted(n for n in set(was) & set(now) if calllog.canonical(was[n]) != calllog.canonical(now[n]))
        gone = sorted(set(was) - set(now))
        new = sorted(set(now) - set(was))
        allowed = {n for n in new if now[n]["purpose"] in ("reasoning", "tools.calls") and n in earlier_without}
        unexplained = [n for n in new if n not in allowed]
        if changed or gone or unexplained:
            parts = []
            if unexplained:
                hidden = [n for n in unexplained if now[n]["purpose"] in ("reasoning", "tools.calls")]
                parts.append(f"it now writes {', '.join(unexplained)}, which earlier turns lack"
                             + (f" (to go on: earlier_without={hidden!r}; earlier turns are shown without them, "
                                f"nothing is rewritten)" if hidden else ""))
            if changed:
                parts.append(f"{', '.join(changed)} changed type")
            if gone:
                parts.append(f"{', '.join(gone)} is no longer one of its fields")
            raise ConversationError("conversation-signature",
                                    f"{program.__name__}: its earlier turns in this conversation were made with other "
                                    f"inputs or outputs: {'; '.join(parts)}")


# ------------------------------------------------------------------ the turn running in this process


class TurnWaiting(BaseException):
    """A turn stops to wait for a person (a BaseException, like ``Cancelled``:
    code inside the call must not catch it and carry on)."""

    code = "turn-waiting"

    def __init__(self, approval: Any):
        super().__init__(f"{approval.path} waits for a person's answer")
        self.approval = approval


_ACTIVE: ContextVar[Optional["_TurnRun"]] = ContextVar("functai_turn", default=None)
_PREPARING: ContextVar[Any] = ContextVar("functai_preparing", default=None)
_RENDERING: ContextVar[Any] = ContextVar("functai_rendering", default=None)
_REPLAY: ContextVar[Any] = ContextVar("functai_replay", default=None)
_APPROVALS_TO: ContextVar[str] = ContextVar("functai_approvals_to", default="owner")


def approvals_to() -> str:
    """Whom an approval is addressed to: ``"owner"``, or ``"caller"`` when a
    served program lets its caller answer (contract/serving.md)."""
    return _APPROVALS_TO.get()


def running(parent: Any) -> Optional["_TurnRun"]:
    """The turn a call made now runs in: its parent's, or, for an outermost
    call, the turn this context is starting."""
    active = _ACTIVE.get()
    if active is not None and not active.root_taken:
        return active                                 # a turn starting now: its own call comes first
    if parent is not None:
        return getattr(parent, "turn_run", None)
    return active


class _TurnRun:
    """One turn being run by this process: its place, what its calls are
    shown, what resuming it replays, and the records it writes as it goes."""

    def __init__(self, conv: "Conversation", turn: str, *, attempt: int, context: Dict[str, Any],
                 replay: Optional[Dict[str, Any]] = None, later: Optional[Dict[str, Any]] = None):
        self.conv = conv
        self.store = conv.store
        self.program = conv.program
        self.turn = turn
        self.attempt = attempt
        self.context = context                          # {"turns", "ids", "entries_fixed", "rows"}
        self.root: Any = None                           # the turn's call (calllog.Call)
        self.root_taken = False
        self.helper_calls: List[Dict[str, Any]] = []   # remembered helpers' calls made in this turn so far
        self.usage: Dict[str, int] = {}
        self.later = later
        self.start_seq = 0                              # the records before this run's own lease: older stops are not its
        self.settings: Dict[str, Any] = {}             # the turn's settings (the conversation's and its own)
        self.holder = _holder()
        self.lock = threading.Lock()
        self.durable = conv._durable()
        r = replay or {}
        self.replies: Dict[str, Deque[Dict[str, Any]]] = {}
        for rec in r.get("replies", []):
            self.replies.setdefault(rec["key"], deque()).append(rec["response"])
        self.tools: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.unfinished: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for t in r.get("tools", []):
            k = (t.get("site"), int(t.get("invocation") or 0))
            if t.get("state") in ("done", "given"):
                self.tools[k] = t
                self.unfinished.pop(k, None)
            elif t.get("state") == "rerun":
                self.tools.pop(k, None)
                self.unfinished.pop(k, None)
            elif t.get("state") == "started" and k not in self.tools:
                self.unfinished[k] = t
        self.answers: Dict[Tuple[Any, int, str], Tuple[bool, Optional[str], Optional[str], bool]] = {}
        last_wait = r.get("waiting_seq", 0)
        for a in r.get("answers", []):
            self.answers[_akey(a)] = (a.get("verdict") == "yes", a.get("reason"), a.get("by"),
                                      a.get("seq", 0) > last_wait)

    # ----- calllog's side

    def call_id_for(self, program: Any, parent: Any) -> Optional[str]:
        """The turn's id for its own call (minted before the call)."""
        if not self.root_taken and program is self.program:
            self.root_taken = True
            return self.turn
        return None

    def later_writer(self, call: Any) -> Optional[Dict[str, Any]]:
        """For a resumed turn's call: this process's claim on its log."""
        if call.id != self.turn or self.later is None:
            return None
        return self.later

    def events_sink(self, call: Any) -> Optional[Callable[[Dict[str, Any]], None]]:
        """An observer that keeps the turn's kept log in the store, so another
        process can watch it."""
        events = stores.events_of(self.store)
        if events is None or call.id != self.turn:
            return None
        warned = []

        def keep(event: Dict[str, Any]) -> None:
            try:
                events.append(event)
            except Exception as exc:  # noqa: BLE001 — watching is best effort; the turn's records are not
                if not warned:
                    warned.append(1)
                    calllog._warn_once(("conversation-events", type(exc).__name__),
                                       f"a turn's events could not be kept in its store ({type(exc).__name__}: "
                                       f"{exc}); the turn goes on, and its records are kept")
        keep.__qualname__ = "conversation store"
        keep._passive = True                          # a copy for other processes: it never streams a request
        return keep

    def call_ended(self, call: Any, value: Any, error: Optional[BaseException]) -> None:
        """A call of this turn ended: its tokens count in the turn's usage; a
        helper the conversation remembers keeps its call for later."""
        pred = call.pred
        if pred is not None:
            with self.lock:
                for k, v in pred.usage.items():
                    self.usage[k] = self.usage.get(k, 0) + v
        if call.id == self.turn:
            self.root = call
            return
        if error is not None or pred is None or getattr(pred, "turn", None) is None:
            return
        memory = self.conv._remembered(call.program)
        if not isinstance(memory, Memory):
            return
        try:
            rec = {"functai_conversation": FORMAT, "kind": "call", "at": _iso(), "turn": self.turn,
                   "attempt": self.attempt, "call": call.id, "site": call.path,
                   "program": {"name": call.program.__name__, "module": _program_key(call.program)[1],
                               "signature": calllog.signature_id(call.program._spec().signature)},
                   "lmcc": json.loads(calllog.canonical(pred.turn.to_dict())), "saw": list(call.saw)}
        except Exception:  # noqa: BLE001 — a call with no JSON form is not remembered
            calllog._warn_once(("remember", call.function), f"{call.function}'s call has a value with no JSON form: "
                                                            f"the conversation cannot remember it")
            return
        with self.lock:
            self.helper_calls.append(rec)
        self.conv._append([rec])

    # ----- what calls are shown

    def helper_context(self, program: Any, memory: Memory) -> Tuple[List[Any], List[str]]:
        """A remembered helper's own earlier calls on this branch (and in this
        turn), as turns (with their steps only when remembered with them)."""
        name, module = _program_key(program)
        found: List[Dict[str, Any]] = []
        if memory.mode == "conversation":
            log = self.conv._read()
            for st in log.branch(self.context.get("parent")):
                if st.state() != "done":
                    continue
                final = int((st.ended or {}).get("attempt") or st.attempt)
                found += [c for c in st.calls if int(c.get("attempt") or 1) == final]
        with self.lock:
            found += list(self.helper_calls)
        mine = [c for c in found if c["program"]["name"] == name and c["program"].get("module") == module]
        turns, ids = [], []
        signature = calllog.signature_id(program._spec().signature)
        for c in mine:
            t = copy.deepcopy(c["lmcc"])
            if not memory.steps:
                t["steps"] = []
            if c["program"].get("signature") != signature:
                t = {"inputs": t.get("inputs", {}), "outputs": t.get("outputs") or {}}
            turns.append(_fit(program, t))
            ids.append(c["call"])
        return turns, ids

    # ----- resuming: what was recorded

    def recorded_reply(self, key: str) -> Any:
        q = self.replies.get(key)
        if q:
            from lm15.serde import response_from_dict
            return response_from_dict(q.popleft())
        return None

    def note_reply(self, key: str, response: Any) -> None:
        if not self.durable:
            return
        from lm15.serde import response_to_dict
        self.conv._append([{"functai_conversation": FORMAT, "kind": "reply", "at": _iso(), "turn": self.turn,
                            "attempt": self.attempt, "key": key,
                            "response": json.loads(calllog.canonical(response_to_dict(response)))}])

    def recorded_tool(self, call: Any, approval: Any) -> Optional[str]:
        k = (call.path, approval.invocation)
        if k in self.tools:
            return self.tools[k].get("output")
        if k in self.unfinished:
            raise ConversationError("turn-unfinished",
                                    f"{approval.path} started before the turn stopped, and whether it ran is not "
                                    f"known: resume with results={{{approval.invocation}: <what it returned>}} or "
                                    f"rerun=[{approval.invocation}]", turn=self.turn)
        return None

    def tool_started(self, call: Any, approval: Any) -> None:
        self.conv._append([{"functai_conversation": FORMAT, "kind": "tool", "at": _iso(), "turn": self.turn,
                            "attempt": self.attempt, "site": call.path, "invocation": approval.invocation,
                            "id": approval.id, "name": approval.name, "input": _json(approval.input),
                            "effects": approval.effects, "state": "started"}])

    def tool_done(self, call: Any, approval: Any, output: str) -> None:
        self.conv._append([{"functai_conversation": FORMAT, "kind": "tool", "at": _iso(), "turn": self.turn,
                            "attempt": self.attempt, "site": call.path, "invocation": approval.invocation,
                            "id": approval.id, "name": approval.name, "state": "done", "output": output}])

    def recorded_approval(self, call: Any, approval: Any) -> Optional[Tuple[bool, Optional[str], Optional[str], bool]]:
        return self.answers.get((call.path, approval.invocation, approval.plugin))

    def note_approval(self, call: Any, approval: Any, allowed: bool, reason: Optional[str], by: Optional[str]) -> None:
        self.conv._append([{"functai_conversation": FORMAT, "kind": "approval", "at": _iso(), "turn": self.turn,
                            "site": call.path, "invocation": approval.invocation, "path": approval.path,
                            "plugin": approval.plugin, "verdict": "yes" if allowed else "no", "by": by,
                            "reason": reason}])

    def pause(self, call: Any, approval: Any) -> None:
        raise TurnWaiting(dataclasses.replace(approval, site=call.path))

    def attach(self, call: Any) -> None:
        """The turn's own call: its record says which conversation and turn it is."""
        self.root = call
        call.conversation = {"id": self.conv.id, "turn": self.turn, "parent": self.context.get("parent")}
        call.changes.extend(copy.deepcopy(self.context.get("changes") or []))


# hooks the engine calls (engine._complete)

def _run_of_current() -> Optional[_TurnRun]:
    call = calllog.current()
    return getattr(call, "turn_run", None) if call is not None else None


def recorded_reply(request: Any, settings: Mapping[str, Any]) -> Any:
    """A reply a resumed turn recorded for this very request, or None (and
    then the turn does something new: its events are shown from here)."""
    run = _run_of_current()
    if run is None:
        return None
    hit = None
    if run.replies:
        from . import replies
        hit = run.recorded_reply(replies.key(request, settings.get("replicate") or 0))
    if hit is None:
        frontier()
    return hit


def frontier() -> None:
    """The call in progress does something its turn had not done before."""
    call = calllog.current()
    log = getattr(call, "log", None)
    if log is not None and log.replaying:
        log.frontier()


def note_reply(request: Any, response: Any, settings: Mapping[str, Any]) -> None:
    """A stored turn keeps every reply, so resuming it pays for none twice."""
    run = _run_of_current()
    if run is None or not run.durable:
        return
    from . import replies
    try:
        run.note_reply(replies.key(request, settings.get("replicate") or 0), response)
    except Exception as exc:  # noqa: BLE001
        calllog._warn_once(("note-reply", type(exc).__name__), f"a reply could not be kept in the conversation "
                                                              f"({type(exc).__name__}: {exc})")


# ------------------------------------------------------------------ what a call is shown


def _fit(program: Any, turn: Dict[str, Any]) -> Dict[str, Any]:
    """A stored turn made for this program's signature (by its type-name-free
    fingerprint), with the fingerprint lmcc checks: this plan's."""
    if "signature" in turn:
        try:
            plan = program.plan()
            turn = {**turn, "signature": plan.fingerprint}
        except Exception:  # noqa: BLE001 — no model to plan for: shown by its values
            turn = {"inputs": turn.get("inputs", {}), "outputs": turn.get("outputs") or {}}
    return turn


def _without(turn: Dict[str, Any], names: Sequence[str]) -> Tuple[Dict[str, Any], List[str]]:
    """A turn with some fields left out (and so shown without its steps), and
    which of its fields were."""
    had = set(turn.get("inputs") or {}) | set(turn.get("outputs") or {})
    gone = sorted(had & set(names))
    if not gone:
        return turn, []
    return {"inputs": {k: v for k, v in (turn.get("inputs") or {}).items() if k not in gone},
            "outputs": {k: v for k, v in (turn.get("outputs") or {}).items() if k not in gone}}, gone


def context_for(program: Any) -> Optional[Tuple[List[Any], List[str], Optional[Callable[[List[Dict[str, Any]]],
                                                                                     List[Dict[str, Any]]]]]]:
    """What a call of ``program`` being prepared now (or rendered) is shown as
    earlier turns: ``(turns, their call ids, a function that finishes the saw
    entries)``, or None when it is shown nothing."""
    call = _PREPARING.get()
    replay = _REPLAY.get()
    if replay is not None and (call is None or call.program is program):
        found = replay.context_for(program, call)
        if found is not None:
            return found
    if call is not None and call.program is program:
        run = call.turn_run
        if run is None:
            return None
        if call.id == run.turn:
            ctx = run.context
            return ctx["turns"], ctx["ids"], ctx["finish"]
        memory = run.conv._remembered(program)
        if isinstance(memory, Memory):
            turns, ids = run.helper_context(program, memory)
            return turns, ids, None
        return None
    rendering = _RENDERING.get()
    if rendering is not None and rendering[0] is program:
        return rendering[1]
    return None


def sections_for(program: Any, call: Any) -> List[str]:
    """The instruction sections a call is given before its own ``before_call``
    hooks: its turn's (the conversation's context hooks gave them), a row's
    asked again, or, rendering, the next turn's."""
    replay = _REPLAY.get()
    if replay is not None and call is not None:
        return replay.sections_for(program, call)
    if call is None:
        rendering = _RENDERING.get()
        if rendering is not None and rendering[0] is program:
            return list(rendering[2] if len(rendering) > 2 else [])
        return []
    run = getattr(call, "turn_run", None)
    if run is not None and call.id == run.turn:
        return list(run.context.get("sections") or [])
    return []


def module_saw(program: Any) -> Optional[List[Dict[str, Any]]]:
    """What a module's call being prepared is shown as the conversation so far
    (its ``saw``), or None: a module's turn, or a row being asked again."""
    call = _PREPARING.get()
    if call is None or call.program is not program:
        return None
    replay = _REPLAY.get()
    if replay is not None and call.parent_call is None:
        return replay.module_saw()
    run = call.turn_run
    if run is not None and call.id == run.turn:
        return list(run.context.get("module_saw", []))
    return None


def earlier() -> List[Dict[str, Any]]:
    '''The conversation so far, as data: inside a module's turn, one row per
    earlier turn it is shown (its inputs and outputs by name); ``[]`` outside
    a conversation.

    For a helper that declares an input for it:

    ```python
    @ai
    def handoff(conversation: list[dict[str, str]]) -> str:
        """Summarize this support conversation for the person who takes it over."""

    @module
    def support(message: str) -> str:
        if topic(message) == "other":
            notify_staff(handoff(functai.earlier()))
            ...
    ```
    '''
    replay = _REPLAY.get()
    call = calllog.current()
    while call is not None and call.parent_call is not None:
        call = call.parent_call
    if replay is not None:
        return replay.rows()
    run = getattr(call, "turn_run", None) if call is not None else None
    if run is None:
        run = _ACTIVE.get()
    if run is None:
        return []
    return copy.deepcopy(run.context.get("rows", []))


# ------------------------------------------------------------------ turns


class Turn:
    """One turn of a conversation, as its records say now.

    ``id`` (also ``call``: the id of the turn's call in the call log),
    ``parent`` (the turn it continues, or None), ``inputs``, ``outputs``,
    ``result`` (the answer), ``state`` (``running``, ``waiting``,
    ``interrupted``, ``done``, ``failed``, ``stopped``, ``abandoned``),
    ``saw`` (the earlier turns it was shown), ``model``, ``error``,
    ``waiting`` (the approvals it waits for), ``unfinished`` (tools that may
    have run when it stopped), ``usage`` (tokens, summed over the calls
    inside it). Another output is an attribute: ``turn.reasoning``."""

    def __init__(self, conv: "Conversation", state: _TurnState):
        self._conv = conv
        self._st = state

    # ----- what it is

    @property
    def id(self) -> str:
        return self._st.id

    call = id

    @property
    def conversation(self) -> str:
        return self._conv.id

    @property
    def parent(self) -> Optional[str]:
        return self._st.parent

    @property
    def request_id(self) -> Optional[str]:
        return self._st.record.get("request_id")

    @property
    def inputs(self) -> Dict[str, Any]:
        return copy.deepcopy(self._st.record.get("inputs") or {})

    @property
    def outputs(self) -> Dict[str, Any]:
        return copy.deepcopy((self._st.ended or {}).get("outputs") or {})

    @property
    def result(self) -> Any:
        """The answer (as the program's code returned it), typed; None until done."""
        ended = self._st.ended or {}
        if "value" in ended:
            return self._conv._typed_answer(ended["value"])
        answer = self._conv._answer_name()
        return self._conv._typed_answer((ended.get("outputs") or {}).get(answer))

    @property
    def state(self) -> str:
        return self._st.state()

    @property
    def model(self) -> Optional[str]:
        return (self._st.ended or {}).get("model") or (self._st.record.get("settings") or {}).get("lm")

    @property
    def error(self) -> Optional[Dict[str, Any]]:
        return copy.deepcopy((self._st.ended or {}).get("error"))

    @property
    def usage(self) -> Dict[str, int]:
        return dict((self._st.ended or {}).get("usage") or {})

    @property
    def reads(self) -> List[str]:
        """A merge: the turns it was made from."""
        return list(self._st.record.get("reads") or [])

    @property
    def made_by(self) -> Optional[Dict[str, Any]]:
        """A merge: the program that made it (its rating goes there)."""
        return copy.deepcopy(self._st.record.get("made_by"))

    @property
    def saw(self) -> List["Turn"]:
        """The earlier turns this turn was shown, in order."""
        entries = (self._st.ended or {}).get("saw") or self._st.record.get("saw") or []
        log = self._conv._read()
        try:
            expanded = calllog.saw(self.id, self._conv._saw_records(log))
        except Exception:  # noqa: BLE001 — not known: say nothing rather than guess
            expanded = [e for e in entries if "call" in e]
        return [Turn(self._conv, log.turns[e["call"]]) for e in expanded if e.get("call") in log.turns]

    @property
    def waiting(self) -> List[Any]:
        from .tools import Approval
        return [Approval.from_dict(a) for a in self._st.unanswered()]

    @property
    def unfinished(self) -> List[Dict[str, Any]]:
        return [{"invocation": t["invocation"], "id": t["id"], "name": t["name"], "input": t.get("input"),
                 "started": t.get("at"), "site": t.get("site")} for t in self._st.unfinished()]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        outputs = (self._st.ended or {}).get("outputs") or {}
        if name in outputs:
            return outputs[name]
        raise AttributeError(f"a turn has no {name!r} (its outputs: {sorted(outputs)})")

    def __repr__(self) -> str:
        ins = ", ".join(f"{k}={_short(v)}" for k, v in (self._st.record.get("inputs") or {}).items())
        st = self.state
        tail = f" → {_short(self.result)}" if st == "done" else f" [{st}]"
        return f"<Turn {self.id[:8]} {ins}{tail}>"

    # ----- what can be done with it

    def wait(self, timeout: Optional[float] = None) -> "Turn":
        """Wait until the turn is no longer running; returns it as it is then."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            st = self._conv._read().turns[self.id]
            self._st = st
            if st.state() != "running":
                return self
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                raise TimeoutError(f"turn {self.id} is still running")
            self._conv._wait(min(0.5, left) if left is not None else 0.5)

    def stop(self) -> None:
        """Stop the turn wherever it runs: it ends ``stopped`` within a second."""
        self._conv.stop(self)

    def approve(self, approval: Any = None, *, by: Optional[str] = None, resume: bool = True) -> Any:
        """Say yes to an approval the turn waits for (``turn.waiting[0]``, its
        invocation number, or None for the only one). When nothing else waits,
        the turn goes on here (``resume=False``: later, ``turn.resume()``);
        returns its result."""
        return self._answer(approval, True, None, by, resume)

    def deny(self, approval: Any = None, reason: Optional[str] = None, *, by: Optional[str] = None,
             resume: bool = True) -> Any:
        """Say no: the model is told the person did not allow it (and why)."""
        return self._answer(approval, False, reason, by, resume)

    def _answer(self, approval: Any, allowed: bool, reason: Optional[str], by: Optional[str], resume: bool) -> Any:
        st = self._conv._read().turns[self.id]
        waiting = st.unanswered()
        if not waiting:
            raise ConversationError("turn-state", f"turn {self.id} waits for no approval (it is {st.state()})",
                                    turn=self.id)
        if approval is None:
            if len(waiting) > 1:
                raise ValueError(f"turn {self.id} waits for {len(waiting)} approvals: name one")
            target = waiting[0]
        else:
            inv = getattr(approval, "invocation", approval.get("invocation") if isinstance(approval, Mapping)
                          else approval)
            site = getattr(approval, "site", None) or (approval.get("site") if isinstance(approval, Mapping) else None)
            ext = getattr(approval, "plugin", None) or (approval.get("plugin") if isinstance(approval, Mapping)
                                                           else None)
            match = [a for a in waiting if int(a["invocation"]) == int(inv) and (site is None or a.get("site") == site)
                     and (ext is None or (a.get("plugin") or "approval") == ext)]
            if not match:
                raise ConversationError("turn-state", f"turn {self.id} waits for no approval {approval!r}", turn=self.id)
            target = match[0]
        self._conv._append([{"functai_conversation": FORMAT, "kind": "approval", "at": _iso(), "turn": self.id,
                             "site": target.get("site"), "invocation": target["invocation"],
                             "path": target.get("path"), "plugin": target.get("plugin") or "approval",
                             "verdict": "yes" if allowed else "no", "by": by, "reason": reason}])
        self._st = self._conv._read().turns[self.id]
        if resume and not self._st.unanswered():
            return self.resume()
        return None

    def resume(self, *, results: Optional[Mapping[Any, Any]] = None, rerun: Iterable[Any] = ()) -> Any:
        """Go on with a turn that waits (every approval answered) or was
        interrupted (its process stopped), in this process: its program runs
        again with the same inputs and earlier turns, each model reply it had
        and each tool result it kept are reused, and it goes on from where it
        stopped. A tool that started and has no result may have run:
        ``results={invocation: output}`` says what it returned, ``rerun=[invocation]``
        runs it again. Returns the turn's answer (or raises ``Waiting`` again)."""
        return self._conv._resume(self.id, dict(results or {}), list(rerun)).result

    def abandon(self) -> None:
        """End a turn that waits or was interrupted, without going on: ``abandoned``."""
        st = self._conv._read().turns[self.id]
        if st.state() not in ("waiting", "interrupted"):
            raise ConversationError("turn-state", f"only a waiting or interrupted turn can be abandoned; turn "
                                                  f"{self.id} is {st.state()}", turn=self.id)
        self._conv._append([{"functai_conversation": FORMAT, "kind": "ended", "at": _iso(), "turn": self.id,
                             "state": "abandoned", "attempt": st.attempt}])
        self._st = self._conv._read().turns[self.id]

    def events(self, after: Optional[Mapping[str, Any]] = None, *, view: str = "kept",
               timeout: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """The turn's events from its store (the kept form, or a view made from
        it: ``view="outside"``), after the event named by ``after`` (a
        position), those kept so far and then each as it is kept, until its
        last. What another process, a page after a reload, reads."""
        from . import views
        stop = lambda: self._conv._read().turns[self.id].state() in ("waiting", "interrupted", "abandoned")  # noqa
        if view == "kept":
            yield from stores.follow(self._conv.store, self.id, after, timeout=timeout, stop=stop)
            return
        # a view is made from the kept form from its first event; a reader resumes it after a position it holds
        v = views.View(view, answer_from=getattr(self._conv.program, "_answer_from", None))
        waiting_for = dict(after) if after is not None else None
        for e in stores.follow(self._conv.store, self.id, None, timeout=timeout, stop=stop):
            shown = v.apply(e)
            if shown is None:
                continue
            if waiting_for is not None:
                if shown["writer"] == waiting_for.get("writer") and shown["seq"] == waiting_for.get("seq"):
                    waiting_for = None
                continue
            yield shown
        if waiting_for is not None:
            from .errors import EventRefused
            raise EventRefused("event-unknown", f"this view of turn {self.id} has no event {after}")

    def calls(self) -> List[Dict[str, Any]]:
        """The calls inside the turn, as its kept log says: ``{"call",
        "parent", "function", "invocation", "ended"}`` each, in the order they
        started."""
        out: Dict[str, Dict[str, Any]] = {}
        events = stores.events_of(self._conv.store)
        if events is None:
            return []
        for e in events.read(self.id, None):
            if e.get("kind") == "started":
                out[e["call"]] = {"call": e["call"], "parent": e.get("parent"), "function": e.get("function"),
                                  "invocation": e.get("invocation"), "ended": None,
                                  "module": (e.get("program") or {}).get("module")}
            elif e.get("kind") in ("done", "failed") and e.get("call") in out:
                out[e["call"]]["ended"] = e["kind"]
        return list(out.values())

    def find(self, program: Any) -> List["CallRef"]:
        """The calls of ``program`` inside this turn (to rate one: ``functai.rate(turn.find(fn)[0], ...)``)."""
        name = program if isinstance(program, str) else program.__name__
        return [CallRef(c["call"], c["function"]) for c in self.calls() if c["function"] == name]

    def tree(self) -> str:
        """The calls inside the turn, as an indented tree."""
        calls = self.calls()
        kids: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for c in calls:
            kids.setdefault(c["parent"] if c["call"] != self.id else None, []).append(c)
        lines: List[str] = []

        def walk(c: Dict[str, Any], prefix: str, last: bool, top: bool) -> None:
            mark = "" if top else ("└─ " if last else "├─ ")
            state = "" if c["ended"] == "done" else f" [{c['ended'] or 'running'}]"
            lines.append(prefix + mark + c["function"] + state)
            children = kids.get(c["call"], [])
            for i, k in enumerate(children):
                walk(k, prefix + ("" if top else ("   " if last else "│  ")), i == len(children) - 1, False)

        for root in kids.get(None, []):
            walk(root, "", True, True)
        return "\n".join(lines)


class CallRef:
    """A call inside a turn, by its id (``functai.rate`` takes it)."""

    def __init__(self, call: str, function: str):
        self.call = call
        self.function = function

    def __repr__(self) -> str:
        return f"<call {self.function} {self.call[:8]}>"


def _short(v: Any, n: int = 60) -> str:
    text = v if isinstance(v, str) else repr(v)
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


# ------------------------------------------------------------------ the conversation


_FOLLOW = object()
_RUNNING: Dict[str, Tuple["_TurnRun", Any]] = {}     # turns this process runs: id → (run, stream)
_RUNNING_LOCK = threading.Lock()
_HOLDER = f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(3)}"


def _holder() -> str:
    global _HOLDER
    if f":{os.getpid()}:" not in _HOLDER:
        _HOLDER = f"{socket.gethostname()}:{os.getpid()}:{secrets.token_hex(3)}"
    return _HOLDER


class Conversation:
    '''A program's conversation: its turns, kept in a store, called like the
    program. Made by ``fn.conversation(...)`` (an AI function or a module).

    Called with the program's inputs, it makes one turn and returns its
    answer; ``predict`` gives the whole call, ``stream`` watches it. Other
    keyword arguments that are settings, not inputs, apply to that turn
    only: ``chat("Why?", lm="claude-sonnet-4-5")``. ``request_id=`` makes a
    send that is repeated (a double click) one turn.

    Parameters
    ----------
    program : AI function or module
        What answers.
    id : str, optional
        The conversation's id (letters, digits, ``.``, ``_``, ``-``). The same
        id in the same store opens the same conversation. Default: a new one.
    store : None, True, folder, or store
        Where it is kept: None (this process's memory), a folder, True (the
        default folder), or any object with ``append`` and ``read``
        (``functai.stores``).
    context : Context, optional
        Which earlier turns are shown: ``functai.all_turns()`` (default) or
        ``functai.last_turns(10)``, each with ``without=[...]``.
    earlier_without : list of str
        Outputs the program now writes that earlier turns lack (reasoning
        turned on, a first tool): earlier turns are shown without them.
    remembers : dict, optional
        A module's helpers' memory: ``{answer: "conversation"}``, ``"turn"``,
        ``functai.remember("conversation", steps=True)``, or ``"own"`` for a
        conversation used inside it. Helpers remember nothing otherwise.
    sends : str
        Two sends at once: ``"queue"`` (default: the second waits and
        continues from the first), ``"refuse"`` (``ConversationError``
        ``conversation-busy``), or ``"branch"`` (the second continues from
        the last finished turn, beside the first).
    **settings
        Settings for every turn (``approve``, ``lm``, ...).
    '''

    def __init__(self, program: Any, id: Optional[str] = None, *, store: Any = None, context: Optional[Context] = None,
                 earlier_without: Iterable[str] = (), remembers: Optional[Mapping[Any, Any]] = None,
                 sends: str = "queue", _head: Any = _FOLLOW, _delegated: bool = False, **settings: Any):
        from .config import check
        from .core import FunctAIFunc
        from .module import FunctAIModule
        if not isinstance(program, (FunctAIFunc, FunctAIModule)):
            raise TypeError(f"a conversation is with an AI function or a module, not {type(program).__name__}")
        self.program = program
        self.id = stores.check_id(id) if id is not None else calllog.new_id()
        self.store = stores.store_for(store)
        self.context = context if context is not None else all_turns()
        if not isinstance(self.context, Context):
            raise TypeError("context is functai.all_turns() or functai.last_turns(n)")
        self.earlier_without = tuple(earlier_without)
        if sends not in ("queue", "refuse", "branch"):
            raise ValueError(f"sends is 'queue', 'refuse' or 'branch', not {sends!r}")
        self.sends = sends
        self.settings = check(dict(settings), "conversation") if settings else {}
        self._remembers: Dict[Any, Any] = {}
        for k, v in (remembers or {}).items():
            if isinstance(program, FunctAIFunc):
                raise TypeError("remembers is for a module's helpers; an AI function's conversation is its own memory")
            self._remembers[k] = _memory(v)
        self._check_remembers()
        self._head = _head
        self._delegated = _delegated                  # a delegate's own conversation, inside the turn that asked
        self._exact = False                           # continue_from(turn): after that very turn
        self._log = _Log()
        self._log_lock = threading.Lock()
        self._send_lock = threading.RLock()
        self._check_content()
        iface = program.interface
        opaque = [f["name"] for d in ("inputs", "outputs") for f in iface[d] if f.get("opaque")]
        if opaque:
            raise ConversationError("conversation-opaque",
                                    f"{program.__name__}: {', '.join(opaque)} may hold values with no JSON form, and "
                                    f"a conversation keeps its turns as data (programs.md, opaque). Give "
                                    f"{'them' if len(opaque) > 1 else 'it'} a type")
        log = self._read()
        head = self._view_head(log)
        _check_signature(self.program, log, log.branch(head), self.earlier_without)

    # ----- checks when opened

    def _check_remembers(self) -> None:
        if not self._remembers:
            return
        reachable = list(self.program.ai_functions())
        for k, v in self._remembers.items():
            if v == "own":
                if not isinstance(k, Conversation) and not callable(k):
                    raise TypeError("remembers={x: 'own'}: x is a conversation (or the program it is with)")
                continue
            if not any(k is f for f in reachable):
                raise ValueError(f"remembers names {getattr(k, '__name__', k)!r}, which {self.program.__name__} "
                                 f"does not call (its AI functions: "
                                 f"{', '.join(f.__name__ for f in reachable) or 'none'})")

    def _check_content(self) -> None:
        if not stores.is_persistent(self.store):
            return
        dropped = calllog.dropped_now(self.program)
        if dropped:
            raise ConversationError("conversation-content",
                                    f"{self.program.__name__}: a log_content setting keeps "
                                    f"{', '.join(dropped)} out of every record, and this store "
                                    f"({type(self.store).__name__}) keeps a conversation's records. A conversation "
                                    f"that must remember what it may not keep refuses rather than forgets: keep it in "
                                    f"memory (store=None), or let the store keep those fields")

    def _durable(self) -> bool:
        """Whether turns record what resuming needs (their replies): a module,
        or an AI function with tools."""
        from .core import FunctAIFunc
        return not isinstance(self.program, FunctAIFunc) or bool(self.program.tools)

    def _remembered(self, program: Any) -> Any:
        for k, v in self._remembers.items():
            if k is program:
                return v
        return None

    # ----- records

    def _read(self) -> _Log:
        with self._log_lock:
            self._log.apply(self.store.read(self.id, self._log.n))
            return self._log

    def _append(self, records: Sequence[Dict[str, Any]], *, expect: Optional[int] = None) -> int:
        return self.store.append(self.id, list(records), expect=expect)

    def _wait(self, timeout: float) -> None:
        stores.wait(self.store, self.id, self._log.n, timeout)

    def _saw_records(self, log: _Log) -> Dict[str, Dict[str, Any]]:
        """Each turn as a record ``calllog.saw`` reads (its id and its saw)."""
        return {tid: {"id": tid, "saw": list((st.ended or {}).get("saw") or st.record.get("saw") or [])}
                for tid, st in log.turns.items()}

    # ----- where it continues from

    def _view_head(self, log: _Log) -> Optional[str]:
        """Where this view continues: a conversation opened by id, the
        conversation's head (its latest turn, or the turn a ``head`` record
        names); once it made a turn, or was made by ``continue_from``, its own
        branch: the latest turn that is that turn or continues it (so a turn
        another process adds after it is followed, and a branch made elsewhere
        is not)."""
        if self._head is _FOLLOW:
            return log.head
        pinned = self._head
        if self._exact:
            return pinned                             # continue_from(turn): after that very turn
        for tid in reversed(log.order):
            t: Optional[str] = tid
            while t is not None:
                if t == pinned:
                    return tid
                st = log.turns.get(t)
                t = st.parent if st is not None else None
        return pinned

    @property
    def head(self) -> Optional[Turn]:
        """The turn this conversation continues from next (None: none yet)."""
        log = self._read()
        h = self._view_head(log)
        return Turn(self, log.turns[h]) if h in log.turns else None

    @head.setter
    def head(self, turn: Any) -> None:
        """Make a turn the conversation's head, for everyone who opens it."""
        tid = self._turn_id(turn)
        self._append([{"functai_conversation": FORMAT, "kind": "head", "at": _iso(), "turn": tid}])
        self._head = tid
        self._exact = True

    @property
    def turns(self) -> List[Turn]:
        """The turns from the first to the head, in order."""
        log = self._read()
        return [Turn(self, st) for st in log.branch(self._view_head(log))]

    def all_turns(self) -> List[Turn]:
        """Every turn of every branch, in the order they were made."""
        log = self._read()
        return [Turn(self, log.turns[t]) for t in log.order]

    def turn(self, turn: Any) -> Turn:
        """One turn, by its id (or a Turn)."""
        tid = self._turn_id(turn)
        return Turn(self, self._read().turns[tid])

    def _turn_id(self, turn: Any) -> str:
        if isinstance(turn, Turn):
            return turn.id
        if isinstance(turn, int) and not isinstance(turn, bool):
            return self.turns[turn].id
        if isinstance(turn, str) and turn in self._read().turns:
            return turn
        raise ConversationError("turn-unknown", f"conversation {self.id} has no turn {turn!r}")

    def continue_from(self, turn: Any) -> "Conversation":
        """This conversation, continuing after ``turn`` (a Turn, its id, or its
        index in ``turns``): the next turn is a new branch. Nothing is deleted."""
        tid = self._turn_id(turn)
        view = object.__new__(Conversation)
        view.__dict__.update(self.__dict__)
        view._head = tid
        view._exact = True
        view._send_lock = threading.RLock()
        view._log = _Log()
        view._log_lock = threading.Lock()
        return view

    # ----- calling it

    def _split(self, args: tuple, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """(inputs by name, settings for this turn) from a call's arguments: a
        keyword that is an input is an input; one that is a setting, a setting."""
        from .config import KNOWN, check
        names = [f["name"] for f in self.program.interface["inputs"]]
        settings = {k: kwargs.pop(k) for k in list(kwargs) if k not in names and k in KNOWN}
        from .core import FunctAIFunc
        if isinstance(self.program, FunctAIFunc):
            given = self.program._bind_args(args, kwargs)
        else:
            given = self.program._given(args, kwargs)
        return given, (check(settings, "a turn") if settings else {})

    def __call__(self, *args: Any, request_id: Optional[str] = None, **kwargs: Any) -> Any:
        """One turn: the program's answer."""
        inputs, settings = self._split(args, dict(kwargs))
        return self._send(inputs, settings, request_id, passive=True).result

    def predict(self, *args: Any, request_id: Optional[str] = None, **kwargs: Any) -> Any:
        """One turn of an AI function: the whole call (``p.turn`` is the lmcc
        turn; ``p.call_id`` the turn's id)."""
        inputs, settings = self._split(args, dict(kwargs))
        return self._send(inputs, settings, request_id, passive=True).prediction

    def stream(self, *args: Any, request_id: Optional[str] = None, **kwargs: Any) -> "TurnStream":
        """One turn, watched while it is made (a ``Stream``, with ``.turn``,
        known at once: the turn is saved before the model is asked)."""
        inputs, settings = self._split(args, dict(kwargs))
        return self._send(inputs, settings, request_id)

    def _check_nested(self) -> None:
        outer = _ACTIVE.get()
        call = calllog.current()
        if call is not None and call.turn_run is not None:
            outer = call.turn_run
        if outer is None or outer.conv.id == self.id and outer.conv.store is self.store or self._delegated:
            return
        declared = outer.conv._remembers
        if any((k is self or k is self.program) and v == "own" for k, v in declared.items()):
            return
        raise ConversationError("conversation-nested",
                                f"conversation {self.id} ({self.program.__name__}) is used inside a turn of "
                                f"conversation {outer.conv.id}, which does not say so: a remembering program inside "
                                f"another is refused unless declared (remembers={{{self.program.__name__}: 'own'}})")

    def _bound_json(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """The turn's inputs as they are bound (the record holds them)."""
        from .core import FunctAIFunc
        from . import interface as _interface
        if isinstance(self.program, FunctAIFunc):
            bound = _interface.bind_call(self.program.interface, inputs, program=self.program.__name__,
                                         dropped=calllog.dropped_now(self.program))
        else:
            own = () if self.program._declared else [
                n for n, p in self.program._signature.parameters.items() if p.default is not p.empty]
            bound = _interface.bind_inputs(self.program.interface, inputs, program=self.program.__name__,
                                           has_default=own)
        return {k: _json(v) for k, v in bound.items()}

    def _send(self, inputs: Dict[str, Any], settings: Dict[str, Any], request_id: Optional[str],
              passive: bool = False) -> "TurnStream":
        self._check_nested()
        self._check_content()                         # a host's rule set since the conversation was opened holds too
        turn_settings = {**self.settings, **settings}
        inputs, start_changes = self._turn_start(inputs, turn_settings)
        values = self._bound_json(inputs)             # refused before anything is recorded
        while True:
            with self._send_lock:
                log = self._read()
                if request_id is not None and request_id in log.request_ids:
                    return self._existing(log.request_ids[request_id])
                parent = self._parent_for(log)
                if parent is not _BUSY:
                    _check_signature(self.program, log, log.branch(parent), self.earlier_without)
                    tid = calllog.new_id()
                    context = self._context(log, parent, turn_settings)
                    context["changes"] = [*start_changes, *context["changes"]]
                    desc = describe(self.program)
                    recs: List[Dict[str, Any]] = []
                    if desc["version"] not in log.programs:
                        recs.append(desc)
                    rec = {"functai_conversation": FORMAT, "kind": "turn", "at": _iso(), "turn": tid,
                           "parent": parent, "program": desc["version"], "inputs": values}
                    if request_id is not None:
                        rec["request_id"] = str(request_id)
                    if isinstance(turn_settings.get("lm"), str):
                        rec["settings"] = {"lm": turn_settings["lm"]}
                    if context["recorded"] is not None:
                        rec["context"] = context["recorded"]
                    if context["changes"]:
                        rec["changes"] = context["changes"]
                    recs.append(rec)
                    recs.append(self._lease(tid, 1))
                    try:
                        start_seq = self._append(recs, expect=log.n)
                    except ConversationError as exc:
                        if exc.code != "store-conflict":
                            raise
                        continue                      # someone appended meanwhile: read again
                    self._head = tid                  # from now on, this view follows its own branch
                    self._exact = False
                    break
            self._wait(0.25)                          # a turn before it is running: queue behind it
        run = _TurnRun(self, tid, attempt=1, context=context)
        run.settings = dict(turn_settings)
        run.start_seq = start_seq
        return self._start(run, inputs, turn_settings, passive=passive)

    def _turn_start(self, inputs: Dict[str, Any], settings: Mapping[str, Any]
                    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """The ``turn_start`` hooks: the turn's inputs as they leave them."""
        from . import plugins
        exts = plugins.around(self.program, [("block", settings)])
        if not any(e.handlers.get("turn_start") for e in exts):
            return inputs, []
        log = self._read()
        event = plugins.TurnStart(dict(inputs), self.id, self._view_head(log), self.program)
        applied = plugins.Applied()
        names = {f["name"] for f in self.program.interface["inputs"]}

        def apply(c: "plugins.Change") -> None:
            unknown = sorted(set(c.inputs) - names)
            if unknown:
                raise ValueError(f"{self.program.__name__} has no input {unknown[0]!r}")
            event.inputs.update(c.inputs)

        plugins.run("turn_start", exts, event, applied, apply)
        return event.inputs, applied.items

    def _turn_end(self, run: "_TurnRun", outcome: Dict[str, Any]) -> None:
        """The ``turn_end`` hooks (they hear; they may keep entries at the
        turn). Run before the turn's end is recorded, so the next turn, which
        waits for that end, sees what they kept. One that fails is reported,
        and changes nothing of the turn."""
        from . import plugins
        settings = {**self.settings, **(run.settings or {})}
        exts = plugins.around(self.program, [("block", settings)])
        if not any(e.handlers.get("turn_end") for e in exts):
            return
        log = self._read()
        st = log.turns[run.turn]
        event = plugins.TurnEnd(run.turn, outcome["state"], dict(st.record.get("inputs") or {}),
                                   dict(outcome.get("outputs") or {}), self, st.parent)
        for ext in exts:
            for fn in ext.handlers.get("turn_end", ()):
                event._ext = ext
                try:
                    fn(event)
                except Exception as exc:  # noqa: BLE001 — a hook that only hears never changes the turn
                    calllog._warn_once(("turn_end", ext.name, type(exc).__name__),
                                       f"plugin {ext.name} failed in turn_end ({type(exc).__name__}: {exc})")
                finally:
                    event._ext = None

    def _parent_for(self, log: _Log) -> Any:
        """The turn a new turn continues from, or ``_BUSY`` while it must wait."""
        head = self._view_head(log)
        if head is None:
            return None
        st = log.turns.get(head)
        if st is None:
            return None
        state = st.state()
        if state == "waiting" and self.sends != "branch":
            raise ConversationError("conversation-busy", f"conversation {self.id}: turn {head} waits for a person's "
                                                         f"answer (answer it, or continue from another turn)",
                                    turn=head)
        if state in ("running", "waiting"):
            if self.sends == "queue":
                return _BUSY
            if self.sends == "refuse":
                raise ConversationError("conversation-busy", f"conversation {self.id}: turn {head} is {state}",
                                        turn=head)
            return log.done_on(st.parent)
        return log.done_on(head)

    def _lease(self, tid: str, attempt: int) -> Dict[str, Any]:
        return {"functai_conversation": FORMAT, "kind": "lease", "at": _iso(), "turn": tid, "holder": _holder(),
                "until": _iso(_now() + LEASE), "attempt": attempt}

    def _shown(self, log: _Log, parent: Optional[str], settings: Mapping[str, Any],
               fixed: Optional[Mapping[str, Any]]) -> Tuple[List["_TurnState"], Dict[str, List[str]], List[str],
                                                         List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """(the earlier turns shown, fields left out of each, sections, the
        changes made, the context to record): the conversation's rule picks
        among the done turns of the branch, then the ``context`` hooks change
        it (contract/plugins.md). ``fixed``: the context a turn recorded
        when it was made (resuming it shows exactly that)."""
        from . import plugins
        done = [st for st in log.branch(parent) if st.state() == "done"]
        if fixed is not None:
            by_id = {st.id: st for st in done}
            picked = [by_id[t] for t in fixed.get("turns", []) if t in by_id]
            without = {k: list(v) for k, v in (fixed.get("without") or {}).items()}
            return picked, without, list(fixed.get("sections") or []), [], None
        picked = self.context.pick(done)
        rule = list(self.context.without)
        without: Dict[str, List[str]] = {}
        for st in picked:
            fields = set(st.record.get("inputs") or {}) | set(((st.ended or {}).get("outputs") or {}))
            gone = sorted(fields & set(rule))
            if gone:
                without[st.id] = gone
        exts = plugins.around(self.program, [("block", settings)])
        if not any(e.handlers.get("context") for e in exts):
            return picked, without, [], [], None
        event = plugins.Context([plugins.ShownTurn(st.id, dict(st.record.get("inputs") or {}),
                                                         dict((st.ended or {}).get("outputs") or {}),
                                                         tuple(without.get(st.id, ()))) for st in picked],
                                   [], self, parent, self.program)
        applied = plugins.Applied()
        state = {"picked": picked}
        branch_done = {st.id: st for st in done}

        def apply(c: "plugins.Change") -> None:
            if c.keep is not None:
                ids = list(c.keep)
                unknown = [i for i in ids if i not in branch_done]
                if unknown:
                    raise ValueError(f"keep names {unknown[0]}, which is not a done turn of this branch")
                state["picked"] = [st for st in done if st.id in set(ids)]
                event.turns = [t for t in event.turns if t.id in set(ids)] + [
                    plugins.ShownTurn(i, dict(branch_done[i].record.get("inputs") or {}),
                                         dict((branch_done[i].ended or {}).get("outputs") or {}))
                    for i in ids if i not in {t.id for t in event.turns}]
            if c.without is not None:
                targets = c.without if isinstance(c.without, Mapping) else {st.id: list(c.without)
                                                                            for st in state["picked"]}
                for tid, names in targets.items():
                    if isinstance(names, str) or not all(isinstance(n, str) for n in names):
                        raise TypeError("without names fields: a list of names, or {turn id: [names]}")
                    without[tid] = sorted(set(without.get(tid, [])) | set(names))
            if c.sections is not None:
                event.sections.extend(plugins._texts(c.sections))

        plugins.run("context", exts, event, applied, apply)
        picked = state["picked"]
        without = {k: v for k, v in without.items() if k in {st.id for st in picked}}
        recorded = {"turns": [st.id for st in picked], "without": without, "sections": list(event.sections)}
        return picked, without, list(event.sections), applied.items, recorded

    def _context(self, log: _Log, parent: Optional[str], settings: Optional[Mapping[str, Any]] = None,
                 fixed: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """What the turn after ``parent`` is shown: the earlier turns
        (``_shown``), each as the lmcc turn it was (with its steps), the fields
        left out taken out; the sections; the ``saw`` entries; the rows
        ``earlier()`` gives; the changes plugins made, and the context to
        record with the turn."""
        from .core import FunctAIFunc
        picked, without, sections, changes, recorded = self._shown(log, parent, settings or self.settings, fixed)
        turns: List[Any] = []
        ids: List[str] = []
        rows: List[Dict[str, Any]] = []
        dropped: Dict[str, List[str]] = {}
        is_ai = isinstance(self.program, FunctAIFunc)
        signature = calllog.signature_id(self.program._spec().signature) if is_ai else None
        for st in picked:
            ended = st.ended or {}
            outputs = ended.get("outputs") or {}
            row = {**(st.record.get("inputs") or {}), **outputs}
            rows.append({k: v for k, v in row.items() if k not in set(without.get(st.id, ()))})
            if not is_ai:
                ids.append(st.id)
                continue
            desc = log.programs.get(st.record.get("program")) or {}
            if "lmcc" in ended and desc.get("signature") == signature:
                t = _fit(self.program, copy.deepcopy(ended["lmcc"]))
            else:
                t = {"inputs": st.record.get("inputs") or {}, "outputs": outputs}
            t, gone = _without(t, without.get(st.id, ()))
            if gone:
                dropped[st.id] = gone
            turns.append(t)
            ids.append(st.id)
        saw_records = self._saw_records(log)

        def finish(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for e in entries:
                e = dict(e)
                gone = dropped.get(e.get("call"))
                if gone:
                    e["without"] = sorted(set(e.get("without") or []) | set(gone))
                    e.pop("steps", None)
                out.append(e)
            return _compress(out, parent, saw_records)

        base = {"rows": rows, "parent": parent, "sections": sections, "changes": changes, "recorded": recorded}
        if not is_ai:
            entries = finish([{"call": i} for i in ids])
            return {**base, "turns": [], "ids": [], "finish": lambda _e: entries, "module_saw": entries}
        return {**base, "turns": turns, "ids": ids, "finish": finish}

    # ----- plugin entries

    def remember(self, plugin: str, kind: str, data: Any, *, turn: Optional[str] = None) -> None:
        """Keep a plugin's entry in this conversation, at a turn (it then
        belongs to the branches through that turn): what the plugin needs
        later, never shown to the model by itself (a summary is shown when a
        ``context`` hook makes it a section)."""
        if not isinstance(kind, str) or not kind:
            raise TypeError("an entry's kind is a name")
        rec = {"functai_conversation": FORMAT, "kind": "entry", "at": _iso(), "plugin": plugin,
               "entry": kind, "data": _json(data)}
        if turn is not None:
            rec["turn"] = turn
        self._append([rec])

    def entries(self, plugin: str, kind: str, *, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """A plugin's entries of a kind on the branch through ``branch``
        (default: this view's head), oldest first: ``{"turn", "data", "at"}``."""
        log = self._read()
        through = branch if branch is not None else self._view_head(log)
        on = {st.id for st in log.branch(through)}
        return [{"turn": e.get("turn"), "data": copy.deepcopy(e.get("data")), "at": e.get("at")}
                for e in log.entries if e.get("plugin") == plugin and e.get("entry") == kind
                and (e.get("turn") is None or e.get("turn") in on)]

    # ----- running a turn

    def _start(self, run: _TurnRun, inputs: Dict[str, Any], settings: Dict[str, Any], *,
               passive: bool = False) -> "TurnStream":
        from .config import forced
        token = _ACTIVE.set(run)
        try:
            with forced(**settings) if settings else _nothing():
                s = TurnStream(self, run, inputs, passive=passive)
        finally:
            _ACTIVE.reset(token)
        return s

    def _existing(self, tid: str) -> Any:
        with _RUNNING_LOCK:
            hit = _RUNNING.get(tid)
        if hit is not None:
            return hit[1]
        return StoredStream(self, tid)

    def _resume(self, tid: str, results: Dict[Any, Any], rerun: List[Any]) -> "TurnStream":
        with self._send_lock:
            while True:
                log = self._read()
                st = log.turns.get(tid)
                if st is None:
                    raise ConversationError("turn-unknown", f"conversation {self.id} has no turn {tid}")
                state = st.state()
                if state not in ("waiting", "interrupted"):
                    raise ConversationError("turn-state", f"turn {tid} is {state}: only a waiting or interrupted turn "
                                                          f"goes on", turn=tid)
                if st.unanswered():
                    raise ConversationError("turn-state", f"turn {tid} still waits for "
                                                          f"{len(st.unanswered())} approval(s)", turn=tid)
                given: List[Dict[str, Any]] = []
                for t in st.unfinished():
                    inv = int(t["invocation"])
                    keys = {inv, str(inv), t.get("id")}
                    hit = next((v for k, v in results.items() if k in keys), _NONE)
                    if hit is not _NONE:
                        given.append({**_tool_base(tid, t), "state": "given", "output": str(hit)})
                    elif any(r in keys for r in rerun):
                        given.append({**_tool_base(tid, t), "state": "rerun"})
                    else:
                        raise ConversationError("turn-unfinished",
                                                f"turn {tid}: {t['name']} (invocation {inv}) started and may have run: "
                                                f"resume(results={{{inv}: <what it returned>}}) or "
                                                f"resume(rerun=[{inv}])", turn=tid)
                attempt = st.attempt + 1
                try:
                    start_seq = self._append([*given, self._lease(tid, attempt)], expect=log.n)
                except ConversationError as exc:
                    if exc.code == "store-conflict":
                        continue
                    raise
                break
        log = self._read()
        st = log.turns[tid]
        later = None
        events = stores.events_of(self.store)
        if events is not None and callable(getattr(events, "claim", None)):
            try:
                claim = events.claim(tid)
                kept = events.read(tid, None)
                requests = max((e.get("request", 0) for e in kept if e.get("kind") == "request"
                                and e.get("call") == tid), default=0)
                later = {"writer": claim["writer"], "after": claim["after"],
                         "at": kept[-1]["at"] if kept else "", "requests": requests}
            except Exception:  # noqa: BLE001 — no log to continue: this writer starts none (watching only)
                later = None
        replay = {"replies": [r for r in st.replies], "tools": list(st.tools),
                  "answers": list(st.answers), "waiting_seq": (st.waiting or {}).get("seq", 0)}
        fixed = st.record.get("context")              # what the turn was shown when it was made (context hooks ran)
        run = _TurnRun(self, tid, attempt=attempt, context=self._context(log, st.parent, self.settings, fixed),
                       replay=replay, later=later)
        run.context["changes"] = []                     # the turn's own changes are on its first record
        run.start_seq = start_seq
        inputs = copy.deepcopy(st.record.get("inputs") or {})
        settings = dict(self.settings)
        if (st.record.get("settings") or {}).get("lm"):
            settings["lm"] = st.record["settings"]["lm"]
        return self._start(run, inputs, settings, passive=True)

    # ----- the rest

    def stop(self, turn: Any) -> None:
        """Stop a running turn, wherever it runs."""
        tid = self._turn_id(turn)
        state = self._read().turns[tid].state()
        if state in ("waiting", "interrupted"):
            Turn(self, self._read().turns[tid]).abandon()      # nothing runs it: stopping it is ending it
            return
        if state != "running":
            return                                            # it ended already
        self._append([{"functai_conversation": FORMAT, "kind": "stop", "at": _iso(), "turn": tid}])
        with _RUNNING_LOCK:
            hit = _RUNNING.get(tid)
        if hit is not None:
            hit[1].close()

    def render(self, *args: Any, call: Any = None, **kwargs: Any) -> Any:
        """The exact request the next turn would send (nothing is sent or
        recorded). ``call=helper``: the request that helper would get, with
        the memory the conversation gives it (inputs: the helper's)."""
        from .core import FunctAIFunc
        log = self._read()
        parent = self._view_head(log)
        parent = log.done_on(parent) if parent is not None else None
        if call is not None:
            memory = self._remembered(call)
            run = _TurnRun(self, calllog.new_id(), attempt=1, context={"parent": parent, "turns": [], "ids": []})
            found = run.helper_context(call, memory) if isinstance(memory, Memory) else ([], [])
            token = _RENDERING.set((call, (found[0], found[1], None)))
            try:
                return call.render(*args, **kwargs)
            finally:
                _RENDERING.reset(token)
        if not isinstance(self.program, FunctAIFunc):
            raise TypeError(f"{self.program.__name__} is a module: render the request one of its helpers would get, "
                            f"chat.render(..., call=helper)")
        ctx = self._context(log, parent, self.settings)
        token = _RENDERING.set((self.program, (ctx["turns"], ctx["ids"], None), ctx["sections"]))
        try:
            return self.program.render(*args, **kwargs)
        finally:
            _RENDERING.reset(token)

    def merge(self, branches: Sequence[Any], fn: Any, **inputs: Any) -> Turn:
        """A turn after this conversation's head made from several branches by
        another AI function (``fn``): its answer becomes this program's
        answer, recorded as a turn (``made_by`` fn, ``reads`` the branches), so
        the next turn sees it. ``inputs``: ``fn``'s, by name; when ``fn`` has
        one input and none is given, it is given the branches' answers
        (``[{"model", <inputs>…, <outputs>…}]``)."""
        from .core import FunctAIFunc
        if not isinstance(self.program, FunctAIFunc):
            raise TypeError("merge makes an AI function's turn; a module's conversation cannot take one")
        log = self._read()
        read = [Turn(self, log.turns[self._turn_id(b)]) for b in branches]
        if not read:
            raise ValueError("merge needs the turns to merge")
        parent = self._view_head(log)
        for t in read:
            if t.parent != parent:
                raise ConversationError("turn-state", f"turn {t.id} does not continue from {parent}: a merge reads "
                                                      f"branches of the turn it follows", turn=t.id)
            if t.state != "done":
                raise ConversationError("turn-state", f"turn {t.id} is {t.state}", turn=t.id)
        if not inputs:
            names = [f["name"] for f in fn.interface["inputs"]]
            if len(names) != 1:
                raise TypeError(f"{fn.__name__} takes {len(names)} inputs: give them by name")
            inputs = {names[0]: [{"model": t.model, **t.inputs, **t.outputs} for t in read]}
        p = fn.predict(**inputs)
        answer = self._answer_name()
        value = p.get(fn._spec().main)
        tid = calllog.new_id()
        base = read[0]
        plan = self.program.plan()
        try:
            lmcc_turn = plan.example(self.program._bind_inputs((), base.inputs), {answer: value})
            lmcc_json = json.loads(calllog.canonical(lmcc_turn.to_dict()))
        except Exception as exc:  # noqa: BLE001
            raise TypeError(f"{fn.__name__}'s answer does not fit {self.program.__name__}'s answer: {exc}") from None
        desc = describe(self.program)
        recs = [] if desc["version"] in log.programs else [desc]
        recs += [{"functai_conversation": FORMAT, "kind": "turn", "at": _iso(), "turn": tid, "parent": parent,
                  "program": desc["version"], "inputs": base.inputs, "reads": [t.id for t in read],
                  "made_by": {"name": fn.__name__, "version": fn.version, "call": p.call_id}},
                 {"functai_conversation": FORMAT, "kind": "ended", "at": _iso(), "turn": tid, "state": "done",
                  "outputs": {answer: _json(value)}, "value": _json(value), "lmcc": lmcc_json, "saw": [],
                  "model": None, "attempt": 1}]
        self._append(recs)
        self._head = tid
        self._exact = False
        return Turn(self, self._read().turns[tid])

    def _answer_name(self) -> str:
        return self.program.interface["outputs"][-1]["name"]

    def _typed_answer(self, value: Any) -> Any:
        from .core import FunctAIFunc
        if value is None:
            return None
        if isinstance(self.program, FunctAIFunc):
            from .signature import coerce
            spec = self.program._spec()
            try:
                return coerce(spec.annotations.get(spec.main), value)
            except Exception:  # noqa: BLE001 — shown as kept
                return value
        return value

    def __repr__(self) -> str:
        lines = [f"<Conversation {self.id} with {self.program.__name__}: {len(self.turns)} turns>"]
        for i, t in enumerate(self.turns, 1):
            ins = " ".join(_short(v, 50) for v in t.inputs.values())
            out = _short(t.result, 70) if t.state == "done" else f"[{t.state}]"
            lines.append(f"{i:>3}. {ins} → {out}")
        return "\n".join(lines)


_BUSY = object()
_NONE = object()


class _nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _tool_base(tid: str, t: Dict[str, Any]) -> Dict[str, Any]:
    return {"functai_conversation": FORMAT, "kind": "tool", "at": _iso(), "turn": tid, "site": t.get("site"),
            "invocation": t["invocation"], "id": t.get("id"), "name": t.get("name")}


def _compress(entries: List[Dict[str, Any]], parent: Optional[str], records: Dict[str, Dict[str, Any]]
              ) -> List[Dict[str, Any]]:
    """``[{"saw_of": parent}, <parent's entry>]`` when the entries are exactly
    what the parent saw, then the parent (contract/calls.md, *Saw*)."""
    if len(entries) < 2 or parent is None or entries[-1].get("call") != parent or parent not in records:
        return entries
    try:
        before = calllog.saw(parent, records)
    except Exception:  # noqa: BLE001
        return entries
    if before == entries[:-1]:
        return [{"saw_of": parent}, entries[-1]]
    return entries


# ------------------------------------------------------------------ streams of turns


from .streaming import Stream  # noqa: E402 — streaming imports nothing of this module


class TurnStream(Stream):
    """A turn, watched while it is made: a ``Stream`` with ``.turn`` (known at
    once), ``approve``/``deny`` for approvals it waits for in this process.
    Its call runs in the background; when it ends, the turn's ``ended`` (or
    ``waiting``) record is written before its result is given."""

    def __init__(self, conv: Conversation, run: _TurnRun, inputs: Dict[str, Any], *, passive: bool = False):
        self._conv = conv
        self._run = run
        self._passive = passive                       # read for its result only: replies are not streamed
        self.turn_id = run.turn
        self._stop_seen = False
        with _RUNNING_LOCK:
            _RUNNING[run.turn] = (run, self)
        self._t_start = time.perf_counter()
        from .core import FunctAIFunc
        if isinstance(conv.program, FunctAIFunc):
            super().__init__(conv.program, (), inputs)
        else:
            super().__init__(conv.program, (), dict(inputs))
        self._beat = threading.Thread(target=self._heartbeat, daemon=True, name=f"functai-turn-{run.turn[:8]}")
        self._beat.start()

    @property
    def turn(self) -> Turn:
        return self._conv.turn(self.turn_id)

    def _receive(self, event: Any, call: Any) -> None:
        # a resumed turn's log goes on without showing its call's start again: this stream knows it anyway
        if self._root is None and call.id == self.turn_id and getattr(event, "kind", "") != "started":
            with self._cond:
                self._root = call.id
                self._programs[call.id] = call.program
                self._answer_names[call.id] = call.answer
                self._fields[call.id] = {}
                self._keeps[call.id] = (call.keep, call.info)
        super()._receive(event, call)

    def _heartbeat(self) -> None:
        """Renew the turn's lease, look for a stop from another process, and
        stop when another process took the turn over (its lease ran out here:
        a store that failed, a process that stalled)."""
        renewed = time.monotonic()
        seen = self._run.start_seq
        while not self._done:
            time.sleep(POLL)
            if self._done:
                return
            try:
                recs = self._conv.store.read(self._conv.id, seen)
                seen += len(recs)
                mine = [r for r in recs if r.get("turn") == self.turn_id]
                if any(r.get("kind") == "stop" for r in mine):
                    self._stop_seen = True
                    self.close()
                if any(r.get("kind") == "lease" and int(r.get("attempt") or 1) > self._run.attempt for r in mine):
                    calllog._warn_once(("lease-lost", self.turn_id), f"turn {self.turn_id} was taken over by another "
                                                                     f"process (its lease ran out here): this one stops")
                    self.close()
                if time.monotonic() - renewed >= RENEW:
                    renewed = time.monotonic()
                    self._conv._append([self._conv._lease(self.turn_id, self._run.attempt)])
            except Exception as exc:  # noqa: BLE001 — a store that fails meanwhile: the lease runs out
                calllog._warn_once(("heartbeat", type(exc).__name__), f"a turn's lease could not be renewed "
                                                                     f"({type(exc).__name__}: {exc})")

    def _work(self, args: tuple, kwargs: Dict[str, Any]) -> None:
        value: Any = None
        error: Optional[BaseException] = None
        try:
            if self._is_ai:
                value = self.program._invoke(args, kwargs)
            else:
                value = self.program(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — recorded, then given to the reader
            error = exc
        try:
            error = self._conclude(value, error)
        finally:
            with _RUNNING_LOCK:
                _RUNNING.pop(self.turn_id, None)
            if error is not None:
                self._finish(error=error)
            else:
                self._finish(value=value)

    def _conclude(self, value: Any, error: Optional[BaseException]) -> Optional[BaseException]:
        """Write the turn's last record; returns the error the reader gets."""
        from .streaming import Cancelled
        run = self._run
        conv = self._conv
        root = run.root
        saw = list(root.saw) if root is not None else run.context.get("module_saw", [])
        base = {"functai_conversation": FORMAT, "at": _iso(), "turn": self.turn_id, "attempt": run.attempt}
        if isinstance(error, TurnWaiting):
            a = error.approval
            conv._append([{**base, "kind": "waiting", "approvals": [{**a.to_dict(), "site": a.site}], "saw": saw}])
            return Waiting(f"turn {self.turn_id} waits for a person's answer: {a.path}({_short(a.input, 80)})",
                           turn=conv.turn(self.turn_id), approvals=conv.turn(self.turn_id).waiting)
        rec: Dict[str, Any] = {**base, "kind": "ended", "saw": saw,
                               "seconds": round(time.perf_counter() - self._t_start, 6), "usage": dict(run.usage)}
        if error is not None:
            rec["state"] = "stopped" if isinstance(error, Cancelled) else "failed"
            rec["error"] = calllog._error(error)
        else:
            rec["state"] = "done"
            try:
                outputs, lmcc_json, model = self._outputs(value)
                rec["outputs"] = outputs
                rec["value"] = _json(value)
                if lmcc_json is not None:
                    rec["lmcc"] = lmcc_json
                rec["model"] = model
            except Exception as exc:  # noqa: BLE001 — an answer with no JSON form cannot be remembered
                rec["state"] = "failed"
                rec["error"] = {"type": type(exc).__name__, "message": calllog.safe_str(exc)}
                error = exc
        conv._turn_end(run, rec)                      # before the end is recorded: the next turn sees what it keeps
        try:
            conv._append([rec])
        except Exception as exc:  # noqa: BLE001 — the turn's end not kept: its lease runs out (interrupted)
            calllog._warn_once(("ended", type(exc).__name__), f"a turn's end could not be kept in its store "
                                                             f"({type(exc).__name__}: {exc})")
        return error

    def _outputs(self, value: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
        from .interface import outputs_of
        if self._is_ai:
            done = next((e for e in reversed(self._events) if getattr(e, "kind", "") == "done"
                         and e.call == self.turn_id), None)
            pred = getattr(done, "prediction", None)
            if pred is None:
                raise TypeError("the turn's call left no prediction")
            outputs = {k: _json(v) for k, v in pred.items()}
            if getattr(pred, "tool_calls", None) is not None and "calls" in {
                    f.name for f in self.program._spec().signature.outputs}:
                outputs["calls"] = _json(pred.tool_calls)
            lmcc_json = json.loads(calllog.canonical(pred.turn.to_dict()))
            model = getattr(pred.response, "model", None) or self._last_model()
            return outputs, lmcc_json, model
        ok, named = outputs_of(self.program.interface, value)
        if not ok:
            named = {self._conv._answer_name(): value}
        return {k: _json(v) for k, v in named.items()}, None, self._last_model()

    def _last_model(self) -> Optional[str]:
        for e in reversed(self._events):
            if getattr(e, "kind", "") == "request" and getattr(e, "model", None):
                return e.model
        return None


def _answer_here(stream: Stream, approval: Any, allowed: bool, reason: Optional[str], by: Optional[str]) -> None:
    from . import tools
    mine = {e.call for e in stream._events if getattr(e, "kind", "") == "started"}
    pending = [a for a in tools.waiting_in_process() if a.call in mine]
    if approval is None:
        if len(pending) != 1:
            raise ValueError(f"{len(pending)} tool calls wait here: name one")
        target = pending[0]
    else:
        call = getattr(approval, "call", None) or (approval.get("call") if isinstance(approval, Mapping) else None)
        inv = getattr(approval, "invocation", None)
        if inv is None:
            inv = approval.get("invocation") if isinstance(approval, Mapping) else approval
        match = [a for a in pending if a.invocation == int(inv) and (call is None or a.call == call)]
        if not match:
            raise ValueError(f"no tool call waits here for {approval!r}")
        target = match[0]
    tools.answer(target.call, target.invocation, allowed, reason=reason, by=by)


class StoredStream:
    """A turn another process runs (or ran), watched through its store: the
    same reading surface as a stream (its answer's text, its events, its
    result), from the kept log."""

    def __init__(self, conv: Conversation, tid: str):
        self._conv = conv
        self.turn_id = tid

    @property
    def turn(self) -> Turn:
        return self._conv.turn(self.turn_id)

    def events(self, after: Optional[Mapping[str, Any]] = None, *, view: str = "kept") -> Iterator[Dict[str, Any]]:
        return self.turn.events(after, view=view)

    def __iter__(self) -> Iterator[str]:
        for e in self.events():
            if e.get("kind") == "text" and e.get("answer") and e.get("call") == self.turn_id:
                yield e["text"]

    @property
    def result(self) -> Any:
        t = self.turn.wait()
        return _outcome(self._conv, t)

    @property
    def prediction(self) -> Any:
        from .data import Prediction
        t = self.turn.wait()
        _outcome(self._conv, t)
        p = Prediction(dict(t.outputs))
        object.__setattr__(p, "call_id", t.id)
        return p

    def close(self) -> None:
        return None


def _outcome(conv: Conversation, t: Turn) -> Any:
    state = t.state
    if state == "done":
        return t.result
    if state == "waiting":
        raise Waiting(f"turn {t.id} waits for a person's answer", turn=t, approvals=t.waiting)
    err = t.error or {}
    raise ConversationError("turn-state", f"turn {t.id} ended {state}"
                                          + (f": {err.get('type')}: {err.get('message', '')}" if err else ""),
                            turn=t.id)


# ------------------------------------------------------------------ rows asked again with their context (stage 5)


class _Replay:
    """A rated row asked again (``evaluate``, the optimizers) with what its
    call was shown: the program's own call gets the row's ``earlier`` turns,
    each helper call the earlier turns its original call was shown (in the
    order they were made), and its record says ``saw_of`` the original. No
    conversation is read or written."""

    def __init__(self, row: Mapping[str, Any], program: Any):
        from .calllog import _meta
        self.program = program
        self.earlier = list(row.get("earlier") or [])
        self.helpers = list(row.get("helpers") or [])
        self.sections = list(row.get("sections") or [])
        self.call = _meta(row, "call")
        self.lock = threading.Lock()
        self.helper_sections: Dict[str, List[str]] = {}       # a helper call → the sections its original had

    def _turn(self, program: Any, t: Mapping[str, Any]) -> Dict[str, Any]:
        signature = calllog.signature_id(program._spec().signature)
        if t.get("steps") and t.get("signature") == signature:
            return _fit(program, {"signature": signature, "inputs": dict(t.get("inputs") or {}),
                                  "steps": copy.deepcopy(t["steps"]), "outputs": dict(t.get("outputs") or {})})
        return {"inputs": dict(t.get("inputs") or {}), "outputs": dict(t.get("outputs") or {})}

    def context_for(self, program: Any, call: Any) -> Any:
        if call is None:
            return None
        if call.parent_call is None and program is self.program:
            if not self.earlier:
                return None
            turns = [self._turn(program, t) for t in self.earlier]
            original = self.call
            return turns, [""] * len(turns), (lambda _e: [{"saw_of": original}] if original else _e)
        with self.lock:
            i = next((i for i, h in enumerate(self.helpers) if h.get("program") == program.__name__), None)
            if i is None:
                return None
            h = self.helpers.pop(i)
            self.helper_sections[call.id] = list(h.get("sections") or [])
        turns = [self._turn(program, t) for t in h.get("earlier") or []]
        if not turns:
            return None
        original = h.get("call")
        return turns, [""] * len(turns), (lambda _e: [{"saw_of": original}] if original else _e)

    def module_saw(self) -> List[Dict[str, Any]]:
        return [{"saw_of": self.call}] if self.call and self.earlier else []

    def sections_for(self, program: Any, call: Any) -> List[str]:
        if call.parent_call is None and program is self.program:
            return list(self.sections)
        with self.lock:
            return list(self.helper_sections.get(call.id, []))

    def rows(self) -> List[Dict[str, Any]]:
        return [{**(t.get("inputs") or {}), **(t.get("outputs") or {})} for t in self.earlier]


class replaying:
    """Ask a rated row again with the context its call had (``with
    replaying(row, program): program(...)``); a row with none changes nothing."""

    def __init__(self, row: Mapping[str, Any], program: Any):
        self.replay = _Replay(row, program) if (row.get("earlier") or row.get("helpers") or row.get("sections")) \
            else None

    def __enter__(self) -> None:
        self.token = _REPLAY.set(self.replay) if self.replay is not None else None

    def __exit__(self, *exc: Any) -> None:
        if self.token is not None:
            _REPLAY.reset(self.token)


# ------------------------------------------------------------------ calls being prepared (calllog's hook)


class preparing:
    """Marks the call whose record and events are being prepared (so
    ``context_for`` knows which call asks)."""

    def __init__(self, call: Any):
        self.call = call

    def __enter__(self) -> None:
        self.token = _PREPARING.set(self.call)

    def __exit__(self, *exc: Any) -> None:
        _PREPARING.reset(self.token)


__all__ = ["Conversation", "Turn", "TurnStream", "StoredStream", "Context", "all_turns", "last_turns", "remember",
           "Memory", "earlier", "CallRef"]
