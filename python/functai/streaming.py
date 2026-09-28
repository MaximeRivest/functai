"""Streaming: the same call, watched while it is made.

    for piece in summarize.stream(article):      # the answer, as it is written
        print(piece, end="", flush=True)

    s = triage.stream(message)                   # a module: every call inside it
    for event in s.events():
        ...
    s.result                                     # what the call returned

A stream runs the call in a thread of its own, started at once, in a copy
of the caller's context (so ``with configure(...)`` and the call log's
parent reach it). The call is the ordinary one: ``calllog.run`` reports
each call's start and end to the stream watching it (``calllog.WATCH``),
``engine.send`` streams each model request instead of waiting for it
(the reply it assembles is the one ``complete`` would return, so reading,
retries, the cache and the log are unchanged) and reports retries and
tools. lmcc's streaming reader (kernel §8) turns the reply's pieces into
each field's text; the reply is still read whole at the end, which is the
only authority on values. The contract is ``contract/streaming.md``.
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import dataclasses
import threading
import time
import warnings
import weakref
from typing import Any, Callable, Dict, Iterator, List, Optional

from . import calllog

_NOTHING = object()


class Cancelled(BaseException):
    """The stream was closed before its call ended.

    A ``BaseException``, like ``asyncio.CancelledError`` and for the same
    reason: code inside the call (a tool, a module's ``except Exception``)
    must not catch it and carry on."""


# ------------------------------------------------------------------ events


def _envelope(**kw: Any) -> Any:
    return dataclasses.field(kw_only=True, **kw)


@dataclasses.dataclass(frozen=True)
class Event:
    """Something that happened in a call tree (contract/streaming.md): ``call``
    is the call's id (as in the call log), ``function`` its program's name.

    Where it is in its tree's log: ``tree`` (the outermost call's id),
    ``writer`` and ``seq`` (its position), ``after`` (the position of the
    event before it in the form being read, None for the first) and ``at``
    (when it was numbered)."""
    call: str
    function: str
    tree: str = _envelope(default="")
    writer: int = _envelope(default=1)
    seq: int = _envelope(default=0)
    after: Optional[Dict[str, int]] = _envelope(default=None, compare=False)
    at: str = _envelope(default="")

    kind = "event"

    @property
    def position(self) -> Dict[str, int]:
        """The event's name in its tree's log: its writer and seq."""
        return {"writer": self.writer, "seq": self.seq}

    def to_dict(self) -> Dict[str, Any]:
        """The event as JSON data (``contract/schema/event.schema.json``, format
        2), for sending to a browser or another process."""
        out: Dict[str, Any] = {"functai_event": 2, "kind": self.kind, "tree": self.tree or self.call,
                               "writer": self.writer, "seq": self.seq,
                               "after": dict(self.after) if self.after is not None else None,
                               "at": self.at or calllog._iso(time.time()), "call": self.call,
                               "function": self.function}
        for f in dataclasses.fields(self):
            if f.name in _ENVELOPE or not f.metadata.get("json", True):
                continue
            out[f.name] = self._json(f.name, getattr(self, f.name))
        return out

    def _json(self, name: str, value: Any) -> Any:
        return value


_ENVELOPE = frozenset({"call", "function", "tree", "writer", "seq", "after", "at"})


@dataclasses.dataclass(frozen=True)
class Started(Event):
    """A call began: its ``inputs``, and the ``parent`` call it runs in;
    ``root`` (the outermost call), ``program`` (the call log's program
    object) and ``saw`` (the earlier calls it is shown) as in the call log."""
    parent: Optional[str]
    inputs: Dict[str, Any]
    root: Optional[str] = None
    program: Optional[Dict[str, Any]] = None
    content: bool = True
    saw: Any = ()
    kind = "started"

    def _json(self, name, value):
        if name == "inputs":
            return {k: calllog.to_json(v)[0] for k, v in value.items()}
        if name == "root":
            return value or self.call
        if name == "program":
            return value if value is not None else {"name": self.function, "kind": "ai", "module": "__main__",
                                                    "version": "sha256:" + "0" * 64,
                                                    "interface": "sha256:" + "0" * 64, "answer": "result"}
        if name == "saw":
            return list(value)
        return value


@dataclasses.dataclass(frozen=True)
class Request(Event):
    """The call began a request to a model (its ``request``-th): the text of
    its fields so far starts again. ``model``: the model asked."""
    request: int
    model: Optional[str] = None
    kind = "request"


@dataclasses.dataclass(frozen=True)
class Text(Event):
    """A piece of an output's text, as the model wrote it. ``answer`` is true
    when the output is the call's answer."""
    field: str
    answer: bool
    text: str
    kind = "text"

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass(frozen=True)
class Thinking(Event):
    """A piece of the model's own thinking (a thinking model's channel that
    no output of the function reads)."""
    text: str
    kind = "thinking"

    def __str__(self) -> str:
        return self.text


@dataclasses.dataclass(frozen=True)
class ToolCall(Event):
    """The model asked for a tool (shown once the request is complete)."""
    id: str
    name: str
    input: Any
    kind = "tool_call"

    def _json(self, name, value):
        return calllog.to_json(value)[0] if name == "input" else value


@dataclasses.dataclass(frozen=True)
class ToolResult(Event):
    """The tool ran; ``output`` is what the model is shown."""
    id: str
    name: str
    output: str
    kind = "tool_result"


@dataclasses.dataclass(frozen=True)
class Retry(Event):
    """The model is asked again for this call's answer: the text of its fields
    shown so far no longer counts. ``wait`` is the pause before asking."""
    reason: str
    wait: Optional[float] = None
    kind = "retry"


@dataclasses.dataclass(frozen=True)
class Done(Event):
    """A call ended: ``value`` is what it returned; ``prediction`` everything
    an AI function's call produced (None for a module)."""
    value: Any
    prediction: Any = dataclasses.field(default=None, compare=False, metadata={"json": False})
    kind = "done"

    def _json(self, name, value):
        return calllog.to_json(value)[0] if name == "value" else value


@dataclasses.dataclass(frozen=True)
class Failed(Event):
    """A call ended with an error."""
    error: BaseException
    kind = "failed"

    def _json(self, name, value):
        return calllog._error(value, True) if name == "error" else value


_KINDS = {c.kind: c for c in (Started, Request, Text, Thinking, ToolCall, ToolResult, Retry, Done, Failed)}


def make_event(kind: str, **fields: Any) -> Event:
    """An event of a kind by its name (``eventlog.TreeLog`` numbers them)."""
    return _KINDS[kind](**fields)


# ------------------------------------------------------------------ one model request, watched


class _Watch:
    """A stream's hold on the call it started: which call that is, and
    whether the stream was closed (which cancels it)."""

    def __init__(self, stream: "Stream"):
        self.stream = stream
        self.root: Optional[str] = None
        self.cancelled = threading.Event()

    def check(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled("the stream was closed")


def request(call: Any, router: Any, request_: Any, plan: Any) -> tuple:
    """Send ``request_`` streaming, showing its fields as they are written
    (the call is watched: a stream, an observer or a journal sees it).
    Returns (the assembled lm15 Response, seconds to its first piece)."""
    import lm15
    call.check()
    show = _Projection(call, plan)
    opener = getattr(router, "stream", None)
    if opener is None:                               # a client that cannot stream (a baked model)
        response = router.complete(request_)
        show.whole(response)
        return response, None
    t0 = time.perf_counter()
    first: Optional[float] = None
    try:
        rs = lm15.ResponseStream(opener(request_), request_)
        try:
            for event in rs.events():
                call.check()
                if event.type == "delta":
                    if first is None:
                        first = time.perf_counter() - t0
                    show.feed(event.delta)
            response = rs.response
        finally:
            rs.close()
    except lm15.UnsupportedFeatureError:
        if first is not None:
            raise
        response = router.complete(request_)          # this model or request cannot stream
        show.whole(response)
        return response, None
    show.finish(response.finish_reason)
    return response, first


def replay(call: Any, plan: Any, response: Any) -> None:
    """A reply that arrived whole (the reply cache): shown as one piece per field."""
    _Projection(call, plan).whole(response)


_thinking_read: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()


def _reads_thinking(plan: Any) -> bool:
    """Does an output of this plan read the model's thinking channel (module="cot")?"""
    try:
        return _thinking_read[plan]
    except (KeyError, TypeError):
        pass
    try:
        rules = plan.describe().get("streaming", {}).get("find", [])
        found = any(str(r.get("from", "")).startswith("part:thinking") for r in rules)
    except Exception:  # noqa: BLE001
        found = False
    try:
        _thinking_read[plan] = found
    except TypeError:
        pass
    return found


class _Projection:
    """One request's reply, piece by piece, as each field's text (lmcc's
    streaming reader). It only shows; the whole reply is read afterwards."""

    def __init__(self, call: Any, plan: Any):
        self.call = call
        self.answer = call.answer
        self.shown = {f.name for f in plan.signature.outputs if f.purpose in ("plain", "reasoning")}
        self.thinking_is_field = _reads_thinking(plan)
        try:
            self.reader = plan.stream()
        except Exception:  # noqa: BLE001 — no view, the call goes on
            self.reader = None

    def _show(self, events: List[Dict[str, Any]]) -> None:
        for e in events:
            if e.get("kind") == "field_delta" and e.get("field") in self.shown and e.get("text"):
                self.call.emit("text", field=e["field"], answer=e["field"] == self.answer, text=e["text"])

    def _feed(self, delta: Any) -> None:
        if self.reader is None:
            return
        try:
            self._show(self.reader.feed(delta))
        except Exception:  # noqa: BLE001 — a piece lmcc cannot place: stop showing, the reply is read whole
            self.reader = None

    def feed(self, delta: Any) -> None:
        kind = getattr(delta, "type", None)
        if kind == "thinking" and not self.thinking_is_field:
            if delta.text:
                self.call.emit("thinking", text=delta.text)
            return
        if kind in ("text", "thinking"):
            from lm15.serde import delta_to_dict
            self._feed(delta_to_dict(delta))
        # tool calls are shown whole, once read (engine.run); media parts are not text

    def whole(self, response: Any) -> None:
        from lm15.serde import message_to_dict
        for part in message_to_dict(response.message).get("parts", []):
            kind = part.get("type")
            if kind == "thinking" and not self.thinking_is_field:
                if part.get("text"):
                    self.call.emit("thinking", text=part["text"])
            elif kind in ("text", "thinking", "data"):
                self._feed(part)
        self.finish(response.finish_reason)

    def finish(self, finish_reason: Optional[str]) -> None:
        if self.reader is None:
            return
        try:
            self._show(self.reader.finish(finish_reason).events)
        except Exception:  # noqa: BLE001 — an unreadable reply: reading it whole says why, and retries
            pass
        self.reader = None


# ------------------------------------------------------------------ the answer so far


def partial_json(text: str) -> Any:
    """The JSON value that the start of a JSON text says so far: complete values
    as they are, a string still being written as its text so far, containers
    closed; a number, true, false or null only once complete. None when the
    text is not JSON so far."""
    s = text.strip()
    if s.startswith("```"):                           # a fenced block
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = s.strip()
    if not s or s[0] not in "[{":
        return None
    value, _i, _ok = _value(s, 0)
    return value if value is not _INCOMPLETE else None


_INCOMPLETE = object()
_WS = " \t\r\n"
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _skip(s: str, i: int) -> int:
    while i < len(s) and s[i] in _WS:
        i += 1
    return i


def _string(s: str, i: int) -> tuple:
    """(text, index after, complete) for the string starting at s[i] == '"'."""
    out = []
    i += 1
    while i < len(s):
        c = s[i]
        if c == '"':
            return "".join(out), i + 1, True
        if c == "\\":
            if i + 1 >= len(s):
                break
            e = s[i + 1]
            if e == "u":
                digits = s[i + 2:i + 6]
                if len(digits) < 4:
                    break
                try:
                    code = int(digits, 16)
                except ValueError:
                    raise ValueError("bad escape") from None
                if 0xD800 <= code < 0xDC00:           # a surrogate pair: wait for its second half
                    low = s[i + 6:i + 12]
                    if len(low) < 6:
                        break
                    code = 0x10000 + ((code - 0xD800) << 10) + (int(low[2:], 16) - 0xDC00)
                    i += 6
                out.append(chr(code))
                i += 6
                continue
            if e not in _ESCAPES:
                raise ValueError("bad escape")
            out.append(_ESCAPES[e])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out), len(s), False


def _value(s: str, i: int) -> tuple:
    """(value or _INCOMPLETE, index after, complete)."""
    i = _skip(s, i)
    if i >= len(s):
        return _INCOMPLETE, i, False
    c = s[i]
    if c == '"':
        return _string(s, i)
    if c == "{":
        out: Dict[str, Any] = {}
        i += 1
        while True:
            i = _skip(s, i)
            if i >= len(s):
                return out, i, False
            if s[i] == "}":
                return out, i + 1, True
            if s[i] == ",":
                i += 1
                continue
            if s[i] != '"':
                raise ValueError("expected a key")
            key, i, done = _string(s, i)
            if not done:
                return out, i, False
            i = _skip(s, i)
            if i >= len(s):
                return out, i, False
            if s[i] != ":":
                raise ValueError("expected ':'")
            value, i, done = _value(s, i + 1)
            if value is not _INCOMPLETE:
                out[key] = value
            if not done:
                return out, i, False
    if c == "[":
        items: List[Any] = []
        i += 1
        while True:
            i = _skip(s, i)
            if i >= len(s):
                return items, i, False
            if s[i] == "]":
                return items, i + 1, True
            if s[i] == ",":
                i += 1
                continue
            value, i, done = _value(s, i)
            if value is not _INCOMPLETE:
                items.append(value)
            if not done:
                return items, i, False
    j = i
    while j < len(s) and s[j] not in ",]}" + _WS:
        j += 1
    if j >= len(s):                                   # a number or a word that may still grow
        return _INCOMPLETE, j, False
    word = s[i:j]
    for lit, val in (("true", True), ("false", False), ("null", None)):
        if word == lit:
            return val, j, True
    try:
        return (int(word) if word.lstrip("-").isdigit() else float(word)), j, True
    except ValueError:
        raise ValueError(f"not JSON: {word!r}") from None


def _partial_kind(program: Any) -> str:
    """How the answer so far is read from its text: "text", "json" or "none"."""
    from .core import FunctAIFunc
    if not isinstance(program, FunctAIFunc):
        return "none"
    spec = program._spec()
    field = next((f for f in spec.signature.outputs if f.name == spec.main), None)
    shape = getattr(field, "shape", None) or {}
    options = shape.get("anyOf") or [shape]
    types = {o.get("type") for o in options if o.get("type") != "null"}
    if types == {"string"} and not any("enum" in o or "const" in o for o in options):
        return "text"
    if types and types <= {"object", "array"}:
        return "json"
    return "none"


# ------------------------------------------------------------------ the stream


_open: "weakref.WeakSet[Stream]" = weakref.WeakSet()


@atexit.register
def _close_at_exit() -> None:
    """A stream still running when the program ends is abandoned: cancel it,
    and give its call a moment to write its line to the call log."""
    pending = [s for s in list(_open) if not s.done]
    for s in pending:
        s._watch.cancelled.set()
    deadline = time.monotonic() + 2.0
    for s in pending:
        s._thread.join(max(0.0, deadline - time.monotonic()))


class Stream:
    '''One call of an AI function or a module, watched while it is made.

    Made by ``fn.stream(...)``; the call starts at once, in the background.
    It is the same call as ``fn(...)``: the same retries, tools and call log
    line, and the same value in the end.

    Iterate it (``for piece in s``, or ``async for``) for the answer's text
    as the model writes it; ``s.events()`` for everything that happens,
    in this call and every call inside it; ``s.show()`` to print it as it
    is written. ``s.result`` waits for the call and returns (or raises)
    what ``fn(...)`` would; ``await s`` does the same in async code.
    Iterating twice replays from the start.

    When the model is asked again (an unreadable reply, a provider error,
    an escalation), the pieces already shown cannot be taken back: the
    next ones are the new answer, and a ``Retry`` event says so. So
    ``"".join(s)`` is the text as shown; the answer is ``s.result``, and
    ``s.text`` is always the answer so far.

    Closing the stream (``s.close()``, or the end of a ``with`` block)
    cancels the call if it is still running: ``s.result`` then raises
    ``Cancelled``. A stream never closed runs to its end.

    Attributes
    ----------
    text : str
        The answer's text so far (in the latest request to the model: it
        starts again when the model is asked again).
    fields : dict
        Every output's text so far, by name.
    partial : optional
        The answer so far as a value: the text for a text answer, the JSON
        read so far for a record or a list (provisional: complete values,
        and strings as far as written), None for other answers until done.
    done : bool
        Whether the call has ended.
    call_id : str
        The call's id in the call log (waits for the call to start).

    Examples
    --------
    ```python
    @ai
    def haiku(topic: str) -> str:
        """A haiku about the topic."""
        ...

    for piece in haiku.stream("autumn rain"):
        print(piece, end="", flush=True)
    ```
    '''

    def __init__(self, program: Any, args: tuple, kwargs: Dict[str, Any]):
        from .core import FunctAIFunc
        self.program = program
        self.function: str = program.__name__
        self._is_ai = isinstance(program, FunctAIFunc)
        self._answer = program._spec().main if self._is_ai else None
        self._partial_kind = _partial_kind(program)
        self._events: List[Event] = []
        self._cond = threading.Condition()
        self._async_waiters: List[tuple] = []
        self._root: Optional[str] = None
        self._answers_for: Dict[str, str] = {}
        self._programs: Dict[str, Any] = {}              # call → its program
        self._answer_names: Dict[str, Optional[str]] = {}   # call → the name of its answer
        self._fields: Dict[str, Dict[str, List[str]]] = {}  # call → its fields' pieces in its latest request
        self._value: Any = _NOTHING
        self._prediction: Any = None
        self._error: Optional[BaseException] = None
        self._done = False
        self._closed = False
        self._error_seen = False
        self._watch = _Watch(self)
        ctx = contextvars.copy_context()
        ctx.run(calllog.WATCH.set, self._watch)
        self._thread = threading.Thread(target=ctx.run, args=(self._work, args, kwargs), daemon=True,
                                        name=f"functai-stream-{self.function}")
        _open.add(self)
        self._thread.start()

    # ----- the call, in its thread

    def _work(self, args: tuple, kwargs: Dict[str, Any]) -> None:
        try:
            value = self.program._invoke(args, kwargs) if self._is_ai else self.program(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — the consumer gets it from .result and iteration
            self._finish(error=exc)
        else:
            self._finish(value=value)

    def _finish(self, *, value: Any = _NOTHING, error: Optional[BaseException] = None) -> None:
        with self._cond:
            if error is not None:
                self._error = error
            else:
                self._value = value
                done = next((e for e in reversed(self._events) if isinstance(e, Done) and e.call == self._root),
                            None)
                self._prediction = done.prediction if done is not None else None
            self._done = True
            self._cond.notify_all()
            waiters, self._async_waiters = self._async_waiters, []
        self._wake(waiters)

    def _receive(self, event: Event, call: Any) -> None:
        """An event of a call this stream sees, in its form (``eventlog.TreeLog``)."""
        with self._cond:
            if isinstance(event, Started):
                if self._root is None:
                    self._root = event.call
                self._programs[event.call] = call.program
                self._answer_names[event.call] = call.answer if call.answer is not None else (
                    call.program._spec().main if hasattr(call.program, "_spec") else None)
                self._fields[event.call] = {}
                if call.answers_for is not None:
                    self._answers_for[event.call] = call.answers_for
            elif isinstance(event, (Request, Retry)):
                self._fields[event.call] = {}
            elif isinstance(event, Text):
                self._fields.setdefault(event.call, {}).setdefault(event.field, []).append(event.text)
            self._events.append(event)
            self._cond.notify_all()
            waiters, self._async_waiters = self._async_waiters, []
        self._wake(waiters)

    @staticmethod
    def _wake(waiters: List[tuple]) -> None:
        for loop, fut in waiters:
            try:
                loop.call_soon_threadsafe(lambda f=fut: f.done() or f.set_result(None))
            except RuntimeError:                  # that event loop is closed
                pass

    # ----- which events are the answer's text

    def _answers(self, call: str, target: Optional[str]) -> bool:
        while call is not None:
            if call == target:
                return True
            call = self._answers_for.get(call)
        return False

    def _is_answer_text(self, e: Event) -> bool:
        return isinstance(e, Text) and e.answer and self._root is not None and self._answers(e.call, self._root)

    # ----- reading, sync

    def _cursor(self, keep: Callable[[Event], Any]) -> "_Cursor":
        return _Cursor(self, keep)

    def __iter__(self) -> Iterator[str]:
        if not self._is_ai:
            raise TypeError(f"{self.function} is a module: its stream has no single answer to show as text. "
                            f"Iterate s.events(), or s.text_of(fn) for one AI function's answer")
        return self._cursor(lambda e: e.text if self._is_answer_text(e) else _NOTHING)

    def events(self) -> "_Cursor":
        """Every event of the call and of the calls inside it, in order:
        ``Started``, ``Request``, ``Text``, ``Thinking``, ``ToolCall``,
        ``ToolResult``, ``Retry``, ``Done``, ``Failed`` (``functai.streaming``).
        Works with ``for`` and ``async for``; each event has ``.kind``,
        ``.position`` (its place in its tree's log) and ``.to_dict()`` (the
        contract's JSON, format 2)."""
        return self._cursor(lambda e: e)

    def text_of(self, fn: Any) -> "_Cursor":
        """The answer's text of every call of ``fn`` inside this stream, as it
        is written (for a module's stream: ``s.text_of(summarize)``)."""
        calls: Dict[str, bool] = {}

        def keep(e: Event) -> Any:
            if isinstance(e, Started):
                calls[e.call] = self._program_of(e.call) is fn
            return e.text if isinstance(e, Text) and e.answer and calls.get(e.call) else _NOTHING
        return self._cursor(keep)

    def _program_of(self, call: str) -> Any:
        return self._programs.get(call)

    def _wait(self, timeout: Optional[float] = None) -> None:
        if self._done:
            return
        try:
            with self._cond:
                if not self._cond.wait_for(lambda: self._done, timeout):
                    raise TimeoutError(f"the stream of {self.function} did not end within {timeout} s")
        except KeyboardInterrupt:
            self.close()
            raise

    def _raise(self) -> None:
        if self._error is not None:
            self._error_seen = True
            raise self._error

    @property
    def result(self) -> Any:
        """What the call returned: waits for it, and raises what it raised."""
        self._wait()
        self._raise()
        return self._value

    @property
    def prediction(self) -> Any:
        """Everything the call produced, as ``fn.predict(...)`` returns it
        (AI functions): waits for it."""
        if not self._is_ai:
            raise TypeError(f"{self.function} is a module: its value is s.result; the predictions of the AI "
                            f"functions it called are on the Done events")
        self._wait()
        self._raise()
        return self._prediction

    def wait(self, timeout: Optional[float] = None) -> "Stream":
        """Wait for the call to end (at most ``timeout`` seconds); returns the stream."""
        self._wait(timeout)
        return self

    @property
    def done(self) -> bool:
        return self._done

    @property
    def call_id(self) -> Optional[str]:
        with self._cond:
            self._cond.wait_for(lambda: self._root is not None or self._done)
            return self._root

    def _answering(self) -> tuple:
        """(answer's name, {output: text}) of the call answering for this stream
        (its own, or the AI function it escalated to), in its latest request."""
        if self._root is None:
            return self._answer, {}
        with self._cond:
            answering = [c for c in self._programs if self._answers(c, self._root)]
            last = answering[-1]
            return self._answer_names.get(last), {k: "".join(v) for k, v in self._fields.get(last, {}).items()}

    @property
    def fields(self) -> Dict[str, str]:
        if not self._is_ai:
            raise TypeError(f"{self.function} is a module; its calls' texts are on the Text events")
        return self._answering()[1]

    @property
    def text(self) -> str:
        if not self._is_ai:
            raise TypeError(f"{self.function} is a module: use s.text_of(fn) or s.events()")
        name, fields = self._answering()
        return fields.get(name or "", "")

    @property
    def partial(self) -> Any:
        if self._done and self._error is None:
            return self._value
        if self._partial_kind == "text":
            return self.text
        if self._partial_kind == "json":
            try:
                return partial_json(self.text)
            except ValueError:
                return None
        if not self._is_ai:
            raise TypeError(f"{self.function} is a module: its value is s.result")
        return None

    # ----- closing

    def close(self) -> None:
        """Stop: cancel the call if it is still running (at the model's next
        piece of text; the provider may bill what it already generated)."""
        self._closed = True
        if not self._done:
            self._watch.cancelled.set()

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> "Stream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.close()

    async def aclose(self) -> None:
        self.close()

    # ----- async

    def __aiter__(self) -> "_Cursor":
        return self.__iter__()

    def __await__(self):
        return self._result_async().__await__()

    async def _result_async(self) -> Any:
        await self._until(lambda: self._done)
        return self.result

    async def _until(self, ready: Callable[[], bool]) -> None:
        loop = asyncio.get_running_loop()
        while True:
            with self._cond:
                if ready():
                    return
                fut = loop.create_future()
                self._async_waiters.append((loop, fut))
            try:
                await fut
            except asyncio.CancelledError:          # the consumer is gone (a client hung up): stop the call
                self.close()
                raise

    def __del__(self) -> None:
        # Like asyncio's "exception was never retrieved": a failure nobody saw
        # is not swallowed.
        if self._done and self._error is not None and not self._error_seen \
                and not isinstance(self._error, Cancelled):
            warnings.warn(f"[functai] the stream of {self.function} failed and nothing read its result: "
                          f"{type(self._error).__name__}: {self._error}", RuntimeWarning, stacklevel=1)

    # ----- display

    def __repr__(self) -> str:
        if not self._done and self._closed:
            state = "closing"
        elif not self._done:
            state = "running"
            if self._is_ai:
                n = len(self.text)
                state += f", {n} character{'s' if n != 1 else ''} of the answer so far" if n else ""
        elif self._error is not None:
            state = f"failed: {type(self._error).__name__}"
        else:
            shown = repr(self._value)
            state = "done: " + (shown if len(shown) <= 60 else shown[:57] + "...")
        return f"<Stream {self.function}: {state}>"

    def show(self, *, file: Any = None) -> None:
        """Print the call as it is written, and wait for its end.

        The answer's text as it arrives; when the function writes several
        outputs (a reasoning, then the answer), each is labelled; tool calls
        and their results get a line each; a retry says why. For a module,
        each AI function it calls, with its answer. In a notebook, the text
        appears as it is written. Raises what the call raised."""
        import sys
        out = file or sys.stdout
        field_of: Dict[str, Optional[str]] = {}          # call → field being printed
        labels: Dict[str, bool] = {}                     # call → several outputs, so label them
        thought: set = set()                             # calls with a thinking channel shown, labelled too
        at_line_start = True

        def write(text: str) -> None:
            nonlocal at_line_start
            if text:
                out.write(text)
                out.flush()
                at_line_start = text.endswith("\n")

        def line(text: str) -> None:
            write(("" if at_line_start else "\n") + text + "\n")

        try:
            for e in self.events():
                if isinstance(e, Started):
                    program = self._programs.get(e.call)
                    labels[e.call] = program is not None and hasattr(program, "_spec") and len(
                        [f for f in program._spec().signature.outputs if f.purpose in ("plain", "reasoning")]) > 1
                    if e.call != self._root and not self._answers(e.call, self._root):
                        line(f"▸ {e.function}")
                elif isinstance(e, (Text, Thinking)):
                    name = e.field if isinstance(e, Text) else "thinking"
                    if isinstance(e, Thinking):
                        thought.add(e.call)
                    if field_of.get(e.call) != name:
                        field_of[e.call] = name
                        if labels.get(e.call) or e.call in thought:
                            write("" if at_line_start else "\n")
                            write(f"{name}: ")
                    write(e.text)
                elif isinstance(e, ToolCall):
                    args = ", ".join(f"{k}={v!r}" for k, v in e.input.items()) if isinstance(e.input, dict) \
                        else repr(e.input)
                    line(f"→ {e.name}({args})")
                    field_of.pop(e.call, None)
                elif isinstance(e, ToolResult):
                    shown = e.output if len(e.output) <= 200 else e.output[:199] + "…"
                    line(f"← {shown}")
                elif isinstance(e, Retry):
                    line(f"[asked again: {e.reason}]")
                    field_of.pop(e.call, None)
                elif isinstance(e, Failed) and e.call != self._root:
                    line(f"✗ {e.function}: {type(e.error).__name__}: {e.error}")
                elif isinstance(e, Done) and e.call != self._root and not self._answers(e.call, self._root):
                    field_of.pop(e.call, None)
        finally:
            if not at_line_start:
                write("\n")
        self._wait()
        self._raise()

    def _ipython_display_(self) -> None:
        """In Jupyter, ``fn.stream(x)`` as a cell's last line shows it as it is written."""
        self.show()


class _Cursor:
    """One reader of a stream's events, from the start: ``for`` and ``async for``.
    ``keep(event)`` gives what to yield, or _NOTHING to skip it."""

    def __init__(self, stream: Stream, keep: Callable[[Event], Any]):
        self.stream = stream
        self.keep = keep
        self.i = 0

    def _next_ready(self) -> Any:
        """The next item if one is there, _NOTHING when the stream must be
        waited for; raises StopIteration or the call's error at the end."""
        s = self.stream
        while self.i < len(s._events):
            item = self.keep(s._events[self.i])
            self.i += 1
            if item is not _NOTHING:
                return item
        if s._done or s._closed:
            if s._error is not None and not s._closed and not isinstance(s._error, Cancelled):
                s._error_seen = True
                raise s._error
            raise StopIteration
        return _NOTHING

    def __iter__(self) -> "_Cursor":
        return self

    def __next__(self) -> Any:
        s = self.stream
        try:
            with s._cond:
                while True:
                    item = self._next_ready()
                    if item is not _NOTHING:
                        return item
                    s._cond.wait()
        except KeyboardInterrupt:
            s.close()
            raise

    def __aiter__(self) -> "_Cursor":
        return self

    async def __anext__(self) -> Any:
        s = self.stream
        while True:
            with s._cond:
                try:
                    item = self._next_ready()
                except StopIteration:
                    raise StopAsyncIteration from None
                if item is not _NOTHING:
                    return item
                n = len(s._events)
            await s._until(lambda n=n: len(s._events) > n or s._done or s._closed)


def stream(program: Any, args: tuple, kwargs: Dict[str, Any]) -> Stream:
    return Stream(program, args, kwargs)


__all__ = ["Stream", "Cancelled", "Event", "Started", "Request", "Text", "Thinking", "ToolCall", "ToolResult",
           "Retry", "Done", "Failed", "partial_json"]
