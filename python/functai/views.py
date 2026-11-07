"""Views: what one kind of reader may see of a call tree's log
(contract/streaming.md, *Views*).

- ``full``: every event and value (only the process running the tree has it);
- ``kept``: what ``log_content`` lets be kept (a store's form);
- ``outside``: a caller who sees only the program's boundary (a served
  program's customer): its ``started`` (its program object without its file
  and line), its answer's text as it is
  written, the approvals addressed to it, and its ``done``, or its
  ``failed`` with the error's type and code and no message. Never a helper's
  answer, a tool call or result, a thinking, or why a request was retried.

A view keeps each event's ``writer`` and ``seq`` and sets ``after`` to the
event before it in the view. It never alters a value it shows and never
invents one: a module's answer shown as it is written is its ``answer_from``
helper's text, re-addressed to the module (its ``call``, ``function`` and
``field``), each call of that helper beginning as a ``request`` of the
module, so a second call empties the first's text.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

NAMES = ("full", "kept", "outside")
_PROGRAM = ("name", "kind", "module", "version", "signature", "interface", "answer")


class View:
    """A view made event by event: ``apply(event)`` gives the event as the view
    shows it (a new dict), or None when the view leaves it out. Events are
    given in the order of one form of one log."""

    def __init__(self, name: str, *, answer_from: Any = None, notes: Any = ()):
        if name not in NAMES:
            raise ValueError(f"a view is one of {', '.join(NAMES)}; not {name!r}")
        self.name = name
        self.answer_from = getattr(answer_from, "__name__", answer_from)
        # the notes the outermost program declares outside (contract/programs.md, *Notes*): shown, re-addressed
        self.notes = frozenset(notes or ())
        self.root: Optional[str] = None
        self.root_function: Optional[str] = None
        self.root_kind: Optional[str] = None
        self.answer: str = "result"
        self.forwarded: Dict[str, bool] = {}           # calls whose answer is shown as the root's
        self.requests = 0
        self.last: Optional[Dict[str, int]] = None

    def _link(self, e: Dict[str, Any]) -> Dict[str, Any]:
        e["after"] = self.last
        self.last = {"writer": e["writer"], "seq": e["seq"]}
        return e

    def apply(self, event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        e = copy.deepcopy(dict(event))
        if self.name in ("full", "kept"):
            return self._link(e)
        kind, call = e.get("kind"), e.get("call")
        if self.root is None:
            if kind != "started":
                return None
            self.root = call
            self.root_function = e.get("function")
            program = e.get("program") or {}
            self.root_kind = program.get("kind", "ai")
            self.answer = program.get("answer") or "result"
        if kind == "note":
            if e.get("name") not in self.notes:
                return None
            e["call"], e["function"] = self.root, self.root_function
            return self._link(e)
        if call == self.root:
            return self._root(e, kind)
        return self._inside(e, kind)

    def _root(self, e: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
        if kind == "started":
            e["program"] = {k: v for k, v in (e.get("program") or {}).items() if k in _PROGRAM}
            e.pop("invocation", None)
            return self._link(e)
        if kind == "request":
            if self.root_kind != "ai":
                return None
            self.requests = max(self.requests, int(e.get("request") or 0))
            return self._link(e)
        if kind == "retry":
            if self.root_kind != "ai":
                return None
            e.pop("reason", None)
            e["content"] = False
            return self._link(e)
        if kind == "text":
            return self._link(e) if e.get("answer") else None
        if kind in ("approval", "approved"):
            return self._approval(e)
        if kind == "done":
            return self._link(e)
        if kind == "failed":
            err = e.get("error") or {}
            e["error"] = {k: v for k, v in err.items() if k in ("type", "code")}
            e["content"] = False
            return self._link(e)
        return None

    def _approval(self, e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if e.get("kind") == "approval" and e.get("to") != "caller":
            return None
        if e.get("kind") == "approved" and not self._asked_caller(e):
            return None
        if e.get("kind") == "approval":
            self._caller_asked = getattr(self, "_caller_asked", set())
            self._caller_asked.add((e.get("call"), e.get("invocation")))
        e["call"], e["function"] = self.root, self.root_function
        return self._link(e)

    def _asked_caller(self, e: Mapping[str, Any]) -> bool:
        return (e.get("call"), e.get("invocation")) in getattr(self, "_caller_asked", set())

    def _inside(self, e: Dict[str, Any], kind: str) -> Optional[Dict[str, Any]]:
        if kind in ("approval", "approved"):
            return self._approval(e)
        if self.root_kind == "ai" or self.answer_from is None:
            return None
        call = e.get("call")
        if kind == "started":
            self.forwarded[call] = e.get("function") == self.answer_from
            if not self.forwarded[call]:
                return None
            return self._as_request(e)
        if not self.forwarded.get(call):
            return None
        if kind in ("request", "retry"):
            return self._as_request(e)
        if kind == "text" and e.get("answer"):
            e["call"], e["function"], e["field"] = self.root, self.root_function, self.answer
            return self._link(e)
        return None

    def _as_request(self, e: Dict[str, Any]) -> Dict[str, Any]:
        """A forwarded helper's new request (or its start): the module's
        answer starts again, as a ``request`` of the module."""
        self.requests += 1
        out = {k: e[k] for k in ("functai_event", "tree", "writer", "seq", "at") if k in e}
        out.update(kind="request", call=self.root, function=self.root_function, request=self.requests, model=None)
        return self._link(out)


def outside(events: Any, *, answer_from: Any = None, notes: Any = ()) -> list:
    """A whole form of a log as the outside view shows it. ``notes``: the
    names of the notes the outermost program declares ``outside``."""
    v = View("outside", answer_from=answer_from, notes=notes)
    return [x for x in (v.apply(e) for e in events) if x is not None]


def outside_notes(program: Any) -> list:
    """The names of the notes a program declares ``outside`` (its interface's)."""
    try:
        iface = program.interface
    except Exception:  # noqa: BLE001 — a program with no interface shows none
        return []
    return [n["name"] for n in (iface.get("notes") or []) if n.get("outside")]


def for_program(name: str, program: Any) -> View:
    """The view ``name`` of a log of ``program``'s calls, with what the program declares for it."""
    return View(name, answer_from=getattr(program, "_answer_from", None), notes=outside_notes(program))


_ALWAYS = frozenset({"started", "request", "retry", "done", "failed"})


class Custom:
    """A view of your own (contract/streaming.md, *Views*): what a served
    program's callers see (``serve(program, view=...)``), or what you read of
    a stream (``s.events(view=...)``)::

        thoughts = views.Custom(shows=lambda e: e if e["kind"] in ("thinking", "text") else None)

    ``calls(started)``: whether a call is shown, given its ``started`` event
    (as JSON); default every call. ``shows(event)``: each event of a shown
    call as the view shows it: the same, a copy with values left out, or
    None; default the event. A call it shows always shows its ``started``,
    ``request``, ``retry``, ``done`` and ``failed`` (with the values
    ``shows`` leaves them), and what ``shows`` gives is cut to the keys the
    contract knows. Never alter a value, nor invent one: a reader takes what
    a view shows as what happened."""

    def __init__(self, *, calls: Optional[Any] = None, shows: Optional[Any] = None):
        for name, f in (("calls", calls), ("shows", shows)):
            if f is not None and not callable(f):
                raise TypeError(f"a view's {name} is a function; not {type(f).__name__}")
        self.calls = calls
        self.shows = shows

    def __repr__(self) -> str:
        return "views.Custom(" + ", ".join(n for n in ("calls", "shows") if getattr(self, n) is not None) + ")"


class _Applied:
    """A view of one's own made event by event, as ``View`` makes a named one."""

    def __init__(self, view: Custom):
        self.view = view
        self.shown: Dict[str, bool] = {}
        self.last: Optional[Dict[str, int]] = None

    def apply(self, event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        from .eventlog import KINDS, known
        e = copy.deepcopy(dict(event))
        kind, call = e.get("kind"), e.get("call")
        if kind not in KINDS:                            # what it cannot judge, it never passes on
            return None
        if kind == "started":
            self.shown[call] = bool(self.view.calls(copy.deepcopy(e))) if self.view.calls is not None else True
        if not self.shown.get(call):
            return None
        given = self.view.shows(copy.deepcopy(e)) if self.view.shows is not None else e
        if given is not None:
            e = known(given)
        elif kind not in _ALWAYS:
            return None
        e["after"] = self.last
        self.last = {"writer": e["writer"], "seq": e["seq"]}
        return e


def as_view(view: Any) -> Any:
    """``view`` as one a reader may name: a name of ``NAMES``, a ``Custom``, a
    function (a ``Custom`` with that ``shows``), or any object with ``calls``
    and ``shows`` (each a function or None). TypeError otherwise."""
    if isinstance(view, str):
        if view in NAMES:
            return view
    elif isinstance(view, Custom):
        return view
    elif callable(view):
        return Custom(shows=view)
    elif hasattr(view, "calls") or hasattr(view, "shows"):
        return Custom(calls=getattr(view, "calls", None), shows=getattr(view, "shows", None))
    raise TypeError(f"a view is one of {', '.join(NAMES)}, a views.Custom, or a function; not {view!r}")


def viewer(view: Any, program: Any) -> Any:
    """The view ``view`` (``as_view``) of a log of ``program``'s calls, made
    event by event: ``.apply(event)`` gives the event as it shows it, or None."""
    view = as_view(view)
    return for_program(view, program) if isinstance(view, str) else _Applied(view)


__all__ = ["View", "Custom", "outside", "outside_notes", "for_program", "as_view", "viewer", "NAMES"]
