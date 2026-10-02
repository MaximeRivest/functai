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

    def __init__(self, name: str, *, answer_from: Any = None):
        if name not in NAMES:
            raise ValueError(f"a view is one of {', '.join(NAMES)}; not {name!r}")
        self.name = name
        self.answer_from = getattr(answer_from, "__name__", answer_from)
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


def outside(events: Any, *, answer_from: Any = None) -> list:
    """A whole form of a log as the outside view shows it."""
    v = View("outside", answer_from=answer_from)
    return [x for x in (v.apply(e) for e in events) if x is not None]


__all__ = ["View", "outside", "NAMES"]
