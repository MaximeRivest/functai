"""Tools that say what they do, and asking a person before they run
(contract/tools.md).

    @functai.tool(effects="reads")
    def read_note(name: str) -> str:
        \"\"\"The text of one note.\"\"\"

    @functai.tool(effects="changes")
    def write_note(name: str, text: str) -> str:
        \"\"\"Replace a note's text.\"\"\"

    @ai(tools=[read_note, write_note])
    def gardener(request: str) -> str: ...

    gardener.stream("Merge groceries.md into todo.md", approve=ask)   # ask(approval) -> True / False / "why not"
    chat = gardener.conversation("notes", store="notes-chats/", approve="changes")
    chat("Tidy todo.md")               # raises functai.Waiting; the turn is saved as waiting
    chat.turns[-1].approve(chat.turns[-1].waiting[0])      # from any process: the turn goes on

A tool that says nothing has unknown effects, and counts as ``"changes"``
for every rule: forgetting to declare is safe. ``approve`` is a setting
(``configure``, ``using``, ``@ai``, a conversation or one turn): a function
asked at once, or a rule. No approval by default.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

EFFECTS = ("reads", "changes")
# a domain a program needs (contract/calls.md, *Reach*): a host name, or "*." and one (every name under it)
_DOMAIN = re.compile(r"(\*\.)?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*")


def needs_of(value: Any, *, where: str) -> Optional[Dict[str, List[str]]]:
    """What code needs to reach, as data (contract/tools.md, *What a tool
    needs*): ``{"domains": [...], "secrets": [...]}``, domains in lower case,
    each list sorted and once each; None when nothing was said."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"domains", "secrets"}:
        raise TypeError(f"{where}: needs is {{'domains': [...], 'secrets': [...]}}: the host names its code reaches "
                        f"and the names of the secrets it uses")
    out: Dict[str, List[str]] = {}
    for key in ("domains", "secrets"):
        items = value.get(key, [])
        if isinstance(items, str) or not isinstance(items, (list, tuple)) or \
                not all(isinstance(x, str) and x.strip() for x in items):
            raise TypeError(f"{where}: needs[{key!r}] is a list of names")
        if key == "domains":
            items = [x.strip().lower() for x in items]
            bad = [x for x in items if not _DOMAIN.fullmatch(x)]
            if bad:
                raise TypeError(f"{where}: {bad[0]!r} is not a host name (api.example.com) or *. and one (*.example.com)")
        out[key] = sorted(set(items))
    return out


def merged_needs(*needs: Optional[Dict[str, List[str]]]) -> Optional[Dict[str, List[str]]]:
    """Several programs' needs as one (each list sorted, once each); None when none says any."""
    said = [n for n in needs if n is not None]
    if not said:
        return None
    return {k: sorted({x for n in said for x in n.get(k, [])}) for k in ("domains", "secrets")}
RULES = ("changes", "all")
DENIED = "The person did not allow this call."


class Tool:
    """A function the model may call, with what it does to the world. Made by
    ``functai.tool``.

    ``effects`` is ``"reads"`` (it only looks), ``"changes"`` (it writes,
    sends or pays) or None (it says nothing, which every approval rule treats
    as ``"changes"``). Called directly, it is the function; its name and
    docstring (or the ``name`` and ``description`` given) are what the
    model is told.
    """

    def __init__(self, fn: Callable[..., Any], *, effects: Optional[str] = None, name: Optional[str] = None,
                 description: Optional[str] = None, notes: Optional[Dict[str, Any]] = None,
                 needs: Optional[Dict[str, Any]] = None):
        if not callable(fn):
            raise TypeError("functai.tool wraps a function")
        if effects is not None and effects not in EFFECTS:
            raise ValueError(f"effects is 'reads' or 'changes' (or left out: unknown, which counts as 'changes'); "
                             f"not {effects!r}")
        functools.update_wrapper(self, fn)
        self._fn = fn
        self.effects = effects
        if name is not None:
            self.__name__ = name
        if description is not None:
            self.__doc__ = description
        from .interface import notes_of
        # the notes its code makes (contract/tools.md, *Notes a tool makes*): every AI function that has it declares them
        self.notes = notes_of(notes, where=f"tool {self.__name__}")
        # what its code reaches (contract/tools.md, *What a tool needs*): checked against the wall it runs behind
        self.needs = needs_of(needs, where=f"tool {self.__name__}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        return inspect.signature(self._fn)

    def __repr__(self) -> str:
        return f"<tool {self.__name__} ({self.effects or 'effects unknown'})>"


def tool(fn: Optional[Callable[..., Any]] = None, *, effects: Optional[str] = None, name: Optional[str] = None,
         description: Optional[str] = None, notes: Optional[Dict[str, Any]] = None,
         needs: Optional[Dict[str, Any]] = None) -> Any:
    '''Make a function a tool that says what it does to the world.

    Parameters
    ----------
    effects : "reads" or "changes", optional
        ``"reads"``: it only looks (a search, reading a file); it is never
        asked about by the ``"changes"`` rule, and a required journal does
        not wait before it. ``"changes"``: it changes something (writes,
        sends, pays). Left out: unknown, which every rule treats as
        ``"changes"``.
    name, description : str, optional
        What the model is told, instead of the function's name and
        docstring.
    notes : dict, optional
        The notes its code makes (``functai.note(name, data)``), by name: a
        type, or ``functai.Note(type, latest=..., outside=...)``. Every AI
        function that has the tool declares them, and each note it makes is
        checked against them (``InterfaceError``, code ``interface-note``).
    needs : dict, optional
        What its code reaches: ``{"domains": ["api.browser-use.com"],
        "secrets": ["BROWSER_USE_API_KEY"]}`` (names, never values). Behind a
        wall that lets through less (the ``wall`` setting), the function that
        has it is refused before its first call (``ReachError``); the reach
        counts in the function's setup.

    Returns
    -------
    Tool
        The function, callable as before, with ``.effects``.

    Examples
    --------
    ```python
    @functai.tool(effects="reads")
    def order_status(order: str) -> str:
        """Where an order is."""
        return "in Leeds"

    order_status.effects
    ```
    '''
    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(f, effects=effects, name=name, description=description, notes=notes, needs=needs)
    return wrap(fn) if fn is not None else wrap


@dataclasses.dataclass(frozen=True)
class Later:
    """A tool's result that comes later (``functai.later(key)``)."""
    key: str


def later(key: str) -> Later:
    '''What a tool returns when its result comes later: it started something
    that ends elsewhere (a job, a cloud browser, a person's reply) and whoever
    learns the result delivers it under ``key``.

    Only in a conversation's turn: the turn runs the step's other tool calls,
    then stops, saved, and waits (``Waiting``); ``chat.deliver(key, result)``,
    from any process, gives the tool its result and the turn goes on. The
    tool does not run again. Anywhere else nothing could wait for it, and the
    call refuses ``later-unavailable``.

    ``key`` names one result in its conversation (put an id in it:
    ``"harvest:8f2c"``). A result delivered before the turn waits is used at
    once.

    Examples
    --------
    ```python
    @functai.tool(effects="changes")
    def start_harvest(url: str) -> str:
        """Visit a site in a cloud browser; returns what it saw."""
        job = harvests.start(url, webhook="https://example.com/done")
        return functai.later(f"harvest:{job.id}")

    # in the webhook, any process:
    chat.deliver(f"harvest:{job_id}", report)
    ```
    '''
    if not isinstance(key, str) or not key.strip():
        raise TypeError("a later result's key is a non-empty text (put an id in it: 'harvest:8f2c')")
    return Later(key)


def effects_of(fn: Any) -> Optional[str]:
    """What a tool says it does: its ``effects``; an AI function used as a tool
    reads, unless one of its own tools changes things (or says nothing)."""
    effects = getattr(fn, "effects", None)
    if effects in EFFECTS:
        return effects
    from .core import FunctAIFunc
    if isinstance(fn, FunctAIFunc):
        inner = [effects_of(t) for t in fn.tools]
        return "reads" if all(e == "reads" for e in inner) else None
    return None                          # a module's own code may do anything: it reads only when it says so


def check_approve(value: Any) -> None:
    """Refuse an ``approve`` setting that is neither a function nor a rule."""
    if value is None or callable(value):
        return
    if isinstance(value, str):
        if value not in RULES:
            raise ValueError(f"approve is a function, 'changes', 'all', or a list of tool names and approval "
                             f"paths; not {value!r}")
        return
    if isinstance(value, (list, tuple, set, frozenset)) and all(isinstance(v, str) and v for v in value):
        return
    raise TypeError(f"approve is a function, 'changes', 'all', or a list of tool names and approval paths; "
                    f"not {value!r}")


@dataclasses.dataclass(frozen=True)
class Approval:
    """One tool call waiting for a person's answer.

    ``call``: the id of the AI function's call that asked for it;
    ``invocation``: the tool call's number in that call; ``id``: the id the
    model gave it; ``name`` and ``input``: the tool and what it would be
    given; ``effects``: what the tool says it does; ``path``: where it is,
    by names (``support/answer/refund``), for rules written before any call."""
    call: str
    invocation: int
    id: str
    name: str
    input: Any
    effects: Optional[str]
    path: str
    site: str = ""          # the asking call's place in its tree ("support#1/answer#1"): what resuming finds it by
    plugin: str = "approval"     # which plugin asks (a tool call may be asked about by several)
    question: Optional[str] = None  # why it asks, in a sentence, when it says

    def to_dict(self) -> Dict[str, Any]:
        from .calllog import to_json
        out = {"call": self.call, "invocation": self.invocation, "id": self.id, "name": self.name,
               "input": to_json(self.input)[0], "effects": self.effects, "path": self.path, "site": self.site,
               "plugin": self.plugin}
        if self.question:
            out["question"] = self.question
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Approval":
        return cls(d["call"], int(d["invocation"]), d["id"], d["name"], d.get("input"), d.get("effects"),
                   d.get("path") or d["name"], d.get("site", ""), d.get("plugin") or "approval",
                   d.get("question"))


def path_of(call: Any, name: str) -> str:
    """A tool call's approval path: the names of the calls from the outermost
    one to the call that asked, then the tool's (``support/answer/refund``)."""
    names = [part.split("#", 1)[0] for part in (call.path or "").split("/") if part]
    return "/".join([*names, name])


def asks(rule: Any, approval: Approval) -> bool:
    """Whether a rule asks a person about this tool call (contract/tools.md):
    ``"changes"`` (and a function, which is asked what ``"changes"`` asks) for
    a tool that changes things or says nothing; ``"all"`` for every one; a
    list for a tool named in it, or whose path is in it or ends with
    ``/<entry>``."""
    if rule is None:
        return False
    if callable(rule) or rule == "changes":
        return approval.effects != "reads"
    if rule == "all":
        return True
    for entry in rule:
        if entry == approval.name or entry == approval.path or approval.path.endswith("/" + entry):
            return True
    return False


def verdict_of(answer: Any) -> Tuple[bool, Optional[str]]:
    """(allowed, reason) from what a function answered: True, False, or a
    reason to refuse (a text)."""
    if answer is True:
        return True, None
    if answer is False or answer is None:
        return False, None
    if isinstance(answer, str):
        return False, answer
    raise TypeError(f"an approve function answers True, False, or a reason to refuse (a text); not {answer!r}")


def denial(reason: Optional[str]) -> str:
    """What the model is shown for a refused tool call."""
    return DENIED + (f" Reason: {reason}" if reason else "")


# ------------------------------------------------------------------ waiting in this process (a stream)


class _Pending:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.answers: Dict[Tuple[str, int, str], Tuple[bool, Optional[str], Optional[str]]] = {}
        self.asked: Dict[Tuple[str, int, str], Approval] = {}


PENDING = _Pending()


def answer(call: str, invocation: int, allowed: bool, *, reason: Optional[str] = None,
           by: Optional[str] = None, plugin: str = "approval") -> None:
    """Answer an approval a call of this process waits for."""
    with PENDING.cond:
        PENDING.answers[(call, int(invocation), plugin)] = (bool(allowed), reason, by)
        PENDING.cond.notify_all()


def waiting_in_process() -> List[Approval]:
    with PENDING.cond:
        return [a for k, a in PENDING.asked.items() if k not in PENDING.answers]


def _wait_here(call: Any, approval: Approval) -> Tuple[bool, Optional[str], Optional[str]]:
    key = (approval.call, approval.invocation, approval.plugin)
    with PENDING.cond:
        PENDING.asked[key] = approval
        try:
            while key not in PENDING.answers:
                call.check()                          # a closed stream stops the wait
                PENDING.cond.wait(0.1)
            return PENDING.answers.pop(key)
        finally:
            PENDING.asked.pop(key, None)


def ask_person(call: Any, approval: Approval, decide: Optional[Callable[[Any], Any]] = None
               ) -> Tuple[bool, Optional[str], Optional[str]]:
    """Ask whether a tool call may run (a plugin's ``tool.ask``):
    (allowed, reason refused, by whom). ``decide`` answers in place of a
    person. A call being resumed takes the answer recorded for it. Emits
    ``approval`` then ``approved``. Otherwise: in a conversation the turn
    waits (it raises: the turn is saved as waiting); on a stream the call
    waits for an answer here; a plain call refuses ``approval-required``."""
    run = getattr(call, "turn_run", None)
    if run is not None:
        known = run.recorded_approval(call, approval)
        if known is not None:
            allowed, reason, by, fresh = known
            if fresh:                                 # answered while the turn waited: the log says so now
                from .conversations import frontier
                frontier()
                call.emit("approved", id=approval.id, invocation=approval.invocation, plugin=approval.plugin,
                          verdict="yes" if allowed else "no", by=by, reason=reason)
            return allowed, reason, by
    from . import conversations
    conversations.frontier()
    to = conversations.approvals_to()
    call.emit("approval", id=approval.id, invocation=approval.invocation, name=approval.name, input=approval.input,
              effects=approval.effects, path=approval.path, to=to, plugin=approval.plugin,
              question=approval.question)
    if decide is not None:
        allowed, reason = verdict_of(decide(approval))
        by = None
    elif run is not None:
        run.pause(call, approval)                     # raises: the turn waits, saved
        raise AssertionError("unreachable")
    elif call.streams:
        allowed, reason, by = _wait_here(call, approval)
    else:
        from .errors import ApprovalError
        raise ApprovalError(f"{approval.path}: this tool call needs a person's answer ({approval.plugin}), and "
                            f"a plain call has nobody to ask. Give approve= a function, stream the call and answer "
                            f"with s.approve(...), or use a conversation, where the turn waits.", approval=approval)
    call.emit("approved", id=approval.id, invocation=approval.invocation, plugin=approval.plugin,
              verdict="yes" if allowed else "no", by=by, reason=reason)
    if run is not None:
        run.note_approval(call, approval, allowed, reason, by)
    return allowed, reason, by


__all__ = ["tool", "Tool", "Approval", "effects_of", "asks", "check_approve", "ask_person", "denial", "EFFECTS"]
