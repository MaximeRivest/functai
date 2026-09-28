"""The errors FunctAI defines by a contract code (contract/README.md, *Refusal
codes FunctAI defines*).

Each carries ``code``, the contract's word for what went wrong, so a program
can branch on it and the call log can record it (``error.code``), the same
in every language. lmcc's own refusals (``lmcc.Refusal``) carry lmcc's codes.
"""

from __future__ import annotations

from typing import Any, Optional


class FunctAIError(Exception):
    """An error with a code from FunctAI's contract (``err.code``)."""

    code: str = ""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class InterfaceError(FunctAIError, ValueError):
    """A program's interface refused what it was given or what it gave back.

    ``code`` is ``"interface-input"`` (a call given what the interface does
    not take), ``"interface-output"`` (the code returned what the interface
    does not give) or ``"interface-malformed"`` (the interface itself breaks
    the rules, when the program is defined or read from a folder). ``field``
    names the field at fault, or is None when the fault is no field's."""

    def __init__(self, code: str, field: Optional[str], message: str):
        super().__init__(code, message)
        self.field = field


class LogContentError(FunctAIError, ValueError):
    """A ``log_content`` map that cannot be honoured (code
    ``"log-content-field"``): a key that is neither a field name nor ``"*"``,
    or, in a program's own settings, a name the program has no field for (a
    misspelling would otherwise write the very value it meant to keep out).
    ``field`` is the key."""

    def __init__(self, field: str, message: str):
        super().__init__("log-content-field", message)
        self.field = field


class SawError(FunctAIError, LookupError):
    """What a call saw cannot be known, or shown again (contract/calls.md,
    *Saw*). ``code`` is one of ``not-recorded``, ``missing-call``,
    ``unknown-key``, ``saw-cycle``, ``not-kept``, ``turn-invalid``; ``call``
    is the call whose record says so."""

    def __init__(self, code: str, call: Optional[str], message: str):
        super().__init__(code, message)
        self.call = call


class EventRefused(FunctAIError):
    """A store's refusal of an append, a claim or a read (contract/streaming.md,
    *The rules a store keeps*): ``event-malformed``, ``event-conflict``,
    ``event-gap``, ``event-after-end``, ``event-start`` or ``event-unknown``.
    ``event`` is the position of the event refused in a batch, when there is
    one. Any other exception a store raises means no answer came."""

    def __init__(self, code: str, message: str = "", *, event: Optional[dict] = None):
        super().__init__(code, message or code)
        self.event = event


class JournalError(FunctAIError, RuntimeError):
    """A journal kept the call from going on, or could not confirm its end.

    ``code`` is one of:

    - ``"journal-policy"``: the settings around the call break the journal
      policy (a program's own setting replacing or removing a host's
      journal; a closer layer replacing, weakening or removing a required
      one). Raised before the call runs.
    - ``"journal-scope"``: a required journal set only around a call inside
      a tree (it cannot keep part of a tree).
    - ``"journal-barrier"``: a required journal did not confirm the call's
      start, or a tool call, so the code or the tool did not run.
    - ``"journal-end"``: a required journal did not confirm the call's end.
      The call itself ended as ``outcome`` says (its value, or its error):
      the journal changes nothing of that. ``journal`` says ``"refused"``
      (it did not keep the end) or ``"unknown"`` (no answer came: it may
      have), ``event`` names the end by its position. ``settle()`` asks
      the journal which it was."""

    def __init__(self, code: str, message: str, *, outcome: Optional["Outcome"] = None,
                 event: Optional[dict] = None, journal: Optional[str] = None, store: Any = None,
                 tree: Optional[str] = None):
        super().__init__(code, message)
        self.outcome = outcome
        self.event = event
        self.journal = journal
        self._store = store
        self._tree = tree

    def settle(self, *, claim: bool = False) -> str:
        """For ``journal-end`` with ``journal == "unknown"``: read the journal
        and say what became of the end, ``"kept"``, ``"not-kept"`` (the log is
        unfinished) or ``"another-end"`` (another writer ended the log, so this
        outcome is not the log's).

        ``"not-kept"`` may still change while the writer keeps sending. With
        ``claim=True`` the log is claimed first, which fences the writer, so
        the answer is final; the log then waits for whoever ends it."""
        from .eventlog import settle
        if self.code != "journal-end" or self._store is None:
            raise ValueError("only a journal-end error of a journal can be settled")
        return settle(self._store, self._tree, self.event, claim=claim)


class Outcome:
    """What a call's program did: its ``value``, or its ``error``.

    ``get()`` gives the value, or raises the error. A ``JournalError``
    (``journal-end``) holds the outcome the journal could not confirm."""

    __slots__ = ("value", "error")

    def __init__(self, value: Any = None, error: Optional[BaseException] = None):
        self.value = value
        self.error = error

    @property
    def failed(self) -> bool:
        return self.error is not None

    def get(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value

    def __repr__(self) -> str:
        if self.error is not None:
            return f"Outcome(error={self.error!r})"
        return f"Outcome(value={self.value!r})"


__all__ = ["FunctAIError", "InterfaceError", "LogContentError", "SawError", "EventRefused", "JournalError",
           "Outcome"]
