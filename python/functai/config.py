"""Settings and their cascade.

Precedence, innermost first:

1. forced overrides (``fn.using(...)``, optimizers running a teacher)
2. the function (``@ai(temperature=0.1)``, ``fn.lm = ...``)
3. a ``with configure(...):`` block (scoped to the current context/thread)
4. ``configure(...)`` called as a statement (process-wide)
5. the defaults below

Settings are resolved at call time, so ``configure(lm=...)`` after a function
was defined still reaches it (unless the function set its own ``lm``).
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Dict, Iterator, List, Tuple

import lm15

from .errors import LogContentError

# lm15 Config fields may be set anywhere a functai setting can:
# temperature, max_tokens, top_p, seed, stop, reasoning, ...
CONFIG_FIELDS = frozenset(lm15.Config.__dataclass_fields__)

DEFAULTS: Dict[str, Any] = {
    # which model, and how to reach it
    "lm": None,                 # the model: "gpt-4.1-mini", "claude:claude-sonnet-4-5", "groq:openai/gpt-oss-120b",
                                # "openai/gpt-4o", or an lm15 BoundClient (a login's connection and model)
    "api_key": None,            # a key for the provider `lm` routes to (beats saved logins and the environment)
    "auth": None,               # saved logins: None/True (lm15's credentials file) | a file path | False (never)
    "base_url": None,           # a base URL for that provider
    "client": None,             # the lm15 connection: an LMRouter, or one provider's LM (OpenAILM(api_key=...))
    "capabilities": None,       # lmcc capability facts that override functai's model table

    # how the call is laid out and run
    "adapter": None,            # None (functai's tags) | "chat" | "json" | "xml" | lmcc.Adapter | artifact dict
    "module": "predict",        # "predict" | "cot" | "react"
    "retries": 1,               # re-asks after an unreadable reply (only spent when a call would fail)
    "api_retries": 3,           # re-sends after a transient provider error (rate limit, 5xx, timeout)
    "max_steps": 8,             # model calls per tool loop
    "tool_errors": "report",    # "report" (the model sees the error) | "raise"
    "on_unreadable": "raise",   # "raise" | "record": keep an unreadable reply as a turn with no values
                                # (prediction.refusal says why); for rollouts that must go on
    "cache_replies": False,     # True: reuse the reply to an identical request (in memory; unreadable replies are not kept); lm15's `cache` is prompt caching

    # escalation: when the model is less sure than escalate_below (its probability for
    # its own answer), escalate_to answers instead (a model name, a baked model, an AI function)
    "escalate_to": None,
    "escalate_below": None,     # default 0.9 when escalate_to is set

    # memory
    "stateful": False,
    "state_window": 5,

    # prompt cosmetics
    "include_fn_name_in_instructions": True,

    # optimization and instruction writing
    "optimizer": None,
    "teacher": None,
    "teacher_lm": None,
    "autocompile": False,
    "autoinstruct": False,
    "instruction_lm": None,
    "autocompile_n": 0,
    "autogen_instructions": True,
    "instruction_autorefine_calls": 0,
    "instruction_autorefine_max_examples": 20,

    # the call log (contract/calls.md): every call, one line of JSON in a folder
    "log_calls": None,         # None: as $FUNCTAI_LOG_CALLS says (off when unset) | True (that folder, else
                                # ~/.local/share/functai/calls) | a folder | False (never)
    "log_content": None,       # None/True: the values and messages too | False: sizes, times and tokens only |
                                # {field: bool, "*": bool}: per field. Only ever removes: a value is written only
                                # when no layer (the function, each block, configure, $FUNCTAI_LOG_CONTENT) drops it
    "caller": None,            # who is calling, added to $FUNCTAI_CALLER: {"kind": "agent", "conversation": ...}

    # receivers of each call tree's events (contract/streaming.md): they add up over the layers
    "observers": None,         # [callable or list, ...]: given the kept form of every event, best effort
    "journal": None,           # a store, functai.Journal(store, required=True), or False (none): one per tree

    # debug
    "debug": False,
}

KNOWN = frozenset(DEFAULTS) | CONFIG_FIELDS

_GLOBAL: Dict[str, Any] = {}
_LOCK = __import__("threading").Lock()
_ABSENT = object()
_SCOPED: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("functai_scoped", default={})
_FORCED: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("functai_forced", default={})
# Each enclosing block's own settings, outermost first: the settings that combine
# over layers instead of the closest one deciding (log_content, observers, journal).
_BLOCKS: contextvars.ContextVar[Tuple[Dict[str, Any], ...]] = contextvars.ContextVar("functai_blocks", default=())


def check(settings: Dict[str, Any], where: str) -> Dict[str, Any]:
    """Unknown names, and values functai could only refuse later, refuse now."""
    unknown = set(settings) - KNOWN
    if unknown:
        hint = " (the connection setting is client= since functai 1.0)" if "router" in unknown else ""
        raise TypeError(
            f"{where}: unknown setting(s) {sorted(unknown)}{hint}. functai settings: {sorted(DEFAULTS)}; "
            f"lm15 Config fields: {sorted(CONFIG_FIELDS)}")
    from . import adapters, models
    try:
        if settings.get("lm") is not None:
            models.check_lm(settings["lm"])
        if settings.get("client") is not None:
            models.check_client(settings["client"])
            if models._is_bound_client(settings.get("lm")) or models._is_baked(settings.get("lm")):
                raise TypeError(f"lm is a {type(settings['lm']).__name__}, which brings its own connection: "
                                f"drop client=")
        esc = settings.get("escalate_to")
        if esc is not None:
            from .core import FunctAIFunc
            if not (isinstance(esc, (str, FunctAIFunc)) or models._is_baked(esc)):
                raise TypeError("escalate_to is a model name, a baked model, or an AI function")
        below = settings.get("escalate_below")
        if below is not None and not (isinstance(below, (int, float)) and 0 < below <= 1):
            raise ValueError(f"escalate_below is a probability in (0, 1], not {below!r}")
        if settings.get("adapter") is not None:
            adapters.resolve_adapter(settings["adapter"])
        if settings.get("log_calls") is not None or settings.get("log_content") is not None \
                or settings.get("caller") is not None:
            from . import calllog
            calllog.check_settings(settings)
        if settings.get("observers") is not None or settings.get("journal") is not None:
            from . import eventlog
            settings = {**settings, **eventlog.check_settings(settings)}
    except LogContentError:
        raise
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{where}: {exc}") from None
    return dict(settings)


def layers(own: Dict[str, Any] | None = None) -> List[Tuple[str, Dict[str, Any]]]:
    """The settings around a call, closest first, each as it was set:
    ``("own", the program's own)``, then each enclosing block's (``"block"``,
    innermost first), then ``("configure", the process-wide ones)``. For the
    settings that combine over layers (log_content only removes; observers
    add up; a host's journal holds against a program's)."""
    out: List[Tuple[str, Dict[str, Any]]] = [("own", dict(own or {}))]
    forced = _FORCED.get()
    if forced:
        out.append(("block", forced))
    out += [("block", b) for b in reversed(_BLOCKS.get())]
    out.append(("configure", dict(_GLOBAL)))
    return out


def effective(fn_settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Every setting's value for a call of a function with ``fn_settings``."""
    return {**DEFAULTS, **_GLOBAL, **_SCOPED.get(), **(fn_settings or {}), **_FORCED.get()}


class configure:
    '''Set defaults for every AI function: the model, sampling, layout, and more.

    Called plainly, the settings apply to the whole program. Used in a
    ``with`` block, they apply inside the block only, in this thread and in
    the threads functai starts from it (``evaluate(num_threads=8)``).

    Settings are looked up at every call, most specific first:
    ``fn.using(...)``, then the function's own (``@ai(...)``), then a
    ``with configure(...)`` block, then ``configure(...)``.

    Parameters
    ----------
    **settings
        Any setting ``@ai`` takes: ``lm``, ``temperature``, ``max_tokens``,
        ``api_key``, ``base_url``, ``auth``, ``client``, ``adapter``,
        ``module``, ``tools``, ``max_steps``, ``stateful``, ``retries``,
        ``api_retries``, ``cache_replies``, ``teacher_lm``, ``debug``...
        An unknown setting raises ``TypeError``. The call log:
        ``log_calls`` (``True``, or a folder: keep every call),
        ``log_content`` (``False``, or ``{"transcript": False}``: what the
        log may not keep; it only ever removes, so a block's ``False``
        holds for every call inside it), and ``caller`` (who is calling, a
        dict). Each call tree's events: ``observers`` (a list of functions
        or lists, given the kept form of every event; they add up over
        blocks) and ``journal`` (a store, ``functai.Journal(store,
        required=True)``, or ``False``: where whole trees are kept while
        they run; a program cannot replace or remove the one you set).

    Returns
    -------
    configure
        Usable as a context manager, to undo the settings at the end of the
        block.

    See Also
    --------
    ai : settings for one function.
    FunctAIFunc.using : a copy of one function with other settings.

    Examples
    --------
    ```python
    functai.configure(lm="gpt-4.1-mini", temperature=0)
    functai.settings.lm
    ```

    For one block only:

    ```python
    @ai
    def capital(country: str) -> str:
        """The country's capital city."""
        ...

    with functai.configure(lm="gpt-4.1-nano"):
        print(capital("Canada"))
    ```
    '''

    def __init__(self, **overrides):
        self._overrides = check(overrides, "configure")
        with _LOCK:
            self._before = {k: _GLOBAL.get(k, _ABSENT) for k in self._overrides}
            _GLOBAL.update(self._overrides)
        self._token = None

    def __enter__(self):
        # Used as a block: undo this process-wide change (only the keys it set, and
        # only where no other thread has set them since), and scope it instead.
        with _LOCK:
            for k, old in self._before.items():
                if _GLOBAL.get(k, _ABSENT) is self._overrides[k]:
                    if old is _ABSENT:
                        _GLOBAL.pop(k, None)
                    else:
                        _GLOBAL[k] = old
        self._token = _SCOPED.set({**_SCOPED.get(), **self._overrides})
        self._block = _BLOCKS.set((*_BLOCKS.get(), self._overrides))
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _SCOPED.reset(self._token)
            _BLOCKS.reset(self._block)
            self._token = None
        return False

    def __repr__(self) -> str:
        return f"configure({', '.join(f'{k}={v!r}' for k, v in self._overrides.items())})"


@contextlib.contextmanager
def scoped(**overrides) -> Iterator[None]:
    """Like ``with configure(...)``, without ever touching the process-wide settings
    (functai's own internal use)."""
    checked = check(overrides, "scoped")
    token = _SCOPED.set({**_SCOPED.get(), **checked})
    block = _BLOCKS.set((*_BLOCKS.get(), checked))
    try:
        yield
    finally:
        _BLOCKS.reset(block)
        _SCOPED.reset(token)


@contextlib.contextmanager
def forced(**overrides) -> Iterator[None]:
    """Overrides that beat even a function's own settings, for this context."""
    token = _FORCED.set({**_FORCED.get(), **check(overrides, "forced")})
    try:
        yield
    finally:
        _FORCED.reset(token)


class _Settings:
    """Read the effective defaults as attributes: ``functai.settings.lm``.
    Assigning an attribute is ``configure(name=value)``."""

    def __getattr__(self, name: str):
        values = effective()
        if name in values:
            return values[name]
        if name in CONFIG_FIELDS:
            return None
        raise AttributeError(f"functai has no setting {name!r}")

    def __setattr__(self, name: str, value) -> None:
        configure(**{name: value})

    def as_dict(self) -> Dict[str, Any]:
        return effective()

    def __repr__(self) -> str:
        changed = {k: v for k, v in effective().items() if DEFAULTS.get(k, None) != v}
        return f"<functai settings {changed or '(defaults)'}>"


settings = _Settings()
