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
from typing import Any, Dict, Iterator

import lm15

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
    "cache_replies": False,     # True: reuse the reply to an identical request (in memory); lm15's `cache` is prompt caching

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

    # debug
    "debug": False,
}

KNOWN = frozenset(DEFAULTS) | CONFIG_FIELDS

_GLOBAL: Dict[str, Any] = {}
_SCOPED: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("functai_scoped", default={})
_FORCED: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("functai_forced", default={})


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
            if models._is_bound_client(settings.get("lm")):
                raise TypeError("lm is an lm15 BoundClient, which brings its own connection: drop client=")
        if settings.get("adapter") is not None:
            adapters.resolve_adapter(settings["adapter"])
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{where}: {exc}") from None
    return dict(settings)


def effective(fn_settings: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Every setting's value for a call of a function with ``fn_settings``."""
    return {**DEFAULTS, **_GLOBAL, **_SCOPED.get(), **(fn_settings or {}), **_FORCED.get()}


class configure:
    """``configure(lm="gpt-4.1-mini", temperature=0)`` sets process-wide defaults.

    ``with configure(temperature=0): ...`` changes them for the block only, and
    only in the current context (threads started by functai inherit it; other
    threads do not see it)."""

    def __init__(self, **overrides):
        self._overrides = check(overrides, "configure")
        self._before = dict(_GLOBAL)
        _GLOBAL.update(self._overrides)
        self._token = None

    def __enter__(self):
        # Used as a block: undo the process-wide change, scope it instead.
        _GLOBAL.clear()
        _GLOBAL.update(self._before)
        self._token = _SCOPED.set({**_SCOPED.get(), **self._overrides})
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _SCOPED.reset(self._token)
            self._token = None
        return False

    def __repr__(self) -> str:
        return f"configure({', '.join(f'{k}={v!r}' for k, v in self._overrides.items())})"


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
