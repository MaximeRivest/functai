"""AI functions on table columns, through dpyr.

    from dpyr import read, col
    reviews = read("reviews.parquet")
    reviews.mutate(topic=classify(col.text))                     # one model call per distinct text
    reviews.mutate(answer=answer(col.question, context=col.doc, tone="formal"))
    reviews.filter(is_complaint(col.text))

Called with a dpyr column expression anywhere in its arguments, an @ai
function (or a @module) returns a column expression instead of calling the
model. dpyr runs it once per distinct row of arguments, remembers the
results for the session, and in a displayed dataframe only runs the rows it
shows. Options: ``classify.vectorize(threads=16, errors="null")(col.text)``.

The column is computed with the prompt the function had when the
expression was written: its instruction, its demos, its model. Optimizing
the function afterwards gives new expressions new results, and never mixes
remembered answers from before and after.
"""

from __future__ import annotations

import hashlib
import inspect
import typing
from typing import Any, Dict, Optional

_THREADS = 8        # model calls wait on the network; providers' rate limits are retried with backoff


def has_column(args: tuple, kwargs: Dict[str, Any]) -> bool:
    """True when an argument is a dpyr expression (checked without importing dpyr)."""
    return any(type(a).__module__.startswith("dpyr.") for a in (*args, *kwargs.values()))


def _dpyr():
    try:
        import dpyr
    except ImportError as err:
        raise ImportError('calling an AI function on columns needs dpyr: pip install "functai[data]"') from err
    return dpyr


def _return_type(fn: Any, default: Any) -> Any:
    try:                                   # resolves `from __future__ import annotations` strings
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 — unresolvable names: fall back to the raw annotation
        hints = {}
    if "return" in hints:
        return hints["return"]
    ann = inspect.signature(fn).return_annotation
    return default if ann is inspect.Signature.empty else ann


def _version(*parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:12]


def vectorize_function(fn: Any, *, dtype: Any = None, threads: Optional[int] = None, errors: str = "raise",
                       version: str = "") -> Any:
    """A dpyr RowFunction for an @ai function, pinned to its current
    instruction, demos and settings."""
    dpyr = _dpyr()
    from .evaluation import with_states
    state = fn.state()
    settings = fn._effective()
    key = (_version(state, sorted((k, repr(v)) for k, v in settings.items()), fn._template), version,
           repr(dtype), threads, errors)
    cached = fn._vectorized.get(key)
    if cached is not None:
        return cached
    pinned = fn.using(lm=settings["lm"]) if settings.get("lm") is not None else fn.using()

    def run(*args: Any, **kwargs: Any) -> Any:
        with with_states({pinned: state}):
            return pinned(*args, **kwargs)

    run.__name__ = fn.__name__
    out_type = dtype if dtype is not None else _return_type(fn._fn, str)   # no annotation: text, as in a call
    try:
        rf = dpyr.RowFunction(run, dtype=out_type, threads=threads or _THREADS, errors=errors,
                              version="-".join(filter(None, (key[0], version))), name=fn.__name__)
    except dpyr.ExprTypeError as err:
        raise TypeError(f"{fn.__name__} returns {out_type!r}, which is not a column type ({err}); "
                        f"use {fn.__name__}.map(table), or {fn.__name__}.vectorize(dtype=...)") from None
    fn._vectorized[key] = rf
    return rf


def vectorize_module(mod: Any, *, dtype: Any = None, threads: Optional[int] = None, errors: str = "raise",
                     version: str = "") -> Any:
    """A dpyr RowFunction for a @module, pinned to the current state of every
    AI function it calls."""
    dpyr = _dpyr()
    from .evaluation import with_states
    fns = mod.ai_functions()
    states = {f: f.state() for f in fns}
    key = (_version([(f.__name__, s) for f, s in states.items()],
                    [sorted((k, repr(v)) for k, v in f._effective().items()) for f in fns]),
           version, repr(dtype), threads, errors)
    cached = mod._vectorized.get(key)
    if cached is not None:
        return cached

    def run(*args: Any, **kwargs: Any) -> Any:
        with with_states(states):
            return mod(*args, **kwargs)

    run.__name__ = mod.__name__
    out_type = dtype if dtype is not None else _return_type(mod._fn, None)
    if out_type is None:
        raise TypeError(f"@module {mod.__name__} has no return annotation, so its column has no type; "
                        f"annotate it (def {mod.__name__}(...) -> list[str]) or use "
                        f"{mod.__name__}.vectorize(dtype=...)")
    rf = dpyr.RowFunction(run, dtype=out_type, threads=threads or _THREADS, errors=errors,
                          version="-".join(filter(None, (key[0], version))), name=mod.__name__)
    mod._vectorized[key] = rf
    return rf
