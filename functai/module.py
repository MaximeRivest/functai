"""``@module``: a plain Python function that calls @ai functions, optimized as one program.

    @module
    def research(claim: str, hops: int = 2):
        facts = []
        for _ in range(hops):
            query = generate_query(claim, facts)
            facts = append_notes(claim, facts, search(query))
        return facts

    research.opt(trainset=..., metric=...)     # tunes generate_query and append_notes together

The metric sees ``Prediction(result=<what the module returned>)``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional

from .core import FunctAIFunc


def _reachable_ai_functions(fn: Callable[..., Any]) -> List[FunctAIFunc]:
    """Every AI function the code reaches: called by name or under another name,
    through plain helper functions, tools and closures (functai.graph reads the
    code; nothing runs)."""
    from .graph import Analysis
    a = Analysis(discover=True)
    try:
        a.program(fn)
    except TypeError:
        return []
    return [n.obj for n in a.nodes.values() if isinstance(n.obj, FunctAIFunc)]


class FunctAIModule:
    """Callable wrapper for an orchestrator function that calls @ai functions."""

    def __init__(self, fn: Callable[..., Any], *, requires: Any = ()):
        if not callable(fn):
            raise TypeError("@module must wrap a callable function")
        self._fn = fn
        self._requires = tuple(requires or ())
        self.__name__ = getattr(fn, "__name__", "module")
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn
        self._globals = getattr(fn, "__globals__", {})
        self.history: List[Any] = []
        self._opt_call_defaults: Dict[str, Any] = {}
        self._vectorized: Dict[Any, Any] = {}

    def __call__(self, *args, **kwargs):
        from .columns import has_column
        if has_column(args, kwargs):                  # research(col.claim): a column, for dpyr
            return self.vectorize()(*args, **kwargs)
        from . import calllog
        from .config import effective
        def inputs():
            bound = inspect.signature(self._fn).bind(*args, **kwargs)
            bound.apply_defaults()
            return dict(bound.arguments)

        return calllog.run(self, effective(), inputs, lambda: self._invoke_original(*args, **kwargs))

    @property
    def version(self) -> str:
        """Which version of the module this is: ``sha256:`` of the code it
        reaches and the versions of the AI functions it calls, so optimizing
        one of them is a new version of the module."""
        from . import calllog
        return calllog.module_version(self)

    def vectorize(self, *, dtype: Any = None, threads: Optional[int] = None, errors: str = "raise"):
        """This module as a dpyr row function (see ``FunctAIFunc.vectorize``);
        its column type is the module's return annotation, or ``dtype``."""
        from .columns import vectorize_module
        return vectorize_module(self, dtype=dtype, threads=threads, errors=errors)

    def __dpyr_vectorize__(self, *, dtype: Any = None, threads: Optional[int] = None, errors: str = "raise",
                           version: str = ""):
        from .columns import vectorize_module
        return vectorize_module(self, dtype=dtype, threads=threads, errors=errors,
                                version=version)

    def __repr__(self) -> str:
        names = ", ".join(self.named_ai_functions())
        return f"<FunctAIModule {self.__name__} | ai functions: {names or '(none found)'}>"

    # ----- the AI functions it calls -----

    def named_ai_functions(self) -> Dict[str, FunctAIFunc]:
        """Every @ai function this module reaches: called by name, under another
        name, or through helper functions (looked up when asked, so functions
        defined after the module are found). Keys are function names, qualified by
        module when two share a name."""
        fns = _reachable_ai_functions(self._fn)
        counts: Dict[str, int] = {}
        for f in fns:
            counts[f.__name__] = counts.get(f.__name__, 0) + 1
        return {(f.__name__ if counts[f.__name__] == 1 else f"{f._fn.__module__}.{f.__name__}"): f for f in fns}

    def ai_functions(self) -> List[FunctAIFunc]:
        return list(self.named_ai_functions().values())

    # ----- optimization -----

    def map(self, data: Any, *, num_threads: int = 1, call_defaults: Optional[Dict[str, Any]] = None):
        """Run on every row of a table; returns the rows with ``pred_result``
        (what the module returned) as a dpyr dataframe. See ``FunctAIFunc.map``."""
        from .evaluation import evaluate
        return evaluate(self, data, (), num_threads=num_threads, call_defaults=call_defaults).table

    def opt(self, *, trainset: Any, metric: Any = None, optimizer: Any = None,
            call_defaults: Optional[Dict[str, Any]] = None, valset: Any = None,
            expected: Any = None, **optimizer_kwargs) -> "FunctAIModule":
        """Tune every @ai function this module calls, against one metric on the
        module's output. ``call_defaults`` fill module arguments the examples lack."""
        from .optimizers import optimize
        optimize(self, trainset=trainset, optimizer=optimizer, metric=metric, valset=valset,
                 call_defaults=call_defaults, expected=expected, **optimizer_kwargs)
        return self

    def undo_opt(self, steps: int = 1) -> None:
        for fn in self.ai_functions():
            fn.undo_opt(steps)

    def save(self, path) -> None:
        """Every AI function's instruction and demos, in one JSON file."""
        import json
        from pathlib import Path
        data = {"functai": 1, "module": self.__name__,
                "functions": {name: fn.state().to_dict() for name, fn in self.named_ai_functions().items()}}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))

    def load(self, path) -> "FunctAIModule":
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        fns = self.named_ai_functions()
        for name, state in (data.get("functions") or {}).items():
            if name in fns:
                fns[name].load_state(state)
        return self

    def _invoke_original(self, *args, **kwargs):
        out = self._fn(*args, **kwargs)
        self.history.append({"args": args, "kwargs": kwargs, "output": out})
        del self.history[:-100]
        return out


def module(fn: Callable[..., Any] | None = None, *, requires: Any = ()):
    '''Make a Python function that calls AI functions into one program.

    The body is ordinary Python: loops, ifs, helpers, several AI functions.
    As a module it can be evaluated, optimized (each AI function inside
    learns from the runs the metric accepts), run on a table, and saved as
    one program. Use it bare (``@module``) or with requirements.

    Parameters
    ----------
    requires : list of str
        Packages the program needs that functai cannot see from the code
        (``["numpy>=2"]``), for ``functai.save``.

    Returns
    -------
    FunctAIModule
        Called like the original. The return annotation is the program's
        output type.

    See Also
    --------
    evaluate : measure the program on rows with known answers.
    save : save it with everything it depends on.

    Examples
    --------
    ```python
    @ai
    def draft(topic: str) -> str:
        """A two-sentence paragraph about the topic."""

    @ai
    def shorten(text: str) -> str:
        """The text in at most twelve words."""

    @module
    def blurb(topic: str) -> str:
        return shorten(draft(topic))

    blurb("why paired comparisons need fewer examples")
    ```
    '''
    if fn is None:
        return lambda real_fn: FunctAIModule(real_fn, requires=requires)
    return FunctAIModule(fn, requires=requires)
