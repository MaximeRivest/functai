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

import ast
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .core import FunctAIFunc


def _find_functai_calls_with_names(fn: Callable[..., Any], globals_dict: Mapping[str, Any]) -> List[Tuple[str, FunctAIFunc]]:
    """(name, FunctAIFunc) for each direct call to a name that resolves to an AI function."""
    try:
        tree = ast.parse(_dedented_source(fn))
    except Exception:
        return []
    closure = {}
    try:
        closure = inspect.getclosurevars(fn).nonlocals
    except Exception:
        pass
    out: List[Tuple[str, FunctAIFunc]] = []
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            obj = closure.get(name, globals_dict.get(name))
            if isinstance(obj, FunctAIFunc) and id(obj) not in seen:
                out.append((name, obj))
                seen.add(id(obj))
    return out


def _dedented_source(fn: Callable[..., Any]) -> str:
    import textwrap
    return textwrap.dedent(inspect.getsource(fn))


class FunctAIModule:
    """Callable wrapper for an orchestrator function that calls @ai functions."""

    def __init__(self, fn: Callable[..., Any]):
        if not callable(fn):
            raise TypeError("@module must wrap a callable function")
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "module")
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn
        self._globals = getattr(fn, "__globals__", {})
        self.history: List[Any] = []
        self._opt_call_defaults: Dict[str, Any] = {}

    def __call__(self, *args, **kwargs):
        return self._invoke_original(*args, **kwargs)

    def __repr__(self) -> str:
        names = ", ".join(self.named_ai_functions())
        return f"<FunctAIModule {self.__name__} | ai functions: {names or '(none found)'}>"

    # ----- the AI functions it calls -----

    def named_ai_functions(self) -> Dict[str, FunctAIFunc]:
        """The @ai functions the body calls by name (looked up when asked, so
        functions defined after the module are found)."""
        return {name: obj for name, obj in _find_functai_calls_with_names(self._fn, self._globals)}

    def ai_functions(self) -> List[FunctAIFunc]:
        return list(self.named_ai_functions().values())

    # ----- optimization -----

    def opt(self, *, trainset: List[Any], metric: Optional[Callable[..., float]] = None, optimizer: Any = None,
            call_defaults: Optional[Dict[str, Any]] = None, valset: Optional[List[Any]] = None,
            **optimizer_kwargs) -> "FunctAIModule":
        """Tune every @ai function this module calls, against one metric on the
        module's output. ``call_defaults`` fill module arguments the examples lack."""
        from .optimizers import optimize
        optimize(self, trainset=trainset, optimizer=optimizer, metric=metric, valset=valset,
                 call_defaults=call_defaults, **optimizer_kwargs)
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


def module(fn: Callable[..., Any] | None = None):
    if fn is None:
        return lambda real_fn: FunctAIModule(real_fn)
    return FunctAIModule(fn)
