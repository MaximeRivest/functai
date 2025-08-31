from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Mapping, Tuple

import dspy

from .core import FunctAIFunc
from .core import settings as functai_settings


def _is_functai_ai_function(obj: Any) -> bool:
    return isinstance(obj, FunctAIFunc)


def _find_functai_calls_with_names(
    fn: Callable[..., Any],
    globals_dict: Mapping[str, Any],
) -> List[Tuple[str, FunctAIFunc]]:
    """Return (name, FunctAIFunc) for each direct call to a Name that resolves to a FunctAIFunc."""
    try:
        source = inspect.getsource(fn)
        tree = ast.parse(source)
    except Exception:
        return []

    out: List[Tuple[str, FunctAIFunc]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in globals_dict:
                obj = globals_dict[name]
                if _is_functai_ai_function(obj):
                    out.append((name, obj))

    # De-duplicate while preserving order
    seen = set()
    uniq: List[Tuple[str, FunctAIFunc]] = []
    for name, obj in out:
        if id(obj) not in seen:
            uniq.append((name, obj))
            seen.add(id(obj))
    return uniq


@contextmanager
def _patched_globals(globals_ns: Dict[str, Any], replacements: Dict[str, Any]):
    """Temporarily patch specified names in a function's globals; restore on exit."""
    sentinel = object()
    backups = {k: globals_ns.get(k, sentinel) for k in replacements}
    try:
        globals_ns.update(replacements)
        yield
    finally:
        for k, old in backups.items():
            if old is sentinel:
                globals_ns.pop(k, None)
            else:
                globals_ns[k] = old


class _DSPyAdapterModule(dspy.Module):
    """Adapter so DSPy optimizers can see predictors inside a FunctAIModule."""

    def __init__(self, outer: "FunctAIModule"):
        super().__init__()
        self._outer = outer

    def _current_alias_map(self) -> Dict[str, Tuple[FunctAIFunc, str]]:
        """Build a fresh alias map: alias -> (FunctAIFunc, predictor_name)."""
        amap: Dict[str, Tuple[FunctAIFunc, str]] = {}
        for fname, faif in self._outer._name_to_ai.items():
            try:
                prog = faif.latest_program(fresh=False)
                n2p = list(getattr(prog, "named_predictors", lambda: [])())
                if not n2p:
                    # fallback
                    amap[f"{fname}__predict"] = (faif, "predict")
                else:
                    for pred_name, _ in n2p:
                        amap[f"{fname}__{pred_name}"] = (faif, pred_name)
            except Exception:
                amap[f"{fname}__predict"] = (faif, "predict")
        return amap

    def named_predictors(self):
        pairs: List[Tuple[str, Any]] = []
        for alias, (faif, pred_name) in self._current_alias_map().items():
            try:
                prog = faif.latest_program(fresh=False)
                name2pred = {n: p for n, p in prog.named_predictors()}
                pred = name2pred.get(pred_name)
                if pred is None and name2pred:
                    # take first as fallback
                    pred = list(name2pred.values())[0]
                if pred is not None:
                    pairs.append((alias, pred))
            except Exception:
                continue
        return pairs

    def predictors(self):
        return [p for _, p in self.named_predictors()]

    def map_named_predictors(self, func: Callable[[Any], Any]):
        """Apply a mapping to each predictor and push replacements back into owning modules."""
        for alias, (faif, pred_name) in list(self._current_alias_map().items()):
            try:
                prog = faif.latest_program(fresh=False)
                name2pred = {n: p for n, p in prog.named_predictors()}
                old_pred = name2pred.get(pred_name)
                if old_pred is None and name2pred:
                    # fallback to first
                    pred_name = list(name2pred.keys())[0]
                    old_pred = name2pred[pred_name]
                if old_pred is None:
                    continue
                new_pred = func(old_pred)
                # Try setattr on the owning module by name
                try:
                    setattr(prog, pred_name, new_pred)
                except Exception:
                    pass
            except Exception:
                continue
        return self

    def set_lm(self, lm: Any):
        for _, (faif, _) in self._alias_map.items():
            try:
                faif.lm = lm
            except Exception:
                pass

    def get_lm(self):
        lms = []
        for _, (faif, _) in self._alias_map.items():
            try:
                lms.append(faif.lm)
            except Exception:
                pass
        uniq = {id(x) for x in lms if x is not None}
        if len(uniq) == 1 and lms:
            return lms[0]
        raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

    def forward(self, *args, **kwargs) -> dspy.Prediction:
        # Merge call defaults for missing orchestrator args (e.g., k/hops)
        try:
            call_defaults = getattr(self._outer, "_opt_call_defaults", {}) or {}
        except Exception:
            call_defaults = {}
        merged_kwargs = dict(call_defaults)
        try:
            merged_kwargs.update(kwargs)
        except Exception:
            pass
        res = self._outer._invoke_original(*args, **merged_kwargs)
        return dspy.Prediction(result=res)

    # Make deepcopy safe (avoid deep-copying the outer reference)
    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        # Shallow copy base state
        try:
            new.__dict__.update({k: v for k, v in self.__dict__.items() if k != "_outer"})
        except Exception:
            pass
        # Preserve same outer reference intentionally
        new._outer = self._outer
        return new


class FunctAIModule:
    """Callable wrapper for an orchestrator function that calls @ai functions."""

    def __init__(self, fn: Callable[..., Any]):
        if not callable(fn):
            raise TypeError("@module must wrap a callable function")
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "module")
        self.__doc__ = fn.__doc__
        self._globals = fn.__globals__

        pairs = _find_functai_calls_with_names(fn, self._globals)
        self._name_to_ai: Dict[str, FunctAIFunc] = {name: obj for name, obj in pairs}

        self.history: List[Any] = []
        # Defaults for missing call args during optimization/evaluation
        self._opt_call_defaults: Dict[str, Any] = {}

    def __call__(self, *args, **kwargs):
        return self._invoke_original(*args, **kwargs)

    def opt(
        self,
        *,
        trainset: List[dspy.Example],
        metric: Callable[..., float] | Any,
        optimizer: Any | None = None,
        call_defaults: Dict[str, Any] | None = None,
        **teleprompter_kwargs,
    ) -> "FunctAIModule":
        # Store call defaults for adapter forward
        if call_defaults:
            self._opt_call_defaults = dict(call_defaults)
        # Pick teleprompter / optimizer
        if optimizer is None:
            optimizer = dspy.teleprompt.BootstrapFewShot

        if isinstance(optimizer, type):
            # Provide metric if ctor expects it
            ctor_kwargs = dict(teleprompter_kwargs)
            try:
                sig = inspect.signature(optimizer.__init__)
                if "metric" in sig.parameters and "metric" not in ctor_kwargs:
                    ctor_kwargs["metric"] = metric
            except Exception:
                pass
            telep = optimizer(**ctor_kwargs)
        else:
            telep = optimizer
            try:
                if hasattr(telep, "metric") and getattr(telep, "metric", None) is None:
                    telep.metric = metric
            except Exception:
                pass

        prog = _DSPyAdapterModule(self)

        # Ensure LM from FunctAI settings is applied across predictors (optional)
        lm_from_config = getattr(functai_settings, "lm", None)
        if lm_from_config is not None:
            try:
                prog.set_lm(lm_from_config)
            except Exception:
                pass

        compiled = telep.compile(prog, trainset=trainset)

        # If a new adapter module was returned, push predictors back into the owned functions
        try:
            named = getattr(compiled, "named_predictors", None)
            if compiled is not prog and callable(named):
                for alias, pred in compiled.named_predictors():
                    try:
                        fname, pred_name = alias.split("__", 1)
                    except Exception:
                        continue
                    faif = self._name_to_ai.get(fname)
                    if faif is None:
                        continue
                    try:
                        mod = faif.latest_program(fresh=False)
                        setattr(mod, pred_name, pred)
                    except Exception:
                        pass
        except Exception:
            pass

        return self

    def named_ai_functions(self) -> Dict[str, FunctAIFunc]:
        return dict(self._name_to_ai)

    def _invoke_original(self, *args, **kwargs):
        if not self._name_to_ai:
            return self._fn(*args, **kwargs)
        replacements = {name: self._name_to_ai[name] for name in self._name_to_ai}
        with _patched_globals(self._globals, replacements):
            out = self._fn(*args, **kwargs)
        self.history.append({"args": args, "kwargs": kwargs, "output": out})
        return out


def module(fn: Callable[..., Any] | None = None):
    if fn is None:
        return lambda real_fn: FunctAIModule(real_fn)
    return FunctAIModule(fn)
