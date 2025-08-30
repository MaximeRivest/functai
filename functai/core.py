# functai.py — v1.0 (refined vision)

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import json
import ast
import typing
from typing import Any, Dict, Optional, List, Tuple, TypedDict

import dspy
from dspy import Signature, InputField, OutputField, Prediction

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAIN_OUTPUT_DEFAULT_NAME = "result"
INCLUDE_FN_NAME_IN_INSTRUCTIONS_DEFAULT = True

# ──────────────────────────────────────────────────────────────────────────────
# Defaults & configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _Defaults:
    # DSPy wiring
    lm: Any = None                 # str | dspy.LM | None
    api_key: Optional[str] = None  # funneled into dspy.LM if lm is a str
    adapter: Any = None            # "chat" | "json" | dspy.Adapter subclass | instance | None
    module: Any = "predict"        # "predict" | "cot" | "react" | dspy.Module subclass | instance | None
    temperature: Optional[float] = None

    # Statefulness (via dspy.History)
    stateful: bool = False
    state_window: int = 5

    # Optimization
    optimizer: Any = None          # Callable[[], Optimizer] | Optimizer | None

    # Prompt cosmetics
    include_fn_name_in_instructions: bool = INCLUDE_FN_NAME_IN_INSTRUCTIONS_DEFAULT

    # Debug/preview
    debug: bool = False

_GLOBAL_DEFAULTS = _Defaults()

def _effective_defaults() -> _Defaults:
    return _GLOBAL_DEFAULTS

def _apply_overrides(target: _Defaults, **overrides):
    for k, v in overrides.items():
        if hasattr(target, k):
            setattr(target, k, v)

class _ConfigContext:
    def __init__(self, snapshot: _Defaults):
        # full snapshot of previous defaults (by value)
        self._prev = dataclasses.replace(snapshot)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # restore field-by-field
        _apply_overrides(_GLOBAL_DEFAULTS, **dataclasses.asdict(self._prev))
        return False

class _ConfigureFacade:
    """
    configure(...): sets global defaults (setter).
    with configure(...): temporary overrides restored on exit (context manager).
    """
    def __call__(self, **overrides):
        # capture snapshot BEFORE applying changes (for potential with-usage)
        snapshot = dataclasses.replace(_GLOBAL_DEFAULTS)
        # apply global changes immediately (setter semantics)
        _apply_overrides(_GLOBAL_DEFAULTS, **overrides)
        # return a context that will restore to the snapshot on exit
        return _ConfigContext(snapshot)

# Public configure
configure = _ConfigureFacade()

# Keep a light 'settings' alias (read-only by convention)
settings = _GLOBAL_DEFAULTS

# ──────────────────────────────────────────────────────────────────────────────
# Adapters & LMs utilities
# ──────────────────────────────────────────────────────────────────────────────

def _select_adapter(adapter: Any) -> Optional[dspy.Adapter]:
    if adapter is None:
        return None
    if isinstance(adapter, str):
        key = adapter.lower().replace("-", "_")
        if key in ("json", "jsonadapter"):
            return dspy.JSONAdapter()
        if key in ("chat", "chatadapter"):
            return dspy.ChatAdapter()
        raise ValueError(f"Unknown adapter string '{adapter}'.")
    if isinstance(adapter, type) and issubclass(adapter, dspy.Adapter):
        return adapter()
    if isinstance(adapter, dspy.Adapter):
        return adapter
    raise TypeError("adapter must be a string, a dspy.Adapter subclass, or a dspy.Adapter instance.")

@contextlib.contextmanager
def _patched_adapter(adapter_instance: Optional[dspy.Adapter]):
    prev = getattr(dspy.settings, "adapter", None)
    try:
        if adapter_instance is not None:
            dspy.settings.adapter = adapter_instance
        yield
    finally:
        dspy.settings.adapter = prev

@contextlib.contextmanager
def _patched_lm(lm_instance: Optional[Any]):
    prev = getattr(dspy.settings, "lm", None)
    try:
        if lm_instance is not None:
            dspy.settings.lm = lm_instance
        yield
    finally:
        dspy.settings.lm = prev

# ──────────────────────────────────────────────────────────────────────────────
# Signature building (docstring-driven)
# ──────────────────────────────────────────────────────────────────────────────

def _mk_signature(fn_name: str, fn: Any, *, doc: str, return_type: Any,
                  extra_outputs: Optional[List[Tuple[str, Any, str]]] = None,
                  main_output: Optional[Tuple[str, Any, str]] = None) -> type[Signature]:
    """Create a dspy.Signature from function params and declared outputs."""
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True) if hasattr(typing, "get_type_hints") else {}
    class_dict: Dict[str, Any] = {}
    ann_map: Dict[str, Any] = {}

    # Inputs (skip FunctAI-reserved names if they existed in user params by accident)
    for pname, p in sig.parameters.items():
        if pname in {"_prediction", "all"}:
            continue
        ann = hints.get(pname, p.annotation if p.annotation is not inspect._empty else str)
        class_dict[pname] = InputField()
        ann_map[pname] = ann

    # Extra outputs
    if extra_outputs:
        for name, typ, desc in extra_outputs:
            if name in class_dict:
                continue
            class_dict[name] = OutputField(desc=str(desc) if desc is not None else "")
            ann_map[name] = typ if typ is not None else str

    # Primary output
    if main_output is None:
        mo_name, mo_type, mo_desc = MAIN_OUTPUT_DEFAULT_NAME, return_type, ""
    else:
        mo_name, mo_type, mo_desc = main_output
        if mo_type is None:
            mo_type = return_type
    if mo_name in class_dict:
        mo_name = MAIN_OUTPUT_DEFAULT_NAME
    class_dict[mo_name] = OutputField(desc=str(mo_desc) if mo_desc is not None else "")
    ann_map[mo_name] = mo_type

    # Attach doc
    if doc:
        class_dict["__doc__"] = doc
    class_dict["__annotations__"] = ann_map

    Sig = type(f"{fn_name.title()}Sig", (Signature,), class_dict)
    return Sig

def _compose_system_doc(fn: Any, *, include_fn_name: bool) -> str:
    # Persona/history are removed. Docstring is the instruction; optional function name.
    parts = []
    if include_fn_name and getattr(fn, "__name__", None):
        parts.append(f"Function: {fn.__name__}")
    base = (fn.__doc__ or "").strip()
    if base:
        parts.append(base)
    return "\n\n".join([p for p in parts if p]).strip()

# ──────────────────────────────────────────────────────────────────────────────
# AST-based collection of declared outputs (x: T = _ai["desc"])
# ──────────────────────────────────────────────────────────────────────────────

def _eval_annotation(expr: ast.AST, env: Dict[str, Any]) -> Any:
    try:
        code = compile(ast.Expression(expr), filename="<ann>", mode="eval")
        return eval(code, env, {})
    except Exception:
        return str

def _extract_desc_from_subscript(node: ast.Subscript) -> str:
    try:
        sl = node.slice
        if isinstance(sl, ast.Index):
            sl = sl.value  # py3.8 compat
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return str(sl.value)
        if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
            second = sl.elts[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str):
                return str(second.value)
    except Exception:
        pass
    return ""

def _collect_ast_outputs(fn: Any) -> List[Tuple[str, Any, str]]:
    try:
        src = inspect.getsource(fn)
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []

    # Find our function node
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return []

    outputs_ordered: List[Tuple[str, Any, str]] = []
    env = dict(fn.__globals__)
    env.setdefault("typing", typing)

    for node in ast.walk(fn_node):
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                continue
            name = node.target.id
            val = node.value
            if val is None:
                continue
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "_ai"
            if not (is_ai or is_ai_sub):
                continue
            typ = _eval_annotation(node.annotation, env) if node.annotation is not None else str
            desc = _extract_desc_from_subscript(val) if is_ai_sub else ""
            if not any(n == name for n, _, _ in outputs_ordered):
                outputs_ordered.append((name, typ, desc))
        elif isinstance(node, ast.Assign):
            if not node.targets:
                continue
            val = node.value
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "_ai"
            if not (is_ai or is_ai_sub):
                continue
            desc = _extract_desc_from_subscript(val) if is_ai_sub else ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    name = tgt.id
                    if not any(n == name for n, _, _ in outputs_ordered):
                        outputs_ordered.append((name, None, desc))
    return outputs_ordered

class _ReturnInfo(TypedDict, total=False):
    mode: str  # 'name' | 'sentinel' | 'ellipsis' | 'empty' | 'other'
    name: Optional[str]

def _collect_return_info(fn: Any) -> _ReturnInfo:
    try:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
    except Exception:
        return {"mode": "other", "name": None}
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return {"mode": "other", "name": None}
    ret: _ReturnInfo = {"mode": "other", "name": None}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return):
            val = node.value
            if val is None:
                ret = {"mode": "empty", "name": None}
            elif isinstance(val, ast.Name):
                if val.id == "_ai":
                    ret = {"mode": "sentinel", "name": None}
                else:
                    ret = {"mode": "name", "name": val.id}
            elif isinstance(val, ast.Constant) and val.value is Ellipsis:
                ret = {"mode": "ellipsis", "name": None}
            else:
                ret = {"mode": "other", "name": None}
    return ret

# ──────────────────────────────────────────────────────────────────────────────
# Module selection
# ──────────────────────────────────────────────────────────────────────────────

def _select_module_kind(module: Any, tools: Optional[List[Any]]) -> Any:
    # If module is not explicitly set or is "predict", and tools are present → ReAct
    if module is None or (isinstance(module, str) and module.lower() in {"predict", "p", ""}):
        if tools:
            return "react"
        return "predict"
    return module

def _instantiate_module(module_kind: Any, Sig: type[Signature], *, tools: Optional[List[Any]], module_kwargs: Optional[Dict[str, Any]]) -> dspy.Module:
    mk = dict(module_kwargs or {})
    if tools:
        mk.setdefault("tools", tools)
    if isinstance(module_kind, str):
        m = module_kind.lower()
        if m in {"predict", "p"}:
            return dspy.Predict(Sig, **mk)
        if m in {"cot", "chainofthought"}:
            return dspy.ChainOfThought(Sig, **mk)
        if m in {"react", "ra"}:
            return dspy.ReAct(Sig, **mk)
        raise ValueError(f"Unknown module '{module_kind}'.")
    if isinstance(module_kind, type) and issubclass(module_kind, dspy.Module):
        return module_kind(Sig, **mk)
    if isinstance(module_kind, dspy.Module):
        try:
            module_kind.signature = Sig
            return module_kind
        except Exception:
            return type(module_kind)(Sig, **mk)
    raise TypeError("module must be a string, a dspy.Module subclass, or an instance.")

# ──────────────────────────────────────────────────────────────────────────────
# Call context and `_ai` sentinel
# ──────────────────────────────────────────────────────────────────────────────

class _CallContext:
    def __init__(self, *, program: "FunctAIFunc", Sig: type[Signature], inputs: Dict[str, Any], adapter: Optional[dspy.Adapter], main_output_name: Optional[str] = None):
        self.program = program
        self.Sig = Sig
        self.inputs = inputs
        self.adapter = adapter
        self.main_output_name = main_output_name or MAIN_OUTPUT_DEFAULT_NAME

        self._materialized = False
        self._pred: Optional[Prediction] = None
        self._value: Any = None
        self._ai_requested = False
        self.collect_only: bool = False
        self._requested_outputs: Dict[str, Tuple[Any, str]] = {}

    def request_ai(self):
        self._ai_requested = True
        return self

    # Dynamic outputs declared via _ai["..."]
    def declare_output(self, *, name: str, typ: Any = str, desc: str = "") -> None:
        if not name:
            return
        if name not in self._requested_outputs:
            self._requested_outputs[name] = (typ or str, desc or "")

    def requested_outputs(self) -> List[Tuple[str, Any, str]]:
        return [(n, t, d) for n, (t, d) in self._requested_outputs.items()]

    def ensure_materialized(self):
        if self._materialized:
            return
        if self.collect_only:
            raise RuntimeError("_ai value accessed before model run; declare outputs with _ai[\"desc\"] and return _ai.")

        # Build/refresh module
        mod = _instantiate_module(
            self.program._module_kind,
            self.Sig,
            tools=self.program._tools,
            module_kwargs=self.program._module_kwargs,
        )

        # Wire LM & generation knobs
        lm_inst = self.program._lm_instance
        if lm_inst is not None:
            try:
                mod.lm = lm_inst
            except Exception:
                pass
        if self.program.temperature is not None:
            try:
                setattr(mod, "temperature", float(self.program.temperature))
            except Exception:
                pass

        # Wire History (DSPy handles how/if it renders)
        if self.program.history is not None:
            try:
                setattr(mod, "history", self.program.history)
            except Exception:
                pass

        # Normalize inputs (strings or pass through)
        in_kwargs = {k: self._to_text(v) for k, v in self.inputs.items() if k in (mod.signature.input_fields or {})}

        with _patched_adapter(self.adapter), _patched_lm(getattr(mod, "lm", None)):
            self._pred = mod(**in_kwargs)

        self._value = dict(self._pred).get(self.main_output_name)
        self._materialized = True

    @staticmethod
    def _to_text(v: Any) -> Any:
        if isinstance(v, list):
            return [_CallContext._to_text(x) for x in v]
        if isinstance(v, (str, dict)):
            return v
        return str(v)

    @property
    def value(self):
        self.ensure_materialized()
        return self._value

    def output_value(self, name: str, typ: Any = str):
        self.ensure_materialized()
        if self._pred is None:
            return None
        data = dict(self._pred)
        return data.get(name)

# Active call ctx
from contextvars import ContextVar
_ACTIVE_CALL: ContextVar[Optional[_CallContext]] = ContextVar("functai_active_call", default=None)

class _AISentinel:
    """Module-level `_ai` sentinel."""

    def __repr__(self):
        return "<_ai>"

    def __getattr__(self, name):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai` can only be used inside an @ai-decorated function call.")
        val = ctx.request_ai().value
        return getattr(val, name)

    def _val(self):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai` can only be used inside an @ai-decorated function call.")
        return ctx.request_ai().value

    # Conversions & operators
    def __str__(self): return str(self._val())
    def __int__(self): return int(self._val())
    def __float__(self): return float(self._val())
    def __bool__(self): return bool(self._val())
    def __len__(self): return len(self._val())
    def __iter__(self): return iter(self._val())
    def __getitem__(self, k): return self._val()[k]
    def __contains__(self, k): return k in self._val()
    def __add__(self, other):   return self._val() + other
    def __radd__(self, other):  return other + self._val()
    def __sub__(self, other):   return self._val() - other
    def __rsub__(self, other):  return other - self._val()
    def __mul__(self, other):   return self._val() * other
    def __rmul__(self, other):  return other * self._val()
    def __truediv__(self, other):  return self._val() / other
    def __rtruediv__(self, other): return other / self._val()
    def __eq__(self, other):    return self._val() == other
    def __ne__(self, other):    return self._val() != other
    def __lt__(self, other):    return self._val() < other
    def __le__(self, other):    return self._val() <= other
    def __gt__(self, other):    return self._val() > other
    def __ge__(self, other):    return self._val() >= other

    def __getitem__(self, spec):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai[...]` can only be used inside an @ai-decorated function call.")
        ctx.request_ai()
        if isinstance(spec, str):
            # Treat bare string as description only. Try to bind the proxy's
            # name to the variable declared in the function (via AST), falling
            # back to a derived placeholder if no AST binding exists (e.g., in
            # non-assignment usages).
            desc = spec
            bound_name: Optional[str] = None
            try:
                # Match by exact description string from the function's AST-collected outputs
                for n, _t, d in _collect_ast_outputs(ctx.program._fn):
                    if d == desc:
                        bound_name = n
                        break
            except Exception:
                bound_name = None

            name = bound_name if bound_name else _derive_output_name(desc)
            # Record the output request with the chosen name and description.
            ctx.declare_output(name=name, typ=str, desc=desc)
            return _AIFieldProxy(ctx, name=name, typ=str)
        if isinstance(spec, tuple) and len(spec) >= 2:
            name = str(spec[0])
            desc = str(spec[1])
            typ = spec[2] if len(spec) >= 3 else str
            ctx.declare_output(name=name, typ=typ, desc=desc)
            return _AIFieldProxy(ctx, name=name, typ=typ)
        raise TypeError("_ai[...] expects a string description or (name, desc[, type]) tuple.")

class _AIFieldProxy:
    def __init__(self, ctx: _CallContext, *, name: str, typ: Any = str):
        self._ctx = ctx
        self._name = name
        self._typ = typ or str

    def _resolve(self):
        if self._ctx.collect_only:
            raise RuntimeError(f"Output '{self._name}' value is not available during signature collection.")
        return self._ctx.output_value(self._name, self._typ)

    def __repr__(self):
        try:
            v = self._resolve()
            return f"<_ai[{self._name!s}]={v!r}>"
        except Exception:
            return f"<_ai[{self._name!s}]>"

    def __str__(self): return str(self._resolve())
    def __int__(self): return int(self._resolve())
    def __float__(self): return float(self._resolve())
    def __bool__(self): return bool(self._resolve())
    def __len__(self): return len(self._resolve())
    def __iter__(self): return iter(self._resolve())
    def __getitem__(self, k): return self._resolve()[k]
    def __contains__(self, k): return k in self._resolve()
    def __add__(self, other):   return self._resolve() + other
    def __radd__(self, other):  return other + self._resolve()
    def __sub__(self, other):   return self._resolve() - other
    def __rsub__(self, other):  return other - self._resolve()
    def __mul__(self, other):   return self._resolve() * other
    def __rmul__(self, other):  return other * self._resolve()
    def __truediv__(self, other):  return self._resolve() / other
    def __rtruediv__(self, other): return other / self._resolve()
    def __eq__(self, other):    return self._resolve() == other
    def __ne__(self, other):    return self._resolve() != other
    def __lt__(self, other):    return self._resolve() < other
    def __le__(self, other):    return self._resolve() <= other
    def __gt__(self, other):    return self._resolve() > other
    def __ge__(self, other):    return self._resolve() >= other

def _derive_output_name(desc: str) -> str:
    s = ''.join(ch if (ch.isalnum() or ch == '_') else ' ' for ch in str(desc))
    s = s.strip().lower()
    if not s:
        return "field"
    return s.split()[0]

_ai = _AISentinel()

# ──────────────────────────────────────────────────────────────────────────────
# Program wrapper returned by @ai
# ──────────────────────────────────────────────────────────────────────────────

class FunctAIFunc:
    """Callable function-like object with live knobs, history, optimizer, and in-place .opt()."""

    def __init__(self, fn, *, lm=None, adapter=None, module=None, tools: Optional[List[Any]] = None,
                 temperature: Optional[float] = None, stateful: Optional[bool] = None, module_kwargs: Optional[Dict[str, Any]] = None):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._sig = inspect.signature(fn)
        self._return_type = (typing.get_type_hints(fn, include_extras=True) or {}).get(
            "return",
            self._sig.return_annotation if self._sig.return_annotation is not inspect._empty else Any  # type: ignore
        )

        # Defaults cascade
        defs = _effective_defaults()
        self._lm = lm if lm is not None else defs.lm
        self._adapter = adapter if adapter is not None else defs.adapter
        self._module_kind = _select_module_kind(module if module is not None else defs.module, tools)
        self._tools: List[Any] = list(tools or [])
        self.temperature: Optional[float] = (float(temperature) if temperature is not None else defs.temperature)
        self._module_kwargs: Dict[str, Any] = dict(module_kwargs or {})

        # History (DSPy)
        self.history: Optional[dspy.History] = None
        use_state = stateful if stateful is not None else defs.stateful
        if use_state:
            try:
                self.history = dspy.History(window=int(defs.state_window))
            except Exception:
                self.history = None

        # Optimizer
        self._optimizer = defs.optimizer  # may be instance or factory

        # Resolved objects
        self._lm_instance = self._to_lm(self._lm)
        self._adapter_instance = _select_adapter(self._adapter)

        # Compiled module (by .opt) and optimization history
        self._compiled: Optional[dspy.Module] = None
        self._opt_stack: List[dspy.Module] = []
        self._initial_module_kind = self._module_kind

        # Debug (preview)
        self._debug = bool(defs.debug)

        # Expose a dunder for helpers (format_prompt / inspection)
        self.__dspy__ = SimpleNamespace(fn=self._fn, program=self)

    # Signature (live)
    @property
    def signature(self):
        try:
            if isinstance(self._module_kind, dspy.Module) and hasattr(self._module_kind, "signature"):
                return self._module_kind.signature
        except Exception:
            pass
        return compute_signature(self)

    # ----- representations -----
    def __repr__(self) -> str:
        try:
            Sig = self.signature
            ins = list((getattr(Sig, "input_fields", {}) or {}).keys())
            outs = list((getattr(Sig, "output_fields", {}) or {}).keys())
            main_out = outs[-1] if outs else MAIN_OUTPUT_DEFAULT_NAME
            parts = []
            if ins:
                parts.append("inputs=" + ", ".join(ins))
            if outs:
                parts.append("outputs=" + ", ".join(outs) + f" (primary={main_out})")
            # Keep overrides terse
            if self._tools:
                parts.append(f"module={type(self._module_kind).__name__ if isinstance(self._module_kind, dspy.Module) else self._module_kind}")
                parts.append("tools=✓")
            return f"<FunctAIFunc {self._fn.__name__} | " + "; ".join(parts) + ">"
        except Exception:
            return f"<FunctAIFunc {getattr(self._fn, '__name__', 'unknown')}>"

    # ----- properties (live-mutable) -----
    @property
    def lm(self): return self._lm
    @lm.setter
    def lm(self, v): self._lm = v; self._lm_instance = self._to_lm(v); self._compiled = None

    @property
    def adapter(self): return self._adapter
    @adapter.setter
    def adapter(self, v): self._adapter = v; self._adapter_instance = _select_adapter(v); self._compiled = None

    @property
    def module(self): return self._module_kind
    @module.setter
    def module(self, v): self._module_kind = v; self._compiled = None

    @property
    def tools(self): return list(self._tools)
    @tools.setter
    def tools(self, seq):
        self._tools = list(seq or [])
        if isinstance(self._module_kind, str) and self._module_kind.lower() in {"predict", "", "p"} and self._tools:
            self._module_kind = "react"   # auto-upgrade to ReAct (DSPy does the work)
        self._compiled = None

    @property
    def optimizer(self): return self._optimizer
    @optimizer.setter
    def optimizer(self, v): self._optimizer = v

    @property
    def debug(self): return self._debug
    @debug.setter
    def debug(self, v: bool): self._debug = bool(v)

    # ----- callable -----
    def __call__(self, *args, all: bool = False, **kwargs):
        # Back-compat: map deprecated _prediction to all
        if "_prediction" in kwargs:
            if kwargs.pop("_prediction"):
                all = True

        # Bind args to the *user* function (discard our control kwargs)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in {"_prediction", "all"}}
        bound = self._sig.bind_partial(*args, **clean_kwargs)
        bound.apply_defaults()
        inputs = {k: v for k, v in bound.arguments.items() if k in self._sig.parameters}

        # Build signature once
        Sig = self.signature
        outs = list((getattr(Sig, "output_fields", {}) or {}).keys())
        main_name = outs[-1] if outs else MAIN_OUTPUT_DEFAULT_NAME

        # Call context
        ctx = _CallContext(program=self, Sig=Sig, inputs=inputs, adapter=self._adapter_instance, main_output_name=main_name)
        token = _ACTIVE_CALL.set(ctx)
        try:
            result = self._fn(*args, **clean_kwargs)

            if self._debug:
                try:
                    print(f"[FunctAI] module={type(self._module_kind).__name__ if isinstance(self._module_kind, dspy.Module) else self._module_kind}; adapter={type(self._adapter_instance).__name__ if self._adapter_instance else type(getattr(dspy.settings,'adapter',None)).__name__}")
                    print(f"[FunctAI] inputs={list(inputs.keys())}; outputs={outs} (primary={main_name})")
                except Exception:
                    pass

            if all:
                _ = ctx.request_ai().value
                return ctx._pred

            if result is _ai or result is Ellipsis:
                return ctx.request_ai().value

            # Unwrap proxies if the user returned them directly/nested
            def _unwrap(x):
                if isinstance(x, _AIFieldProxy):
                    return x._resolve()
                if isinstance(x, list):
                    return [_unwrap(i) for i in x]
                if isinstance(x, tuple):
                    return tuple(_unwrap(i) for i in x)
                if isinstance(x, dict):
                    return {k: _unwrap(v) for k, v in x.items()}
                return x

            result = _unwrap(result)
            if result is None and not ctx._ai_requested:
                return ctx.request_ai().value
            return result
        finally:
            _ACTIVE_CALL.reset(token)

    # ----- optimization -----
    def opt(self, *, trainset: Optional[List[Any]] = None, optimizer: Any = None, **opts) -> None:
        """
        Compile with a DSPy optimizer and mutate in place.
        - optimizer: instance or factory; if None, uses self.optimizer or global default.
        - trainset: list of (inputs, outputs) pairs or DSPy's preferred format.
        Additional **opts are forwarded to the optimizer constructor if it is a factory.
        """
        Sig = self.signature

        # Build a base module to optimize
        if self._compiled is not None:
            try:
                self._compiled.signature = Sig
                base_mod = self._compiled
            except Exception:
                base_mod = type(self._compiled)(Sig, **(self._module_kwargs or {}))
                if self._tools:
                    try:
                        base_mod.tools = self._tools
                    except Exception:
                        pass
        else:
            base_mod = _instantiate_module(self._module_kind, Sig, tools=self._tools, module_kwargs=self._module_kwargs)

        # Pick optimizer
        opt = optimizer if optimizer is not None else (self._optimizer if self._optimizer is not None else dspy.BootstrapFewShot)
        if isinstance(opt, type):
            optimizer_instance = opt(**opts)  # factory/class
        else:
            optimizer_instance = opt           # already an instance; opts ignored

        new_prog = optimizer_instance.compile(base_mod, trainset=trainset)

        # Track history and swap in compiled program
        if self._compiled is not None:
            self._opt_stack.append(self._compiled)
        self._compiled = new_prog
        self._module_kind = self._compiled  # instance; signature will rebind per call

    def undo_opt(self, steps: int = 1) -> None:
        steps = max(1, int(steps))
        for _ in range(steps):
            if self._opt_stack:
                self._compiled = self._opt_stack.pop()
                self._module_kind = self._compiled
            else:
                self._compiled = None
                self._module_kind = self._initial_module_kind
                break

    # ----- helpers -----
    @staticmethod
    def _to_lm(v: Any):
        if v is None:
            return None
        if isinstance(v, str):
            ak = _effective_defaults().api_key
            try:
                return dspy.LM(v, api_key=ak) if ak is not None else dspy.LM(v)
            except TypeError:
                # older DSPy may not accept api_key kwarg
                return dspy.LM(v)
        return v

class SimpleNamespace:
    def __init__(self, **kw): self.__dict__.update(kw)

# ──────────────────────────────────────────────────────────────────────────────
# Decorator: @ai  (works with @ai and @ai(...))
# ──────────────────────────────────────────────────────────────────────────────

def ai(_fn=None, **cfg):
    """Decorator that turns a typed Python function into a single-call DSPy program.

    Usage:
        @ai
        def f(...)->T:
            # declare outputs
            y: T = _ai["..."]
            return _ai  # last declared is the default primary

        @ai(lm="openai:gpt-4o-mini", tools=[...], temperature=0.2, stateful=True)
        def g(...)->T:
            x: T = _ai["..."]
            return _ai
    """
    def _decorate(fn):
        return FunctAIFunc(fn, **cfg)
    if _fn is not None and callable(_fn):
        return _decorate(_fn)
    return _decorate

# ──────────────────────────────────────────────────────────────────────────────
# Prompt preview (kept minimal; adapters ultimately render)
# ──────────────────────────────────────────────────────────────────────────────

def _default_user_content(sig: Signature, inputs: Dict[str, Any]) -> str:
    lines = []
    doc = (getattr(sig, "__doc__", "") or "").strip()
    if doc:
        lines.append(doc)
        lines.append("")
    if inputs:
        lines.append("Inputs:")
        for k, v in inputs.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    outs = getattr(sig, "output_fields", {}) or {}
    if outs:
        lines.append("Please produce the following outputs:")
        for k in outs.keys():
            lines.append(f"- {k}")
    return "\n".join(lines).strip()

def format_prompt(fn_or_prog, /, **inputs) -> Dict[str, Any]:
    """Return a minimal preview of adapter-formatted messages for a call."""
    prog: FunctAIFunc
    if isinstance(fn_or_prog, FunctAIFunc):
        prog = fn_or_prog
    elif hasattr(fn_or_prog, "__wrapped__") and isinstance(getattr(fn_or_prog, "__wrapped__"), FunctAIFunc):
        prog = getattr(fn_or_prog, "__wrapped__")
    elif hasattr(fn_or_prog, "__dspy__") and hasattr(fn_or_prog.__dspy__, "program"):  # type: ignore
        prog = fn_or_prog.__dspy__.program  # type: ignore
    else:
        raise TypeError("format_prompt(...) expects an @ai-decorated function/program.")

    Sig = prog.signature
    adapter_inst = prog._adapter_instance or getattr(dspy.settings, "adapter", None) or dspy.ChatAdapter()

    in_text = {k: str(v) for k, v in inputs.items() if k in (Sig.input_fields or {})}

    # Adapter-native formatting if available
    system_content = (getattr(Sig, "__doc__", "") or "").strip()
    user_content = None
    try:
        if hasattr(adapter_inst, "format_user_message_content"):
            user_content = adapter_inst.format_user_message_content(Sig, in_text)
    except Exception:
        user_content = None
    if not user_content:
        user_content = _default_user_content(Sig, in_text)

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    render_lines = []
    render_lines.append(f"Adapter: {type(adapter_inst).__name__}")
    render_lines.append(f"Module: {prog._module_kind.__class__.__name__ if isinstance(prog._module_kind, dspy.Module) else str(prog._module_kind)}")
    if system_content:
        render_lines.append("")
        render_lines.append("System:")
        render_lines.append(system_content)
    render_lines.append("")
    render_lines.append("User:")
    render_lines.append(user_content)
    render = "\n".join(render_lines).strip()

    return {
        "adapter": type(adapter_inst).__name__,
        "module": prog._module_kind.__class__.__name__ if isinstance(prog._module_kind, dspy.Module) else str(prog._module_kind),
        "inputs": in_text,
        "messages": messages,
        "render": render,
        "signature": Sig,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Signature helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_signature(fn_or_prog):
    """Build and return the computed dspy.Signature for a function/program."""
    prog: FunctAIFunc
    if isinstance(fn_or_prog, FunctAIFunc):
        prog = fn_or_prog
    elif hasattr(fn_or_prog, "__wrapped__") and isinstance(getattr(fn_or_prog, "__wrapped__"), FunctAIFunc):
        prog = getattr(fn_or_prog, "__wrapped__")
    elif hasattr(fn_or_prog, "__dspy__") and hasattr(fn_or_prog.__dspy__, "program"):
        prog = fn_or_prog.__dspy__.program  # type: ignore
    else:
        raise TypeError("compute_signature(...) expects an @ai-decorated function/program.")

    sysdoc = _compose_system_doc(prog._fn, include_fn_name=bool(_effective_defaults().include_fn_name_in_instructions))

    ast_outputs = _collect_ast_outputs(prog._fn)
    ret_info = _collect_return_info(prog._fn)
    order_names = [n for n, _, _ in ast_outputs]

    if ret_info.get("mode") == "name" and ret_info.get("name") in order_names:
        main_name = typing.cast(str, ret_info.get("name"))
    elif ret_info.get("mode") in {"sentinel", "ellipsis", "empty"}:
        main_name = MAIN_OUTPUT_DEFAULT_NAME
    elif order_names:
        main_name = order_names[-1]
    else:
        main_name = MAIN_OUTPUT_DEFAULT_NAME

    ast_map: Dict[str, Tuple[Any, str]] = {n: (t, d) for n, t, d in ast_outputs}
    if main_name in ast_map:
        t0, d0 = ast_map[main_name]
        if ret_info.get("mode") in {"sentinel", "ellipsis"} or (ret_info.get("mode") == "name" and ret_info.get("name") == main_name):
            main_typ = prog._return_type
        else:
            main_typ = t0 if t0 is not None else str
        main_desc = d0
    else:
        main_typ = prog._return_type
        main_desc = ""

    extras = [(n, (ast_map[n][0] if ast_map[n][0] is not None else str), ast_map[n][1]) for n in order_names if n != main_name]
    Sig = _mk_signature(prog._fn.__name__, prog._fn, doc=sysdoc, return_type=prog._return_type, extra_outputs=extras, main_output=(main_name, main_typ, main_desc))
    return Sig

def signature_text(fn_or_prog) -> str:
    """Return a tiny, human-readable summary of the computed Signature."""
    Sig = compute_signature(fn_or_prog)
    anns: Dict[str, Any] = getattr(Sig, "__annotations__", {}) or {}
    inputs = list((getattr(Sig, "input_fields", {}) or {}).keys())
    outputs = list((getattr(Sig, "output_fields", {}) or {}).keys())
    doc = (getattr(Sig, "__doc__", "") or "").strip()
    main_name = outputs[-1] if outputs else MAIN_OUTPUT_DEFAULT_NAME

    def _tostr(t: Any) -> str:
        try:
            return str(t)
        except Exception:
            return repr(t)

    lines: List[str] = []
    lines.append(f"Signature: {Sig.__name__}")
    if doc:
        lines.append(" | Doc: " + (doc[:120] + ("…" if len(doc) > 120 else "")))
    if inputs:
        lines.append(" | Inputs: " + ", ".join(f"{k}:{_tostr(anns.get(k, str))}" for k in inputs))
    if outputs:
        lines.append(" | Outputs: " + ", ".join(f"{k}{'*' if k == main_name else ''}" for k in outputs))
    return "".join(lines).strip()

def inspect_history_text() -> str:
    """Return dspy.inspect_history() as text (best effort)."""
    import io
    import contextlib as _ctx
    buf = io.StringIO()
    try:
        with _ctx.redirect_stdout(buf):
            try:
                dspy.inspect_history()
            except Exception:
                pass
    except Exception:
        return ""
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ai",
    "_ai",
    "configure",
    "format_prompt",
    "inspect_history_text",
    "settings",
    "compute_signature",
    "signature_text",
]
