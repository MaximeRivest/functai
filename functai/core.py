# functai_v02.py
# New-style FunctAI core: @ai decorator + bare `_ai` sentinel
# Single-call programs; docstring-driven; stateful; auto-ReAct; live knobs; in-place .opt()

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import is_dataclass, fields as dataclass_fields
import dataclasses
import functools
import inspect
import json
import typing
from typing import Any, Dict, Optional, List, Tuple, get_origin, get_args, TypedDict

import dspy
from dspy import Signature, InputField, OutputField, Prediction


# ──────────────────────────────────────────────────────────────────────────────
# Defaults & configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _Defaults:
    lm: Any = None                 # str | dspy.LM | None
    temperature: Optional[float] = None
    adapter: Any = None            # "json" | "chat" | dspy.Adapter subclass | instance | None
    module: Any = "predict"        # "predict" | "cot" | "react" | dspy.Module subclass | instance | None
    stateful: bool = False

_GLOBAL_DEFAULTS = _Defaults()
_CTX_DEFAULTS: ContextVar[_Defaults] = ContextVar("functai_defaults_ctx", default=None)  # type: ignore


def configure(*, lm: Any = None, temperature: Optional[float] = None,
              adapter: Any = None, module: Any = None, stateful: Optional[bool] = None) -> None:
    """Set project-wide defaults used when @ai() has no explicit kwargs."""
    if lm is not None:
        _GLOBAL_DEFAULTS.lm = lm
    if temperature is not None:
        _GLOBAL_DEFAULTS.temperature = float(temperature)
    if adapter is not None:
        _GLOBAL_DEFAULTS.adapter = adapter
    if module is not None:
        _GLOBAL_DEFAULTS.module = module
    if stateful is not None:
        _GLOBAL_DEFAULTS.stateful = bool(stateful)


@contextlib.contextmanager
def defaults(**overrides):
    """Temporarily override defaults within a 'with' block."""
    base = _CTX_DEFAULTS.get() or dataclasses.replace(_GLOBAL_DEFAULTS)
    temp = dataclasses.replace(base)
    for k, v in overrides.items():
        if hasattr(temp, k):
            setattr(temp, k, v)
    token = _CTX_DEFAULTS.set(temp)
    try:
        yield
    finally:
        _CTX_DEFAULTS.reset(token)


def _effective_defaults() -> _Defaults:
    ctx = _CTX_DEFAULTS.get()
    if ctx is not None:
        return ctx
    return _GLOBAL_DEFAULTS


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
# Type coercion helper
# ──────────────────────────────────────────────────────────────────────────────

def _from_text(txt: Any, typ: Any) -> Any:
    if typ is Any or typ is None:
        return txt
    if not isinstance(txt, str):
        return txt

    try:
        origin = get_origin(typ)
        args = get_args(typ)

        if typ is str:
            return txt
        if typ is float:
            return float(txt)
        if typ is int:
            s = txt.strip()
            return int(float(s)) if "." in s else int(s)
        if typ is bool:
            return txt.strip().lower() in {"true", "1", "yes", "y"}

        if origin is list and args:
            inner = args[0]
            s = txt.strip()
            data = json.loads(s) if s.startswith("[") and s.endswith("]") else [x.strip() for x in txt.split(",") if x.strip()]
            return [_from_text(x if isinstance(x, str) else json.dumps(x), inner) for x in data]

        if origin is dict and args:
            kt, vt = args
            s = txt.strip()
            data = json.loads(s) if s.startswith("{") and s.endswith("}") else {}
            return { _from_text(str(k), kt): _from_text(v if isinstance(v, str) else json.dumps(v), vt) for k, v in data.items() }

        if is_dataclass(typ):
            data = json.loads(txt) if txt.strip().startswith("{") else {}
            vals = {}
            for f in dataclass_fields(typ):
                vals[f.name] = _from_text(data.get(f.name), f.type) if f.name in data else None
            return typ(**vals)  # type: ignore

    except Exception:
        return txt

    return txt


# ──────────────────────────────────────────────────────────────────────────────
# Signature building (docstring-driven)
# ──────────────────────────────────────────────────────────────────────────────

def _mk_signature(fn_name: str, fn: Any, *, doc: str, return_type: Any) -> type[Signature]:
    """Create a dspy.Signature with inputs from fn params and single 'result' output."""
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True) if hasattr(typing, "get_type_hints") else {}
    class_dict: Dict[str, Any] = {}
    ann_map: Dict[str, Any] = {}

    # Inputs (skip reserved)
    for pname, p in sig.parameters.items():
        if pname in {"_prediction"}:
            continue
        ann = hints.get(pname, p.annotation if p.annotation is not inspect._empty else str)  # default str
        class_dict[pname] = InputField()
        ann_map[pname] = ann

    # Single output "result" typed to return annotation
    class_dict["result"] = OutputField(desc="")
    ann_map["result"] = return_type

    # Attach doc
    if doc:
        class_dict["__doc__"] = doc
    class_dict["__annotations__"] = ann_map

    Sig = type(f"{fn_name.title()}Sig", (Signature,), class_dict)
    return Sig


def _compose_system_doc(fn: Any, persona: Optional[str], state_block: Optional[str]) -> str:
    parts = []
    if persona:
        parts.append(f"Persona: {persona}")
    base = (fn.__doc__ or "").strip()
    if base:
        parts.append(base)
    if state_block:
        parts.append("\nRecent context:\n" + state_block.strip())
    return "\n\n".join([p for p in parts if p]).strip()


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
        # Rebind signature if possible
        try:
            module_kind.signature = Sig
            return module_kind
        except Exception:
            # fallback: rebuild a fresh instance of its class
            return type(module_kind)(Sig, **mk)
    raise TypeError("module must be a string, a dspy.Module subclass, or an instance.")


# ──────────────────────────────────────────────────────────────────────────────
# State management
# ──────────────────────────────────────────────────────────────────────────────

class _State:
    def __init__(self):
        self.enabled = False
        self.window = 0
        self._turns: List[Dict[str, Any]] = []

    def enable(self, window: int = 5) -> "_State":
        self.enabled = True
        self.window = int(window)
        return self

    def disable(self) -> "_State":
        self.enabled = False
        return self

    def clear(self) -> None:
        self._turns.clear()

    def add_turn(self, inputs: Dict[str, Any], outputs: Any) -> None:
        self._turns.append({"inputs": inputs, "outputs": outputs})
        if self.window and len(self._turns) > self.window:
            del self._turns[0 : len(self._turns) - self.window]

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._turns)

    def render_block(self) -> Optional[str]:
        if not (self.enabled and self._turns):
            return None
        # compact JSON-ish block
        try:
            return json.dumps(self._turns[-self.window:], ensure_ascii=False)
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Call context and `_ai` sentinel
# ──────────────────────────────────────────────────────────────────────────────

class _CallContext:
    def __init__(self, *, program: "FunctAIFunc", Sig: type[Signature], inputs: Dict[str, Any], adapter: Optional[dspy.Adapter]):
        self.program = program
        self.Sig = Sig
        self.inputs = inputs
        self.adapter = adapter

        self._materialized = False
        self._pred: Optional[Prediction] = None
        self._value: Any = None
        self._ai_requested = False  # whether `_ai` has been accessed

    def request_ai(self):
        """Mark that `_ai` is being used and return self for Proxy access."""
        self._ai_requested = True
        return self

    def ensure_materialized(self):
        if self._materialized:
            return
        # Build/refresh module for this call
        mod = _instantiate_module(
            self.program._module_kind,  # may be compiled instance swapped by .opt()
            self.Sig,
            tools=self.program._tools,
            module_kwargs=self.program._module_kwargs,
        )
        # Apply LM & generation knobs
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

        # Normalize inputs to strings where needed
        in_kwargs = {k: self._to_text(v) for k, v in self.inputs.items() if k in (mod.signature.input_fields or {})}

        with _patched_adapter(self.adapter), _patched_lm(getattr(mod, "lm", None)):
            self._pred = mod(**in_kwargs)

        # Extract single "result"
        raw = dict(self._pred).get("result")
        self._value = _from_text(raw, self.program._return_type)
        self._materialized = True

        # Record state
        if self.program.state.enabled:
            self.program.state.add_turn(inputs=self.inputs, outputs=self._value)

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

# Active call ctx (per-thread/async)
_ACTIVE_CALL: ContextVar[Optional[_CallContext]] = ContextVar("functai_active_call", default=None)


class _AISentinel:
    """Module-level `_ai` sentinel: when used inside an @ai function, resolves to the call's final value.

    You can return `_ai` directly or assign it to a variable and post-process before returning.
    """

    # Representations
    def __repr__(self):
        return "<_ai>"

    # Proxy: attribute access forwards to resolved value
    def __getattr__(self, name):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai` can only be used inside an @ai-decorated function call.")
        val = ctx.request_ai().value
        return getattr(val, name)

    # Conversions & operators for ergonomic post-processing
    def _val(self):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai` can only be used inside an @ai-decorated function call.")
        return ctx.request_ai().value

    def __str__(self):   return str(self._val())
    def __int__(self):   return int(self._val())
    def __float__(self): return float(self._val())
    def __bool__(self):  return bool(self._val())
    def __len__(self):   return len(self._val())
    def __iter__(self):  return iter(self._val())
    def __getitem__(self, k): return self._val()[k]
    def __contains__(self, k): return k in self._val()

    # Arithmetic-ish
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


# Public sentinel instance
_ai = _AISentinel()


# ──────────────────────────────────────────────────────────────────────────────
# Program wrapper returned by @ai
# ──────────────────────────────────────────────────────────────────────────────

class FunctAIFunc:
    """Callable function-like object with live knobs, state, and in-place .opt()."""

    def __init__(self, fn, *, lm=None, adapter=None, module=None, tools: Optional[List[Any]] = None,
                 temperature: Optional[float] = None, stateful: Optional[bool] = None, module_kwargs: Optional[Dict[str, Any]] = None):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._sig = inspect.signature(fn)
        self._return_type = (typing.get_type_hints(fn, include_extras=True) or {}).get("return",
                           self._sig.return_annotation if self._sig.return_annotation is not inspect._empty else Any)  # type: ignore

        # Defaults cascade
        defs = _effective_defaults()
        self._lm = lm if lm is not None else defs.lm
        self._adapter = adapter if adapter is not None else defs.adapter
        self._module_kind = _select_module_kind(module if module is not None else defs.module, tools)
        self._tools: List[Any] = list(tools or [])
        self.temperature: Optional[float] = (float(temperature) if temperature is not None else defs.temperature)
        self.state = _State()
        if stateful if stateful is not None else defs.stateful:
            self.state.enable(window=5)
        self.persona: Optional[str] = None
        self._module_kwargs: Dict[str, Any] = dict(module_kwargs or {})

        # Resolved objects
        self._lm_instance = self._to_lm(self._lm)
        self._adapter_instance = _select_adapter(self._adapter)

        # Compiled module (by .opt)
        self._compiled: Optional[dspy.Module] = None

        # Expose a dunder for helpers (format_prompt / inspection)
        self.__dspy__ = SimpleNamespace(   # type: ignore[name-defined]
            fn=self._fn,
            program=self,
        )

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
        # auto-upgrade to ReAct when no explicit module chosen
        if isinstance(self._module_kind, str) and self._module_kind.lower() in {"predict", "", "p"} and self._tools:
            self._module_kind = "react"
        self._compiled = None

    # ----- callable -----
    def __call__(self, *args, _prediction: bool = False, **kwargs):
        # Bind args
        bound = self._sig.bind_partial(*args, **{k: v for k, v in kwargs.items() if k != "_prediction"})
        bound.apply_defaults()
        inputs = {k: v for k, v in bound.arguments.items() if k in self._sig.parameters}

        # Compose dynamic system doc (docstring + persona + state)
        state_block = self.state.render_block()
        sysdoc = _compose_system_doc(self._fn, self.persona, state_block)

        # Signature for this call
        Sig = _mk_signature(self._fn.__name__, self._fn, doc=sysdoc, return_type=self._return_type)

        # Build call context & activate
        ctx = _CallContext(program=self, Sig=Sig, inputs=inputs, adapter=self._adapter_instance)
        token = _ACTIVE_CALL.set(ctx)
        try:
            result = self._fn(*args, **kwargs)
            if _prediction:
                # force materialization to expose raw prediction
                _ = ctx.request_ai().value
                return ctx._pred
            # If they returned the sentinel itself, resolve to real value
            if result is _ai:
                return ctx.request_ai().value
            # If they never touched _ai, but function is "empty-style" (returned None), auto-mode:
            if result is None and not ctx._ai_requested:
                return ctx.request_ai().value
            # Otherwise, return what they returned (likely post-processed Python value)
            return result
        finally:
            _ACTIVE_CALL.reset(token)

    # ----- in-place optimize -----
    def opt(self, *, trainset: Optional[List[Any]] = None, strategy: str = "launch", **opts) -> None:
        """Compile the program with a DSPy teleprompter and mutate in place."""
        # Build a temporary module using current signature to compile against
        Sig = _mk_signature(self._fn.__name__, self._fn,
                            doc=_compose_system_doc(self._fn, self.persona, self.state.render_block()),
                            return_type=self._return_type)
        base_mod = _instantiate_module(self._module_kind, Sig, tools=self._tools, module_kwargs=self._module_kwargs)

        # Choose optimizer
        if strategy == "launch":
            optimizer = dspy.BootstrapFewShot(**opts)
        else:
            # Fallback to BFS; you can wire more strategies here
            optimizer = dspy.BootstrapFewShot(**opts)

        self._compiled = optimizer.compile(base_mod, trainset=trainset)
        # After compilation, prefer the compiled module kind directly:
        self._module_kind = self._compiled  # instance; future calls will rebind signature

    # ----- helpers -----
    @staticmethod
    def _to_lm(v: Any):
        if v is None:
            return None
        if isinstance(v, str):
            return dspy.LM(v)
        return v


# Shim for __dspy__ reference in FunctAIFunc.__init__
class SimpleNamespace:
    def __init__(self, **kw): self.__dict__.update(kw)


# ──────────────────────────────────────────────────────────────────────────────
# Decorator: @ai  (works with @ai  and @ai(...))
# ──────────────────────────────────────────────────────────────────────────────

def ai(_fn=None, **cfg):
    """Decorator that turns a typed Python function into a single-call LLM program.

    Usage:
        @ai
        def f(...)->T:
            return _ai

        @ai(lm="gpt-4.1", tools=[...], temperature=0.2, stateful=True)
        def g(...)->T:
            raw = _ai
            return postprocess(raw)
    """
    def _decorate(fn):
        return FunctAIFunc(fn, **cfg)

    if _fn is not None and callable(_fn):
        # Bare @ai
        return _decorate(_fn)
    # @ai(...)
    return _decorate


# ──────────────────────────────────────────────────────────────────────────────
# Prompt preview & history utilities
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
    """Return a best-effort preview of adapter-formatted messages for a call."""
    # Accept either the wrapper object or the original function if wrapped
    prog: FunctAIFunc
    if isinstance(fn_or_prog, FunctAIFunc):
        prog = fn_or_prog
    elif hasattr(fn_or_prog, "__wrapped__") and isinstance(getattr(fn_or_prog, "__wrapped__"), FunctAIFunc):
        prog = getattr(fn_or_prog, "__wrapped__")
    elif hasattr(fn_or_prog, "__dspy__") and hasattr(fn_or_prog.__dspy__, "program"):  # type: ignore
        prog = fn_or_prog.__dspy__.program  # type: ignore
    else:
        raise TypeError("format_prompt(...) expects an @ai-decorated function/program.")

    # Compose signature
    sysdoc = _compose_system_doc(prog._fn, prog.persona, prog.state.render_block())
    Sig = _mk_signature(prog._fn.__name__, prog._fn, doc=sysdoc, return_type=prog._return_type)
    adapter_inst = prog._adapter_instance or getattr(dspy.settings, "adapter", None) or dspy.ChatAdapter()

    # Normalize inputs
    in_text = {k: str(v) for k, v in inputs.items() if k in (Sig.input_fields or {})}

    # Try adapter-native formatting
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
    "defaults",
    "format_prompt",
    "inspect_history_text",
]
