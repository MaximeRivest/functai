from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import is_dataclass, fields as dataclass_fields
import dataclasses
import functools
import inspect
import json
import ast
import typing
from typing import Any, Dict, Optional, List, Tuple, get_origin, get_args, TypedDict

import dspy
from dspy import Signature, InputField, OutputField, Prediction

# Main output field name when no explicit name is chosen
MAIN_OUTPUT_DEFAULT_NAME = "result"
# Default toggle for including function name in system instructions
INCLUDE_FN_NAME_IN_INSTRUCTIONS_DEFAULT = True


# ──────────────────────────────────────────────────────────────────────────────
# Defaults & configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _Defaults:
    lm: Any = None                 # str | dspy.LM | None
    temperature: Optional[float] = None
    # "json" | "chat" | dspy.Adapter subclass | instance | None
    adapter: Any = None
    # "predict" | "cot" | "react" | dspy.Module subclass | instance | None
    module: Any = "predict"
    stateful: bool = False
    include_fn_name_in_instructions: bool = INCLUDE_FN_NAME_IN_INSTRUCTIONS_DEFAULT


_GLOBAL_DEFAULTS = _Defaults()
_CTX_DEFAULTS: ContextVar[_Defaults] = ContextVar(
    "functai_defaults_ctx", default=None)  # type: ignore


def configure(*, lm: Any = None, temperature: Optional[float] = None,
              adapter: Any = None, module: Any = None, stateful: Optional[bool] = None,
              include_fn_name_in_instructions: Optional[bool] = None) -> None:
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
    if include_fn_name_in_instructions is not None:
        _GLOBAL_DEFAULTS.include_fn_name_in_instructions = bool(
            include_fn_name_in_instructions)


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
    raise TypeError(
        "adapter must be a string, a dspy.Adapter subclass, or a dspy.Adapter instance.")


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
    """Create a dspy.Signature with inputs from fn params and outputs.

    Always includes a 'result' OutputField for the function return value.
    Optionally includes additional outputs specified by `extra_outputs` as
    (name, type, description) tuples.
    """
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True) if hasattr(
        typing, "get_type_hints") else {}
    class_dict: Dict[str, Any] = {}
    ann_map: Dict[str, Any] = {}

    # Inputs (skip reserved)
    for pname, p in sig.parameters.items():
        if pname in {"_prediction"}:
            continue
        ann = hints.get(
            pname, p.annotation if p.annotation is not inspect._empty else str)  # default str
        class_dict[pname] = InputField()
        ann_map[pname] = ann

    # Optional additional outputs (e.g., think)
    if extra_outputs:
        for name, typ, desc in extra_outputs:
            # Do not shadow inputs
            if name in class_dict:
                continue
            class_dict[name] = OutputField(
                desc=str(desc) if desc is not None else "")
            ann_map[name] = typ if typ is not None else str

    # Primary output
    if main_output is None:
        mo_name, mo_type, mo_desc = MAIN_OUTPUT_DEFAULT_NAME, return_type, ""
    else:
        mo_name, mo_type, mo_desc = main_output
        if mo_type is None:
            mo_type = return_type
    # Avoid collision with inputs or extras
    if mo_name in class_dict:
        # If the main name shadows an input, rename fallback to default
        mo_name = MAIN_OUTPUT_DEFAULT_NAME
    class_dict[mo_name] = OutputField(
        desc=str(mo_desc) if mo_desc is not None else "")
    ann_map[mo_name] = mo_type

    # Attach doc
    if doc:
        class_dict["__doc__"] = doc
    class_dict["__annotations__"] = ann_map

    Sig = type(f"{fn_name.title()}Sig", (Signature,), class_dict)
    return Sig


def _compose_system_doc(fn: Any, persona: Optional[str], state_block: Optional[str], *, include_fn_name: bool) -> str:
    parts = []
    if persona:
        parts.append(f"Persona: {persona}")
    if include_fn_name and getattr(fn, "__name__", None):
        parts.append(f"Function: {fn.__name__}")
    base = (fn.__doc__ or "").strip()
    if base:
        parts.append(base)
    if state_block:
        parts.append("\nRecent context:\n" + state_block.strip())
    return "\n\n".join([p for p in parts if p]).strip()


# ──────────────────────────────────────────────────────────────────────────────
# AST-based collection of declared outputs (e.g., think: str = _ai)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_annotation(expr: ast.AST, env: Dict[str, Any]) -> Any:
    try:
        code = compile(ast.Expression(expr), filename="<ann>", mode="eval")
        return eval(code, env, {})
    except Exception:
        return str


def _extract_desc_from_subscript(node: ast.Subscript) -> str:
    # Accept _ai["desc"] or _ai[(..., "desc", ...)]
    try:
        sl = node.slice
        # Python 3.8+: ast.Index removed in 3.9; handle both
        if isinstance(sl, ast.Index):
            sl = sl.value
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
    """Collect outputs declared via assignments to `_ai` (annotated or not).

    Returns a list in the order of appearance with (name, type_or_str, desc).
    For unannotated assignments the type defaults to str here and may be
    overridden for the main output to match the function return type.
    """
    try:
        src = inspect.getsource(fn)
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []

    # Find the right function node
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
    # Traverse for AnnAssign and Assign nodes with value referencing _ai
    for node in ast.walk(fn_node):
        if isinstance(node, ast.AnnAssign):
            # Only handle simple names on LHS
            if not isinstance(node.target, ast.Name):
                continue
            name = node.target.id
            val = node.value
            if val is None:
                continue
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(
                val.value, ast.Name) and val.value.id == "_ai"
            if not (is_ai or is_ai_sub):
                continue
            # Type
            typ = _eval_annotation(
                node.annotation, env) if node.annotation is not None else str
            # Desc from subscript if any
            desc = _extract_desc_from_subscript(val) if is_ai_sub else ""
            # preserve order and de-dupe by name (first occurrence wins)
            if not any(n == name for n, _, _ in outputs_ordered):
                outputs_ordered.append((name, typ, desc))
        elif isinstance(node, ast.Assign):
            # Handle simple "x = _ai" or "x = _ai[...]" forms; allow multiple targets but only simple names
            if not node.targets:
                continue
            val = node.value
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(
                val.value, ast.Name) and val.value.id == "_ai"
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
    # Use last return statement if present
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


def _merge_outputs(primary: List[Tuple[str, Any, str]], secondary: List[Tuple[str, Any, str]]) -> List[Tuple[str, Any, str]]:
    by_name: Dict[str, Tuple[Any, str]] = {}
    for name, typ, desc in secondary:
        by_name[name] = (typ, desc)
    for name, typ, desc in primary:
        # primary wins, but fill missing desc/type with secondary if blank
        old = by_name.get(name)
        if old:
            t2, d2 = old
            by_name[name] = (typ or t2, desc if desc is not None else d2)
        else:
            by_name[name] = (typ, desc)
    return [(n, t, d) for n, (t, d) in by_name.items()]


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
    raise TypeError(
        "module must be a string, a dspy.Module subclass, or an instance.")


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
            del self._turns[0: len(self._turns) - self.window]

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
    def __init__(self, *, program: "FunctAIFunc", Sig: type[Signature], inputs: Dict[str, Any], adapter: Optional[dspy.Adapter], main_output_name: Optional[str] = None):
        self.program = program
        self.Sig = Sig
        self.inputs = inputs
        self.adapter = adapter
        self.main_output_name = main_output_name or MAIN_OUTPUT_DEFAULT_NAME

        self._materialized = False
        self._pred: Optional[Prediction] = None
        self._value: Any = None
        self._ai_requested = False  # whether `_ai` has been accessed
        self.collect_only: bool = False  # pre-scan to collect requested outputs
        # name -> (type, desc)
        self._requested_outputs: Dict[str, Tuple[Any, str]] = {}

    def request_ai(self):
        """Mark that `_ai` is being used and return self for Proxy access."""
        self._ai_requested = True
        return self

    # Dynamic outputs requested via _ai["..."]
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
            raise RuntimeError(
                "_ai value accessed before model run; declare outputs with _ai[""desc""] and return _ai.")
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
        in_kwargs = {k: self._to_text(v) for k, v in self.inputs.items() if k in (
            mod.signature.input_fields or {})}

        with _patched_adapter(self.adapter), _patched_lm(getattr(mod, "lm", None)):
            self._pred = mod(**in_kwargs)

        # Extract main output (typed by DSPy via Signature annotations)
        self._value = dict(self._pred).get(self.main_output_name)
        self._materialized = True

        # Record state
        if self.program.state.enabled:
            self.program.state.add_turn(
                inputs=self.inputs, outputs=self._value)

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


# Active call ctx (per-thread/async)
_ACTIVE_CALL: ContextVar[Optional[_CallContext]] = ContextVar(
    "functai_active_call", default=None)


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
            raise RuntimeError(
                "`_ai` can only be used inside an @ai-decorated function call.")
        val = ctx.request_ai().value
        return getattr(val, name)

    # Conversions & operators for ergonomic post-processing
    def _val(self):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError(
                "`_ai` can only be used inside an @ai-decorated function call.")
        return ctx.request_ai().value

    def __str__(self): return str(self._val())
    def __int__(self): return int(self._val())
    def __float__(self): return float(self._val())
    def __bool__(self): return bool(self._val())
    def __len__(self): return len(self._val())
    def __iter__(self): return iter(self._val())
    def __getitem__(self, k): return self._val()[k]
    def __contains__(self, k): return k in self._val()

    # Indexing to declare and/or access additional outputs
    def __getitem__(self, spec):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError(
                "`_ai[...]` can only be used inside an @ai-decorated function call.")
        ctx.request_ai()
        # Accept strings for desc; derive name by sanitizing leading token
        if isinstance(spec, str):
            name = _derive_output_name(spec)
            desc = spec
            ctx.declare_output(name=name, typ=str, desc=desc)
            return _AIFieldProxy(ctx, name=name, typ=str)
        # Tuple: (name, desc[, type])
        if isinstance(spec, tuple) and len(spec) >= 2:
            name = str(spec[0])
            desc = str(spec[1])
            typ = spec[2] if len(spec) >= 3 else str
            ctx.declare_output(name=name, typ=typ, desc=desc)
            return _AIFieldProxy(ctx, name=name, typ=typ)
        raise TypeError(
            "_ai[...] expects a string description or (name, desc[, type]) tuple.")


class _AIFieldProxy:
    """A proxy for a specific output field declared via _ai[...].

    During pre-scan (collect_only), any attempt to use this value will raise.
    During the actual run, it resolves to the model-produced output field.
    """

    def __init__(self, ctx: _CallContext, *, name: str, typ: Any = str):
        self._ctx = ctx
        self._name = name
        self._typ = typ or str

    def _resolve(self):
        if self._ctx.collect_only:
            raise RuntimeError(
                f"Output '{self._name}' value is not available during signature collection.")
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


def _derive_output_name(desc: str) -> str:
    """Derive a field name from a human description like 'think!'.

    Keeps alphanumerics and underscores, lowercases, takes the first token.
    """
    s = ''.join(ch if (ch.isalnum() or ch == '_') else ' ' for ch in str(desc))
    s = s.strip().lower()
    if not s:
        return "field"
    # first token before whitespace
    return s.split()[0]

    # Arithmetic-ish
    def __add__(self, other): return self._val() + other
    def __radd__(self, other): return other + self._val()
    def __sub__(self, other): return self._val() - other
    def __rsub__(self, other): return other - self._val()
    def __mul__(self, other): return self._val() * other
    def __rmul__(self, other): return other * self._val()
    def __truediv__(self, other): return self._val() / other
    def __rtruediv__(self, other): return other / self._val()
    def __eq__(self, other): return self._val() == other
    def __ne__(self, other): return self._val() != other
    def __lt__(self, other): return self._val() < other
    def __le__(self, other): return self._val() <= other
    def __gt__(self, other): return self._val() > other
    def __ge__(self, other): return self._val() >= other


# Public sentinel instance
_ai = _AISentinel()

# Public settings handle for global defaults
settings = _GLOBAL_DEFAULTS


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
        self._module_kind = _select_module_kind(
            module if module is not None else defs.module, tools)
        self._tools: List[Any] = list(tools or [])
        self.temperature: Optional[float] = (
            float(temperature) if temperature is not None else defs.temperature)
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
    def lm(self, v): self._lm = v; self._lm_instance = self._to_lm(
        v); self._compiled = None

    @property
    def adapter(self): return self._adapter

    @adapter.setter
    def adapter(self, v): self._adapter = v; self._adapter_instance = _select_adapter(
        v); self._compiled = None

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
        bound = self._sig.bind_partial(
            *args, **{k: v for k, v in kwargs.items() if k != "_prediction"})
        bound.apply_defaults()
        inputs = {k: v for k, v in bound.arguments.items()
                  if k in self._sig.parameters}

        # Compose dynamic system doc (docstring + persona + state)
        state_block = self.state.render_block()
        sysdoc = _compose_system_doc(self._fn, self.persona, state_block, include_fn_name=bool(
            _effective_defaults().include_fn_name_in_instructions))

        # Phase 1: pre-scan to collect dynamic outputs via _ai["..."] without running the model
        pre_sig = _mk_signature(
            self._fn.__name__, self._fn, doc=sysdoc, return_type=self._return_type)
        pre_ctx = _CallContext(program=self, Sig=pre_sig,
                               inputs=inputs, adapter=self._adapter_instance)
        pre_ctx.collect_only = True
        token1 = _ACTIVE_CALL.set(pre_ctx)
        try:
            prelim = self._fn(*args, **kwargs)
        except RuntimeError as e:
            # If user tried to use _ai value during pre-scan, fall back to single-pass behavior
            _ACTIVE_CALL.reset(token1)
            return self._single_pass_call(inputs=inputs, sysdoc=sysdoc, args=args, kwargs=kwargs, _prediction=_prediction)
        finally:
            with contextlib.suppress(Exception):
                _ACTIVE_CALL.reset(token1)

        # If function didn't use _ai at all, just return their value unless they returned the sentinel/ellipsis
        # or it's an "empty-style" function returning None/ellipsis/implicit None → trigger model run
        needs_model = (
            pre_ctx._ai_requested
            or (prelim is _ai)
            or isinstance(prelim, _AIFieldProxy)
            or (prelim is Ellipsis)
            or (prelim is None)
        )
        if not needs_model and not pre_ctx._requested_outputs:
            return prelim

        # Build final signature including any requested outputs (e.g., think)
        dyn_extras = pre_ctx.requested_outputs()
        # Also include outputs declared via assignments (think: str = _ai, sentiment = _ai)
        ast_outputs = _collect_ast_outputs(self._fn)

        # Determine main output name
        ret_info = _collect_return_info(self._fn)
        order_names = [n for n, _, _ in ast_outputs]
        main_name: str
        if ret_info.get("mode") == "name" and ret_info.get("name") in order_names:
            main_name = typing.cast(str, ret_info.get("name"))
        elif ret_info.get("mode") in {"sentinel", "ellipsis", "empty"}:
            main_name = MAIN_OUTPUT_DEFAULT_NAME
        elif order_names:
            main_name = order_names[-1]
        else:
            main_name = MAIN_OUTPUT_DEFAULT_NAME

        # Validate/resolve types and merge extras
        # Convert to dict for easy lookup
        ast_map: Dict[str, Tuple[Any, str]] = {
            n: (t, d) for n, t, d in ast_outputs}
        # Main output type: must match function return type if annotated on var
        if main_name in ast_map:
            t0, d0 = ast_map[main_name]
            if t0 is not None and t0 != self._return_type:
                raise TypeError(
                    f"Type of '{main_name}' ({t0}) conflicts with function return type ({self._return_type}).")
            main_typ = self._return_type
            main_desc = d0
        else:
            main_typ = self._return_type
            main_desc = ""

        # Extras: all AST outputs except the main
        extras: List[Tuple[str, Any, str]] = [(n, (ast_map[n][0] if ast_map[n][0] is not None else str), ast_map[n][1])
                                              for n in order_names if n != main_name]
        # Merge in dynamic extras (from _ai[...])
        if dyn_extras:
            extras = _merge_outputs(primary=extras, secondary=dyn_extras)

        # Remove any accidental duplication of the main name
        extras = [(n, t, d) for (n, t, d) in extras if n != main_name]

        Sig = _mk_signature(
            self._fn.__name__,
            self._fn,
            doc=sysdoc,
            return_type=self._return_type,
            extra_outputs=extras,
            main_output=(main_name, main_typ, main_desc),
        )

        # Build real call context
        ctx = _CallContext(program=self, Sig=Sig, inputs=inputs,
                           adapter=self._adapter_instance, main_output_name=main_name)

        # Materialize prediction up-front so proxies can resolve during re-run
        ctx.ensure_materialized()

        # Phase 2: run function again to allow access to declared outputs via proxies
        token2 = _ACTIVE_CALL.set(ctx)
        try:
            result = self._fn(*args, **kwargs)
            if _prediction:
                return ctx._pred
            if result is _ai or result is Ellipsis:
                return ctx.request_ai().value
            if result is None and not ctx._ai_requested:
                return ctx.request_ai().value
            return result
        finally:
            _ACTIVE_CALL.reset(token2)

    # Backwards-compatible path: original single-pass call (no dynamic outputs)
    def _single_pass_call(self, *, inputs, sysdoc, args, kwargs, _prediction: bool = False):
        # Build signature including AST-declared outputs and main
        ast_outputs = _collect_ast_outputs(self._fn)
        ret_info = _collect_return_info(self._fn)
        order_names = [n for n, _, _ in ast_outputs]
        if ret_info.get("mode") == "name" and ret_info.get("name") in order_names:
            main_name = typing.cast(str, ret_info.get("name"))
        elif ret_info.get("mode") in {"sentinel", "ellipsis", "empty"}:
            main_name = MAIN_OUTPUT_DEFAULT_NAME
        elif order_names:
            main_name = order_names[-1]
        else:
            main_name = MAIN_OUTPUT_DEFAULT_NAME
        ast_map: Dict[str, Tuple[Any, str]] = {
            n: (t, d) for n, t, d in ast_outputs}
        if main_name in ast_map:
            t0, d0 = ast_map[main_name]
            if t0 is not None and t0 != self._return_type:
                raise TypeError(
                    f"Type of '{main_name}' ({t0}) conflicts with function return type ({self._return_type}).")
            main_desc = d0
        else:
            main_desc = ""
        extras = [(n, (ast_map[n][0] if ast_map[n][0] is not None else str),
                   ast_map[n][1]) for n in order_names if n != main_name]
        Sig = _mk_signature(self._fn.__name__, self._fn, doc=sysdoc, return_type=self._return_type,
                            extra_outputs=extras, main_output=(main_name, self._return_type, main_desc))
        ctx = _CallContext(program=self, Sig=Sig, inputs=inputs,
                           adapter=self._adapter_instance, main_output_name=main_name)
        token = _ACTIVE_CALL.set(ctx)
        try:
            result = self._fn(*args, **kwargs)
            if _prediction:
                _ = ctx.request_ai().value
                return ctx._pred
            if result is _ai or result is Ellipsis:
                return ctx.request_ai().value
            if result is None and not ctx._ai_requested:
                return ctx.request_ai().value
            return result
        finally:
            _ACTIVE_CALL.reset(token)

    # ----- in-place optimize -----
    def opt(self, *, trainset: Optional[List[Any]] = None, strategy: str = "launch", **opts) -> None:
        """Compile the program with a DSPy teleprompter and mutate in place."""
        # Build a temporary module using current signature to compile against
        Sig = _mk_signature(
            self._fn.__name__,
            self._fn,
            doc=_compose_system_doc(
                self._fn,
                self.persona,
                self.state.render_block(),
                include_fn_name=bool(
                    _effective_defaults().include_fn_name_in_instructions),
            ),
            return_type=self._return_type,
        )
        base_mod = _instantiate_module(
            self._module_kind, Sig, tools=self._tools, module_kwargs=self._module_kwargs)

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
        raise TypeError(
            "format_prompt(...) expects an @ai-decorated function/program.")

    # Compose signature
    sysdoc = _compose_system_doc(prog._fn, prog.persona, prog.state.render_block(
    ), include_fn_name=bool(_effective_defaults().include_fn_name_in_instructions))
    # Include AST-declared outputs and main in preview
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
    ast_map: Dict[str, Tuple[Any, str]] = {
        n: (t, d) for n, t, d in ast_outputs}
    if main_name in ast_map:
        main_desc = ast_map[main_name][1]
    else:
        main_desc = ""
    extras = [(n, (ast_map[n][0] if ast_map[n][0] is not None else str),
               ast_map[n][1]) for n in order_names if n != main_name]
    Sig = _mk_signature(prog._fn.__name__, prog._fn, doc=sysdoc, return_type=prog._return_type,
                        extra_outputs=extras, main_output=(main_name, prog._return_type, main_desc))
    adapter_inst = prog._adapter_instance or getattr(
        dspy.settings, "adapter", None) or dspy.ChatAdapter()

    # Normalize inputs
    in_text = {k: str(v) for k, v in inputs.items()
               if k in (Sig.input_fields or {})}

    # Try adapter-native formatting
    system_content = (getattr(Sig, "__doc__", "") or "").strip()
    user_content = None
    try:
        if hasattr(adapter_inst, "format_user_message_content"):
            user_content = adapter_inst.format_user_message_content(
                Sig, in_text)
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
    render_lines.append(
        f"Module: {prog._module_kind.__class__.__name__ if isinstance(prog._module_kind, dspy.Module) else str(prog._module_kind)}")
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
    "settings",
]
