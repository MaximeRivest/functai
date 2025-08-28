# funnydspy.py
# -----------------------------------------------------------------------------
# FunnyDSPy — DSPy-first, single-call LLM functions with step()/final() markers
#
# NEW: Adapter support
#   - @magic(..., adapter=..., adapter_kwargs=...) to set a DSPy adapter per function.
#   - Works with strings ("json", "chat", "two_step"), classes, or instances.
#   - Per-call patch of dspy.settings.adapter so all modules honor it (Predict/COT/ReAct/etc.).
#   - parallel(...) wraps calls so each item uses the right adapter without global races.
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import json
import textwrap
import typing
from dataclasses import is_dataclass, fields as dataclass_fields
from typing import Any, Dict, List, Optional, Tuple
from contextvars import ContextVar
import contextlib
from types import SimpleNamespace

import dspy
from dspy import Signature, InputField, OutputField, Prediction


# ──────────────────────────────────────────────────────────────────────────────
# Runtime call-context: step()/final() return lazy proxies tied to the active call
# -----------------------------------------------------------------------------

_current_ctx: ContextVar["CallContext|None"] = ContextVar("funnydspy_ctx", default=None)


def step(*args, desc: str = "") -> Any:
    """Declare and request the next STEP output (lazy proxy).

    Accepts `step("desc")` or `step(desc="desc")`.
    """
    ctx = _current_ctx.get()
    if ctx is None:
        raise RuntimeError("step() can only be used inside a @magic function call.")
    return ctx.request_proxy(kind="step")


def final(*args, desc: str = "") -> Any:
    """Declare and request the (single) FINAL output (lazy proxy).

    Accepts `final("desc")` or `final(desc="desc")`.
    """
    ctx = _current_ctx.get()
    if ctx is None:
        raise RuntimeError("final() can only be used inside a @magic function call.")
    return ctx.request_proxy(kind="final")


# ──────────────────────────────────────────────────────────────────────────────
# Spec and parsing: collect markers & types at DECORATION time (no execution)
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class OutputDef:
    var_name: str  # Python variable name as written by user
    field_name: str  # DSPy output field base name (sanitized)
    typ: Any
    is_final: bool
    desc: str
    dspy_fields: List[str]  # DSPy output fields (sanitized)
    kind: str  # "simple" | "dataclass" | "namedtuple"
    type_name: Optional[str] = None
    field_types: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ParsedSpec:
    mode: str  # "minimal" or "markers"
    inputs: Dict[str, Any]
    outputs: List[OutputDef]
    return_ann: Any
    final_var: Optional[str]
    doc: str


def _eval_type_expr(expr: ast.expr, g: Dict[str, Any]) -> Any:
    code = ast.unparse(expr) if hasattr(ast, "unparse") else None
    env = {"typing": typing, "__builtins__": {}}
    env.update(g)
    try:
        return eval(code, env, {})
    except Exception:
        return typing.Any


def _is_namedtuple_type(tp: Any) -> bool:
    try:
        return isinstance(tp, type) and issubclass(tp, tuple) and hasattr(tp, "_fields")
    except Exception:
        return False


def _expand_struct_for_var(var: str, typ: Any) -> Tuple[str, Dict[str, Any], List[str], str]:
    tname = getattr(typ, "__name__", "Struct")
    prefix = f"{var}__{tname}"
    if is_dataclass(typ):
        ftypes = {f.name: f.type for f in dataclass_fields(typ)}
        fields = [f"{prefix}_{fname}" for fname in ftypes]
        return tname, ftypes, fields, "dataclass"
    if _is_namedtuple_type(typ):
        ann = getattr(typ, "__annotations__", {})
        ftypes = {fname: ann.get(fname, str) for fname in typ._fields}
        fields = [f"{prefix}_{fname}" for fname in ftypes]
        return tname, ftypes, fields, "namedtuple"
    return "", {}, [var], "simple"


def _parse_markers(fn) -> ParsedSpec:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)

    inputs: Dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name == "_prediction":
            continue
        ann = hints.get(name, p.annotation if p.annotation is not inspect.Parameter.empty else str)
        inputs[name] = ann

    ret_ann = hints.get("return", sig.return_annotation if sig.return_annotation is not inspect._empty else typing.Any)

    def _extract_desc_from_call(call: ast.Call) -> str:
        desc_s = ""
        # Keyword form: final(desc="...") / step(desc="...")
        for kw in call.keywords or []:
            if isinstance(kw, ast.keyword) and kw.arg == "desc" and isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    desc_s = kw.value.value
        # Positional form: final("...") / step("...")
        if not desc_s and getattr(call, "args", None):
            first = call.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                desc_s = first.value
        return desc_s

    try:
        src = textwrap.dedent(inspect.getsource(fn))
        mod = ast.parse(src)
    except Exception:
        return ParsedSpec(
            mode="minimal", inputs=inputs, outputs=[], return_ann=ret_ann, final_var="result", doc=(fn.__doc__ or "")
        )

    fdef = None
    for node in mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fdef = node
            break
    if fdef is None:
        return ParsedSpec(
            mode="minimal", inputs=inputs, outputs=[], return_ann=ret_ann, final_var="result", doc=(fn.__doc__ or "")
        )

    g = getattr(fn, "__globals__", {})
    outputs: List[OutputDef] = []

    for node in fdef.body:
        # Pattern 1: Annotated assignment: var: Type = step(...)/final(...)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.value, ast.Call):
            tgt = node.target.id
            callee = node.value.func
            if isinstance(callee, ast.Name) and callee.id in ("step", "final"):
                vtype = _eval_type_expr(node.annotation, g) if node.annotation is not None else typing.Any
                desc = _extract_desc_from_call(node.value)
                # Sanitize field base name for DSPy (strip leading underscores to appease linters)
                field_base = tgt.lstrip('_') or tgt
                tname, ftypes, dspy_fields, kind = _expand_struct_for_var(field_base, vtype)
                outputs.append(
                    OutputDef(
                        var_name=tgt,
                        field_name=field_base,
                        typ=vtype,
                        is_final=(callee.id == "final"),
                        desc=desc,
                        dspy_fields=dspy_fields,
                        kind=kind,
                        type_name=(tname or None),
                        field_types=(ftypes or None),
                    )
                )
            continue
        # Pattern 2: Unannotated assignment: var = step(...)/final(...)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
            tgt = node.targets[0].id
            callee = node.value.func
            if isinstance(callee, ast.Name) and callee.id in ("step", "final"):
                vtype = typing.Any
                desc = _extract_desc_from_call(node.value)
                field_base = tgt.lstrip('_') or tgt
                tname, ftypes, dspy_fields, kind = _expand_struct_for_var(field_base, vtype)
                outputs.append(
                    OutputDef(
                        var_name=tgt,
                        field_name=field_base,
                        typ=vtype,
                        is_final=(callee.id == "final"),
                        desc=desc,
                        dspy_fields=dspy_fields,
                        kind=kind,
                        type_name=(tname or None),
                        field_types=(ftypes or None),
                    )
                )

    explicit_finals = [o for o in outputs if o.is_final]
    ret_final: Optional[str] = None
    if not explicit_finals:
        # Case A: return <name> where <name> is a declared output
        for node in ast.walk(fdef):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                if any(o.var_name == node.value.id for o in outputs):
                    ret_final = node.value.id
                    break
        # Case B: return final(...) — synthesize a default final output named 'result'
        if ret_final is None:
            final_calls = []
            for node in ast.walk(fdef):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                    callee = node.value.func
                    if isinstance(callee, ast.Name) and callee.id == "final":
                        final_calls.append(node)
            if len(final_calls) > 1:
                raise ValueError(f"{fn.__name__}: multiple final() markers found; only one is allowed.")
            if len(final_calls) == 1:
                fnode = final_calls[0]
                desc = _extract_desc_from_call(fnode.value)
                vtype = ret_ann
                field_base = "result"
                tname, ftypes, dspy_fields, kind = _expand_struct_for_var(field_base, vtype)
                outputs.append(
                    OutputDef(
                        var_name=field_base,
                        field_name=field_base,
                        typ=vtype,
                        is_final=True,
                        desc=desc,
                        dspy_fields=dspy_fields,
                        kind=kind,
                        type_name=(tname or None),
                        field_types=(ftypes or None),
                    )
                )
                ret_final = field_base
    else:
        if len(explicit_finals) > 1:
            raise ValueError(f"{fn.__name__}: multiple final() markers found; only one is allowed.")
        ret_final = explicit_finals[0].var_name

    mode = "markers" if outputs else "minimal"
    return ParsedSpec(
        mode=mode,
        inputs=inputs,
        outputs=outputs,
        return_ann=ret_ann,
        final_var=(ret_final if mode == "markers" else "result"),
        doc=(fn.__doc__ or ""),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Signature builder
# -----------------------------------------------------------------------------


def _mk_signature(fn_name: str, spec: ParsedSpec, *, system: Optional[str], name: Optional[str], module_kind: str):
    class_dict: Dict[str, Any] = {}
    ann_map: Dict[str, Any] = {}

    for in_name, in_type in spec.inputs.items():
        class_dict[in_name] = InputField()
        ann_map[in_name] = in_type

    if spec.mode == "minimal":
        class_dict["result"] = OutputField(desc="")
        ann_map["result"] = spec.return_ann
    else:
        for o in spec.outputs:
            if o.kind == "simple":
                class_dict[o.field_name] = OutputField(desc=o.desc or "")
                ann_map[o.field_name] = o.typ
            else:
                for full in o.dspy_fields:
                    base_field = full.split("_")[-1]
                    ftyp = (o.field_types or {}).get(base_field, typing.Any)
                    class_dict[full] = OutputField(desc=o.desc or "")
                    ann_map[full] = ftyp

    doc = (system or spec.doc or "").strip()
    if doc:
        class_dict["__doc__"] = doc
    class_dict["__annotations__"] = ann_map
    Sig = type(name or f"{fn_name.title()}Sig", (Signature,), class_dict)
    return Sig


# ──────────────────────────────────────────────────────────────────────────────
# Materialization helpers (text → typed)
# -----------------------------------------------------------------------------


def _from_text(txt: Any, typ: Any) -> Any:
    if not isinstance(txt, str):
        return txt
    try:
        origin = typing.get_origin(typ)
        args = typing.get_args(typ)
        if typ in (str, typing.Any, Any):
            return txt
        if typ is float:
            return float(txt)
        if typ is int:
            s = txt.strip()
            return int(float(s)) if "." in s else int(s)
        if typ is bool:
            return txt.strip().lower() in ("true", "1", "yes", "y")
        if origin is list and args:
            inner = args[0]
            s = txt.strip()
            data = (
                json.loads(s)
                if (s.startswith("[") and s.endswith("]"))
                else [x.strip() for x in txt.split(",") if x.strip()]
            )
            return [_from_text(x if isinstance(x, str) else json.dumps(x), inner) for x in data]
        if origin is dict and args:
            kt, vt = args
            s = txt.strip()
            data = json.loads(s) if (s.startswith("{") and s.endswith("}")) else {}
            return {
                _from_text(str(k), kt): _from_text(v if isinstance(v, str) else json.dumps(v), vt)
                for k, v in data.items()
            }
        if is_dataclass(typ):
            data = json.loads(txt) if txt.strip().startswith("{") else {}
            vals = {}
            for f in dataclass_fields(typ):
                vals[f.name] = _from_text(data.get(f.name), f.type) if f.name in data else None
            return typ(**vals)
        if _is_namedtuple_type(typ):
            ann = getattr(typ, "__annotations__", {})
            data = json.loads(txt) if txt.strip().startswith("{") else {}
            return typ(**{k: _from_text(data.get(k), ann.get(k, str)) for k in typ._fields})
    except Exception:
        return txt
    return txt


# ──────────────────────────────────────────────────────────────────────────────
# Adapter helpers  (NEW)
# -----------------------------------------------------------------------------


def _select_adapter(adapter: Any, adapter_kwargs: Optional[Dict[str, Any]]) -> Optional[dspy.Adapter]:
    if adapter is None:
        return None
    if isinstance(adapter, str):
        key = adapter.lower().replace("-", "_")
        if key in ("json", "jsonadapter"):
            return dspy.JSONAdapter()
        if key in ("chat", "chatadapter"):
            return dspy.ChatAdapter()
        if key in ("two_step", "twostep", "two_step_adapter", "twostepadapter"):
            kwargs = adapter_kwargs or {}
            if "extraction_model" not in kwargs:
                raise ValueError("TwoStepAdapter requires `extraction_model=LM(...)` in adapter_kwargs.")
            return dspy.TwoStepAdapter(**kwargs)
        raise ValueError(f"Unknown adapter string '{adapter}'.")
    if isinstance(adapter, type) and issubclass(adapter, dspy.Adapter):
        return adapter(**(adapter_kwargs or {}))
    if isinstance(adapter, dspy.Adapter):
        return adapter
    raise TypeError("adapter must be a string, a dspy.Adapter subclass, or a dspy.Adapter instance.")


@contextlib.contextmanager
def _patched_adapter(adapter_instance: Optional[dspy.Adapter]):
    """Temporarily patch dspy.settings.adapter for the duration of one call."""
    if adapter_instance is None:
        yield
        return
    prev = getattr(dspy.settings, "adapter", None)
    try:
        dspy.settings.adapter = adapter_instance
        yield
    finally:
        dspy.settings.adapter = prev


@contextlib.contextmanager
def _patched_lm(lm_instance: Optional[Any]):
    """Temporarily patch dspy.settings.lm for the duration of one call.

    This ensures nested modules (e.g., ReAct's internal Predict) see an LM
    even if their own `.lm` isn't set.
    """
    if lm_instance is None:
        yield
        return
    prev = getattr(dspy.settings, "lm", None)
    try:
        dspy.settings.lm = lm_instance
        yield
    finally:
        dspy.settings.lm = prev


# ──────────────────────────────────────────────────────────────────────────────
# Call context and lazy proxies
# -----------------------------------------------------------------------------


class CallContext:
    def __init__(
        self, *, module: dspy.Module, spec: ParsedSpec, inputs: Dict[str, Any], adapter: Optional[dspy.Adapter]
    ):
        self.module = module
        self.spec = spec
        self.inputs = inputs
        self.adapter = adapter
        self._materialized = False
        self._pred: Optional[Prediction] = None
        self._cache: Dict[str, Any] = {}
        self._assigned_step = 0
        self._assigned_final = False

        self._steps = [o for o in spec.outputs if not o.is_final]
        self._final = next((o for o in spec.outputs if o.is_final), None)

    def request_proxy(self, *, kind: str):
        if self.spec.mode != "markers":
            raise RuntimeError("step()/final() used but no markers were detected at decoration time.")
        if kind == "final":
            if self._final is None:
                if self.spec.final_var:
                    ov = next((o for o in self.spec.outputs if o.var_name == self.spec.final_var), None)
                else:
                    ov = None
                if ov is None:
                    raise RuntimeError("final() used but no final output is declared or inferred.")
                self._final = ov
            if self._assigned_final:
                return _Proxy(self, self._final)
            self._assigned_final = True
            return _Proxy(self, self._final)

        if self._assigned_step >= len(self._steps):
            if self._final is not None and not self._assigned_final:
                self._assigned_final = True
                return _Proxy(self, self._final)
            raise RuntimeError("More step() calls at runtime than declared markers in the function.")
        ov = self._steps[self._assigned_step]
        self._assigned_step += 1
        return _Proxy(self, ov)

    def ensure_materialized(self):
        if self._materialized:
            return
        in_kwargs = {k: self._to_text(v) for k, v in self.inputs.items()}
        # Ensure the chosen adapter is used even if the module looks at dspy.settings.adapter.
        with _patched_adapter(self.adapter), _patched_lm(getattr(self.module, "lm", None)):
            self._pred = self.module(**in_kwargs)
        pm = dict(self._pred)

        for o in self.spec.outputs:
            if o.kind == "simple":
                raw = pm.get(o.field_name)
                # Fallback: some modules (e.g., ReAct) return only 'result' as the final field.
                if raw is None and o.is_final and "result" in pm:
                    raw = pm.get("result")
                self._cache[o.var_name] = _from_text(raw, o.typ)
            else:
                data = {}
                for full in o.dspy_fields:
                    base_field = full.split("_")[-1]
                    raw = pm.get(full)
                    ftyp = (o.field_types or {}).get(base_field, typing.Any)
                    data[base_field] = _from_text(raw, ftyp)
                if o.kind == "dataclass":
                    cls = o.typ
                    self._cache[o.var_name] = cls(**data)
                else:
                    cls = o.typ
                    ordered = [data[k] for k in cls._fields]
                    self._cache[o.var_name] = cls(*ordered)

        self._materialized = True

    def get_value(self, var_name: str) -> Any:
        self.ensure_materialized()
        if var_name not in self._cache:
            raise KeyError(f"Unknown output variable '{var_name}' — was it declared with step()/final()? ")
        return self._cache[var_name]

    def get_prediction(self) -> Prediction:
        self.ensure_materialized()
        assert self._pred is not None
        return self._pred

    @staticmethod
    def _to_text(v: Any) -> Any:
        if isinstance(v, list):
            return [CallContext._to_text(x) for x in v]
        if isinstance(v, (str, dict)):
            return v
        return str(v)


class _Proxy:
    """Lazy proxy for a step()/final() output. Materializes all model outputs on first use."""

    __slots__ = ("_ctx", "_odef")

    def __init__(self, ctx: CallContext, odef: OutputDef):
        self._ctx = ctx
        self._odef = odef

    @property
    def value(self):
        return self._ctx.get_value(self._odef.var_name)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        if not self._ctx._materialized:
            return f"<{self._odef.var_name}:lazy>"
        return repr(self.value)

    def __str__(self):
        v = self.value
        return v if isinstance(v, str) else str(v)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, k):
        return self.value[k]

    def __contains__(self, k):
        return k in self.value

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other


# ──────────────────────────────────────────────────────────────────────────────
# Module selection / building
# -----------------------------------------------------------------------------


def _select_module(module):
    """Normalize the `module` arg.

    Returns either a dspy.Module subclass (class) or an instance.
    Do not instantiate here to allow passing `module_kwargs` later.
    """
    if isinstance(module, str):
        m = module.lower()
        if m in ("predict", "p"):
            return dspy.Predict
        if m in ("chainofthought", "cot"):
            return dspy.ChainOfThought
        if m in ("react", "ra"):
            return dspy.ReAct
        raise ValueError(f"Unknown module string '{module}'.")
    if isinstance(module, type) and issubclass(module, dspy.Module):
        return module
    if isinstance(module, dspy.Module):
        return module
    raise TypeError("module must be a string, a dspy.Module subclass, or a dspy.Module instance.")


# ──────────────────────────────────────────────────────────────────────────────
# The decorator: @magic
# -----------------------------------------------------------------------------


def magic(
    _fn=None,
    *,
    module: Any = "predict",
    system: Optional[str] = None,
    lm: Any = None,
    gen: Optional[Dict[str, Any]] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    # NEW: per-function adapter
    adapter: Any = None,
    adapter_kwargs: Optional[Dict[str, Any]] = None,
    # Convenience: allow passing `tools=[...]` directly for modules like ReAct
    tools: Optional[List[Any]] = None,
):
    """
    Turn a typed function with body markers into a single-call DSPy program.

    `_prediction=True` → returns raw `dspy.Prediction`.

    NEW:
      - `adapter`: string | dspy.Adapter subclass | instance (e.g., "json", dspy.JSONAdapter()).
      - `adapter_kwargs`: passed if `adapter` is a subclass or "two_step".
    """

    def _decorate(fn):
        # Merge convenience kwargs into module_kwargs
        mk = dict(module_kwargs or {})
        if tools is not None:
            mk.setdefault("tools", tools)

        spec = _parse_markers(fn)
        Sig = _mk_signature(fn.__name__, spec, system=system, name=name, module_kind=str(module))

        mod = _select_module(module)
        # If a class was provided/selected, instantiate it with our signature and module kwargs
        if isinstance(mod, type) and issubclass(mod, dspy.Module):
            try:
                mod = mod(Sig, **mk)
            except TypeError as e:
                raise TypeError(
                    f"Failed to instantiate module {mod.__name__}: {e}. "
                    f"Pass required args via module_kwargs, e.g., module_kwargs={{'tools': [...]}}."
                ) from e
        else:
            # Instance provided; try to set or rebuild with signature
            if getattr(mod, "signature", None) is None:
                try:
                    MCls = type(mod)
                    mod = MCls(Sig, **mk)
                except Exception:
                    try:
                        mod.signature = Sig
                    except Exception:
                        pass
            else:
                try:
                    mod.signature = Sig
                except Exception:
                    pass

        if lm is not None:
            try:
                # If a string is provided, let DSPy resolve the provider via dspy.LM(thestr)
                if isinstance(lm, str):
                    mod.lm = dspy.LM(lm)
                else:
                    mod.lm = lm
            except Exception:
                pass
        if gen:
            for k, v in gen.items():
                try:
                    setattr(mod, k, v)
                except Exception:
                    pass

        # Build adapter instance once at decoration
        adapter_inst = _select_adapter(adapter, adapter_kwargs)

        @functools.wraps(fn)
        def wrapper(*args, _prediction: bool = False, **kwargs):
            bound = inspect.signature(fn).bind_partial(*args, **{k: v for k, v in kwargs.items() if k != "_prediction"})
            bound.apply_defaults()
            in_kwargs = {k: v for k, v in bound.arguments.items() if k in spec.inputs}

            if spec.mode == "minimal":
                with _patched_adapter(adapter_inst), _patched_lm(getattr(mod, "lm", None)):
                    pred: Prediction = mod(**{k: CallContext._to_text(v) for k, v in in_kwargs.items()})
                if _prediction:
                    return pred
                raw = dict(pred).get("result")
                return _from_text(raw, spec.return_ann)

            ctx = CallContext(module=mod, spec=spec, inputs=in_kwargs, adapter=adapter_inst)
            token = _current_ctx.set(ctx)
            try:
                result = fn(*args, **kwargs)
                if _prediction:
                    return ctx.get_prediction()
                return result
            finally:
                _current_ctx.reset(token)

        # Expose DSPy internals + adapter
        wrapper.__dspy__ = SimpleNamespace(  # type: ignore[attr-defined]
            signature=Sig, module=mod, spec=spec, adapter=adapter_inst
        )

        _sig = inspect.signature(fn)
        params = list(_sig.parameters.values())
        if "_prediction" not in _sig.parameters:
            params.append(
                inspect.Parameter(
                    name="_prediction",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=False,
                    annotation=bool,
                )
            )
        wrapper.__signature__ = _sig.replace(parameters=params, return_annotation=spec.return_ann)
        return wrapper

    if _fn is not None:
        return _decorate(_fn)
    return _decorate


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer / Parallel integration
# -----------------------------------------------------------------------------


def optimize(fn_or_wrapper, *, optimizer: Optional[Any] = None, trainset: Optional[List[Any]] = None):
    """
    Compile the underlying DSPy module with a teleprompter/optimizer.
    Preserves the configured adapter.
    """
    if not hasattr(fn_or_wrapper, "__dspy__"):
        raise TypeError("optimize(...) expects a @magic-decorated function.")

    spec: ParsedSpec = fn_or_wrapper.__dspy__.spec
    mod: dspy.Module = fn_or_wrapper.__dspy__.module
    adapter_inst: Optional[dspy.Adapter] = getattr(fn_or_wrapper.__dspy__, "adapter", None)

    if optimizer is None:
        optimizer = dspy.BootstrapFewShot()
    compiled = optimizer.compile(mod, trainset=trainset)

    @functools.wraps(fn_or_wrapper)
    def wrapped(*args, _prediction: bool = False, **kwargs):
        if spec.mode == "minimal":
            bound = inspect.signature(fn_or_wrapper).bind_partial(
                *args, **{k: v for k, v in kwargs.items() if k != "_prediction"}
            )
            bound.apply_defaults()
            in_kwargs = {k: v for k, v in bound.arguments.items() if k in compiled.signature.input_fields}
            with _patched_adapter(adapter_inst), _patched_lm(getattr(compiled, "lm", None)):
                pred: Prediction = compiled(**{k: CallContext._to_text(v) for k, v in in_kwargs.items()})
            if _prediction:
                return pred
            raw = dict(pred).get("result")
            return _from_text(raw, spec.return_ann)

        bound = inspect.signature(fn_or_wrapper).bind_partial(
            *args, **{k: v for k, v in kwargs.items() if k != "_prediction"}
        )
        bound.apply_defaults()
        in_kwargs = {k: v for k, v in bound.arguments.items() if k in spec.inputs}

        ctx = CallContext(module=compiled, spec=spec, inputs=in_kwargs, adapter=adapter_inst)
        token = _current_ctx.set(ctx)
        try:
            result = fn_or_wrapper(*args, **kwargs)
            if _prediction:
                return ctx.get_prediction()
            return result
        finally:
            _current_ctx.reset(token)

    wrapped.__dspy__ = fn_or_wrapper.__dspy__
    wrapped.__dspy__.module = compiled  # type: ignore[attr-defined]
    return wrapped


class _AdapterBoundCallable:
    """Tiny callable wrapper used in parallel() to ensure per-item adapter usage."""

    def __init__(self, module: Any, adapter_inst: Optional[dspy.Adapter]):
        self._module = module
        self._adapter = adapter_inst
        # propagate signature if present (not strictly required by Parallel)
        self.signature = getattr(module, "signature", None)

    def __call__(self, *args, **kwargs):
        with _patched_adapter(self._adapter), _patched_lm(getattr(self._module, "lm", None)):
            return self._module(*args, **kwargs)


def parallel(fn_or_wrapper, inputs: List[Dict[str, Any]], *, prediction: bool = False):
    """
    Batch execution using dspy.Parallel over the underlying module.

    Returns LLM outputs only (not full Python body). Honors per-function adapter.
    """
    if not hasattr(fn_or_wrapper, "__dspy__"):
        raise TypeError("parallel(...) expects a @magic-decorated function.")
    spec: ParsedSpec = fn_or_wrapper.__dspy__.spec
    mod: dspy.Module = fn_or_wrapper.__dspy__.module
    adapter_inst: Optional[dspy.Adapter] = getattr(fn_or_wrapper.__dspy__, "adapter", None)

    if not inputs:
        return []

    # Wrap module so each call patches settings.adapter for that item.
    wrapped_mod = _AdapterBoundCallable(mod, adapter_inst)
    pairs = [(wrapped_mod, {k: CallContext._to_text(v) for k, v in item.items()}) for item in inputs]
    preds: List[Prediction] = dspy.Parallel().forward(pairs)

    if prediction:
        return preds

    out = []
    if spec.mode == "minimal":
        for p in preds:
            raw = dict(p).get("result")
            out.append(_from_text(raw, spec.return_ann))
        return out

    for p in preds:
        pm = dict(p)
        if spec.final_var:
            odef = next((o for o in spec.outputs if o.var_name == spec.final_var), None)
            if odef:
                if odef.kind == "simple":
                    val = pm.get(odef.field_name)
                    if val is None and "result" in pm:
                        val = pm.get("result")
                    out.append(_from_text(val, odef.typ))
                else:
                    data = {}
                    for full in odef.dspy_fields:
                        base_field = full.split("_")[-1]
                        raw = pm.get(full)
                        ftyp = (odef.field_types or {}).get(base_field, typing.Any)
                        data[base_field] = _from_text(raw, ftyp)
                    if odef.kind == "dataclass":
                        cls = odef.typ
                        out.append(cls(**data))
                    else:
                        cls = odef.typ
                        out.append(cls(*[data[k] for k in cls._fields]))
                continue
        record = {}
        for o in spec.outputs:
            if o.kind == "simple":
                val = pm.get(o.field_name)
                if val is None and o.is_final and "result" in pm:
                    val = pm.get("result")
                record[o.var_name] = _from_text(val, o.typ)
            else:
                data = {}
                for full in o.dspy_fields:
                    base_field = full.split("_")[-1]
                    raw = pm.get(full)
                    ftyp = (o.field_types or {}).get(base_field, typing.Any)
                    data[base_field] = _from_text(raw, ftyp)
                if o.kind == "dataclass":
                    cls = o.typ
                    record[o.var_name] = cls(**data)
                else:
                    cls = o.typ
                    record[o.var_name] = cls(*[data[k] for k in cls._fields])
        out.append(record)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# -----------------------------------------------------------------------------

__all__ = [
    "magic",
    "step",
    "final",
    "optimize",
    "parallel",
    "format_prompt",
    "inspect_history_text",
]


def use(*_args):
    """No-op helper to mark variables as 'used' for linters.

    Example:
        sentiment: str = step(...)
        use(sentiment)
    """
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Prompt formatting and history helpers
# -----------------------------------------------------------------------------

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


def format_prompt(fn_or_wrapper, /, **inputs) -> Dict[str, Any]:
    """Return a best-effort preview of adapter-formatted messages for a call.

    Attempts to use the configured adapter's formatting (e.g., ChatAdapter.format_user_message_content).
    Falls back to a readable text rendering if adapter-specific hooks are not available.
    """
    if not hasattr(fn_or_wrapper, "__dspy__"):
        raise TypeError("format_prompt(...) expects a @magic-decorated function.")

    spec: ParsedSpec = fn_or_wrapper.__dspy__.spec  # type: ignore[attr-defined]
    sig: Signature = fn_or_wrapper.__dspy__.signature  # type: ignore[attr-defined]
    adapter_inst = getattr(fn_or_wrapper.__dspy__, "adapter", None)  # type: ignore[attr-defined]
    adapter_inst = adapter_inst or getattr(dspy.settings, "adapter", None) or dspy.ChatAdapter()

    # Normalize inputs to text the same way the runtime would.
    in_text = {k: CallContext._to_text(v) for k, v in inputs.items() if k in (sig.input_fields or {})}

    # Try adapter-native formatting.
    system_content = (getattr(sig, "__doc__", "") or "").strip()
    user_content = None
    try:
        if hasattr(adapter_inst, "format_user_message_content"):
            user_content = adapter_inst.format_user_message_content(sig, in_text)
    except Exception:
        user_content = None
    if not user_content:
        user_content = _default_user_content(sig, in_text)

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    # Human-friendly rendering
    render_lines = []
    render_lines.append(f"Adapter: {type(adapter_inst).__name__}")
    render_lines.append(f"Module: {type(getattr(fn_or_wrapper.__dspy__, 'module', None)).__name__}")
    if system_content:
        render_lines.append("")
        render_lines.append("System:")
        render_lines.append(system_content)
    render_lines.append("")
    render_lines.append("User:")
    render_lines.append(user_content)

    # Best-effort demo extraction
    def _maybe_to_plain(val):
        try:
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            return json.dumps(val)
        except Exception:
            return str(val)

    def _demo_io_from_obj(obj) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # If it exposes explicit inputs/outputs
        for attr in ("inputs", "input"):
            din = getattr(obj, attr, None)
            if isinstance(din, dict):
                dout = getattr(obj, "outputs", None) or getattr(obj, "output", None)
                if isinstance(dout, dict):
                    return din, dout
        # If it's mapping-like, split by signature fields
        if isinstance(obj, dict):
            in_keys = set((sig.input_fields or {}).keys())
            out_keys = set((sig.output_fields or {}).keys())
            din = {k: obj[k] for k in obj.keys() if k in in_keys}
            dout = {k: obj[k] for k in obj.keys() if k in out_keys}
            if din or dout:
                return din, dout
        # Fallback: try dict(obj)
        try:
            od = dict(obj)
            return _demo_io_from_obj(od)
        except Exception:
            return {}, {}

    demos = []
    demo_renders = []
    try:
        mod = getattr(fn_or_wrapper.__dspy__, "module", None)
        # Look for a top-level demos attr or nested in common submodules
        candidate_lists = []
        for attr in ("demos", "demo", "trainset"):
            if hasattr(mod, attr):
                candidate_lists.append(getattr(mod, attr))
        for sub in ("predict", "react", "extract", "program", "backbone"):
            if hasattr(mod, sub):
                obj = getattr(mod, sub)
                for attr in ("demos", "demo"):
                    if hasattr(obj, attr):
                        candidate_lists.append(getattr(obj, attr))
        # Flatten and normalize
        flat = []
        for c in candidate_lists:
            if c is None:
                continue
            try:
                flat.extend(list(c))
            except Exception:
                continue
        # Deduplicate by id
        seen = set()
        for ex in flat:
            if id(ex) in seen:
                continue
            seen.add(id(ex))
            din, dout = _demo_io_from_obj(ex)
            if not (din or dout):
                continue
            demos.append({"inputs": din, "outputs": dout})
        if demos:
            render_lines.append("")
            render_lines.append(f"Demos ({len(demos)}):")
            for i, d in enumerate(demos, 1):
                render_lines.append(f"- Demo {i}:")
                if d["inputs"]:
                    render_lines.append("  Inputs:")
                    for k, v in d["inputs"].items():
                        render_lines.append(f"    - {k}: {_maybe_to_plain(v)}")
                if d["outputs"]:
                    render_lines.append("  Outputs:")
                    for k, v in d["outputs"].items():
                        render_lines.append(f"    - {k}: {_maybe_to_plain(v)}")
    except Exception:
        pass

    render = "\n".join(render_lines).strip()

    return {
        "adapter": type(adapter_inst).__name__,
        "module": type(getattr(fn_or_wrapper.__dspy__, "module", None)).__name__,
        "inputs": in_text,
        "messages": messages,
        "render": render,
        "signature": sig,
        "demos": demos,
    }


def inspect_history_text() -> str:
    """Return dspy.inspect_history() as text (best effort).

    Captures stdout output of dspy.inspect_history() for convenience.
    """
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
