"""The ``@ai`` decorator, the ``_ai`` sentinel, and the AI function object.

The function definition is the prompt, and the function body is the program:

    @ai
    def solve(problem: str) -> float:
        \"\"\"Solve the word problem.\"\"\"
        reasoning: str = _ai["Think step by step."]
        return _ai

Parameters are the inputs, ``_ai`` declarations and the return type are the
outputs, the docstring and comments are the instruction. lmcc lays out the
call and reads the reply; lm15 talks to the provider (see ``engine``).
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import threading
import warnings
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import lmcc

from . import adapters, engine, models
from .config import DEFAULTS, check, configure, effective, settings  # noqa: F401 — re-exported
from .data import Prediction
from .docments import (UNSET, docments, docstring, extract_docstrings, flexiclass,  # noqa: F401
                       get_dataclass_source, get_name, get_source, isdataclass, parse_docstring,
                       qual_name, sig2str)
from .engine import inspect_history, phistory  # noqa: F401 — re-exported
from .signature import (MAIN_OUTPUT_DEFAULT_NAME, Spec, _collect_ast_outputs, _extract_return_names,
                        build_spec, describe_signature)

# ──────────────────────────────────────────────────────────────────────────────
# Program state: what optimization changes
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ProgramState:
    """What an optimizer tunes in one AI function: the instruction (None: the one
    written from the code) and the demos (lmcc turns, or ``{"inputs", "outputs"}``)."""
    instructions: Optional[str] = None
    demos: Tuple[Any, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"instructions": self.instructions,
                "demos": [d.to_dict() if isinstance(d, lmcc.Turn) else d for d in self.demos]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgramState":
        return cls(instructions=data.get("instructions"), demos=tuple(data.get("demos") or ()))


# Optimizers try candidate states without touching the function: id(fn) → state.
_STATE_OVERRIDE: ContextVar[Dict[int, ProgramState]] = ContextVar("functai_state_override", default={})
# While an optimizer bootstraps, every finished call is recorded here: (fn, prediction).
_TRACE: ContextVar[Optional[List[Tuple["FunctAIFunc", Prediction]]]] = ContextVar("functai_trace", default=None)


# ──────────────────────────────────────────────────────────────────────────────
# Call context and the `_ai` sentinel
# ──────────────────────────────────────────────────────────────────────────────


class _CallContext:
    def __init__(self, *, program: "FunctAIFunc", spec: Spec, inputs: Dict[str, Any]):
        self.program = program
        self.spec = spec
        self.inputs = inputs
        self.main_output_name = spec.main
        self._materialized = False
        self._pred: Optional[Prediction] = None
        self._value: Any = None
        self._ai_requested = False
        self.collect_only: bool = False
        self._requested_outputs: Dict[str, Tuple[Any, str]] = {}

    def request_ai(self):
        self._ai_requested = True
        return self

    def declare_output(self, *, name: str, typ: Any = str, desc: str = "") -> None:
        if name and name not in self._requested_outputs:
            self._requested_outputs[name] = (typ or str, desc or "")

    def requested_outputs(self) -> List[Tuple[str, Any, str]]:
        return [(n, t, d) for n, (t, d) in self._requested_outputs.items()]

    def ensure_materialized(self):
        if self._materialized:
            return
        if self.collect_only:
            raise RuntimeError("_ai value accessed before model run; declare outputs with _ai[\"desc\"] and return _ai.")
        self._pred = self.program._run(self.inputs, self.spec)
        self._value = self._pred.get(self.main_output_name)
        self._materialized = True

    @property
    def value(self):
        self.ensure_materialized()
        return self._value

    def output_value(self, name: str, typ: Any = str):
        self.ensure_materialized()
        if name not in self._pred:
            raise KeyError(f"{self.program.__name__}: the model was not asked for an output named {name!r} "
                           f"(outputs: {list(self._pred)}). Declare it in the body as "
                           f"`{name}: T = _ai[\"...\"]` so it becomes part of the signature.")
        return self._pred[name]


_ACTIVE_CALL: ContextVar[Optional[_CallContext]] = ContextVar("functai_active_call", default=None)


def _active() -> _CallContext:
    ctx = _ACTIVE_CALL.get()
    if ctx is None:
        raise RuntimeError("`_ai` can only be used inside an @ai-decorated function call.")
    return ctx


class _AISentinel:
    """Module-level `_ai` sentinel: stands for the model's output inside an @ai body."""

    def __repr__(self):
        return "<_ai>"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(_active().request_ai().value, name)

    def _val(self):
        return _active().request_ai().value

    # Conversions & operators
    def __str__(self): return str(self._val())
    def __format__(self, spec): return format(self._val(), spec)
    def __int__(self): return int(self._val())
    def __float__(self): return float(self._val())
    def __index__(self): return self._val().__index__()
    def __round__(self, n=None): return round(self._val(), n)
    def __abs__(self): return abs(self._val())
    def __neg__(self): return -self._val()
    def __bool__(self): return bool(self._val())
    def __len__(self): return len(self._val())
    def __iter__(self): return iter(self._val())
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
    __hash__ = None

    def __getitem__(self, spec):
        ctx = _active()
        ctx.request_ai()
        if isinstance(spec, str):
            # A bare string is a description; bind the proxy to the variable the
            # body assigns it to (found by the AST), else derive a name.
            desc = spec
            bound_name: Optional[str] = None
            for n, _t, d in _collect_ast_outputs(ctx.program._fn):
                if d == desc:
                    bound_name = n
                    break
            name = bound_name if bound_name else _derive_output_name(desc)
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
    """One declared output (``x: T = _ai["..."]``), resolved when first used."""

    def __init__(self, ctx: _CallContext, *, name: str, typ: Any = str):
        self._ctx = ctx
        self._name = name
        self._typ = typ or str

    def _resolve(self):
        if self._ctx.collect_only:
            raise RuntimeError(f"Output '{self._name}' value is not available during signature collection.")
        return self._ctx.output_value(self._name, self._typ)

    def __repr__(self):
        if self._ctx._materialized:
            return f"<_ai[{self._name!s}]={self._resolve()!r}>"
        return f"<_ai[{self._name!s}]>"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def __str__(self): return str(self._resolve())
    def __format__(self, spec): return format(self._resolve(), spec)
    def __int__(self): return int(self._resolve())
    def __float__(self): return float(self._resolve())
    def __index__(self): return self._resolve().__index__()
    def __round__(self, n=None): return round(self._resolve(), n)
    def __abs__(self): return abs(self._resolve())
    def __neg__(self): return -self._resolve()
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
    __hash__ = None


def _derive_output_name(desc: str) -> str:
    s = ''.join(ch if (ch.isalnum() or ch == '_') else ' ' for ch in str(desc))
    s = s.strip().lower()
    if not s:
        return "field"
    return s.split()[0]


_ai = _AISentinel()


# ──────────────────────────────────────────────────────────────────────────────
# The AI function
# ──────────────────────────────────────────────────────────────────────────────

_MODULES = {"predict": "predict", "p": "predict", "": "predict",
            "cot": "cot", "chainofthought": "cot", "chain_of_thought": "cot",
            "react": "react", "ra": "react"}

# Decorator arguments that are settings (stored per function, resolved at call time).
_SETTING_ALIASES = {
    "n_auto_examples": "autocompile_n",
    "autoinstruct_improve_calls": "instruction_autorefine_calls",
}


def _module_name(module: Any) -> str:
    if module is None:
        return "predict"
    if isinstance(module, str) and module.lower().replace("-", "") in _MODULES:
        return _MODULES[module.lower().replace("-", "")]
    raise TypeError(f"module must be 'predict', 'cot' or 'react', not {module!r}. (functai 1.0 no longer "
                    f"runs DSPy modules; chain of thought is module='cot' or a `reasoning: str = _ai[...]` "
                    f"line, tools run in the tool loop.)")


class FunctAIFunc:
    """A typed Python function whose body is a model call. Build with ``@ai``."""

    def __init__(self, fn, *, tools: Optional[List[Any]] = None, template: Any = None, messages: Any = None,
                 module_kwargs: Optional[Dict[str, Any]] = None, examples: Any = None, **cfg):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._sig = inspect.signature(fn)
        for old, new in _SETTING_ALIASES.items():
            if old in cfg:
                value = cfg.pop(old)
                if value is not None:
                    cfg.setdefault(new, value)
        mk = dict(module_kwargs or {})
        if "max_iters" in mk:
            cfg.setdefault("max_steps", mk.pop("max_iters"))
        if mk:
            raise TypeError(f"@ai on {fn.__name__}: module_kwargs {sorted(mk)} were DSPy module arguments; "
                            f"functai 1.0 accepts only max_iters (the tool loop's max_steps)")
        self._settings: Dict[str, Any] = check({k: v for k, v in cfg.items() if v is not None},
                                               f"@ai on {fn.__name__}")
        if "module" in self._settings:
            self._settings["module"] = _module_name(self._settings["module"])
        if template is not None and messages is not None:
            raise TypeError("give template=[...] or messages=[...], not both")
        template = template if template is not None else messages
        if template is not None and "adapter" in self._settings:
            raise TypeError("give adapter=... or template=[...], not both: a template is an adapter")
        self._template = tuple(template) if template is not None else None
        if self._template is not None:
            adapters.template_adapter(self._template)       # template syntax errors surface now
        self._tools: List[Callable] = list(tools or [])
        self._tool_specs = [engine.tool_spec(t) for t in self._tools]
        self._state = ProgramState()
        self.history: List[lmcc.Turn] = []
        self._lock = threading.RLock()
        self._spec_cache: Dict[Tuple, Spec] = {}
        self._plan_cache: Dict[Tuple, lmcc.Plan] = {}
        self._opt_stack: List[ProgramState] = []
        self._opt_runs: List[Dict[str, Any]] = []
        self._states_history: List[ProgramState] = []
        self._instr_observed: List[Dict[str, Any]] = []
        self._instr_refined = 0
        self._instr_frozen = False
        self._autoinstructed = False
        self._spec()                                          # signature errors surface at definition
        if examples:
            self._state = ProgramState(demos=tuple(self._example_demo(e) for e in examples))

    # ----- settings -----

    def _effective(self) -> Dict[str, Any]:
        return effective(self._settings)

    def _set(self, name: str, value: Any) -> None:
        with self._lock:
            if value is None:
                self._settings.pop(name, None)
            else:
                self._settings[name] = value
            self._plan_cache.clear()

    @property
    def lm(self): return self._settings.get("lm")
    @lm.setter
    def lm(self, v): self._set("lm", v)

    @property
    def adapter(self): return self._settings.get("adapter")
    @adapter.setter
    def adapter(self, v):
        self._template = None
        self._set("adapter", v)

    @property
    def template(self): return self._template
    @template.setter
    def template(self, messages):
        self._template = tuple(messages) if messages is not None else None
        if self._template is not None:
            adapters.template_adapter(self._template)
            self._settings.pop("adapter", None)
        self._plan_cache.clear()

    @property
    def module(self): return self._effective().get("module")
    @module.setter
    def module(self, v): self._set("module", _module_name(v))

    @property
    def temperature(self): return self._settings.get("temperature")
    @temperature.setter
    def temperature(self, v): self._set("temperature", v)

    @property
    def tools(self): return list(self._tools)
    @tools.setter
    def tools(self, seq):
        self._tools = list(seq or [])
        self._tool_specs = [engine.tool_spec(t) for t in self._tools]
        self._spec_cache.clear()
        self._plan_cache.clear()

    @property
    def optimizer(self): return self._effective().get("optimizer")
    @optimizer.setter
    def optimizer(self, v): self._set("optimizer", v)

    @property
    def debug(self): return bool(self._effective().get("debug"))
    @debug.setter
    def debug(self, v: bool): self._set("debug", bool(v))

    def using(self, **settings) -> "FunctAIFunc":
        """A copy of this function with other settings (``f.using(lm="gpt-4.1")(x)``).
        The copy starts with the same instruction and demos; the original is untouched."""
        clone = object.__new__(FunctAIFunc)
        clone.__dict__.update(self.__dict__)
        clone._settings = {**self._settings,
                           **{k: v for k, v in check(settings, "using").items() if v is not None}}
        if "module" in settings:
            clone._settings["module"] = _module_name(settings["module"])
        clone.history = []
        clone._lock = threading.RLock()
        clone._spec_cache, clone._plan_cache = {}, {}
        clone._opt_stack, clone._opt_runs, clone._states_history = [], [], []
        return clone

    # ----- state (what optimizers change) -----

    def _current_state(self) -> ProgramState:
        return _STATE_OVERRIDE.get().get(id(self), self._state)

    def state(self) -> ProgramState:
        """The instruction and demos in use."""
        return self._current_state()

    def load_state(self, state: "ProgramState | Dict[str, Any]") -> "FunctAIFunc":
        self._state = state if isinstance(state, ProgramState) else ProgramState.from_dict(state)
        return self

    @property
    def instructions(self) -> str:
        """The instruction the model gets: an optimized one, or the one written from the code."""
        return self._spec().signature.instructions

    @instructions.setter
    def instructions(self, text: Optional[str]) -> None:
        self._state = dataclasses.replace(self._state, instructions=text)

    @property
    def demos(self) -> List[Any]:
        return list(self._current_state().demos)

    @demos.setter
    def demos(self, items) -> None:
        self._state = dataclasses.replace(self._state, demos=tuple(self._example_demo(e) for e in (items or ())))

    def _example_demo(self, item: Any) -> Any:
        """A demo as stored: a Turn as is, else ``{"inputs", "outputs"}``."""
        if isinstance(item, lmcc.Turn):
            return item
        if isinstance(item, dict) and "signature" in item and "inputs" in item:
            return item
        from .data import as_example
        if isinstance(item, tuple) and len(item) == 2 and not isinstance(item[0], dict):
            names = list(self._sig.parameters)
            item = ({names[0]: item[0]}, {self._spec().main: item[1]})
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict) and not isinstance(item[1], dict):
            item = (item[0], {self._spec().main: item[1]})
        ex = as_example(item, self._sig.parameters)
        ins = {k: ex[k] for k in ex.input_keys or ()}
        return {"inputs": ins, "outputs": {k: v for k, v in ex.items() if k not in ins}}

    def save(self, path: "str | Path") -> None:
        """Write the instruction and demos to a JSON file (``load`` reads it back)."""
        data = {"functai": 1, "function": self.__name__, **self._state.to_dict()}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))

    def load(self, path: "str | Path") -> "FunctAIFunc":
        data = json.loads(Path(path).read_text())
        if data.get("functai") != 1:
            raise ValueError(f"{path}: not a functai program file")
        return self.load_state(ProgramState.from_dict(data))

    def reset(self) -> None:
        """Forget the conversation (stateful functions)."""
        self.history.clear()

    # ----- signature and plan -----

    def _spec(self, instructions: Optional[str] = None) -> Spec:
        s = self._effective()
        if instructions is None:
            instructions = self._current_state().instructions
        key = (instructions, bool(s.get("include_fn_name_in_instructions")),
               _module_name(s.get("module")) == "cot", bool(self._tools))
        spec = self._spec_cache.get(key)
        if spec is None:
            spec = build_spec(self._fn, instructions=instructions, include_fn_name=key[1], reasoning=key[2],
                              tools=key[3], registry=adapters.REGISTRY)
            self._spec_cache[key] = spec
        return spec

    @property
    def signature(self) -> lmcc.SignatureCore:
        """The lmcc signature: inputs, outputs, instruction."""
        return self._spec().signature

    def _layout(self, settings: Dict[str, Any]) -> adapters.Layout:
        if self._template is not None:
            return adapters.Layout(template=self._template)
        return adapters.Layout(adapter=settings.get("adapter"))

    def _plan_for(self, spec: Spec, settings: Dict[str, Any]):
        router, model, route = models.resolve(settings)
        caps = models.model_capabilities(settings, route)
        layout = self._layout(settings)
        key = (layout.key(), json.dumps(caps, sort_keys=True), spec.signature.instructions,
               spec.reasoning, spec.tools, route.provider)
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = adapters.bind(layout, spec.signature, caps, route.provider)
            with self._lock:
                if len(self._plan_cache) > 64:
                    self._plan_cache.clear()
                self._plan_cache[key] = plan
        return plan, router, model, route

    def plan(self) -> lmcc.Plan:
        """The lmcc plan for the current model: ``.explain()``, ``.describe()``, ``.render(...)``."""
        return self._plan_for(self._spec(), self._effective())[0]  # (plan, router, model, route)

    def explain(self) -> str:
        """How calls are laid out for the current model: adapter, reader, transports, formats."""
        return self.plan().explain()

    def _past(self, plan: lmcc.Plan, spec: Spec, settings: Dict[str, Any]) -> List[lmcc.Turn]:
        past = [t for t in (engine.fit_turn(plan, spec, d) for d in self._current_state().demos) if t is not None]
        if settings.get("stateful") and self.history:
            window = int(settings.get("state_window") or 0)
            recent = self.history[-window:] if window > 0 else self.history
            past += [t for t in (engine.fit_turn(plan, spec, h) for h in recent) if t is not None]
        return past

    def render(self, *args, **kwargs):
        """The exact lm15 request the first model call would send. No network."""
        inputs = self._bind_inputs(args, kwargs)
        spec, s = self._spec(), self._effective()
        plan, _router, model, route = self._plan_for(spec, s)
        s = models.adjust(s, route)
        values = engine.prepare_inputs(spec, inputs)
        if spec.tools:
            values["tools"] = list(self._tool_specs)
        rendered = plan.render(plan.turn(values), turns=self._past(plan, spec, s))
        import lmcc_lm15
        return lmcc_lm15.request(rendered, model=model, config=engine.config_of(s))

    # ----- representations -----

    def __repr__(self) -> str:
        try:
            spec = self._spec()
            ins = [f.name for f in spec.signature.inputs if f.purpose == "plain"]
            outs = [f.name for f in spec.signature.outputs if f.purpose != "tools.calls"]
            parts = []
            if ins:
                parts.append("inputs=" + ", ".join(ins))
            if outs:
                parts.append("outputs=" + ", ".join(outs) + f" (primary={spec.main})")
            if self._tools:
                parts.append("tools=" + ", ".join(t.name for t in self._tool_specs))
            return f"<FunctAIFunc {self._fn.__name__} | " + "; ".join(parts) + ">"
        except Exception:
            return f"<FunctAIFunc {getattr(self._fn, '__name__', 'unknown')}>"

    # ----- calling -----

    def _bind_inputs(self, args, kwargs) -> Dict[str, Any]:
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _run(self, inputs: Dict[str, Any], spec: Spec) -> Prediction:
        """One model call (or tool loop) for these inputs; used by the body's `_ai`."""
        s = self._effective()
        if (s.get("autocompile") or s.get("autoinstruct")) and not self._autoinstructed:
            self._autoinstruct(s)
            spec = self._spec()
        plan, router, model, route = self._plan_for(spec, s)
        s = models.adjust(s, route)
        past = self._past(plan, spec, s)
        pred = engine.run(function=self.__name__, plan=plan, spec=spec, inputs=inputs, past=past, settings=s,
                          router=router, model=model, tools={t.__name__: t for t in self._tools if callable(t)},
                          tool_specs=self._tool_specs)
        if s.get("stateful"):
            with self._lock:
                self.history.append(pred.turn)
                window = int(s.get("state_window") or 0)
                if window > 0 and len(self.history) > window:
                    del self.history[:-window]
        trace = _TRACE.get()
        if trace is not None:
            trace.append((self, pred))
        if s.get("debug"):
            print(f"[functai] {self.__name__}: model={model}; adapter={plan.adapter.name}; "
                  f"outputs={list(pred)} (primary={spec.main}); tokens={pred.usage}")
        if int(s.get("instruction_autorefine_calls") or 0) > 0 and not self._instr_frozen:
            self._record_and_maybe_refine(inputs, dict(pred), s)
        return pred

    def __call__(self, *args, all: bool = False, **kwargs):
        # Back-compat: map deprecated _prediction to all
        if "_prediction" in kwargs:
            if kwargs.pop("_prediction"):
                all = True
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in {"_prediction", "all"}}
        inputs = self._bind_inputs(args, clean_kwargs)
        spec = self._spec()
        ctx = _CallContext(program=self, spec=spec, inputs=inputs)
        token = _ACTIVE_CALL.set(ctx)
        try:
            result = self._fn(*args, **clean_kwargs)

            if all:
                _ = ctx.request_ai().value
                return ctx._pred

            if result is _ai or result is Ellipsis:
                return ctx.request_ai().value

            # Unwrap proxies and realize bare `_ai` placeholders inside containers.
            def _has_bare_ai(x):
                if x is _ai:
                    return True
                if isinstance(x, (list, tuple)):
                    return any(_has_bare_ai(i) for i in x)
                if isinstance(x, dict):
                    return any(_has_bare_ai(v) for v in x.values())
                return False

            # Return field order for mapping bare `_ai` occurrences.
            ret_names = _extract_return_names(self._fn) or [n for n, _t, _d in _collect_ast_outputs(self._fn)]
            names_pool = [n for n in ret_names if n in spec.outputs]

            def _next_name():
                return names_pool.pop(0) if names_pool else None

            def _unwrap_and_realize(x):
                if isinstance(x, _AIFieldProxy):
                    return x._resolve()
                if x is _ai:
                    ctx.request_ai().value
                    return ctx.output_value(_next_name() or spec.main)
                if isinstance(x, list):
                    return [_unwrap_and_realize(i) for i in x]
                if isinstance(x, tuple):
                    return tuple(_unwrap_and_realize(i) for i in x)
                if isinstance(x, dict):
                    return {k: _unwrap_and_realize(v) for k, v in x.items()}
                return x

            if _has_bare_ai(result):
                ctx.request_ai().value
            result = _unwrap_and_realize(result)
            if result is None and not ctx._ai_requested:
                return ctx.request_ai().value
            return result
        finally:
            _ACTIVE_CALL.reset(token)

    # ----- optimization -----

    def opt(self, *, trainset: Optional[List[Any]] = None, optimizer: Any = None,
            metric: Optional[Callable] = None, valset: Optional[List[Any]] = None, **opts) -> "FunctAIFunc":
        """Optimize the instruction and/or demos on examples, in place (``undo_opt`` reverts).

        - ``trainset``: Examples, dicts, or ``(inputs, outputs)`` pairs
        - ``optimizer``: an optimizer class or instance (default ``BootstrapFewShot``)
        - ``metric``: ``metric(example, prediction[, trace]) -> float | bool``
          (default: exact match on the labeled outputs)
        - ``teacher`` / ``teacher_lm``: a stronger model (or AI function) to learn from
        - ``n_synth``: first synthesize this many examples with the teacher
        - other keywords go to the optimizer class
        """
        from .optimizers import optimize
        optimize(self, trainset=trainset, optimizer=optimizer, metric=metric, valset=valset, **opts)
        return self

    def _apply_state(self, state: ProgramState, *, log: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._opt_stack.append(self._state)
            self._state = state
            self._states_history.append(state)
            if log is not None:
                self._opt_runs.append(log)

    def undo_opt(self, steps: int = 1) -> None:
        """Revert the last ``steps`` optimizations."""
        for _ in range(max(1, int(steps))):
            if not self._opt_stack:
                break
            self._state = self._opt_stack.pop()

    def programs(self) -> List[ProgramState]:
        """Every state optimization produced for this function, oldest first."""
        return list(self._states_history)

    def latest_program(self, fresh: bool = False) -> ProgramState:
        """The state in use (``fresh=True``: the unoptimized one)."""
        return ProgramState() if fresh else self._current_state()

    def optimization_runs(self) -> List[Dict[str, Any]]:
        return list(self._opt_runs)

    def to_dspy(self, deepcopy: bool = False):
        raise NotImplementedError("functai 1.0 no longer runs on DSPy. The optimized program is "
                                  "`fn.state()` (instruction and demos); `fn.save(path)` writes it as JSON.")

    # ----- instruction writing and refinement (opt-in) -----

    def freeze(self) -> "FunctAIFunc":
        """Stop further automatic instruction refinement."""
        self._instr_frozen = True
        return self

    def _autoinstruct(self, s: Dict[str, Any]) -> None:
        self._autoinstructed = True
        from . import meta
        lm = s.get("instruction_lm") or s.get("teacher_lm") or s.get("lm")
        try:
            text = meta.write_instruction(self, lm=lm)
        except Exception as exc:  # noqa: BLE001 — the function still works with its own instruction
            warnings.warn(f"[functai] {self.__name__}: writing an instruction failed ({exc}); "
                          f"using the one from the code")
            return
        self._state = dataclasses.replace(self._state, instructions=text)

    def _record_and_maybe_refine(self, inputs: Dict[str, Any], outputs: Dict[str, Any], s: Dict[str, Any]) -> None:
        cap = max(1, int(s.get("instruction_autorefine_max_examples") or 20))
        with self._lock:
            self._instr_observed.append({"inputs": inputs, "outputs": outputs})
            del self._instr_observed[:-cap]
            if self._instr_refined >= int(s.get("instruction_autorefine_calls") or 0):
                return
            self._instr_refined += 1
            observed = list(self._instr_observed)
        from . import meta
        lm = s.get("instruction_lm") or s.get("teacher_lm") or s.get("lm")
        try:
            text = meta.refine_instruction(self, observed, lm=lm)
        except Exception as exc:  # noqa: BLE001
            if s.get("debug"):
                warnings.warn(f"[functai] {self.__name__}: instruction refinement failed ({exc})")
            return
        self._state = dataclasses.replace(self._state, instructions=text)


# ──────────────────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────────────────


def ai(_fn=None, **cfg):
    """Turn a typed Python function into an AI function.

        @ai
        def summarize(text: str) -> str:
            \"\"\"Summarize the text in one sentence.\"\"\"
            return _ai

        @ai(lm="claude-haiku-4-5", temperature=0.2, tools=[search], module="cot",
            template=[system("{instruction}"), turns(), user("{text}")])
        def g(text: str) -> str: ...

    Options: ``lm``, ``adapter`` ('xml', 'chat', 'json', an lmcc adapter), ``template``
    (a chat template), ``module`` ('predict', 'cot', 'react'), ``tools``, ``stateful``,
    ``examples``, ``retries``, ``max_steps``, ``capabilities``, ``cache``, any lm15
    Config field (``temperature``, ``max_tokens``, ``seed``, ...), and the
    optimization settings (``teacher``, ``optimizer``, ``autoinstruct``, ...).
    """
    def _decorate(fn):
        return FunctAIFunc(fn, **cfg)
    if _fn is not None and callable(_fn):
        return _decorate(_fn)
    return _decorate


# ──────────────────────────────────────────────────────────────────────────────
# Signature helpers
# ──────────────────────────────────────────────────────────────────────────────


def _program_of(fn_or_prog) -> FunctAIFunc:
    if isinstance(fn_or_prog, FunctAIFunc):
        return fn_or_prog
    wrapped = getattr(fn_or_prog, "__wrapped__", None)
    if isinstance(wrapped, FunctAIFunc):
        return wrapped
    raise TypeError("expected an @ai-decorated function")


def compute_signature(fn_or_prog) -> lmcc.SignatureCore:
    """The lmcc signature of an @ai function: inputs, outputs, instruction."""
    return _program_of(fn_or_prog).signature


def signature_text(fn_or_prog) -> str:
    """A one-line summary of the signature."""
    prog = _program_of(fn_or_prog)
    return describe_signature(prog._spec(), prog._fn.__name__)


__all__ = [
    "ai", "_ai", "configure", "settings", "phistory", "inspect_history", "compute_signature", "signature_text",
    "FunctAIFunc", "ProgramState", "MAIN_OUTPUT_DEFAULT_NAME", "DEFAULTS",
    "flexiclass", "UNSET", "docstring", "parse_docstring", "docments", "isdataclass",
    "get_dataclass_source", "get_source", "get_name", "qual_name", "sig2str", "extract_docstrings",
]
