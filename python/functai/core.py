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

import asyncio
import copy
import dataclasses
import functools
import inspect
import json
import threading
import warnings
from contextvars import ContextVar
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, List, Optional, ParamSpec, Protocol, Tuple, TypeVar, overload,
                    runtime_checkable)

import lmcc

from . import adapters, calllog, engine, models, streaming
from .config import DEFAULTS, check, configure, effective, settings  # noqa: F401 — re-exported
from .data import Prediction
from .docments import (UNSET, docments, docstring, extract_docstrings, flexiclass,  # noqa: F401
                       get_dataclass_source, get_name, get_source, isdataclass, parse_docstring,
                       qual_name, sig2str)
from .bake.baked import is_baked
from .engine import inspect_history, phistory  # noqa: F401 — re-exported
from .signature import (MAIN_OUTPUT_DEFAULT_NAME, Spec, _collect_ast_outputs, _return_slots, bind_named_outputs,
                        model_writes_body,
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

    def __repr__(self) -> str:
        def short(v: Any) -> str:
            text = repr(v)
            return text if len(text) <= 70 else text[:67] + "..."

        def io(d: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            if isinstance(d, lmcc.Turn):
                return dict(d.inputs), dict(d.outputs or {})
            if isinstance(d, dict) and "signature" in d:           # a saved turn
                return dict(d.get("inputs") or {}), dict(d.get("outputs") or {})
            return dict(d.get("inputs") or {}), dict(d.get("outputs") or {})

        lines = ["instruction: " + ("(written from the code)" if self.instructions is None
                                     else short(self.instructions))]
        lines.append(f"examples: {len(self.demos)}" if self.demos else "examples: none")
        for i, d in enumerate(self.demos, 1):
            ins, outs = io(d)
            lines.append(f"  {i}. " + ", ".join(f"{k}={short(v)}" for k, v in ins.items())
                         + "  →  " + ", ".join(f"{k}={short(v)}" for k, v in outs.items()))
        return "\n".join(lines)


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


# Typed as Any so that `return _ai` and `x: T = _ai` type-check as the value they stand for.
_ai: Any = _AISentinel()
"""The model's answer, inside an AI function's body.

A bare ``_ai`` is always the answer, and behaves like the value it stands
for: ``return _ai``, ``return round(_ai, 2)``, ``return critique, _ai``.
``x: T = _ai`` declares one more output, named ``x`` and written before the
answer; a comment on the line (or ``_ai["..."]``) describes it."""


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


_KEEP = object()   # `using` left the template as it was


def _checked_template(messages: Any) -> Optional[tuple]:
    """A template as stored (a tuple of messages), refused now if lmcc cannot
    compile it; None for no template."""
    if messages is None:
        return None
    if isinstance(messages, (str, dict)):
        raise TypeError("a template is a list of messages: [system(...), turns(), user(...)]")
    checked = tuple(messages)
    adapters.template_adapter(checked)
    return checked


def _module_name(module: Any) -> str:
    if module is None:
        return "predict"
    if isinstance(module, str) and module.lower().replace("-", "") in _MODULES:
        return _MODULES[module.lower().replace("-", "")]
    raise TypeError(f"module must be 'predict', 'cot' or 'react', not {module!r}. (functai 1.0 no longer "
                    f"runs DSPy modules; chain of thought is module='cot' or a `reasoning: str = _ai[...]` "
                    f"line, tools run in the tool loop.)")


P = ParamSpec("P")
R = TypeVar("R", covariant=True)


@runtime_checkable
class ColumnExpr(Protocol):
    """A table column expression (dpyr's ``col.message``): an AI function called
    on one gives a column, one model call per row."""

    def is_na(self) -> Any: ...

    def str_to_lower(self) -> Any: ...


class FunctAIFunc(Generic[P, R]):
    """A typed Python function whose body is a model call. Build with ``@ai``.

    Calling it runs the model and gives the answer, typed as the function's
    return type; ``predict`` gives every output; ``acall`` and ``apredict`` are
    the same in async code (an ``async def`` AI function is awaited directly).
    """

    def __init__(self, fn, *, tools: Optional[List[Any]] = None, template: Any = None, messages: Any = None,
                 module_kwargs: Optional[Dict[str, Any]] = None, examples: Any = None,
                 requires: Optional[List[str]] = None, **cfg):
        functools.update_wrapper(self, fn)
        # requirements the code reaches in ways functai.check cannot see ("numpy>=2")
        self._requires: Tuple[str, ...] = tuple(requires or ())
        self._async = inspect.iscoroutinefunction(fn)
        if self._async and not model_writes_body(fn):
            raise TypeError(f"@ai on async def {fn.__name__}: an async AI function's body is the model call only "
                            f"(a docstring, `...` or `return _ai`, and output declarations). For code around the "
                            f"model, write it with `def` and await `{fn.__name__}.acall(...)`.")
        self._fn = bind_named_outputs(fn)     # `label = _ai` is the output `label`
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
        self._template = _checked_template(template)        # template syntax errors surface now
        self._tools: List[Callable] = list(tools or [])
        self._tool_specs = [engine.tool_spec(t) for t in self._tools]
        self._state = ProgramState()
        self.history: List[lmcc.Turn] = []
        self._lock = threading.RLock()
        self._spec_cache: Dict[Tuple, Spec] = {}
        self._plan_cache: Dict[Tuple, lmcc.Plan] = {}
        self._opt_runs: List[Dict[str, Any]] = []             # how this function was improved, oldest first
        self._vectorized: Dict[Any, Any] = {}                 # dpyr row functions, by prompt version
        self._instr_observed: List[Dict[str, Any]] = []
        self._instr_refined = 0
        self._instr_frozen = False
        self._autoinstructed = False
        self._history_calls: List[str] = []                   # the call ids of `history`'s turns
        self._interface_cache: Optional[Tuple[Any, Dict[str, Any]]] = None
        self._spec()                                          # signature errors surface at definition
        self._check_definition()                              # and interface and log_content ones
        if examples:
            self._state = ProgramState(demos=tuple(self._example_demo(e) for e in examples))

    def _check_definition(self) -> None:
        """What every language refuses when an AI function is defined (after
        lmcc accepted its signature): an interface that breaks programs.md's
        rules (an optional input whose default does not fit its shape, one with
        no JSON default...), and a ``log_content`` map naming a field it lacks."""
        from . import interface as _interface
        _interface.check(self.interface, ai=True, program=self.__name__)
        self._check_log_content()

    def _check_log_content(self) -> None:
        content = self._settings.get("log_content")
        if isinstance(content, dict):
            ins, outs, added = self._fields()
            calllog.check_log_content(content, [*ins, *outs, *added], program=self.__name__)

    # ----- settings -----

    def _effective(self) -> Dict[str, Any]:
        return effective(self._settings)

    def _set(self, name: str, value: Any) -> None:
        if value is not None:
            check({name: value}, f"{self.__name__}.{name}")
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
        """A layout: a name ('xml', 'chat', 'json'), an lmcc adapter, or a saved
        adapter artifact. Setting one replaces the function's template."""
        self._set("adapter", v)
        self._template = None

    @property
    def template(self): return self._template
    @template.setter
    def template(self, messages):
        """A chat template; it replaces the function's adapter. None removes it
        (the function then uses its adapter setting, or the default layout)."""
        checked = _checked_template(messages)
        with self._lock:
            self._template = checked
            if checked is not None:
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

    def using(self, *, template: Any = _KEEP, **settings) -> "FunctAIFunc[P, R]":
        '''A copy of this function with other settings or another layout.

        The copy starts with the same instruction and demos; the original is
        untouched. A setting given as None is no longer set by the copy: it
        comes from ``configure`` or the defaults. An adapter replaces the
        template, and a template replaces the adapter.

        Parameters
        ----------
        template : list, optional
            A chat template for the copy.
        **settings
            Any setting ``@ai`` takes: ``lm``, ``temperature``, ``adapter``,
            ``client``, ``tools``...

        Returns
        -------
        FunctAIFunc
            The copy.

        Examples
        --------
        ```python
        @ai
        def capital(country: str) -> str:
            """The country's capital city."""
            ...

        capital.using(lm="gpt-4.1-nano")("Chile")
        ```
        '''
        checked = check(settings, "using")
        if template is not _KEEP and checked.get("adapter") is not None and template is not None:
            raise TypeError("give adapter=... or template=[...], not both: a template is an adapter")
        clone = object.__new__(FunctAIFunc)
        clone.__dict__.update(self.__dict__)
        merged = dict(self._settings)
        for k, v in checked.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = _module_name(v) if k == "module" else v
        clone._settings = merged
        if template is not _KEEP:
            clone._template = _checked_template(template)
            if clone._template is not None:
                merged.pop("adapter", None)
        elif checked.get("adapter") is not None:
            clone._template = None
        clone.history = []
        clone._history_calls = []
        clone._interface_cache = None
        clone._lock = threading.RLock()
        clone._spec_cache, clone._plan_cache = {}, {}
        clone._opt_runs = list(self._opt_runs)
        clone._vectorized = {}
        if "log_content" in checked or "module" in checked or "tools" in checked:
            clone._check_definition()
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
        """A demo as stored: a Turn as is, else ``{"inputs", "outputs"}``.

        Written by hand, a demo is a row (``{"text": ..., "result": ...}``: the
        keys named like parameters are inputs, the rest outputs) or an
        ``(inputs, outputs)`` pair, each side a dict or a single value."""
        if isinstance(item, lmcc.Turn):
            return item
        if isinstance(item, dict) and "signature" in item and "inputs" in item:
            return item                                       # a saved Turn
        names = list(self._sig.parameters)
        if isinstance(item, dict) and set(item) == {"inputs", "outputs"} and isinstance(item["inputs"], dict) \
                and "inputs" not in names:
            return {"inputs": dict(item["inputs"]), "outputs": dict(item["outputs"])}
        if isinstance(item, tuple) and len(item) == 2:
            ins, outs = item
            ins = dict(ins) if isinstance(ins, dict) else {names[0]: ins}
            outs = dict(outs) if isinstance(outs, dict) else {self._spec().main: outs}
            return {"inputs": ins, "outputs": outs}
        if isinstance(item, dict):
            return {"inputs": {k: v for k, v in item.items() if k in names},
                    "outputs": {k: v for k, v in item.items() if k not in names}}
        raise TypeError(f"a demo is a row dict ({{{names[0]!r}: ..., {self._spec().main!r}: ...}}) or an "
                        f"(input, output) pair, not {type(item).__name__}")

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
        with self._lock:
            self.history.clear()
            self._history_calls.clear()

    # ----- the interface, and what a call records -----

    @property
    def interface(self) -> Dict[str, Any]:
        """What a caller gives and gets, as data (contract/programs.md): the
        docstring as ``description``, the parameters as ``inputs`` (a parameter
        with a default is ``optional``, its default in the shape), the outputs
        the body declares and the answer (last) as ``outputs``; not the fields
        FunctAI adds (reasoning, tools). The call log records its signature as
        ``program.interface``; ``functai.save`` writes it; another language
        reads it."""
        from . import interface as _interface
        spec = self._spec()
        cached = self._interface_cache
        if cached is None or cached[0] is not spec:
            cached = self._interface_cache = (spec, _interface.of_ai(self))
        return copy.deepcopy(cached[1])

    def _fields(self) -> Tuple[List[str], List[str], List[str]]:
        """(inputs, outputs, added) of this function's calls, in the record's
        order: the interface's, and the outputs FunctAI adds (reasoning, calls)."""
        spec = self._spec()
        ins = [f.name for f in spec.signature.inputs if f.purpose == "plain"]
        outs = [f.name for f in spec.signature.outputs]
        added = [f.name for f in spec.signature.outputs if f.purpose in ("reasoning", "tools.calls")]
        return ins, outs, added

    def _saw(self) -> Tuple[List[Dict[str, Any]], Any]:
        """(what a call is shown as context, entries of the call log's ``saw``;
        the turns themselves): a stateful function's latest turns, as the ids
        of the calls they were, each shown with its steps."""
        s = self._effective()
        if not s.get("stateful"):
            return [], None
        with self._lock:
            history = list(self.history)
            ids = list(self._history_calls)[-len(history):] if history else []
        ids = [""] * (len(history) - len(ids)) + ids          # turns put in `history` by hand have no call
        pairs = list(zip(ids, history))
        window = int(s.get("state_window") or 0)
        if window > 0:
            pairs = pairs[-window:]
        # A turn no logged call made is shown too: an entry no reader knows says so (unknown-key).
        entries = [({"call": cid, "steps": True} if getattr(turn, "steps", None) else {"call": cid}) if cid
                   else {"unrecorded": True} for cid, turn in pairs]
        return entries, [turn for _cid, turn in pairs]

    # ----- signature and plan -----

    def _spec(self, instructions: Optional[str] = None) -> Spec:
        s = self._effective()
        if instructions is None:
            instructions = self._current_state().instructions
        baked = s.get("lm") if is_baked(s.get("lm")) else None
        cot = _module_name(s.get("module")) == "cot" and (baked is None or bool(baked.meta.get("reasoning")))
        key = (instructions, bool(s.get("include_fn_name_in_instructions")), cot, bool(self._tools))
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

    @property
    def version(self) -> str:
        """The function's version: a fingerprint of what it sends besides its inputs.

        ``sha256:`` of the request it renders for a sample input
        (instruction, worked examples, layout, tools), and of its code when
        code of its own runs beside the model (``return round(_ai, 2)``);
        so the same function in another language has the same version. Optimizing it,
        editing its docstring or changing its layout makes a new version;
        choosing another model does not. Logged calls carry it, and a saved
        folder names the version it holds."""
        return calllog.ai_version(self)

    def _layout(self, settings: Dict[str, Any]) -> adapters.Layout:
        if self._template is not None:
            return adapters.Layout(template=self._template)
        return adapters.Layout(adapter=settings.get("adapter"))

    def _plan_for(self, spec: Spec, settings: Dict[str, Any]):
        router, model, route = models.resolve(settings)
        caps = models.model_capabilities(settings, route)
        baked = settings.get("lm") if is_baked(settings.get("lm")) else None
        if baked is not None:
            # A baked model reads its inputs through the layout it was trained on,
            # whatever this function's own adapter or template says.
            _check_baked_signature(self, spec, baked)
            layout = adapters.Layout(adapter=baked.layout)
        else:
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
            call = calllog.current()
            if call is not None and call.program is self and call.context is not None:
                recent = call.context                      # the turns its record says it saw
            else:
                window = int(settings.get("state_window") or 0)
                recent = self.history[-window:] if window > 0 else self.history
            past += [t for t in (engine.fit_turn(plan, spec, h) for h in recent) if t is not None]
        return past

    def render(self, *args, **kwargs):
        '''The exact request the next call would send, without sending it.

        Parameters
        ----------
        *args, **kwargs
            The call's inputs, as for calling the function.

        Returns
        -------
        lm15.Request
            ``.system``, ``.messages``, ``.tools``, ``.config``, ``.model``.

        See Also
        --------
        phistory : what was actually sent.

        Examples
        --------
        ```python
        @ai
        def capital(country: str) -> str:
            """The country's capital city."""
            ...

        request = capital.render("Chile")
        print(request.system)
        print(request.messages[0].parts[0].text)
        ```
        '''
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
        """One call for these inputs (the model, the tool loop, and escalation when
        the first model is unsure); used by the body's `_ai`."""
        s = self._effective()
        if (s.get("autocompile") or s.get("autoinstruct")) and not self._autoinstructed:
            self._autoinstruct(s)
            spec = self._spec()
        pred, model, plan = self._call_model(inputs, spec, s)
        escalate_to = s.get("escalate_to")
        if escalate_to is not None:
            pred, model, plan = self._maybe_escalate(pred, inputs, s, escalate_to, (model, plan))
        if s.get("stateful"):
            call = calllog.current()
            with self._lock:
                self.history.append(pred.turn)
                self._history_calls.append(call.id if call is not None and call.program is self else "")
                window = int(s.get("state_window") or 0)
                if window > 0 and len(self.history) > window:
                    del self.history[:-window]
                    del self._history_calls[:-window]
        trace = _TRACE.get()
        if trace is not None:
            trace.append((self, pred))
        if s.get("debug"):
            print(f"[functai] {self.__name__}: model={model}; adapter={plan.adapter.name}; "
                  f"outputs={list(pred)} (primary={spec.main}); tokens={pred.usage}"
                  + (f"; escalated (first answer's confidence {pred.first.confidence:.2f})" if pred.escalated else ""))
        if int(s.get("instruction_autorefine_calls") or 0) > 0 and not self._instr_frozen:
            self._record_and_maybe_refine(inputs, dict(pred), s)
        calllog.attach(self, pred)
        return pred

    def _maybe_escalate(self, pred: Prediction, inputs: Dict[str, Any], s: Dict[str, Any], escalate_to: Any,
                        first: Tuple[Any, Any]):
        """When the first model is less sure than ``escalate_below``, ask ``escalate_to``
        (a model, a baked model, or an AI function) and return its answer, with the
        first one kept as ``.first``."""
        from .config import forced
        conf = pred.confidence
        if conf is None:
            raise ValueError(f"{self.__name__}: escalate_to needs a first model that measures its confidence (a baked "
                             f"model, Jev, or probabilities='required'); {first[0]} gave no probabilities")
        threshold = float(s.get("escalate_below") if s.get("escalate_below") is not None else 0.9)
        if conf >= threshold:
            return pred, first[0], first[1]
        call = calllog.current()
        if call is not None:
            target = escalate_to.__name__ if isinstance(escalate_to, FunctAIFunc) else models.model_string(escalate_to) \
                if not isinstance(escalate_to, str) else escalate_to
            call.emit("retry", reason=f"the first model was {conf:.0%} sure (less than {threshold:.0%}); {target} "
                                      f"answers instead", wait=None)
            if isinstance(escalate_to, FunctAIFunc):
                call.delegating = True                   # the next call inside this one answers for it
        if isinstance(escalate_to, FunctAIFunc):
            # the target follows its own escalate_to (a longer chain), never a global one
            from .config import scoped
            with scoped(escalate_to=None):
                second = escalate_to._invoke((), inputs, full=True)
            model, plan = escalate_to.__name__, first[1]
        else:
            # This call only: the escalation model, and no second escalation from it.
            with forced(lm=escalate_to, escalate_to=None):
                s2 = self._effective()
                second, model, plan = self._call_model(inputs, self._spec(), s2)
        object.__setattr__(second, "escalated", True)
        object.__setattr__(second, "first", pred)
        return second, model, plan

    def _call_model(self, inputs: Dict[str, Any], spec: Spec, s: Dict[str, Any]):
        plan, router, model, route = self._plan_for(spec, s)
        rec = engine.RECORDING.get()
        if rec is not None and s.get("lm") is not None:
            rec["routes"][models.model_string(s["lm"]) if isinstance(s["lm"], str) else model] = \
                [route.provider, route.model, model]
        s = models.adjust(s, route)
        calllog.route(route.provider)
        past = self._past(plan, spec, s)
        pred = engine.run(function=self.__name__, plan=plan, spec=spec, inputs=inputs, past=past, settings=s,
                          router=router, model=model, tools={t.__name__: t for t in self._tools if callable(t)},
                          tool_specs=self._tool_specs)
        return pred, model, plan

    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    @overload
    def __call__(self, column: ColumnExpr, /, *args: Any, **kwargs: Any) -> Any: ...

    @overload
    def __call__(self, **columns: ColumnExpr) -> Any: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The answer. On a table's columns (``fn(col.message)``), a column."""
        from .columns import has_column
        if has_column(args, kwargs):                  # classify(col.text): a column, for dpyr
            return self.vectorize()(*args, **kwargs)
        if self._async:
            return self._in_thread(args, kwargs, full=False)
        return self._invoke(args, kwargs)

    def predict(self, *args: P.args, **kwargs: P.kwargs) -> Prediction:
        '''The call, with everything it produced: every output (``p.result``,
        ``p.reasoning``...), the tokens, the model's replies, the call's id.

        Examples
        --------
        ```python
        @ai
        def solve(question: str) -> float:
            """Solve the word problem."""
            reasoning: str = _ai     # step by step
            return _ai

        p = solve.predict("3 pencils cost $1.20. How much do 10 cost?")
        p.result, p.reasoning
        ```
        '''
        return self._invoke(args, kwargs, full=True)

    async def acall(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """``await fn.acall(...)``: the answer, in async code. The call runs in a
        worker thread, so the event loop is free while the model answers."""
        return await self._in_thread(args, kwargs, full=False)

    async def apredict(self, *args: P.args, **kwargs: P.kwargs) -> Prediction:
        """``await fn.apredict(...)``: ``predict`` in async code."""
        return await self._in_thread(args, kwargs, full=True)

    async def _in_thread(self, args: tuple, kwargs: Dict[str, Any], full: bool) -> Any:
        self._bind_inputs(args, kwargs)               # wrong arguments fail here, in the caller
        return await asyncio.to_thread(self._invoke, args, kwargs, full)   # settings and states go along

    def _invoke(self, args: tuple, kwargs: Dict[str, Any], full: bool = False) -> Any:
        """One call, made here and now, followed in the call log."""
        return calllog.run(self, self._effective(), lambda: self._bind_inputs(args, kwargs),
                           lambda: self._call(args, kwargs, full))

    def _call(self, args: tuple, kwargs: Dict[str, Any], full: bool = False) -> Any:
        inputs = self._bind_inputs(args, kwargs)
        spec = self._spec()
        ctx = _CallContext(program=self, spec=spec, inputs=inputs)
        token = _ACTIVE_CALL.set(ctx)
        try:
            if self._async:                           # the body is the model call: nothing of its own to run
                result = _ai
            else:
                result = self._fn(*args, **kwargs)

            if full:
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

            # Which output each `_ai` left in a returned tuple or list holds: a
            # bare `_ai` is the answer, `x = _ai` is the output `x`.
            names_pool = [spec.main if n is None or n not in spec.outputs else n
                          for n in _return_slots(self._fn)]

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

    def stream(self, *args, **kwargs) -> "streaming.Stream":
        '''Call the function and watch the answer being written.

        The call starts at once, in the background, and is the same call as
        ``fn(...)``: the same retries, tools and call log line, the same value
        in the end. Iterate the stream for the answer's text as it arrives.

        Parameters
        ----------
        *args, **kwargs
            The call's inputs, as for calling the function.

        Returns
        -------
        Stream
            ``for piece in s`` (or ``async for``): the answer's text, piece
            by piece. ``s.result``: the value (waits); ``await s`` in async
            code. ``s.events()``: everything, with the reasoning, tool calls
            and retries. ``s.text``, ``s.partial``: the answer so far.
            ``s.close()``: stop the call.

        See Also
        --------
        Stream : what this returns.

        Examples
        --------
        ```python
        @ai
        def haiku(topic: str) -> str:
            """A haiku about the topic."""
            ...

        for piece in haiku.stream("the first snow"):
            print(piece, end="", flush=True)
        ```

        Everything the model writes, reasoning first:

        ```python
        @ai
        def solve(problem: str) -> float:
            """Solve the word problem."""
            reasoning: str = _ai["Step by step."]
            return _ai

        s = solve.stream("3 pencils cost $1.20. How much do 10 cost?")
        for event in s.events():
            if event.kind == "text":
                print(event.text, end="", flush=True)
        s.result
        ```
        '''
        from .columns import has_column
        if has_column(args, kwargs):
            raise TypeError(f"{self.__name__}.stream watches one call; on columns, use {self.__name__}(col.x)")
        self._bind_inputs(args, kwargs)               # wrong arguments fail here, not in the background
        return streaming.Stream(self, args, kwargs)

    # ----- optimization -----

    def vectorize(self, *, dtype: Any = None, threads: Optional[int] = None, errors: str = "raise"):
        '''This function as a column expression, with options.

        Calling an AI function on a dpyr column (``fn(col.text)``) is the same
        with the defaults. Each distinct input is sent once; answers are
        remembered for the session; the prompt in use now is the one the column
        is computed with.

        Parameters
        ----------
        dtype : optional
            The column type. Default: the return annotation (text when there is
            none).
        threads : int, optional
            How many rows run at once (default 8).
        errors : str
            ``"raise"`` (default): raise after every row ran; running again
            retries only the failures. ``"null"``: a failed row is null.

        Returns
        -------
        function
            Call it on columns: ``fn.vectorize(threads=16)(col.text)``.

        Examples
        --------
        ```python
        from dpyr import read, col

        @ai
        def capital(country: str) -> str:
            """The country's capital city."""
            ...

        read([{"country": "Norway"}, {"country": "Ghana"}]).mutate(
            capital=capital.vectorize(threads=2)(col.country))
        ```
        '''
        from .columns import vectorize_function
        return vectorize_function(self, dtype=dtype, threads=threads, errors=errors)

    def unpack(self, *args: Any, threads: Optional[int] = None, errors: str = "raise", prefix: str = "",
               **kwargs: Any) -> Dict[str, Any]:
        '''One column per field of the answer, to spread into a table.

        For a function whose answer is a record (a dataclass, a pydantic
        model, a TypedDict): ``table.mutate(**fn.unpack(col.text))`` adds
        one column per field. Each distinct row still costs one model call.

        Parameters
        ----------
        *args, **kwargs
            The inputs, as columns (``col.note``) or constants.
        threads : int, optional
            How many rows run at once (default 8).
        errors : str
            ``"raise"`` (default) or ``"null"``, as for ``vectorize``.
        prefix : str
            Put before each column's name (``prefix="ai_"`` → ``ai_species``).

        Returns
        -------
        dict
            ``{field: column expression}``, for ``mutate(**...)``.

        See Also
        --------
        FunctAIFunc.vectorize : the whole answer as one column.

        Examples
        --------
        ```python
        from dataclasses import dataclass
        from dpyr import read, col

        @dataclass
        class Contact:
            name: str
            city: str | None   # None when the text does not say

        @ai
        def contact(text: str) -> Contact:
            """The person the text is about."""
            ...

        people = read([{"text": "Ada Lovelace wrote to us from London."},
                       {"text": "Grace Hopper called."}])
        people.mutate(**contact.unpack(col.text))
        ```
        '''
        from .columns import unpack_function
        return unpack_function(self, args, kwargs, threads=threads, errors=errors, prefix=prefix)

    def __dpyr_vectorize__(self, *, dtype: Any = None, threads: Optional[int] = None, errors: str = "raise",
                           version: str = ""):
        from .columns import vectorize_function
        return vectorize_function(self, dtype=dtype, threads=threads, errors=errors,
                                  version=version)

    def map(self, data: Any, *, num_threads: int = 1):
        '''Run on every row of a table, and return the run table.

        ``evaluate`` without the scoring: the rows, the predictions
        (``pred_<output>``), and each row's ``error``, ``seconds``, tokens and
        ``model``. Needs ``pip install "functai[data]"``.

        Parameters
        ----------
        data : list of dict, or a table
            Anything ``dpyr.read()`` takes; columns named like the parameters
            are the inputs.
        num_threads : int
            How many rows run at once.

        Returns
        -------
        dpyr dataframe

        See Also
        --------
        FunctAIFunc.vectorize : the function as a column expression.

        Examples
        --------
        ```python
        @ai
        def capital(country: str) -> str:
            """The country's capital city."""
            ...

        capital.map([{"country": "Norway"}, {"country": "Ghana"}], num_threads=2)
        ```
        '''
        from .evaluation import evaluate
        return evaluate(self, data, (), num_threads=num_threads).table

    def opt(self, data: Any = None, *, optimizer: Any = None, metric: Any = None, valset: Any = None,
            **opts) -> "FunctAIFunc[P, R]":
        '''An improved copy: its instruction and worked examples chosen from rows
        with known answers. This function is unchanged.

        Only what the function sends besides its inputs changes: the
        instruction and the demos. Code, types and layout are never touched.
        ``functai.labeled_few_shot``, ``functai.bootstrap_few_shot`` and
        ``functai.gepa`` are the common cases, by name.

        Parameters
        ----------
        data : list of dict, or a table
            Rows as for ``evaluate``: columns named like the parameters are the
            inputs, the others the expected outputs.
        expected : str or dict, optional
            The column holding the right answers, as for ``evaluate``:
            ``expected="category"``.
        optimizer : optimizer class or instance
            Default ``BootstrapFewShot``. See the Optimizers section.
        metric : function or dpyr expression
            As for ``evaluate``. Default: exact match on the expected outputs.
        valset : list of dict, or a table
            Rows for optimizers that choose between candidates.
        teacher_lm : str, optional
            A stronger model that runs the examples; its good runs become demos.
        teacher : AI function, optional
            Or a teacher function.
        n_synth : int, optional
            With a teacher: first write this many training rows.
        **opts
            Passed to the optimizer.

        Returns
        -------
        FunctAIFunc
            The improved copy, with its own version.

        See Also
        --------
        evaluate : measure before and after.

        Examples
        --------
        ```python
        from typing import Literal

        @ai
        def category(message: str) -> Literal["shipping", "billing", "product"]:
            """The support category of the message."""
            ...

        train = [
            {"message": "The vase came smashed.", "result": "shipping"},
            {"message": "Money back please, the chair wobbles.", "result": "billing"},
            {"message": "The handle came off after two uses.", "result": "product"},
        ]
        taught = category.opt(train)
        [d.inputs["message"] for d in taught.demos]
        ```
        '''
        from .optimizers import optimize
        states, logs = optimize(self, trainset=data, optimizer=optimizer, metric=metric, valset=valset, **opts)
        return self._improved(states[self], logs[self])

    def _improved(self, state: ProgramState, log: Optional[Dict[str, Any]] = None) -> "FunctAIFunc[P, R]":
        """A copy with this state, and the improvement on its record."""
        copy = self.using()
        copy._state = state
        if log is not None:
            copy._opt_runs = [*self._opt_runs, log]
        return copy

    @property
    def trials(self) -> List[Dict[str, Any]]:
        """What the search that made this copy tried (``GEPA``'s candidates,
        ``InstructionSearch``'s trials), as rows; empty for a function no search made.
        Scores measured on the rows it chose with flatter the one it chose."""
        return list(self._opt_runs[-1].get("trials") or []) if self._opt_runs else []

    def bake(self, data: Any, **options):
        """Train weights that answer this function; returns the baked model.
        ``fast = fn.using(lm=baked)`` runs the function on them. See
        ``functai.bake.bake`` for the options (student, teacher, labels, test, ...)."""
        from .bake import bake
        return bake(self, data, **options)

    def optimization_runs(self) -> List[Dict[str, Any]]:
        """How this function was improved, oldest first: the optimizer, the
        examples, what changed (and, for a search, its ``trials``)."""
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


def _check_baked_signature(fn: "FunctAIFunc", spec: Spec, baked: Any) -> None:
    """A baked model answers the signature it was trained for, or refuses."""
    from .bake.examples import BakeError, head_fields, head_signature
    if baked.kind == "head":
        try:
            signature = head_signature(spec, head_fields(spec))
        except BakeError as exc:
            raise BakeError(f"{fn.__name__} cannot run on the baked model {baked.name!r}: {exc}") from None
    else:
        signature = spec.signature
    if lmcc.signature_fingerprint(signature) != baked.fingerprint:
        was = {f.name: (f.direction, f.type) for f in baked.signature.fields}
        now = {f.name: (f.direction, f.type) for f in signature.fields}
        diff = sorted(set(was.items()) ^ set(now.items()))
        raise BakeError(f"{fn.__name__} has changed since {baked.name!r} was baked (inputs, outputs or their "
                        f"answers differ: {diff[:6]}); bake it again")


# ──────────────────────────────────────────────────────────────────────────────
# Decorator
# ──────────────────────────────────────────────────────────────────────────────


@overload
def ai(_fn: Callable[P, R], /) -> FunctAIFunc[P, R]: ...


@overload
def ai(_fn: None = None, /, **cfg: Any) -> Callable[[Callable[P, R]], FunctAIFunc[P, R]]: ...


def ai(_fn: Any = None, /, **cfg: Any) -> Any:
    '''Turn a typed Python function into an AI function.

    The function's parts are the prompt: its name is the task, the docstring
    the instruction, the parameters the inputs, the return type the output
    (and the type the reply is read back into). Comments on parameters,
    fields and the return line are guidance. A body that is only a
    docstring, ``...`` or ``return _ai`` means "the model's answer is the
    return value"; otherwise ``_ai`` stands for the model's answer inside
    the body. Use it bare (``@ai``) or with settings (``@ai(lm=...)``).

    Parameters
    ----------
    lm : str
        The model: ``"gpt-4.1-mini"``, ``"claude-haiku-4-5"``,
        ``"groq:openai/gpt-oss-120b"``, ``"claude:claude-sonnet-4-5"`` (a
        subscription)... Default: the one set with ``configure``.
    temperature, max_tokens, seed, top_p, stop : optional
        Sampling settings; any lm15 ``Config`` field is accepted.
    module : str
        ``"predict"`` (default), ``"cot"`` (reasoning before the answer:
        the model's thinking channel when it has one), or ``"react"``.
    tools : list of functions
        Typed Python functions the model may call. A call then runs the tool
        loop: at most ``max_steps`` model calls (default 8).
    stateful : bool
        Remember the conversation between calls (the last ``state_window``
        turns, default 5).
    adapter : str or lmcc.Adapter
        The prompt layout: ``"xml"`` (default), ``"chat"``, ``"json"``, or an
        lmcc adapter.
    template : list
        A chat template, ``[system(...), turns(), user(...)]``: write the
        conversation yourself. Replaces ``adapter``.
    examples : list
        Worked examples shown before the question: pairs
        ``("input", "output")`` or rows ``{"text": ..., "result": ...}``.
    retries : int
        How many times an unreadable reply is asked again (default 1).
    api_retries : int
        How many times a provider error is re-sent (default 3).
    log_calls : bool or folder, optional
        Keep this function's calls in the call log (see ``functai.calls``);
        ``False`` keeps them out, whatever ``configure`` says.
    log_content : bool or dict, optional
        ``False``: log only sizes, times and tokens, never the values (for a
        function that sees secrets). ``{"transcript": False}``: every value
        but that input's; ``{"*": False, "question": True}``: only the
        question's. It only removes: a host's ``configure`` or block that
        drops a value wins over the function's own ``True``. A name the
        function has no field for is an error (``LogContentError``).
    observers : list, optional
        Functions (or lists) given each event of this function's calls as
        they happen, in the form a log keeps (``functai.eventlog``), beside
        the host's observers.
    journal : store or Journal, optional
        Where the call tree's events are kept while it runs (a
        ``functai.MemoryStore``, or ``functai.Journal(store,
        required=True)``); only where the host sets none.
    **settings
        Any other setting ``configure`` takes (``api_key``, ``client``,
        ``cache_replies``, ``teacher``, ``optimizer``, ``debug``...). An
        unknown setting is an error.

    Returns
    -------
    FunctAIFunc
        The AI function. Call it like the original; ``fn.predict(...)`` returns
        a ``Prediction`` with every output and the tokens used; ``await
        fn.acall(...)`` in async code (an ``async def`` AI function is awaited
        directly).

    See Also
    --------
    configure : settings for every function at once.
    module : a Python function that calls several AI functions, as one program.

    Examples
    --------
    ```python
    @ai
    def sentiment(text: str) -> str:
        """Is the text 'positive', 'negative' or 'neutral'?"""
        ...

    sentiment("The update broke my favourite feature.")
    ```

    ``_ai`` in the body: ``reasoning: str = _ai`` is one more output, written
    before the answer (its comment describes it); a bare ``_ai`` is the
    answer, and plain Python runs on it.

    ```python
    @ai
    def solve(question: str) -> float:
        """Solve the word problem."""
        reasoning: str = _ai     # step by step, the calculation
        return round(_ai, 2)

    p = solve.predict("3 pencils cost $1.20. How much do 10 cost?")
    p.result, p.reasoning
    ```

    Settings in the decorator:

    ```python
    @ai(lm="gpt-4.1-nano", temperature=0)
    def headline(article: str) -> str:
        """A headline of at most eight words."""
        ...

    headline("The council voted to turn the old rail yard into a park with a pool.")
    ```
    '''
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
