"""``@module``: a plain Python function that calls @ai functions, optimized as one program.

    @module
    def research(claim: str, hops: int = 2) -> list[str]:
        facts = []
        for _ in range(hops):
            query = generate_query(claim, facts)
            facts = append_notes(claim, facts, search(query))
        return facts

    better = research.opt(rows, metric=...)   # a copy with generate_query and append_notes tuned together
    research.interface                        # what it takes and gives, as data (checked on every call)

The metric sees ``Prediction(result=<what the module returned>)``.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .core import FunctAIFunc
from .errors import InterfaceError

# The settings a module takes for itself: where its calls go and what is kept
# of them. Model settings belong to the AI functions it calls (or to a block).
MODULE_SETTINGS = ("log_calls", "log_content", "caller", "observers", "journal")


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
    """A Python function that calls AI functions, as one program: called,
    streamed, evaluated, optimized and saved as a whole. Build with ``@module``.

    Its ``interface`` (what it takes and gives, as data) is checked on every
    call: its inputs before its code runs, its outputs when it returns
    (``InterfaceError``)."""

    def __init__(self, fn: Callable[..., Any], *, requires: Any = (), interface: Optional[Mapping[str, Any]] = None,
                 outputs: Optional[Mapping[str, Any]] = None, **settings: Any):
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
        # the improved states of the AI functions it calls, for an improved copy
        # (applied while it runs; a candidate state being tried still wins)
        self._states: Dict[FunctAIFunc, Any] = {}
        other = sorted(set(settings) - set(MODULE_SETTINGS))
        if other:
            raise TypeError(f"@module on {self.__name__}: {other} are settings of the AI functions it calls (set "
                            f"them on those, or in a `with functai.configure(...)` block around the call); a module "
                            f"takes {list(MODULE_SETTINGS)}")
        from .config import check
        self._settings: Dict[str, Any] = check({k: v for k, v in settings.items() if v is not None},
                                               f"@module on {self.__name__}")
        if interface is not None and outputs is not None:
            raise TypeError(f"@module on {self.__name__}: give interface= (everything, as data) or outputs=, not both")
        self._declared = interface is not None
        self._outputs = dict(outputs) if outputs is not None else None
        self._interface: Optional[Dict[str, Any]] = None
        if interface is not None:
            self._interface = copy.deepcopy(dict(interface))
        self._signature = inspect.signature(fn)
        self._check_definition()

    # ----- the interface -----

    def _derive(self) -> Dict[str, Any]:
        from . import interface as _interface
        if self._interface is None:
            self._interface = _interface.of_function(self._fn, outputs=self._outputs)
            _interface.check(self._interface, program=self.__name__)
        return self._interface

    def _check_definition(self) -> None:
        """Refuse, when the module is defined, an interface every language
        would refuse, a declared one its code cannot take, and a log_content
        map naming a field it lacks. A name its annotations use that is not
        defined yet is looked up at the first call instead."""
        from . import interface as _interface
        if self._declared:
            _interface.check(self._interface, program=self.__name__)
            self._check_code_takes(self._interface)
        else:
            try:
                self._derive()
            except NameError:
                self._interface = None            # a forward reference: derived when first used
        content = self._settings.get("log_content")
        if isinstance(content, dict) and self._interface is not None:
            from .calllog import check_log_content
            ins, outs, _added = self._fields()
            check_log_content(content, [*ins, *outs], program=self.__name__)

    def _check_code_takes(self, iface: Mapping[str, Any]) -> None:
        params = self._signature.parameters
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        names = {f["name"] for f in iface["inputs"]}
        for f in iface["inputs"]:
            p = params.get(f["name"])
            if (p is None and not takes_any) or (p is not None and p.kind is inspect.Parameter.POSITIONAL_ONLY):
                raise TypeError(f"@module on {self.__name__}: its interface has the input {f['name']!r}, which its "
                                f"code does not take by name")
        for name, p in params.items():
            if p.default is inspect.Parameter.empty and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                                    inspect.Parameter.KEYWORD_ONLY) \
                    and name not in names:
                raise TypeError(f"@module on {self.__name__}: its code needs {name!r}, which its interface does not "
                                f"give")

    @property
    def interface(self) -> Dict[str, Any]:
        """What the module takes and gives, as data (contract/programs.md).

        Derived from the function: each parameter an input (``Any``,
        ``object`` or no annotation: opaque, any value, never checked;
        ``functai.JSON``: any JSON value; a default makes it optional), the
        return annotation one output, ``result`` (``outputs={...}`` declares
        several). Or declared whole with ``interface={...}``."""
        return copy.deepcopy(self._derive())

    def _fields(self) -> Tuple[List[str], List[str], List[str]]:
        iface = self._derive()
        return [f["name"] for f in iface["inputs"]], [f["name"] for f in iface["outputs"]], []

    def _given(self, args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """The inputs a call gives, by name, as the interface names them."""
        if self._declared:
            names = [f["name"] for f in self._interface["inputs"]]     # type: ignore[index]
            if len(args) > len(names):
                raise TypeError(f"{self.__name__}() takes {len(names)} inputs, {len(args)} were given")
            given = dict(zip(names, args))
            for k, v in kwargs.items():
                if k in given:
                    raise TypeError(f"{self.__name__}() got two values for {k!r}")
                given[k] = v
            return given
        bound = self._signature.bind(*args, **kwargs)
        given = {}
        for name, value in bound.arguments.items():
            kind = self._signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_POSITIONAL:
                given[name] = list(value)
            elif kind is inspect.Parameter.VAR_KEYWORD:
                given[name] = dict(value)
            else:
                given[name] = value
        return given

    def _invoke_checked(self, args: tuple, kwargs: Dict[str, Any]) -> Any:
        """Check the inputs, run the code, check what it returned."""
        from . import interface as _interface
        iface = self._derive()
        given = self._given(args, kwargs)
        own_defaults = [n for n, p in self._signature.parameters.items() if p.default is not inspect.Parameter.empty]
        checked = _interface.bind_inputs(iface, given, program=self.__name__, has_default=own_defaults)
        if self._declared:
            out = self._invoke_original(**checked)
        else:
            out = self._invoke_original(*args, **kwargs)
        _interface.check_outputs(iface, out, program=self.__name__)
        return out

    def _own_states(self):
        """Run with this copy's states, under any states already set (an optimizer's candidates)."""
        from .core import _STATE_OVERRIDE
        from .evaluation import with_states
        taken = _STATE_OVERRIDE.get()
        return with_states({fn: st for fn, st in self._states.items() if id(fn) not in taken})

    def _with_states(self, states: Dict[FunctAIFunc, Any]) -> "FunctAIModule":
        copy = object.__new__(FunctAIModule)
        copy.__dict__.update(self.__dict__)
        copy._states = {**self._states, **states}
        copy.history, copy._vectorized = [], {}
        return copy

    def __call__(self, *args, **kwargs):
        from .columns import has_column
        if has_column(args, kwargs):                  # research(col.claim): a column, for dpyr
            return self.vectorize()(*args, **kwargs)
        from . import calllog
        from . import interface as _interface
        from .config import effective

        def inputs():
            return _interface.recorded_inputs(self._derive(), self._given(args, kwargs))

        with self._own_states():
            return calllog.run(self, effective(self._settings), inputs, lambda: self._invoke_checked(args, kwargs))

    def stream(self, *args, **kwargs):
        """Call the module and watch every AI function it calls, as it works.

        The call starts at once, in the background, and is the same call as
        ``module(...)``. ``s.events()`` shows each call inside it (started,
        its text as it is written, tool calls, retries, done);
        ``s.text_of(fn)`` one AI function's answer as it is written;
        ``s.result`` what the module returned (waits). See ``Stream``."""
        from . import streaming
        self._given(args, kwargs)                              # wrong arguments fail here
        return streaming.Stream(self, args, kwargs)

    @property
    def version(self) -> str:
        """The module's version: a fingerprint of its code and its AI functions.

        ``sha256:`` of the code it reaches and of the versions of the AI
        functions it calls, so optimizing one of them is a new version of the
        module."""
        from . import calllog
        with self._own_states():
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

    def opt(self, data: Any, *, metric: Any = None, optimizer: Any = None,
            call_defaults: Optional[Dict[str, Any]] = None, valset: Any = None,
            expected: Any = None, **optimizer_kwargs) -> "FunctAIModule":
        """An improved copy: every @ai function this module calls tuned against
        one metric on the module's output. This module and its functions are
        unchanged. ``call_defaults`` fill module arguments the rows lack."""
        from .optimizers import optimize
        with self._own_states():
            states, _logs = optimize(self, trainset=data, optimizer=optimizer, metric=metric, valset=valset,
                                     call_defaults=call_defaults, expected=expected, **optimizer_kwargs)
        return self._with_states(states)

    def state(self) -> Dict[str, Any]:
        """The instruction and demos each AI function runs with in this module, by name."""
        with self._own_states():
            return {name: fn.state() for name, fn in self.named_ai_functions().items()}

    def save(self, path) -> None:
        """Every AI function's instruction and demos, in one JSON file."""
        import json
        from pathlib import Path
        data = {"functai": 1, "module": self.__name__,
                "functions": {name: st.to_dict() for name, st in self.state().items()}}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))

    def load(self, path) -> "FunctAIModule":
        """A copy running with the states a ``save`` wrote."""
        import json
        from pathlib import Path
        from .core import ProgramState
        data = json.loads(Path(path).read_text())
        fns = self.named_ai_functions()
        return self._with_states({fns[name]: ProgramState.from_dict(state)
                                  for name, state in (data.get("functions") or {}).items() if name in fns})

    def _invoke_original(self, *args, **kwargs):
        out = self._fn(*args, **kwargs)
        self.history.append({"args": args, "kwargs": kwargs, "output": out})
        del self.history[:-100]
        return out


def module(fn: Callable[..., Any] | None = None, *, requires: Any = (), interface: Optional[Mapping[str, Any]] = None,
           outputs: Optional[Mapping[str, Any]] = None, **settings: Any):
    '''Make a Python function that calls AI functions into one program.

    The body is ordinary Python: loops, ifs, helpers, several AI functions.
    As a module it can be evaluated, optimized (each AI function inside
    learns from the runs the metric accepts), run on a table, and saved as
    one program. Use it bare (``@module``) or with requirements.

    Its interface (``blurb.interface``) is derived from the function and
    checked on every call: inputs before the code runs, outputs when it
    returns (``InterfaceError``). ``Any``, ``object`` or no annotation is an
    opaque field (any value, never checked, for data frames and the like);
    ``functai.JSON`` is any JSON value; a parameter with a default is
    optional.

    Parameters
    ----------
    requires : list of str
        Packages the program needs that functai cannot see from the code
        (``["numpy>=2"]``), for ``functai.save``.
    outputs : dict, optional
        Several outputs, by name and type: ``outputs={"team": str, "minutes":
        int, "result": Reply}``; the code returns a dict of them. The last is
        the answer.
    interface : dict, optional
        The whole interface as data (contract/programs.md), instead of
        deriving it; the code is then called with the inputs by keyword.
    log_calls, log_content, caller, observers, journal
        The call log and receiver settings, for this module's calls (as for
        ``@ai``). ``log_content={"transcript": False}`` keeps an input out of
        the log.

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
        ...

    @ai
    def shorten(text: str) -> str:
        """The text in at most twelve words."""
        ...

    @module
    def blurb(topic: str) -> str:
        return shorten(draft(topic))

    blurb("why paired comparisons need fewer examples")
    ```
    '''
    if fn is None:
        return lambda real_fn: FunctAIModule(real_fn, requires=requires, interface=interface, outputs=outputs,
                                             **settings)
    return FunctAIModule(fn, requires=requires, interface=interface, outputs=outputs, **settings)
