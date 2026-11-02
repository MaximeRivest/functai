"""Plugins: code that changes what programs do, through a small set of
hooks, with every change returned as data and recorded (contract/plugins.md).

    modes = functai.Plugin("modes", version="1.0.0")

    @modes.before_call
    def careful(call):
        return functai.Change(sections=["Answer carefully. Cite the file you read."], tools=["read_note"])

    functai.configure(plugins=[modes])                 # every call; or per conversation, or per program

A hook either **changes** something (it returns a ``Change``, or None for no
opinion) or only **hears** of something (``turn_end``). Changes are data:
"add this section to the instruction", "show these earlier turns", "offer
only these tools", "this tool call may not run". The call's record keeps
every change (``changes``).

Two kinds of change are told apart when a rated call is asked again
(``evaluate``, the optimizers). What a call was shown **of its
conversation** (its earlier turns, and the sections ``context`` hooks gave:
a summary) is part of its input: the row carries it (``earlier``,
``sections``) and it is shown again, with no plugin re-run. How the
**host shaped** the call (``before_call``: instruction, sections, model,
settings, tools) is the environment's: asked again, the call is shaped by
the plugins of the process that asks, never by recorded ones, so an
improved instruction is what is measured. The escape hatch (``request``:
rewrite the provider request) is allowed and recorded: that call cannot be
rebuilt (``replayable`` false, no ``request_hash``).

Built-in features use these hooks and nothing private: ``approve=`` is the
``approval`` plugin (``tool_call``), ``functai.compaction(...)`` keeps a
long conversation short (``turn_end`` and ``context``), and
``functai.delegate(program)`` is a tool that runs another program's
conversation.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import FunctAIError

API = 1                     # the plugin API this FunctAI implements
SUPPORTED = (1,)

# hook → (what it does, the Change fields it may return)
HOOKS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "turn_start": ("change", ("inputs",)),
    "context": ("change", ("keep", "without", "sections")),
    "before_call": ("change", ("instruction", "sections", "lm", "settings", "tools", "stop")),
    "step": ("change", ("tools", "reminder", "replace", "stop")),
    "request": ("change", ()),                      # returns an lm15 Request: the escape hatch
    "tool_call": ("change", ("inputs", "block")),
    "tool_result": ("change", ("output",)),
    "turn_end": ("hear", ()),
}
_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class PluginError(FunctAIError):
    """A plugin refused or failed (contract/plugins.md). ``code`` is one
    of ``plugin-api`` (it targets an API version this FunctAI does not
    implement), ``plugin-hook`` (no such hook), ``plugin-name``,
    ``plugin-change`` (a change this hook cannot make, or a value that does
    not fit), ``plugin-failed`` (its handler raised: the call stops, since
    a change it was meant to make, a redaction say, did not happen),
    ``plugin-stopped`` (a ``before_call`` or ``step`` handler stopped the call,
    with its reason: a limit reached),
    ``plugin-load`` (a file that does not define one). ``plugin`` and
    ``hook`` name where."""

    def __init__(self, code: str, message: str, *, plugin: Optional[str] = None, hook: Optional[str] = None):
        super().__init__(code, message)
        self.plugin = plugin
        self.hook = hook


@dataclasses.dataclass(frozen=True)
class Change:
    """What a hook changes. Each hook accepts some fields (HOOKS); a field it
    does not accept refuses ``plugin-change``.

    - ``instruction``: the instruction itself, replaced for this call (a
      mode's own system prompt) (``before_call``);
    - ``sections``: text added to the instruction, in order (``before_call``,
      ``context``);
    - ``lm``: the model for this call; ``settings``: lm15 settings for it
      (``reasoning``, ``temperature``, ``max_tokens``...) (``before_call``);
    - ``tools``: the names of the tools offered to the model, from the
      function's own (``before_call``);
    - ``keep``: the ids of the earlier turns shown (``context``); ``without``:
      fields left out of earlier turns, a list for every turn or ``{turn id:
      [names]}`` (``context``);
    - ``inputs``: inputs replaced, by name (``turn_start``: the turn's;
      ``tool_call``: the tool's);
    - ``block``: the tool call may not run, and why (``tool_call``);
    - ``output``: the tool's result as the model is shown it: text, or a list
      of lm15 parts (a picture with a caption) (``tool_result``);
    - ``reminder``: text the model is shown after this step's tool results,
      in this request and the ones after (``step``);
    - ``replace``: earlier tool results, by invocation, shown as this from
      this request on: text, or a list of lm15 parts (``step``);
    - ``stop``: the call stops here, and why (``before_call``, ``step``): it
      raises ``PluginError`` (``plugin-stopped``).
    """
    instruction: Optional[str] = None
    sections: Optional[Sequence[str]] = None
    lm: Optional[str] = None
    settings: Optional[Mapping[str, Any]] = None
    tools: Optional[Sequence[str]] = None
    keep: Optional[Sequence[str]] = None
    without: Any = None
    inputs: Optional[Mapping[str, Any]] = None
    block: Optional[str] = None
    output: Any = None
    reminder: Optional[str] = None
    replace: Optional[Mapping[int, Any]] = None
    stop: Optional[str] = None

    def fields(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if getattr(self, f.name) is not None}


class Plugin:
    '''A named, versioned set of hooks.

    Parameters
    ----------
    name : str
        Lower-case letters, digits, ``_`` and ``-``: how its changes and
        entries are named in records.
    version : str
        Its own version, recorded with every change it makes.
    api : int
        The plugin API it was written for (1). A version this FunctAI
        does not implement refuses ``plugin-api``.

    Hooks are registered with decorators (``@ext.tool_call``) or
    ``ext.on("tool_call", fn)``; each is given an event and returns a
    ``Change`` or None.

    Examples
    --------
    ```python
    guard = functai.Plugin("no-deletes", version="1.0.0")

    @guard.tool_call
    def refuse_deletes(tool):
        if tool.name == "delete_file":
            return functai.Change(block="deleting files is not allowed here")

    functai.configure(plugins=[guard])
    ```
    '''

    def __init__(self, name: str, *, version: str = "0.0.0", api: int = API, description: str = ""):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise PluginError("plugin-name", f"a plugin's name is lower-case letters, digits, '_' or '-' "
                                                   f"(at most 64), starting with a letter; not {name!r}")
        if api not in SUPPORTED:
            raise PluginError("plugin-api", f"plugin {name} is written for plugin API {api}; this "
                                                  f"FunctAI implements {', '.join(map(str, SUPPORTED))}",
                                 plugin=name)
        self.name = name
        self.version = str(version)
        self.api = api
        self.description = description
        self.handlers: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, hook: str, fn: Optional[Callable[..., Any]] = None) -> Any:
        """Register ``fn`` for ``hook`` (as a decorator when ``fn`` is left out)."""
        if hook not in HOOKS:
            raise PluginError("plugin-hook", f"{self.name}: there is no hook {hook!r}; hooks: "
                                                   f"{', '.join(HOOKS)}", plugin=self.name, hook=hook)

        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(f):
                raise TypeError(f"{self.name}.{hook}: a handler is a function of one event")
            self.handlers.setdefault(hook, []).append(f)
            return f
        return register(fn) if fn is not None else register

    def turn_start(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """A turn is about to be recorded: may replace its inputs."""
        return self.on("turn_start", fn)

    def context(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Which earlier turns a turn is shown, and sections about them."""
        return self.on("context", fn)

    def before_call(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """An AI function is about to be asked: sections, model, settings, tools; or stop it."""
        return self.on("before_call", fn)

    def step(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """A call with tools is about to ask the model again, after its tools ran:
        the tools offered, a reminder, earlier results shown otherwise; or stop it."""
        return self.on("step", fn)

    def request(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """The escape hatch: rewrite the provider request (the call is then not replayable)."""
        return self.on("request", fn)

    def tool_call(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """A tool is about to run: change its input, block it, or ask a person."""
        return self.on("tool_call", fn)

    def tool_result(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """A tool ran: change what the model is shown."""
        return self.on("tool_result", fn)

    def turn_end(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """A turn ended (hears only; may keep entries in its conversation)."""
        return self.on("turn_end", fn)

    def describe(self) -> Dict[str, Any]:
        """Its manifest, as data: name, version, api, the hooks it uses."""
        return {"name": self.name, "version": self.version, "api": self.api, "description": self.description,
                "hooks": sorted(self.handlers)}

    def __repr__(self) -> str:
        return f"<Plugin {self.name} {self.version}: {', '.join(sorted(self.handlers)) or 'no hooks'}>"


# ------------------------------------------------------------------ loading


_loaded: Dict[Tuple[str, float], Plugin] = {}
_loaded_lock = threading.Lock()


def load_plugin(path: "str | os.PathLike[str]") -> Plugin:
    """A plugin from a Python file that defines ``plugin`` (an
    ``Plugin``). Loading runs the file: load only code you trust. A file is
    read again when it changed (``reload`` after editing it)."""
    p = Path(os.fspath(path)).expanduser().absolute()
    try:
        stamp = p.stat().st_mtime
    except OSError as exc:
        raise PluginError("plugin-load", f"{p}: {exc}") from None
    key = (str(p), stamp)
    with _loaded_lock:
        hit = _loaded.get(key)
        if hit is not None:
            return hit
    name = f"_functai_plugin_{abs(hash(key)):x}"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise PluginError("plugin-load", f"{p} is not a Python file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    ext = getattr(module, "plugin", None)
    if not isinstance(ext, Plugin):
        raise PluginError("plugin-load", f"{p} defines no `plugin = functai.Plugin(...)`")
    with _loaded_lock:
        _loaded[key] = ext
    return ext


def check_setting(value: Any) -> Tuple[Any, ...]:
    """An ``plugins`` setting: a list of Plugins (or files defining one)."""
    if isinstance(value, (Plugin, str, os.PathLike)) or not isinstance(value, (list, tuple)):
        raise TypeError("plugins is a list: plugins=[my_plugin, 'plugins/modes.py']")
    for e in value:
        if not isinstance(e, (Plugin, str, os.PathLike)):
            raise TypeError(f"a plugin is a functai.Plugin or the path of a file that defines one; "
                            f"not {type(e).__name__}")
    return tuple(value)


def _resolve(e: Any) -> Plugin:
    return e if isinstance(e, Plugin) else load_plugin(e)


def in_order(layers: Sequence[Tuple[str, Mapping[str, Any]]]) -> List[Plugin]:
    """The plugins around a call, in the order their handlers run: the
    program's own first, then each enclosing block from the innermost out,
    then ``configure``'s, so the host's handlers see (and have the last word
    on) what the program's did. Within a layer, plugins run in the order it
    lists them. A plugin set in several layers runs once,
    in its outermost one. A host layer's ``program_plugins=False`` drops
    the program's own."""
    vetoed = any(layer.get("program_plugins") is False for w, layer in layers if w != "own")
    seen: set = set()
    kept: List[List[Plugin]] = []
    for where, layer in reversed(layers):                  # outermost first, for "runs in its outermost"
        if vetoed and where == "own":
            kept.append([])
            continue
        mine = []
        for e in layer.get("plugins") or ():
            ext = _resolve(e)
            if id(ext) not in seen:
                seen.add(id(ext))
                mine.append(ext)
        kept.append(mine)
    return [p for layer in reversed(kept) for p in layer]  # innermost layer first, each in the order it lists


# ------------------------------------------------------------------ running hooks


def _json(value: Any) -> Any:
    from .calllog import to_json
    return to_json(value)[0]


def _record(change: Change) -> Dict[str, Any]:
    out = {}
    for k, v in change.fields().items():
        out[k] = list(v) if k in ("sections", "tools", "keep") else _json(dict(v)) if k in ("settings", "inputs") \
            else {str(i): _json(r) for i, r in dict(v).items()} if k == "replace" else _json(v)
    return out


class Applied:
    """The changes made in one place, for the record: ``[{"plugin",
    "version", "hook", "change"}]``."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, ext: Plugin, hook: str, change: Dict[str, Any]) -> None:
        self.items.append({"plugin": ext.name, "version": ext.version, "hook": hook, "change": change})


def run(hook: str, plugins: Sequence[Plugin], event: Any, applied: Applied,
        apply: Callable[[Change], None]) -> None:
    """Every handler of ``hook``, in order: each is given the event as the
    changes before it left it, and its change is checked, recorded, and
    applied. A handler that raises stops the call (``plugin-failed``)."""
    allowed = HOOKS[hook][1]
    for ext in plugins:
        for fn in ext.handlers.get(hook, ()):
            event._ext = ext
            try:
                got = fn(event)
            except FunctAIError:
                raise
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise                               # Cancelled, KeyboardInterrupt, a turn that waits
                raise PluginError("plugin-failed", f"plugin {ext.name} failed in {hook}: "
                                                         f"{type(exc).__name__}: {exc}",
                                     plugin=ext.name, hook=hook) from exc
            finally:
                event._ext = None
            if got is None:
                continue
            if not isinstance(got, Change):
                raise PluginError("plugin-change", f"plugin {ext.name}: {hook} returns a functai.Change "
                                                         f"or None, not {type(got).__name__}",
                                     plugin=ext.name, hook=hook)
            fields = got.fields()
            wrong = sorted(set(fields) - set(allowed))
            if wrong:
                raise PluginError("plugin-change", f"plugin {ext.name}: {hook} cannot change "
                                                         f"{', '.join(wrong)} (it may change "
                                                         f"{', '.join(allowed) or 'nothing'})",
                                     plugin=ext.name, hook=hook)
            if not fields:
                continue
            event._ext = ext                            # a change that stops the call names its plugin
            try:
                apply(got)
            except PluginError:
                raise
            except (TypeError, ValueError) as exc:
                raise PluginError("plugin-change", f"plugin {ext.name}: {hook}: {exc}",
                                     plugin=ext.name, hook=hook) from None
            finally:
                event._ext = None
            applied.add(ext, hook, _record(got))


def _texts(sections: Any) -> List[str]:
    if isinstance(sections, str) or not all(isinstance(s, str) for s in sections):
        raise TypeError("sections is a list of texts")
    return [s for s in sections if s.strip()]


# ------------------------------------------------------------------ events


class _Event:
    _ext: Optional[Plugin] = None
    _turn_run: Any = None

    def entries(self, kind: str) -> List[Dict[str, Any]]:
        """This plugin's entries of a kind on the branch of the turn the call
        runs in (``[]`` outside a conversation)."""
        run = self._turn_run
        if run is None:
            return []
        return run.conv.entries(self._ext.name if self._ext else "", kind, branch=run.turn)

    def remember(self, kind: str, data: Any) -> None:
        """Keep an entry of this plugin at the turn the call runs in (outside a
        conversation, it is kept nowhere: a ``ValueError``)."""
        run = self._turn_run
        if run is None:
            raise ValueError("remember keeps an entry in a conversation; this call runs in none")
        run.conv.remember(self._ext.name if self._ext else "", kind, data, turn=run.turn)


@dataclasses.dataclass
class TurnStart(_Event):
    """A conversation's turn, before it is recorded: ``inputs`` (by name, as
    given), ``conversation`` (its id), ``parent`` (the turn it continues),
    ``program``."""
    inputs: Dict[str, Any]
    conversation: str
    parent: Optional[str]
    program: Any


@dataclasses.dataclass
class ShownTurn:
    """An earlier turn a turn would be shown: ``id``, ``inputs``, ``outputs``,
    the fields already left out (``without``)."""
    id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    without: Tuple[str, ...] = ()


@dataclasses.dataclass
class Context(_Event):
    """What a turn is shown: ``turns`` (the earlier turns the conversation's
    rule picked, in order), ``sections`` so far, ``conversation`` (the
    ``Conversation``), ``parent`` (the turn it continues), ``program``.
    ``entries(kind)`` reads this plugin's entries on the branch."""
    turns: List[ShownTurn]
    sections: List[str]
    conversation: Any
    parent: Optional[str]
    program: Any

    def entries(self, kind: str) -> List[Dict[str, Any]]:
        return self.conversation.entries(self._ext.name if self._ext else "", kind, branch=self.parent)


class Spent:
    """What a call tree spent so far in this process (contract/plugins.md,
    *What a tree spent*): each request a provider answered, as ``(model,
    usage)`` (lm15's token counts, by name); replies from the reply cache and
    replies a resumed turn reuses cost nothing now and are not in it.
    ``requests``, ``usage`` (summed: a counter only when every request
    reported it), ``tokens`` (``total_tokens`` summed, or None when a request
    did not report it), ``seconds`` (since the tree's outermost call started)
    and ``cost(prices)``."""

    def __init__(self, exchanges: Sequence[Tuple[str, Mapping[str, int]]], seconds: float):
        self.exchanges = [(m, dict(u)) for m, u in exchanges]
        self.seconds = seconds

    @property
    def requests(self) -> int:
        return len(self.exchanges)

    @property
    def usage(self) -> Dict[str, int]:
        from .data import sum_usage
        return sum_usage(u for _, u in self.exchanges)

    @property
    def tokens(self) -> Optional[int]:
        return self.usage.get("total_tokens") if self.exchanges else 0

    def cost(self, prices: Mapping[str, Any]) -> Optional[float]:
        """The list-price cost of the requests, from ``prices`` (a model's
        name, as asked, to an lm15 ``InferencePricing`` or a dict of its
        fields): exact or None, never a guess (a model without a price, or a
        request whose counts do not settle its price, makes it None)."""
        total = 0.0
        for model, usage in self.exchanges:
            price = pricing(prices, model)
            if price is None:
                return None
            got = price.estimate(_usage_obj(usage))
            if got is None:
                return None
            total += got
        return total

    def __repr__(self) -> str:
        return f"<Spent {self.requests} requests, {self.tokens} tokens, {self.seconds:.1f}s>"


def pricing(prices: Mapping[str, Any], model: str) -> Any:
    """``prices[model]`` as an lm15 ``InferencePricing``, or None."""
    from lm15.models import InferencePricing
    p = prices.get(model) if prices else None
    if p is None and isinstance(model, str) and ":" in model:
        p = prices.get(model.split(":", 1)[1])          # "anthropic:claude-..." priced by the model's own name
    if p is None:
        return None
    if isinstance(p, InferencePricing):
        return p
    if isinstance(p, Mapping):
        return InferencePricing(**dict(p))
    raise TypeError(f"a price is an lm15 InferencePricing (lm15.models) or a dict of its fields, "
                    f"not {type(p).__name__}")


def _usage_obj(usage: Mapping[str, int]) -> Any:
    import lm15
    names = {f.name for f in dataclasses.fields(lm15.Usage)}
    return lm15.Usage(**{k: v for k, v in usage.items() if k in names})


def spent_of(call: Any) -> Spent:
    """What ``call``'s tree spent so far (its outermost call in this process)."""
    import time as _time
    if call is None:
        return Spent([], 0.0)
    top = call.top
    from .calllog import _id_lock
    with _id_lock:
        exchanges = list(top.spent)
    return Spent(exchanges, max(0.0, _time.time() - top.started))


@dataclasses.dataclass
class BeforeCall(_Event):
    """An AI function about to be asked: ``function`` (its name), ``program``,
    ``inputs`` (bound), ``instruction`` (as it stands: the program's, or a
    replacement), ``lm`` and ``settings`` (as they stand), ``tools`` (the
    names offered), ``all_tools`` (the function's own), ``sections`` so far,
    ``path`` (its place in the call tree: ``support/answer``),
    ``conversation`` and ``turn`` (when it runs in one), ``spent`` (what its
    call tree spent so far: ``Spent``); ``entries(kind)``,
    ``remember(kind, data)``: this plugin's entries in that conversation."""
    instruction: str
    function: str
    program: Any
    inputs: Dict[str, Any]
    lm: Any
    settings: Dict[str, Any]
    tools: List[str]
    all_tools: List[str]
    sections: List[str]
    path: str
    conversation: Optional[str] = None
    turn: Optional[str] = None
    spent: Optional[Spent] = None


@dataclasses.dataclass
class StepResult:
    """A tool result of the call so far, as the model is shown it now:
    ``invocation``, ``name``, ``input``, ``output`` (text, or a list of lm15
    parts in their JSON form), ``pictures`` (how many image parts it holds)."""
    invocation: int
    name: str
    input: Any
    output: Any

    @property
    def pictures(self) -> int:
        return 0 if isinstance(self.output, str) else sum(1 for p in self.output if p.get("type") == "image")


@dataclasses.dataclass
class Step(_Event):
    """A call with tools is about to ask the model again, after the tools of
    its last reply ran: ``function``, ``path``, ``step`` (the number of the
    request about to be made: 2 for the first after tools), ``max_steps``
    (the last request the call may make), ``results`` (every tool result of
    the call so far, oldest first, as shown now: ``StepResult``), ``tools``
    (the names offered now), ``all_tools`` (the names the call was offered),
    ``lm``, ``spent`` (what its call tree spent so far: ``Spent``),
    ``conversation`` and ``turn``; ``entries(kind)``, ``remember(kind, data)``."""
    function: str
    path: str
    step: int
    max_steps: int
    results: List[StepResult]
    tools: List[str]
    all_tools: List[str]
    lm: Any
    spent: Spent
    conversation: Optional[str] = None
    turn: Optional[str] = None

    @property
    def last_step(self) -> bool:
        """Whether this is the last request the call may make (``max_steps``)."""
        return self.step >= self.max_steps


@dataclasses.dataclass
class Request(_Event):
    """The provider request about to be sent (an lm15 ``Request``): a handler
    may return another. ``function``, ``path``."""
    request: Any
    function: str
    path: str


@dataclasses.dataclass
class ToolCall(_Event):
    """A tool about to run: ``name``, ``input`` (as it stands), ``effects``,
    ``path`` (``support/answer/refund``), ``invocation``, ``id``, ``function``
    (the AI function that asked), ``approval`` (the ``Approval`` a person
    would be shown), ``settings`` (the asking call's).

    ``ask(reason=None, decide=None)`` asks a person whether it may run: in a
    conversation the turn waits, saved, and goes on when someone answers; on
    a stream the call waits for ``s.approve()``; a plain call refuses
    (``approval-required``). ``decide`` (a function of the ``Approval``
    returning True, False or a reason) answers in place of a person. Returns
    True, or False (and the person's reason is the result the model sees)."""
    name: str
    input: Any
    effects: Optional[str]
    path: str
    invocation: int
    id: str
    function: str
    approval: Any
    settings: Dict[str, Any]
    _call: Any = None
    _refused: Optional[str] = None

    def ask(self, reason: Optional[str] = None, *, decide: Optional[Callable[[Any], Any]] = None) -> bool:
        from . import tools
        ext = self._ext.name if self._ext else "approval"
        approval = dataclasses.replace(self.approval, input=self.input, plugin=ext, question=reason)
        allowed, why, _by = tools.ask_person(self._call, approval, decide)
        if not allowed:
            self._refused = tools.denial(why)
        return allowed


@dataclasses.dataclass
class ToolResult(_Event):
    """A tool ran: ``name``, ``input``, ``output`` (as it stands: text, or a
    list of lm15 parts in their JSON form when it holds a picture, a sound or
    a document), ``path``, ``invocation``, ``function``."""
    name: str
    input: Any
    output: Any
    path: str
    invocation: int
    function: str


@dataclasses.dataclass
class TurnEnd(_Event):
    """A conversation's turn ended (before its end is recorded, so the next
    turn sees what this hook keeps): ``turn`` (its id), ``state`` (``done``,
    ``failed``, ``stopped``), ``inputs``, ``outputs``, ``conversation`` (the
    ``Conversation``), ``parent``. ``turns()``: the done turns of its branch,
    this one last when it is done. ``remember(kind, data)`` keeps an entry of
    this plugin at this turn; ``entries(kind)`` reads them on the branch."""
    turn: str
    state: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    conversation: Any
    parent: Optional[str]

    def turns(self) -> List[ShownTurn]:
        log = self.conversation._read()
        out = [ShownTurn(st.id, dict(st.record.get("inputs") or {}), dict((st.ended or {}).get("outputs") or {}))
               for st in log.branch(self.parent) if st.state() == "done"]
        if self.state == "done":
            out.append(ShownTurn(self.turn, dict(self.inputs), dict(self.outputs)))
        return out

    def remember(self, kind: str, data: Any) -> None:
        self.conversation.remember(self._ext.name if self._ext else "", kind, data, turn=self.turn)

    def entries(self, kind: str) -> List[Dict[str, Any]]:
        found = self.conversation.entries(self._ext.name if self._ext else "", kind, branch=self.parent)
        mine = [e for e in self.conversation._read().entries
                if e.get("turn") == self.turn and e.get("plugin") == (self._ext.name if self._ext else "")
                and e.get("entry") == kind]
        return found + [{"turn": e.get("turn"), "data": copy.deepcopy(e.get("data")), "at": e.get("at")}
                        for e in mine]


# ------------------------------------------------------------------ what the engine asks


def around(program: Any, extra: Sequence[Tuple[str, Mapping[str, Any]]] = ()) -> List[Plugin]:
    """The plugins around a call of ``program`` made now (``extra``: layers
    to add just outside the program's own, a conversation's)."""
    from .config import layers
    found = layers(getattr(program, "_settings", None) or {})
    if extra:
        found = [found[0], *extra, *found[1:]]
    return in_order(found)


@dataclasses.dataclass
class Shaped:
    """A call as its hooks left it: settings, the instruction (None: the
    program's own), sections (its context's, then the hooks'), the sections
    its context gave (what it was shown of its conversation: a rated row
    carries them), the tools offered, and what was changed."""
    settings: Dict[str, Any]
    instruction: Optional[str]
    sections: List[str]
    context_sections: List[str]
    tools: Optional[List[str]]
    applied: Applied


def before_call(program: Any, inputs: Mapping[str, Any], settings: Dict[str, Any], call: Any) -> Shaped:
    """``before_call`` for an AI function's call: the sections its context
    gives first (a conversation's turn, a row asked again), then each
    handler's."""
    from .config import CONFIG_FIELDS
    applied = Applied()
    given = list(given_sections(program, call))
    sections = list(given)
    all_tools = [getattr(t, "__name__", "") for t in getattr(program, "_tools", [])]
    exts = around(program)
    if not any(e.handlers.get("before_call") for e in exts):
        return Shaped(dict(settings), None, sections, given, None, applied)
    conv = (call.conversation or {}) if call is not None else {}
    turn_run = getattr(call, "turn_run", None) if call is not None else None
    event = BeforeCall(program._spec().signature.instructions, program.__name__, program, dict(inputs),
                       settings.get("lm"),
                       {k: v for k, v in settings.items() if k in CONFIG_FIELDS and v is not None},
                       list(all_tools), list(all_tools), list(sections), _names(call),
                       conversation=conv.get("id") or (turn_run.conv.id if turn_run is not None else None),
                       turn=conv.get("turn") or (turn_run.turn if turn_run is not None else None),
                       spent=spent_of(call if call is not None else _current()))
    event._turn_run = turn_run
    out = dict(settings)
    offered: List[Optional[List[str]]] = [None]
    instruction: List[Optional[str]] = [None]

    def apply(c: Change) -> None:
        if c.instruction is not None:
            if not isinstance(c.instruction, str) or not c.instruction.strip():
                raise TypeError("instruction is the text of an instruction")
            instruction[0] = event.instruction = c.instruction
        if c.sections is not None:
            event.sections.extend(_texts(c.sections))
        if c.lm is not None:
            if not isinstance(c.lm, str) or not c.lm.strip():
                raise TypeError("lm is a model's name")
            out["lm"] = event.lm = c.lm
        if c.settings is not None:
            bad = sorted(set(c.settings) - set(CONFIG_FIELDS))
            if bad:
                raise TypeError(f"settings are lm15 settings ({', '.join(sorted(CONFIG_FIELDS))}), not {bad}")
            out.update(c.settings)
            event.settings.update(c.settings)
        if c.tools is not None:
            names = list(c.tools)
            unknown = sorted(set(names) - set(all_tools))
            if unknown:
                raise ValueError(f"{program.__name__} has no tool {unknown[0]!r} (its tools: "
                                 f"{', '.join(all_tools) or 'none'})")
            offered[0] = event.tools = [n for n in all_tools if n in names]
        if c.stop is not None:
            stop_here(c.stop, event, "before_call", applied, program.__name__, call)

    run("before_call", exts, event, applied, apply)
    return Shaped(out, instruction[0], event.sections, given, offered[0], applied)


def _current() -> Any:
    from .calllog import current
    return current()


def stop_here(reason: Any, event: Any, hook: str, applied: "Applied", function: str, call: Any) -> None:
    """A ``stop`` change: kept with the call's changes, then the call stops
    (``plugin-stopped``)."""
    if not isinstance(reason, str) or not reason.strip():
        raise TypeError("stop is the reason, a text")
    ext = event._ext
    applied.add(ext, hook, {"stop": reason})
    if call is not None:
        call.changes.extend(applied.items)
        applied.items.clear()
    raise PluginError("plugin-stopped", f"{function}: stopped by plugin {ext.name}: {reason}",
                      plugin=ext.name, hook=hook)


def _names(call: Any) -> str:
    if call is None:
        return ""
    return "/".join(p.split("#", 1)[0] for p in (call.path or "").split("/") if p)


def given_sections(program: Any, call: Any) -> List[str]:
    """Sections a call is given before any handler: its turn's (the
    conversation's context) or the row's it is asked again for."""
    from . import conversations
    return conversations.sections_for(program, call)


def request(request_: Any, call: Any, function: str) -> Tuple[Any, bool]:
    """The ``request`` hook (the escape hatch): the request to send, and
    whether a handler replaced it (the call is then not replayable)."""
    if call is None:
        return request_, False
    exts = around(call.program)
    if not any(e.handlers.get("request") for e in exts):
        return request_, False
    import lm15
    event = Request(request_, function, _names(call))
    changed = False
    for ext in exts:
        for fn in ext.handlers.get("request", ()):
            event._ext = ext
            try:
                got = fn(event)
            except Exception as exc:  # noqa: BLE001
                raise PluginError("plugin-failed", f"plugin {ext.name} failed in request: "
                                                         f"{type(exc).__name__}: {exc}",
                                     plugin=ext.name, hook="request") from exc
            finally:
                event._ext = None
            if got is None or got is event.request:
                continue
            if not isinstance(got, lm15.Request):
                raise PluginError("plugin-change", f"plugin {ext.name}: request returns an lm15 Request "
                                                         f"or None", plugin=ext.name, hook="request")
            event.request = got
            changed = True
            call.changes.append({"plugin": ext.name, "version": ext.version, "hook": "request",
                                 "change": {"request": "replaced"}})
    if changed:
        call.replayable = False
    return event.request, changed


class Steering:
    """The ``step`` hook of one call's tool loop (contract/plugins.md): what
    its handlers changed so far (the tools offered, earlier results shown
    otherwise, reminders), and the request it makes next, as a view of the
    call's turn: the turn itself keeps every result as the tool gave it."""

    def __init__(self, call: Any, function: str, max_steps: int, tool_specs: Sequence[Any], lm: Any):
        self.call = call
        self.function = function
        self.max_steps = max_steps
        self.all_tools = [t.name for t in tool_specs]
        self.offered: List[str] = list(self.all_tools)
        self.lm = lm
        self.exts = [e for e in around(call.program) if e.handlers.get("step")] if call is not None else []
        self.results: List[StepResult] = []         # every tool result, as shown now
        self.given: Dict[int, Any] = {}             # invocation -> its result as the tool gave it

    def ran(self, invocation: int, name: str, tool_input: Any, output: Any) -> None:
        self.results.append(StepResult(invocation, name, copy.deepcopy(tool_input), output))
        self.given[invocation] = output

    def before(self, step: int) -> None:
        """Run the ``step`` handlers before request ``step`` (2 or more)."""
        if not self.exts:
            return
        from .engine import check_result
        call = self.call
        turn_run = getattr(call, "turn_run", None)
        conv = call.conversation or {}
        event = Step(self.function, _names(call), step, self.max_steps, copy.deepcopy(self.results),
                     list(self.offered), list(self.all_tools), self.lm, spent_of(call),
                     conversation=conv.get("id") or (turn_run.conv.id if turn_run is not None else None),
                     turn=conv.get("turn") or (turn_run.turn if turn_run is not None else None))
        event._turn_run = turn_run
        applied = Applied()
        by_inv = {r.invocation: r for r in self.results}

        def apply(c: Change) -> None:
            if c.tools is not None:
                names = list(c.tools)
                unknown = sorted(set(names) - set(self.all_tools))
                if unknown:
                    raise ValueError(f"{self.function} was not offered tool {unknown[0]!r} (offered: "
                                     f"{', '.join(self.all_tools) or 'none'})")
                self.offered = event.tools = [n for n in self.all_tools if n in names]
            if c.replace is not None:
                if not isinstance(c.replace, Mapping):
                    raise TypeError("replace is a dict: {invocation: what the model is shown instead}")
                for inv, out in dict(c.replace).items():
                    n = int(inv)
                    if n not in by_inv:
                        raise ValueError(f"{self.function} has no tool result {n} (it has "
                                         f"{', '.join(map(str, by_inv)) or 'none'})")
                    by_inv[n].output = check_result(out, f"replace[{n}]")
                event.results = copy.deepcopy(self.results)
            if c.reminder is not None:
                if not isinstance(c.reminder, str) or not c.reminder.strip():
                    raise TypeError("reminder is a text")
                if not self.results:
                    raise ValueError("a reminder follows a tool result, and the call has none")
                last = self.results[-1]
                last.output = (last.output + "\n\n" + c.reminder if isinstance(last.output, str)
                               else [*last.output, {"type": "text", "text": c.reminder}])
                event.results = copy.deepcopy(self.results)
            if c.stop is not None:
                stop_here(c.stop, event, "step", applied, self.function, call)

        try:
            run("step", self.exts, event, applied, apply)
        finally:
            call.changes.extend(applied.items)

    def view(self, turn: Any) -> Any:
        """The call's turn as the next request shows it: earlier results as the
        handlers left them, and only the tools they offer."""
        import lmcc
        shown = {r.invocation: r.output for r in self.results if r.output is not self.given.get(r.invocation)}
        if not shown and self.offered == self.all_tools:
            return turn
        steps, k = [], 0
        invocations = [r.invocation for r in self.results]
        for s in turn.steps:
            if isinstance(s, lmcc.ToolStep):
                inv = invocations[k] if k < len(invocations) else None
                k += 1
                if inv in shown:
                    out = shown[inv]
                    s = dataclasses.replace(s, output=((({"type": "text", "text": out},) if isinstance(out, str)
                                                        else tuple(dict(p) for p in out))))
            steps.append(s)
        inputs = turn.inputs
        if self.offered != self.all_tools:
            inputs = {**inputs, "tools": [t for t in inputs.get("tools", []) if t.name in self.offered]}
        return dataclasses.replace(turn, steps=tuple(steps), inputs=inputs)


def tool_call(call: Any, approval: Any, settings: Mapping[str, Any]) -> Tuple[Any, Optional[str]]:
    """``tool_call`` for one tool call: (the input it runs with, or None and
    what the model is shown instead). Handlers run in order, then the
    ``approval`` plugin (``approve=``) last, on the input they left: a
    host's rule sees what will run. A handler that raises blocks the tool."""
    exts = [*around(call.program), APPROVAL]
    event = ToolCall(approval.name, copy.deepcopy(approval.input), approval.effects, approval.path,
                     approval.invocation, approval.id, call.function, approval, dict(settings), _call=call)
    event._turn_run = getattr(call, "turn_run", None)
    applied = Applied()
    blocked: List[Optional[str]] = [None]

    def apply(c: Change) -> None:
        if c.inputs is not None:
            if not isinstance(c.inputs, Mapping):
                raise TypeError("inputs is a dict of the tool's inputs")
            event.input = {**(event.input if isinstance(event.input, Mapping) else {}), **dict(c.inputs)}
        if c.block is not None:
            if not isinstance(c.block, str):
                raise TypeError("block is the reason, a text")
            blocked[0] = c.block

    try:
        for ext in exts:
            if blocked[0] is not None or event._refused is not None:
                break
            run("tool_call", [ext], event, applied, apply)
    except PluginError as exc:
        if exc.code != "plugin-failed":
            raise
        from .calllog import _warn_once
        _warn_once(("tool-call", exc.plugin), f"{exc}: the tool does not run")
        blocked[0] = f"a check on this tool call failed ({exc.plugin})"
        applied.items.append({"plugin": exc.plugin, "version": "", "hook": "tool_call",
                              "change": {"block": blocked[0]}})
    call.changes.extend(applied.items)
    if blocked[0] is not None:
        ext = applied.items[-1]["plugin"] if applied.items else "?"
        return None, f"This call was blocked ({ext}): {blocked[0]}"
    if event._refused is not None:
        return None, event._refused
    return event.input, None


def tool_result(call: Any, approval: Any, tool_input: Any, output: Any) -> Any:
    """``tool_result``: what the model is shown of a tool's result."""
    exts = around(call.program)
    if not any(e.handlers.get("tool_result") for e in exts):
        return output
    event = ToolResult(approval.name, tool_input, output, approval.path, approval.invocation, call.function)
    event._turn_run = getattr(call, "turn_run", None)
    applied = Applied()

    def apply(c: Change) -> None:
        from .engine import check_result
        event.output = check_result(c.output, "output")

    run("tool_result", exts, event, applied, apply)
    call.changes.extend(applied.items)
    return event.output


# ------------------------------------------------------------------ the approval plugin (approve=)


APPROVAL = Plugin("approval", version="1.0.0", description="approve=: ask before tools run, as a rule says")


@APPROVAL.tool_call
def _approve(tool: ToolCall) -> Optional[Change]:
    """``approve=`` as a plugin: a rule names which tool calls a person is
    asked about; a function answers in place of a person."""
    from . import tools
    rule = tool.settings.get("approve")
    if not tools.asks(rule, tool.approval):
        return None
    tool.ask(decide=rule if callable(rule) else None)
    return None


__all__ = ["Plugin", "Change", "PluginError", "load_plugin", "HOOKS", "API", "APPROVAL",
           "TurnStart", "Context", "ShownTurn", "BeforeCall", "Request", "ToolCall", "ToolResult", "TurnEnd"]
