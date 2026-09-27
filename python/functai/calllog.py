"""The call log: every call of an AI function or module, one line of JSON in a folder.

    functai.configure(log_calls=True)                # ~/.local/share/functai/calls
    p = team("I was charged twice", all=True)
    functai.rate(p, "wrong", answer="billing")       # a correction is a row of data
    functai.calls(team)                              # what happened, as a table
    functai.rated(team)                              # rows with known answers: evaluate, .opt

Off unless asked for (``log_calls``, or ``$FUNCTAI_LOG_CALLS``, which rat and
Chattering set for the processes they start). The format is the contract in
``contract/calls.md``: the folder is the interface, so a TypeScript program
and a dashboard read and write the same thing.

How a call is followed: ``run`` wraps every call of an AI function or a
module; the call in progress is a ContextVar, so a call made inside it (a
module's steps, a tool, an escalation, ``evaluate``'s threads) is its child.
``engine.send`` reports each model exchange to it and ``_run`` its
prediction. The line is written when the call ends, in the caller's thread,
as one append; writing never raises into the call.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import functools
import getpass
import hashlib
import inspect
import json
import os
import platform
import re
import secrets
import socket
import sys
import textwrap
import threading
import time
import uuid
import warnings
import weakref
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import lmcc

FORMAT = 1
ENV_FOLDER = "FUNCTAI_LOG_CALLS"
ENV_CONTENT = "FUNCTAI_LOG_CONTENT"
ENV_CALLER = "FUNCTAI_CALLER"
MAX_LINE = 8 * 1024 * 1024           # bytes; a longer record loses its messages, then its values
_REPR_MAX = 2000
_OFF = {"", "0", "false", "no", "off"}
_ON = {"1", "true", "yes", "on"}
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_NOTHING = object()


# ------------------------------------------------------------------ ids and times


_id_lock = threading.Lock()
_id_last = [0, 0]                    # last millisecond, counter within it


def new_id() -> str:
    """A UUIDv7 (RFC 9562): 48 bits of Unix milliseconds, a 12-bit counter
    that keeps ids made in the same millisecond in order, 62 random bits."""
    ms = time.time_ns() // 1_000_000
    with _id_lock:
        last, counter = _id_last
        if ms <= last:
            ms, counter = last, counter + 1
            if counter > 0xFFF:
                ms, counter = last + 1, 0
        else:
            counter = secrets.randbits(11)          # headroom for the counter
        _id_last[:] = [ms, counter]
    value = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | counter << 64 | 0b10 << 62 | secrets.randbits(62)
    return str(uuid.UUID(int=value))


def _iso(t: float) -> str:
    """RFC 3339 UTC with exactly six fraction digits, so times sort as text."""
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------------------------ JSON


def canonical(value: Any) -> str:
    """lmcc kernel §3a: keys sorted by code point, no white space, UTF-8, no
    NaN; numbers spelled by §7a (integers in decimal, other numbers as
    ECMAScript writes them: ``1.0`` is ``1``, ``1e-07`` is ``1e-7``), so every
    language writes the same bytes for the same data."""
    parts: List[str] = []
    _write(value, parts)
    return "".join(parts)


def _write(value: Any, out: List[str]) -> None:
    if value is None or value is True or value is False:
        out.append("null" if value is None else "true" if value else "false")
    elif isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, int):
        out.append(str(int(value)))
    elif isinstance(value, float):
        out.append(ecmascript_number(value))
    elif isinstance(value, Mapping):
        out.append("{")
        for i, key in enumerate(sorted(value)):
            if not isinstance(key, str):
                raise TypeError(f"a JSON object's keys are text, not {type(key).__name__}")
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _write(value[key], out)
        out.append("}")
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _write(item, out)
        out.append("]")
    else:
        raise TypeError(f"{type(value).__name__} has no JSON form")


def ecmascript_number(x: float) -> str:
    """ECMAScript's Number::toString of a finite double (lmcc kernel §7a)."""
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{x} has no JSON form")
    if x == 0:
        return "0"
    if x < 0:
        return "-" + ecmascript_number(-x)
    mantissa, _, exp = repr(x).partition("e")          # repr: the shortest digits that round-trip
    whole, _, frac = mantissa.partition(".")
    digits = (whole + frac).lstrip("0")
    n = len(whole.lstrip("0")) + (int(exp) if exp else 0) if whole.strip("0") else \
        (int(exp) if exp else 0) - (len(frac) - len(frac.lstrip("0")))
    digits = digits.rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + digits
    e = n - 1
    sign = "+" if e >= 0 else "-"
    return (digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + sign + str(abs(e)))


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_json(value: Any) -> Tuple[Any, int]:
    """A value as the JSON its type describes, and its size (code points of its
    canonical JSON). A value with no JSON form is described instead."""
    try:
        data = lmcc.turn.to_json(value, where="value")
        return data, len(canonical(data))
    except Exception:  # noqa: BLE001 — anything else is described, never refused
        text = repr(value)
        data = {"$type": type(value).__qualname__,
                "$repr": text if len(text) <= _REPR_MAX else text[:_REPR_MAX - 1] + "…"}
        return data, len(canonical(data))


# ------------------------------------------------------------------ settings


def check_settings(settings: Mapping[str, Any]) -> None:
    """Refuse log settings that could only fail later (called by ``config.check``)."""
    where = settings.get("log_calls")
    if where is not None and not isinstance(where, (bool, str, os.PathLike)):
        raise TypeError(f"log_calls is a folder, True or False, not {type(where).__name__}")
    if isinstance(where, str) and not where.strip():
        raise ValueError("log_calls is a folder, True or False, not an empty text")
    content = settings.get("log_content")
    if content is not None and not isinstance(content, bool):
        raise TypeError(f"log_content is True or False, not {content!r}")
    caller = settings.get("caller")
    if caller is not None:
        if not isinstance(caller, Mapping) or not all(isinstance(k, str) for k in caller):
            raise TypeError("caller is a dict with text keys: {'kind': 'agent', 'conversation': '...'}")
        try:
            canonical(dict(caller))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"caller holds only JSON values (text, numbers, lists, dicts): {exc}") from None


def default_folder() -> Path:
    """Where calls are logged when no folder is named."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "functai" / "calls"


def _env_folder() -> Tuple[bool, Optional[Path]]:
    """(on, folder) as $FUNCTAI_LOG_CALLS says; folder None: the default one."""
    raw = os.environ.get(ENV_FOLDER, "").strip()
    if raw.lower() in _OFF:
        return False, None
    if raw.lower() in _ON:
        return True, None
    return True, Path(raw).expanduser()


_folders: Dict[tuple, Optional[Path]] = {}


def _folder_of(setting: Any) -> Optional[Path]:
    """The folder calls are logged to under this setting, or None when off."""
    if setting is False:
        return None
    key = (setting if isinstance(setting, (str, bool, type(None))) else os.fspath(setting),
           os.environ.get(ENV_FOLDER), os.environ.get("XDG_DATA_HOME"), os.getcwd())
    try:
        return _folders[key]
    except KeyError:
        pass
    if len(_folders) > 64:
        _folders.clear()
    _folders[key] = found = _resolve_folder(setting)
    return found


def _resolve_folder(setting: Any) -> Optional[Path]:
    on, env = _env_folder()
    if setting is None and not on:
        return None
    if setting is None or setting is True:
        folder = env or default_folder()
    else:
        folder = Path(os.fspath(setting)).expanduser()
    return folder.absolute()


def _content_of(setting: Any) -> bool:
    if setting is not None:
        return bool(setting)
    return os.environ.get(ENV_CONTENT, "").strip().lower() not in _OFF - {""}


_env_caller_cache: Dict[str, Dict[str, Any]] = {}


def _env_caller() -> Dict[str, Any]:
    raw = os.environ.get(ENV_CALLER, "")
    if not raw:
        return {}
    hit = _env_caller_cache.get(raw)
    if hit is None:
        try:
            hit = json.loads(raw)
            if not isinstance(hit, dict):
                raise ValueError("not a JSON object")
        except ValueError as exc:
            _warn_once(("caller", raw), f"${ENV_CALLER} is not a JSON object ({exc}); ignored")
            hit = {}
        _env_caller_cache.clear()
        _env_caller_cache[raw] = hit
    return hit


def caller_of(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Who is calling: $FUNCTAI_CALLER, with the ``caller`` setting's keys over it."""
    return {**_env_caller(), **dict(settings.get("caller") or {})}


@contextlib.contextmanager
def part_of(key: str, value: str) -> Iterator[None]:
    """Calls made in this block say they are part of something (``evaluation``,
    ``optimization``) in their ``caller``: they answer known questions, they
    are not use."""
    from .config import effective, scoped
    with scoped(caller={**caller_of(effective()), key: value}):
        yield


def tagged(key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A function whose calls are each ``part_of(key, <a new id>)``."""
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with part_of(key, new_id()):
                return fn(*args, **kwargs)
        return wrapper
    return decorate


@dataclasses.dataclass(frozen=True)
class _Target:
    folder: Path
    content: bool
    caller: Dict[str, Any]


def _target(settings: Mapping[str, Any]) -> Optional[_Target]:
    folder = _folder_of(settings.get("log_calls"))
    if folder is None:
        return None
    return _Target(folder, _content_of(settings.get("log_content")), caller_of(settings))


def reading_folder(folder: Any = None) -> Path:
    """The folder to read: the one given, else the one calls are logged to
    here, else the default one."""
    if folder is not None:
        return Path(os.fspath(folder)).expanduser().absolute()
    from .config import effective
    return _folder_of(effective().get("log_calls")) or _folder_of(True)


_warned: set = set()


def _warn_once(key: Any, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(f"[functai] {message}", stacklevel=4)


# ------------------------------------------------------------------ the call in progress


class Call:
    """One call being made. Only its id and timing exist when nothing is logged."""

    __slots__ = ("id", "parent", "root", "program", "started", "t0", "target", "inputs", "sizes", "pred",
                 "exchanges", "provider")

    def __init__(self, program: Any, parent: Optional["Call"]):
        self.id = new_id()
        self.parent = parent.id if parent is not None else None
        self.root = parent.root if parent is not None else self.id
        self.program = program
        self.started = time.time()
        self.t0 = time.perf_counter()
        self.target: Optional[_Target] = None
        self.inputs: Optional[Dict[str, Any]] = None
        self.sizes: Dict[str, int] = {}
        self.pred: Any = None
        self.exchanges: List[tuple] = []
        self.provider: Optional[str] = None


_CURRENT: ContextVar[Optional[Call]] = ContextVar("functai_call", default=None)
# The stream watching the calls made in this context (functai.streaming), or None.
WATCH: ContextVar[Any] = ContextVar("functai_watch", default=None)


def current() -> Optional[Call]:
    return _CURRENT.get()


def run(program: Any, settings: Mapping[str, Any], inputs: Callable[[], Mapping[str, Any]],
        invoke: Callable[[], Any]) -> Any:
    """Make one call of ``program`` (``invoke()``), followed as a call: an id, a
    parent, and a line in the log when logging is on (``inputs()`` binds the
    arguments to their names; only then)."""
    watch = WATCH.get()
    if watch is not None:
        watch.check()                             # a closed stream starts no new call
    call = Call(program, _CURRENT.get())
    bound: Optional[Mapping[str, Any]] = None
    try:
        call.target = _target(settings)
        if call.target is not None or watch is not None:
            try:
                bound = inputs()
            except TypeError:                     # wrong arguments: the call itself says so
                bound = {}
        if call.target is not None:
            values = {k: to_json(v) for k, v in bound.items()}
            call.sizes = {k: n for k, (_v, n) in values.items()}
            if call.target.content:
                call.inputs = {k: v for k, (v, _n) in values.items()}
    except Exception as exc:  # noqa: BLE001 — logging never stands in the way of a call
        _warn_once(("start", type(exc).__name__), f"calls are not logged: {type(exc).__name__}: {exc}")
        call.target = None
    token = _CURRENT.set(call)
    try:
        if watch is not None:
            watch.started(call, dict(bound or {}))
        out = invoke()
    except BaseException as exc:
        if call.target is not None:
            _finish(call, error=exc)
        if watch is not None:
            watch.ended(call, error=exc)
        raise
    else:
        if call.target is not None:
            _finish(call, returned=out)
        if watch is not None:
            watch.ended(call, value=out)
        return out
    finally:
        _CURRENT.reset(token)


def attach(program: Any, pred: Any) -> None:
    """The prediction ``program`` produced in the call in progress (``_run``)."""
    call = _CURRENT.get()
    if call is not None and call.program is program:
        call.pred = pred
        object.__setattr__(pred, "call_id", call.id)


def route(provider: Optional[str]) -> None:
    """The provider the next exchanges go to (``_call_model``)."""
    call = _CURRENT.get()
    if call is not None:
        call.provider = provider


def exchange(model: str, request: Any, response: Any = None, *, started: float, seconds: float,
             cached: bool = False, error: Optional[BaseException] = None, streamed: bool = False,
             first_delta: Optional[float] = None) -> None:
    """One request of the call in progress and its reply or error (``engine.send``);
    ``first_delta``: seconds to the first piece of a streamed reply."""
    call = _CURRENT.get()
    if call is not None and call.target is not None:
        call.exchanges.append((model, call.provider, started, seconds, cached, request, response, error,
                               streamed, first_delta))


# ------------------------------------------------------------------ versions


_code_hashes: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()


def code_hash(fn: Any) -> str:
    """``sha256:`` of a function's source, dedented and without decorators (as
    ``functai.save`` writes it); of its compiled code when there is no source.
    Read once per function object, so editing the file later does not change it."""
    try:
        return _code_hashes[fn]
    except (KeyError, TypeError):
        pass
    from .graph import _strip_decorators
    try:
        text = _strip_decorators(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError):
        import marshal
        code = getattr(fn, "__code__", None)
        text = "code:" + (hashlib.sha256(marshal.dumps(code)).hexdigest() if code is not None else repr(fn))
    digest = _sha(text)
    try:
        _code_hashes[fn] = digest
    except TypeError:
        pass
    return digest


def _same(a: Any, b: Any) -> bool:
    return a is b or (isinstance(a, str) and a == b)


_ai_facts: "weakref.WeakKeyDictionary[Any, Tuple[tuple, Tuple[str, str, str]]]" = weakref.WeakKeyDictionary()


def ai_facts(fn: Any) -> Tuple[str, str, str]:
    """(version, signature fingerprint, answer's name) of an AI function as it
    is now; worked out again only when something they depend on changed."""
    from . import saved
    settings = fn._effective()
    spec = fn._spec()
    state = fn._current_state()
    refs = (state, spec, fn._template, settings.get("adapter"), settings.get("lm"), fn._tool_specs)
    hit = _ai_facts.get(fn)
    if hit is not None and len(hit[0]) == len(refs) and all(_same(a, b) for a, b in zip(hit[0], refs)):
        return hit[1]
    plan, past = saved.probe_plan(fn, spec, settings)
    try:
        request = saved.request_fingerprint(saved.probe_request(fn, spec, plan, past, saved._sample_inputs(spec)))
    except lmcc.Refusal as exc:
        request = f"refused:{exc.code}"
    version = _sha(canonical(version_document(fn, request)))
    facts = (version, signature_id(spec.signature), spec.main)
    _ai_facts[fn] = (refs, facts)
    return facts


def version_document(fn: Any, request: str) -> Dict[str, str]:
    """What an AI function's version hashes: ``{"request": R}`` when the model
    writes the whole body, ``{"code": C, "request": R}`` when code of its own
    runs beside the model (contract/calls.md, *Versions*). So the same AI
    function written in two languages has one version."""
    body = fn.__wrapped__
    return {"request": request} if _model_body(body) else {"code": code_hash(body), "request": request}


_model_bodies: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()


def _model_body(fn: Any) -> bool:
    """``signature.model_writes_body``, read once per function object (like its code hash)."""
    try:
        return _model_bodies[fn]
    except (KeyError, TypeError):
        pass
    from .signature import model_writes_body
    answer = model_writes_body(fn)
    try:
        _model_bodies[fn] = answer
    except TypeError:
        pass
    return answer


def signature_id(signature: Any) -> str:
    """A call's ``program.signature``: lmcc's signature fingerprint with every
    field's host type name left out (``"type": ""``), so the fields' names,
    directions, purposes and JSON shapes decide it, not how one language
    spells a type (``str``, ``string``, ``character``) (contract/calls.md)."""
    return lmcc.signature_fingerprint(lmcc.SignatureCore(
        signature.instructions, [dataclasses.replace(f, type=None) for f in signature.fields]))


def ai_version(fn: Any) -> str:
    """An AI function's version: ``sha256:`` of ``{"request"}`` or ``{"code",
    "request"}``, where ``request`` is what it sends for a sample input
    (contract/calls.md)."""
    return ai_facts(fn)[0]


_module_graphs: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()


def _module_graph(m: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """({key: code hash} of the plain code, {key: AI function}) a module
    reaches, read once and read again when a name it used is rebound (a
    function redefined in a notebook)."""
    hit = _module_graphs.get(m)
    if hit is not None and all(vars(mod).get(name, _NOTHING) is obj for mod, name, obj in hit[0]):
        return hit[1], hit[2]
    from .core import FunctAIFunc
    from .graph import Analysis
    bindings: List[tuple] = []
    code: Dict[str, str] = {}
    ai: Dict[str, Any] = {}
    try:
        a = Analysis(discover=True)
        a.program(m)
        nodes = list(a.nodes.values())
    except Exception:  # noqa: BLE001 — a program the analysis cannot follow: its own code only
        nodes = []
    for n in nodes:
        key = f"{_original(n.module)}:{n.name}"
        if isinstance(n.obj, FunctAIFunc):
            ai[key] = n.obj
        else:
            code[key] = _sha(n.source) if n.source else code_hash(getattr(n.obj, "_fn", n.obj))
        mod = sys.modules.get(n.module)
        if mod is not None and vars(mod).get(n.name, _NOTHING) is n.obj:
            bindings.append((mod, n.name, n.obj))
    if not nodes:
        code[f"{_original(getattr(m._fn, '__module__', None))}:{m.__name__}"] = code_hash(m._fn)
    _module_graphs[m] = (bindings, code, ai)
    return code, ai


def module_version(m: Any) -> str:
    """A module's version: ``sha256:`` of the code it reaches and the versions
    of the AI functions it calls (contract/calls.md)."""
    code, ai = _module_graph(m)
    return _sha(canonical({"code": code, "ai": {k: ai_version(fn) for k, fn in ai.items()}}))


def _original(module: Optional[str]) -> str:
    from .saved import origin
    return origin(module)[0]


def program_info(program: Any) -> Dict[str, Any]:
    """The ``program`` part of a call record."""
    from .core import FunctAIFunc
    is_ai = isinstance(program, FunctAIFunc)
    fn = program.__wrapped__ if is_ai else program._fn
    from .saved import origin
    module, saved_id = origin(getattr(fn, "__module__", None))
    info: Dict[str, Any] = {"name": program.__name__, "kind": "ai" if is_ai else "module", "module": module}
    if is_ai:
        info["version"], info["signature"], info["answer"] = ai_facts(program)
    else:
        info["version"] = module_version(program)
        info["answer"] = "result"
    if saved_id:
        info["saved"] = saved_id
    code = getattr(fn, "__code__", None)
    if code is not None:
        info["file"], info["line"] = code.co_filename, code.co_firstlineno
    return info


# ------------------------------------------------------------------ the record


_process: Dict[str, Any] = {}


def _process_info() -> Dict[str, Any]:
    pid = os.getpid()
    if _process.get("pid") != pid:
        from . import __version__
        try:
            user = getpass.getuser()
        except Exception:  # noqa: BLE001
            user = None
        _process.clear()
        _process.update(host=socket.gethostname(), pid=pid, user=user, language="python",
                        runtime=platform.python_version(), functai=__version__)
    return dict(_process)


def _error(exc: BaseException, content: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": type(exc).__name__}
    code = getattr(exc, "code", None)
    if isinstance(exc, lmcc.Refusal) and isinstance(code, str):
        out["code"] = code
    if content:
        out["message"] = str(exc)
    return out


def _usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None or not dataclasses.is_dataclass(usage):
        return {}
    out = {}
    for f in dataclasses.fields(usage):
        v = getattr(usage, f.name)
        if isinstance(v, int) and not isinstance(v, bool):
            out[f.name] = v
    return out


def _exchange_record(ex: tuple, content: bool) -> Dict[str, Any]:
    from lm15.serde import request_to_dict, response_to_dict
    model, provider, started, seconds, cached, request, response, error, streamed, first_delta = ex
    out: Dict[str, Any] = {"model": model, "provider": provider, "started": _iso(started),
                           "seconds": round(seconds, 6), "cached": cached}
    if streamed:
        out["streamed"] = True
        out["first_delta"] = round(first_delta, 6) if first_delta is not None else None
    if response is not None:
        out["finish"] = getattr(response, "finish_reason", None)
        out["usage"] = _usage(response)
    if error is not None:
        out["error"] = _error(error, content)
    if content:
        out["request"] = request_to_dict(request)
        if response is not None:
            out["response"] = response_to_dict(response)
    return out


def _record(call: Call, *, returned: Any = _NOTHING, error: Optional[BaseException] = None) -> Dict[str, Any]:
    from .core import FunctAIFunc
    from .data import Prediction
    target = call.target
    content = target.content
    program = program_info(call.program)
    pred = call.pred
    outputs: Optional[Dict[str, Any]] = None
    out_sizes: Dict[str, int] = {}
    shown: Any = _NOTHING
    if isinstance(call.program, FunctAIFunc):
        if pred is not None:
            values = {k: to_json(v) for k, v in pred.items()}
            outputs = {k: v for k, (v, _n) in values.items()}
            out_sizes = {k: n for k, (_v, n) in values.items()}
        if returned is not _NOTHING and not isinstance(returned, Prediction):
            value, _n = to_json(returned)
            if outputs is None or outputs.get(program["answer"], _NOTHING) != value:
                shown = value
    elif error is None:
        value, n = to_json(returned)
        outputs, out_sizes = {"result": value}, {"result": n}
    refusal = getattr(pred, "refusal", None) if pred is not None else None
    failure = error if error is not None else refusal
    answered = [ex for ex in call.exchanges if ex[6] is not None]
    usage: Dict[str, int] = {}
    for ex in answered:
        for k, v in _usage(ex[6]).items():
            usage[k] = usage.get(k, 0) + v
    rec: Dict[str, Any] = {
        "functai_call": FORMAT, "id": call.id, "parent": call.parent, "root": call.root, "program": program,
        "started": _iso(call.started), "seconds": round(time.perf_counter() - call.t0, 6), "content": content,
    }
    if content:
        rec["inputs"] = call.inputs or {}
        rec["outputs"] = outputs
        if shown is not _NOTHING:
            rec["returned"] = shown
    rec["sizes"] = {"inputs": call.sizes, "outputs": out_sizes}
    rec["error"] = _error(failure, content) if failure is not None else None
    rec["model"] = answered[-1][0] if answered else None
    rec["usage"] = usage
    rec["confidence"] = getattr(pred, "confidence", None) if pred is not None else None
    if content and pred is not None and getattr(pred, "probabilities", None):
        rec["probabilities"] = {k: {str(a): float(p) for a, p in d.items()} for k, d in pred.probabilities.items()}
    if pred is not None and getattr(pred, "escalated", False):
        rec["escalated"] = True
    rec["exchanges"] = [_exchange_record(ex, content) for ex in call.exchanges]
    rec["caller"] = dict(target.caller)
    rec["process"] = _process_info()
    return rec


def _line(rec: Dict[str, Any]) -> bytes:
    def dump(r: Dict[str, Any]) -> bytes:
        return (json.dumps(r, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")

    data = dump(rec)
    if len(data) <= MAX_LINE:
        return data
    rec = {**rec, "truncated": True,
           "exchanges": [{k: v for k, v in ex.items() if k not in ("request", "response")} for ex in rec["exchanges"]]}
    data = dump(rec)
    if len(data) <= MAX_LINE:
        return data
    for k in ("inputs", "outputs", "returned", "probabilities"):
        rec.pop(k, None)
    return dump(rec)


def _finish(call: Call, *, returned: Any = _NOTHING, error: Optional[BaseException] = None) -> None:
    try:
        line = _line(_record(call, returned=returned, error=error))
        _writer(call.target.folder).write(line)
    except Exception as exc:  # noqa: BLE001 — logging never stands in the way of a call
        _warn_once((str(call.target.folder), type(exc).__name__),
                   f"could not log a call of {getattr(call.program, '__name__', '?')} to {call.target.folder} "
                   f"({type(exc).__name__}: {exc}); calls go on, unlogged")


# ------------------------------------------------------------------ writing


class _Writer:
    """This process's file in a log folder: one per UTC day, opened for
    appending, readable by its owner only; a forked child opens its own."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.lock = threading.Lock()
        self.fd: Optional[int] = None
        self.day = ""
        self.pid = 0

    def _open(self, day: str, pid: int) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None
        folder = self.folder / day
        os.makedirs(folder, mode=0o700, exist_ok=True)
        host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname()) or "host"
        path = folder / f"{host}-{pid}-{secrets.token_hex(3)}.jsonl"
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0), 0o600)
        self.day, self.pid = day, pid

    def write(self, data: bytes) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self.lock:
            pid = os.getpid()
            if self.fd is None or day != self.day or pid != self.pid:
                self._open(day, pid)
            view = memoryview(data)
            while view:
                view = view[os.write(self.fd, view):]


_writers: Dict[Path, _Writer] = {}
_writers_lock = threading.Lock()


def _writer(folder: Path) -> _Writer:
    with _writers_lock:
        w = _writers.get(folder)
        if w is None:
            w = _writers[folder] = _Writer(folder)
        return w


def _after_fork() -> None:
    """A forked child writes its own files, with locks no thread of the parent holds."""
    global _writers_lock, _id_lock
    _writers_lock, _id_lock = threading.Lock(), threading.Lock()
    for w in _writers.values():
        w.lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


# ------------------------------------------------------------------ reading


def _since(since: Any) -> Optional[_dt.datetime]:
    if since is None:
        return None
    now = _dt.datetime.now(_dt.timezone.utc)
    if isinstance(since, _dt.timedelta):
        return now - since
    if isinstance(since, _dt.datetime):
        return since if since.tzinfo else since.astimezone()
    if isinstance(since, _dt.date):
        return _dt.datetime(since.year, since.month, since.day, tzinfo=_dt.timezone.utc)
    if isinstance(since, str):
        m = re.fullmatch(r"\s*(\d+)\s*([hdw])\s*", since)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            return now - _dt.timedelta(hours=n) * {"h": 1, "d": 24, "w": 24 * 7}[unit]
        try:
            return _since(_dt.datetime.fromisoformat(since.replace("Z", "+00:00")))
        except ValueError:
            pass
    raise ValueError(f"since is a date, a datetime, a timedelta, or text like '2026-09-20', '7d', '12h', '2w'; "
                     f"not {since!r}")


def _files(folder: Path, since: Optional[_dt.datetime]) -> List[Path]:
    if not folder.is_dir():
        return []
    first = since.astimezone(_dt.timezone.utc).strftime("%Y-%m-%d") if since else ""
    out = sorted(folder.glob("*.jsonl"))
    for day in sorted(p for p in folder.iterdir() if p.is_dir() and _DAY.match(p.name)):
        if day.name >= first:
            out += sorted(day.glob("*.jsonl"))
    return out


def read(folder: Any = None, *, since: Any = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(calls, ratings) logged in a folder, as the dicts of their lines. Lines
    that do not parse (one being written) and unknown records are skipped;
    with ``since``, calls and ratings from before it are left out."""
    root = reading_folder(folder)
    start = _since(since)
    cutoff = _iso(start.timestamp()) if start else ""
    calls: List[Dict[str, Any]] = []
    ratings: List[Dict[str, Any]] = []
    for path in _files(root, start):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for raw in data.split(b"\n"):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("functai_call") == FORMAT and rec.get("started", "") >= cutoff:
                calls.append(rec)
            elif rec.get("functai_rating") == FORMAT and rec.get("at", "") >= cutoff:
                ratings.append(rec)
    return calls, ratings


def _order(rec: Dict[str, Any], time_key: str) -> Tuple[str, str]:
    return (rec.get(time_key) or "", rec.get("id") or "")


def current_ratings(ratings: Iterable[Dict[str, Any]], *, by: Optional[str] = None
                    ) -> Dict[str, List[Dict[str, Any]]]:
    """For each call, the ratings that count: each person's latest, none for
    someone who withdrew theirs (verdict null); only ``by``'s when given."""
    latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in ratings:
        if by is not None and r.get("by") != by:
            continue
        key = (r.get("call"), r.get("by"))
        if key not in latest or _order(r, "at") > _order(latest[key], "at"):
            latest[key] = r
    out: Dict[str, List[Dict[str, Any]]] = {}
    for (call, _by), r in latest.items():
        if r.get("verdict") in ("right", "wrong"):
            out.setdefault(call, []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: _order(r, "at"))
    return out


def _says(rating: Dict[str, Any], call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The right values a rating gives ({output: value}), or None when it gives none."""
    answer = (call.get("program") or {}).get("answer") or "result"
    if rating.get("verdict") == "right":
        outputs = call.get("outputs") or {}
        return {answer: outputs[answer]} if answer in outputs else None
    values: Dict[str, Any] = {}
    if "answer" in rating:
        values[answer] = rating["answer"]
    values.update({k: v for k, v in (rating.get("outputs") or {}).items() if k not in values})
    return values or None


_META = 7                              # columns rated_rows adds after the data


def _add_meta(row: Dict[str, Any], meta: Mapping[str, Any]) -> None:
    """Add columns about the row without hiding its data: a name an input or
    output already has gets underscores in front until it is free."""
    for key, value in meta.items():
        while key in row:
            key = "_" + key
        row[key] = value


def matches(call: Dict[str, Any], name: str, module: Optional[str] = None) -> bool:
    program = call.get("program") or {}
    return program.get("name") == name and (module is None or program.get("module") == module)


def rated_rows(calls: Iterable[Dict[str, Any]], ratings: Iterable[Dict[str, Any]], *, name: str,
               module: Optional[str] = None, signature: Optional[str] = None, by: Optional[str] = None
               ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Rows with known answers from rated calls, and how many rated calls were
    left out and why (``other_signature``, ``no_content``, ``no_answer``).
    The rules are contract/calls.md's "Rows with known answers"."""
    counting = current_ratings(ratings, by=by)
    left = {"other_signature": 0, "no_content": 0, "no_answer": 0}
    rows: List[Dict[str, Any]] = []
    for call in sorted((c for c in calls if matches(c, name, module)), key=lambda c: _order(c, "started")):
        rs = counting.get(call.get("id"))
        if not rs:
            continue
        program = call.get("program") or {}
        if signature is not None and program.get("signature") != signature:
            left["other_signature"] += 1
            continue
        if not call.get("content") or "inputs" not in call:
            left["no_content"] += 1
            continue
        said = [(r, _says(r, call)) for r in rs]
        usable = [(r, v) for r, v in said if v is not None]
        if not usable:
            left["no_answer"] += 1
            continue
        rating, values = usable[-1]
        verdicts = {r.get("verdict") for r in rs}
        disputed = len(verdicts) > 1 or len({canonical(v) for _r, v in usable}) > 1
        answer = program.get("answer") or "result"
        row = dict(call.get("inputs") or {})
        row[answer] = values.get(answer)
        row.update({k: v for k, v in values.items() if k != answer})
        if answer not in values:
            del row[answer]
        _add_meta(row, {"call": call.get("id"), "version": program.get("version"), "rating": rating.get("verdict"),
                        "rated_by": rating.get("by"), "origin": rating.get("origin") or "review",
                        "sample": rating.get("sample"), "disputed": disputed})
        rows.append(row)
    return rows, left


# ------------------------------------------------------------------ the public functions


def _program_key(program: Any) -> Tuple[str, Optional[str], Optional[str]]:
    """(name, module, current signature) a program's calls are found by."""
    from .core import FunctAIFunc
    from .module import FunctAIModule
    if isinstance(program, str):
        return program, None, None
    if isinstance(program, FunctAIFunc):
        from .saved import origin
        return (program.__name__, origin(getattr(program.__wrapped__, "__module__", None))[0],
                signature_id(program._spec().signature))
    if isinstance(program, FunctAIModule):
        return program.__name__, _original(getattr(program._fn, "__module__", None)), None
    raise TypeError(f"expected an AI function, a module, or a program's name, not {type(program).__name__}")


def _parse_time(text: Optional[str]) -> Optional[_dt.datetime]:
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def purpose(call: Dict[str, Any]) -> str:
    """Why a call was made: ``"optimization"`` or ``"evaluation"`` (it answered
    known questions) or ``"use"``, from its caller."""
    caller = call.get("caller") or {}
    return "optimization" if "optimization" in caller else "evaluation" if "evaluation" in caller else "use"


def _error_text(err: Optional[Dict[str, Any]]) -> Optional[str]:
    if not err:
        return None
    head = err.get("type") or "Error"
    if err.get("code"):
        head += f" [{err['code']}]"
    return f"{head}: {err['message']}" if err.get("message") else head


def calls(program: Any = None, *, folder: Any = None, since: Any = None):
    '''Every logged call, as a table.

    Parameters
    ----------
    program : AI function, module or name, optional
        Only this program's calls. Its inputs then get a column each, and
        its outputs ``pred_<output>`` columns, as in ``evaluate``'s table.
        Default: every program's, with ``inputs`` and ``outputs`` as JSON
        text.
    folder : str or path, optional
        The log folder. Default: the one calls are logged to here, else the
        default one (``~/.local/share/functai/calls``).
    since : date, datetime, timedelta or text, optional
        Only calls from then on: ``"2026-09-20"``, ``"7d"``, ``"12h"``.

    Returns
    -------
    dpyr dataframe
        One row per call, oldest first: the inputs and ``pred_<output>``
        (for one program; else ``program``, ``inputs`` and ``outputs``),
        ``rating`` (the latest judgement: ``right``, ``wrong``, or null),
        ``error``, ``started``, ``seconds``, ``input_tokens``,
        ``output_tokens``, ``reasoning_tokens``, ``total_tokens`` (Gemini's
        ``output_tokens`` leave its hidden reasoning out; a cost is
        ``total_tokens - input_tokens`` at the output price), ``model``, ``version``, ``purpose`` (``use``, or
        ``evaluation`` and ``optimization`` for calls that answered known
        questions), ``caller`` (who called, as JSON), ``call`` (its id)
        and ``parent`` (the call it ran in).

    See Also
    --------
    rated : the rated calls as rows with known answers.
    rate : judge a call.

    Examples
    --------
    ```python
    import tempfile
    from typing import Literal

    @ai
    def team(message: str) -> Literal["shipping", "billing", "product"]:
        """Which team should answer this customer message?"""

    with functai.configure(log_calls=tempfile.mkdtemp()):
        team("My parcel never came.")
        team("I was charged twice.")
        table = functai.calls(team)
    table.select("message", "pred_result", "rating", "model", "seconds")
    ```
    '''
    from .evaluation import _cell, _dpyr, _tabular
    _dpyr()
    root = reading_folder(folder)
    found, ratings = read(root, since=since)
    name = module = None
    if program is not None:
        name, module, _sig = _program_key(program)
        found = [c for c in found if matches(c, name, module)]
    if not found:
        what = f"of {name} " if name else ""
        raise ValueError(f"no calls {what}logged in {root}" + ("" if root.is_dir() else
                         " (nothing is logged there yet: functai.configure(log_calls=True) turns logging on)"))
    found.sort(key=lambda c: _order(c, "started"))
    latest: Dict[str, str] = {}
    for call_id, rs in current_ratings(ratings).items():
        latest[call_id] = rs[-1]["verdict"]
    inputs: List[str] = []
    outputs: List[str] = []
    if program is not None:
        for c in found:
            inputs += [k for k in (c.get("inputs") or {}) if k not in inputs]
            outputs += [k for k in (c.get("outputs") or {}) if k not in outputs]
    records = []
    for c in found:
        p = c.get("program") or {}
        usage = c.get("usage") or {}
        rec: Dict[str, Any] = {}
        if program is not None:
            rec.update({k: _cell((c.get("inputs") or {}).get(k)) for k in inputs})
            rec.update({f"pred_{k}": _cell((c.get("outputs") or {}).get(k)) for k in outputs})
        else:
            rec["program"] = p.get("name")
            rec["inputs"] = canonical(c["inputs"]) if "inputs" in c else None
            rec["outputs"] = canonical(c["outputs"]) if c.get("outputs") is not None else None
        _add_meta(rec, dict(rating=latest.get(c.get("id")), error=_error_text(c.get("error")),
                            started=_parse_time(c.get("started")), seconds=c.get("seconds"),
                            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                            reasoning_tokens=usage.get("reasoning_tokens"), total_tokens=usage.get("total_tokens"),
                            model=c.get("model"), version=p.get("version"), purpose=purpose(c),
                            caller=canonical(c.get("caller") or {}), call=c.get("id"), parent=c.get("parent")))
        records.append(rec)
    names = list(records[0])
    return _dpyr().read(_tabular(names, records))


def rated(program: Any, *, folder: Any = None, by: Optional[str] = None, since: Any = None):
    '''The calls people rated, as rows with known answers.

    Each row is a call someone judged: its inputs under their names, and
    the right answer under the output's name (the answer it gave, when it
    was marked right; the correction, when it was marked wrong and
    corrected). Hand it to ``evaluate`` or ``.opt`` as it is.

    A call marked wrong without the right answer is left out: it says what
    the answer isn't, not what it is. So are calls made when the program
    had other inputs or outputs (another signature), and calls logged
    without their values; a warning says how many.

    Rated calls are the ones people chose to look at, not a fair sample:
    a score on them says how the program does on those. To measure how
    often it is right, rate a random draw (``rate(..., sample=...)``) and
    keep the rows of that draw (``col.sample == "..."``).

    Parameters
    ----------
    program : AI function, module or name
        Whose calls. Given the function itself, only calls with its current
        signature are used (calls from earlier versions of the prompt are
        kept: their inputs and outputs are the same).
    folder : str or path, optional
        The log folder (default as for ``calls``).
    by : str, optional
        Only this person's ratings. Default: everyone's; when people
        disagree, the latest rating is used and ``disputed`` is true.
    since : date, datetime, timedelta or text, optional
        Only calls and ratings from then on.

    Returns
    -------
    dpyr dataframe
        The inputs, the right answer (named like the output), then
        ``rating`` (``right`` or ``wrong``), ``rated_by``, ``origin``
        (``review`` or ``edit``), ``disputed``, ``sample``, ``version`` and
        ``call``.

    See Also
    --------
    rate : judge a call.
    calls : every logged call.
    evaluate : score a program on these rows.

    Examples
    --------
    ```python
    import tempfile
    from typing import Literal

    @ai
    def team(message: str) -> Literal["shipping", "billing", "product"]:
        """Which team should answer this customer message?"""

    with functai.configure(log_calls=tempfile.mkdtemp()):
        a = team("My parcel never came.", all=True)
        b = team("The chair arrived with a snapped leg.", all=True)
        functai.rate(a, "right")
        functai.rate(b, "wrong", answer="shipping")    # broken on the way: the carrier's fault
        rows = functai.rated(team)
    rows.select("message", "result", "rating")
    ```
    '''
    from .evaluation import _cell, _dpyr, _tabular
    _dpyr()
    root = reading_folder(folder)
    name, module, signature = _program_key(program)
    found, ratings = read(root, since=since)
    rows, left = rated_rows(found, ratings, name=name, module=module, signature=signature, by=by)
    reasons = {"no_answer": "marked wrong without the right answer",
               "other_signature": "made when its inputs or outputs were different",
               "no_content": "logged without their values"}
    dropped = [f"{n} {reasons[k]}" for k, n in left.items() if n]
    if not rows:
        raise ValueError(f"no rated calls of {name} in {root}" + (f" ({'; '.join(dropped)})" if dropped else ""))
    if dropped:
        _warn_once(("rated", name, tuple(left.items())), f"{name}: rated calls left out: " + "; ".join(dropped))
    meta = list(dict.fromkeys(k for r in rows for k in list(r)[-_META:]))          # each row ends with them
    first = ["rating", "rated_by", "origin", "disputed", "sample", "version", "call"]
    meta.sort(key=lambda k: first.index(k.lstrip("_")))
    names = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in meta] + meta
    return _dpyr().read(_tabular(names, [{k: _cell(v) for k, v in r.items()} for r in rows]))


_VERDICTS = {True: "right", False: "wrong", "right": "right", "wrong": "wrong"}


def _call_id(call: Any) -> str:
    from .data import Prediction
    if isinstance(call, Prediction):
        found = getattr(call, "call_id", None)
        if found is None:
            raise ValueError("this prediction has no call id (it was not made by calling an AI function)")
        return found
    if isinstance(call, Mapping) and "call" in call:
        call = call["call"]
    if isinstance(call, str) and _UUID.match(call):
        return call
    raise TypeError("rate what? a prediction (fn(..., all=True)), a call id, or a row of calls()/rated() "
                    f"with a 'call' column; not {call!r}")


def rate(call: Any, verdict: Any = _NOTHING, *, answer: Any = _NOTHING, outputs: Optional[Mapping[str, Any]] = None,
         note: Optional[str] = None, reasons: Iterable[str] = (), by: Optional[str] = None,
         origin: Optional[str] = None, sample: Optional[str] = None, folder: Any = None) -> Dict[str, Any]:
    '''Say whether a call's answer is right, and if not, what it should have been.

    "Right" means correct for this input, not "nice". A correction becomes a
    row of data: ``rated`` gives it back with the inputs, for ``evaluate``
    and ``.opt``. The rating is written to the call log, next to the call.

    Parameters
    ----------
    call : Prediction, call id, or row
        The call: what ``fn(..., all=True)`` returned, its ``call_id``, or a
        row of ``calls()`` or ``rated()``.
    verdict : "right", "wrong", True, False or None
        Is the answer right? None withdraws your earlier rating. May be
        left out when ``answer`` is given (then it is ``"wrong"``).
    answer : optional
        The right answer, in the answer's type (a label, a number, a
        dataclass...).
    outputs : dict, optional
        Right values for other named outputs: ``{"priority": 2}``.
    note : str, optional
        Why, in a sentence.
    reasons : list of str, optional
        Short tags: ``["wrong category"]``.
    by : str, optional
        Who is judging. Default: the caller's ``user``, else this computer's
        account. One person's later rating of a call replaces their earlier one.
    origin : str, optional
        ``"review"`` (default: someone judged the answer) or ``"edit"``
        (someone changed the output while using it).
    sample : str, optional
        The id of a random draw of calls this rating is part of: only a
        random draw measures how often a program is right.
    folder : str or path, optional
        The log folder. Default: the one calls are logged to here.

    Returns
    -------
    dict
        The rating, as written.

    See Also
    --------
    rated : the ratings as rows with known answers.
    calls : every logged call.

    Examples
    --------
    ```python
    import tempfile
    from typing import Literal

    @ai
    def team(message: str) -> Literal["shipping", "billing", "product"]:
        """Which team should answer this customer message?"""

    with functai.configure(log_calls=tempfile.mkdtemp()):
        p = team("I was charged twice for one order.", all=True)
        rating = functai.rate(p, "right")
    rating["verdict"]
    ```
    '''
    call_id = _call_id(call)
    corrected = answer is not _NOTHING or bool(outputs)
    if verdict is _NOTHING:
        if not corrected:
            raise TypeError("say whether the answer is right: rate(p, 'right'), rate(p, 'wrong'), or "
                            "rate(p, answer=...) with the right answer")
        verdict = "wrong"
    if verdict is not None:
        if verdict not in _VERDICTS:
            raise ValueError(f"verdict is 'right', 'wrong', True, False, or None (withdraw); not {verdict!r}")
        verdict = _VERDICTS[verdict]
    if corrected and verdict != "wrong":
        raise ValueError("a right answer needs no correction: give answer= or outputs= only with 'wrong'")
    root = Path(os.fspath(folder)).expanduser().absolute() if folder is not None else None
    from .config import effective
    settings = effective()
    if root is None:
        root = _folder_of(settings.get("log_calls"))
        if root is None:
            raise ValueError("ratings are written to the call log, and no call is logged here: "
                             "functai.configure(log_calls=True), or rate(..., folder=...)")
    if by is None:
        by = caller_of(settings).get("user") or _process_info().get("user") or "unknown"
    rec: Dict[str, Any] = {"functai_rating": FORMAT, "id": new_id(), "call": call_id, "at": _iso(time.time()),
                           "by": str(by), "verdict": verdict}
    if answer is not _NOTHING:
        rec["answer"] = to_json(answer)[0]
    if outputs:
        rec["outputs"] = {str(k): to_json(v)[0] for k, v in outputs.items()}
    if reasons:
        rec["reasons"] = [str(r) for r in reasons]
    if note:
        rec["note"] = str(note)
    if origin:
        rec["origin"] = str(origin)
    if sample:
        rec["sample"] = str(sample)
    _writer(root).write((json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    return rec


__all__ = ["calls", "rated", "rate", "default_folder"]
