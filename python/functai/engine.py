"""One call of an AI function: lay it out (lmcc), send it (lm15), read the reply
(lmcc), run tools until the model answers, and keep a record.

What this module owns, and nothing else: the tool loop, re-asking once after
an unreadable reply, re-sending after a transient provider error, the reply
cache, and the call records ``phistory`` prints. No prompt text is written
here; every byte of the request comes from the plan.
"""

from __future__ import annotations

import collections
import contextvars
import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import lm15
import lmcc
import lmcc_lm15
from lm15.serde import message_to_dict, request_to_dict, response_to_dict
from lmcc_std.tools import Tool, ToolCall

from . import calllog
from .config import CONFIG_FIELDS
from .data import Prediction
from .signature import Spec, coerce, mismatch, shape_of


class LoginRequired(RuntimeError):
    """No usable credential for the model's provider: sign in or pass a key.
    ``.provider`` names it; ``__cause__`` is lm15's error."""

    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(message)
        self.provider = provider


_SIGN_IN_REASONS = {"login_required", "credential_rejected", "indeterminate", "connection_changed"}


def _login_error(exc: Exception) -> Optional[LoginRequired]:
    """lm15's credential errors, said the functai way (what to type next)."""
    from .accounts import ACCOUNTS, CLI_LOGINS
    if isinstance(exc, lm15.AuthOperationError) and getattr(exc, "reason", None) not in _SIGN_IN_REASONS:
        return None
    if not isinstance(exc, (lm15.MissingCredentialError, lm15.AuthOperationError, lm15.AuthError)):
        return None
    if isinstance(exc, lm15.AuthError) and getattr(exc, "status", None) not in (None, 401):
        return None          # 403 and the like: the credential works but is not allowed this (a spent key)
    provider = getattr(exc, "provider", None)
    friendly = next((alias for alias, p in (("claude", "claude-code"), ("chatgpt", "openai-codex"),
                                            ("copilot", "github-copilot"), ("kimi", "kimi-code"))
                     if p == provider), provider)
    is_account = provider in {p for p, _ in ACCOUNTS}
    lines = [f"{provider}: {exc}"]
    if provider:
        lines.append(f"Sign in with functai.login({friendly!r})" if is_account
                     else f"Save a key with functai.login({friendly!r}), set "
                          + " or ".join(f"${k}" for k in (getattr(exc, 'env_keys', ()) or ())[:1] or ("its API key",))
                          + ", or pass api_key=...")
        if provider in CLI_LOGINS:
            lines.append(f"(or sign in to the {CLI_LOGINS[provider][0]} on this machine)")
    return LoginRequired("\n".join(lines), provider)


class StepLimit(RuntimeError):
    """The tool loop reached ``max_steps`` without an answer. ``.turn`` is the turn so far."""

    def __init__(self, message: str, turn: lmcc.Turn):
        super().__init__(message)
        self.turn = turn


# ------------------------------------------------------------------ records


@dataclasses.dataclass
class CallRecord:
    """One request sent (or answered from the cache) and its reply."""
    function: str
    model: str
    request: Any                      # lm15.Request
    response: Any = None              # lm15.Response, None when the provider raised
    cached: bool = False
    error: Optional[str] = None
    timestamp: float = dataclasses.field(default_factory=time.time)


_RECORDS: "collections.deque[CallRecord]" = collections.deque(maxlen=500)
_RECORDS_LOCK = threading.Lock()


def _record(rec: CallRecord) -> None:
    with _RECORDS_LOCK:
        _RECORDS.append(rec)


def inspect_history(n: int = 1) -> List[CallRecord]:
    """The last ``n`` requests functai sent (or answered from its cache), oldest first."""
    with _RECORDS_LOCK:
        return list(_RECORDS)[-max(0, int(n)):] if n else []


def clear_history() -> None:
    with _RECORDS_LOCK:
        _RECORDS.clear()


class _Text(str):
    """A string that shows itself unquoted at a REPL prompt."""

    def __repr__(self) -> str:
        return str(self)


def _part_text(part: Any) -> str:
    kind = getattr(part, "type", None) or type(part).__name__
    text = getattr(part, "text", None)
    if kind == "text" or (text is not None and kind in ("TextPart",)):
        return text or ""
    if kind == "thinking" or type(part).__name__ == "ThinkingPart":
        return f"[thinking]\n{text or ''}"
    if kind == "tool_call" or type(part).__name__ == "ToolCallPart":
        args = getattr(part, "input", None)
        return f"[tool call {getattr(part, 'name', '?')}({json.dumps(args, ensure_ascii=False, default=str)})]"
    if kind == "data" or type(part).__name__ == "DataPart":
        text = json.dumps(getattr(part, "value", None), ensure_ascii=False, default=str)
        probs = getattr(part, "probabilities", None) or {}
        for field, dist in probs.items():
            top = sorted(dist.items(), key=lambda kv: -kv[1])[:3]
            text += f"\n  {field}: " + ", ".join(f"{k} {p:.3f}" for k, p in top) + \
                (f"  ({getattr(part, 'method', '')})" if getattr(part, "method", None) else "")
        return text
    if kind == "tool_result" or type(part).__name__ == "ToolResultPart":
        content = getattr(part, "content", None)
        inner = " ".join(_part_text(p) for p in content) if isinstance(content, (list, tuple)) else str(content)
        return f"[tool result {getattr(part, 'id', '')}] {inner}"
    return f"[{kind}]" + (f" {text}" if text else "")


def _message_text(message: Any) -> str:
    return "\n".join(_part_text(p) for p in getattr(message, "parts", ()) or ())


def phistory(n: int = 1) -> _Text:
    '''The last model calls, as readable text: what was sent, what came back.

    Parameters
    ----------
    n : int
        How many calls, most recent last.

    Returns
    -------
    text
        Every message of each call, the reply, the finish reason and the
        tokens. Displays as is at a notebook prompt; ``print`` it elsewhere.

    See Also
    --------
    inspect_history : the same calls as lm15 request and response objects.
    FunctAIFunc.render : the request a call would send, without sending it.

    Examples
    --------
    ```python
    @ai
    def capital(country: str) -> str:
        """The country's capital city."""
        ...

    capital("Japan")
    print(phistory())
    ```
    '''
    out: List[str] = []
    for rec in inspect_history(n):
        when = _dt.datetime.fromtimestamp(rec.timestamp).isoformat(timespec="seconds")
        head = f"[{when}] {rec.function} → {rec.model}" + (" (from cache)" if rec.cached else "")
        lines = [head, ""]
        req = rec.request
        if getattr(req, "system", None):
            system = req.system if isinstance(req.system, str) else _message_text(req.system)
            lines += ["System message:", "", system, ""]
        for m in getattr(req, "messages", ()) or ():
            lines += [f"{str(m.role).capitalize()} message:", "", _message_text(m), ""]
        tools = getattr(req, "tools", None)
        if tools:
            lines += ["Tools: " + ", ".join(getattr(t, "name", "?") for t in tools), ""]
        if rec.response is not None:
            lines += ["Response:", "", _message_text(rec.response.message), ""]
            usage = rec.response.usage
            lines.append(f"(finish: {rec.response.finish_reason}; tokens in {usage.input_tokens}, "
                         f"out {usage.output_tokens})")
        elif rec.error:
            lines += ["Error:", "", rec.error]
        out.append("\n".join(lines).rstrip())
    return _Text(("\n\n" + "─" * 60 + "\n\n").join(out) if out else "(no model calls yet)")


# ------------------------------------------------------------------ cache


class _Cache:
    """Replies by request, in memory: an identical request (model, messages,
    settings) gets the same reply without a new model call. A reply that
    could not be read is forgotten (``discard``), so a passing failure never
    becomes a permanent one."""

    def __init__(self, capacity: int = 20_000):
        self.capacity = capacity
        self._data: "collections.OrderedDict[str, Any]" = collections.OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(request: Any) -> str:
        data = json.dumps(request_to_dict(request), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any:
        with self._lock:
            hit = self._data.get(key)
            if hit is not None:
                self._data.move_to_end(key)
            return hit

    def put(self, key: str, response: Any) -> None:
        with self._lock:
            self._data[key] = response
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def discard(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


CACHE = _Cache()


def clear_cache() -> None:
    """Forget every cached reply."""
    CACHE.clear()


# ------------------------------------------------------------------ sending


# functai.save(record=...) records every model exchange here, to replay it in verify.
RECORDING: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar("functai_recording",
                                                                                        default=None)


def _remember(request: Any, response: Any) -> None:
    rec = RECORDING.get()
    if rec is not None:
        rec["exchanges"].append({"request": request_to_dict(request), "response": response_to_dict(response)})


def send(router: Any, request: Any, *, function: str, model: str, settings: Dict[str, Any],
         plan: Optional[lmcc.Plan] = None, request_hash: Optional[str] = None) -> Any:
    """One model call: from the cache when allowed, else through the router,
    re-sent after transient errors (rate limit, 5xx, timeout) with backoff.
    Each attempt is an exchange of the call, and begins a ``request`` event.
    When the call is watched (a stream, an observer or a journal sees it),
    the reply is streamed and shown field by field as it arrives (``plan``
    reads it); what is returned is the same whole reply. ``request_hash``:
    lmcc's hash of the rendered request this one was made from."""
    call = calllog.current()
    watched = call is not None and call.watched and plan is not None
    use_cache = bool(settings.get("cache_replies"))
    key = CACHE.key(request) if use_cache else None
    if key is not None:
        hit = CACHE.get(key)
        if hit is not None:
            if call is not None:
                call.request(model)
            _record(CallRecord(function, model, request, hit, cached=True))
            _remember(request, hit)
            calllog.exchange(model, request, hit, started=time.time(), seconds=0.0, cached=True,
                             request_hash=request_hash)
            if watched:
                from . import streaming
                streaming.replay(call, plan, hit)
            return hit
    retries = max(0, int(settings.get("api_retries") or 0))
    first: Optional[float] = None
    for attempt in range(retries + 1):
        if call is not None:
            call.request(model)
        started, t0 = time.time(), time.perf_counter()
        try:
            if watched:
                from . import streaming
                response, first = streaming.request(call, router, request, plan)
            else:
                response = router.complete(request)
            break
        except lm15.RETRYABLE_ERRORS as exc:
            calllog.exchange(model, request, None, started=started, seconds=time.perf_counter() - t0, error=exc,
                             streamed=watched, request_hash=request_hash)
            if attempt == retries:
                _record(CallRecord(function, model, request, None, error=f"{type(exc).__name__}: {exc}"))
                raise
            wait = getattr(exc, "retry_after", None)
            wait = float(wait) if isinstance(wait, (int, float)) and wait > 0 \
                else min(30.0, 2 ** attempt) * (0.5 + random.random())
            if call is not None:
                call.emit("retry", reason=f"the provider failed ({type(exc).__name__}); sending again in "
                                          f"{wait:.1f} s", wait=wait)
                call.sleep(wait)
            else:
                time.sleep(wait)
        except Exception as exc:
            calllog.exchange(model, request, None, started=started, seconds=time.perf_counter() - t0, error=exc,
                             streamed=watched, request_hash=request_hash)
            _record(CallRecord(function, model, request, None, error=f"{type(exc).__name__}: {exc}"))
            friendly = _login_error(exc)
            if friendly is not None:
                raise friendly from exc
            raise
        except BaseException as exc:                 # a closed stream, Ctrl-C: the exchange still counts
            calllog.exchange(model, request, None, started=started, seconds=time.perf_counter() - t0, error=exc,
                             streamed=watched, request_hash=request_hash)
            raise
    calllog.exchange(model, request, response, started=started, seconds=time.perf_counter() - t0,
                     streamed=watched and first is not None, first_delta=first, request_hash=request_hash)
    _record(CallRecord(function, model, request, response))
    _remember(request, response)
    if key is not None and response.finish_reason not in ("error",):
        CACHE.put(key, response)
    return response


def config_of(settings: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Optional[lm15.Config]:
    s = {**settings, **(overrides or {})}
    fields = {k: v for k, v in s.items() if k in CONFIG_FIELDS and v is not None and v != ()}
    return lm15.Config(**fields) if fields else None


# ------------------------------------------------------------------ tools


def tool_spec(fn: Callable) -> Tool:
    """A Python function as a tool: its name, its docstring, and a JSON Schema of
    its parameters (lowered like a signature's inputs); a parameter without a
    default is required."""
    if isinstance(fn, Tool):
        return fn
    try:
        hints = __import__("typing").get_type_hints(fn)
    except Exception:
        hints = dict(getattr(fn, "__annotations__", {}) or {})
    props: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"tool {fn.__name__}: *{name} has no JSON Schema; name every parameter")
        shape = shape_of(hints.get(name, str), where=f"{fn.__name__}.{name}")
        if param.default is param.empty:
            required.append(name)
        else:
            try:
                json.dumps(param.default)
                shape = {**shape, "default": param.default}
            except (TypeError, ValueError):
                pass
        props[name] = shape
    # no "additionalProperties": Gemini refuses the keyword in function declarations (live, 2026-09-26)
    parameters = {"type": "object", "properties": props, "required": required}
    return Tool(fn.__name__, inspect.cleandoc(fn.__doc__ or ""), parameters)


def run_tool(tools: Dict[str, Callable], call: ToolCall, *, errors: str) -> str:
    fn = tools.get(call.name)
    if fn is None:
        if errors == "raise":
            raise KeyError(f"the model called unknown tool {call.name!r}")
        return f"error: there is no tool named {call.name!r}"
    try:
        out = fn(**(call.input or {}))
    except Exception as exc:  # noqa: BLE001 — reported to the model unless asked to raise
        if errors == "raise":
            raise
        return f"error: {type(exc).__name__}: {exc}"
    return out if isinstance(out, str) else json.dumps(out, default=str, ensure_ascii=False)


# ------------------------------------------------------------------ the call


def _model_step(plan: lmcc.Plan, rendered: Any, response: Any, values: Dict[str, Any]) -> lmcc.ModelStep:
    return lmcc.ModelStep(values, message_to_dict(response.message), lmcc.turn.sha256(rendered.request()),
                          plan.calls_field)


def _misfit(annotations: Dict[str, Any], values: Dict[str, Any]) -> Optional[str]:
    """The first output value that does not fit its declared type (a choice
    outside a Literal inside a record, text where a number goes), or None."""
    for name, value in values.items():
        if name in annotations:
            problem = mismatch(annotations[name], value, name)
            if problem:
                return problem
    return None


def _asked_again(err: lmcc.Refusal) -> str:
    """Why the model is asked again, in a sentence (a stream's Retry event)."""
    if err.code == "parse-truncated":
        return "the reply was cut off; asking again with a larger token budget"
    return f"the reply could not be read ({err.hint}); asking again"


def _complete(plan, rendered, *, router, model, settings, function, responses, annotations=None) -> tuple:
    """One model call; after an unreadable reply, up to ``retries`` follow-ups
    that send the reader's hint back (a cut reply is re-sent with twice the
    token budget instead)."""
    request = lmcc_lm15.request(rendered, model=model, config=config_of(settings))
    rendered_hash = lmcc.turn.sha256(rendered.request())
    retries = max(0, int(settings.get("retries") or 0))
    overrides: Dict[str, Any] = {}
    for attempt in range(retries + 1):
        response = send(router, request, function=function, model=model, settings=settings, plan=plan,
                        request_hash=rendered_hash)
        responses.append(response)
        try:
            reading = lmcc_lm15.read(plan, response)
            problem = _misfit(annotations or {}, reading.values)
            if problem:
                raise lmcc.Refusal("parse-value", problem)
            return response, reading
        except lmcc.Refusal as err:
            if settings.get("cache_replies"):
                CACHE.discard(CACHE.key(request))     # not kept: asked again, it is asked of the model
            thought = getattr(response.usage, "reasoning_tokens", None) or 0
            if err.code == "parse-truncated" and thought:
                err = lmcc.Refusal(err.code, f"{err.hint} (the model spent {thought} of its tokens thinking "
                                             f"first; raise max_tokens)", fix=err.fix, partial=err.partial)
            if attempt == retries or not (err.code.startswith("parse-") or err.code == "format-read-error"):
                raise err
            if err.code == "parse-truncated" and "max_tokens" in (settings.get("_dropped") or ()):
                raise err                     # this provider takes no token budget to raise
            if err.code == "parse-truncated":
                current = (config_of(settings, overrides) or lm15.Config()).max_tokens or 1024
                overrides["max_tokens"] = current * 2
                request = lmcc_lm15.request(rendered, model=model, config=config_of(settings, overrides))
            else:
                request = dataclasses.replace(request, messages=request.messages + (
                    response.message,
                    lm15.Message.user(f"Your reply could not be read: {err.hint}. Reply again, in exactly "
                                      f"the form the instructions give."),))
            call = calllog.current()
            if call is not None:
                call.emit("retry", reason=_asked_again(err), wait=None)
    raise AssertionError("unreachable")


def prepare_inputs(spec: Spec, values: Dict[str, Any]) -> Dict[str, Any]:
    """Values as their fields expect them. A text field given another object
    gets its text: JSON for dataclasses, pydantic models, dicts and lists;
    ``str()`` otherwise."""
    out: Dict[str, Any] = {}
    for f in spec.signature.inputs:
        if f.name not in values:
            continue
        v = values[f.name]
        if f.shape.get("type") == "string" and "enum" not in f.shape and v is not None and not isinstance(v, str):
            v = to_text(v)
        out[f.name] = v
    return out


def to_text(v: Any) -> str:
    try:
        plain = lmcc.turn.to_json(v)
    except lmcc.Refusal:
        return str(v)
    if isinstance(plain, (dict, list)) and not isinstance(v, (str,)):
        if isinstance(v, (dict, list, tuple)) or dataclasses.is_dataclass(v) or hasattr(v, "model_dump"):
            return json.dumps(plain, ensure_ascii=False, indent=2)
    return str(v)


def run(*, function: str, plan: lmcc.Plan, spec: Spec, inputs: Dict[str, Any], past: Sequence[lmcc.Turn],
        settings: Dict[str, Any], router: Any, model: str, tools: Dict[str, Callable],
        tool_specs: Sequence[Tool]) -> Prediction:
    """Call the model (and run tools until it answers). Returns everything."""
    values = prepare_inputs(spec, inputs)
    if spec.tools:
        values["tools"] = list(tool_specs)
    turn = plan.turn(values)
    responses: List[Any] = []
    max_steps = max(1, int(settings.get("max_steps") or 1))
    for _ in range(max_steps):
        rendered = plan.render(turn, turns=list(past))
        try:
            response, reading = _complete(plan, rendered, router=router, model=model, settings=settings,
                                          function=function, responses=responses, annotations=spec.annotations)
        except lmcc.Refusal as err:
            if settings.get("on_unreadable") != "record" or not err.code.startswith("parse-") or not responses:
                raise
            # Keep the reply as it came, with no values: a verbatim replay still writes it into
            # the next request, so a multi-turn exchange stays one conversation (lmcc D-48).
            last = responses[-1]
            turn = turn.with_step(lmcc.ModelStep({}, message_to_dict(last.message),
                                                 lmcc.turn.sha256(rendered.request()), plan.calls_field))
            turn = dataclasses.replace(turn.finish(), meta={**turn.meta, "refusal": err.describe()})
            pred = Prediction({}, turn=turn, response=last, responses=responses, attempts=len(responses))
            object.__setattr__(pred, "refusal", err)
            return pred
        turn = turn.with_step(_model_step(plan, rendered, response, reading.values))
        calls = reading.values.get("calls") or []
        if not calls:
            turn = turn.finish()
            outputs = {k: coerce(spec.annotations.get(k), v) for k, v in (turn.outputs or {}).items()
                       if k != "calls"}
            return Prediction(outputs, turn=turn, response=response, responses=responses,
                              repairs=reading.repairs, attempts=len(responses),
                              probabilities=reading.probabilities, measured_by=reading.measured_by)
        current = calllog.current()
        for call in calls:
            if current is not None:
                event = current.emit("tool_call", id=call.id, name=call.name, input=call.input)
                # a required journal keeps the request before the tool runs
                calllog.tool_barrier(event.seq if event is not None else None)
            output = run_tool(tools, call, errors=settings.get("tool_errors") or "report")
            if current is not None:
                current.emit("tool_result", id=call.id, name=call.name, output=output)
            turn = turn.tool(call.id, output)
    raise StepLimit(f"{function}: no answer after {max_steps} model steps", turn)


def fit_turn(plan: lmcc.Plan, spec: Spec, demo: Any) -> Optional[lmcc.Turn]:
    """A demo or past turn written for this plan's signature: a turn of the same
    signature as it is; a recorded turn of another signature (the function
    changed since) or an ``{"inputs", "outputs"}`` dict as an example of its
    values — the fields this signature still has."""
    if isinstance(demo, lmcc.Turn):
        if demo.signature == plan.fingerprint:
            return demo
        inputs, outputs = dict(demo.inputs), dict(demo.outputs or {})
    elif isinstance(demo, dict) and "signature" in demo and "inputs" in demo:
        if demo["signature"] == plan.fingerprint:
            return plan.load_turn(demo)
        inputs, outputs = dict(demo["inputs"]), dict(demo.get("outputs") or {})
    elif isinstance(demo, dict) and "inputs" in demo:
        inputs, outputs = dict(demo["inputs"]), dict(demo.get("outputs") or {})
    else:
        raise TypeError(f"a demo is an lmcc Turn or {{'inputs': ..., 'outputs': ...}}, not {type(demo).__name__}")
    names_in = {f.name for f in plan.signature.inputs if f.purpose == "plain"}
    names_out = {f.name for f in plan.signature.outputs if f.purpose == "plain" or f.purpose == "reasoning"}
    ins = prepare_inputs(spec, {k: v for k, v in inputs.items() if k in names_in})
    outs = {k: v for k, v in outputs.items() if k in names_out}
    if not outs:
        return None
    try:
        return plan.example(ins, outs)
    except lmcc.Refusal:
        return None


__all__ = ["StepLimit", "LoginRequired", "CallRecord", "inspect_history", "phistory", "clear_cache", "clear_history", "run",
           "fit_turn", "tool_spec"]
