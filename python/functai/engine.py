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
    """A function with tools asked the model ``max_steps`` times (8 by
    default) and still had no answer.

    ``err.turn`` is the exchange so far (every tool call and result), to see
    what the model kept doing. Raise the limit with
    ``@ai(tools=[...], max_steps=20)`` (or ``fn.using(max_steps=20)``), or
    make the instruction say when to stop looking things up.
    """

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
    """The last ``n`` requests FunctAI sent (or answered from its cache), oldest first.

    Each is a record with ``function``, ``model``, ``request`` and
    ``response`` (lm15's objects: exactly what went to the provider and what
    came back; ``response`` is None when the provider raised), ``cached``,
    ``error`` and ``timestamp``. The last 500 are kept, in this process only;
    ``phistory()`` prints them as readable text. For every call, kept on
    disk: ``configure(log_calls=True)`` and ``functai.calls()``.

    Parameters
    ----------
    n : int
        How many (default 1: the last one).

    Returns
    -------
    list of CallRecord
    """
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


def clear_cache(which: Any = None) -> None:
    """Forget the replies the reply cache kept, so the next identical requests
    reach the model again.

    Parameters
    ----------
    which : optional
        None (default): the cache in this process's memory
        (``cache_replies=True``). ``"disk"``: the shared file
        ``cache_replies="disk"`` uses (in the user's cache folder,
        ``functai/replies.sqlite``). A folder or ``.sqlite`` path: that file.

    Examples
    --------
    ```python
    # not run: it deletes the replies kept on this machine
    functai.clear_cache()            # this process's memory
    functai.clear_cache("disk")      # the replies kept across runs
    ```
    """
    from . import replies
    replies.clear(which)


# ------------------------------------------------------------------ sending


# functai.save(record=...) records every model exchange here, to replay it in verify.
RECORDING: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar("functai_recording",
                                                                                        default=None)


def _remember(request: Any, response: Any) -> None:
    rec = RECORDING.get()
    if rec is not None:
        rec["exchanges"].append({"request": request_to_dict(request), "response": response_to_dict(response)})


def send(router: Any, request: Any, *, function: str, model: str, settings: Dict[str, Any],
         plan: Optional[lmcc.Plan] = None, request_hash: Optional[str] = None, hit: Any = None) -> Any:
    """One model call: ``hit`` when the reply is already known (the reply
    cache, or a turn being resumed: contract/tools.md), else through the
    router, re-sent after transient errors (rate limit, 5xx, timeout) with
    backoff. Each attempt is an exchange of the call, and begins a
    ``request`` event. When the call is watched (a stream, an observer or a
    journal sees it), the reply is streamed and shown field by field as it
    arrives (``plan`` reads it); what is returned is the same whole reply.
    ``request_hash``: lmcc's hash of the rendered request this one was made
    from."""
    call = calllog.current()
    watched = call is not None and call.wants_pieces and plan is not None
    whole = call is not None and not watched and call.watched and plan is not None   # shown, not live
    if hit is not None:
        if call is not None:
            call.request(model)
        _record(CallRecord(function, model, request, hit, cached=True))
        _remember(request, hit)
        calllog.exchange(model, request, hit, started=time.time(), seconds=0.0, cached=True,
                         request_hash=request_hash)
        if watched or whole:
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
            if isinstance(exc, lm15.ContextLengthError):
                exc.add_note(f"{function}: the request no longer fits the model. In a conversation, show fewer "
                             f"earlier turns: context=functai.last_turns(10)")
            raise
        except BaseException as exc:                 # a closed stream, Ctrl-C: the exchange still counts
            calllog.exchange(model, request, None, started=started, seconds=time.perf_counter() - t0, error=exc,
                             streamed=watched, request_hash=request_hash)
            raise
    calllog.exchange(model, request, response, started=started, seconds=time.perf_counter() - t0,
                     streamed=watched and first is not None, first_delta=first, request_hash=request_hash)
    if whole:
        from . import streaming
        streaming.replay(call, plan, response)          # a reply that arrived whole: one piece per field
    _record(CallRecord(function, model, request, response))
    _remember(request, response)
    return response


def config_of(settings: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Optional[lm15.Config]:
    s = {**settings, **(overrides or {})}
    fields = {k: v for k, v in s.items() if k in CONFIG_FIELDS and v is not None and v != ()}
    return lm15.Config(**fields) if fields else None


# ------------------------------------------------------------------ tools


def tool_spec(fn: Callable) -> Tool:
    """A Python function as a tool: its name, its docstring, and a JSON Schema of
    its parameters (lowered like a signature's inputs); a parameter without a
    default is required. A function that holds its inputs' shapes as data (an
    AI function loaded from a manifest: ``_tool_parameters``) gives them
    itself, as its interface states them: they are what the original's
    annotations lower to, which the Python types written for its signature
    are not."""
    if isinstance(fn, Tool):
        return fn
    own = getattr(fn, "_tool_parameters", None)
    if callable(own):
        return Tool(fn.__name__, inspect.cleandoc(fn.__doc__ or ""), own())
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
    if isinstance(out, str):
        return out
    from .interface import json_form
    ok, data = json_form(out)          # a record as its JSON (a pydantic model, a dataclass), not its str()
    return json.dumps(data if ok else out, default=str, ensure_ascii=False)


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


LMCC_TRUNCATED_ADVICE = "; raise max_tokens or ask for less"


def cut_off(err: lmcc.Refusal, response: Any, limit: Optional[int]) -> lmcc.Refusal:
    """A ``parse-truncated`` refusal that says what happened (functions.md,
    "When the reply cannot be read"): how much went to thinking, whose limit
    it was and whether it can be raised, and what lm15 changed in the request.
    ``limit``: the ``max_tokens`` this request set, or None."""
    hint = err.hint[:-len(LMCC_TRUNCATED_ADVICE)] if err.hint.endswith(LMCC_TRUNCATED_ADVICE) else err.hint
    usage = getattr(response, "usage", None)
    thought = getattr(usage, "reasoning_tokens", None) or 0
    total = getattr(usage, "output_tokens", None) or 0
    if thought:
        hint += (f"; the model spent {thought} of its {total} output tokens thinking" if total >= thought
                 else f"; the model spent {thought} tokens thinking")
    notes = tuple(getattr(response, "adaptations", None) or ())
    chosen = next((a.applied for a in notes if a.field == "config.max_tokens" and a.action == "defaulted"), None)
    if limit is not None:
        hint += f"; raise max_tokens (it was {limit}) or ask for less"
    elif chosen is not None:
        hint += (f"; no max_tokens was set, and lm15 sent {chosen}, the most it knows this model to allow: "
                 "lower the reasoning effort or ask for less")
    else:
        hint += ("; no max_tokens was set, so the provider used its own maximum: "
                 "lower the reasoning effort or ask for less")
    other = [f"{a.field} {a.action}: {a.reason}" for a in notes if a.field != "config.max_tokens"]
    if other:
        hint += " (lm15 adapted the request: " + "; ".join(other) + ")"
    return lmcc.Refusal(err.code, hint, fix=err.fix, partial=err.partial)


def _complete(plan, rendered, *, router, model, settings, function, responses, annotations=None) -> tuple:
    """One model call; after an unreadable reply, up to ``retries`` follow-ups
    that send the reader's hint back. A cut reply is re-sent with twice the
    token budget instead, when a budget was set: without one the reply already
    had the most the call allows (lm15's default for the model, or the
    provider's own maximum), and a guessed number would only shrink it."""
    request = lmcc_lm15.request(rendered, model=model, config=config_of(settings))
    asked = rendered.request()
    rendered_hash = lmcc.turn.sha256(asked)
    retries = max(0, int(settings.get("retries") or 0))
    overrides: Dict[str, Any] = {}
    from . import conversations, replies
    call = calllog.current()
    cancelled = call.check if call is not None else None
    from . import plugins
    for attempt in range(retries + 1):
        # the escape hatch: a plugin may replace the provider request; no one can rebuild that request, so
        # its exchange has no request_hash and the call is not replayable
        request, replaced = plugins.request(request, call, function)
        sent_hash = None if replaced else rendered_hash
        # a reply already known: the turn being resumed recorded it, or the reply cache kept it
        hit = conversations.recorded_reply(request, settings)
        flight = replies.begin(settings, request, cancelled) if hit is None else None
        try:
            if hit is None and flight is not None:
                hit = flight.reply
            response = send(router, request, function=function, model=model, settings=settings, plan=plan,
                            request_hash=sent_hash, hit=hit)
            responses.append(response)
            conversations.note_reply(request, response, settings)    # a stored turn keeps every reply
            try:
                reading = lmcc_lm15.read(plan, response)
                problem = _misfit(annotations or {}, reading.values)
                if problem:
                    raise lmcc.Refusal("parse-value", problem)
            except lmcc.Refusal:
                if flight is not None:
                    flight.drop()                     # a kept reply that no longer reads is forgotten
                raise
            if flight is not None:
                flight.keep(response)                 # only a reply that was read is kept
            return response, reading
        except lmcc.Refusal as err:
            limit = None
            if err.code == "parse-truncated":
                # the budget this request set; none when this provider takes none
                if "max_tokens" not in (settings.get("_dropped") or ()):
                    limit = (config_of(settings, overrides) or lm15.Config()).max_tokens
                err = cut_off(err, response, limit)
            if attempt == retries or not (err.code.startswith("parse-") or err.code == "format-read-error"):
                raise err
            if err.code == "parse-truncated" and limit is None:
                raise err                     # no budget was set: nothing larger to give
            if err.code == "parse-truncated":
                overrides["max_tokens"] = limit * 2
                request = lmcc_lm15.request(rendered, model=model, config=config_of(settings, overrides))
            else:
                correction = (f"Your reply could not be read: {err.hint}. Reply again, in exactly "
                              f"the form the instructions give.")
                request = dataclasses.replace(request, messages=request.messages + (
                    response.message, lm15.Message.user(correction),))
                # the exchange's request_hash is the hash of what it sends (contract/calls.md, exchanges)
                asked = {**asked, "messages": [*asked["messages"], message_to_dict(response.message),
                                               {"role": "user", "parts": [{"type": "text", "text": correction}]}]}
                rendered_hash = lmcc.turn.sha256(asked)
            if call is not None:
                call.emit("retry", reason=_asked_again(err), wait=None)
        finally:
            if flight is not None:
                flight.end()
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
    asked: List[Any] = []                   # every tool call the model asked for, across steps (outputs.calls)
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
        asked.extend(calls)
        if not calls:
            turn = turn.finish()
            outputs = {k: coerce(spec.annotations.get(k), v) for k, v in (turn.outputs or {}).items()
                       if k != "calls"}
            pred = Prediction(outputs, turn=turn, response=response, responses=responses,
                              repairs=reading.repairs, attempts=len(responses),
                              probabilities=reading.probabilities, measured_by=reading.measured_by)
            if spec.tools:
                object.__setattr__(pred, "tool_calls", list(asked))
            return pred
        current = calllog.current()
        for call in calls:
            output = _one_tool(current, tools, call, settings)
            turn = turn.tool(call.id, output)
    raise StepLimit(f"{function}: no answer after {max_steps} model steps", turn)


def _one_tool(current: Any, tools: Dict[str, Callable], call: ToolCall, settings: Dict[str, Any]) -> str:
    """One tool call of the call in progress (contract/tools.md,
    contract/plugins.md): numbered (its invocation), shown, given to the
    ``tool_call`` hooks (which may change its input, block it, or ask a
    person: ``approve=`` is one of them), kept before and after it runs when
    it changes things (a stored turn, a required journal), run, its result
    given to the ``tool_result`` hooks, and shown."""
    from . import plugins, tools as _tools
    errors = settings.get("tool_errors") or "report"
    if current is None:
        return run_tool(tools, call, errors=errors)
    current.invocations += 1
    n = current.invocations
    fn = tools.get(call.name)
    effects = _tools.effects_of(fn) if fn is not None else "reads"     # an unknown tool runs nothing
    event = current.emit("tool_call", id=call.id, name=call.name, input=call.input, invocation=n)
    approval = _tools.Approval(current.id, n, call.id, call.name, call.input, effects,
                               _tools.path_of(current, call.name), current.path)
    tool_input, refused = plugins.tool_call(current, approval, settings)
    if refused is not None:
        output = refused
    else:
        run = current.turn_run
        known = run.recorded_tool(current, approval) if run is not None else None
        if known is not None:
            output = known                            # a turn resumed: this tool ran before; its result is kept
        else:
            from .conversations import frontier
            frontier()
            if effects != "reads":
                # a required journal keeps the request before a tool that changes things runs
                calllog.tool_barrier(event.seq if event is not None else None)
                if run is not None:
                    run.tool_started(current, dataclasses.replace(approval, input=tool_input))
            asked = ToolCall(call.id, call.name, tool_input) if tool_input is not call.input else call
            token = calllog.INVOCATION.set(n)
            try:
                output = run_tool(tools, asked, errors=errors)
            finally:
                calllog.INVOCATION.reset(token)
            # the result as the model is shown it; a turn resumed later reuses it, hooks and all
            output = plugins.tool_result(current, approval, tool_input, output)
            if run is not None:
                run.tool_done(current, approval, output)
    current.emit("tool_result", id=call.id, name=call.name, output=output, invocation=n)
    return output


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
