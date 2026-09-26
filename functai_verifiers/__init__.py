"""functai programs as a verifiers harness: ``--env.agent.harness.id functai-verifiers``.

Reinforcement learning and teacher rollouts on Prime Intellect (prime-rl,
``prime train``) for any @ai function, with any lmcc layout:

    vf-eval functai-rows --env.taskset.program mytask:classify --env.taskset.data rows.jsonl \\
        --env.agent.harness.id functai-verifiers --env.agent.harness.program mytask:classify

verifiers records a rollout's trace from the model calls that pass through its
interception endpoint, so this harness only has to send every call there: the
trace is verifiers' own, whatever layout writes the calls. The program runs
in-process, one conversation per rollout (the function's memory, fresh per
rollout), so a multi-turn exchange keeps its turns between segments.

Two rules keep a recorded conversation one training sample:

- past replies are replayed verbatim (lmcc ``replay: "verbatim"``), so each
  request extends the previous one exactly, repaired or unreadable;
- no retries: a follow-up outside the turn record would break that.

A reply that cannot be read becomes a turn with no values (the refusal kept in
``trace.info``) and the exchange goes on; the reward decides what it is worth.
``on_unreadable="raise"`` ends the rollout instead.

Ported from lmfn_verifiers (2026-09-23), where it trained and distilled
banking77 and alphabet-sort students.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import importlib
import json
import re
import typing
from pathlib import Path
from typing import Any, Literal

import lmcc
from lmcc import turn as lmcc_turn
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

import functai
from functai import adapters
from functai.core import FunctAIFunc

INFO_KEY = "functai"
ENVELOPE = "functai_inputs"

__all__ = ["FunctaiHarness", "FunctaiHarnessConfig", "load_program", "user_turn", "decode_inputs", "turns",
           "strip_reasoning", "AnswerOnlyTask", "AnswerOnlyTaskConfig", "RowTaskData", "OneTurnEnv", "sft_rows"]


# ------------------------------------------------------------------ config


class FunctaiHarnessConfig(HarnessConfig):
    program: str = ""
    """The @ai function to run: ``pkg.module:function``."""
    adapter: str | None = None
    """The lmcc layout: a name in the program module's ``ADAPTERS`` dict, one of
    functai's ("xml", "chat", "json"), ``pkg.module:object``, or a path to an
    lmcc adapter JSON file. None: the function's own (its template or adapter)."""
    capabilities: dict[str, bool] = {"instruct": True, "stop_sequences": True}
    """What the served model can do (lmcc capability facts). The model behind
    verifiers is an OpenAI-compatible server functai cannot identify, so this is
    declared, never guessed. Text tool calls unless ``native_function_calling``."""
    read_timeout: float = 600.0
    """Seconds to wait for a reply; thinking teachers are slow."""
    on_unreadable: Literal["record", "raise"] = "record"
    """``record``: an unreadable reply becomes a turn with no values and the
    exchange goes on; ``raise``: the rollout errors."""


# ------------------------------------------------------------------ wire


@dataclasses.dataclass
class _Route:
    provider: str
    model: str
    capabilities: Any = None


class EndpointClient:
    """functai's client face over verifiers' interception endpoint (OpenAI Chat
    Completions at ``endpoint``, bearer ``secret``)."""

    def __init__(self, endpoint: str, secret: str, read_timeout: float = 600.0):
        import lm15
        from lm15.transports._sync import StdlibTransport
        self.lm = lm15.OpenAIChatLM(api_key=secret, base_url=endpoint,
                                    transport=StdlibTransport(read_timeout=read_timeout))

    def resolve(self, model: str) -> _Route:
        return _Route("verifiers", model)

    def complete(self, request):
        return self.lm.complete(request)

    def stream(self, request):
        return self.lm.stream(request)


# ------------------------------------------------------------------ loading


def _load_object(path: str):
    module, _, attr = path.rpartition(":")
    if not module or not attr:
        raise ValueError(f"{path!r} must be 'pkg.module:name'")
    return getattr(importlib.import_module(module), attr)


def _adapter_of(fn: FunctAIFunc, spec: str | None, program: str) -> lmcc.Adapter:
    if spec is None:
        if fn._template is not None:
            return adapters.template_adapter(fn._template)
        return adapters.resolve_adapter(fn._effective().get("adapter") or "xml")
    if spec.endswith(".json"):
        return lmcc.load(json.loads(Path(spec).read_text()), registry=adapters.REGISTRY)
    if ":" in spec:
        return adapters.resolve_adapter(_load_object(spec))
    module = importlib.import_module(program.rpartition(":")[0])
    named = getattr(module, "ADAPTERS", {})
    if spec in named:
        return adapters.resolve_adapter(named[spec])
    if spec.lower() in adapters.NAMED:
        return adapters.resolve_adapter(spec)
    raise KeyError(f"adapter {spec!r} is not in {module.__name__}.ADAPTERS ({sorted(named)}) nor one of "
                   f"functai's ({sorted(set(adapters.NAMED))})")


def load_program(config: FunctaiHarnessConfig) -> FunctAIFunc:
    """The configured function, on the configured layout, replaying past replies verbatim."""
    if not config.program:
        raise ValueError("functai-verifiers needs --env.agent.harness.program pkg.module:function")
    fn = _load_object(config.program)
    if not isinstance(fn, FunctAIFunc):
        raise TypeError(f"{config.program!r} is not an @ai function")
    adapter = _adapter_of(fn, config.adapter, config.program)
    # the recorded conversation must be the model's own text (lmcc D-48)
    return fn.using(adapter=dataclasses.replace(adapter, replay="verbatim"))


# ------------------------------------------------------------------ inputs


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "value"):          # Enum
        return value.value
    raise TypeError(f"cannot send {type(value).__name__} as a turn input")


def user_turn(**inputs) -> str:
    """The user message that carries one turn's inputs to the program."""
    return json.dumps({ENVELOPE: inputs}, ensure_ascii=False, default=_jsonable)


def _text_of(message) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(getattr(p, "text", None) or (p.get("text", "") if isinstance(p, dict) else "")
                       for p in content)
    return ""


def decode_inputs(fn: FunctAIFunc, text: str) -> dict:
    """A turn's inputs: the ``user_turn`` envelope, or, for a function with a
    single text input, the text itself (so plain chat users work too)."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    params = [f for f in fn.signature.inputs if f.purpose == "plain"]
    if isinstance(data, dict) and ENVELOPE in data:
        raw = data[ENVELOPE]
        by_name = {f.name: f for f in params}
        unknown = set(raw) - set(by_name)
        if unknown:
            raise ValueError(f"inputs {sorted(unknown)} are not parameters of {fn.__name__}")
        return {k: lmcc_turn.lift(by_name[k].annotation, v) for k, v in raw.items()}
    if len(params) == 1 and params[0].shape.get("type") == "string":
        return {params[0].name: text}
    raise ValueError(f"{fn.__name__} takes {[p.name for p in params]}: send the turn with "
                     f"functai_verifiers.user_turn(...), not plain text")


# ------------------------------------------------------------------ session


class FunctaiSession(HarnessSession):
    """One rollout: a fresh copy of the function (its own conversation memory),
    one call per verifiers segment."""

    def __init__(self, harness: "FunctaiHarness", ctx: ModelContext, trace: Trace, runtime: Runtime, endpoint: str,
                 secret: str, mcp_urls: dict[str, str], data: TaskData, tool_interception_url: str | None = None):
        super().__init__(harness, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url)
        config = harness.config
        sampling = {k: v for k, v in ctx.sampling.model_dump(exclude_none=True).items()
                    if k in ("temperature", "top_p", "max_tokens")}
        base = load_program(config)
        self.fn = base.using(lm=ctx.model, client=EndpointClient(endpoint, secret, config.read_timeout),
                             capabilities=dict(config.capabilities), retries=0, stateful=True, state_window=0,
                             on_unreadable=config.on_unreadable, escalate_to=None, cache_replies=False, **sampling)

    async def _run(self, messages) -> ProgramResult:
        if messages is None:                  # a prompted task opens the exchange
            prompt = self.data.prompt
            text = prompt if isinstance(prompt, str) else _text_of(prompt[-1]) if prompt else ""
        else:
            users = [m for m in messages if getattr(m, "role", None) == "user"
                     or (isinstance(m, dict) and m.get("role") == "user")]
            text = _text_of(users[-1]) if users else ""
        inputs = decode_inputs(self.fn, text)
        context = contextvars.copy_context()
        pred = await asyncio.to_thread(context.run, lambda: self.fn(**inputs, all=True))
        _record(self.trace, pred)
        return ProgramResult(exit_code=0, stdout="", stderr="")


def _record(trace: Trace, pred) -> None:
    turn = pred.turn
    entry: dict[str, Any] = {"inputs": lmcc_turn.to_json(turn.inputs),
                             "outputs": lmcc_turn.to_json(turn.outputs or {}),
                             "repairs": pred.repairs, "steps": len(turn.steps)}
    if pred.refusal is not None:
        entry["refusal"] = pred.refusal.describe()
    trace.info.setdefault(INFO_KEY, {"turns": []})["turns"].append(entry)


# ------------------------------------------------------------------ harness


class FunctaiHarness(Harness[FunctaiHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_RESUME = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def setup(self, runtime: Runtime) -> None:
        load_program(self.config)            # fail at setup, not mid-rollout

    async def session(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                      tool_interception_url=None) -> HarnessSession:
        return FunctaiSession(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url)

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                     tool_interception_url=None) -> ProgramResult:
        """One segment on its own (verifiers normally drives ``session``)."""
        session = FunctaiSession(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data)
        return await session._run(None)


# ------------------------------------------------------------------ rewards


def turns(trace: Trace, fn: FunctAIFunc | None = None) -> list[dict]:
    """Each turn this rollout recorded: ``inputs``, ``outputs`` (JSON values;
    lifted to the function's types when ``fn`` is given), ``repairs``, and
    ``refusal`` when the reply could not be read."""
    recorded = [dict(t) for t in trace.info.get(INFO_KEY, {}).get("turns", [])]
    if fn is not None:
        by_name = {f.name: f for f in fn.signature.fields}
        for t in recorded:
            t["outputs"] = {k: lmcc_turn.lift(by_name[k].annotation, v) if k in by_name else v
                            for k, v in t["outputs"].items()}
    return recorded


# ------------------------------------------------------------------ distillation
#
# In SFT distillation the teacher writes the rollouts and the student is
# trained on every token of them (prime-rl `loss = "sft"`). The student's
# renderer writes a reply's `reasoning_content` into the training sample as a
# <think> block, so a thinking teacher would teach the student to think. A task
# that drops the reasoning at the response boundary (before the trace records
# it) keeps the teacher's thinking out of what the student learns: the answer.

import verifiers.v1 as vf  # noqa: E402

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_reasoning(response: "vf.Response") -> "vf.Response | None":
    """The response without its reasoning (the field, the opaque signed state,
    and a leading inline <think> block); None when there was none."""
    message = response.message
    content = message.content
    inline = isinstance(content, str) and _THINK.match(content) is not None
    if not message.reasoning_content and not message.provider_state and not inline:
        return None
    update = {"reasoning_content": None, "provider_state": None}
    if inline:
        update["content"] = _THINK.sub("", content, count=1)
    return response.model_copy(update={"message": message.model_copy(update=update)})


class AnswerOnlyTaskConfig(vf.TaskConfig):
    teacher_reasoning: Literal["drop", "keep"] = "drop"
    """``drop``: the recorded reply is the answer only (distilling a thinking
    teacher into a model that answers directly). ``keep``: record it all."""


ConfigT = typing.TypeVar("ConfigT", bound=AnswerOnlyTaskConfig)


class RowTaskData(vf.TaskData):
    info: dict
    """``inputs`` (the function's inputs for this row) and any row data the
    reward needs (a label, a reference answer)."""


class AnswerOnlyTask(vf.Task[RowTaskData, vf.State, ConfigT]):
    """A task base for distillation: drops the model's reasoning before the
    trace records the reply (``teacher_reasoning = "drop"``, the default)."""

    @vf.intercept
    def drop_teacher_reasoning(self, response: vf.Response) -> vf.Response | None:
        if self.config.teacher_reasoning != "drop":
            return None
        return strip_reasoning(response)

    @vf.metric
    async def readable(self, trace: vf.Trace) -> float:
        """Share of turns whose reply lmcc could read."""
        recorded = turns(trace)
        return sum("refusal" not in t for t in recorded) / len(recorded) if recorded else 0.0


class OneTurnEnv(vf.SingleAgentEnv):
    """The scripted user for a row-per-task taskset: sends the task's
    ``info["inputs"]`` as one turn."""

    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(user_turn(**task.data.info["inputs"]))


# ------------------------------------------------------------------ SFT export


def sft_rows(traces_jsonl: str | Path, *, min_reward: float | None = None, readable_only: bool = True) -> list[dict]:
    """Teacher rollouts (a vf-eval ``traces.jsonl``) as SFT rows
    ``{"prompt": [messages], "completion": [assistant message]}``, the format
    prime-rl's SFT trainer reads: the exact request the teacher answered and its
    reply as recorded (reasoning already dropped by ``AnswerOnlyTask``).
    ``readable_only`` skips turns lmcc could not read; ``min_reward`` skips
    rollouts scored below it."""
    from verifiers.v1.trace import Trace as _Trace
    rows: list[dict] = []
    for line in Path(traces_jsonl).read_text().splitlines():
        record = json.loads(line)
        for data in record.get("traces", [record]):
            if min_reward is not None:
                score = sum((r or {}).get("score", 0) * (r or {}).get("weight", 1)
                            for r in (data.get("rewards") or {}).values())
                if score < min_reward:
                    continue
            recorded = data.get("info", {}).get(INFO_KEY, {}).get("turns", [])
            if readable_only and any("refusal" in t for t in recorded):
                continue
            trace = _Trace.model_validate(data)
            for branch in trace.branches:
                messages = [m.model_dump(exclude_none=True) for m in branch.messages]
                for i, m in enumerate(messages):
                    if m["role"] == "assistant":
                        m.pop("reasoning_content", None)
                        m.pop("provider_state", None)
                        rows.append({"prompt": messages[:i], "completion": [m]})
    return rows


_ = functai  # the package is imported for its side effects (lmcc_std on the default registry)
