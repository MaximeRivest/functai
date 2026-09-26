"""Which model, how to reach it, and what it can do.

lm15 routes a model string to a provider. lmcc never sniffs what a model
can do: someone declares it. functai declares it from the provider the
model routes to and a small table of model-name prefixes, because lm15's
model information does not say yet whether a model calls tools natively.
Every fact is overridable: ``@ai(capabilities={"native_reasoning": False})``.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Tuple

import lm15
from lm15.router import openai_chat_model_string

# providers whose API has native tool calling, stop sequences and an enforced
# JSON schema (lm15 maps response_format for all four)
NATIVE_PROVIDERS = {"openai", "anthropic", "gemini", "xai"}

# OpenAI's Chat Completions door ("openai/gpt-4o" routes here): native tools and
# JSON schema; no thinking text comes back through it.
CHAT_COMPLETIONS = {"openai-chat", "azure-chat"}

# hosted OpenAI-compatible servers whose current models take native tool calls;
# with text tool calls, reasoning models there wrote the call inside their
# thinking and never answered (gpt-oss-120b on Groq, live 2026-09-26). A model
# without tools fails loudly at the provider; override with
# capabilities={"native_function_calling": False}.
NATIVE_TOOL_HOSTS = {"groq", "openrouter", "deepseek", "together", "fireworks", "deepinfra", "moonshotai",
                     "zai", "parasail", "meta-chat", "bedrock-chat"}

# providers that answer only judgments: typed questions with a probability for
# every declared answer (lm15 MAP-14; TypeSafe's Jev), never free text
JUDGMENT_ONLY = {"typesafe"}

# model-name prefixes with an API-level thinking channel lm15 can request
_REASONING_PREFIXES = {
    "openai": ("o1", "o3", "o4", "gpt-5"),
    "anthropic": ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4-5", "claude-3-7-sonnet"),
    "gemini": ("gemini-2.5", "gemini-3"),
    "xai": ("grok-3-mini", "grok-4"),
}


def capabilities(provider: str, model: str) -> Dict[str, bool]:
    """The lmcc capability facts functai declares for ``model`` served by ``provider``."""
    if provider in JUDGMENT_ONLY:
        return {"native_structured_output": True}
    caps = {"instruct": True}
    if provider in NATIVE_PROVIDERS:
        caps["native_function_calling"] = True
        caps["native_structured_output"] = True
        # lm15's `openai` provider speaks the Responses API, which has no `stop`
        caps["stop_sequences"] = provider != "openai"
        caps["native_reasoning"] = model.startswith(_REASONING_PREFIXES.get(provider, ()))
        # Anthropic continues a trailing assistant message (not with extended thinking)
        caps["assistant_prefill"] = provider == "anthropic" and not caps["native_reasoning"]
    elif provider in CHAT_COMPLETIONS:
        caps.update(native_function_calling=True, native_structured_output=True, stop_sequences=False,
                    native_reasoning=False)
    else:
        caps["native_function_calling"] = provider in NATIVE_TOOL_HOSTS
        # OpenAI-compatible hosts (Groq, OpenRouter, DeepSeek, Ollama, ...) vary by
        # model: text tool calls and tags work everywhere. No stop sequences: many
        # models there think in the same token stream (gpt-oss, qwen3, deepseek-r1),
        # and a stop sequence fires inside the thinking (seen live on Groq's
        # gpt-oss-120b, 2026-09-26: the reply ended mid-thought, before any answer).
        caps["stop_sequences"] = False
    return caps


# ------------------------------------------------------------------ routing

_lock = threading.Lock()
_routers: Dict[Tuple, Any] = {}


def model_string(lm: Any) -> str:
    """The lm15 model string for an ``lm`` setting: a string as given, or the
    ``.model`` of an object that has one (a DSPy/litellm LM, for migration)."""
    if isinstance(lm, str):
        return lm
    model = getattr(lm, "model", None)
    if isinstance(model, str):
        return model
    raise TypeError(f"lm must be a model string like 'gpt-4.1-mini' or 'groq:openai/gpt-oss-120b', "
                    f"not {type(lm).__name__}")


def _shared(key: Tuple, make):
    with _lock:
        r = _routers.get(key)
        if r is None:
            r = _routers[key] = make()
        return r


def resolve(settings: Dict[str, Any]) -> Tuple[Any, str, Any]:
    """``(router, model, route)`` for the effective settings. ``model`` is the
    string lm15 routes (litellm's ``provider/model`` is read the way lm15's
    own ingest reads it); ``route`` has ``.provider`` and ``.model``."""
    if settings.get("lm") is None:
        raise RuntimeError("no model configured: call functai.configure(lm='gpt-4.1-mini') "
                           "or pass lm=... to @ai")
    model = model_string(settings["lm"])
    router = settings.get("router")
    if router is None:
        router = _shared(("env",), lm15.LMRouter)
    try:
        route = router.resolve(model)
    except lm15.UnknownModelError:
        if ":" in model or "/" not in model:
            raise
        model = openai_chat_model_string(model)          # "openai/gpt-4o" → "openai-chat:gpt-4o"
        route = router.resolve(model)
    if settings.get("router") is None and (settings.get("api_key") or settings.get("base_url")):
        provider = route.provider
        key, url = settings.get("api_key"), settings.get("base_url")
        router = _shared(("explicit", provider, key, url), lambda: lm15.LMRouter(lm15.RouterConfig(
            api_keys={provider: key} if key else None,
            base_urls={provider: url} if url else None)))
        route = router.resolve(model)
    return router, model, route


def model_capabilities(settings: Dict[str, Any], route: Any) -> Dict[str, bool]:
    caps = capabilities(route.provider, route.model)
    temperature = settings.get("temperature")
    if route.provider == "anthropic" and caps.get("native_reasoning") and temperature not in (None, 1, 1.0):
        # Anthropic's extended thinking only runs at temperature 1. A caller who set
        # another temperature gets reasoning written in the reply instead (and, with
        # thinking off, Anthropic continues a prefill again).
        caps["native_reasoning"] = False
        caps["assistant_prefill"] = True
    return {**caps, **(settings.get("capabilities") or {})}
