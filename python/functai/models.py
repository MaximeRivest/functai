"""Which model, how to reach it, and what it can do.

lm15 routes a model string to a provider. lmcc never sniffs what a model
can do: someone declares it. functai declares it from the provider the
model routes to and a small table of model-name prefixes, because lm15's
model information does not say yet whether a model calls tools natively.
Every fact is overridable: ``@ai(capabilities={"native_reasoning": False})``.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import lm15
from lm15.router import openai_chat_model_string

# providers whose API has native tool calling, stop sequences and an enforced
# JSON schema (lm15 maps response_format for all four)
NATIVE_PROVIDERS = {"openai", "anthropic", "gemini", "xai"}

# Subscription providers that speak another provider's API: the same abilities.
SUBSCRIPTION_AS = {"claude-code": "anthropic", "openai-codex": "openai"}

# OpenAI's Chat Completions door ("openai/gpt-4o" routes here): native tools and
# JSON schema; no thinking text comes back through it.
CHAT_COMPLETIONS = {"openai-chat", "azure-chat"}

# hosted OpenAI-compatible servers whose current models take native tool calls;
# with text tool calls, reasoning models there wrote the call inside their
# thinking and never answered (gpt-oss-120b on Groq, live 2026-09-26). A model
# without tools fails loudly at the provider; override with
# capabilities={"native_function_calling": False}.
NATIVE_TOOL_HOSTS = {"github-copilot", "kimi-code", "groq", "openrouter", "deepseek", "together", "fireworks", "deepinfra", "moonshotai",
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
        # lm15's `openai` provider speaks the Responses API, which has no `stop`;
        # xAI's current models refuse it (grok-3-mini, grok-4-fast, live 2026-09-26)
        caps["stop_sequences"] = provider not in ("openai", "xai")
        caps["native_reasoning"] = model.startswith(_REASONING_PREFIXES.get(provider, ()))
        # Anthropic continues a trailing assistant message (not with extended thinking)
        caps["assistant_prefill"] = provider == "anthropic" and not caps["native_reasoning"]
    elif provider in SUBSCRIPTION_AS:
        return capabilities(SUBSCRIPTION_AS[provider], model)
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
    """The lm15 model string for an ``lm`` setting: a string as given (with the
    friendly account prefixes: ``claude:`` → ``claude-code:``, ``chatgpt:`` →
    ``openai-codex:``, ``copilot:`` → ``github-copilot:``, ``kimi:`` → ``kimi-code:``),
    or the ``.model`` of a DSPy/litellm LM object (for migration)."""
    if not isinstance(lm, str):
        check_lm(lm)
        lm = lm.model
    head, sep, rest = lm.partition(":")
    if sep and head.lower() in MODEL_PREFIXES:
        return f"{MODEL_PREFIXES[head.lower()]}:{rest}"
    return lm


def _is_baked(obj: Any) -> bool:
    return getattr(type(obj), "__functai_baked__", False) is True


def _is_bound_client(obj: Any) -> bool:
    """lm15's BoundClient (a login's connection and one model), duck-typed."""
    selection = getattr(obj, "selection", None)
    return selection is not None and callable(getattr(obj, "complete", None)) \
        and isinstance(getattr(selection, "routed", None), str)


def _is_provider_lm(obj: Any) -> bool:
    """One provider's lm15 LM (OpenAILM, ClaudeCodeLM, ...), duck-typed: lm15's
    ``ProviderLM`` is a protocol that cannot be checked with isinstance."""
    return isinstance(getattr(obj, "provider", None), str) and callable(getattr(obj, "complete", None)) \
        and not callable(getattr(obj, "resolve", None)) and not _is_bound_client(obj)


def _is_async(obj: Any) -> bool:
    import inspect
    return inspect.iscoroutinefunction(getattr(obj, "complete", None))


def check_lm(lm: Any) -> None:
    """Refuse, with the fix, an ``lm`` value functai cannot use."""
    if lm is None or isinstance(lm, str) or _is_bound_client(lm) or _is_baked(lm):
        return
    if _is_provider_lm(lm):
        raise TypeError(
            f"lm is the model; an lm15 {type(lm).__name__} is a connection to {lm.provider!r} that does not know "
            f"which model to use. Pass it as client= and the model name as lm=, e.g. "
            f"@ai(lm='gpt-4.1', client=lm15.OpenAILM(api_key=...))")
    if isinstance(getattr(lm, "model", None), str):
        return                                      # a DSPy/litellm LM: its model name is read
    raise TypeError("lm must be a model name like 'gpt-4.1-mini', 'claude:claude-sonnet-4-5' or "
                    f"'groq:openai/gpt-oss-120b', or an lm15 BoundClient; not {type(lm).__name__}")


def check_client(client: Any) -> None:
    """Refuse, with the fix, a ``client`` value functai cannot send through."""
    if client is None or _is_router(client):
        return
    if _is_bound_client(client):
        raise TypeError("an lm15 BoundClient carries its model: pass it as lm=, not client=")
    if _is_provider_lm(client):
        if _is_async(client):
            name = type(client).__name__
            raise TypeError(f"{name} is asynchronous; functai calls are synchronous: use the synchronous "
                            f"lm15.{name[5:] if name.startswith('Async') else name}")
        return
    raise TypeError(f"client must be an lm15 LMRouter or provider LM (OpenAILM, AnthropicLM, ClaudeCodeLM, ...), "
                    f"not {type(client).__name__}")


def _is_router(obj: Any) -> bool:
    return callable(getattr(obj, "resolve", None)) and callable(getattr(obj, "complete", None))


class _Route:
    """What functai needs to know about where a request goes. ``capabilities``:
    what the destination declares itself (a baked model), instead of the table."""

    def __init__(self, provider: str, model: str, capabilities: Optional[Dict[str, bool]] = None):
        self.provider, self.model, self.capabilities = provider, model, capabilities

    def __repr__(self) -> str:
        return f"{self.provider}:{self.model}"


def _one_provider(client: Any, model: str) -> Tuple[str, "_Route"]:
    """The wire model for a provider LM: its own prefix (or a friendly one for
    it) is dropped; another provider's prefix is a mistake, said as one."""
    provider = client.provider
    head, sep, rest = model.partition(":")
    if sep and (head == provider or MODEL_PREFIXES.get(head) == provider):
        model = rest
    elif sep and (head in MODEL_PREFIXES.values() or head in _known_providers()):
        raise ValueError(f"lm={model!r} names provider {head!r}, but client is a {provider!r} connection "
                         f"({type(client).__name__}); write lm={rest!r}, or pass a client for {head!r}")
    return model, _Route(provider, model)


def _known_providers() -> frozenset:
    from lm15.registry import PROVIDERS
    from lm15.login.declared import DECLARED_PROVIDERS
    return frozenset(PROVIDERS) | frozenset(d.id for d in DECLARED_PROVIDERS)


# Account-style prefixes a model string may use (lm15 names always work too).
MODEL_PREFIXES = {"claude": "claude-code", "chatgpt": "openai-codex", "codex": "openai-codex",
                  "copilot": "github-copilot", "kimi": "kimi-code"}


def _shared(key: Tuple, make):
    with _lock:
        r = _routers.get(key)
        if r is None:
            r = _routers[key] = make()
        return r


def _declared(provider: str) -> tuple:
    from lm15.login.declared import DECLARED_PROVIDERS
    return tuple(d for d in DECLARED_PROVIDERS if d.id == provider)


def resolve(settings: Dict[str, Any]) -> Tuple[Any, str, Any]:
    """``(router, model, route)`` for the effective settings. ``model`` is the
    string lm15 routes (litellm's ``provider/model`` is read the way lm15's
    own ingest reads it); ``route`` has ``.provider`` and ``.model``.

    Where a call goes, first match wins: ``lm=`` an lm15 BoundClient (its login
    and model); ``client=`` (an lm15 router, or one provider's LM); ``api_key=``;
    the saved login (``functai.login``) for the model's provider; the
    environment and CLI logins (lm15's own rules)."""
    from . import accounts
    lm, client = settings.get("lm"), settings.get("client")
    if _is_baked(lm):
        # weights of our own: the model is its own client, and says what it can do
        return lm, lm.model, _Route(lm.provider, lm.name, lm.capabilities)
    if _is_bound_client(lm):
        # its own connection: a client= from configure does not apply to it (both
        # in one place is refused by config.check, where the contradiction is written)
        return lm, lm.selection.routed, _Route(lm.provider, lm.model)
    if lm is None and client is None:
        picked = accounts.default_model(settings.get("auth"))
        if picked is None:
            raise RuntimeError("no model configured, and no API key or login found to pick one: "
                               "set OPENAI_API_KEY (or ANTHROPIC_API_KEY, GEMINI_API_KEY, ...), or sign in "
                               "with functai.login(), or name a model: functai.configure(lm='gpt-4.1-mini')")
        lm = picked[0]
        accounts.say_default(*picked)
    if lm is None:
        raise RuntimeError("no model configured: call functai.configure(lm='gpt-4.1-mini') "
                           "or pass lm=... to @ai (functai.logins() shows what you can use)")
    model = model_string(lm)
    if client is not None:
        check_client(client)
        if not _is_router(client):
            wire, route = _one_provider(client, model)
            return client, wire, route
        try:
            return client, model, client.resolve(model)
        except lm15.UnknownModelError:
            if ":" in model or "/" not in model:
                raise
            model = openai_chat_model_string(model)
            return client, model, client.resolve(model)
    auth = accounts.auth_for(settings.get("auth"))
    # A router that knows every provider lm15 can connect (declared ones like
    # github-copilot included) resolves the name; no credential is read here.
    namer = _shared(("auth", id(auth)), lambda: lm15.LMRouter(lm15.RouterConfig(auth=auth))) if auth \
        else _shared(("env",), lm15.LMRouter)
    try:
        route = namer.resolve(model)
    except lm15.UnknownModelError:
        if ":" in model or "/" not in model:
            raise
        model = openai_chat_model_string(model)          # "openai/gpt-4o" → "openai-chat:gpt-4o"
        route = namer.resolve(model)
    provider = route.provider
    key, url = settings.get("api_key"), settings.get("base_url")
    if key or url:
        # an explicit key wins; with only a base URL, a saved login still supplies the credential
        use_auth = auth if (not key and auth is not None and provider in accounts.saved_providers(auth)) else None
        router = _shared(("explicit", provider, key, url, id(use_auth)), lambda: lm15.LMRouter(lm15.RouterConfig(
            api_keys={provider: key} if key else None, base_urls={provider: url} if url else None,
            auth=use_auth, providers=_declared(provider) if use_auth is None else ())))
    elif auth is not None and provider in accounts.saved_providers(auth):
        router = namer
    elif _declared(provider) and auth is not None:
        router = namer                                   # reachable only by a login: its error says so
    else:
        router = _shared(("env",), lm15.LMRouter)
    return router, model, route


# lm15 Config fields a provider or model refuses (seen live, 2026-09-26): functai
# leaves them out of the request, and says so once per process, instead of
# failing every call of a program configured for another model.
_SAMPLING = ("temperature", "top_p")


def refused_settings(provider: str, model: str) -> tuple:
    if provider == "openai-codex":
        # the ChatGPT subscription backend: no sampling knobs, and no output cap
        # (lm15 refuses to drop a cap on metered APIs; this one is not billed per token)
        return _SAMPLING + ("max_tokens",)
    if SUBSCRIPTION_AS.get(provider, provider) in ("openai", "openai-chat") and \
            model.startswith(_REASONING_PREFIXES["openai"]):
        return _SAMPLING                          # o-series and GPT-5 reject temperature
    return ()


_warned: set = set()


def adjust(settings: Dict[str, Any], route: Any) -> Dict[str, Any]:
    """The settings as this provider can take them."""
    drop = [k for k in refused_settings(route.provider, route.model) if settings.get(k) is not None]
    if not drop:
        return settings
    key = (route.provider, tuple(drop))
    if key not in _warned:
        _warned.add(key)
        import warnings
        warnings.warn(f"[functai] {route.provider}:{route.model} does not take {', '.join(drop)}; "
                      f"left out of its requests", stacklevel=2)
    return {**settings, **{k: None for k in drop}, "_dropped": tuple(drop)}


def model_capabilities(settings: Dict[str, Any], route: Any) -> Dict[str, bool]:
    declared = getattr(route, "capabilities", None)
    if declared is not None:
        return {**declared, **(settings.get("capabilities") or {})}
    caps = capabilities(route.provider, route.model)
    temperature = settings.get("temperature")
    if route.provider in ("anthropic", "claude-code") and caps.get("native_reasoning") and temperature not in (None, 1, 1.0):
        # Anthropic's extended thinking only runs at temperature 1. A caller who set
        # another temperature gets reasoning written in the reply instead (and, with
        # thinking off, Anthropic continues a prefill again).
        caps["native_reasoning"] = False
        caps["assistant_prefill"] = True
    return {**caps, **(settings.get("capabilities") or {})}
