"""Sign in once, from functai: subscriptions (Claude, ChatGPT, GitHub Copilot,
xAI, Kimi Code), OpenRouter, or an API key for any provider.

    import functai
    functai.login("claude")                  # sign in; saved for every later session
    functai.configure(lm="claude:claude-sonnet-4-5")
    functai.logins()                         # everything you can use, and how
    functai.logout("claude")

Logins are lm15's managed logins, saved in lm15's one credentials file
(``~/.config/lm15/credentials.json``, or ``$LM15_CREDENTIALS_PATH``), which
lm15 in every language, and tools built on it, share. functai only picks
which one a call uses (``models.resolve``):

1. ``api_key=`` in functai's settings, when given;
2. the saved login for the model's provider, when there is one;
3. otherwise the environment (``OPENAI_API_KEY``, ...) and the Claude Code /
   Codex CLI logins on this machine, read in place.

A saved login that expired or was signed out is an error that says how to
sign in again: it never falls back to a metered key behind your back.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lm15
from lm15 import auth as _lm15_auth
from lm15.login import Auth, SelectOption, SelectPrompt, TerminalUI, TextPrompt
from lm15.registry import PROVIDERS

# Friendly names, for login("...") and model strings ("claude:claude-sonnet-4-5").
ALIASES: Dict[str, str] = {
    "claude": "claude-code",
    "chatgpt": "openai-codex",
    "codex": "openai-codex",
    "copilot": "github-copilot",
    "github": "github-copilot",
    "kimi": "kimi-code",
    "grok": "xai",
}

# The accounts people sign in to (not API keys), in the order a picker shows them.
ACCOUNTS: Tuple[Tuple[str, str], ...] = (
    ("claude-code", "Claude (Pro/Max subscription)"),
    ("openai-codex", "ChatGPT (Plus/Pro subscription)"),
    ("github-copilot", "GitHub Copilot"),
    ("xai", "xAI / Grok (subscription)"),
    ("kimi-code", "Kimi Code (subscription)"),
    ("openrouter", "OpenRouter (approve in the browser; spends OpenRouter credits)"),
)

# A model to suggest after signing in: what to write in configure(lm=...).
EXAMPLE_MODEL: Dict[str, str] = {
    "claude-code": "claude:claude-sonnet-4-5",
    "openai-codex": "chatgpt:gpt-5.5",
    "github-copilot": "copilot:gpt-4.1",
    "xai": "grok-4",
    "openrouter": "openrouter:openai/gpt-4.1-mini",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "groq": "groq:openai/gpt-oss-120b",
}

# The CLI logins lm15 reads in place, when the CLI is signed in on this machine.
CLI_LOGINS: Dict[str, Tuple[str, Path]] = {
    "claude-code": ("Claude Code CLI", Path(_lm15_auth.CLAUDE_CODE_CREDENTIALS_PATH)),
    "openai-codex": ("Codex CLI", Path(_lm15_auth.CODEX_CLI_AUTH_PATH)),
}


def canonical(provider: str) -> str:
    """``"claude"`` → ``"claude-code"``; lm15 names pass through."""
    key = provider.strip().lower()
    return ALIASES.get(key, key)


# ------------------------------------------------------------------ the store


_lock = threading.Lock()
_auths: Dict[Any, Auth] = {}


def auth_for(setting: Any) -> Optional[Auth]:
    """The lm15 ``Auth`` an ``auth`` setting names: None/True → the default
    store; a path → that store; an ``Auth`` as is; False → none."""
    if setting is False:
        return None
    if isinstance(setting, Auth):
        return setting
    if setting is None or setting is True:
        key: Any = ("default", os.environ.get("LM15_CREDENTIALS_PATH"))
        path = None
    elif isinstance(setting, (str, os.PathLike)):
        key = ("path", str(Path(setting).expanduser()))
        path = Path(setting).expanduser()
    else:
        raise TypeError(f"auth must be True, False, a credentials file path, or an lm15 login.Auth, "
                        f"not {type(setting).__name__}")
    with _lock:
        a = _auths.get(key)
        if a is None:
            a = _auths[key] = Auth.local(path)
        return a


def store_path(a: Auth) -> Optional[Path]:
    p = getattr(a.store, "path", None)
    return Path(p) if p is not None else None


_saved_cache: Dict[Tuple, frozenset] = {}


def saved_providers(a: Auth) -> frozenset:
    """Providers with a saved, not signed-out connection in this store (cached
    until the file changes)."""
    path = store_path(a)
    try:
        stamp = (id(a), path.stat().st_mtime_ns if path else None)
    except FileNotFoundError:
        return frozenset()
    if path is not None and stamp in _saved_cache:
        return _saved_cache[stamp]
    found = frozenset(route for c in a.connections() for route in (c.routes or (c.provider,)))
    if path is not None:
        _saved_cache.clear()
        _saved_cache[stamp] = found
    return found


# ------------------------------------------------------------------ records


@dataclasses.dataclass(frozen=True)
class Login:
    """One way functai can reach a provider."""
    provider: str
    label: str
    source: str                 # "saved login" | "saved key" | "CLI login" | "environment" | "local server"
    status: str                 # "ready" | "renewal due" | "needs login" | "found" | ...
    expires: Optional[str] = None
    example: Optional[str] = None

    def __repr__(self) -> str:
        return f"Login({self.label}: {self.source}, {self.status}" + \
            (f", until {self.expires}" if self.expires and self.expires != "never" else "") + ")"


class Logins(list):
    """``functai.logins()``: a list of ``Login`` that prints as a table."""

    def __repr__(self) -> str:
        if not self:
            return "(no logins, saved keys or API keys found; functai.login() to add one)"
        rows = [("provider", "how", "status", "try")]
        for x in self:
            status = x.status + (f" until {x.expires}" if x.expires and x.expires not in ("never", "unknown") else "")
            rows.append((x.label if x.label != x.provider else x.provider, x.source, status, x.example or ""))
        widths = [max(len(r[i]) for r in rows) for i in range(4)]
        lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
        lines.insert(1, "  ".join("─" * w for w in widths))
        return "\n".join(lines)

    __str__ = __repr__


def _label(provider: str) -> str:
    for p, label in ACCOUNTS:
        if p == provider:
            return label.split(" (")[0]
    return provider


def _expires(value: Optional[str]) -> Optional[str]:
    if not value or value in ("never", "unknown"):
        return value
    return value[:16].replace("T", " ") + (" UTC" if value.endswith("Z") else "")


def logins(*, auth: Any = None) -> Logins:
    """Everything functai can use right now: saved logins and keys, CLI logins
    found on this machine, and API keys in the environment. Reads files and
    environment variables only; no network."""
    a = auth_for(True if auth is None else auth)
    out = Logins()
    seen = set()
    if a is not None:
        for c in a.connections():
            st = a.status(c.provider)
            source = "saved key" if c.kind == "api_key" else "saved login"
            if c.method_id == "env":
                source = "saved: use env key"
            elif (c.method_id or "").startswith("external:"):
                source = "saved: use CLI login"
            out.append(Login(c.provider, _label(c.provider), source, st.usability.replace("_", " "),
                             _expires(st.expires_at), EXAMPLE_MODEL.get(c.provider)))
            seen.add(c.provider)
    for provider, (name, path) in CLI_LOGINS.items():
        if provider not in seen and path.exists():
            out.append(Login(provider, _label(provider), name, "found", None, EXAMPLE_MODEL.get(provider)))
            seen.add(provider)
    keys_seen = set()
    for provider, d in PROVIDERS.items():
        if provider in seen or getattr(d, "placeholder_key", None):
            continue
        keys = [k for k in (getattr(d, "env_keys", None) or ()) if os.environ.get(k) and k not in keys_seen]
        keys_seen.update(getattr(d, "env_keys", None) or ())    # one row per key: openai-chat shares openai's
        if keys:
            out.append(Login(provider, provider, f"environment (${keys[0]})", "found", None,
                             EXAMPLE_MODEL.get(provider)))
    return out


# ------------------------------------------------------------------ login / logout


def _has_display() -> bool:
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _ui(open_browser: Optional[bool]) -> TerminalUI:
    return TerminalUI(out=sys.stdout, open_browser=_has_display() if open_browser is None else open_browser)


def _say(text: str) -> None:
    print(text, flush=True)


def _choose_method(a: Auth, provider: str, method: Optional[str]) -> Tuple[Optional[str], bool]:
    """``(method id, unverified)``. Prefers a proven way; for Claude and ChatGPT,
    the CLI login on this machine when there is one. A provider whose only
    account login lm15 marks unverified gets it, said out loud."""
    methods = [m for m in a.methods(provider) if m.availability != "unavailable"]
    if method is not None:
        chosen = next((m for m in methods if m.id == method), None)
        if chosen is None:
            raise ValueError(f"{provider}: no login method {method!r}; one of {[m.id for m in methods]}")
        return chosen.id, chosen.availability == "unverified"
    cli = CLI_LOGINS.get(provider)
    accounts = [m for m in methods if m.kind == "account"]
    usable = [m for m in accounts if not m.id.startswith("external:") or (cli and cli[1].exists())]
    if cli and cli[1].exists():
        ext = next((m for m in accounts if m.id.startswith("external:")), None)
        if ext is not None:
            return ext.id, False
    proven = [m for m in usable if m.availability == "supported" and not m.id.startswith("external:")]
    if proven:
        return proven[0].id, False
    others = [m for m in usable if not m.id.startswith("external:")]
    if len(others) == 1:
        return others[0].id, others[0].availability == "unverified"
    if others:
        return None, any(m.availability == "unverified" for m in others)      # the UI asks
    if any(m.kind == "api_key" and m.id == "api_key" for m in methods):
        return "api_key", False
    local = next((m for m in methods if m.kind == "local_server"), None)
    if local is not None:
        return local.id, False
    raise ValueError(f"{provider}: lm15 has no way to sign in here")


def login(provider: Optional[str] = None, *, key: Optional[str] = None, method: Optional[str] = None,
          again: bool = False, open_browser: Optional[bool] = None, auth: Any = None) -> Login:
    """Sign in to a provider once; functai (and lm15) use it from then on.

    - ``login("claude")``, ``login("chatgpt")``, ``login("copilot")``, ``login("grok")``,
      ``login("kimi")``, ``login("openrouter")``: an account. For Claude and ChatGPT,
      a Claude Code / Codex CLI already signed in on this machine is used as is.
    - ``login("openai")`` asks for an API key; ``login("groq", key="gsk-...")`` saves one.
    - ``login()`` asks which.

    Already signed in: says so and does nothing, unless ``again=True``.
    ``method`` picks a specific lm15 login method (see ``functai.login_methods``).
    ``open_browser``: default, when this machine has a display."""
    a = auth_for(True if auth is None else auth)
    if a is None:
        raise ValueError("login needs a credentials store; auth=False has none")
    ui = _ui(open_browser)
    if provider is None:
        options = tuple(SelectOption(p, label) for p, label in ACCOUNTS) + (
            SelectOption("api_key", "An API key for another provider (OpenAI, Anthropic, Gemini, Groq, ...)"),)
        provider = ui.prompt(SelectPrompt("provider", "Sign in to:", options))
        if provider == "api_key":
            provider = ui.prompt(TextPrompt("provider", "Provider (e.g. openai, anthropic, gemini, groq)"))
    asked = provider
    provider = canonical(provider)
    try:
        a.descriptor(provider)
    except lm15.AuthOperationError:
        known = sorted({*ALIASES, *(d.id for d in a.providers() if d.methods)})
        raise ValueError(f"unknown provider {provider!r}; one of: {', '.join(known)}") from None
    current = a.status(provider)
    replace = current.connection.id if current.connection is not None else None
    if replace and not again and key is None and method is None and current.usability in ("ready", "renewal_due"):
        record = next(x for x in logins(auth=a) if x.provider == provider)
        _say(f"Already signed in: {record!r}. Use functai.login({asked!r}, again=True) to sign in again.")
        return record
    if key is not None:
        a.set_api_key(provider, key, replace=replace)
        _say(f"Saved the {provider} API key." + _try_hint(provider))
        return next(x for x in logins(auth=a) if x.provider == provider)
    chosen, unverified = _choose_method(a, provider, method)
    if unverified:
        note = next((m.reason for m in a.methods(provider) if m.id == chosen), None) if chosen else None
        _say(f"Note: lm15 has not yet certified this {_label(provider)} sign-in"
             + (f" ({note})" if note else "") + ". Whether an account may be used this way, "
             "and how it is billed, is the provider's decision.")
    if chosen and chosen.startswith("external:"):
        name = CLI_LOGINS[provider][0]
        conn = a.configure(provider, method=chosen, replace=replace)
        _say(f"Using your {name} login for {_label(provider)} (read in place, renewed by lm15)."
             + _try_hint(provider))
    else:
        conn = a.login(provider, chosen, ui=ui, replace=replace, allow_unverified=unverified)
        _say(f"Signed in: {conn.label or _label(provider)}." + _try_hint(provider))
    return next(x for x in logins(auth=a) if x.provider == provider)


def _try_hint(provider: str) -> str:
    example = EXAMPLE_MODEL.get(provider)
    return f" Try: functai.configure(lm={example!r})" if example else ""


def logout(provider: str, *, auth: Any = None) -> None:
    """Forget the saved login or key for a provider, on this machine (the account
    itself is untouched). API keys in the environment still work afterwards, except
    where lm15 blocks them on purpose: a signed-out xAI subscription does not fall
    back to ``XAI_API_KEY``."""
    a = auth_for(True if auth is None else auth)
    if a is None:
        raise ValueError("logout needs a credentials store; auth=False has none")
    provider = canonical(provider)
    if a.status(provider).connection is None:
        _say(f"Not signed in to {provider}.")
        return
    a.logout(provider)
    cli = CLI_LOGINS.get(provider)
    still = (f" Your {cli[0]} is still signed in on this machine, and functai still uses it; "
             f"sign out there to stop that.") if cli and cli[1].exists() else ""
    _say(f"Signed out of {_label(provider)}." + still)


def login_methods(provider: str) -> List[Dict[str, str]]:
    """The ways lm15 can sign in to a provider, and whether each is proven."""
    a = auth_for(True)
    return [{"method": m.id, "kind": m.kind, "status": m.availability, **({"note": m.reason} if m.reason else {})}
            for m in a.methods(canonical(provider))]


__all__ = ["login", "logout", "logins", "login_methods", "Login", "canonical", "ALIASES"]
