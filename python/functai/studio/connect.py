"""AI accounts, from the dashboard: what is connected, adding an API key, signing in
with a subscription, signing out, and which models each connection can use.

Everything goes through FunctAI's logins (``functai.accounts``), which are lm15's
managed logins: saved in lm15's one credentials file (``~/.config/lm15/credentials.json``,
or ``$LM15_CREDENTIALS_PATH``, private to the user), shared by every lm15 program and
every FunctAI script on the computer. Studio never stores a key itself, never sends
one back to the browser, and accepts one only over a connection from this computer
(the server checks; a tunnel such as ``tailscale serve`` counts, plain http from the
network does not).

A sign-in (Claude, ChatGPT, GitHub Copilot, xAI, Kimi, OpenRouter) runs lm15's own
flow; this module is the "UI" lm15 talks to (``lm15.login.AuthUI``): the links,
device codes and questions of the flow go to the page, the person's answers come
back. One sign-in at a time.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .project import StudioError

# The companies and subscriptions a person picks from, in this order, with what to say about each.
SUBSCRIPTIONS: Tuple[Tuple[str, str, str], ...] = (
    ("claude-code", "Claude", "your Claude Pro or Max subscription: no extra cost per use, within its limits"),
    ("openai-codex", "ChatGPT", "your ChatGPT Plus or Pro subscription: no extra cost per use, within its limits"),
    ("github-copilot", "GitHub Copilot", "your Copilot subscription"),
    ("xai", "xAI / Grok", "your xAI subscription"),
    ("kimi-code", "Kimi Code", "your Kimi Code subscription"),
    ("openrouter", "OpenRouter", "one account for many companies' models, paid with OpenRouter credits"),
)
KEY_PROVIDERS: Tuple[Tuple[str, str, str], ...] = (
    ("anthropic", "Anthropic", "https://console.anthropic.com/settings/keys"),
    ("openai", "OpenAI", "https://platform.openai.com/api-keys"),
    ("gemini", "Google (Gemini)", "https://aistudio.google.com/apikey"),
    ("openrouter", "OpenRouter", "https://openrouter.ai/settings/keys"),
    ("groq", "Groq", "https://console.groq.com/keys"),
    ("xai", "xAI", "https://console.x.ai"),
    ("deepseek", "DeepSeek", "https://platform.deepseek.com/api_keys"),
    ("mistral", "Mistral", "https://console.mistral.ai/api-keys"),
)
LABELS = {p: label for p, label, _ in SUBSCRIPTIONS} | {p: label for p, label, _ in KEY_PROVIDERS} | {
    "ollama": "This computer (Ollama)", "lmstudio": "This computer (LM Studio)", "meta": "Meta", "zai": "Z.AI",
    "moonshotai": "Moonshot (Kimi)", "deepinfra": "DeepInfra", "together": "Together", "fireworks": "Fireworks",
    "parasail": "Parasail", "typesafe": "TypeSafe", "azure": "Azure", "bedrock-chat": "Amazon Bedrock",
    "vertex-express": "Google Vertex"}
# Other doors to the same company with the same key: shown once, under the company
SAME_KEY = ("-chat", "-anthropic", "-responses")
SUBSCRIPTION_ROUTES = {"claude-code", "openai-codex", "github-copilot", "kimi-code"}
LOCAL_ROUTES = {"ollama", "lmstudio", "llamacpp", "vllm", "local"}

# Models to offer for each connection, newest first. Checked against the providers' own model
# lists on 2026-11-02; "another model…" in the dashboard takes any other name.
CATALOGUE: Dict[str, Tuple[str, ...]] = {
    "anthropic": ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"),
    "openai": ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.1-sol"),
    "gemini": ("gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-pro-preview"),
    "claude-code": ("claude:claude-haiku-5-5", "claude:claude-sonnet-5-5", "claude:claude-opus-5-5"),
    "openai-codex": ("chatgpt:gpt-6-luna", "chatgpt:gpt-6.1-terra", "chatgpt:gpt-6.1-sol"),
    "openrouter": ("openrouter:anthropic/claude-haiku-5.5", "openrouter:openai/gpt-6-luna",
                   "openrouter:google/gemini-3.5-flash-lite", "openrouter:anthropic/claude-sonnet-5.5"),
    "groq": ("groq:openai/gpt-oss-120b",),
}
# The model Studio picks for a new step when the person has not chosen: small, fast and cheap,
# from the first connection they have, in this order (pay-per-use keys first, then subscriptions).
PREFERRED = ("anthropic", "openai", "gemini", "openrouter", "claude-code", "openai-codex", "groq")


# ------------------------------------------------------------------ what is connected

def _auth():
    from .. import accounts
    return accounts.auth_for(True)


def connections() -> List[Dict[str, Any]]:
    """Every way this computer can reach a model right now, for the page (no secrets)."""
    from .. import accounts
    out: List[Dict[str, Any]] = []
    found = accounts.logins()
    names = {x.provider for x in found}
    for x in found:
        if any(x.provider.endswith(t) and x.provider[: -len(t)] in names for t in SAME_KEY):
            continue
        ready = x.status in ("ready", "found") or x.status.startswith("ready") or x.status == "renewal due"
        how = x.source
        if how == "saved key":
            how = "a key saved here"
        elif how == "saved login":
            how = "signed in here"
        elif how.startswith("saved: use CLI") or how.endswith("CLI"):
            how = "through " + {"claude-code": "Claude Code", "openai-codex": "Codex"}.get(x.provider, "its app") + \
                  "'s sign-in on this computer"
        elif how.startswith("environment ("):
            how = "a key set in this computer's settings (" + how[len("environment ($"):].rstrip(")") + ")"
        elif how == "saved: use env key":
            how = "a key set in this computer's settings"
        out.append({"provider": x.provider, "label": LABELS.get(x.provider, x.label), "how": how,
                    "status": "ready" if ready else x.status, "expires": x.expires,
                    "kind": "subscription" if x.provider in SUBSCRIPTION_ROUTES or x.source.endswith("login")
                    else "key", "can_sign_out": x.source.startswith("saved"),
                    "models": list(CATALOGUE.get(x.provider, ()))})
    return out


def usable() -> Dict[str, Dict[str, Any]]:
    return {c["provider"]: c for c in connections() if c["status"] == "ready"}


_PROVIDER_CACHE: Dict[str, str] = {}


def provider_of(model: Optional[str]) -> Optional[str]:
    """Which connection a model name goes through (``claude:…`` → claude-code)."""
    if not model:
        return None
    if model in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[model]
    try:
        from .. import models
        prov = models.resolve({"lm": model})[2].provider
    except Exception:  # noqa: BLE001 — an unknown name: say so where it is shown
        prov = model.split(":", 1)[0] if ":" in model else None
    if prov:
        _PROVIDER_CACHE[model] = prov
    return prov


def model_status(model: Optional[str], ready: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Whether a model can run here, through what, and who pays."""
    ready = usable() if ready is None else ready
    prov = provider_of(model)
    if prov in LOCAL_ROUTES:
        return {"provider": prov, "label": LABELS.get(prov, "this computer"), "ready": True, "pays": "nobody",
                "local": True}
    conn = ready.get(prov or "")
    sub = prov in SUBSCRIPTION_ROUTES
    return {"provider": prov, "label": LABELS.get(prov or "", prov or "unknown"), "ready": conn is not None,
            "how": conn["how"] if conn else None,
            "pays": ("your subscription" if sub else "your account, per use") if conn else None,
            "subscription": sub}


def catalogue(ready: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The models to offer: those of connected accounts first, then the rest (marked not connected)."""
    ready = usable() if ready is None else ready
    out, seen = [], set()
    order = [p for p in PREFERRED if p in ready] + [p for p in ready if p not in PREFERRED] + \
        [p for p in CATALOGUE if p not in ready]
    for prov in order:
        for m in CATALOGUE.get(prov, ()):
            if m not in seen:
                seen.add(m)
                out.append({"model": m, "provider": prov, "label": LABELS.get(prov, prov), "ready": prov in ready,
                            "subscription": prov in SUBSCRIPTION_ROUTES})
    return out


def default_model(explicit: Optional[str] = None) -> Optional[str]:
    """The model for steps that name none: the person's choice when it can run here, else the
    first small model of a connected account; None when nothing is connected."""
    ready = usable()
    if explicit and model_status(explicit, ready)["ready"]:
        return explicit
    for prov in PREFERRED:
        if prov in ready and CATALOGUE.get(prov):
            return CATALOGUE[prov][0]
    for prov in ready:
        if CATALOGUE.get(prov):
            return CATALOGUE[prov][0]
    return explicit


def can_add() -> Dict[str, Any]:
    """The ways to connect, for the page: subscriptions to sign in to, companies to give a key for."""
    from .. import accounts
    a = _auth()
    subs = []
    for prov, label, about in SUBSCRIPTIONS:
        try:
            methods = [m for m in a.methods(prov) if m.availability != "unavailable" and m.kind == "account"]
        except Exception:  # noqa: BLE001 — a provider this lm15 does not know
            continue
        cli = accounts.CLI_LOGINS.get(prov)
        # a CLI's own sign-in counts only when that CLI is signed in on this computer
        methods = [m for m in methods if not m.id.startswith("external:") or (cli and cli[1].exists())]
        if not methods:
            continue
        proven = any(m.availability == "supported" for m in methods)
        subs.append({"provider": prov, "label": label, "about": about, "certified": proven,
                     "note": None if proven else "lm15 has not yet certified this sign-in: it may work, and "
                     "whether an account may be used this way is the provider's decision"})
    return {"subscriptions": subs,
            "keys": [{"provider": p, "label": label, "get_one": url} for p, label, url in KEY_PROVIDERS]}


# ------------------------------------------------------------------ keys

def add_key(provider: str, key: str) -> Dict[str, Any]:
    """Save an API key in lm15's credentials file, then check it (a model list, no answer asked)."""
    from .. import accounts
    key = (key or "").strip()
    if len(key) < 8 or any(c.isspace() for c in key):
        raise StudioError("that does not look like an API key (it is usually one long word, often starting "
                          "with sk-)")
    a = _auth()
    prov = accounts.canonical(provider)
    try:
        a.descriptor(prov)
    except Exception:  # noqa: BLE001
        raise StudioError(f"unknown company {provider!r}") from None
    current = a.status(prov)
    a.set_api_key(prov, key, replace=current.connection.id if current.connection is not None else None)
    _PROVIDER_CACHE.clear()
    try:
        v = a.verify(prov)
        result, detail = v.result, v.detail
    except Exception as exc:  # noqa: BLE001 — the key is saved; the check is extra
        result, detail = "unverified", f"{type(exc).__name__}: {exc}"
    if result == "rejected":
        a.logout(prov)
        raise StudioError(f"{LABELS.get(prov, prov)} refused that key ({detail or 'not accepted'}), so it was not "
                          f"kept. Check that it was copied whole and that the account has credit.")
    return {"provider": prov, "label": LABELS.get(prov, prov), "checked": result == "valid",
            "ends_with": key[-4:]}


def sign_out(provider: str) -> Dict[str, Any]:
    from .. import accounts
    a = _auth()
    prov = accounts.canonical(provider)
    if a.status(prov).connection is None:
        raise StudioError(f"nothing saved for {LABELS.get(prov, prov)} (a key in the environment, or a "
                          f"program's own sign-in, is removed where it was set)")
    a.logout(prov)
    _PROVIDER_CACHE.clear()
    cli = accounts.CLI_LOGINS.get(prov)
    still = (f" Your {cli[0]} is still signed in on this computer, and is still used; sign out there to stop "
             f"that.") if cli and cli[1].exists() else ""
    return {"provider": prov, "say": f"Signed out of {LABELS.get(prov, prov)}." + still}


# ------------------------------------------------------------------ sign-ins, driven from the page

class Cancelled(Exception):
    pass


class WebLogin:
    """One sign-in in progress: lm15's flow talks to this object; the page reads it and answers."""

    def __init__(self, provider: str):
        self.id = uuid.uuid4().hex[:10]
        self.provider = provider
        self.label = LABELS.get(provider, provider)
        self.state = "starting"              # starting, waiting (for the person), working, done, failed, cancelled
        self.notices: List[Dict[str, Any]] = []
        self.prompt: Optional[Dict[str, Any]] = None
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self._answer: Optional[str] = None
        self._event = threading.Event()
        self._cancel = False
        self.started = time.time()

    # ---- lm15.login.AuthUI
    def prompt_(self, p: Any) -> str:
        if self._cancel:
            from lm15.login import LoginCancelled
            raise LoginCancelled("cancelled")
        d = {"type": p.type, "field": p.field_id, "label": p.label,
             "placeholder": getattr(p, "placeholder", None), "accepted": getattr(p, "accepted", None),
             "options": [{"id": o.id, "label": o.label, "description": o.description}
                         for o in getattr(p, "options", ())]}
        self.prompt, self._answer, self.state = d, None, "waiting"
        self._event.clear()
        while not self._event.wait(0.5):
            if self._cancel:
                from lm15.login import LoginCancelled
                raise LoginCancelled("cancelled")
            if time.time() - self.started > 15 * 60:
                from lm15.login import LoginCancelled
                raise LoginCancelled("timed out")
        self.prompt, self.state = None, "working"
        return self._answer or ""

    def notify(self, n: Any) -> None:
        d = {"type": n.type}
        for k in ("url", "instructions", "user_code", "verification_url", "expires_in_s", "stage", "message"):
            if hasattr(n, k):
                d[k] = getattr(n, k)
        if hasattr(n, "links"):
            d["links"] = [list(x) for x in n.links]
        self.notices.append(d)
        if self.state == "starting":
            self.state = "working"

    def dismiss(self, p: Any) -> None:
        if self.prompt and self.prompt.get("field") == getattr(p, "field_id", None):
            self.prompt = None
            self._event.set()

    def view(self) -> Dict[str, Any]:
        return {"id": self.id, "provider": self.provider, "label": self.label, "state": self.state,
                "notices": self.notices[-6:], "prompt": self.prompt, "result": self.result, "error": self.error}

    def answer(self, value: str) -> None:
        if self.prompt is None:
            raise StudioError("the sign-in is not waiting for an answer")
        self._answer = value
        self._event.set()

    def cancel(self) -> None:
        self._cancel = True
        self._event.set()


class _UI:
    """The object handed to lm15 (its prompt method is called ``prompt``)."""

    def __init__(self, w: WebLogin):
        self.w = w

    def prompt(self, p: Any) -> str:
        return self.w.prompt_(p)

    def notify(self, n: Any) -> None:
        self.w.notify(n)

    def dismiss(self, p: Any) -> None:
        self.w.dismiss(p)


class Logins:
    """The sign-ins of one Studio server (one at a time)."""

    def __init__(self) -> None:
        self.current: Optional[WebLogin] = None
        self.lock = threading.Lock()

    def start(self, provider: str, method: Optional[str] = None) -> WebLogin:
        from .. import accounts
        with self.lock:
            if self.current and self.current.state in ("starting", "waiting", "working"):
                if time.time() - self.current.started < 15 * 60:
                    raise StudioError(f"a sign-in to {self.current.label} is in progress: finish or cancel it first")
                self.current.cancel()
            w = self.current = WebLogin(accounts.canonical(provider))

        def run() -> None:
            said: List[str] = []
            try:
                accounts.login(w.provider, method=method, again=True, ui=_UI(w), say=said.append,
                               open_browser=False)
                w.result = " ".join(said) or f"Signed in to {w.label}."
                w.state = "done"
            except Exception as exc:  # noqa: BLE001 — shown to the person
                cancelled = type(exc).__name__ == "LoginCancelled" or w._cancel
                w.state = "cancelled" if cancelled else "failed"
                w.error = None if cancelled else f"{exc}"
            finally:
                _PROVIDER_CACHE.clear()
        threading.Thread(target=run, daemon=True).start()
        return w

    def get(self, login_id: str) -> WebLogin:
        if not self.current or self.current.id != login_id:
            raise StudioError("no such sign-in (it may have finished)")
        return self.current
