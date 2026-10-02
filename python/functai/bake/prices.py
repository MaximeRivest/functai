"""Prices bake shows in a plan: what a teacher's answers and a service's
training will cost, before any of it is spent.

Language model prices come from models.dev (a public catalog of providers'
prices, kept by the opencode project); Tinker's from its own published table.
Both are read once a day and kept in ``~/.cache/functai/prices``; offline, the
last copy is used, and with none a price is shown as unknown. A plan never
waits on them for more than a few seconds; ``FUNCTAI_PRICES=off`` never asks.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

MODELS_DEV = "https://models.dev/api.json"
TINKER_MODELS = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
DAY = 86400

# functai's provider names → models.dev's
_PROVIDERS = {"gemini": "google", "anthropic": "anthropic", "openai": "openai", "xai": "xai",
              "mistral": "mistral", "groq": "groq", "deepseek": "deepseek", "openrouter": "openrouter",
              "together": "togetherai", "fireworks": "fireworks-ai", "openai-codex": "openai",
              "claude-code": "anthropic"}


def _home() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "functai" / "prices"


def _fetch(url: str, name: str, *, max_age: float = DAY, timeout: float = 4.0) -> Optional[Any]:
    path = _home() / name
    if os.environ.get("FUNCTAI_PRICES") == "off":       # never ask the network (tests, air-gapped machines)
        try:
            return json.loads(path.read_text()) if path.exists() else None
        except ValueError:
            return None
    if path.exists() and time.time() - path.stat().st_mtime < max_age:
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "functai-bake"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
        return data
    except Exception:  # noqa: BLE001 — offline: the last copy, else unknown
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                return None
        return None


def _names(model: str):
    yield model
    yield model.replace(".", "-")
    yield re.sub(r"-\d{8}$", "", model)
    yield re.sub(r"-latest$", "", model)


def model_price(provider: Optional[str], model: str) -> Optional[Tuple[float, float]]:
    """(dollars per million input tokens, per million output tokens) for a
    language model, or None when the catalog does not know it (or it is billed
    by subscription: ``openai-codex``, ``claude-code``)."""
    if provider in ("openai-codex", "claude-code"):
        return None          # a subscription: no per-token bill
    data = _fetch(MODELS_DEV, "models.dev.json")
    if not data:
        return None
    if "/" in model and provider is None:
        provider, model = model.split("/", 1)
    keys = [_PROVIDERS.get(provider or "", provider)] if provider else list(data)
    for key in keys:
        models = (data.get(key) or {}).get("models") or {}
        for name in _names(model):
            cost = (models.get(name) or {}).get("cost")
            if cost and cost.get("input") is not None:
                return float(cost["input"]), float(cost.get("output") or 0.0)
    return None


def _dollars(text: Any) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def tinker_models() -> Dict[str, Dict[str, Any]]:
    """Tinker's trainable models: ``{tinker_id: {"train", "sample", "prefill", "context"}}``
    (dollars per million tokens; context in tokens)."""
    data = _fetch(TINKER_MODELS, "tinker-models.json") or []
    out = {}
    for m in data if isinstance(data, list) else data.get("models", []):
        ctx = str(m.get("context") or "")
        mult = 1024 if ctx.upper().endswith("K") else 1
        try:
            context = int(float(ctx.rstrip("Kk")) * mult) if ctx else None
        except ValueError:
            context = None
        out[m["tinker_id"]] = {"train": _dollars(m.get("train")), "sample": _dollars(m.get("sample")),
                               "prefill": _dollars(m.get("prefill")), "context": context}
    return out


__all__ = ["model_price", "tinker_models"]
