"""
FunctAI — DSPy-powered, single-call AI functions.

New API:
- Decorator: @ai  (works bare or with options)
- Sentinel:  _ai  (bare; docstring + return type drive behavior)
- Defaults:  configure(...), defaults(...)
- Utils:     format_prompt(...), inspect_history_text()
"""

from .core import (
    ai,
    _ai,
    configure,
    defaults,
    format_prompt,
    inspect_history_text,
    settings,
)

__version__ = "0.2.0"

__all__ = [
    "ai",
    "_ai",
    "configure",
    "defaults",
    "format_prompt",
    "inspect_history_text",
    "settings",
]
