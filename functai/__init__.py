"""
FunctAI — DSPy-powered, single-call AI functions.

API:
- Decorator: @ai        (bare or with options)
- Sentinel:  _ai        (bare; docstring + type hints drive outputs)
- Defaults:  configure(...) (global and context manager)
- Utils:     format_prompt(...), inspect_history_text()
"""

from .core import (
    ai,
    _ai,
    configure,
    format_prompt,
    inspect_history_text,
    settings,
    compute_signature,
    signature_text,
)

__version__ = "0.3.0"

__all__ = [
    "ai",
    "_ai",
    "configure",
    "format_prompt",
    "inspect_history_text",
    "settings",
    "compute_signature",
    "signature_text",
]
