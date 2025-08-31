"""
FunctAI — DSPy-powered, single-call AI functions.

API:
- Decorator: @ai        (bare or with options)
- Sentinel:  _ai        (bare; docstring + type hints drive outputs)
- Defaults:  configure(...) (global and context manager)
- Utils:     inspect_history_text()
"""

from .core import (
    ai,
    _ai,
    configure,
    inspect_history_text,
    settings,
    compute_signature,
    signature_text,
)
from .module import module, FunctAIModule

__version__ = "0.7.0"

__all__ = [
    "ai",
    "_ai",
    "configure",
    "inspect_history_text",
    "settings",
    "compute_signature",
    "signature_text",
    "module",
    "FunctAIModule",
]
