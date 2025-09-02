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
    # NEW exports
    flexiclass,
    UNSET,
    docstring,
    parse_docstring,
    docments,
    isdataclass,
    get_dataclass_source,
    get_source,
    get_name,
    qual_name,
    sig2str,
    extract_docstrings,
)
from .module import module, FunctAIModule

__version__ = "0.8.1"

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
    # NEW
    "flexiclass",
    "UNSET",
    "docstring",
    "parse_docstring",
    "docments",
    "isdataclass",
    "get_dataclass_source",
    "get_source",
    "get_name",
    "qual_name",
    "sig2str",
    "extract_docstrings",
]
