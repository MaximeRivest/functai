"""
FunctAI — the function is the prompt, the body is the program.

Built on lmcc (how each value is written into the prompt and read back) and
lm15 (every provider, one wire).

API:
- Decorator: @ai        (bare or with options)
- Sentinel:  _ai        (bare, or _ai["description"] for extra outputs)
- Defaults:  configure(...) (process-wide, or `with configure(...):` for a block)
- Accounts:  login("claude"), logins(), logout(...)  (subscriptions, OpenRouter, API keys)
- Templates: system(...), user(...), assistant(...), developer(...), turns()  → @ai(template=[...])
- Programs:  @module; fn.opt(...), fn.map(table), and the optimizers
- Evaluation: evaluate(fn, data, metric) → Evaluation (.score, .summary, .table); compare(a, b); runs(folder)
- Saving:    check(program), save(program, path), verify(path), load(path), file("data.txt")
- Utils:     phistory(), inspect_history(), clear_cache()
"""

from .accounts import login, login_methods, logins, logout
from .adapters import (assistant, chat_adapter, developer, json_adapter, system, template_adapter,  # noqa: F401
                       turns, user, xml_adapter)
from .core import (
    UNSET,
    FunctAIFunc,
    ProgramState,
    _ai,
    ai,
    compute_signature,
    configure,
    docments,
    docstring,
    extract_docstrings,
    flexiclass,
    get_dataclass_source,
    get_name,
    get_source,
    inspect_history,
    isdataclass,
    parse_docstring,
    phistory,
    qual_name,
    settings,
    sig2str,
    signature_text,
)
from .data import Prediction
from .engine import LoginRequired, StepLimit, clear_cache, clear_history  # noqa: F401
from .evaluation import Evaluation, compare, evaluate, exact_match, runs
from .graph import Problem, Refused, Report, check
from .module import FunctAIModule, module
from .saved import LoadRefused, Verification, file, load, save, verify
from .optimizers import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    InstructionSearch,
    LabeledFewShot,
    Optimizer,
)

__version__ = "1.0.0"

# `from functai import *` brings the decorator, the sentinel and the program
# vocabulary. The template helpers (system, user, ...) are generic names, so
# they are imported explicitly: `from functai import system, user, turns`.
__all__ = [
    "ai",
    "_ai",
    "configure",
    "login",
    "logins",
    "logout",
    "login_methods",
    "LoginRequired",
    "phistory",
    "inspect_history",
    "settings",
    "compute_signature",
    "signature_text",
    "module",
    "FunctAIModule",
    "check",
    "save",
    "load",
    "verify",
    "file",
    "Report",
    "Problem",
    "Refused",
    "LoadRefused",
    "Verification",
    "FunctAIFunc",
    "ProgramState",
    "Prediction",
    "StepLimit",
    "evaluate",
    "Evaluation",
    "compare",
    "runs",
    "exact_match",
    "Optimizer",
    "LabeledFewShot",
    "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch",
    "InstructionSearch",
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
