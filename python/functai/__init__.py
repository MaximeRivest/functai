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
- Call log:  configure(log_calls=True); calls(fn), rate(prediction, "right"), rated(fn); fn.version
- Streaming: fn.stream(...) → Stream (for piece in s; s.events(); s.result; await s)
- Utils:     phistory(), inspect_history(), clear_cache()
- Data:      datasets.tickets(), datasets.field_notes()  (small labelled tables to learn with)
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
from .calllog import calls, rate, rated
from .streaming import Cancelled, Stream
from .data import Prediction
from .engine import LoginRequired, StepLimit, clear_cache, clear_history  # noqa: F401
from . import datasets  # noqa: F401  (functai.datasets.tickets(), ...)
from .evaluation import Evaluation, compare, evaluate, exact_match, runs
from .graph import Problem, Refused, Report, check
from .module import FunctAIModule, module
from .saved import LoadRefused, Verification, file, load, save, verify
from .optimizers import (
    BootstrapFewShot,
    BootstrapFewShotWithRandomSearch,
    InstructionSearch,
    GEPA,
    LabeledFewShot,
    Optimizer,
    bootstrap_few_shot,
    gepa,
    labeled_few_shot,
)

__version__ = "1.1.0"

# `from functai import *` brings the decorator, the sentinel and the program
# vocabulary. The template helpers (system, user, ...) are generic names, so
# they are imported explicitly: `from functai import system, user, turns`;
# other helpers (docments, flexiclass, parse_docstring, compute_signature,
# signature_text, ...) stay reachable as `functai.<name>`.
__all__ = [
    "ai", "_ai", "configure", "login", "logins", "logout", "login_methods", "LoginRequired",
    "phistory", "inspect_history",
    "module", "FunctAIModule", "FunctAIFunc", "ProgramState", "Prediction", "StepLimit",
    "check", "save", "load", "verify", "file", "Report", "Problem", "Refused", "LoadRefused", "Verification",
    "evaluate", "Evaluation", "compare", "runs", "exact_match",
    "calls", "rate", "rated",
    "Stream", "Cancelled",
    "labeled_few_shot", "bootstrap_few_shot", "gepa",
    "Optimizer", "LabeledFewShot", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "InstructionSearch", "GEPA",
]
