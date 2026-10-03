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
- Call log:  configure(log_calls=True, log_content={"transcript": False}); calls(fn), rate(prediction, "right"),
             rated(fn); fn.version
- Interfaces: fn.interface, module.interface (checked on every call: InterfaceError); JSON; describe(path)
- Streaming: fn.stream(...) → Stream (for piece in s; s.events(); s.result; await s)
- Event logs: configure(observers=[...], journal=Journal(store, required=True)); MemoryStore; Store; Follower;
             flush() (function observers and best-effort journals catch up)
- Conversations: fn.conversation(id, store="folder/", context=last_turns(10)) → Conversation (chat(...),
             chat.turns, chat.continue_from(turn), chat.render(...), turn.approve()); all_turns, last_turns,
             remember, earlier(); FolderStore
- Tools:     @tool(effects="reads"|"changes"); approve=ask or "changes"; Waiting, ApprovalError
- Plugins:   Plugin(name, version=...) with hooks (turn_start, context, before_call, request, tool_call,
             tool_result, turn_end) returning Change(...); configure(plugins=[...]); compaction(), delegate()
- Serving:   serve(program), Service; remote(url, key=...)
- Long runs: configure(cache_replies="disk"), replicate=; fn.map(rows, threads=8) (a progress line; run again to
             resume); quotes_found(text, quotes); prune_calls(older_than="90d"); split(rows, by="conversation")
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
from .errors import (ApprovalError, ConversationError, EventRefused, FunctAIError, InterfaceError, JournalError,
                     LogContentError, Outcome, SawError, ServeError, Waiting)
from .eventlog import Follower, Journal, MemoryStore, Store, flush
from .interface import JSON
from .streaming import Cancelled, Stream
from .conversations import Conversation, Turn, all_turns, earlier, last_turns, remember
from .stores import FolderStore, MemoryConversations
from .tools import Approval, Tool, tool
from .plugins import Change, Plugin, PluginError, load_plugin
from .builtins import compaction, delegate
from .serving import Service, serve
from .remote import RemoteProgram, remote
from .judges import quotes_found
from .calllog import prune_calls, split
from .data import Prediction
from .engine import LoginRequired, StepLimit, clear_cache, clear_history  # noqa: F401
from . import datasets  # noqa: F401  (functai.datasets.tickets(), ...)
from .evaluation import Evaluation, compare, evaluate, exact_match, runs
from .graph import Problem, Refused, Report, check
from .module import FunctAIModule, module
from .saved import LoadRefused, Verification, describe, file, load, save, verify
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

__version__ = "1.2.0"

# `from functai import *` brings the decorator, the sentinel and the program
# vocabulary. The template helpers (system, user, ...) are generic names, so
# they are imported explicitly: `from functai import system, user, turns`;
# other helpers (docments, flexiclass, parse_docstring, compute_signature,
# signature_text, ...) stay reachable as `functai.<name>`.
__all__ = [
    "ai", "_ai", "configure", "login", "logins", "logout", "login_methods", "LoginRequired",
    "phistory", "inspect_history",
    "module", "FunctAIModule", "FunctAIFunc", "ProgramState", "Prediction", "StepLimit",
    "check", "save", "load", "verify", "describe", "file", "Report", "Problem", "Refused", "LoadRefused",
    "Verification",
    "evaluate", "Evaluation", "compare", "runs", "exact_match",
    "calls", "rate", "rated",
    "JSON", "InterfaceError", "LogContentError", "JournalError", "EventRefused", "SawError", "FunctAIError", "Outcome",
    "Stream", "Cancelled", "Journal", "MemoryStore", "Store", "Follower", "flush",
    "Conversation", "Turn", "all_turns", "last_turns", "remember", "earlier", "FolderStore", "MemoryConversations",
    "tool", "Tool", "Approval", "Waiting", "ApprovalError", "ConversationError", "ServeError",
    "Plugin", "Change", "PluginError", "load_plugin", "compaction", "delegate",
    "Service", "serve", "remote", "RemoteProgram", "quotes_found", "prune_calls", "split",
    "labeled_few_shot", "bootstrap_few_shot", "gepa",
    "Optimizer", "LabeledFewShot", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "InstructionSearch", "GEPA",
]
