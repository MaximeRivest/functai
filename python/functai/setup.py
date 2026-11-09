"""What a program runs on (contract/calls.md, *Setups*; design/18).

A version says what a program **is**: its instruction, examples, layout,
code. Its **setup** says what it runs on: which model answers each step,
with which settings and plugins, with which tool code, able to reach what.
Two runs of one version on two models have two setups, so a score, a
comparison or a release belongs to exactly one thing that ran.

``fn.setup`` is ``"sha256:"`` of the canonical JSON of
``fn.setup_document()``, computed where the program runs, from the settings
in force there, before its first call and without the network. Every call
records it (``program.setup``), beside where it ran (``placement``).
"""

from __future__ import annotations

import copy
import contextvars
import json
import weakref
from typing import Any, Dict, List, Mapping, Optional, Tuple

# FunctAI's settings that change what the model is asked or what comes back, with their defaults
SETTINGS = (("max_steps", 8), ("retries", 1), ("tool_errors", "report"), ("on_unreadable", "raise"))
CONFIG = ("temperature", "max_tokens", "top_p", "seed", "stop", "reasoning")

# setups being worked out in this context: an escalation that comes back to itself names no setup
_WORKING: contextvars.ContextVar[Tuple[int, ...]] = contextvars.ContextVar("functai_setup_working", default=())


def _sha(text: str) -> str:
    from .calllog import _sha as sha
    return sha(text)


def _canonical(value: Any) -> str:
    from .calllog import canonical
    return canonical(value)


# ------------------------------------------------------------------ the model


def baked_id(baked: Any) -> str:
    """A baked model by its files (weights included): ``sha256:`` of its ``baked.json``'s hashes;
    weights on a service (not in the folder) by their URI too."""
    hashes = baked.meta.get("hashes", {})
    weights = baked.meta.get("weights") or {}
    if weights.get("form") == "remote":
        return _sha(_canonical({"hashes": hashes, "uri": weights.get("uri")}))
    return _sha(_canonical(hashes))


def model_of(lm: Any, settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A model as a setup names it: ``{"provider", "model"}`` as lm15 routes it,
    ``{"baked": id}``, ``{"provider": None, "model": name}`` for a name lm15
    cannot route, or None when none can be named (none set, none found)."""
    from . import accounts, models
    if models._is_baked(lm):
        return {"baked": baked_id(lm)}
    if lm is None and settings.get("client") is None:
        picked = accounts.default_model(settings.get("auth"))       # quietly: a call says which, once
        if picked is None:
            return None
        lm = picked[0]
    if lm is None:
        return None
    try:
        _router, _wire, route = models.resolve({**settings, "lm": lm})
    except Exception:  # noqa: BLE001 — a name lm15 cannot route: named, not routed (the call says why)
        return {"provider": None, "model": models.model_string(lm) if isinstance(lm, str) else str(lm)}
    return {"provider": route.provider, "model": route.model}


def _config(settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The lm15 settings set in any layer, as lm15's canonical JSON."""
    import lm15
    import lm15.serde
    from .config import CONFIG_FIELDS
    given = {k: v for k, v in settings.items() if k in CONFIG_FIELDS and v is not None and v != ()}
    if not given:
        return None
    return lm15.serde.config_to_dict(lm15.Config(**given)) or None


# ------------------------------------------------------------------ tools


_tool_code: "weakref.WeakKeyDictionary[Any, Tuple[str, Dict[str, Any]]]" = weakref.WeakKeyDictionary()


def _code_of(fn: Any) -> Tuple[str, Dict[str, Any]]:
    """(hash of the plain code a tool reaches, {key: AI function} it reaches), read once per function."""
    try:
        return _tool_code[fn]
    except (KeyError, TypeError):
        pass
    from .calllog import _original, code_hash
    from .core import FunctAIFunc
    from .graph import Analysis
    from .module import FunctAIModule
    code: Dict[str, str] = {}
    ai: Dict[str, Any] = {}
    try:
        a = Analysis(discover=True)
        a.program(fn)
        for n in a.nodes.values():
            key = f"{_original(n.module)}:{n.name}"
            if isinstance(n.obj, (FunctAIFunc, FunctAIModule)):
                ai[key] = n.obj
            else:
                code[key] = _sha(n.source) if n.source else code_hash(n.obj)
    except Exception:  # noqa: BLE001 — code the analysis cannot follow: its own text
        code = {}
    if not code:
        code[f"{_original(getattr(fn, '__module__', None))}:{getattr(fn, '__name__', 'tool')}"] = code_hash(fn)
    got = (_sha(_canonical(code)), ai)
    try:
        _tool_code[fn] = got
    except TypeError:
        pass
    return got


def tool_entry(tool: Any) -> Dict[str, Any]:
    """A tool as a setup counts it: its code, or the setup of the program it is."""
    from .core import FunctAIFunc
    from .module import FunctAIModule
    from .tools import Tool
    if isinstance(tool, (FunctAIFunc, FunctAIModule)):
        return {"setup": setup_of(tool)}
    if isinstance(tool, Tool):
        fn = tool._fn
    elif callable(tool):
        fn = tool
    else:                                            # lmcc's Tool: its code, when it carries one
        fn = getattr(tool, "fn", None)
    if fn is None:                                   # described by data, run elsewhere: its description is in the version
        return {"described": True}
    code, ai = _code_of(fn)
    entry: Dict[str, Any] = {"code": {"language": "python", "code": code}}
    if getattr(tool, "code_id", None) is not None:
        entry["code"]["id"] = tool.code_id
    if ai:
        entry["ai"] = {k: setup_of(p) for k, p in sorted(ai.items()) if p is not tool}
    return entry


def _tool_name(tool: Any) -> str:
    from .tools import Tool
    if isinstance(tool, Tool) or (callable(tool) and not hasattr(tool, "parameters")):
        return getattr(tool, "__name__", "tool")
    return getattr(tool, "name", None) or getattr(tool, "__name__", "tool")


# ------------------------------------------------------------------ reach


def needs_of(program: Any, _seen: Optional[set] = None) -> Optional[Dict[str, List[str]]]:
    """What a program's code needs to reach (contract/calls.md, *Reach*): its
    own, its tools', and (a module) every AI function's it reaches."""
    from .core import FunctAIFunc
    from .module import FunctAIModule
    from .tools import merged_needs
    seen = _seen if _seen is not None else set()
    if id(program) in seen:
        return None
    seen.add(id(program))
    found: List[Optional[Dict[str, List[str]]]] = [getattr(program, "_needs", None)]
    if isinstance(program, FunctAIFunc):
        for t in program._tools:
            found.append(needs_of(t, seen) if isinstance(t, (FunctAIFunc, FunctAIModule)) else getattr(t, "needs", None))
    elif isinstance(program, FunctAIModule):
        from .calllog import _module_graph
        for p in _module_graph(program)[1].values():
            found.append(needs_of(p, seen))
    return merged_needs(*found)


def wall_of(settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    w = settings.get("wall")
    if w is None:
        return None
    return {"name": w} if isinstance(w, str) else dict(w)


def reach_of(program: Any, settings: Mapping[str, Any]) -> Any:
    """``"open"`` with no wall; else what the wall lets through (or the program's needs), and its limits."""
    wall = wall_of(settings)
    if wall is None:
        return "open"
    needs = needs_of(program) or {}
    out: Dict[str, Any] = {k: sorted(set(wall.get(k, needs.get(k, [])))) for k in ("domains", "secrets")}
    if "limits" in wall:
        out["limits"] = copy.deepcopy(wall["limits"])
    return out


def _lets_through(granted: List[str], domain: str) -> bool:
    for g in granted:
        if g == domain or (g.startswith("*.") and domain.endswith(g[1:])):
            return True
    return False


def check_reach(program: Any, settings: Mapping[str, Any]) -> None:
    """Refuse a program that needs what its wall does not let through (``ReachError``, ``needs-unmet``)."""
    wall = wall_of(settings)
    if wall is None or ("domains" not in wall and "secrets" not in wall):
        return
    needs = needs_of(program) or {}
    missing: Dict[str, List[str]] = {"domains": [], "secrets": []}
    if "domains" in wall:
        missing["domains"] = [d for d in needs.get("domains", []) if not _lets_through(wall["domains"], d)]
    if "secrets" in wall:
        missing["secrets"] = [s for s in needs.get("secrets", []) if s not in wall["secrets"]]
    if missing["domains"] or missing["secrets"]:
        from .errors import ReachError
        what = ", ".join([*missing["domains"], *(f"the secret {s}" for s in missing["secrets"])])
        raise ReachError(f"{getattr(program, '__name__', 'program')} needs {what}, which the wall "
                         f"{wall['name']!r} does not let through", missing)


def placement_of(settings: Mapping[str, Any], *, ai: bool) -> Optional[Dict[str, Any]]:
    """Where a call runs (the record's ``placement``): the wall's name, and an AI function's server address."""
    wall = wall_of(settings)
    out: Dict[str, Any] = {}
    if wall is not None:
        out["wall"] = wall["name"]
    if ai and isinstance(settings.get("base_url"), str):
        out["base_url"] = settings["base_url"]
    return out or None


# ------------------------------------------------------------------ the document


def _plugin(p: Any) -> Dict[str, Any]:
    out = {"name": p.name, "version": p.version}
    if getattr(p, "config", None) is not None:
        out["config"] = p.config
    return out


def ai_document(fn: Any) -> Dict[str, Any]:
    """An AI function's setup document, with the settings in force now."""
    from . import plugins
    from .core import FunctAIFunc
    s = fn._effective()
    doc: Dict[str, Any] = {"version": fn.version, "model": model_of(s.get("lm"), s)}
    config = _config(s)
    if config:
        doc["config"] = config
    doc["settings"] = {k: s.get(k) if s.get(k) is not None else d for k, d in SETTINGS}
    if s.get("capabilities") is not None:
        doc["settings"]["capabilities"] = json.loads(_canonical(s["capabilities"]))
    target = s.get("escalate_to")
    if target is not None:
        to = {"setup": setup_of(target)} if isinstance(target, FunctAIFunc) else model_of(target, s)
        below = s.get("escalate_below")
        doc["escalate"] = {"to": to, "below": 0.9 if below is None else below}
    running = plugins.around(fn)
    if running:
        doc["plugins"] = [_plugin(p) for p in running]
    if fn._tools:
        doc["tools"] = {_tool_name(t): tool_entry(t) for t in fn._tools}
    doc["reach"] = reach_of(fn, s)
    return doc


def module_document(m: Any) -> Dict[str, Any]:
    """A module's setup document: its version, the setup of each AI function it reaches, its reach."""
    from .calllog import _module_graph, module_version
    from .config import effective
    _code, ai = _module_graph(m)
    return {"version": module_version(m), "ai": {k: setup_of(p) for k, p in sorted(ai.items())},
            "reach": reach_of(m, effective(getattr(m, "_settings", None) or {}))}


def document_of(program: Any) -> Dict[str, Any]:
    """A program's setup document (an AI function's or a module's), with the settings in force now."""
    from .core import FunctAIFunc
    working = _WORKING.get()
    if id(program) in working:
        return {"cycle": getattr(program, "__name__", "program")}
    token = _WORKING.set(working + (id(program),))
    try:
        return ai_document(program) if isinstance(program, FunctAIFunc) else module_document(program)
    finally:
        _WORKING.reset(token)


def setup_of(program: Any) -> str:
    """``sha256:`` of the canonical JSON of a program's setup document."""
    return _sha(_canonical(document_of(program)))


# ------------------------------------------------------------------ what differs


def _differences(a: Any, b: Any, path: str, out: List[Tuple[str, Any, Any]]) -> None:
    """Where two documents differ: down into objects both sides have, whole elsewhere (a value that changed kind,
    a list, a leaf)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            _differences(a.get(k), b.get(k), f"{path}.{k}" if path else str(k), out)
    elif _canonical(a) != _canonical(b):
        out.append((path, a, b))


def _documents(source: Any) -> Dict[str, Dict[str, Any]]:
    """Each program's setup document by key: a program here (and the AI functions it reaches), or a saved folder's nodes."""
    from .core import FunctAIFunc
    from .module import FunctAIModule
    if isinstance(source, (FunctAIFunc, FunctAIModule)):
        out = {source.__name__: document_of(source)}
        if isinstance(source, FunctAIModule):
            from .calllog import _module_graph
            for k, p in sorted(_module_graph(source)[1].items()):
                out[k] = document_of(p)
        return out
    from pathlib import Path
    path = Path(source)
    manifest = json.loads((path / "functai.json" if path.is_dir() else path).read_text())
    out = {}
    entry = manifest.get("entry")
    for key, node in manifest.get("nodes", {}).items():
        if "setup" in node:
            out[node["name"] if key == entry else key] = node["setup"]
    return out


def setup_diff(a: Any, b: Any) -> List[Dict[str, Any]]:
    """What differs between two setups, step by step: each ``{"program", "field",
    "before", "after"}``, ``field`` a dotted path into the setup document
    (``model.model``, ``config.temperature``, ``plugins``). ``a`` and ``b``:
    programs here, or saved folders (their ``functai.json``)."""
    da, db = _documents(a), _documents(b)
    out: List[Dict[str, Any]] = []
    for program in sorted(set(da) | set(db)):
        found: List[Tuple[str, Any, Any]] = []
        # a module's "ai" names its steps' setups by hash: each step is compared on its own instead
        _differences({k: v for k, v in (da.get(program) or {}).items() if k != "ai"},
                     {k: v for k, v in (db.get(program) or {}).items() if k != "ai"}, "", found)
        out += [{"program": program, "field": f, "before": x, "after": y} for f, x, y in found]
    return out


__all__ = ["setup_of", "document_of", "setup_diff", "check_reach", "reach_of", "needs_of", "placement_of"]
