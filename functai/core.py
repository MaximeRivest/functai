from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import json
import ast
import typing
import re
import tokenize
import io
import linecache
from typing import Any, Dict, Optional, List, Tuple, TypedDict, Callable

import dspy
from dspy import Signature, InputField, OutputField, Prediction

# ──────────────────────────────────────────────────────────────────────────────
# UNSET sentinel and flexiclass
# ──────────────────────────────────────────────────────────────────────────────

class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _UnsetType()


def _is_classvar(anno: Any) -> bool:
    try:
        return typing.get_origin(anno) is typing.ClassVar
    except Exception:
        return False


def _is_initvar(anno: Any) -> bool:
    try:
        return typing.get_origin(anno) is dataclasses.InitVar
    except Exception:
        return False


def flexiclass(cls):
    """
    Convert `cls` to a dataclass IN PLACE, giving UNSET defaults to
    any annotated field that doesn't already have a default.

    Usages:
        @flexiclass
        class Person: name: str; age: int; city: str = "Unknown"

        # or
        class Person: ...
        flexiclass(Person)

    Returns
    -------
    dataclass
        The same class object, mutated to be a dataclass.
    """
    # If already a dataclass, nothing to do (keep behavior stable)
    if dataclasses.is_dataclass(cls):
        return cls

    anns = getattr(cls, "__annotations__", {}) or {}
    # Try harvesting same-line comments for fields to embed as metadata
    field_docs: Dict[str, str] = {}
    try:
        src = get_source(cls)
        if src:
            m = re.search(r"class\s+" + re.escape(cls.__name__) + r"\b.*:\s*(?:#.*)?\n", src)
            start_idx = m.end() if m else 0
            body = src[start_idx:]
            blines = body.splitlines()
            field_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*[^#\n]+?(?:=\s*[^#\n]+)?\s*(?:#\s*(.+))?$")
            for ln in blines:
                mm = field_re.match(ln)
                if mm:
                    nm = mm.group(1)
                    cmt = (mm.group(2) or "").strip()
                    if cmt:
                        field_docs[nm] = cmt
    except Exception:
        field_docs = {}

    # Assign defaults for fields; preserve explicit defaults but wrap to keep docs in metadata.
    for name, anno in list(anns.items()):
        if _is_classvar(anno) or _is_initvar(anno):
            continue
        if name in cls.__dict__:
            val = cls.__dict__[name]
            # Optionally attach metadata when the default is already a dataclasses.field
            try:
                if isinstance(val, dataclasses.Field):
                    meta = dict(val.metadata or {})
                    if "doc" not in meta and field_docs.get(name):
                        meta["doc"] = field_docs.get(name)
                        # Recreate field carefully to avoid default/default_factory conflict
                        kwargs = {"metadata": meta}
                        if val.default is not dataclasses.MISSING and val.default_factory is dataclasses.MISSING:
                            kwargs["default"] = val.default
                        elif val.default is dataclasses.MISSING and val.default_factory is not dataclasses.MISSING:
                            kwargs["default_factory"] = val.default_factory
                        setattr(cls, name, dataclasses.field(**kwargs))
            except Exception:
                pass
            continue
        # No explicit default: use None (schema-safe), mark to flip to UNSET
        setattr(cls, name, dataclasses.field(default=None, metadata={"functai_unset": True, "doc": field_docs.get(name)}))

    # Convert in place
    cls = dataclasses.dataclass(cls)

    # Attach a __post_init__ to flip None defaults (our marked ones) to UNSET
    orig_post = getattr(cls, "__post_init__", None)

    def __functai_post_init__(self):
        # Convert marked None values to UNSET
        try:
            for f in dataclasses.fields(self):
                try:
                    if f.metadata.get("functai_unset") and getattr(self, f.name) is None:
                        setattr(self, f.name, UNSET)
                except Exception:
                    continue
        except Exception:
            pass
        # Chain to user-defined __post_init__ if present
        if orig_post is not None:
            try:
                orig_post(self)
            except Exception:
                pass

    # Install our post-init only once
    setattr(cls, "__post_init__", __functai_post_init__)
    return cls

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAIN_OUTPUT_DEFAULT_NAME = "result"
INCLUDE_FN_NAME_IN_INSTRUCTIONS_DEFAULT = True

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight "docments" utilities (inline-comment powered)
# ──────────────────────────────────────────────────────────────────────────────


def get_source(s: Any) -> str:
    "Get source code for string, function object, class, or dataclass."
    if isinstance(s, str):
        return s
    try:
        return inspect.getsource(s)
    except Exception:
        return ""


def docstring(sym: Any) -> str:
    "Get cleaned docstring for functions and classes."
    return (inspect.getdoc(sym) or "").strip()


def isdataclass(s: Any) -> bool:
    "Check if s is a dataclass *class* (not an instance)."
    return isinstance(s, type) and dataclasses.is_dataclass(s)


def get_dataclass_source(s: Any) -> str:
    "Get source code for dataclass s."
    if not isdataclass(s):
        return ""
    return get_source(s)


def get_name(obj: Any) -> str:
    return getattr(obj, "__name__", obj.__class__.__name__)


def qual_name(obj: Any) -> str:
    mod = getattr(obj, "__module__", "")
    qn = getattr(obj, "__qualname__", get_name(obj))
    return f"{mod}.{qn}" if mod else qn


_NUMPY_PARAM_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*([^#\n]+?)\s*$")
_NUMPY_RET_RE = re.compile(r"^\s*([A-Za-z_][\w\.\[\], ]*|None)\s*$")


def parse_docstring(sym: Any) -> Dict[str, str]:
    """
    Parse a subset of numpy-style docstrings:
      Parameters
      ----------
      name : type
          description...
      Returns
      -------
      type
          description...
    Returns dict with 'param:<name>' and 'return' keys when found.
    """
    ds = docstring(sym)
    if not ds:
        return {}

    lines = [l.rstrip() for l in ds.splitlines()]
    i, n = 0, len(lines)
    out: Dict[str, str] = {}

    def skip_blanks(j):
        while j < n and not lines[j].strip():
            j += 1
        return j

    while i < n:
        line = lines[i].strip()
        if line.lower() in {"parameters", "args", "arguments"}:
            # underline
            i += 1
            if i < n and set(lines[i].strip()) == {"-"}:
                i += 1
            i = skip_blanks(i)
            while i < n:
                m = _NUMPY_PARAM_RE.match(lines[i])
                if not m:
                    break
                name = m.group(1)
                i += 1
                desc_lines: List[str] = []
                while i < n and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                    desc_lines.append(lines[i].strip())
                    i += 1
                if desc_lines:
                    out[f"param:{name}"] = "\n".join(desc_lines).strip()
                i = skip_blanks(i)
            continue
        if line.lower() in {"returns", "return"}:
            i += 1
            if i < n and set(lines[i].strip()) == {"-"}:
                i += 1
            i = skip_blanks(i)
            if i < n:
                _ = _NUMPY_RET_RE.match(lines[i].strip())
                i += 1
            desc_lines: List[str] = []
            while i < n and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                desc_lines.append(lines[i].strip())
                i += 1
            if desc_lines:
                out["return"] = "\n".join(desc_lines).strip()
            continue
        i += 1
    return out


def _function_def_block(fn: Any) -> Tuple[List[str], int, int]:
    "Return (lines, base_lineno, header_end_line_index) for the function source."
    src = get_source(fn)
    if not src:
        return [], 0, -1
    lines = src.splitlines()
    base_lineno = (
        inspect.getsourcelines(fn)[1] if hasattr(inspect, "getsourcelines") else 1
    )
    header = "\n".join(lines)
    m = re.search(r"def\s+" + re.escape(fn.__name__) + r"\s*\(", header)
    if not m:
        return lines, base_lineno, -1
    start = m.end() - 1
    depth, idx = 0, start
    flat = header
    while idx < len(flat):
        ch = flat[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        elif ch == "#":
            while idx < len(flat) and flat[idx] != "\n":
                idx += 1
        idx += 1
    end_pos = idx
    header_up_to_end = flat[:end_pos]
    header_end_line_index = header_up_to_end.count("\n")
    return lines, base_lineno, header_end_line_index


def _harvest_inline_param_and_return_comments(fn: Any) -> Tuple[Dict[str, str], Optional[str]]:
    """
    Collect inline comments attached to parameters (same-line '# ...' or
    contiguous comment lines immediately above a parameter inside the header)
    and a comment after the return annotation:  `)->Type:  # comment`.
    """
    lines, base_lineno, hdr_end_idx = _function_def_block(fn)
    if not lines:
        return {}, None
    sig = inspect.signature(fn)
    pnames = list(sig.parameters.keys())

    if hdr_end_idx < 0:
        header_lines = lines[:1]
    else:
        header_lines = lines[: hdr_end_idx + 1]

    # Return comment: look on the last header line after ')->...: # ...'
    return_comment = None
    header_last = header_lines[-1] if header_lines else ""
    if "#" in header_last and (")-" in header_last or "):" in header_last or "->" in header_last):
        try:
            code, cmt = header_last.split("#", 1)
            cmt = cmt.strip()
            if "->" in code or "):" in code:
                return_comment = cmt or None
        except Exception:
            pass

    name_pattern = r"[A-Za-z_]\w*"
    param_line_re = re.compile(r"^\s*(\*{0,2})(?P<name>" + name_pattern + r")\s*(?:[:=,)]|$)")

    # Above-blocks immediately preceding a parameter line
    above_blocks: Dict[int, str] = {}
    acc: List[str] = []
    for i, ln in enumerate(header_lines):
        stripped = ln.strip()
        if stripped.startswith("#"):
            acc.append(stripped[1:].strip())
            continue
        m = param_line_re.match(ln)
        if m and acc:
            above_blocks[i] = "\n".join(acc).strip()
            acc = []
        else:
            acc = []

    param_comments: Dict[str, str] = {}
    for i, ln in enumerate(header_lines):
        # same-line
        if "#" in ln:
            code, cmt = ln.split("#", 1)
            m = param_line_re.match(code)
            if m:
                nm = m.group("name")
                if nm in pnames:
                    param_comments[nm] = (param_comments.get(nm) or cmt.strip())
        # above-block
        if i in above_blocks:
            j = i
            while j < len(header_lines):
                m = param_line_re.match(header_lines[j])
                if m:
                    nm = m.group("name")
                    if nm in pnames and nm not in param_comments:
                        param_comments[nm] = above_blocks[i]
                    break
                j += 1

    parsed = parse_docstring(fn)
    for k, v in parsed.items():
        if k.startswith("param:"):
            nm = k.split(":", 1)[1]
            param_comments.setdefault(nm, v)
        elif k == "return":
            if return_comment is None:
                return_comment = v

    return param_comments, return_comment


def _harvest_ai_output_inline_comments(fn: Any) -> Dict[str, str]:
    """
    Find comments placed after `_ai` declarations, e.g.:
        clues: str = _ai  # mention words...
    Returns { 'clues': 'mention words...' }.
    """
    src = get_source(fn)
    if not src:
        return {}
    out: Dict[str, str] = {}
    for line in src.splitlines():
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*(?::[^\=]+)?=\s*_ai(?:\[[^\]]*\])?\s*(?:#\s*(.+)\s*)?$", line)
        if m:
            name = m.group(1)
            cmt = (m.group(2) or "").strip()
            if cmt:
                out[name] = cmt
    return out


def _class_field_docments(cls: Any) -> Dict[str, str]:
    """
    Extract inline/above comments for annotated class fields (dataclass or plain class).
    Tries multiple strategies to locate and parse the class body.
    """
    def _parse(text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not text:
            return out
        pat = re.compile(
            r"(?:^|\n)class\s+" + re.escape(getattr(cls, "__name__", "")) + r"\b[^\n]*:\s*(?:#.*)?\n"
            r"(?P<body>(?:[ \t].*(?:\n|$))+)",
            flags=re.MULTILINE,
        )
        m = pat.search(text)
        if not m:
            return out
        body = m.group("body") or ""
        blines = body.splitlines()
        acc: List[str] = []
        field_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*[^#\n]+?(?:=\s*[^#\n]+)?\s*(?:#\s*(.+))?$")
        for ln in blines:
            s = ln.strip()
            if not s:
                acc = []
                continue
            if s.startswith("#"):
                acc.append(s[1:].strip())
                continue
            mm = field_re.match(ln)
            if mm:
                nm = mm.group(1)
                same = (mm.group(2) or "").strip()
                if same:
                    out[nm] = same
                elif acc:
                    out[nm] = "\n".join(acc).strip()
                acc = []
            else:
                acc = []
        return out

    # Strategy 1: direct class source
    src = get_source(cls)
    parsed = _parse(src)
    if parsed:
        return parsed

    # Strategy 2: module source
    try:
        mod = inspect.getmodule(cls)
    except Exception:
        mod = None
    if mod is not None:
        try:
            mod_src = inspect.getsource(mod)
        except Exception:
            mod_src = ""
        parsed = _parse(mod_src)
        if parsed:
            return parsed
        # Strategy 2b: file via linecache
        try:
            fname = getattr(mod, "__file__", None) or getattr(getattr(mod, "__spec__", None), "origin", None)
            if fname:
                all_text = "".join(linecache.getlines(fname) or [])
                parsed = _parse(all_text)
                if parsed:
                    return parsed
        except Exception:
            pass

    # Strategy 3: scan stack files
    try:
        for fr in inspect.stack():
            try:
                text = "".join(linecache.getlines(fr.filename) or [])
                parsed = _parse(text)
                if parsed:
                    return parsed
            except Exception:
                continue
    except Exception:
        pass

    return {}


def docments(
    elt: Any,
    full: bool = False,
    args_kwargs: bool = False,
    returns: bool = True,
    eval_str: bool = False,
) -> Dict[str, Any]:
    """
    Generate comment docs for functions or classes.

    For functions: returns {param_name: comment, 'return': comment?}
    For classes:   returns {field_name: comment}
    If full=True, each value becomes {'anno': ..., 'default': ..., 'docment': ...}.
    """
    if isinstance(elt, type):
        anns = getattr(elt, "__annotations__", {}) or {}
        fd = _class_field_docments(elt)
        if not full:
            return {k: fd.get(k) for k in anns.keys()}
        out: Dict[str, Any] = {}
        for k, anno in anns.items():
            default = getattr(elt, k, inspect._empty)
            out[k] = {"anno": anno, "default": default, "docment": fd.get(k)}
        return out

    if callable(elt):
        sig = inspect.signature(elt)
        param_docs, ret_cmt = _harvest_inline_param_and_return_comments(elt)
        if not full:
            d = {k: param_docs.get(k) for k in sig.parameters.keys()}
            if returns:
                d["return"] = ret_cmt
            if args_kwargs:
                for nm, p in sig.parameters.items():
                    if p.kind == inspect.Parameter.VAR_POSITIONAL and "args" not in d:
                        d["args"] = None if d.get(nm) is None else d.get(nm)
                    if p.kind == inspect.Parameter.VAR_KEYWORD and "kwargs" not in d:
                        d["kwargs"] = None if d.get(nm) is None else d.get(nm)
            return d
        out: Dict[str, Any] = {}
        for nm, p in sig.parameters.items():
            out[nm] = {
                "anno": (p.annotation if p.annotation is not inspect._empty else str),
                "default": (
                    p.default if p.default is not inspect._empty else inspect._empty
                ),
                "docment": param_docs.get(nm),
            }
        if returns:
            out["return"] = {
                "anno": (
                    sig.return_annotation
                    if sig.return_annotation is not inspect._empty
                    else inspect._empty
                ),
                "default": inspect._empty,
                "docment": ret_cmt,
            }
        return out

    return {}


def sig2str(func: Any) -> str:
    """
    Generate a function signature string with inline docments comments.
    """
    sig = inspect.signature(func)
    d = docments(func)
    params = []
    for nm, p in sig.parameters.items():
        base = str(p)
        cmt = d.get(nm)
        params.append(f"{base}  # {cmt}" if cmt else base)
    header = ",\n    ".join(params)
    ret_cmt = d.get("return")
    ret_ann = (
        "" if sig.return_annotation is inspect._empty else f"->{inspect.formatannotation(sig.return_annotation)}"
    )
    tail = (f"  # {ret_cmt}" if ret_cmt else "")
    return f"def {func.__name__}(\n    {header}\n){ret_ann}:{tail}"


def extract_docstrings(code: str) -> Dict[str, Tuple[str, str]]:
    """
    Return mapping {name: (docstring, paramlist)} for top-level symbols in code.
    """
    out: Dict[str, Tuple[str, str]] = {}
    try:
        tree = ast.parse(code)
    except Exception:
        return out
    module_doc = ast.get_docstring(tree) or ""
    out["_module"] = (module_doc, "")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            ds = ast.get_docstring(node) or ""
            arglist = ", ".join(a.arg for a in node.args.args)
            if node.args.vararg:
                arglist += (", *" + node.args.vararg.arg) if arglist else ("*" + node.args.vararg.arg)
            if node.args.kwarg:
                arglist += (", **" + node.args.kwarg.arg) if arglist else ("**" + node.args.kwarg.arg)
            out[node.name] = (ds, arglist)
        elif isinstance(node, ast.ClassDef):
            ds = ast.get_docstring(node) or "This class has no separate docstring."
            arglist = ""
            for n2 in node.body:
                if isinstance(n2, ast.FunctionDef) and n2.name == "__init__":
                    arglist = ", ".join(a.arg for a in n2.args.args)
            out[node.name] = (ds, arglist)
            for n2 in node.body:
                if isinstance(n2, ast.FunctionDef) and not n2.name.startswith("_"):
                    ds2 = ast.get_docstring(n2) or ""
                    arglist2 = ", ".join(a.arg for a in n2.args.args)
                    out[f"{node.name}.{n2.name}"] = (ds2, arglist2)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Defaults & configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _Defaults:
    # DSPy wiring
    lm: Any = None                 # str | dspy.LM | None
    api_key: Optional[str] = None  # funneled into dspy.LM if lm is a str
    adapter: Any = None            # "chat" | "json" | dspy.Adapter subclass | instance | None
    module: Any = "predict"        # "predict" | "cot" | "react" | dspy.Module subclass | instance | None
    temperature: Optional[float] = None

    # Statefulness (via dspy.History)
    stateful: bool = False
    state_window: int = 5

    # Optimization
    optimizer: Any = None          # Callable[[], Optimizer] | Optimizer | None

    # Prompt cosmetics
    include_fn_name_in_instructions: bool = INCLUDE_FN_NAME_IN_INSTRUCTIONS_DEFAULT

    # Debug/preview
    debug: bool = False

_GLOBAL_DEFAULTS = _Defaults()

def _effective_defaults() -> _Defaults:
    return _GLOBAL_DEFAULTS

def _apply_overrides(target: _Defaults, **overrides):
    for k, v in overrides.items():
        if hasattr(target, k):
            setattr(target, k, v)

class _ConfigContext:
    def __init__(self, snapshot: _Defaults, *, prev_dspy_lm: Any = None, prev_dspy_adapter: Any = None):
        # full snapshot of previous defaults (by value)
        self._prev = dataclasses.replace(snapshot)
        # capture previous DSPy settings to restore on exit for context usage
        self._prev_dspy_lm = prev_dspy_lm
        self._prev_dspy_adapter = prev_dspy_adapter

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # restore field-by-field
        _apply_overrides(_GLOBAL_DEFAULTS, **dataclasses.asdict(self._prev))
        # restore DSPy global config (LM/adapter) if captured
        try:
            dspy.configure(lm=self._prev_dspy_lm, adapter=self._prev_dspy_adapter)
        except Exception:
            pass
        return False

class _ConfigureFacade:
    """
    configure(...): sets global defaults (setter).
    with configure(...): temporary overrides restored on exit (context manager).
    """
    def __call__(self, **overrides):
        # capture snapshot BEFORE applying changes (for potential with-usage)
        snapshot = dataclasses.replace(_GLOBAL_DEFAULTS)
        # capture previous DSPy settings
        try:
            prev_dspy_lm = getattr(dspy.settings, "lm", None)
        except Exception:
            prev_dspy_lm = None
        try:
            prev_dspy_adapter = getattr(dspy.settings, "adapter", None)
        except Exception:
            prev_dspy_adapter = None
        # apply global changes immediately (setter semantics)
        _apply_overrides(_GLOBAL_DEFAULTS, **overrides)
        # propagate relevant settings to DSPy global settings
        try:
            # LM propagation: accept str or LM instance
            if "lm" in overrides:
                v = overrides.get("lm")
                ak = overrides.get("api_key", _GLOBAL_DEFAULTS.api_key)
                lm_inst = None
                try:
                    if v is None:
                        lm_inst = None
                    elif isinstance(v, str):
                        try:
                            lm_inst = dspy.LM(v, api_key=ak) if ak is not None else dspy.LM(v)
                        except TypeError:
                            lm_inst = dspy.LM(v)
                    else:
                        lm_inst = v
                except Exception:
                    lm_inst = v
                dspy.configure(lm=lm_inst)
            # Adapter propagation: accept string or adapter instance
            if "adapter" in overrides:
                adapter_inst = _select_adapter(overrides.get("adapter"))
                dspy.configure(adapter=adapter_inst)
        except Exception:
            # best-effort; do not block configuration if DSPy is unavailable or incompatible
            pass
        # return a context that will restore to the snapshot on exit
        return _ConfigContext(snapshot, prev_dspy_lm=prev_dspy_lm, prev_dspy_adapter=prev_dspy_adapter)

# Public configure
configure = _ConfigureFacade()

# Keep a light 'settings' alias (read-only by convention)
settings = _GLOBAL_DEFAULTS

# ──────────────────────────────────────────────────────────────────────────────
# Adapters & LMs utilities
# ──────────────────────────────────────────────────────────────────────────────

def _select_adapter(adapter: Any) -> Optional[dspy.Adapter]:
    """
    Accept multiple adapter forms:
    - String aliases ("json", "chat", "xml", "two").
    - A dspy.Adapter subclass or instance.
    - Any callable instance (duck-typed adapter), or a class that instantiates
      to a callable instance. This relaxes strict type coupling across
      different dspy import paths (e.g., dspy.adapters.Adapter) so users can
      pass custom adapters like MaxAdapter() without type errors.
    """
    if adapter is None:
        return None

    # Strings → known adapters
    if isinstance(adapter, str):
        key = adapter.lower().replace("-", "_")
        if key in ("json", "jsonadapter"):
            return dspy.JSONAdapter()
        if key in ("chat", "chatadapter"):
            return dspy.ChatAdapter()
        if key in ("xml", "xmladapter"):
            return dspy.XMLAdapter()
        if key in ("two", "twostepadapter"):
            return dspy.TwoStepAdapter()
        raise ValueError(f"Unknown adapter string '{adapter}'.")

    # Classes → try recognized dspy.Adapter subclass first; otherwise instantiate
    # and accept if the instance is callable (duck-typed adapter).
    if isinstance(adapter, type):
        try:
            if issubclass(adapter, dspy.Adapter):  # type: ignore[arg-type]
                return adapter()
        except Exception:
            # Not a dspy.Adapter subclass (or dspy not available here)
            pass
        try:
            inst = adapter()
            if callable(inst):
                return inst  # type: ignore[return-value]
        except Exception:
            pass

    # Instances → prefer exact dspy.Adapter instance; otherwise accept any
    # callable instance (duck-typed adapter).
    try:
        if isinstance(adapter, dspy.Adapter):
            return adapter
    except Exception:
        # dspy.Adapter may not be import-compatible; fall through to callable check
        pass
    if callable(adapter):
        return adapter  # type: ignore[return-value]

    raise TypeError(
        "adapter must be a string, a dspy.Adapter subclass/instance, or any callable adapter instance."
    )

@contextlib.contextmanager
def _patched_adapter(adapter_instance: Optional[dspy.Adapter]):
    prev = getattr(dspy.settings, "adapter", None)
    try:
        if adapter_instance is not None:
            dspy.settings.adapter = adapter_instance
        yield
    finally:
        dspy.settings.adapter = prev

@contextlib.contextmanager
def _patched_lm(lm_instance: Optional[Any]):
    prev = getattr(dspy.settings, "lm", None)
    try:
        if lm_instance is not None:
            dspy.settings.lm = lm_instance
        yield
    finally:
        dspy.settings.lm = prev

# ──────────────────────────────────────────────────────────────────────────────
# Signature building (docstring-driven)
# ──────────────────────────────────────────────────────────────────────────────

def _mk_signature(fn_name: str, fn: Any, *, doc: str, return_type: Any,
                  extra_outputs: Optional[List[Tuple[str, Any, str]]] = None,
                  main_output: Optional[Tuple[str, Any, str]] = None,
                  include_history_input: bool = False) -> type[Signature]:
    """Create a dspy.Signature from function params and declared outputs."""
    sig = inspect.signature(fn)
    hints = _safe_get_type_hints(fn)
    class_dict: Dict[str, Any] = {}
    ann_map: Dict[str, Any] = {}

    # Inputs (skip FunctAI-reserved names if they existed in user params by accident)
    for pname, p in sig.parameters.items():
        if pname in {"_prediction", "all"}:
            raise ValueError(f"Function parameter name '{pname}' is reserved by FunctAI.")
        ann = hints.get(pname, p.annotation if p.annotation is not inspect._empty else str)
        class_dict[pname] = InputField()
        ann_map[pname] = ann

    # Optionally add conversation history input (stateful programs)
    if include_history_input and "history" not in class_dict:
        try:
            class_dict["history"] = InputField()
            ann_map["history"] = dspy.History
        except Exception:
            # If dspy.History is unavailable for some reason, skip gracefully.
            pass

    # Extra outputs
    if extra_outputs:
        for name, typ, desc in extra_outputs:
            if name in class_dict:
                continue
            class_dict[name] = OutputField(desc=str(desc) if desc is not None else "")
            ann_map[name] = typ if typ is not None else str

    # Primary output
    if main_output is None:
        mo_name, mo_type, mo_desc = MAIN_OUTPUT_DEFAULT_NAME, return_type, ""
    else:
        mo_name, mo_type, mo_desc = main_output
        if mo_type is None:
            mo_type = return_type
    if mo_name in class_dict:
        mo_name = MAIN_OUTPUT_DEFAULT_NAME
    class_dict[mo_name] = OutputField(desc=str(mo_desc) if mo_desc is not None else "")
    ann_map[mo_name] = mo_type

    # Attach doc
    if doc:
        class_dict["__doc__"] = doc
    class_dict["__annotations__"] = ann_map

    Sig = type(f"{fn_name.title()}Sig", (Signature,), class_dict)
    return Sig


def _compose_system_doc(fn: Any, *, include_fn_name: bool) -> str:
    # history are removed. Docstring is the instruction; optional function name.
    parts = []
    if include_fn_name and getattr(fn, "__name__", None):
        parts.append(f"Function: {fn.__name__}")
    base = (fn.__doc__ or "").strip()
    if base:
        parts.append(base)
    return "\n\n".join([p for p in parts if p]).strip()

# ──────────────────────────────────────────────────────────────────────────────
# AST-based collection of declared outputs (x: T = _ai["desc"])
# ──────────────────────────────────────────────────────────────────────────────

def _eval_annotation(expr: ast.AST, env: Dict[str, Any]) -> Any:
    try:
        code = compile(ast.Expression(expr), filename="<ann>", mode="eval")
        return eval(code, env, {})
    except Exception:
        return str

def _extract_desc_from_subscript(node: ast.Subscript) -> str:
    try:
        sl = node.slice
        if isinstance(sl, ast.Index):
            sl = sl.value  # py3.8 compat
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return str(sl.value)
        if isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
            second = sl.elts[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, str):
                return str(second.value)
    except Exception:
        pass
    return ""

def _collect_ast_outputs(fn: Any) -> List[Tuple[str, Any, str]]:
    try:
        src = inspect.getsource(fn)
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except Exception:
        return []

    # Find our function node
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return []

    outputs_ordered: List[Tuple[str, Any, str]] = []
    env = dict(fn.__globals__)
    env.setdefault("typing", typing)

    for node in ast.walk(fn_node):
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                continue
            name = node.target.id
            val = node.value
            if val is None:
                continue
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "_ai"
            if not (is_ai or is_ai_sub):
                continue
            typ = _eval_annotation(node.annotation, env) if node.annotation is not None else str
            desc = _extract_desc_from_subscript(val) if is_ai_sub else ""
            if not any(n == name for n, _, _ in outputs_ordered):
                outputs_ordered.append((name, typ, desc))
        elif isinstance(node, ast.Assign):
            if not node.targets:
                continue
            val = node.value
            is_ai = isinstance(val, ast.Name) and val.id == "_ai"
            is_ai_sub = isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "_ai"
            if not (is_ai or is_ai_sub):
                continue
            desc = _extract_desc_from_subscript(val) if is_ai_sub else ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    name = tgt.id
                    if not any(n == name for n, _, _ in outputs_ordered):
                        outputs_ordered.append((name, None, desc))
    return outputs_ordered

class _ReturnInfo(TypedDict, total=False):
    mode: str  # 'name' | 'sentinel' | 'ellipsis' | 'empty' | 'other'
    name: Optional[str]

def _collect_return_info(fn: Any) -> _ReturnInfo:
    try:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
    except Exception:
        return {"mode": "other", "name": None}
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return {"mode": "other", "name": None}
    ret: _ReturnInfo = {"mode": "other", "name": None}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return):
            val = node.value
            if val is None:
                ret = {"mode": "empty", "name": None}
            elif isinstance(val, ast.Name):
                if val.id == "_ai":
                    ret = {"mode": "sentinel", "name": None}
                else:
                    ret = {"mode": "name", "name": val.id}
            elif isinstance(val, ast.Constant) and val.value is Ellipsis:
                ret = {"mode": "ellipsis", "name": None}
            else:
                ret = {"mode": "other", "name": None}
    return ret

def _extract_return_names(fn: Any) -> List[str]:
    """Best-effort: extract variable names referenced in `return (...)` or
    `return [...]` constructs. Used to map bare `_ai` placeholders to concrete
    output field names by position.

    Example: for `return (id, email)`, returns ["id", "email"].
    """
    try:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
    except Exception:
        return []
    fn_node: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            fn_node = node
            break
    if fn_node is None:
        return []
    names: List[str] = []
    last_ret: Optional[ast.Return] = None
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return):
            last_ret = node
    if last_ret is None or last_ret.value is None:
        return []
    val = last_ret.value
    elts: List[ast.AST] = []
    if isinstance(val, (ast.Tuple, ast.List)):
        elts = list(val.elts)
    elif isinstance(val, ast.Name):
        return [val.id]
    else:
        return []
    for e in elts:
        if isinstance(e, ast.Name):
            names.append(e.id)
    return names

def _safe_get_type_hints(fn: Any) -> Dict[str, Any]:
    """Best-effort type_hints that won't error on unknown/forward-ref annotations.
    Falls back to raw __annotations__ if evaluation fails.
    """
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:
        anns = getattr(fn, "__annotations__", {}) or {}
        return dict(anns)

def _return_label_from_ast(fn: Any) -> Optional[str]:
    """Extract a textual label from the return annotation (e.g., -> "french")."""
    try:
        src = inspect.getsource(fn)
        tree = ast.parse(src)
    except Exception:
        # Fallback: inspect raw annotations
        try:
            anns = getattr(fn, "__annotations__", {}) or {}
            ret = anns.get("return")
            if isinstance(ret, str):
                return ret
        except Exception:
            pass
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn.__name__:
            ann = node.returns
            if isinstance(ann, ast.Name):
                return ann.id
            if isinstance(ann, ast.Attribute):
                # attr chain like lang.French -> "French" or "lang.French"
                parts: List[str] = []
                cur = ann
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                return ".".join(reversed(parts)) if parts else None
            if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
                return str(ann.value)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Module selection
# ──────────────────────────────────────────────────────────────────────────────

def _select_module_kind(module: Any, tools: Optional[List[Any]]) -> Any:
    # If module is not explicitly set or is "predict", and tools are present → ReAct
    if module is None or (isinstance(module, str) and module.lower() in {"predict", "p", ""}):
        if tools:
            return "react"
        return "predict"
    return module

def _instantiate_module(module_kind: Any, Sig: type[Signature], *, tools: Optional[List[Any]], module_kwargs: Optional[Dict[str, Any]]) -> dspy.Module:
    mk = dict(module_kwargs or {})
    if tools:
        mk.setdefault("tools", tools)
    if isinstance(module_kind, str):
        m = module_kind.lower()
        if m in {"predict", "p"}:
            return dspy.Predict(Sig, **mk)
        if m in {"cot", "chainofthought"}:
            return dspy.ChainOfThought(Sig, **mk)
        if m in {"react", "ra"}:
            return dspy.ReAct(Sig, **mk)
        raise ValueError(f"Unknown module '{module_kind}'.")
    if isinstance(module_kind, type) and issubclass(module_kind, dspy.Module):
        return module_kind(Sig, **mk)
    if isinstance(module_kind, dspy.Module):
        try:
            module_kind.signature = Sig
            return module_kind
        except Exception:
            return type(module_kind)(Sig, **mk)
    raise TypeError("module must be a string, a dspy.Module subclass, or an instance.")

# ──────────────────────────────────────────────────────────────────────────────
# Call context and `_ai` sentinel
# ──────────────────────────────────────────────────────────────────────────────

class _CallContext:
    def __init__(self, *, program: "FunctAIFunc", Sig: type[Signature], inputs: Dict[str, Any], adapter: Optional[dspy.Adapter], main_output_name: Optional[str] = None):
        self.program = program
        self.Sig = Sig
        self.inputs = inputs
        self.adapter = adapter
        self.main_output_name = main_output_name or MAIN_OUTPUT_DEFAULT_NAME

        self._materialized = False
        self._pred: Optional[Prediction] = None
        self._value: Any = None
        self._ai_requested = False
        self.collect_only: bool = False
        self._requested_outputs: Dict[str, Tuple[Any, str]] = {}

    def request_ai(self):
        self._ai_requested = True
        return self

    # Dynamic outputs declared via _ai["..."]
    def declare_output(self, *, name: str, typ: Any = str, desc: str = "") -> None:
        if not name:
            return
        if name not in self._requested_outputs:
            self._requested_outputs[name] = (typ or str, desc or "")

    def requested_outputs(self) -> List[Tuple[str, Any, str]]:
        return [(n, t, d) for n, (t, d) in self._requested_outputs.items()]

    def ensure_materialized(self):
        if self._materialized:
            return
        if self.collect_only:
            raise RuntimeError("_ai value accessed before model run; declare outputs with _ai[\"desc\"] and return _ai.")

        # Build/refresh module
        mod = _instantiate_module(
            self.program._module_kind,
            self.Sig,
            tools=self.program._tools,
            module_kwargs=self.program._module_kwargs,
        )

        # Wire LM & generation knobs
        lm_inst = self.program._lm_instance
        if lm_inst is not None:
            try:
                mod.lm = lm_inst
            except Exception:
                pass
        if self.program.temperature is not None:
            try:
                setattr(mod, "temperature", float(self.program.temperature))
            except Exception:
                pass

        # Normalize inputs (strings or pass through)
        try:
            expected_inputs = (self.Sig.input_fields or {})
        except Exception:
            expected_inputs = {}
        in_kwargs = {k: self._to_text(v) for k, v in self.inputs.items() if k in expected_inputs}

        # Inject conversation history as an input for stateful programs
        needs_history = ("history" in expected_inputs) or bool(getattr(self.program, "_stateful", False))
        if needs_history:
            if self.program.history is None:
                try:
                    self.program.history = dspy.History(messages=[])
                except Exception:
                    self.program.history = None
            if self.program.history is not None:
                in_kwargs["history"] = self.program.history

        # Prefer setting attributes on the module rather than mutating global dspy.settings
        # to avoid cross-thread issues when evaluating with parallel executors.
        try:
            adapter_inst = self.program._adapter_instance
            if adapter_inst is not None:
                if hasattr(mod, "adapter"):
                    try:
                        mod.adapter = adapter_inst
                    except Exception:
                        pass
                else:
                    # Fallback: set adapter on each predictor if supported
                    try:
                        for _, predictor in getattr(mod, "named_predictors", lambda: [])():
                            try:
                                setattr(predictor, "adapter", adapter_inst)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        # Ensure callbacks attribute is a list (DSPy 3.0.2 may mis-set this)
        try:
            cb = getattr(mod, "callbacks", [])
            if not isinstance(cb, list):
                setattr(mod, "callbacks", [])
        except Exception:
            pass

        # Execute with a temporary adapter override so DSPy modules that read
        # from global settings (e.g., Predict/CoT/ReAct) pick up per-call
        # adapters provided via @ai(adapter=...). This mirrors project-wide
        # functai.configure(adapter=...) but scoped to this invocation.
        try:
            adapter_inst = self.program._adapter_instance
        except Exception:
            adapter_inst = None
        with _patched_adapter(adapter_inst):
            self._pred = mod(**in_kwargs)

        self._value = dict(self._pred).get(self.main_output_name)
        self._materialized = True

        # Append this turn to the persistent history if enabled
        if needs_history and self.program.history is not None:
            try:
                turn: Dict[str, Any] = {}
                try:
                    for k in expected_inputs.keys():
                        if k == "history":
                            continue
                        if k in in_kwargs:
                            turn[k] = in_kwargs[k]
                except Exception:
                    pass
                try:
                    turn.update(dict(self._pred))
                except Exception:
                    pass
                self.program.history.messages.append(turn)  # type: ignore[attr-defined]
                # Trim window if configured
                try:
                    win = int(getattr(self.program, "_state_window", 0) or 0)
                except Exception:
                    win = 0
                if win > 0:
                    msgs = self.program.history.messages  # type: ignore[attr-defined]
                    while isinstance(msgs, list) and len(msgs) > win:
                        try:
                            msgs.pop(0)
                        except Exception:
                            break
            except Exception:
                pass

    @staticmethod
    def _to_text(v: Any) -> Any:
        # Normalize structured inputs for the model
        try:
            # Pydantic BaseModel instance
            if hasattr(v, "model_dump") and callable(getattr(v, "model_dump")) and not inspect.isclass(v):
                try:
                    return v.model_dump()
                except Exception:
                    pass
            # Dataclass instance
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                try:
                    return dataclasses.asdict(v)
                except Exception:
                    pass
        except Exception:
            pass
        if isinstance(v, list):
            return [_CallContext._to_text(x) for x in v]
        if isinstance(v, (str, dict)):
            return v
        return str(v)

    @property
    def value(self):
        self.ensure_materialized()
        return self._value

    def output_value(self, name: str, typ: Any = str):
        self.ensure_materialized()
        if self._pred is None:
            return None
        data = dict(self._pred)
        return data.get(name)

# Active call ctx
from contextvars import ContextVar
_ACTIVE_CALL: ContextVar[Optional[_CallContext]] = ContextVar("functai_active_call", default=None)

class _AISentinel:
    """Module-level `_ai` sentinel."""

    def __repr__(self):
        return "<_ai>"

    def __getattr__(self, name):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai` can only be used inside an @ai-decorated function call.")
        val = ctx.request_ai().value
        return getattr(val, name)

    def _val(self):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai` can only be used inside an @ai-decorated function call.")
        return ctx.request_ai().value

    # Conversions & operators
    def __str__(self): return str(self._val())
    def __int__(self): return int(self._val())
    def __float__(self): return float(self._val())
    def __bool__(self): return bool(self._val())
    def __len__(self): return len(self._val())
    def __iter__(self): return iter(self._val())
    def __getitem__(self, k): return self._val()[k]
    def __contains__(self, k): return k in self._val()
    def __add__(self, other):   return self._val() + other
    def __radd__(self, other):  return other + self._val()
    def __sub__(self, other):   return self._val() - other
    def __rsub__(self, other):  return other - self._val()
    def __mul__(self, other):   return self._val() * other
    def __rmul__(self, other):  return other * self._val()
    def __truediv__(self, other):  return self._val() / other
    def __rtruediv__(self, other): return other / self._val()
    def __eq__(self, other):    return self._val() == other
    def __ne__(self, other):    return self._val() != other
    def __lt__(self, other):    return self._val() < other
    def __le__(self, other):    return self._val() <= other
    def __gt__(self, other):    return self._val() > other
    def __ge__(self, other):    return self._val() >= other

    def __getitem__(self, spec):
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            raise RuntimeError("`_ai[...]` can only be used inside an @ai-decorated function call.")
        ctx.request_ai()
        if isinstance(spec, str):
            # Treat bare string as description only. Try to bind the proxy's
            # name to the variable declared in the function (via AST), falling
            # back to a derived placeholder if no AST binding exists (e.g., in
            # non-assignment usages).
            desc = spec
            bound_name: Optional[str] = None
            try:
                # Match by exact description string from the function's AST-collected outputs
                for n, _t, d in _collect_ast_outputs(ctx.program._fn):
                    if d == desc:
                        bound_name = n
                        break
            except Exception:
                bound_name = None

            name = bound_name if bound_name else _derive_output_name(desc)
            # Record the output request with the chosen name and description.
            ctx.declare_output(name=name, typ=str, desc=desc)
            return _AIFieldProxy(ctx, name=name, typ=str)
        if isinstance(spec, tuple) and len(spec) >= 2:
            name = str(spec[0])
            desc = str(spec[1])
            typ = spec[2] if len(spec) >= 3 else str
            ctx.declare_output(name=name, typ=typ, desc=desc)
            return _AIFieldProxy(ctx, name=name, typ=typ)
        raise TypeError("_ai[...] expects a string description or (name, desc[, type]) tuple.")

class _AIFieldProxy:
    def __init__(self, ctx: _CallContext, *, name: str, typ: Any = str):
        self._ctx = ctx
        self._name = name
        self._typ = typ or str

    def _resolve(self):
        if self._ctx.collect_only:
            raise RuntimeError(f"Output '{self._name}' value is not available during signature collection.")
        return self._ctx.output_value(self._name, self._typ)

    def __repr__(self):
        try:
            v = self._resolve()
            return f"<_ai[{self._name!s}]={v!r}>"
        except Exception:
            return f"<_ai[{self._name!s}]>"

    def __str__(self): return str(self._resolve())
    def __int__(self): return int(self._resolve())
    def __float__(self): return float(self._resolve())
    def __bool__(self): return bool(self._resolve())
    def __len__(self): return len(self._resolve())
    def __iter__(self): return iter(self._resolve())
    def __getitem__(self, k): return self._resolve()[k]
    def __contains__(self, k): return k in self._resolve()
    def __add__(self, other):   return self._resolve() + other
    def __radd__(self, other):  return other + self._resolve()
    def __sub__(self, other):   return self._resolve() - other
    def __rsub__(self, other):  return other - self._resolve()
    def __mul__(self, other):   return self._resolve() * other
    def __rmul__(self, other):  return other * self._resolve()
    def __truediv__(self, other):  return self._resolve() / other
    def __rtruediv__(self, other): return other / self._resolve()
    def __eq__(self, other):    return self._resolve() == other
    def __ne__(self, other):    return self._resolve() != other
    def __lt__(self, other):    return self._resolve() < other
    def __le__(self, other):    return self._resolve() <= other
    def __gt__(self, other):    return self._resolve() > other
    def __ge__(self, other):    return self._resolve() >= other

def _derive_output_name(desc: str) -> str:
    s = ''.join(ch if (ch.isalnum() or ch == '_') else ' ' for ch in str(desc))
    s = s.strip().lower()
    if not s:
        return "field"
    return s.split()[0]

_ai = _AISentinel()

# ──────────────────────────────────────────────────────────────────────────────
# Type normalization for schema friendliness
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_schema_types(tp: Any, _seen: Optional[set] = None) -> None:
    """Recursively ensure nested types are schema-friendly (dataclasses or BaseModel).
    Converts plain classes-with-annotations into dataclasses via flexiclass.
    """
    if _seen is None:
        _seen = set()
    try:
        if id(tp) in _seen:
            return
        _seen.add(id(tp))
    except Exception:
        pass

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is not None:
        for a in args:
            _ensure_schema_types(a, _seen)
        return

    # Non typing-constructed type
    if not isinstance(tp, type):
        return

    # Skip obvious builtins
    if tp in (str, int, float, bool, dict, list, tuple, set, type(None)):
        return

    # Skip pydantic BaseModel subclasses if available
    try:
        import pydantic as _p
        from pydantic import BaseModel as _BM  # type: ignore
        if isinstance(tp, type) and issubclass(tp, _BM):
            return
    except Exception:
        pass

    anns = getattr(tp, "__annotations__", None)
    if anns:
        # Convert class-with-annotations to dataclass if needed
        try:
            if not dataclasses.is_dataclass(tp):
                flexiclass(tp)
        except Exception:
            pass
        # Recurse into fields
        try:
            hints = typing.get_type_hints(tp, include_extras=True)
        except Exception:
            hints = anns
        for _n, _t in (hints or {}).items():
            _ensure_schema_types(_t, _seen)

# ──────────────────────────────────────────────────────────────────────────────
# Program wrapper returned by @ai
# ──────────────────────────────────────────────────────────────────────────────

class FunctAIFunc:
    """Callable function-like object with live knobs, history, optimizer, and in-place .opt()."""

    def __init__(self, fn, *, lm=None, adapter=None, module=None, tools: Optional[List[Any]] = None,
                 temperature: Optional[float] = None, stateful: Optional[bool] = None, module_kwargs: Optional[Dict[str, Any]] = None):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._sig = inspect.signature(fn)
        hints_rt = _safe_get_type_hints(fn)
        raw_ret = hints_rt.get(
            "return",
            self._sig.return_annotation if self._sig.return_annotation is not inspect._empty else Any  # type: ignore
        )
        # Coerce unknown/forward-ref/string annotations to str
        try:
            from typing import ForwardRef  # type: ignore
            if isinstance(raw_ret, ForwardRef):
                raw_ret = str
        except Exception:
            pass
        if not isinstance(raw_ret, type):
            raw_ret = str
        self._return_type = raw_ret

        # Defaults cascade
        defs = _effective_defaults()
        self._lm = lm if lm is not None else defs.lm
        self._adapter = adapter if adapter is not None else defs.adapter
        self._module_kind = _select_module_kind(module if module is not None else defs.module, tools)
        self._tools: List[Any] = list(tools or [])
        self.temperature: Optional[float] = (float(temperature) if temperature is not None else defs.temperature)
        self._module_kwargs: Dict[str, Any] = dict(module_kwargs or {})

        # History (DSPy)
        self._stateful: bool = bool(stateful if stateful is not None else defs.stateful)
        self._state_window: int = int(defs.state_window or 0)
        self.history: Optional[dspy.History] = None
        if self._stateful:
            try:
                self.history = dspy.History(messages=[])
            except Exception:
                self.history = None

        # Optimizer
        self._optimizer = defs.optimizer  # may be instance or factory

        # Resolved objects
        self._lm_instance = self._to_lm(self._lm)
        self._adapter_instance = _select_adapter(self._adapter)

        # Compiled module (by .opt) and optimization history
        self._compiled: Optional[dspy.Module] = None
        self._opt_stack: List[dspy.Module] = []
        self._initial_module_kind = self._module_kind
        self._modules_history: List[dspy.Module] = []  # all compiled/programs that backed this function

        # Debug (preview)
        self._debug = bool(defs.debug)

        # Expose a dunder for helpers (format_prompt / inspection)
        self.__dspy__ = SimpleNamespace(fn=self._fn, program=self)

    # Signature (live)
    @property
    def signature(self):
        try:
            if isinstance(self._module_kind, dspy.Module) and hasattr(self._module_kind, "signature"):
                return self._module_kind.signature
        except Exception:
            pass
        return compute_signature(self)

    # ----- representations -----
    def __repr__(self) -> str:
        try:
            Sig = self.signature
            ins = list((getattr(Sig, "input_fields", {}) or {}).keys())
            outs = list((getattr(Sig, "output_fields", {}) or {}).keys())
            main_out = outs[-1] if outs else MAIN_OUTPUT_DEFAULT_NAME
            parts = []
            if ins:
                parts.append("inputs=" + ", ".join(ins))
            if outs:
                parts.append("outputs=" + ", ".join(outs) + f" (primary={main_out})")
            # Keep overrides terse
            if self._tools:
                parts.append(f"module={type(self._module_kind).__name__ if isinstance(self._module_kind, dspy.Module) else self._module_kind}")
                parts.append("tools=✓")
            return f"<FunctAIFunc {self._fn.__name__} | " + "; ".join(parts) + ">"
        except Exception:
            return f"<FunctAIFunc {getattr(self._fn, '__name__', 'unknown')}>"

    # ----- properties (live-mutable) -----
    @property
    def lm(self): return self._lm
    @lm.setter
    def lm(self, v): self._lm = v; self._lm_instance = self._to_lm(v); self._compiled = None

    @property
    def adapter(self): return self._adapter
    @adapter.setter
    def adapter(self, v): self._adapter = v; self._adapter_instance = _select_adapter(v); self._compiled = None

    @property
    def module(self): return self._module_kind
    @module.setter
    def module(self, v): self._module_kind = v; self._compiled = None

    @property
    def tools(self): return list(self._tools)
    @tools.setter
    def tools(self, seq):
        self._tools = list(seq or [])
        if isinstance(self._module_kind, str) and self._module_kind.lower() in {"predict", "", "p"} and self._tools:
            self._module_kind = "react"   # auto-upgrade to ReAct (DSPy does the work)
        self._compiled = None

    @property
    def optimizer(self): return self._optimizer
    @optimizer.setter
    def optimizer(self, v): self._optimizer = v

    @property
    def debug(self): return self._debug
    @debug.setter
    def debug(self, v: bool): self._debug = bool(v)

    # ----- callable -----
    def __call__(self, *args, all: bool = False, **kwargs):
        # Back-compat: map deprecated _prediction to all
        if "_prediction" in kwargs:
            if kwargs.pop("_prediction"):
                all = True

        # Bind args to the *user* function (discard our control kwargs)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in {"_prediction", "all"}}
        bound = self._sig.bind_partial(*args, **clean_kwargs)
        bound.apply_defaults()
        inputs = {k: v for k, v in bound.arguments.items() if k in self._sig.parameters}

        # Build signature once
        Sig = self.signature
        outs = list((getattr(Sig, "output_fields", {}) or {}).keys())
        main_name = outs[-1] if outs else MAIN_OUTPUT_DEFAULT_NAME

        # Call context
        ctx = _CallContext(program=self, Sig=Sig, inputs=inputs, adapter=self._adapter_instance, main_output_name=main_name)
        token = _ACTIVE_CALL.set(ctx)
        try:
            result = self._fn(*args, **clean_kwargs)

            if self._debug:
                try:
                    print(f"[FunctAI] module={type(self._module_kind).__name__ if isinstance(self._module_kind, dspy.Module) else self._module_kind}; adapter={type(self._adapter_instance).__name__ if self._adapter_instance else type(getattr(dspy.settings,'adapter',None)).__name__}")
                    print(f"[FunctAI] inputs={list(inputs.keys())}; outputs={outs} (primary={main_name})")
                except Exception:
                    pass

            if all:
                _ = ctx.request_ai().value
                return ctx._pred

            if result is _ai or result is Ellipsis:
                return ctx.request_ai().value

            # Unwrap proxies and realize bare `_ai` placeholders inside containers.
            def _has_bare_ai(x):
                if x is _ai:
                    return True
                if isinstance(x, (list, tuple)):
                    return any(_has_bare_ai(i) for i in x)
                if isinstance(x, dict):
                    return any(_has_bare_ai(v) for v in x.values())
                return False

            # Pre-compute return field order for mapping bare `_ai` occurrences.
            ret_names = _extract_return_names(self._fn)
            if not ret_names:
                try:
                    ret_names = [n for n, _t, _d in _collect_ast_outputs(self._fn)]
                except Exception:
                    ret_names = []
            names_pool = list(ret_names)

            def _next_name():
                return names_pool.pop(0) if names_pool else None

            def _unwrap_and_realize(x):
                # Field-specific proxies
                if isinstance(x, _AIFieldProxy):
                    return x._resolve()
                # Bare `_ai` placeholder: map by return position/name
                if x is _ai:
                    # Ensure prediction is ready
                    ctx.request_ai().value
                    name = _next_name()
                    if not name:
                        try:
                            outs2 = list((getattr(Sig, "output_fields", {}) or {}).keys())
                            name = outs2[-1] if outs2 else MAIN_OUTPUT_DEFAULT_NAME
                        except Exception:
                            name = MAIN_OUTPUT_DEFAULT_NAME
                    return ctx.output_value(name)
                if isinstance(x, list):
                    return [_unwrap_and_realize(i) for i in x]
                if isinstance(x, tuple):
                    return tuple(_unwrap_and_realize(i) for i in x)
                if isinstance(x, dict):
                    return {k: _unwrap_and_realize(v) for k, v in x.items()}
                return x

            if _has_bare_ai(result):
                # Prepare prediction once
                ctx.request_ai().value
            result = _unwrap_and_realize(result)
            if result is None and not ctx._ai_requested:
                return ctx.request_ai().value
            return result
        finally:
            _ACTIVE_CALL.reset(token)

    # ----- optimization -----
    def opt(self, *, trainset: Optional[List[Any]] = None, optimizer: Any = None, metric: Optional[Callable] = None, **opts) -> None:
        """
        Compile with a DSPy optimizer and mutate in place.
        - optimizer: instance or factory; if None, uses self.optimizer or global default.
        - trainset: list of (inputs, outputs) pairs or DSPy's preferred format.
        - metric: a Callable metric passed to optimizers that accept/require it
          (e.g., SIMBA, MIPROv2). Required when the chosen optimizer's constructor
          needs a metric.
        Additional **opts are forwarded to the optimizer constructor if it is a factory.
        """
        Sig = self.signature

        # Build a base module to optimize
        if self._compiled is not None:
            try:
                self._compiled.signature = Sig
                base_mod = self._compiled
            except Exception:
                base_mod = type(self._compiled)(Sig, **(self._module_kwargs or {}))
                if self._tools:
                    try:
                        base_mod.tools = self._tools
                    except Exception:
                        pass
        else:
            base_mod = _instantiate_module(self._module_kind, Sig, tools=self._tools, module_kwargs=self._module_kwargs)

        # Pick optimizer
        opt = optimizer if optimizer is not None else (self._optimizer if self._optimizer is not None else dspy.BootstrapFewShot)

        # If the optimizer accepts a 'metric' argument, require the caller to provide it
        # when the parameter is required (no default). Otherwise, pass it through when given.
        if isinstance(opt, type):
            ctor_kwargs = dict(opts)
            try:
                sig = inspect.signature(opt.__init__)
                # Require/pass metric when appropriate
                if "metric" in sig.parameters:
                    param = sig.parameters["metric"]
                    required = param.default is inspect._empty
                    if required and metric is None:
                        raise ValueError("This optimizer requires a 'metric'. Please call .opt(metric=...) with a callable metric.")
                    if metric is not None:
                        ctor_kwargs["metric"] = metric
                # Provide prompt_model / task_model from our program LM if requested
                for name in ("prompt_model", "task_model"):
                    if name in sig.parameters and name not in ctor_kwargs and self._lm_instance is not None:
                        ctor_kwargs[name] = self._lm_instance
            except (ValueError, TypeError):
                # if signature inspection fails, just pass through opts
                pass
            optimizer_instance = opt(**ctor_kwargs)  # factory/class
        else:
            optimizer_instance = opt           # already an instance; opts ignored
            # If instance exposes a 'metric' attribute and user provided one, set it.
            if metric is not None:
                try:
                    setattr(optimizer_instance, "metric", metric)
                except Exception:
                    pass

        new_prog = optimizer_instance.compile(base_mod, trainset=trainset)

        # Track history and swap in compiled program
        if self._compiled is not None:
            self._opt_stack.append(self._compiled)
        self._compiled = new_prog
        try:
            self._modules_history.append(new_prog)
        except Exception:
            pass
        self._module_kind = self._compiled  # instance; signature will rebind per call

    def undo_opt(self, steps: int = 1) -> None:
        steps = max(1, int(steps))
        for _ in range(steps):
            if self._opt_stack:
                self._compiled = self._opt_stack.pop()
                self._module_kind = self._compiled
            else:
                self._compiled = None
                self._module_kind = self._initial_module_kind
                break

    # ----- helpers -----
    @staticmethod
    def _to_lm(v: Any):
        if v is None:
            return None
        if isinstance(v, str):
            ak = _effective_defaults().api_key
            try:
                return dspy.LM(v, api_key=ak) if ak is not None else dspy.LM(v)
            except TypeError:
                # older DSPy may not accept api_key kwarg
                return dspy.LM(v)
        return v

    # ----- export / history helpers -----
    def programs(self) -> List[dspy.Module]:
        """Return a list of all DSPy modules that have backed this function via .opt()."""
        items = list(self._modules_history)
        if self._compiled is not None and (not items or items[-1] is not self._compiled):
            items.append(self._compiled)
        return items

    def latest_program(self, fresh: bool = False) -> dspy.Module:
        """Return the latest DSPy module.
        - If compiled, return the compiled program (unless fresh=True).
        - Otherwise, instantiate a fresh module with the current signature and config.
        """
        if self._compiled is not None and not fresh:
            return self._compiled
        Sig = self.signature
        mod = _instantiate_module(self._module_kind, Sig, tools=self._tools, module_kwargs=self._module_kwargs)
        try:
            if self._lm_instance is not None:
                mod.lm = self._lm_instance
        except Exception:
            pass
        try:
            if self._adapter_instance is not None and hasattr(mod, "adapter"):
                mod.adapter = self._adapter_instance
        except Exception:
            pass
        try:
            if self.temperature is not None:
                setattr(mod, "temperature", float(self.temperature))
        except Exception:
            pass
        # Ensure callbacks attribute is a list to be compatible with DSPy callback wrapper.
        try:
            cb = getattr(mod, "callbacks", [])
            if not isinstance(cb, list):
                setattr(mod, "callbacks", [])
        except Exception:
            pass
        return mod

    def to_dspy(self, deepcopy: bool = False) -> dspy.Module:
        """Convenience: export the latest module for direct DSPy use.
        If deepcopy=True, returns a deep-copied module.
        """
        mod = self.latest_program(fresh=False)
        try:
            return mod.deepcopy() if deepcopy and hasattr(mod, "deepcopy") else mod
        except Exception:
            return mod

class SimpleNamespace:
    def __init__(self, **kw): self.__dict__.update(kw)

# ──────────────────────────────────────────────────────────────────────────────
# Decorator: @ai  (works with @ai and @ai(...))
# ──────────────────────────────────────────────────────────────────────────────

def ai(_fn=None, **cfg):
    """Decorator that turns a typed Python function into a single-call DSPy program.

    Usage:
        @ai
        def f(...)->T:
            # declare outputs
            y: T = _ai["..."]
            return _ai  # last declared is the default primary

        @ai(lm="openai:gpt-4o-mini", tools=[...], temperature=0.2, stateful=True)
        def g(...)->T:
            x: T = _ai["..."]
            return _ai
    """
    def _decorate(fn):
        return FunctAIFunc(fn, **cfg)
    if _fn is not None and callable(_fn):
        return _decorate(_fn)
    return _decorate

# ──────────────────────────────────────────────────────────────────────────────
# Prompt preview (kept minimal; adapters ultimately render)
# ──────────────────────────────────────────────────────────────────────────────

def _build_instruction_appendix(
    fn: Any,
    *,
    main_name: str,
    order_names: List[str],
    ast_map: Dict[str, Tuple[Any, str]],
    main_output_type: Any,
) -> str:
    """
    Build extra instruction text from docments:
      - parameter inline comments
      - return comment (if present)
      - comments after `_ai` lines
      - field docs for return type classes/dataclasses
    """
    parts: List[str] = []

    # Param docs
    param_docs, ret_cmt = _harvest_inline_param_and_return_comments(fn)
    if param_docs:
        parts.append("Parameter guidance:")
        for k, v in param_docs.items():
            if v:
                parts.append(f"- {k}: {v}")
        parts.append("")

    # Output docs from AST map (which may be enriched with _ai line comments)
    if order_names:
        parts.append("Output guidance:")
        for nm in order_names:
            typ, dsc = ast_map.get(nm, (str, ""))
            parts.append(f"- {nm}: {dsc}" if dsc else f"- {nm}")
        parts.append("")

    # Return guidance
    if ret_cmt:
        parts.append(f"Return guidance: {ret_cmt}")
        parts.append("")

    # If the main output is a class/dataclass, include its field docs (qualified)
    try:
        if isinstance(main_output_type, type) and getattr(main_output_type, "__annotations__", None):
            # Collect docs from source and from dataclass metadata, then merge per field
            src_docs = _class_field_docments(main_output_type) or {}
            meta_docs: Dict[str, Optional[str]] = {}
            if dataclasses.is_dataclass(main_output_type):
                try:
                    meta_docs = {f.name: (f.metadata.get("doc") if f.metadata else None) for f in dataclasses.fields(main_output_type)}
                except Exception:
                    meta_docs = {}
            # Determine field order using annotations (preserve declaration order)
            all_fields = list(getattr(main_output_type, "__annotations__", {}).keys())
            if not all_fields and dataclasses.is_dataclass(main_output_type):
                try:
                    all_fields = [f.name for f in dataclasses.fields(main_output_type)]
                except Exception:
                    all_fields = []
            if all_fields:
                cls_name = getattr(main_output_type, "__name__", "Object")
                for k in all_fields:
                    v = src_docs.get(k) or meta_docs.get(k)
                    parts.append(f"- {cls_name}.{k}: {v}" if v else f"- {cls_name}.{k}")
                parts.append("")
    except Exception:
        pass

    text = "\n".join([p for p in parts if p is not None]).strip()
    return text

def _default_user_content(sig: Signature, inputs: Dict[str, Any]) -> str:
    lines = []
    doc = (getattr(sig, "__doc__", "") or "").strip()
    if doc:
        lines.append(doc)
        lines.append("")
    if inputs:
        lines.append("Inputs:")
        for k, v in inputs.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    outs = getattr(sig, "output_fields", {}) or {}
    if outs:
        lines.append("Please produce the following outputs:")
        for k in outs.keys():
            lines.append(f"- {k}")
    return "\n".join(lines).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Signature helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_signature(fn_or_prog):
    """Build and return the computed dspy.Signature for a function/program."""
    prog: FunctAIFunc
    if isinstance(fn_or_prog, FunctAIFunc):
        prog = fn_or_prog
    elif hasattr(fn_or_prog, "__wrapped__") and isinstance(getattr(fn_or_prog, "__wrapped__"), FunctAIFunc):
        prog = getattr(fn_or_prog, "__wrapped__")
    elif hasattr(fn_or_prog, "__dspy__") and hasattr(fn_or_prog.__dspy__, "program"):
        prog = fn_or_prog.__dspy__.program  # type: ignore
    else:
        raise TypeError("compute_signature(...) expects an @ai-decorated function/program.")

    sysdoc_base = _compose_system_doc(prog._fn, include_fn_name=bool(_effective_defaults().include_fn_name_in_instructions))

    ast_outputs = _collect_ast_outputs(prog._fn)
    ret_label = _return_label_from_ast(prog._fn)
    ret_info = _collect_return_info(prog._fn)
    order_names = [n for n, _, _ in ast_outputs]

    if ret_info.get("mode") == "name" and ret_info.get("name") in order_names:
        main_name = typing.cast(str, ret_info.get("name"))
    elif order_names:
        main_name = order_names[-1]
    else:
        # No explicit outputs declared; use textual return label if provided.
        if ret_label and str(ret_label).isidentifier():
            main_name = str(ret_label)
        else:
            main_name = MAIN_OUTPUT_DEFAULT_NAME

    # Merge _ai[...] desc with comments after _ai lines
    ai_inline = _harvest_ai_output_inline_comments(prog._fn)
    ast_map: Dict[str, Tuple[Any, str]] = {n: (t if t is not None else str, (d or "")) for n, t, d in ast_outputs}
    for nm, (t, d) in list(ast_map.items()):
        extra = ai_inline.get(nm)
        if extra and d:
            ast_map[nm] = (t, f"{d} — {extra}")
        elif extra:
            ast_map[nm] = (t, extra)
    if main_name in ast_map:
        t0, d0 = ast_map[main_name]
        if ret_info.get("mode") in {"sentinel", "ellipsis"} or (ret_info.get("mode") == "name" and ret_info.get("name") == main_name):
            main_typ = prog._return_type if isinstance(prog._return_type, type) else str
        else:
            main_typ = t0 if t0 is not None else str
        main_desc = d0
    else:
        main_typ = prog._return_type if isinstance(prog._return_type, type) else str
        main_desc = ""

    # If main output is a plain class with annotations, convert it in-place to a dataclass
    try:
        if isinstance(main_typ, type) and getattr(main_typ, "__annotations__", None) and not dataclasses.is_dataclass(main_typ):
            flexiclass(main_typ)
    except Exception:
        pass

    # Recursively ensure nested types inside main output are dataclass/base-model friendly
    _ensure_schema_types(main_typ)

    # Ensure any extra output typed classes are dataclasses as well
    try:
        for n in order_names:
            if n == main_name:
                continue
            t_extra = ast_map[n][0]
            if isinstance(t_extra, type) and getattr(t_extra, "__annotations__", None) and not dataclasses.is_dataclass(t_extra):
                flexiclass(t_extra)
            _ensure_schema_types(t_extra)
    except Exception:
        pass

    extras = [(n, (ast_map[n][0] if ast_map[n][0] is not None else str), ast_map[n][1]) for n in order_names if n != main_name]

    # Append docments-derived guidance to the system doc
    appendix = _build_instruction_appendix(
        prog._fn,
        main_name=main_name,
        order_names=order_names,
        ast_map=ast_map,
        main_output_type=main_typ,
    )
    sysdoc = sysdoc_base if not appendix else (sysdoc_base + ("\n\n" if sysdoc_base else "") + appendix)
    Sig = _mk_signature(
        prog._fn.__name__,
        prog._fn,
        doc=sysdoc,
        return_type=prog._return_type,
        extra_outputs=extras,
        main_output=(main_name, main_typ, main_desc),
        include_history_input=bool(getattr(prog, "_stateful", False)),
    )
    return Sig

def signature_text(fn_or_prog) -> str:
    """Return a tiny, human-readable summary of the computed Signature."""
    Sig = compute_signature(fn_or_prog)
    anns: Dict[str, Any] = getattr(Sig, "__annotations__", {}) or {}
    inputs = list((getattr(Sig, "input_fields", {}) or {}).keys())
    outputs = list((getattr(Sig, "output_fields", {}) or {}).keys())
    doc = (getattr(Sig, "__doc__", "") or "").strip()
    main_name = outputs[-1] if outputs else MAIN_OUTPUT_DEFAULT_NAME

    def _tostr(t: Any) -> str:
        try:
            return str(t)
        except Exception:
            return repr(t)

    lines: List[str] = []
    lines.append(f"Signature: {Sig.__name__}")
    if doc:
        lines.append(" | Doc: " + (doc[:120] + ("…" if len(doc) > 120 else "")))
    if inputs:
        lines.append(" | Inputs: " + ", ".join(f"{k}:{_tostr(anns.get(k, str))}" for k in inputs))
    if outputs:
        lines.append(" | Outputs: " + ", ".join(f"{k}{'*' if k == main_name else ''}" for k in outputs))
    return "".join(lines).strip()

def inspect_history_text() -> str:
    """Return dspy.inspect_history() as text (best effort)."""
    import io
    import contextlib as _ctx
    buf = io.StringIO()
    try:
        with _ctx.redirect_stdout(buf):
            try:
                dspy.inspect_history()
            except Exception:
                pass
    except Exception:
        return ""
    return buf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ai",
    "_ai",
    "configure",
    "inspect_history_text",
    "settings",
    "compute_signature",
    "signature_text",
    # new exports
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
