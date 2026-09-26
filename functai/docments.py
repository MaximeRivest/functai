"""Docments: documentation harvested from code (inline comments, docstrings),
plus ``flexiclass`` and the ``UNSET`` sentinel.

Nothing here talks to a model. ``functai.signature`` uses these helpers to
turn comments into instructions and field descriptions.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import linecache
import re
import textwrap
import typing
from typing import Any, Dict, List, Optional, Tuple

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
    """Make a plain annotated class a dataclass, as ``@ai`` does for types.

    
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
        # Chain to user-defined __post_init__ if present (its errors are the user's to see)
        if orig_post is not None:
            orig_post(self)

    # Install our post-init only once
    setattr(cls, "__post_init__", __functai_post_init__)
    return cls

# ──────────────────────────────────────────────────────────────────────────────
# Lightweight "docments" utilities (inline-comment powered)
# ──────────────────────────────────────────────────────────────────────────────


def get_source(s: Any) -> str:
    "Get source code for string, function object, class, or dataclass."
    if isinstance(s, str):
        return s
    try:
        return class_source(s) if isinstance(s, type) else inspect.getsource(s)
    except Exception:
        return ""


def class_source(cls: type) -> str:
    """A class's source, also for classes defined in notebook cells (where
    ``inspect.getsource`` cannot find the file): the cell of one of its methods,
    else the most recent cell defining a class of that name at that line with
    those fields."""
    try:
        found = inspect.getsource(cls)
        tree = ast.parse(textwrap.dedent(found))
        if tree.body and isinstance(tree.body[0], ast.ClassDef) and tree.body[0].name == cls.__name__:
            return found
    except (OSError, TypeError, SyntaxError):
        pass            # not found, or found in the wrong file (a class made in a notebook)
    marker = f"class {cls.__name__}"
    files: List[str] = []
    for v in vars(cls).values():                  # a method written in the class points to its cell
        f = getattr(v, "__func__", v)
        code = getattr(f, "__code__", None)
        if code is not None and marker in "".join(linecache.getlines(code.co_filename)):
            files.append(code.co_filename)
    files += [k for k in reversed(list(linecache.cache))
              if ("ipykernel" in k or "ipython-input" in k) and k not in files]
    first = getattr(cls, "__firstlineno__", None)
    fields = list(getattr(cls, "__annotations__", {}) or {})
    for name in files:
        text = "".join(linecache.getlines(name))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == cls.__name__):
                continue
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            if first is not None and first not in (node.lineno, start):
                continue
            declared = [s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
            if fields and declared != fields:
                continue
            lines = text.splitlines(keepends=True)
            return "".join(lines[start - 1:node.end_lineno])
    raise OSError(f"no source for class {cls.__name__}")


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
    """Split a numpy-style docstring into its parts.

    
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

    # Strategy 1: the class's own source (also found in notebook cells)
    try:
        src = class_source(cls)
    except (OSError, TypeError):
        src = ""
    parsed = _parse(textwrap.dedent(src))
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
    """The documentation of a function's parameters and return, read from its comments.

    
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

__all__ = [
    "UNSET", "flexiclass", "docstring", "parse_docstring", "docments", "isdataclass",
    "get_dataclass_source", "get_source", "get_name", "qual_name", "sig2str", "extract_docstrings",
]
