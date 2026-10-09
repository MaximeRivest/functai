"""What a function's code tells the model beyond its docstring, read with
Python's parser and tokenizer so each comment is tied to the code it sits
next to:

- a comment after each choice of a ``Literal[...]`` written over several
  lines, and after each member of an ``Enum``: that choice's meaning;
- a comment after the closing ``]`` of an annotation written over several
  lines: the field's (or parameter's, or output's) meaning;
- ``Annotated[T, "words"]`` on a record's field;
- every record a function reads or writes, nested ones included
  (``list[Analysis]``, ``Optional[Obs]``), with its docstring and the
  meaning of its fields.

``docments.py``'s line patterns find the one-line cases (a comment after a
field or a parameter on the same line, comment lines above it); this module
finds what they cannot, and where both find words, theirs are kept, so a
prompt that was right stays the same. A comment cannot be read when the code
has no source (``python - <<EOF``, ``exec``; ``python -c`` before 3.13): then
nothing here, and nothing of ``docments``, reaches the model (``python -c``
keeps its source from Python 3.13 on), and
``warn_without_source`` says so once per function.
"""
from __future__ import annotations

import ast
import dataclasses
import enum
import io
import textwrap
import tokenize
import typing
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .docments import class_source, get_source, strip_pragmas


@dataclasses.dataclass
class Note:
    """The words the code gives one name: its own, and each choice's."""
    words: str = ""
    choices: Dict[Any, str] = dataclasses.field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.words or self.choices)


# ──────────────────────────────────────────────────────────────────────────────
# Source, tree and comments
# ──────────────────────────────────────────────────────────────────────────────

def _comments(src: str) -> Dict[int, Tuple[int, str]]:
    """Line number → (column, text) of the comment on that line. A line that
    holds only a comment has its column negated (``-1 - column``), so a
    comment after code and a comment alone are told apart."""
    out: Dict[int, Tuple[int, str]] = {}
    code: set = set()
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = (tok.start[1], strip_pragmas(tok.string[1:]))   # a pragma is no words
            elif tok.type not in skip:
                code.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return {line: (col if line in code else -1 - col, text) for line, (col, text) in out.items()}


def _parsed(src: str) -> Tuple[Optional[ast.Module], Dict[int, Tuple[int, str]]]:
    src = textwrap.dedent(src)
    try:
        return ast.parse(src), _comments(src)
    except SyntaxError:
        return None, {}


def _comment_after(comments: Dict[int, Tuple[int, str]], line: int, col: int) -> str:
    """The comment after the code at (line, col), with the comment-only lines
    right below it that start in the same column: one comment the writer
    wrapped over several lines."""
    c = comments.get(line)
    if not c or c[0] < col:
        return ""
    words, k = [c[1]], line + 1
    while k in comments and comments[k][0] == -1 - c[0]:
        words.append(comments[k][1])
        k += 1
    return " ".join(w for w in words if w)


def _choice_words(ann: Optional[ast.expr], comments: Dict[int, Tuple[int, str]],
                  used: Optional[set] = None) -> Dict[Any, str]:
    """For every ``Literal[...]`` inside an annotation: each choice → the comment
    after it on its line, when it is the only choice on that line. On the line
    where the ``Literal`` closes (``"lab"]  # …``), the comment is the last
    choice's when every other choice line has one (the pattern the writer
    set), and the field's otherwise. ``used`` collects the lines whose comment
    went to a choice."""
    out: Dict[Any, str] = {}
    if ann is None:
        return out
    for node in ast.walk(ann):
        if not (isinstance(node, ast.Subscript) and _name_of(node.value) == "Literal"):
            continue
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        per_line: Dict[int, List[ast.Constant]] = {}
        for c in items:
            if isinstance(c, ast.Constant):
                per_line.setdefault(c.lineno, []).append(c)
        single = {line: here[0] for line, here in per_line.items() if len(here) == 1}
        said = {line: _comment_after(comments, line, c.end_col_offset) for line, c in single.items()}
        for line, c in single.items():
            if not said[line]:
                continue
            if line == node.end_lineno:
                others = [ln for ln in single if ln != line]
                if not others or not all(said[ln] for ln in others):
                    continue                      # the field's comment, after its closing bracket
            out[c.value] = said[line]
            if used is not None:
                used.add(line)
    return out


def _name_of(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _trailing(stmt: ast.AST, comments: Dict[int, Tuple[int, str]]) -> str:
    """The comment after a statement written over several lines (on its last line)."""
    if stmt.end_lineno == stmt.lineno:
        return ""
    return _comment_after(comments, stmt.end_lineno, stmt.end_col_offset)


def _annotated_words(ann: Optional[ast.expr]) -> str:
    """``Annotated[T, "words"]``: the first string after the type."""
    if isinstance(ann, ast.Subscript) and _name_of(ann.value) == "Annotated" and isinstance(ann.slice, ast.Tuple):
        for e in ann.slice.elts[1:]:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                return e.value.strip()
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Classes: records and enums
# ──────────────────────────────────────────────────────────────────────────────

def _class_node(cls: type) -> Tuple[Optional[ast.ClassDef], Dict[int, Tuple[int, str]]]:
    try:
        src = class_source(cls)
    except (OSError, TypeError):
        return None, {}
    tree, comments = _parsed(src)
    if tree is None:
        return None, {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return node, comments
    return None, {}


def record_notes(cls: type) -> Tuple[str, Dict[str, Note]]:
    """A record's docstring (as written: never a dataclass's generated one) and
    what its source says about each field."""
    node, comments = _class_node(cls)
    if node is None:
        return "", {}
    doc = (ast.get_docstring(node) or "").strip()
    out: Dict[str, Note] = {}
    for stmt in node.body:
        if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
            continue
        used: set = set()
        choices = _choice_words(stmt.annotation, comments, used)
        if stmt.end_lineno == stmt.lineno:                          # one line: the comment after it
            words = _comment_after(comments, stmt.lineno, stmt.end_col_offset)
        else:                                                       # several: the comment after the last
            words = "" if stmt.end_lineno in used else _trailing(stmt, comments)
        words = words or _above(comments, stmt.lineno) or _annotated_words(stmt.annotation)
        out[stmt.target.id] = Note(words, choices)
    return doc, out


def enum_notes(cls: type) -> Tuple[str, Dict[Any, str]]:
    """An Enum's docstring, and each member's value → the comment after it (or
    the comment lines right above it)."""
    node, comments = _class_node(cls)
    if node is None:
        return "", {}
    by_name = {m.name: m.value for m in cls}           # type: ignore[attr-defined]
    out: Dict[Any, str] = {}
    for stmt in node.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        name = stmt.targets[0].id
        if name not in by_name:
            continue
        words = _comment_after(comments, stmt.end_lineno, stmt.end_col_offset) or _above(comments, stmt.lineno)
        if words:
            out[by_name[name]] = words
    return (ast.get_docstring(node) or "").strip(), out


def _above(comments: Dict[int, Tuple[int, str]], line: int) -> str:
    """The comment-only lines right above a line, unless they continue a
    comment after the code above them (wrapped, starting in its column)."""
    acc: List[str] = []
    k = line - 1
    while k in comments and comments[k][0] < 0:
        acc.insert(0, comments[k][1])
        k -= 1
    top = comments.get(k + 1)
    before = comments.get(k)
    if acc and before is not None and before[0] >= 0 and top is not None and top[0] == -1 - before[0]:
        return ""                                                   # a wrapped comment of the line above
    return "\n".join(acc).strip()


def is_record(tp: Any) -> bool:
    """A class whose fields the model fills or reads: a dataclass, a TypedDict, a
    NamedTuple, a pydantic model, a plain annotated class. Not an Enum, not a builtin."""
    if not isinstance(tp, type) or issubclass(tp, enum.Enum) or tp.__module__ == "builtins":
        return False
    return bool(getattr(tp, "__annotations__", None)) and tp.__module__ not in ("typing", "collections")


def is_pydantic(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "model_fields") and hasattr(tp, "model_json_schema")


def records_in(tp: Any, seen: Optional[set] = None) -> Iterator[type]:
    """Every record class a type holds, depth first, each once: the type itself,
    then its fields' (``list[Analysis]``, ``Optional[Obs]``, ``dict[str, Row]``)."""
    seen = set() if seen is None else seen
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Annotated:
        yield from records_in(args[0], seen)
        return
    if origin is not None:
        if origin is typing.Literal:
            return
        for a in args:
            if a is not Ellipsis:
                yield from records_in(a, seen)
        return
    if not is_record(tp) or tp in seen:
        return
    seen.add(tp)
    yield tp
    try:
        hints = typing.get_type_hints(tp, include_extras=True)
    except Exception:  # noqa: BLE001 — unresolvable names: walk no further than what is known
        hints = dict(getattr(tp, "__annotations__", {}) or {})
    for h in hints.values():
        yield from records_in(h, seen)


def enum_in(tp: Any) -> Optional[type]:
    """The Enum class a field's type is (through Optional and Annotated), if any."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Annotated:
        return enum_in(args[0])
    if origin is not None and origin is not typing.Literal:
        found = [e for e in (enum_in(a) for a in args if a is not type(None)) if e is not None]
        return found[0] if len(found) == 1 else None
    return tp if isinstance(tp, type) and issubclass(tp, enum.Enum) else None


# ──────────────────────────────────────────────────────────────────────────────
# Functions: parameters, `_ai` outputs, the return annotation
# ──────────────────────────────────────────────────────────────────────────────

def function_notes(fn: Any) -> Tuple[Dict[str, Note], Dict[str, Note], Note]:
    """What the source says about each parameter, each ``_ai`` output, and the
    return annotation, beyond the one-line comments ``docments`` reads."""
    src = get_source(fn)
    if not src:
        return {}, {}, Note()
    tree, comments = _parsed(src)
    if tree is None:
        return {}, {}, Note()
    node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == getattr(fn, "__name__", None)), None)
    if node is None:
        return {}, {}, Note()
    params: Dict[str, Note] = {}
    a = node.args
    args = [*a.posonlyargs, *a.args, *([a.vararg] if a.vararg else []), *a.kwonlyargs, *([a.kwarg] if a.kwarg else [])]
    for i, arg in enumerate(args):
        # the comment after a parameter, on the line where it ends, when no other parameter follows on that line
        # (``def f(x: str,  # words`` too, which docments' line patterns cannot see)
        follows = any(b.lineno == arg.end_lineno for b in args[i + 1:])
        used: set = set()
        choices = _choice_words(arg.annotation, comments, used)
        words = "" if follows or arg.end_lineno in used else _comment_after(comments, arg.end_lineno, arg.end_col_offset)
        note = Note(words, choices)
        if note:
            params[arg.arg] = note
    outputs: Dict[str, Note] = {}
    for stmt in ast.walk(node):
        if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None
                and _is_ai(stmt.value)):
            used: set = set()
            choices = _choice_words(stmt.annotation, comments, used)
            note = Note("" if stmt.end_lineno in used else _trailing(stmt, comments), choices)
            if note:
                outputs[stmt.target.id] = note
    returns = Note("", _choice_words(node.returns, comments))
    return params, outputs, returns


def _is_ai(value: ast.expr) -> bool:
    if isinstance(value, ast.Subscript):
        value = value.value
    return isinstance(value, ast.Name) and value.id == "_ai"


# ──────────────────────────────────────────────────────────────────────────────
# Words for the instruction
# ──────────────────────────────────────────────────────────────────────────────

def with_choices(words: str, choices: Dict[Any, str]) -> str:
    """A name's words, then one line per described choice (``  - lab: indoors``);
    with described choices and no words of its own, the words are ``one of:``."""
    if not choices:
        return words
    return "\n".join([words or "one of:", *[f"  - {_choice(v)}: {w}" for v, w in choices.items()]])


def _choice(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def record_sections(types: List[Any], field_docs: Any) -> List[str]:
    """For every record the types hold (each once, in order): its described
    fields, as instruction lines, after its docstring when it has described
    fields (a docstring alone adds nothing, so a prompt that was complete
    stays the same). The parser's words are used when it could read the class;
    ``field_docs(cls)`` (docments' line patterns) only when it could not, and
    ``field_docs(cls, metadata_only=True)`` (a dataclass field's
    ``metadata["doc"]``) where neither found words. A pydantic model's
    docstring and ``Field(description=...)`` are in its JSON schema already,
    so they are not repeated."""
    out: List[str] = []
    seen: set = set()
    for tp in types:
        for cls in records_in(tp, seen):
            doc, notes = record_notes(cls)
            old = {} if notes else (field_docs(cls) or {})          # the parser, when it could read the class
            meta = field_docs(cls, metadata_only=True) or {}
            pyd = is_pydantic(cls)
            described = {name for name, f in getattr(cls, "model_fields", {}).items() if f.description} if pyd else set()
            try:
                hints = typing.get_type_hints(cls, include_extras=True)
            except Exception:  # noqa: BLE001
                hints = dict(getattr(cls, "__annotations__", {}) or {})
            lines = []
            for name in getattr(cls, "__annotations__", {}):
                if name in described:
                    continue
                note = notes.get(name, Note())
                words = note.words or old.get(name) or meta.get(name, "")
                choices = dict(note.choices)
                kind = enum_in(hints.get(name))
                if kind is not None:
                    edoc, members = enum_notes(kind)
                    words = words or edoc
                    choices = {**members, **choices}
                if words or choices:
                    lines.append(f"- {cls.__name__}.{name}: {with_choices(words, choices)}")
            if doc and not pyd and lines:
                out.append(f"{cls.__name__}: {doc}")
            if lines:
                out += [f"{cls.__name__} fields:", *lines, ""]
    return out


def mine(words: Optional[str], note: Optional[Note]) -> str:
    """The one-line words docments found for a name, unless they are a choice's
    comment it read as the name's (``x: Literal["a",  # …`` on the name's line)."""
    if not words or (note is not None and words in note.choices.values()):
        return ""
    return words


_warned: set = set()


def warn_without_source(fn: Any) -> None:
    """Once per function: its code has no source file, so its comments cannot
    reach the model."""
    code = getattr(fn, "__code__", None)
    where = getattr(code, "co_filename", "")
    if where not in ("<stdin>", "<string>") or get_source(fn):
        return
    key = (where, getattr(fn, "__qualname__", ""), getattr(code, "co_firstlineno", 0))
    if key in _warned:
        return
    _warned.add(key)
    run = "from standard input (python - <<EOF)" if where == "<stdin>" else "by exec() (or python -c, before Python 3.13)"
    warnings.warn(
        f"functai: {getattr(fn, '__name__', 'this function')} was defined {run}, so Python keeps no source for it: "
        f"its comments (and those of the records it uses) do not reach the model, and outputs declared in its body "
        f"(`name: T = _ai`) are not asked for. Put the code in a file and run that file; or use what needs no "
        f"source: the docstring, a returned record (dataclass) for several outputs, "
        f"pydantic Field(description=...).", stacklevel=4)


__all__ = ["mine", "Note", "record_notes", "enum_notes", "function_notes", "records_in", "record_sections",
           "with_choices", "warn_without_source", "is_record", "enum_in"]
