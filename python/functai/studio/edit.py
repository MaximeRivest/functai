"""Changes a person makes in the dashboard, written into the code (and the files it reads).

The code stays the truth: each change is a precise edit of the file's text (the
rest, formatting and comments, untouched), checked to still be Python before it is
written, refused when the file changed since the person looked (``base``), kept
in a history so it can be undone, and visible in git like any other change.

What can be changed:

- a step's model (``@ai(lm=...)``), one step or all;
- a step's instruction (its docstring);
- a step's kind of answer (text, one of a few choices, a number, yes/no, a list);
- which value goes into a step's input (the pipeline's inputs, earlier steps'
  answers, values from the file);
- a fixed value in the pipeline (``"Love it!"``);
- the text of a file a step reads (``FAQ = open("club_faq.md").read()``);
- a new step after another (its answer replaces the other's, downstream), or
  after the end; removing a step whose answer has the type of one of its inputs;
- the whole code, as text.
"""

from __future__ import annotations

import ast
import datetime
import difflib
import hashlib
import json
import keyword
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .project import Project, StudioError


class Conflict(StudioError):
    """The file changed since the person looked at it."""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ text and positions

def _offset(text: str, line: int, col: int) -> int:
    """Character offset of (1-based line, 0-based UTF-8 byte column)."""
    lines = text.splitlines(keepends=True)
    head = "".join(lines[: line - 1])
    return len(head) + len(lines[line - 1].encode("utf-8")[:col].decode("utf-8", "ignore"))


def _span(text: str, node: ast.AST) -> Tuple[int, int]:
    return _offset(text, node.lineno, node.col_offset), _offset(text, node.end_lineno, node.end_col_offset)


def _replace(text: str, edits: List[Tuple[int, int, str]]) -> str:
    for a, b, new in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:a] + new + text[b:]
    return text


def _fdef(tree: ast.Module, name: str) -> ast.FunctionDef:
    f = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
             None)
    if f is None:
        raise StudioError(f"no function {name} at the top of the file")
    return f  # type: ignore[return-value]


def _is_ai(dec: ast.AST) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id == "ai"
    return isinstance(target, ast.Attribute) and target.attr == "ai"


def _ai_def(tree: ast.Module, step: str) -> ast.FunctionDef:
    f = _fdef(tree, step)
    if not any(_is_ai(d) for d in f.decorator_list):
        raise StudioError(f"{step} is not an @ai step: change it in the code")
    return f


def _indent_of(text: str, node: ast.AST) -> str:
    line = text.splitlines()[node.lineno - 1]
    return line[: len(line) - len(line.lstrip())]


def _calls_in(fn: ast.AST, name: str) -> List[ast.Call]:
    return sorted((n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == name), key=lambda n: (n.lineno, n.col_offset))


def _ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or "") or keyword.iskeyword(name):
        raise StudioError(f"{name!r} is not a name Python accepts (letters, digits and _, not starting with a digit)")
    return name


# ------------------------------------------------------------------ writing, history, undo

def _history_dir(project: Project) -> Path:
    return project.studio_dir / "edits"


def history(project: Project) -> List[Dict[str, Any]]:
    path = _history_dir(project) / "history.jsonl"
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return []


def _keep_version(project: Project) -> None:
    from .. import calllog
    from .data import keep_version_code
    try:
        keep_version_code(project, calllog.program_info(project.load())["version"])
    except StudioError:
        pass


def _write(project: Project, path: Path, new: str, *, base: Optional[str], what: str) -> Dict[str, Any]:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if base is not None and path == project.file and sha(old) != base:
        raise Conflict("the file changed since you opened it (your coding agent, or another window): "
                       "look at it again, then make your change")
    if path.suffix == ".py":
        try:
            ast.parse(new)
        except SyntaxError as exc:
            raise StudioError(f"not saved: that would not be valid Python (line {exc.lineno}: {exc.msg})") from None
    if new == old:
        return {"what": what, "diff": [], "sha": sha(new), "unchanged": True}
    if path == project.file:
        _keep_version(project)
    folder = _history_dir(project)
    folder.mkdir(parents=True, exist_ok=True)
    entries = history(project)
    n = (entries[-1]["n"] + 1) if entries else 1
    (folder / f"{n:05d}.before").write_text(old, encoding="utf-8")
    entry = {"n": n, "what": what, "file": str(path.relative_to(project.root)),
             "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "after": sha(new)}
    with (folder / "history.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    tmp = path.with_name(path.name + ".studio-tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(path)
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1))[2:]
    return {"what": what, "diff": diff, "sha": sha(new) if path == project.file else None, "n": n}


def undo(project: Project) -> Dict[str, Any]:
    """Undo the last change made in the dashboard (only if the file is as that change left it)."""
    entries = history(project)
    if not entries:
        raise StudioError("nothing to undo")
    last = entries[-1]
    path = project.root / last["file"]
    now = path.read_text(encoding="utf-8") if path.exists() else ""
    if sha(now) != last["after"]:
        raise Conflict(f"{last['file']} changed after “{last['what']}” (your coding agent, or by hand): "
                       f"undo it there, or use the version list")
    before = (_history_dir(project) / f"{last['n']:05d}.before").read_text(encoding="utf-8")
    tmp = path.with_name(path.name + ".studio-tmp")
    tmp.write_text(before, encoding="utf-8")
    tmp.replace(path)
    lines = [json.dumps(e) for e in entries[:-1]]
    (_history_dir(project) / "history.jsonl").write_text("".join(line + "\n" for line in lines))
    return {"undone": last["what"], "file": last["file"]}


def _text(project: Project) -> Tuple[str, ast.Module]:
    text = project.file.read_text(encoding="utf-8")
    return text, ast.parse(text)


# ------------------------------------------------------------------ models

def set_model(project: Project, step: str, model: str, *, base: Optional[str] = None) -> Dict[str, Any]:
    """Make ``step``'s ``@ai`` say ``lm="model"``."""
    if not model or any(c in model for c in "\"'\n\\"):
        raise StudioError(f"not a model name: {model!r}")
    text, tree = _text(project)
    new, line = _model_edit(text, tree, step, model)
    out = _write(project, project.file, new, base=base, what=f"{step} now uses {model}")
    out.update(step=step, model=model, line=line, before=text.splitlines()[line - 1].strip(),
               after=new.splitlines()[line - 1].strip())
    return out


def _model_edit(text: str, tree: ast.Module, step: str, model: str) -> Tuple[str, int]:
    fdef = _fdef(tree, step)
    dec = next((d for d in fdef.decorator_list if _is_ai(d)), None)
    if dec is None:
        raise StudioError(f"{step} is not decorated with @ai, so its model is not set there")
    lit = json.dumps(model)
    if isinstance(dec, ast.Call):
        kw = next((k for k in dec.keywords if k.arg == "lm"), None)
        if kw is not None:
            a, b = _span(text, kw.value)
            return text[:a] + lit + text[b:], dec.lineno
        at = _offset(text, dec.func.end_lineno, dec.func.end_col_offset)
        close = text.index("(", at) + 1
        sep = ", " if (dec.args or dec.keywords) else ""
        return text[:close] + f"lm={lit}" + sep + text[close:], dec.lineno
    at = _offset(text, dec.end_lineno, dec.end_col_offset)
    return text[:at] + f"(lm={lit})" + text[at:], dec.lineno


def set_all_models(project: Project, steps: List[str], model: str, *, base: Optional[str] = None) -> Dict[str, Any]:
    if not model or any(c in model for c in "\"'\n\\"):
        raise StudioError(f"not a model name: {model!r}")
    text = project.file.read_text(encoding="utf-8")
    for step in steps:
        text, _ = _model_edit(text, ast.parse(text), step, model)
    return _write(project, project.file, text, base=base, what=f"every step now uses {model}")


# ------------------------------------------------------------------ instruction, kind of answer

def _docstring(text: str, indent: str) -> str:
    body = text.strip().replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if not body:
        raise StudioError("an instruction cannot be empty")
    lines = body.splitlines()
    rest = "".join("\n" + (indent + line if line.strip() else "") for line in lines[1:])
    end = ("\n" + indent) if len(lines) > 1 else ""
    return '"""' + lines[0] + rest + end + '"""'


def set_instruction(project: Project, step: str, instruction: str, *, base: Optional[str] = None) -> Dict[str, Any]:
    text, tree = _text(project)
    fdef = _fdef(tree, step)
    first = fdef.body[0]
    indent = _indent_of(text, first)
    lit = _docstring(instruction, indent)
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        a, b = _span(text, first.value)
        new = text[:a] + lit + text[b:]
    else:
        at = _offset(text, first.lineno, 0)
        new = text[:at] + indent + lit + "\n" + text[at:]
    return _write(project, project.file, new, base=base, what=f"changed what {step} is asked")


ANSWERS = {"text": "str", "number": "float", "whole number": "int", "yes/no": "bool", "list of texts": "list[str]"}


def _ensure_literal(text: str, tree: ast.Module) -> str:
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module == "typing" and any(a.name == "Literal" for a in n.names):
            return text
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if imports:
        at = _offset(text, imports[0].lineno, 0)
    else:
        doc = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) else None
        at = _offset(text, doc.end_lineno + 1, 0) if doc else 0
    return text[:at] + "from typing import Literal\n" + text[at:]


def set_answer(project: Project, step: str, kind: str, choices: Optional[List[str]] = None, *,
               base: Optional[str] = None) -> Dict[str, Any]:
    """The kind of answer a step gives: one of ``ANSWERS``, or ``choices`` (one of a few)."""
    text, tree = _text(project)
    fdef = _ai_def(tree, step)
    if kind == "choices":
        cs = [c.strip() for c in (choices or []) if c and c.strip()]
        if len(cs) < 2 or len(set(cs)) != len(cs):
            raise StudioError("give at least two different choices")
        ann = "Literal[" + ", ".join(json.dumps(c, ensure_ascii=False) for c in cs) + "]"
    elif kind in ANSWERS:
        ann = ANSWERS[kind]
    else:
        raise StudioError(f"a kind of answer is one of: choices, {', '.join(ANSWERS)}")
    if fdef.returns is not None:
        a, b = _span(text, fdef.returns)
        new = text[:a] + ann + text[b:]
    else:
        start = _offset(text, fdef.lineno, fdef.col_offset)
        body = _offset(text, fdef.body[0].lineno, 0)
        colon = text.rfind(":", start, body)
        new = text[:colon] + f" -> {ann}" + text[colon:]
    if kind == "choices":
        new = _ensure_literal(new, ast.parse(new))
    return _write(project, project.file, new, base=base, what=f"{step} now answers with {_answer_words(kind, choices)}")


def _answer_words(kind: str, choices: Optional[List[str]]) -> str:
    return "one of: " + ", ".join(c for c in (choices or []) if c) if kind == "choices" else kind


# ------------------------------------------------------------------ wiring: what goes into a step

def _module_def(project: Project, tree: ast.Module) -> ast.FunctionDef:
    project.load()
    return _fdef(tree, project.name)


def _assigned_before(fn: ast.FunctionDef, line: int) -> List[str]:
    names: List[str] = []
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and n.lineno < line:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name) and x.id not in names:
                        names.append(x.id)
    return names


def _module_values(tree: ast.Module) -> List[str]:
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if not isinstance(n.value, (ast.Lambda,)):
                out.append(n.targets[0].id)
    return out


def wiring(project: Project) -> Dict[str, Any]:
    """For each step called in the pipeline: each call's inputs (parameter, what is given) and the names
    that could be given instead at that place. Also the file each module-level value is read from."""
    text, tree = _text(project)
    out: Dict[str, Any] = {"steps": {}, "files": {}}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            for c in ast.walk(n.value):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "open" and c.args \
                        and isinstance(c.args[0], ast.Constant) and isinstance(c.args[0].value, str):
                    out["files"][n.targets[0].id] = c.args[0].value
    try:
        mod = _module_def(project, tree)
    except StudioError:
        return out
    params = [a.arg for a in mod.args.args + mod.args.kwonlyargs]
    values = _module_values(tree)
    steps = {f.name for f in tree.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name != mod.name}
    labels: Dict[str, str] = {p: "the pipeline's input" for p in params}
    for v in values:
        labels.setdefault(v, f"read from {out['files'][v]}" if v in out["files"] else "a value set in the file")
    for n in ast.walk(mod):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            made = [c.func.id for c in ast.walk(n.value) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id in steps]
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(t, ast.Name) and t.id not in labels:
                    fixed = isinstance(n.value, ast.IfExp) and any(
                        isinstance(x, ast.Constant) for x in (n.value.body, n.value.orelse))
                    labels[t.id] = ((" or ".join(f"what {m} gave" for m in dict.fromkeys(made))
                                     + (", or a fixed value" if fixed else ""))
                                    if made else "a value worked out in the pipeline")
    out["labels"] = labels
    for f in tree.body:
        if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) or f.name == mod.name:
            continue
        calls = _calls_in(mod, f.name)
        if not calls:
            continue
        fparams = [a.arg for a in f.args.args]
        entries = []
        for call in calls:
            given = []
            for i, arg in enumerate(call.args):
                if i < len(fparams):
                    given.append({"param": fparams[i], "given": ast.get_source_segment(text, arg),
                                  "simple": isinstance(arg, ast.Name)})
            for kw in call.keywords:
                if kw.arg:
                    given.append({"param": kw.arg, "given": ast.get_source_segment(text, kw.value),
                                  "simple": isinstance(kw.value, ast.Name)})
            sources = params + [n for n in _assigned_before(mod, call.lineno) if n not in params] + \
                [v for v in values if v not in params]
            entries.append({"line": call.lineno, "col": call.col_offset, "inputs": given, "sources": sources})
        out["steps"][f.name] = entries
    return out


def set_input(project: Project, step: str, param: str, source: str, *, line: Optional[int] = None,
              base: Optional[str] = None) -> Dict[str, Any]:
    """Give ``step``'s input ``param`` the value named ``source`` (at the call on ``line``, else every call)."""
    text, tree = _text(project)
    mod = _module_def(project, tree)
    fdef = _fdef(tree, step)
    fparams = [a.arg for a in fdef.args.args]
    allowed = {e["line"]: e["sources"] for e in wiring(project)["steps"].get(step, [])}
    edits = []
    for call in _calls_in(mod, step):
        if line is not None and call.lineno != line:
            continue
        if source not in allowed.get(call.lineno, []):
            raise StudioError(f"{source} is not available where {step} is called (line {call.lineno})")
        target = None
        if param in fparams and fparams.index(param) < len(call.args):
            target = call.args[fparams.index(param)]
        else:
            target = next((k.value for k in call.keywords if k.arg == param), None)
        if target is None:
            raise StudioError(f"{step}'s call on line {call.lineno} gives nothing for {param}")
        a, b = _span(text, target)
        edits.append((a, b, source))
    if not edits:
        raise StudioError(f"the pipeline does not call {step} there")
    return _write(project, project.file, _replace(text, edits), base=base, what=f"{step}'s {param} now comes from {source}")


# ------------------------------------------------------------------ fixed values, files

def set_value(project: Project, line: int, col: int, value: Any, *, base: Optional[str] = None) -> Dict[str, Any]:
    """A fixed value written in the pipeline (``"Love it!"``), at its place in the file."""
    text, tree = _text(project)
    mod = _module_def(project, tree)
    node = next((n for n in ast.walk(mod) if isinstance(n, ast.Constant) and n.lineno == line and n.col_offset == col),
                None)
    if node is None:
        raise StudioError("that value is no longer at that place in the file: look again")
    if isinstance(node.value, str):
        lit = json.dumps(str(value), ensure_ascii=False)
    elif isinstance(node.value, bool):
        lit = "True" if str(value).lower() in ("true", "1", "yes") else "False"
    elif isinstance(node.value, (int, float)):
        try:
            lit = repr(type(node.value)(value))
        except (TypeError, ValueError):
            raise StudioError(f"{value!r} is not a {type(node.value).__name__}") from None
    else:
        raise StudioError("only text, numbers and yes/no values can be changed here")
    a, b = _span(text, node)
    return _write(project, project.file, text[:a] + lit + text[b:], base=base, what="changed a fixed value")


def set_file(project: Project, name: str, content: str) -> Dict[str, Any]:
    """The text of a file a module-level value is read from (``FAQ = open("club_faq.md").read()``)."""
    rel = wiring(project)["files"].get(name)
    if rel is None:
        raise StudioError(f"{name} is not read from a file")
    path = (project.root / rel).resolve()
    if project.root.resolve() not in path.parents:
        raise StudioError("that file is outside the project")
    return _write(project, path, content, base=None, what=f"changed {rel}")


# ------------------------------------------------------------------ adding and removing steps

def _annotation(text: str, node: Optional[ast.AST], default: str = "str") -> str:
    return ast.get_source_segment(text, node) if node is not None else default


def add_step(project: Project, after: str, name: str, instruction: str, *, answer: str = "same",
             choices: Optional[List[str]] = None, extra: Optional[List[str]] = None,
             base: Optional[str] = None) -> Dict[str, Any]:
    """A new step that takes ``after``'s answer (and the pipeline inputs in ``extra``) and whose answer
    goes on in its place. ``after="(end)"``: the step takes the pipeline's result before it is returned."""
    _ident(name)
    text, tree = _text(project)
    mod = _module_def(project, tree)
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name for n in tree.body) \
            or name in _module_values(tree):
        raise StudioError(f"the file already has something named {name}")
    mparams = {a.arg: _annotation(text, a.annotation) for a in mod.args.args + mod.args.kwonlyargs}
    extra = [e for e in (extra or []) if e]
    for e in extra:
        if e not in mparams:
            raise StudioError(f"{e} is not one of the pipeline's inputs ({', '.join(mparams)})")
    if after == "(end)":
        first, first_type = "result", _annotation(text, mod.returns)
        anchor: ast.AST = mod
        targets = [n.value for n in ast.walk(mod) if isinstance(n, ast.Return) and n.value is not None
                   and not isinstance(n.value, ast.Constant)]
    else:
        prev = _fdef(tree, after)
        first, first_type = after, _annotation(text, prev.returns)
        anchor = prev
        targets = _calls_in(mod, after)
    if not targets:
        raise StudioError(f"the pipeline does not use {after}" if after != "(end)" else "the pipeline returns no answer to work on")
    if answer == "same":
        ret = first_type
    elif answer == "choices":
        cs = [c.strip() for c in (choices or []) if c and c.strip()]
        if len(cs) < 2:
            raise StudioError("give at least two choices")
        ret = "Literal[" + ", ".join(json.dumps(c, ensure_ascii=False) for c in cs) + "]"
    elif answer in ANSWERS:
        ret = ANSWERS[answer]
    else:
        raise StudioError("unknown kind of answer")
    params = ", ".join([f"{first}: {first_type}"] + [f"{e}: {mparams[e]}" for e in extra])
    deco = "@ai"                       # a new step after another uses the same decorator (its model too)
    d = next((d for d in getattr(anchor, "decorator_list", []) if _is_ai(d)), None) if after != "(end)" else None
    if d is not None:
        deco = "@" + ast.get_source_segment(text, d)
    func = f"{deco}\ndef {name}({params}) -> {ret}:\n    {_docstring(instruction, '    ')}\n"
    if after == "(end)":               # before the pipeline
        insert = _offset(text, mod.decorator_list[0].lineno if mod.decorator_list else mod.lineno, 0)
        func = func + "\n\n"
    else:                              # right after the step it follows
        insert = _offset(text, anchor.end_lineno, 0) + len(text.splitlines(keepends=True)[anchor.end_lineno - 1])
        func = "\n\n" + func
    edits = []
    for t in targets:
        a, b = _span(text, t)
        edits.append((a, b, f"{name}({text[a:b]}{''.join(', ' + e for e in extra)})"))
    edits.append((insert, insert, func))
    new = _replace(text, edits)
    if answer == "choices":
        new = _ensure_literal(new, ast.parse(new))
    where = "at the end" if after == "(end)" else f"after {after}"
    return _write(project, project.file, new, base=base, what=f"added the step {name} {where}")


def remove_step(project: Project, step: str, *, base: Optional[str] = None) -> Dict[str, Any]:
    """Take a step out: each call is replaced by its input of the same type as its answer."""
    text, tree = _text(project)
    mod = _module_def(project, tree)
    fdef = _ai_def(tree, step)
    ret = _annotation(text, fdef.returns, "")
    params = [a.arg for a in fdef.args.args]
    same = next((i for i, a in enumerate(fdef.args.args) if ret and _annotation(text, a.annotation, "") == ret), None)
    if same is None:
        raise StudioError(f"{step} cannot simply be taken out: none of its inputs is the same kind as its answer, "
                          f"so what replaces it is a choice for you or your coding agent to make in the code")
    edits = []
    for call in _calls_in(mod, step):
        arg = call.args[same] if same < len(call.args) else next(
            (k.value for k in call.keywords if k.arg == params[same]), None)
        if arg is None:
            raise StudioError(f"{step}'s call on line {call.lineno} gives nothing for {params[same]}")
        edits.append((*_span(text, call), ast.get_source_segment(text, arg)))
    new = _replace(text, edits)
    tree2 = ast.parse(new)
    still = any(isinstance(n, ast.Name) and n.id == step for n in ast.walk(tree2))
    if not still:
        f2 = _fdef(tree2, step)
        lines = new.splitlines(keepends=True)
        start = (f2.decorator_list[0].lineno if f2.decorator_list else f2.lineno) - 1
        end = f2.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        while start > 0 and not lines[start - 1].strip() and end < len(lines):
            start -= 1
        new = "".join(lines[:start] + (["\n"] if start else []) + lines[end:])
    return _write(project, project.file, new, base=base, what=f"took out the step {step}")


def set_source(project: Project, code: str, *, base: Optional[str] = None) -> Dict[str, Any]:
    return _write(project, project.file, code, base=base, what="changed the code")


# ------------------------------------------------------------------ examples

def _read_examples(project: Project) -> Tuple[Path, List[str], List[Dict[str, Any]]]:
    """(file, columns, rows) as they are in the file; a new CSV when there is none yet."""
    import csv
    path = project.examples_path()
    names = project.input_names()
    if not path.exists():
        return path, names, []
    if path.suffix == ".jsonl":
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        cols = list(dict.fromkeys(names + [k for r in rows for k in r]))
        return path, cols, rows
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = list(dict.fromkeys(list(r.fieldnames or []) + names))
    return path, cols, rows


def _write_examples(project: Project, path: Path, cols: List[str], rows: List[Dict[str, Any]], what: str) -> Dict[str, Any]:
    import csv
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    else:
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
        text = out.getvalue()
    res = _write(project, path, text, base=None, what=what)
    res["count"] = len(rows)
    return res


def _clean(project: Project, values: Dict[str, Any]) -> Dict[str, Any]:
    names = project.input_names()
    row = {n: values.get(n, "") for n in names}
    if not any(str(v).strip() for v in row.values()):
        raise StudioError("an example needs at least one input filled in")
    return row


def add_examples(project: Project, rows: List[Dict[str, Any]], *, what: Optional[str] = None) -> Dict[str, Any]:
    path, cols, have = _read_examples(project)
    new = [_clean(project, r) for r in rows]
    seen = {json.dumps({c: str(r.get(c, "")) for c in project.input_names()}, sort_keys=True) for r in have}
    fresh = [r for r in new if json.dumps({c: str(r.get(c, "")) for c in project.input_names()}, sort_keys=True) not in seen]
    if not fresh:
        raise StudioError("those examples are already there")
    n = len(fresh)
    out = _write_examples(project, path, cols, have + fresh,
                          what or ("added an example" if n == 1 else f"added {n} examples"))
    out["added"] = n
    out["skipped"] = len(new) - n
    return out


def update_example(project: Project, n: int, values: Dict[str, Any]) -> Dict[str, Any]:
    path, cols, have = _read_examples(project)
    if not 1 <= n <= len(have):
        raise StudioError(f"no example #{n}")
    have[n - 1] = {**have[n - 1], **_clean(project, values)}
    return _write_examples(project, path, cols, have, f"changed example #{n}")


def delete_example(project: Project, n: int) -> Dict[str, Any]:
    path, cols, have = _read_examples(project)
    if not 1 <= n <= len(have):
        raise StudioError(f"no example #{n}")
    del have[n - 1]
    return _write_examples(project, path, cols, have, f"removed example #{n}")


def paste_examples(project: Project, text: str) -> Dict[str, Any]:
    """Many examples at once: one per line when the pipeline takes one input; a table with a
    header row (CSV, or columns separated by tabs, as copied from a spreadsheet) otherwise,
    or when the first line names the inputs."""
    import csv
    import io
    names = project.input_names()
    text = (text or "").replace("\r\n", "\n").strip("\n")
    if not text.strip():
        raise StudioError("nothing to add: paste one example per line, or a table with a header row")
    first = text.split("\n", 1)[0]
    delim = "\t" if "\t" in first else ","
    header = [h.strip().lower() for h in next(csv.reader([first], delimiter=delim))]
    if set(names) <= set(header) or (len(names) > 1):
        if not set(names) <= set(header):
            raise StudioError(f"the first line must name the inputs ({', '.join(names)}), separated by commas or tabs")
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames or []]
        rows = [{n: (r.get(n) or "").strip() for n in names} for r in reader]
    else:
        rows = [{names[0]: line.strip()} for line in text.split("\n") if line.strip()]
    rows = [r for r in rows if any(v for v in r.values())]
    return add_examples(project, rows)
