"""A new pipeline from a plan, written as ordinary FunctAI code.

A plan is plain data (what a person fills in on the dashboard's "New pipeline"
page, or what the AI helper drafts for them to check first)::

    {"name": "club_inbox", "description": "Sort a club's messages and reply.",
     "inputs": [{"name": "message", "about": "a message sent to the club"}],
     "facts": [{"name": "faq", "about": "the club's FAQ"}],
     "steps": [{"name": "kind", "instruction": "What kind of message is it?",
                "takes": ["message"], "answer": "choices", "choices": ["question", "idea", "spam"]},
               {"name": "reply", "instruction": "Answer from the FAQ.", "takes": ["message", "faq"],
                "answer": "text"}],
     "examples": [{"message": "When is the next meeting?"}],
     "model": "claude-haiku-5-5"}

The code is written by this module, never by a model: one ``@ai`` function per
step, one ``@module`` calling them in order, each fact read from its own file
(``faq.md``, for the person to fill in), and the examples in
``examples/<name>.csv`` (inputs only, never answers). Steps run one after the
other; branches and loops come later, in the dashboard's code view or with a
coding agent.
"""

from __future__ import annotations

import csv
import io
import json
import keyword
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from .edit import ANSWERS, _docstring
from .project import StudioError

PLACEHOLDER = "(Write {about} here. Until you do, the AI only sees this sentence.)"


def slug(text: str, fallback: str = "step") -> str:
    """A Python name from words: "Sort messages!" → sort_messages."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", (text or "").strip().lower()).strip("_")
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"{fallback}_{s}"
    if keyword.iskeyword(s) or s in ("ai", "module", "functai", "open", "print", "str", "int", "list"):
        s = f"{s}_"
    return s[:40]


def check(plan: Dict[str, Any]) -> Dict[str, Any]:
    """The plan made consistent: names Python accepts and unique, each step taking only what exists
    before it, at least one step. Raises StudioError with what to fix."""
    name = slug(plan.get("name") or plan.get("description") or "pipeline", "pipeline")
    used = {name}

    def unique(n: str, fallback: str) -> str:
        n = slug(n, fallback)
        base, i = n, 2
        while n in used:
            n, i = f"{base}_{i}", i + 1
        used.add(n)
        return n

    renamed: Dict[str, str] = {}
    inputs, facts = [], []
    for kind, src, out_, fallback in (("input", plan.get("inputs"), inputs, "text"), ("fact", plan.get("facts"), facts, "notes")):
        for x in src or []:
            orig = (x.get("name") or "").strip()
            if not orig:
                continue
            n = unique(orig, fallback)
            renamed[orig] = n
            about = (x.get("about") or "").strip() or (orig if kind == "fact" else "")
            out_.append({"name": n, "about": about})
    if not inputs:
        inputs.append({"name": unique("text", "text"), "about": "what the pipeline works on"})
    known = {x["name"]: "str" for x in inputs} | {x["name"]: "str" for x in facts}
    steps = []
    for x in plan.get("steps") or []:
        instr = (x.get("instruction") or "").strip()
        if not instr:
            continue
        n = unique(x.get("name") or instr.split(".")[0], "step")
        answer = x.get("answer") or "text"
        choices = [str(c).strip() for c in x.get("choices") or [] if str(c).strip()]
        if answer == "choices" and len(set(choices)) < 2:
            answer = "text"
        if answer not in ANSWERS and answer != "choices":
            answer = "text"
        takes = []
        for t in x.get("takes") or []:
            t = renamed.get(str(t).strip(), slug(str(t)))
            if t in known and t not in takes:
                takes.append(t)
        if not takes:              # a step that takes nothing gets what came just before it
            takes = [steps[-1]["name"]] if steps else [inputs[0]["name"]]
        steps.append({"name": n, "instruction": instr, "takes": takes, "answer": answer, "choices": choices})
        known[n] = _type(answer, choices)
        renamed[(x.get("name") or "").strip()] = n
    if not steps:
        raise StudioError("a pipeline needs at least one step, with what it is asked")
    names = [x["name"] for x in inputs]
    examples = []
    for e in plan.get("examples") or []:
        if isinstance(e, dict):
            back = {n: o for o, n in renamed.items()}
            row = {n: str(e.get(n, e.get(back.get(n, n), ""))) for n in names}
            if any(v.strip() for v in row.values()):
                examples.append(row)
    return {"name": name, "description": (plan.get("description") or "").strip(), "inputs": inputs, "facts": facts,
            "steps": steps, "examples": examples, "model": (plan.get("model") or "").strip() or None}


def _type(answer: str, choices: List[str]) -> str:
    if answer == "choices":
        return "Literal[" + ", ".join(json.dumps(c, ensure_ascii=False) for c in choices) + "]"
    return ANSWERS.get(answer, "str")


def code(plan: Dict[str, Any]) -> str:
    """The pipeline's Python file."""
    p = plan
    types = {x["name"]: "str" for x in p["inputs"]} | {x["name"]: "str" for x in p["facts"]}
    var = {x["name"]: x["name"] for x in p["inputs"]} | {x["name"]: x["name"].upper() for x in p["facts"]}
    lines = [_docstring((p["description"] or p["name"].replace("_", " ").capitalize()) +
                        "\n\nMade with FunctAI Studio. See it, run it and judge it: functai studio open", ""), ""]
    if any(s["answer"] == "choices" for s in p["steps"]):
        lines.append("from typing import Literal\n")
    lines.append("from functai import ai, module\n")
    for f in p["facts"]:
        lines.append(f'{f["name"].upper()} = open("{f["name"]}.md", encoding="utf-8").read()   # {f["about"]}')
    if p["facts"]:
        lines.append("")
    deco = f'@ai(lm={json.dumps(p["model"])})' if p.get("model") else "@ai"
    for s in p["steps"]:
        params = ", ".join(f"{t}: {types[t]}" for t in s["takes"])
        ret = _type(s["answer"], s["choices"])
        text = "\n".join(textwrap.fill(par, 92) for par in s["instruction"].split("\n"))
        lines += ["", deco, f"def {s['name']}({params}) -> {ret}:", "    " + _docstring(text, "    "), ""]
        types[s["name"]] = ret
    module_params = ", ".join(f"{x['name']}: str" for x in p["inputs"])
    last = p["steps"][-1]
    body = []
    for s in p["steps"]:
        v = f"{s['name']}_result"
        body.append(f"    {v} = {s['name']}({', '.join(var[t] for t in s['takes'])})")
        var[s["name"]] = v
    body.append(f"    return {var[last['name']]}")
    lines += ["", "@module", f"def {p['name']}({module_params}) -> {types[last['name']]}:",
              "    " + _docstring(p["description"] or "The pipeline.", "    ")] + body + [""]
    return "\n".join(lines)


def examples_csv(plan: Dict[str, Any]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=[x["name"] for x in plan["inputs"]], lineterminator="\n")
    w.writeheader()
    for e in plan["examples"]:
        w.writerow(e)
    return out.getvalue()


def write(folder: Path, plan: Dict[str, Any]) -> Path:
    """Write the pipeline in ``folder/<name>/``; returns its .py file. Refuses an existing folder."""
    plan = check(plan)
    root = folder / plan["name"]
    if root.exists() and any(root.iterdir()):
        raise StudioError(f"{root} already exists: choose another name")
    src = code(plan)
    compile(src, f"{plan['name']}.py", "exec")         # never write code that does not even parse
    (root / "examples").mkdir(parents=True, exist_ok=True)
    (root / f"{plan['name']}.py").write_text(src, encoding="utf-8")
    (root / "examples" / f"{plan['name']}.csv").write_text(examples_csv(plan), encoding="utf-8")
    for f in plan["facts"]:
        (root / f"{f['name']}.md").write_text(PLACEHOLDER.format(about=f["about"]) + "\n", encoding="utf-8")
    return root / f"{plan['name']}.py"


def blank(name: str = "my_helper") -> Dict[str, Any]:
    return {"name": name, "description": "", "inputs": [{"name": "text", "about": "what it works on"}], "facts": [],
            "steps": [{"name": "answer", "instruction": "Say what this text is about, in one sentence.",
                       "takes": ["text"], "answer": "text", "choices": []}], "examples": []}
