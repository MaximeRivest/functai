"""Grade the `files` task: answer questions from a PDF report and a spreadsheet.

Checks (from answers.csv and the recording proxy's log of every request the task sent)
- files:    answers.csv with id and answer for the 5 questions
- real:     the documents' content reached the model (text from report.pdf and budget.xlsx)
- picture:  the chart reached the model as a picture (an image part in some request)
- chart:    the two answers only the chart holds (q1, q2) are right
- text:     the three answers in the text and the sheet (q3, q4, q5) are right
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common import check, csv_rows, norm, recorded, summarize

HERE = Path(__file__).parent
TRUTH = json.loads((HERE / "truth.json").read_text())


def _norm(answer: str) -> str:
    a = norm(answer).replace("$", "").replace(",", "").replace(" dollars", "").rstrip(".")
    a = re.sub(r"\.0+$", "", a)
    return a


def right(qid: str, answer: str) -> bool:
    a = _norm(answer)
    return any(a == x or re.fullmatch(rf"(the )?{re.escape(x)}( region| department)?", a) for x in TRUTH[qid]["accept"])


def grade(work: Path, env: dict, python: str, out: Path) -> dict:
    c = {}
    rows = csv_rows(work / "answers.csv")
    ok = rows is not None and rows and {"id", "answer"} <= set(rows[0])
    by = {r["id"].strip(): r["answer"] for r in rows} if ok else {}
    ok = ok and set(TRUTH) <= set(by)
    c["files"] = check(ok, detail="answers.csv with id and answer for q1..q5")
    sent = recorded(env.get("BENCH_RECORD"))
    text = "\n".join(x["text"] for x in sent)
    seen = ["Larkspur Supplies" in text or "board approved a dividend" in text, "Marketing" in text and "515000" in text.replace(",", "")]
    c["real"] = check(all(seen), None, f"report text sent: {seen[0]}; spreadsheet sent: {seen[1]}")
    c["picture"] = check(any(x["images"] for x in sent), None,
                         f"{sum(1 for x in sent if x['images'])} requests carried an image part")
    if not ok:
        c["chart"] = check(False, 0.0, "no usable output")
        c["text"] = check(False, 0.0, "no usable output")
        return summarize(c, required=list(c))
    chart = [q for q in ("q1", "q2") if right(q, by[q])]
    rest = [q for q in ("q3", "q4", "q5") if right(q, by[q])]
    c["chart"] = check(len(chart) == 2, len(chart) / 2, "answers: " + ", ".join(f"{q}={by[q]!r}" for q in ("q1", "q2")))
    c["text"] = check(len(rest) == 3, len(rest) / 3, "answers: " + ", ".join(f"{q}={by[q]!r}" for q in ("q3", "q4", "q5")))
    return summarize(c, required=list(c))
