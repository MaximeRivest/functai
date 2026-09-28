"""The cases in rated/. Each case is a log (call and rating records) and
the rows ``rated`` must make from it, written from the rules in
../calls.md (section "Rows with known answers"), not from any output.
make.py writes them.

A case file: {"description", "records": [...], "rated": {name, module,
signature, by}, "expect": {"rows": [...], "left_out": {...}}}. An
implementation passes a case when rated(records, **rated) gives exactly
``expect`` (rows in order, keys and values; left_out counts).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIG = "sha256:" + "5" * 64
SIG_OLD = "sha256:" + "4" * 64
V1 = "sha256:" + "1" * 64
V2 = "sha256:" + "2" * 64
LEFT = {"other_signature": 0, "no_content": 0, "no_answer": 0}


def cid(n):
    return f"01926a8e-{n:04x}-7000-8000-000000000000"


def rid(n):
    return f"01926a90-{n:04x}-7000-8000-000000000000"


def at(minute, second=0):
    return f"2026-09-26T10:{minute:02d}:{second:02d}.000000Z"


def call(n, minute, inputs, outputs, *, name="team", module="support", version=V1, signature=SIG,
         answer="result", content=True, error=None, omitted=None):
    """``omitted``: {"inputs": [...], "outputs": [...]}, the fields written as their size only
    (calls.md, "Content"); the record then keeps the other values, with content false."""
    rec = {"functai_call": 1, "id": cid(n), "parent": None, "root": cid(n),
           "program": {"name": name, "kind": "ai", "module": module, "version": version, "signature": signature,
                       "answer": answer},
           "started": at(minute), "seconds": 0.4, "content": content if omitted is None else False}
    if omitted is not None:
        rec["omitted"] = omitted
        kept_in = {k: v for k, v in inputs.items() if k not in omitted["inputs"]}
        kept_out = None if outputs is None else {k: v for k, v in outputs.items() if k not in omitted["outputs"]}
        if kept_in:
            rec["inputs"] = kept_in
        if kept_out is None or kept_out:
            rec["outputs"] = kept_out
    elif content:
        rec["inputs"], rec["outputs"] = inputs, outputs
    rec["sizes"] = {"inputs": {k: len(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
                               for k, v in inputs.items()},
                    "outputs": {k: len(json.dumps(v, ensure_ascii=False, separators=(",", ":")))
                                for k, v in (outputs or {}).items()}}
    rec.update(error=error, model="gpt-4.1-mini", usage={"input_tokens": 200, "output_tokens": 3}, confidence=None,
               exchanges=[], caller={}, process={"host": "lambda", "pid": 1, "user": "maxime", "language": "python",
                                                 "runtime": "3.13.1", "functai": "1.1.0"})
    return rec


def rating(n, call_n, minute, by, verdict, **extra):
    return {"functai_rating": 1, "id": rid(n), "call": cid(call_n), "at": at(minute), "by": by,
            "verdict": verdict, **extra}


def row(inputs, answers, n, *, rating_, by, version=V1, origin="review", sample=None, disputed=False):
    return {**inputs, **answers, "call": cid(n), "version": version, "rating": rating_, "rated_by": by,
            "origin": origin, "sample": sample, "disputed": disputed}


M1 = {"message": "I was charged twice for one order."}
M2 = {"message": "My parcel never came."}
M3 = {"message": "The kettle broke on day one."}
TEAM = {"name": "team", "module": "support", "signature": SIG, "by": None}

CASES = {
    "01-right-means-the-answer-it-gave": dict(
        description="A right verdict makes the call's own answer the known answer. Unrated calls make no rows.",
        records=[call(1, 1, M1, {"result": "billing"}), call(2, 2, M2, {"result": "shipping"}),
                 rating(1, 1, 5, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [row(M1, {"result": "billing"}, 1, rating_="right", by="maxime")], "left_out": LEFT}),
    "02-a-correction-is-the-answer": dict(
        description="A wrong verdict with an answer makes that answer the known one; the rating's origin, "
                    "sample and version travel with the row.",
        records=[call(1, 1, M2, {"result": "billing"}, version=V2),
                 rating(1, 1, 5, "maxime", "wrong", answer="shipping", origin="edit", sample="draw-1",
                        reasons=["wrong team"], note="A lost parcel is shipping.")],
        rated=TEAM,
        expect={"rows": [row(M2, {"result": "shipping"}, 1, rating_="wrong", by="maxime", version=V2,
                             origin="edit", sample="draw-1")], "left_out": LEFT}),
    "03-wrong-without-an-answer-is-left-out": dict(
        description="A wrong verdict with no answer says what the answer is not: no row, counted no_answer. "
                    "So is a right verdict on a call that has no answer (it failed).",
        records=[call(1, 1, M1, {"result": "billing"}), call(2, 2, M2, None, error={"type": "StepLimit"}),
                 call(3, 3, M3, {"result": "product"}),
                 rating(1, 1, 5, "maxime", "wrong"), rating(2, 2, 6, "maxime", "right"),
                 rating(3, 3, 7, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [row(M3, {"result": "product"}, 3, rating_="right", by="maxime")],
                "left_out": {**LEFT, "no_answer": 2}}),
    "04-each-persons-latest-rating-counts": dict(
        description="A person's later rating replaces their earlier one (records in any order); a null "
                    "verdict withdraws it; equal times are ordered by id.",
        records=[call(1, 1, M1, {"result": "billing"}), call(2, 2, M2, {"result": "billing"}),
                 call(3, 3, M3, {"result": "billing"}),
                 rating(2, 1, 9, "ana", None), rating(1, 1, 5, "ana", "right"),
                 rating(4, 2, 7, "ben", "right"), rating(3, 2, 7, "ben", "wrong", answer="shipping"),
                 rating(5, 3, 7, "cy", "wrong", answer="product"), rating(6, 3, 7, "cy", "right")],
        rated=TEAM,
        expect={"rows": [row(M2, {"result": "billing"}, 2, rating_="right", by="ben"),
                         row(M3, {"result": "billing"}, 3, rating_="right", by="cy")], "left_out": LEFT}),
    "05-disagreement-is-disputed": dict(
        description="When people's current ratings differ, the latest usable one gives the values and "
                    "disputed is true. Two corrections to different answers disagree too; a wrong verdict "
                    "without an answer next to a right one disagrees but gives no values.",
        records=[call(1, 1, M1, {"result": "billing"}), call(2, 2, M2, {"result": "billing"}),
                 call(3, 3, M3, {"result": "product"}),
                 rating(1, 1, 5, "ana", "right"), rating(2, 1, 6, "ben", "wrong", answer="shipping"),
                 rating(3, 2, 5, "ana", "wrong", answer="shipping"), rating(4, 2, 6, "ben", "wrong", answer="product"),
                 rating(5, 3, 5, "ana", "right"), rating(6, 3, 6, "ben", "wrong")],
        rated=TEAM,
        expect={"rows": [row(M1, {"result": "shipping"}, 1, rating_="wrong", by="ben", disputed=True),
                         row(M2, {"result": "product"}, 2, rating_="wrong", by="ben", disputed=True),
                         row(M3, {"result": "product"}, 3, rating_="right", by="ana", disputed=True)],
                "left_out": LEFT}),
    "06-another-signature-is-left-out": dict(
        description="Knowing the program's signature, calls made with another are left out "
                    "(other_signature); without it (signature null), all are used.",
        records=[call(1, 1, M1, {"result": "billing"}, signature=SIG_OLD), call(2, 2, M2, {"result": "shipping"}),
                 rating(1, 1, 5, "maxime", "right"), rating(2, 2, 5, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [row(M2, {"result": "shipping"}, 2, rating_="right", by="maxime")],
                "left_out": {**LEFT, "other_signature": 1}}),
    "07-no-content-is-left-out": dict(
        description="A call logged without its values has no inputs to learn from (no_content).",
        records=[call(1, 1, M1, {"result": "billing"}, content=False), rating(1, 1, 5, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [], "left_out": {**LEFT, "no_content": 1}}),
    "08-other-outputs-name-and-module": dict(
        description="A correction may give other outputs (after the answer, in the rating's order); the "
                    "answer's name comes from the call's program.answer; a same-named program of another "
                    "module is not this one; only corrected outputs become columns.",
        records=[call(1, 1, M1, {"reasoning": "They mention a charge.", "category": "billing", "priority": 1},
                      answer="category"),
                 call(2, 2, M2, {"reasoning": "Lost parcel.", "category": "shipping", "priority": 1},
                      answer="category"),
                 call(3, 3, M3, {"result": "product"}, module="other"),
                 rating(1, 1, 5, "maxime", "wrong", answer="billing", outputs={"priority": 3}),
                 rating(2, 2, 5, "maxime", "right"), rating(3, 3, 5, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [row(M1, {"category": "billing", "priority": 3}, 1, rating_="wrong", by="maxime"),
                         row(M2, {"category": "shipping"}, 2, rating_="right", by="maxime")],
                "left_out": LEFT}),
    "09-one-persons-ratings-only": dict(
        description="With by, only that person's ratings count; the others' do not dispute them.",
        records=[call(1, 1, M1, {"result": "billing"}), call(2, 2, M2, {"result": "billing"}),
                 rating(1, 1, 5, "ana", "right"), rating(2, 1, 6, "ben", "wrong", answer="shipping"),
                 rating(3, 2, 5, "ben", "wrong", answer="shipping")],
        rated={**TEAM, "by": "ana"},
        expect={"rows": [row(M1, {"result": "billing"}, 1, rating_="right", by="ana")], "left_out": LEFT}),
    "10-rows-follow-the-calls-order": dict(
        description="Rows are in the order of the calls' started, then id (not the id alone, nor the order "
                    "of the lines); "
                    "structured values stay JSON.",
        records=[call(2, 1, {"order": {"id": 7, "items": ["mug"]}}, {"result": "shipping"}),
                 call(1, 1, {"order": {"id": 6, "items": []}}, {"result": "billing"}),
                 call(3, 0, {"order": {"id": 5, "items": ["lamp", "cord"]}}, {"result": "product"}),
                 rating(1, 3, 5, "maxime", "right"), rating(2, 2, 5, "maxime", "right"),
                 rating(3, 1, 5, "maxime", "right")],
        rated={**TEAM, "signature": None},
        expect={"rows": [row({"order": {"id": 5, "items": ["lamp", "cord"]}}, {"result": "product"}, 3,
                             rating_="right", by="maxime"),
                         row({"order": {"id": 6, "items": []}}, {"result": "billing"}, 1, rating_="right", by="maxime"),
                         row({"order": {"id": 7, "items": ["mug"]}}, {"result": "shipping"}, 2, rating_="right",
                             by="maxime")],
                "left_out": LEFT}),
    "11-data-keeps-its-names": dict(
        description="An input or output named like a column about the rating keeps its name; that column "
                    "gets underscores in front until its name is free.",
        records=[call(1, 1, {"call": "support line", "version": "2", "_version": "draft"}, {"result": "billing"}),
                 rating(1, 1, 5, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [{"call": "support line", "version": "2", "_version": "draft", "result": "billing",
                          "_call": cid(1), "__version": V1, "rating": "right", "rated_by": "maxime",
                          "origin": "review", "sample": None, "disputed": False}],
                "left_out": LEFT}),
    "12-some-inputs-not-written": dict(
        description="A call whose record kept some values but not every input (content false, omitted.inputs "
                    "not empty) has no inputs to ask again: left out, no_content, as a call with no values.",
        records=[call(1, 1, {**M1, "transcript": "…"}, {"result": "billing"},
                      omitted={"inputs": ["transcript"], "outputs": []}),
                 rating(1, 1, 5, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [], "left_out": {**LEFT, "no_content": 1}}),
    "13-every-input-written-an-output-not": dict(
        description="A call whose record kept every input but not an output makes rows: a correction gives "
                    "the answer; a right verdict gives nothing when the answer itself was not written "
                    "(no_answer), and the call's other outputs when they were.",
        records=[call(1, 1, M1, {"summary": "Charged twice.", "result": "billing"}, answer="result",
                      omitted={"inputs": [], "outputs": ["result"]}),
                 call(2, 2, M2, {"summary": "Lost parcel.", "result": "shipping"}, answer="result",
                      omitted={"inputs": [], "outputs": ["result"]}),
                 call(3, 3, M3, {"summary": "Broken kettle.", "result": "product"}, answer="result",
                      omitted={"inputs": [], "outputs": ["summary"]}),
                 rating(1, 1, 5, "maxime", "wrong", answer="shipping"), rating(2, 2, 5, "maxime", "right"),
                 rating(3, 3, 5, "maxime", "right")],
        rated=TEAM,
        expect={"rows": [row(M1, {"result": "shipping"}, 1, rating_="wrong", by="maxime"),
                         row(M3, {"result": "product"}, 3, rating_="right", by="maxime")],
                "left_out": {**LEFT, "no_answer": 1}}),
}
