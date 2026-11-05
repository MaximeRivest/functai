"""The cases in measures/, written from ../measures.md by the rules in
measure_rules.py (never from an implementation's output). make.py writes them.

Every case is {"description", "kind", ...the operation's data..., "expect"}.
Its kind is the prefix of its file name; ../cases/README.md, `measures/`,
says what a harness does with each. Numbers are compared within 1e-9.
"""

import json

import measure_rules as m

S1 = "sha256:" + "a1" * 32
S2 = "sha256:" + "b2" * 32
S3 = "sha256:" + "c3" * 32
V1 = "sha256:" + "1" * 64
V2 = "sha256:" + "2" * 64
IFACE = "sha256:" + "9" * 64
JUDGE = "sha256:" + "d4" * 32
JUDGE2 = "sha256:" + "e5" * 32
CODE = "sha256:" + "f6" * 32


def cid(n):
    return f"01926c00-{n:04x}-7000-8000-000000000000"


def rid(n):
    return f"01926c10-{n:04x}-7000-8000-000000000000"


def at(minute, second=0):
    return f"2026-11-05T10:{minute:02d}:{second:02d}.000000Z"


def size(v):
    return len(json.dumps(v, ensure_ascii=False, separators=(",", ":")))


def call(n, minute, inputs, outputs, *, setup=S1, version=V1, name="study", kind="module", error=None,
         case=None, repeat=None, parent=None, root=None, model=None, usage=None, spent=None, saw=None,
         content=True, fmt=2, answer="result", seconds=2.0):
    """A call record (calls.md, format 2 unless fmt says 1)."""
    program = {"name": name, "kind": kind, "module": "groups.demo", "version": version}
    if kind == "ai":
        program["signature"] = IFACE
    if fmt >= 2:
        program["interface"] = IFACE
    program["answer"] = answer
    if setup is not None and fmt >= 2:
        program["setup"] = setup
    rec = {"functai_call": fmt, "id": cid(n), "parent": cid(parent) if parent else None,
           "root": cid(root or parent or n), "program": program, "started": at(minute), "seconds": seconds,
           "content": content}
    if content:
        rec["inputs"], rec["outputs"] = inputs, outputs
    elif fmt >= 2:
        rec["omitted"] = {"inputs": list(inputs), "outputs": list(outputs or {})}
    rec["sizes"] = {"inputs": {k: size(v) for k, v in inputs.items()},
                    "outputs": {k: size(v) for k, v in (outputs or {}).items()}}
    rec.update(error=error, model=model, usage=usage if usage is not None else {}, confidence=None, exchanges=[])
    if spent is not None:
        rec["spent"] = {"usd": spent}
    if fmt >= 2:
        rec["saw"] = saw or []
    rec["caller"] = {}
    rec["process"] = {"host": "lambda", "pid": 7, "user": "maxime", "language": "python", "runtime": "3.13.1",
                      "functai": "1.3.0"}
    if case is not None:
        rec["case"] = case
    if repeat is not None:
        rec["repeat"] = repeat
    return rec


def r1(n, call_n, minute, by, verdict, **extra):
    who = {"by": by} if by is not None else {"account": "maxime"}
    return {"functai_rating": 1, "id": rid(n), "call": cid(call_n), "at": at(minute), **who, "verdict": verdict, **extra}


def r2(n, target, minute, *, criterion, kind, value, by=None, account=None, judge=None, item=None, **extra):
    """A format-2 rating. target: a call number, or two (a pair, in the order shown)."""
    rec = {"functai_rating": 2, "id": rid(n), "at": at(minute)}
    if isinstance(target, tuple):
        rec["calls"] = [cid(target[0]), cid(target[1])]
    else:
        rec["call"] = cid(target)
    if item is not None:
        rec["item"] = item
    rec.update(criterion=criterion, kind=kind)
    rec.update(extra)
    rec["value"] = value
    if by is not None:
        rec["by"] = by
    elif judge is not None:
        rec["judge"] = judge
    else:
        rec["account"] = account or "maxime"
    return rec


REAL = {"name": "real_problem", "kind": "label", "items": "/insights"}
ANSWERS = {"yes": 1, "no": 0, "unsure": None}
LABEL = dict(criterion="real_problem", kind="label", question="Is this a real problem?", answers=ANSWERS)
CORRECT = {"name": "correct", "kind": "verdict"}
QUALITY = {"name": "quality", "kind": "score"}
PREF = {"name": "preferred", "kind": "choice"}
PROF = {"name": "professor", "setup": JUDGE}
PROF2 = {"name": "professor", "setup": JUDGE2}
CHECKER = {"name": "quotes_found", "code": CODE}
STUDY = {"name": "study"}
SITE = [{"url": f"https://site{i}.example"} for i in range(10)]


def ins(*titles):
    return {"insights": [{"title": t} for t in titles]}


def key(inputs):
    return m.case_key(inputs)


def jkey(obj):
    """JSON keys in cases: a group or a case as it is, null as "null"."""
    return "null" if obj is None else obj


def cases_list(cases):
    return [{"group": g, "case": k, "value": v} for g, cs in cases.items() for k, v in cs.items()]


# ------------------------------------------------------------------ key


def key_cases():
    out = {}
    rows = {
        "01-only-the-inputs": ("A row's key is its values of the program's inputs: other columns (the answer, a "
                               "tag, the split) do not change it.",
                               {"message": "Charged twice.", "result": "billing", "tag": "x", "split": "train"},
                               ["message"]),
        "02-member-order-does-not-count": ("Canonical JSON: the order of an object's members does not count.",
                                           {"order": {"b": 2, "a": 1}}, ["order"]),
        "03-numbers-by-value": ("1 and 1.0 are one number.", {"n": 1.0}, ["n"]),
        "04-an-input-left-out": ("An optional input the row leaves out is not in the key (the program's default "
                                 "is the version's, not the case's).", {"text": "hi"}, ["text", "tone"]),
        "05-earlier-turns": ("A row that keeps its conversation's earlier turns: they are part of the case.",
                             {"question": "And tomorrow?", "earlier": [{"inputs": {"question": "Weather today?"},
                                                                       "outputs": {"result": "Sunny."}}],
                              "conversation": "c-1"}, ["question"]),
        "06-no-earlier-turns": ("An empty `earlier` is a first turn: the same case as a row without it.",
                                {"question": "Weather today?", "earlier": [], "conversation": "c-1"}, ["question"]),
        "07-an-input-named-earlier": ("An input named like a context key is an input.",
                                      {"earlier": "yesterday"}, ["earlier"]),
        "08-pictures-by-their-bytes": ("A picture is its data, as a record writes it.",
                                       {"page": {"type": "image", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                                       ["page"]),
    }
    for name, (text, row, inputs) in rows.items():
        out[name] = {"description": text, "kind": "key-row", "row": row, "inputs": inputs,
                     "expect": {"key": m.row_key(row, inputs)}}
    recs = {
        "09-a-calls-own-case": ("A call made by evaluate carries its case: that is its key.",
                                call(1, 1, {"text": "hi", "tone": "kind"}, {"result": "ok"}, case=key({"text": "hi"}))),
        "10-a-live-call": ("A call without `case` that saw nothing: the key of its recorded inputs (a shadow run "
                           "on rows made from them meets it).",
                           call(2, 2, {"text": "hi", "tone": "kind"}, {"result": "ok"})),
        "11-a-turn-that-saw-earlier-calls": ("A call without `case` that saw earlier calls has no key: its context "
                                             "is not in its inputs.",
                                             call(3, 3, {"text": "hi"}, {"result": "ok"}, saw=[{"call": cid(1)}])),
        "12-values-not-recorded": ("A call whose inputs were not recorded has no key.",
                                   call(4, 4, {"text": "hi"}, {"result": "ok"}, content=False)),
        "13-format-1": ("A format-1 record: what it saw was not recorded, so it has no key.",
                        call(5, 5, {"text": "hi"}, {"result": "ok"}, fmt=1, setup=None)),
    }
    for name, (text, rec) in recs.items():
        out[name] = {"description": text, "kind": "key-call", "record": rec, "expect": {"key": m.call_key(rec)}}
    return out


# ------------------------------------------------------------------ split


def split_cases():
    out = {}
    single = {
        "01-its-own-split": ("A row's `split` is its own.", {"q": "a", "split": "held_out"}, ["q"], 0.2, ""),
        "02-train-stays": ("Even when its hash would hold it out.", {"q": "a", "split": "train"}, ["q"], 1.0, ""),
        "03-share-0": ("A share of 0 holds nothing out.", {"q": "a"}, ["q"], 0.0, ""),
        "04-share-1": ("A share of 1 holds everything out.", {"q": "a"}, ["q"], 1.0, ""),
        "05-an-input-named-split": ("An input named `split` is an input, not a side.", {"split": "x"}, ["split"], 0.5, ""),
    }
    for name, (text, row, inputs, share, salt) in single.items():
        out[name] = {"description": text, "kind": "split", "rows": [row], "inputs": inputs, "share": share, "salt": salt,
                     "expect": {"splits": [m.split_of(row, inputs, share, salt)]}}
    rows = [{"q": f"question {i}"} for i in range(40)]
    out["06-forty-rows"] = {"description": "Forty rows, a share of 0.25: each side by the hash of its key.",
                            "kind": "split", "rows": rows, "inputs": ["q"], "share": 0.25, "salt": "",
                            "expect": {"splits": [m.split_of(r, ["q"], 0.25, "") for r in rows]}}
    out["07-a-salt"] = {"description": "A salt draws another split of the same rows.", "kind": "split",
                        "rows": rows, "inputs": ["q"], "share": 0.25, "salt": "second",
                        "expect": {"splits": [m.split_of(r, ["q"], 0.25, "second") for r in rows]}}
    conv = [{"q": f"turn {i}", "conversation": f"c-{i // 4}"} for i in range(24)]
    out["08-a-conversation-on-one-side"] = {
        "description": "Rows of one conversation are on one side: the hash is of the conversation's id.",
        "kind": "split", "rows": conv, "inputs": ["q"], "share": 0.4, "salt": "",
        "expect": {"splits": [m.split_of(r, ["q"], 0.4, "") for r in conv]}}
    out["09-an-unknown-split"] = {"description": "A split that is neither train nor held_out is refused.",
                                  "kind": "split", "rows": [{"q": "a", "split": "test"}], "inputs": ["q"],
                                  "share": 0.2, "salt": "", "expect": {"refuses": "case-split"}}
    return out


# ------------------------------------------------------------------ draw


def draw_cases():
    ids = [cid(i) for i in range(1, 31)]
    return {
        "01-five-of-thirty": {"description": "Five ids of thirty, by the hash of seed and id.", "kind": "draw",
                              "ids": ids, "n": 5, "seed": "draw-1", "expect": {"ids": m.draw(ids, 5, "draw-1")}},
        "02-another-seed": {"description": "Another seed, another draw.", "kind": "draw", "ids": ids, "n": 5,
                            "seed": "draw-2", "expect": {"ids": m.draw(ids, 5, "draw-2")}},
        "03-more-than-there-are": {"description": "Asking for more than there are gives all, ranked; an id given "
                                                  "twice counts once.", "kind": "draw", "ids": ids[:4] + ids[:2],
                                   "n": 10, "seed": "s", "expect": {"ids": m.draw(ids[:4] + ids[:2], 10, "s")}},
        "04-a-larger-population": {"description": "Drawing from more ids keeps the earlier picks that still rank "
                                                  "(stable).", "kind": "draw", "ids": ids + [cid(i) for i in range(31, 61)],
                                   "n": 5, "seed": "draw-1",
                                   "expect": {"ids": m.draw(ids + [cid(i) for i in range(31, 61)], 5, "draw-1")}},
    }


# ------------------------------------------------------------------ ratings


def counted(records, criterion):
    return [{"id": j["id"], "calls": j["calls"], "item": j["item"], "value": j["value"],
             "source": list(j["source"])} for j in m.counting(records, criterion)]


def ratings_cases():
    c = [call(1, 1, SITE[0], ins("Slow search", "Hidden menu")), call(2, 2, SITE[0], ins("Hidden menu"), setup=S2)]
    sets = {
        "01-format-1-is-correct": ("A format-1 rating is a verdict under the criterion `correct`, about the whole call.",
                                   [r1(1, 1, 5, "megan", "right"), r1(2, 2, 6, None, "wrong", blind=True)], CORRECT),
        "02-a-persons-latest": ("A person's later rating of the same item replaces the earlier one; a null value "
                                "withdraws it.",
                                [r2(1, 1, 5, by="megan", item="/insights/0", value="no", **LABEL),
                                 r2(2, 1, 6, by="megan", item="/insights/0", value="yes", **LABEL),
                                 r2(3, 1, 5, by="megan", item="/insights/1", value="yes", **LABEL),
                                 r2(4, 1, 7, by="megan", item="/insights/1", value=None, **LABEL)], REAL),
        "03-items-are-targets": ("Two items of one answer are two targets; two people on one item both count.",
                                 [r2(1, 1, 5, by="megan", item="/insights/0", value="yes", **LABEL),
                                  r2(2, 1, 6, by="ana", item="/insights/0", value="no", **LABEL),
                                  r2(3, 1, 7, by="ana", item="/insights/1", value="unsure", **LABEL)], REAL),
        "04-accounts-count-on-their-own": ("Ratings under an account each count; a null one withdraws nothing.",
                                           [r2(1, 1, 5, account="lab", item="/insights/0", value="yes", **LABEL),
                                            r2(2, 1, 6, account="lab", item="/insights/0", value="no", **LABEL),
                                            r2(3, 1, 7, account="lab", item="/insights/0", value=None, **LABEL)], REAL),
        "05-a-judge-per-order": ("A judge's picks in both orders both count; a person's second pick of the same pair "
                                 "replaces their first, whichever side each answer was on.",
                                 [r2(1, (1, 2), 5, criterion="preferred", kind="choice", value=cid(2), judge=PROF),
                                  r2(2, (2, 1), 6, criterion="preferred", kind="choice", value=cid(2), judge=PROF),
                                  r2(3, (1, 2), 7, criterion="preferred", kind="choice", value=cid(1), by="megan",
                                     blind=True),
                                  r2(4, (2, 1), 8, criterion="preferred", kind="choice", value="tie", by="megan",
                                     blind=True)], PREF),
        "06-two-judges": ("Two setups of one judge are two judges; a judge's later rating replaces its earlier one.",
                          [r2(1, 1, 5, judge=PROF, item="/insights/0", value="no", **LABEL),
                           r2(2, 1, 6, judge=PROF, item="/insights/0", value="yes", **LABEL),
                           r2(3, 1, 7, judge=PROF2, item="/insights/0", value="no", **LABEL)], REAL),
        "07-other-criteria-and-kinds": ("Only ratings of the criterion's name and kind count; a format no reader knows "
                                        "is skipped.",
                                        [r2(1, 1, 5, by="megan", item="/insights/0", value="yes", **LABEL),
                                         r2(2, 1, 6, by="megan", item="/insights/0", criterion="real_problem",
                                            kind="score", scale=[0, 10], value=7),
                                         r2(3, 1, 7, by="megan", value="right", criterion="correct", kind="verdict"),
                                         {"functai_rating": 3, "id": rid(4), "call": cid(1), "at": at(8), "by": "megan",
                                          "criterion": "real_problem", "kind": "label", "value": "yes"}], REAL),
    }
    out = {}
    for name, (text, rs, crit) in sets.items():
        out[name] = {"description": text, "kind": "ratings", "records": c + rs, "criterion": crit,
                     "expect": {"counting": counted(rs, crit)}}
    return out


# ------------------------------------------------------------------ values


def values_cases():
    k = [key(s) for s in SITE]
    calls = [
        call(1, 1, SITE[0], ins("A", "B"), case=k[0], repeat=1),
        call(2, 2, SITE[0], ins("A", "C", "D"), case=k[0], repeat=2),
        call(3, 3, SITE[1], ins("E"), case=k[1]),
        call(4, 4, SITE[2], None, case=k[2], error={"type": "StepLimit", "message": "too many steps"}),
        call(5, 5, SITE[0], ins("A"), setup=S2, case=k[0]),
        call(6, 6, SITE[1], ins(), setup=S2, case=k[1]),
        call(7, 7, SITE[3], ins("F", "G"), setup=S2),
    ]
    ratings = [
        r2(1, 1, 10, by="megan", item="/insights/0", value="yes", **LABEL),
        r2(2, 1, 10, by="megan", item="/insights/1", value="no", **LABEL),
        r2(3, 2, 11, by="megan", item="/insights/0", value="yes", **LABEL),
        r2(4, 2, 11, by="megan", item="/insights/2", value="unsure", **LABEL),
        r2(5, 3, 12, by="megan", item="/insights/0", value="yes", sample="draw-1", **LABEL),
        r2(6, 3, 12, by="ana", item="/insights/0", value="no", sample="draw-1", **LABEL),
        r2(7, 5, 13, by="megan", item="/insights/0", value="no", **LABEL),
        r2(8, 7, 14, by="megan", item="/insights/1", value="yes", sample="draw-1", **LABEL),
        r2(9, 1, 15, judge=PROF, item="/insights/0", value="yes", **LABEL),
        r2(10, 1, 15, judge=PROF, item="/insights/1", value="yes", **LABEL),
        r2(11, 5, 15, judge=PROF, item="/insights/0", value="no", **LABEL),
    ]
    recs = calls + ratings
    sum_crit = dict(REAL, aggregate="sum")
    out = {}

    def add(name, text, crit, source, drawn=False, by="setup", records=recs):
        v = m.values(records, STUDY, crit, source, by=by, drawn=drawn)
        out[name] = {"description": text, "kind": "values", "records": records, "program": STUDY, "criterion": crit,
                     "source": source, "drawn": drawn, "by": by,
                     "expect": {"results": v["results"], "cases": cases_list(v["cases"])}}

    add("01-share-of-good-items", "An item criterion: each item's mean judgement, then the share of good items "
        "among those judged ('unsure' counts for nothing); the case's value is its repeats' mean; a failed call is "
        "the worst answer (no share); a call nobody judged has no value.", REAL, "people")
    add("02-how-many-good-items", "With aggregate sum: the share times the number of items; a failed call or an "
        "empty list counts 0.", sum_crit, "people")
    add("03-people-drawn", "Only the ratings of a random draw.", REAL, "people", drawn=True)
    add("04-a-judge", "One judge's ratings.", REAL, JUDGE)
    add("05-by-version", "Grouped by version instead of setup.", REAL, "people", by="version")
    whole = [call(11, 1, {"review": "Love it."}, {"result": "happy"}, kind="ai", name="mood", case=key({"review": "Love it."})),
             call(12, 2, {"review": "Broke."}, {"result": "happy"}, kind="ai", name="mood", case=key({"review": "Broke."})),
             call(13, 3, {"review": "Meh."}, None, kind="ai", name="mood", case=key({"review": "Meh."}),
                  error={"type": "RateLimitError", "message": "slow down"}),
             call(14, 4, {"review": "Ok."}, {"result": "mixed"}, kind="ai", name="mood", case=key({"review": "Ok."})),
             r1(21, 11, 9, "megan", "right"), r1(22, 12, 9, "megan", "wrong", answer="unhappy"),
             r2(23, 14, 9, by="megan", criterion="correct", kind="verdict", value="right", blind=True)]
    v = m.values(whole, {"name": "mood"}, CORRECT, "people")
    out["06-right-or-wrong"] = {"description": "A whole-answer verdict: format 1 and format 2 alike; a failed "
                                "call counts 0 without a rating.", "kind": "values", "records": whole,
                                "program": {"name": "mood"}, "criterion": CORRECT, "source": "people",
                                "drawn": False, "by": "setup",
                                "expect": {"results": v["results"], "cases": cases_list(v["cases"])}}
    scored = [call(31, 1, {"q": "a"}, {"result": "x"}, kind="ai", name="writer"),
              call(32, 2, {"q": "b"}, {"result": "y"}, kind="ai", name="writer"),
              r2(33, 31, 5, by="megan", criterion="quality", kind="score", scale=[1, 5], value=4),
              r2(34, 31, 6, by="ana", criterion="quality", kind="score", scale=[1, 5], value=2),
              r2(35, 32, 7, judge=CHECKER, criterion="quality", kind="score", value=0.25)]
    v = m.values(scored, {"name": "writer"}, QUALITY, "people")
    out["07-scores-on-a-scale"] = {"description": "A score on its scale from 0 to 1, people averaged.",
                                   "kind": "values", "records": scored, "program": {"name": "writer"},
                                   "criterion": QUALITY, "source": "people", "drawn": False, "by": "setup",
                                   "expect": {"results": v["results"], "cases": cases_list(v["cases"])}}
    v = m.values(scored, {"name": "writer"}, QUALITY, CODE)
    out["08-a-code-judge"] = {"description": "A code judge, known by its code's hash; a score with no scale "
                              "counts as it is.", "kind": "values", "records": scored, "program": {"name": "writer"},
                              "criterion": QUALITY, "source": CODE, "drawn": False, "by": "setup",
                              "expect": {"results": v["results"], "cases": cases_list(v["cases"])}}
    return out


# ------------------------------------------------------------------ side by side


def preference_cases():
    k = [key(s) for s in SITE]
    calls = [call(1, 1, SITE[0], ins("A"), case=k[0]), call(2, 2, SITE[0], ins("B"), setup=S2, case=k[0]),
             call(3, 3, SITE[1], ins("C"), case=k[1]), call(4, 4, SITE[1], ins("D"), setup=S2, case=k[1]),
             call(5, 5, SITE[2], None, case=k[2], error={"type": "StepLimit"}),
             call(6, 6, SITE[2], ins("E"), setup=S2, case=k[2]),
             call(7, 7, SITE[3], None, case=k[3], error={"type": "StepLimit"}),
             call(8, 8, SITE[3], None, setup=S2, case=k[3], error={"type": "StepLimit"}),
             call(9, 9, SITE[4], ins("F"), case=k[4]), call(10, 10, SITE[4], ins("G"), setup=S2, case=k[4]),
             call(11, 11, SITE[4], ins("H"), setup=S2, case=k[4], repeat=2)]
    picks = [r2(1, (1, 2), 20, criterion="preferred", kind="choice", value=cid(2), judge=PROF),
             r2(2, (2, 1), 20, criterion="preferred", kind="choice", value=cid(1), judge=PROF),
             r2(3, (3, 4), 21, criterion="preferred", kind="choice", value=cid(4), judge=PROF),
             r2(4, (4, 3), 21, criterion="preferred", kind="choice", value=cid(4), judge=PROF),
             r2(5, (9, 10), 22, criterion="preferred", kind="choice", value="tie", judge=PROF),
             r2(6, (11, 9), 22, criterion="preferred", kind="choice", value=cid(11), judge=PROF),
             r2(7, (1, 2), 23, criterion="preferred", kind="choice", value=cid(2), by="megan", blind=True),
             r2(8, (3, 4), 23, criterion="preferred", kind="choice", value=cid(3), by="megan", blind=True,
                sample="draw-1")]
    recs = calls + picks
    out = {}
    for name, text, src, drawn in [
            ("01-a-judge-both-orders", "A judge's picks in both orders are averaged; a failed answer loses (both "
             "failed: a tie); every pair of results on a case counts (repeats).", JUDGE, False),
            ("02-people", "People's picks.", "people", False),
            ("03-people-drawn", "People's picks of a random draw.", "people", True)]:
        got = m.preferences(recs, STUDY, PREF, src, S1, S2, drawn=drawn)
        out[name] = {"description": text, "kind": "preferences", "records": recs, "program": STUDY, "criterion": PREF,
                     "source": src, "drawn": drawn, "by": "setup", "a": S1, "b": S2,
                     "expect": {"cases": [{"case": c, "value": v} for c, v in got.items()]}}
    return out


# ------------------------------------------------------------------ statistics


def stat_cases():
    out = {}
    paired = {
        "01-thirty-right-for-both": ("Thirty cases right for both: the difference is 0, and the range says how little "
                                     "thirty cases can show (the paired t said exactly 0).", [1] * 30, [1] * 30, None),
        "02-one-case-better": ("One case of twenty better, none worse.", [1] * 19 + [0], [1] * 20, None),
        "03-clearly-better": ("Twelve better, one worse, of forty.", [0] * 12 + [1] + [1] * 20 + [0] * 7,
                              [1] * 12 + [0] + [1] * 20 + [0] * 7, None),
        "04-clearly-worse": ("The other way.", [1] * 12 + [0] + [1] * 20 + [0] * 7,
                             [0] * 12 + [1] + [1] * 20 + [0] * 7, None),
        "05-fractions": ("Values that are not all 0 or 1 (repeats, items): the paired t interval.",
                         [0.5, 1, 0.25, 0.75, 1, 0], [0.75, 1, 0.5, 0.75, 0.5, 0.5], None),
        "06-one-case": ("One shared case: a difference, no range.", [0], [1], None),
        "07-no-cases": ("No shared case: nothing.", [], [], None),
        "08-not-worse-within-a-margin": ("With a margin of 0.1: not worse.", [1] * 150 + [0] * 50,
                                         [1] * 149 + [0] + [0] * 2 + [1] * 2 + [0] * 46, 0.1),
        "09-the-same-within-a-margin": ("Within ±0.15 both ways: the same.", [1] * 200 + [0] * 200,
                                        [1] * 199 + [0] + [1] + [0] * 199, 0.15),
        "10-not-better-within-a-margin": ("Below +0.1 but maybe worse than −0.1: not better.",
                                          [1] * 30 + [0] * 10, [1] * 27 + [0] * 3 + [0] * 10, 0.1),
        "11-zero-margin": ("A margin of 0 is no margin.", [1] * 10, [1] * 10, 0),
    }
    for name, (text, a, b, margin) in paired.items():
        out["paired-" + name] = {"description": text, "kind": "paired", "a": a, "b": b, "margin": margin,
                                 "bounded": False, "expect": m.paired(a, b, margin)}
    a, b = [0.5, 1, 0.25, 1, 1, 0.75], [1, 1, 1, 1, 0.75, 1]
    out["paired-12-bounded-fractions"] = {"description": "Fractions of a bounded criterion: the t range kept within "
                                          "-1 and 1.", "kind": "paired", "a": a, "b": b, "margin": None,
                                          "bounded": True, "expect": m.paired(a, b, None, True)}
    wins = {
        "01-all-ties": ("Every pick a tie: one half, and a range that is not zero.", [0.5] * 12, None),
        "02-wins-and-losses": ("Nine wins, three losses.", [1] * 9 + [0] * 3, None),
        "03-averaged-picks": ("Shares from several picks per case.", [1, 0.75, 0.5, 1, 0.25, 1, 1, 0.5], None),
        "04-one-case": ("One case: a rate, no range.", [1], None),
        "05-a-margin": ("With a margin of 0.1 around one half.", [1] * 40 + [0] * 38 + [0.5] * 22, 0.1),
    }
    for name, (text, v, margin) in wins.items():
        out["winrate-" + name] = {"description": text, "kind": "winrate", "values": v, "margin": margin,
                                  "expect": m.win_rate(v, margin)}
    frag = {"01-largest-first": [1, 1, 1, 0, 0, -1], "02-negative": [-1, -1, 0, 1], "03-zero": [1, -1],
            "04-one-big": [0.9, -0.1, -0.1, -0.1], "05-empty": []}
    for name, d in frag.items():
        out["fragility-" + name] = {"description": "The fewest values whose removal takes the mean to 0 or across "
                                    "it.", "kind": "fragility", "values": d, "expect": {"fragility": m.fragility(d)}}
    keys = [key(s) for s in SITE] + [key({"url": f"https://more{i}.example"}) for i in range(30)]
    truth = [(i * 7) % 10 / 10 for i in range(len(keys))]
    judge = {k: min(1.0, max(0.0, t + (0.1 if i % 3 == 0 else -0.1 if i % 3 == 1 else 0))) for i, (k, t) in
             enumerate(zip(keys, truth))}
    people = {k: t for i, (k, t) in enumerate(zip(keys, truth)) if i % 5 == 0}
    ppis = {
        "01-a-good-judge": ("A judge close to people on 8 drawn cases, and 32 more cases only it judged.", judge, people),
        "02-a-useless-judge": ("A judge that says the same of everything: λ is 0, people alone (with the "
                               "pseudo-observations).", {k: 0.5 for k in keys}, people),
        "03-too-few-people": ("Fewer than three cases people judged: nothing.", judge, dict(list(people.items())[:2])),
        "04-nothing-unlabeled": ("People judged every case the judge did: people alone.",
                                 {k: judge[k] for k in people}, people),
    }
    for name, (text, j, p) in ppis.items():
        out["ppi-" + name] = {"description": text, "kind": "ppi", "judge": j, "people": p, "bounded": True,
                              "expect": m.ppi(j, p)}
    counts_j = {k: round(v * 6) for k, v in judge.items()}
    counts_p = {k: round(v * 6) + (1 if i % 2 else 0) for i, (k, v) in enumerate(people.items())}
    out["ppi-06-counts"] = {"description": "Counts (how many good items): no bounds, pseudo-observations half the "
                            "range of the values seen.", "kind": "ppi", "judge": counts_j, "people": counts_p,
                            "bounded": False, "expect": m.ppi(counts_j, counts_p, False)}
    ja = {k: v for k, v in judge.items()}
    jb = {k: min(1.0, v + 0.2) for k, v in judge.items()}
    pa = people
    pb = {k: min(1.0, v + 0.3) for k, v in people.items()}
    out["ppi-05-paired"] = {"description": "B against A, both by the judge on every case and by people on a drawn "
                            "few: PPI++ on the differences.", "kind": "ppi-paired", "judge_a": ja, "judge_b": jb,
                            "people_a": pa, "people_b": pb, "margin": None, "bounded": True,
                            "expect": m.ppi_paired(ja, jb, pa, pb)}
    agree = {
        "01-right-or-wrong": ("0 or 1 both sides: agreement and kappa.",
                              dict(zip("abcdefghij", [1, 1, 0, 1, 0, 1, 1, 0, 1, 1])),
                              dict(zip("abcdefghij", [1, 1, 0, 0, 0, 1, 1, 1, 1, 1])), False),
        "02-scores": ("Scores: mean absolute difference and correlation.",
                      dict(zip("abcde", [0.2, 0.4, 0.6, 0.8, 1.0])), dict(zip("abcde", [0.1, 0.5, 0.5, 0.9, 0.8])), False),
        "03-picks": ("Side by side: the same winner (first, tie or second).",
                     dict(zip("abcdef", [1, 0, 0.5, 1, 0.75, 0])), dict(zip("abcdef", [1, 0, 1, 0.5, 1, 0.25])), True),
        "04-nothing-shared": ("No target both rated.", {"a": 1}, {"b": 1}, False),
        "05-everyone-says-yes": ("Both always 1: agreement 1, kappa undefined.", dict(zip("abc", [1, 1, 1])),
                                 dict(zip("abc", [1, 1, 1])), False),
    }
    for name, (text, a, b, choice) in agree.items():
        exp = m.agreement(a, b, choice)
        out["agreement-" + name] = {"description": text, "kind": "agreement", "a": a, "b": b, "choice": choice,
                                    "expect": exp}
    selfs = {
        "01-repeats": ("Cases with two or more results: pairwise agreement and the spread within cases; a case with "
                       "one result says nothing.", {"k1": [1, 1, 1], "k2": [1, 0, 1], "k3": [0, 0], "k4": [1]}),
        "02-fractions": ("Fractions: one minus the mean difference.", {"k1": [0.5, 0.75], "k2": [1, 0.5, 0.25]}),
        "03-nothing-repeated": ("No case repeated.", {"k1": [1], "k2": [0]}),
    }
    for name, (text, pc) in selfs.items():
        out["self-" + name] = {"description": text, "kind": "self", "per_case": pc, "expect": m.self_agreement(pc)}
    versions = ["v1", "v2", "v3", "v4"]
    oc = []
    for i in range(12):
        oc.append((f"case-{i:02d}", "v1", "v2", 0.0 if i % 4 else 1.0))
        oc.append((f"case-{i:02d}", "v2", "v3", 1.0 if i % 3 else 0.5))
        if i % 2 == 0:
            oc.append((f"case-{i:02d}", "v1", "v3", 0.5))
    ranks = {
        "01-three-versions": ("Three versions, side by side on twelve cases; v4 never met another: no strength.",
                              versions, oc),
        "02-one-never-lost": ("A version that never lost stays finite (the virtual tie).", ["a", "b"],
                              [(f"c{i}", "a", "b", 1.0) for i in range(5)]),
        "03-a-thin-lead": ("A lead resting on one case.", ["a", "b", "c"],
                           [("c1", "a", "b", 1.0), ("c2", "a", "b", 0.0), ("c3", "a", "c", 1.0), ("c4", "b", "c", 0.5),
                            ("c5", "a", "b", 1.0)]),
        "04-nothing": ("No comparisons.", ["a", "b"], []),
    }
    for name, (text, vs, o) in ranks.items():
        out["rank-" + name] = {"description": text, "kind": "rank", "versions": vs,
                               "outcomes": [{"case": c, "first": i, "second": j, "first_share": s} for c, i, j, s in o],
                               "expect": m.rank(vs, o)}
    return out


# ------------------------------------------------------------------ positions, scoreboard, compare, disagreements


def board_records():
    k = [key(s) for s in SITE]
    recs = []
    n = 0
    lists = {S1: [["A", "B"], ["C"], ["D", "E", "F"], ["G"], ["H", "I"], ["J"], ["K", "L"], ["M"]],
             S2: [["A"], ["C", "N"], ["D"], ["G", "O"], ["H"], ["J", "P"], ["K"], ["M", "Q"]]}
    for s, lsts in lists.items():
        for i, titles in enumerate(lsts):
            n += 1
            root = n
            recs.append(call(n, n % 50, SITE[i], ins(*titles), setup=s, case=k[i], seconds=10.0 + i,
                             spent=0.05))
            n += 1
            recs.append(call(n, n % 50, {"site": SITE[i]["url"]}, {"result": "notes"}, name="walker", kind="ai",
                             parent=root, model="claude-haiku-5-5", usage={"input_tokens": 1000 + i, "output_tokens": 50},
                             setup=S3))
    if True:
        n += 1
        recs.append(call(n, 40, SITE[8], None, case=k[8], error={"type": "StepLimit", "message": "100 steps"}))
    calls = {c["id"]: c for c in recs if c["program"]["name"] == "study"}
    ratings = []
    rn = 0
    for c in calls.values():
        if c.get("error"):
            continue
        idx = list(calls).index(c["id"])
        for i, it in enumerate(c["outputs"]["insights"]):
            good = (ord(it["title"]) + i) % 3 != 0
            rn += 1
            ratings.append(r2(rn, int(c["id"][9:13], 16), 45, judge=PROF, item=f"/insights/{i}",
                              value="yes" if good else "no", **LABEL))
            if idx % 2 == 0:
                rn += 1
                agree = (ord(it["title"]) % 4) != 0
                drawn = {"sample": "draw-1"} if idx % 8 != 2 else {}
                ratings.append(r2(rn, int(c["id"][9:13], 16), 46, by="megan", item=f"/insights/{i}",
                                  value=("yes" if good else "no") if agree else ("no" if good else "yes"),
                                  blind=True, **drawn, **LABEL))
    return recs + ratings


def board_cases():
    recs = board_records()
    out = {}
    got = m.scoreboard(recs, STUDY, [REAL, dict(REAL, aggregate="sum")])
    out["scoreboard-01-a-study"] = {
        "description": "Per setup: results, cases, failures, seconds, tokens of the results' trees (a pipeline's "
                       "steps), what code spent; then each criterion by people, people drawn, each judge, and the "
                       "judge combined with people drawn.", "kind": "scoreboard", "records": recs, "program": STUDY,
        "criteria": [REAL, dict(REAL, aggregate="sum")], "by": "setup", "expect": got}
    for name, text, src, drawn, margin in [
            ("01-by-a-judge", "Two setups compared on their shared cases by a judge; cases only one has are counted "
             "apart.", JUDGE, False, None),
            ("02-by-people", "The same by people's ratings, with a margin.", "people", False, 0.1)]:
        out["compare-" + name] = {"description": text, "kind": "compare", "records": recs, "program": STUDY,
                                  "criterion": REAL, "source": src, "drawn": drawn, "by": "setup", "a": S1, "b": S2,
                                  "margin": margin,
                                  "expect": m.compare_groups(recs, STUDY, REAL, src, S1, S2, drawn=drawn, margin=margin)}
    for name, text, src in [("01-by-a-judge", "Cases on which the setups differ, the largest spread first.", JUDGE)]:
        out["disagreements-" + name] = {"description": text, "kind": "disagreements", "records": recs,
                                        "program": STUDY, "criterion": REAL, "source": src, "by": "setup",
                                        "expect": {"cases": m.disagreements(recs, STUDY, REAL, src)}}
    k = [key(s) for s in SITE]
    rep = [call(1, 1, SITE[0], ins("A"), case=k[0], repeat=1), call(2, 2, SITE[0], ins("A", "B"), case=k[0], repeat=2),
           call(3, 3, SITE[1], ins("C"), case=k[1], repeat=1), call(4, 4, SITE[1], ins("C"), case=k[1], repeat=2),
           r2(1, 1, 9, judge=PROF, item="/insights/0", value="yes", **LABEL),
           r2(2, 2, 9, judge=PROF, item="/insights/0", value="yes", **LABEL),
           r2(3, 2, 9, judge=PROF, item="/insights/1", value="no", **LABEL),
           r2(4, 3, 9, judge=PROF, item="/insights/0", value="yes", **LABEL),
           r2(5, 4, 9, judge=PROF, item="/insights/0", value="yes", **LABEL)]
    out["disagreements-02-repeats"] = {"description": "A case whose repeats differ.", "kind": "disagreements",
                                       "records": rep, "program": STUDY, "criterion": REAL, "source": JUDGE,
                                       "by": "setup", "expect": {"cases": m.disagreements(rep, STUDY, REAL, JUDGE)}}
    pos = [call(1, 1, SITE[0], ins("A"), case=k[0]), call(2, 2, SITE[0], ins("B"), setup=S2, case=k[0]),
           call(3, 3, SITE[1], ins("C"), case=k[1]), call(4, 4, SITE[1], ins("D"), setup=S2, case=k[1]),
           call(5, 5, SITE[2], ins("E"), case=k[2]), call(6, 6, SITE[2], ins("F"), setup=S2, case=k[2]),
           r2(1, (1, 2), 9, criterion="preferred", kind="choice", value=cid(1), judge=PROF),
           r2(2, (2, 1), 9, criterion="preferred", kind="choice", value=cid(2), judge=PROF),
           r2(3, (3, 4), 9, criterion="preferred", kind="choice", value=cid(4), judge=PROF),
           r2(4, (4, 3), 9, criterion="preferred", kind="choice", value=cid(4), judge=PROF),
           r2(5, (5, 6), 9, criterion="preferred", kind="choice", value="tie", judge=PROF),
           r2(6, (6, 5), 9, criterion="preferred", kind="choice", value="tie", judge=PROF)]
    out["positions-01-both-orders"] = {"description": "A judge shown each pair both ways: how often it picked the "
                                       "same answer both times, and how often the first shown won.",
                                       "kind": "positions", "records": pos, "criterion": PREF, "judge": JUDGE,
                                       "expect": m.positions(pos, PREF, JUDGE)}
    return out


# ------------------------------------------------------------------ rated with format 2


def rated_cases():
    import rated as r
    calls = [r.call(1, 1, r.M1, {"result": "billing"}, fmt=2), r.call(2, 2, r.M2, {"result": "billing"}, fmt=2),
             r.call(3, 3, r.M3, {"result": "product"}, fmt=2)]
    recs = calls + [
        {"functai_rating": 2, "id": r.rid(1), "call": r.cid(1), "at": r.at(5), "criterion": "correct",
         "kind": "verdict", "value": "right", "by": "maxime", "blind": True},
        {"functai_rating": 2, "id": r.rid(2), "call": r.cid(2), "at": r.at(5), "criterion": "correct",
         "kind": "verdict", "value": "wrong", "answer": "shipping", "by": "maxime"},
        {"functai_rating": 2, "id": r.rid(3), "call": r.cid(3), "at": r.at(5), "criterion": "correct",
         "kind": "verdict", "value": "right", "judge": {"name": "checker", "setup": JUDGE}},
        {"functai_rating": 2, "id": r.rid(4), "call": r.cid(3), "at": r.at(6), "criterion": "polite",
         "kind": "label", "answers": {"yes": 1, "no": 0}, "value": "yes", "by": "maxime"},
        r.rating(5, 3, 7, "ana", "right", blind=True),
    ]
    rows = [r.row(r.M1, {"result": "billing"}, 1, rating_="right", by="maxime"),
            r.row(r.M2, {"result": "shipping"}, 2, rating_="wrong", by="maxime"),
            r.row(r.M3, {"result": "product"}, 3, rating_="right", by="ana")]
    return {"rated-01-format-2-verdicts": {
        "description": "rated reads a person's or an account's format-2 verdict on a whole call under `correct` as a "
                       "format-1 rating; a judge's verdict and other criteria make no known answer; format 1 with "
                       "blind is read as before.",
        "kind": "rated", "records": recs, "rated": r.TEAM, "expect": {"rows": rows, "left_out": r.LEFT}}}


def cases() -> dict:
    out = {}
    for prefix, fn in [("key-", key_cases), ("split-", split_cases), ("draw-", draw_cases),
                       ("ratings-", ratings_cases), ("values-", values_cases), ("preferences-", preference_cases)]:
        for name, case in fn().items():
            out[prefix + name] = case
    out.update(stat_cases())
    out.update(board_cases())
    out.update(rated_cases())
    return out


def records_of(case: dict) -> list:
    """The records a case holds, to check against the schemas."""
    if "records" in case:
        return case["records"]
    if "record" in case:
        return [case["record"]]
    return []
