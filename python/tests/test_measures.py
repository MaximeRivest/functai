"""Measuring programs (contract/measures.md, design/20), offline: the same calls
on four shapes of program, a classifier with known answers, a study judged
item by item, a conversation, and an agent judged side by side."""

import re

import pytest

import functai
from functai import _ai, ai, evaluate, module
from functai.measures import paired

MOODS = [
    {"review": "Love it. Works exactly as described.", "result": "happy"},
    {"review": "Broke after two days.", "result": "unhappy"},
    {"review": "Great sound, but the battery is weak.", "result": "mixed"},
    {"review": "Arrived early and works.", "result": "happy"},
]


def review_of(req):
    text = "".join(getattr(p, "text", "") or "" for p in req.messages[-1].parts)
    m = re.search(r"<review>\n(.*?)\n</review>", text, re.S)
    return m.group(1) if m else ""


def mood_answer(req):
    r = review_of(req).lower()
    mood = "mixed" if "but" in r else "unhappy" if "broke" in r else "happy"
    return f"<result>\n{mood}\n</result>"


def make_mood():
    @ai
    def mood(review: str) -> str:
        """The customer's overall mood: happy, unhappy or mixed."""
        return _ai
    return mood


# ------------------------------------------------------------------ 1. known answers


def test_calls_say_their_case_and_repeat(fake):
    fake(responder=mood_answer)
    log = functai.MemoryCalls()
    with functai.configure(log_calls=log):
        ev = evaluate(make_mood(), MOODS, repeats=2)
    calls = [r for r in log.read() if "functai_call" in r]
    assert len(calls) == 8 and ev.score == 1.0
    keys = {functai.case_key(["review"], r) for r in MOODS}
    assert {c["case"] for c in calls} == keys
    assert sorted(c["repeat"] for c in calls) == [1, 1, 1, 1, 2, 2, 2, 2]
    t = ev.table.collect().to_dicts()
    assert {r["call_id"] for r in t} == {c["id"] for c in calls} and not any(r["reused"] for r in t)
    s = ev.summary.collect().to_dicts()[0]
    assert s["n"] == 4                                       # the range is over cases, not runs
    assert "× 2 repeats" in repr(ev)


def test_reuse_asks_only_what_is_missing(fake):
    router = fake(responder=mood_answer)
    log = functai.MemoryCalls()
    mood = make_mood()
    first = evaluate(mood, MOODS[:2], reuse=log)
    assert len(router.requests) == 2
    again = evaluate(mood, MOODS, repeats=2, reuse=log)       # 2 cases known once: 6 runs to ask
    assert len(router.requests) == 2 + 6
    t = again.table.collect().to_dicts()
    assert sum(r["reused"] for r in t) == 2 and "2 reused" in repr(again)
    assert again.score == first.score == 1.0
    other = mood.using(lm="gpt-4.1")                         # another setup: nothing to reuse
    evaluate(other, MOODS[:1], reuse=log)
    assert len(router.requests) == 9


def test_thirty_right_for_both_says_how_little_it_shows(fake):
    rows = [{"review": f"Love it, number {i}.", "result": "happy"} for i in range(30)]
    fake(responder=mood_answer)
    a = evaluate(make_mood(), rows)
    b = evaluate(make_mood(), rows)
    c = functai.compare(a, b, margin=0.05).collect().to_dicts()[0]
    assert c["diff"] == 0 and c["method"] == "mover-wilson"
    assert c["low"] == pytest.approx(-0.1135, abs=1e-4) and c["outcome"] == "unclear"   # the old t said [0, 0]


def test_a_failed_run_counts_zero_and_the_outcome_says_worse(fake):
    rows = [{"review": f"Love it, number {i}.", "result": "happy"} for i in range(40)]
    fake(responder=mood_answer)
    a = evaluate(make_mood(), rows)
    fake(responder=lambda req: RuntimeError("down") if "number 1" in review_of(req) else mood_answer(req))
    functai.configure(api_retries=0)
    b = evaluate(make_mood(), rows)                          # 11 of 40 fail ("number 1", "number 10" … "19")
    c = functai.compare(a, b).collect().to_dicts()[0]
    assert c["worse"] == 11 and c["outcome"] == "worse" and c["fragility"] == 11   # the 11 cases it rests on


def test_held_out_rows_never_reach_an_optimizer(fake):
    fake(responder=mood_answer)
    rows = [{**r, "split": "held_out" if i == 0 else "train"} for i, r in enumerate(MOODS)]
    with pytest.raises(ValueError, match="1 of the 4 rows of the trainset are held out"):
        functai.labeled_few_shot(make_mood(), rows)


def test_split_of_is_stable_and_keeps_a_conversation_together():
    rows = [{"q": f"question {i}"} for i in range(200)]
    sides = [functai.split_of(r, ["q"], share=0.25) for r in rows]
    assert 30 < sides.count("held_out") < 70
    more = rows + [{"q": f"question {i}"} for i in range(200, 400)]
    assert [functai.split_of(r, ["q"], share=0.25) for r in more[:200]] == sides     # adding rows moves none
    turns = [{"q": f"turn {i}", "conversation": f"c{i // 5}"} for i in range(50)]
    by_conv = {}
    for t in turns:
        by_conv.setdefault(t["conversation"], set()).add(functai.split_of(t, ["q"], share=0.3))
    assert all(len(s) == 1 for s in by_conv.values())
    with pytest.raises(functai.MeasureError, match="'train' or 'held_out'"):
        functai.split_of({"q": "x", "split": "test"}, ["q"])


def test_a_rating_format_1_can_say_is_format_1(fake):
    fake(responder=mood_answer)
    log = functai.MemoryCalls()
    with functai.configure(log_calls=log):
        p = make_mood().predict(review="Broke after two days.")
        r1 = functai.rate(p, "right", by="megan", blind=True)
        r2 = functai.rate(p, "yes", criterion=functai.Criterion("polite", "label", answers={"yes": 1, "no": 0}),
                          by="megan")
    assert r1["functai_rating"] == 1 and r1["blind"] is True
    assert r2["functai_rating"] == 2 and r2["kind"] == "label" and r2["answers"] == {"yes": 1, "no": 0}
    rows = functai.rated(make_mood(), folder=log).collect().to_dicts()
    assert len(rows) == 1 and rows[0]["result"] == "unhappy"


def test_as_case_turns_a_corrected_call_into_a_row(fake):
    fake(responder=mood_answer)
    log = functai.MemoryCalls()
    with functai.configure(log_calls=log):
        p = make_mood().predict(review="Fine I guess, but slow.")
        functai.rate(p, "wrong", answer="unhappy", by="megan")
        row = functai.as_case(p, folder=log)
    assert row["review"] == "Fine I guess, but slow." and row["result"] == "unhappy"
    assert row["split"] == functai.split_of({"review": row["review"]}, ["review"])


# ------------------------------------------------------------------ 2. a study judged item by item

SITES = [{"url": f"https://site{i}.example"} for i in range(6)]
REAL = functai.Criterion("real_problem", "label", items="/result", answers={"yes": 1, "no": 0, "unsure": None},
                         question="Is this a real problem?")


def problem_answer(req):
    text = "".join(getattr(p, "text", "") or "" for p in req.messages[-1].parts)
    n = int(re.search(r"site(\d)", text).group(1))
    return f"<result>\n{'Slow search' if n % 2 else 'Hidden menu'} on site {n}\n</result>"


@ai
def find_problem(url: str) -> str:
    """One usability problem a visitor would meet on this site."""
    return _ai


def make_study(k=2):
    @module
    def study(url: str) -> list:
        return [{"title": find_problem(url=url)} for _ in range(k)]
    return study


def professor(item, url):
    """A code judge: a problem that mentions search is real."""
    return "yes" if "search" in item["title"].lower() else "no"


def test_a_study_on_the_scoreboard(fake):
    fake(responder=problem_answer)
    log = functai.MemoryCalls()
    study = make_study()
    with functai.configure(log_calls=log):
        ev = evaluate(study, SITES, criteria=[REAL.with_judge(professor)])
    assert ev.metrics == ["real_problem"]
    assert ev.score == pytest.approx(0.5)                    # odd sites' problems are real
    judged = [r for r in log.read() if r.get("functai_rating") == 2]
    assert len(judged) == 12 and all(r["judge"]["name"] == "professor" for r in judged)
    calls = {r["id"]: r for r in log.read() if r.get("functai_call") and r["program"]["name"] == "study"}
    ids = list(calls)
    drawn = functai.draw(ids, 4, seed="draw-1")
    for cid in drawn:                                        # Megan judges a random draw, blind
        for i, item in enumerate(calls[cid]["outputs"]["result"]):
            functai.rate(cid, "yes" if "search" in item["title"].lower() else "no", criterion=REAL, item=i,
                         by="megan", sample="draw-1", blind=True, folder=log)
    board = functai.scoreboard(study, [REAL], records=log)
    sources = [m["source"] for m in board.groups[0]["measures"]]
    assert sources == ["people", "people drawn", "judge", "combined"]
    assert board.groups[0]["input_tokens"] == 6 * 2 * 3      # the steps' tokens count
    agree = functai.agreement(study, REAL, "people", professor, records=log)
    assert agree["agree"] == 1.0 and agree["n"] == 8
    assert "real_problem [mean] by combined (professor)" in repr(board)


def test_a_judges_ratings_are_reused(fake):
    fake(responder=problem_answer)
    log = functai.MemoryCalls()
    asked = []

    def counting_judge(item):
        asked.append(item["title"])
        return "yes"

    crit = REAL.with_judge(counting_judge)
    evaluate(make_study(), SITES[:2], criteria=[crit], reuse=log)
    evaluate(make_study(), SITES[:2], criteria=[crit], reuse=log)
    assert len(asked) == 4                                   # the second run reused calls and ratings


def test_two_setups_compared_by_a_judge(fake):
    fake(responder=problem_answer)
    log = functai.MemoryCalls()
    @module
    def study(url: str) -> list:
        return [{"title": find_problem(url=url)}]
    one = study

    @module
    def study(url: str) -> list:                             # noqa: F811 — the next version: three problems
        return [{"title": find_problem(url=url)} for _ in range(3)]
    three = study
    assert one.setup != three.setup
    evaluate(one, SITES, criteria=[REAL.with_judge(professor)], reuse=log)
    evaluate(three, SITES, criteria=[REAL.with_judge(professor)], reuse=log)
    many = functai.Criterion("real_problem", "label", items="/result", aggregate="sum",
                             answers={"yes": 1, "no": 0, "unsure": None})
    c = functai.measures.compare_of(log.read(), {"name": "study"}, many.data(), functai.measures.source_id(professor),
                                    one.setup, three.setup)
    assert c["n"] == 6 and c["diff"] == pytest.approx(1.0) and c["method"] == "paired-t"


# ------------------------------------------------------------------ 3. a conversation's turns


def test_a_turn_with_earlier_turns_is_its_own_case():
    first = {"question": "Weather today?"}
    later = {"question": "And tomorrow?", "earlier": [{"inputs": first, "outputs": {"result": "Sunny."}}],
             "conversation": "c1"}
    alone = {"question": "And tomorrow?"}
    assert functai.case_key(["question"], later) != functai.case_key(["question"], alone)
    assert functai.case_key(["question"], {**first, "earlier": []}) == functai.case_key(["question"], first)


# ------------------------------------------------------------------ 4. side by side, ranked


@functai.tool(effects="reads")
def lookup(what: str) -> str:
    """Look something up."""
    return f"found {what}"


def make_agent(style):
    @ai(tools=[lookup])
    def agent(task: str) -> str:
        """Do the task."""
        return _ai
    return agent.using(lm=f"gpt-4.1-{style}")


def test_agents_with_tools_judged_side_by_side_and_ranked(fake):
    import lm15

    def reply(req):
        parts = [p for msg in req.messages for p in msg.parts]
        if not any(type(p).__name__ == "ToolResultPart" for p in parts):
            return [lm15.ToolCallPart(id="c1", name="lookup", input={"what": "the task"})]
        if "fail" in "".join(getattr(p, "text", "") or "" for p in parts) and req.model.endswith("bad"):
            return RuntimeError("the model gave up")
        return f"<result>\ndone by {req.model}\n</result>"
    fake(responder=reply)
    functai.configure(api_retries=0)
    log = functai.MemoryCalls()
    tasks = [{"task": f"task {i}"} for i in range(5)] + [{"task": "fail please"}]
    agents = {s: make_agent(s) for s in ("good", "mid", "bad")}
    for a in agents.values():
        evaluate(a, tasks, metric=lambda row, pred: 1.0, reuse=log)
    calls = [r for r in log.read() if r.get("functai_call")]
    assert all(any(st["kind"] == "tool" for st in c["steps"]) for c in calls if c["error"] is None)
    by = {}
    for c in calls:
        by.setdefault((c["program"]["setup"], c["case"]), c)

    def picker(first, second):
        """Prefers the good agent, then mid; position-biased on ties."""
        rank = {"good": 2, "mid": 1, "bad": 0}
        a, b = (rank[x.rsplit("-", 1)[1]] for x in (first, second))
        return "first" if a > b else "second" if b > a else "first"

    keys = sorted({c["case"] for c in calls})
    pairs = []
    for k in keys:
        g, m_, b = (by[(agents[s].setup, k)] for s in ("good", "mid", "bad"))
        pairs += [(g, m_), (m_, b), (g, b)]
    pairs = [(x, y) for x, y in pairs if x.get("error") is None and y.get("error") is None]
    written = functai.judge_pairs(picker, pairs, records=log)
    assert len(written) == 2 * len(pairs)
    pos = functai.positions(picker, records=log)
    assert pos["consistent"] == 1.0
    order = functai.rank("agent", records=log, source=picker)
    assert order["order"] == [agents["good"].setup, agents["mid"].setup, agents["bad"].setup]
    wr = functai.win_rate("agent", agents["bad"].setup, agents["good"].setup, source=picker, records=log)
    assert wr["rate"] == 1.0 and wr["n"] == 6                 # the bad agent's failure loses without a judge


def test_prefer_records_a_blind_pick(fake):
    fake(responder=mood_answer)
    log = functai.MemoryCalls()
    with functai.configure(log_calls=log):
        a = make_mood().predict(review="Love it.")
        b = make_mood().using(lm="gpt-4.1").predict(review="Love it.")
    first, second = functai.shown_order(a, b, seed="queue-1")
    r = functai.prefer(first, second, winner=second, by="megan", blind=True, folder=log)
    assert r["calls"] == [first, second] and r["value"] == second and r["kind"] == "choice"
    assert functai.shown_order(b, a, seed="queue-1") == (first, second)


def test_paired_t_range_kept_within_its_bounds():
    got = paired([0.5, 0.75], [1.0, 1.0], None, True)
    assert got["high"] <= 1.0 and got["method"] == "paired-t"


# ------------------------------------------------------------------ what is written passes the contract's schemas


def test_every_record_written_passes_the_schemas(fake):
    from contract_support import assert_valid, validator
    call_v, rating_v = validator("call"), validator("rating")
    fake(responder=problem_answer)
    log = functai.MemoryCalls()
    study = make_study()
    with functai.configure(log_calls=log):
        evaluate(study, SITES[:3], criteria=[REAL.with_judge(professor)], repeats=2)
    calls = [r for r in log.read() if r.get("functai_call") and r["program"]["name"] == "study"]
    functai.rate(calls[0]["id"], "unsure", criterion=REAL, item=0, by="megan", sample="d", blind=True, folder=log)
    functai.rate(calls[0]["id"], "right", by="megan", blind=True, folder=log)
    functai.prefer(calls[0]["id"], calls[1]["id"], "tie", by="megan", blind=True, folder=log)
    functai.judge_pairs(lambda first, second: "second", [(calls[0], calls[2])], records=log)
    score = functai.Criterion("clarity", "score", scale=(1, 5))
    functai.judge(lambda answer: 4, calls[:1], score, records=log)
    kinds = set()
    for rec in log.read():
        if "functai_call" in rec:
            assert_valid(call_v, rec, rec["program"]["name"])
        else:
            assert_valid(rating_v, rec, rec.get("criterion", "correct"))
            kinds.add((rec["functai_rating"], rec.get("kind")))
    assert kinds == {(1, None), (2, "label"), (2, "choice"), (2, "score")}
