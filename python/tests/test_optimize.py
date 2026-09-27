"""Evaluation and optimization, offline: the fake model answers by rules."""

import re

import pytest

import functai
from functai import (BootstrapFewShotWithRandomSearch, InstructionSearch, LabeledFewShot, _ai, ai, evaluate,
                     module)

TRAIN = [
    {"user_query": "I need to reserve a room.", "result": "booking"},
    {"user_query": "How do I get there?", "result": "information"},
    {"user_query": "Cancel my reservation.", "result": "cancelation"},
    {"user_query": "Book me a suite.", "result": "booking"},
]


def _last_user(req):
    return "".join(p.text for p in req.messages[-1].parts)


def _label(query):
    q = query.lower()
    return "booking" if ("reserve" in q or "book" in q) else "cancelation" if "cancel" in q else "information"


def make_classifier():
    @ai
    def classify_intent(user_query: str) -> str:
        """Classify user intent as 'booking', 'cancelation', or 'information'."""
        return _ai
    return classify_intent


def query_of(req):
    m = re.search(r"<user_query>\n(.*?)\n</user_query>", _last_user(req), re.S)
    return m.group(1) if m else ""


def test_bootstrap_keeps_the_runs_the_metric_accepts(fake):
    f = make_classifier()
    # the model gets "How do I get there?" wrong; the others right
    r = fake(responder=lambda req: "<result>\n" + ("booking" if "get there" in query_of(req)
                                                   else _label(query_of(req))) + "\n</result>")
    original = f
    f = f.opt(TRAIN)
    demos = f.demos
    boot = [d for d in demos if not isinstance(d, dict)]
    labeled = [d for d in demos if isinstance(d, dict)]
    assert len(boot) == 3 and len(labeled) == 1          # the wrong run was not kept; its label fills in
    assert labeled[0]["outputs"] == {"result": "information"}
    n = len(r.requests)
    f("Can I book for Tuesday?")
    req = r.requests[n]
    assert [m.role for m in req.messages] == ["user", "assistant"] * 4 + ["user"]
    assert original.demos == []                          # an improved copy; the function given is unchanged
    assert original.version != f.version
    assert [r["optimizer"] for r in f.optimization_runs()] == ["BootstrapFewShot"]
    assert original.optimization_runs() == []


def test_labeled_few_shot(fake):
    f = make_classifier()
    fake()
    f = f.opt(TRAIN, optimizer=LabeledFewShot, k=2)
    assert len(f.demos) == 2 and all(isinstance(d, dict) for d in f.demos)


def test_training_data_can_be_a_table(fake, tmp_path):
    import dpyr
    f = make_classifier()
    fake()
    dpyr.read(TRAIN).write(str(tmp_path / "train.parquet"))
    f = f.opt(str(tmp_path / "train.parquet"), optimizer=LabeledFewShot(k=5))
    assert {d["inputs"]["user_query"] for d in f.demos} == {r["user_query"] for r in TRAIN}


def test_training_data_must_have_the_inputs(fake):
    f = make_classifier()
    fake()
    with pytest.raises(ValueError, match="is 'user_querry' a typo"):
        f.opt([{"user_querry": "a", "result": "booking"}])


def test_random_search_picks_the_best_candidate(fake):
    f = make_classifier()

    def responder(req):
        # right only when the prompt carries worked examples
        has_demos = len(req.messages) > 1
        return "<result>\n" + (_label(query_of(req)) if has_demos else "information") + "\n</result>"

    fake(responder=responder)
    opt = BootstrapFewShotWithRandomSearch(num_candidate_programs=2, max_labeled_demos=2)
    f = f.opt(TRAIN, optimizer=opt)
    best = max(opt.candidates, key=lambda c: c["score"])
    assert f.demos and best["candidate"] != "zero-shot"
    assert evaluate(f, TRAIN, functai.exact_match).score == 1.0


def test_instruction_search_finds_the_instruction_that_works(fake):
    f = make_classifier()
    proposals = iter(["Be vague.", "Use the labels exactly: booking, cancelation, information.", "Guess."])

    def responder(req):
        if "Propose a new instruction" in (req.system or ""):
            return "<result>\n" + next(proposals, "Other.") + "\n</result>"
        good = "Use the labels exactly" in (req.system or "")
        return "<result>\n" + (_label(query_of(req)) if good else "information") + "\n</result>"

    fake(responder=responder)
    opt = InstructionSearch(num_candidates=4, num_trials=10, max_bootstrapped_demos=0, max_labeled_demos=0)
    better = f.opt(TRAIN, optimizer=opt)
    assert "Use the labels exactly" in better.instructions
    assert better.demos == []
    assert "Use the labels exactly" not in f.instructions
    assert better.trials and better.trials == opt.trials


def test_a_teacher_model_bootstraps_the_demos(fake):
    f = make_classifier()

    def responder(req):
        smart = req.model == "gpt-4.1"
        return "<result>\n" + (_label(query_of(req)) if smart else "information") + "\n</result>"

    r = fake(responder=responder)
    f = f.opt(TRAIN, teacher_lm="gpt-4.1", max_labeled_demos=4)
    assert len([d for d in f.demos if not isinstance(d, dict)]) == 4
    assert {req.model for req in r.requests} == {"gpt-4.1"}
    f("book it")
    assert r.requests[-1].model == "gpt-4.1-mini"


def test_synthesized_examples_from_a_teacher(fake):
    f = make_classifier()

    def responder(req):
        if "diverse, realistic examples" in (req.system or ""):
            return '<result>\n[{"user_query": "reserve a table", "result": "booking"}, ' \
                   '{"user_query": "where is it", "result": "information"}]\n</result>'
        return "<result>\n" + _label(query_of(req)) + "\n</result>"

    fake(responder=responder)
    f = f.opt(n_synth=2, teacher_lm="gpt-4.1", optimizer=LabeledFewShot)
    assert sorted(d["inputs"]["user_query"] for d in f.demos) == ["reserve a table", "where is it"]


def test_dspy_optimizers_are_refused_with_a_way_forward(fake):
    class MIPROv2:
        def __init__(self, **kw): ...
    MIPROv2.__module__ = "dspy.teleprompt.mipro_optimizer_v2"
    f = make_classifier()
    fake()
    with pytest.raises(TypeError, match="InstructionSearch"):
        f.opt(TRAIN, optimizer=MIPROv2)


def test_save_and_load(fake, tmp_path):
    f = make_classifier()
    fake(responder=lambda req: "<result>\n" + _label(query_of(req)) + "\n</result>")
    f = f.opt(TRAIN, max_bootstrapped_demos=2, max_labeled_demos=3)
    f.instructions = "Classify carefully."
    path = tmp_path / "classify.json"
    f.save(path)
    g = make_classifier().load(path)
    assert g.instructions == "Classify carefully."
    assert len(g.demos) == 3
    req = g.render("book a room")
    assert [m.role for m in req.messages] == ["user", "assistant"] * 3 + ["user"]


# ------------------------------------------------------------------ modules


@ai
def generate_query(claim: str, key_facts: list[str]) -> str:
    """Produce a search query from a claim and the facts so far."""


@ai
def append_notes(claim: str, key_facts: list[str], new_docs: list[str]) -> list[str]:
    """Extend the key facts with what the new documents say."""


@module
def research_hop(claim: str, hops: int = 2):
    key_facts: list[str] = []
    for i in range(hops):
        query = generate_query(claim, key_facts)
        key_facts = append_notes(claim, key_facts, [f"doc about: {query}"])
    return key_facts


def test_a_module_is_optimized_as_one_program(fake):
    def responder(req):
        if "search query" in req.system:
            return "<result>\nparis tower\n</result>"
        return '<result>\n["fact: paris"]\n</result>'

    r = fake(responder=responder)
    assert set(research_hop.named_ai_functions()) == {"generate_query", "append_notes"}
    trainset = [{"claim": "The Eiffel Tower is in Paris.", "result": ["fact: paris"]}]

    def metric(row, prediction):
        return 1.0 if prediction.result else 0.0

    better = research_hop.opt(trainset, metric=metric, call_defaults=dict(hops=1))
    assert generate_query.demos == [] and append_notes.demos == []     # the functions themselves are unchanged
    assert {k: len(st.demos) for k, st in better.state().items()} == {"generate_query": 1, "append_notes": 1}
    assert better.version != research_hop.version
    n = len(r.requests)
    better("Big Ben is in London.", 1)
    assert [m.role for m in r.requests[n].messages] == ["user", "assistant", "user"]
    n = len(r.requests)
    research_hop("Big Ben is in London.", 1)                          # the original runs as it was
    assert [m.role for m in r.requests[n].messages] == ["user"]


def test_autoinstruct_is_opt_in_and_runs_at_the_first_call(fake):
    def responder(req):
        if "Write one clear, complete instruction" in (req.system or ""):
            return "<result>\nLabel the query: booking, cancelation or information.\n</result>"
        return "<result>\nbooking\n</result>"

    r = fake(responder=responder)
    plain = make_classifier()
    plain("book it")
    assert len(r.requests) == 1                     # no instruction writing by default

    @ai(autoinstruct=True)
    def classify(user_query: str) -> str:
        """Classify the intent."""

    assert len(r.requests) == 1                     # nothing at definition
    classify("book it")
    assert classify.instructions == "Label the query: booking, cancelation or information."
    assert r.requests[-1].system.startswith("Label the query")


def test_global_autorefine_does_not_recurse_into_functais_own_helpers(fake):
    r = fake(responder=lambda req: "<result>\nbooking\n</result>")
    functai.configure(instruction_autorefine_calls=1)
    f = make_classifier()
    f("book it")
    assert len(r.requests) == 2                     # the call, then one refinement
    assert f.instructions == "booking"


# ------------------------------------------------------------------ GEPA

GOOD = "Use the labels exactly: booking, cancelation, information."
MORE = TRAIN + [
    {"user_query": "Please book a table for two.", "result": "booking"},
    {"user_query": "What time is breakfast?", "result": "information"},
    {"user_query": "I want to cancel tonight.", "result": "cancelation"},
    {"user_query": "Is there parking?", "result": "information"},
]


def _reflecting(req):
    return "You improve the instruction" in (req.system or "")


def test_gepa_rewrites_the_instruction_from_its_mistakes(fake):
    from functai import GEPA
    f = make_classifier()

    def responder(req):
        if _reflecting(req):
            return f"<result>\n{GOOD}\n</result>"
        good = "Use the labels exactly" in (req.system or "")
        return "<result>\n" + (_label(query_of(req)) if good else "information") + "\n</result>"

    r = fake(responder=responder)
    opt = GEPA(budget=60, seed=1)
    f = f.opt(MORE, optimizer=opt)
    assert f.instructions == GOOD
    reflections = [q for q in r.requests if _reflecting(q)]
    shown = "".join(_last_user(q) for q in reflections)
    assert "wrong: the right answer is" in shown                       # feedback in words
    chosen = [t for t in opt.trials if t["chosen"]]
    assert len(chosen) == 1 and chosen[0]["kind"] == "reflect" and chosen[0]["score"] == 1.0
    assert opt.calls <= 60
    # the selection rows (half, drawn with the seed) are never shown to the reflection model
    import random
    rows = list(MORE)
    random.Random(1).shuffle(rows)
    selection = {q["user_query"] for q in rows[: len(rows) // 2]}
    shown_queries = {line.split(": ", 1)[1] for line in shown.splitlines() if line.strip().startswith("user_query:")}
    assert shown_queries and not (shown_queries & selection)


def test_gepa_runs_a_row_once_per_instruction(fake):
    from functai import GEPA
    f = make_classifier()
    seen = []

    def responder(req):
        if _reflecting(req):
            return "<result>\nStill vague.\n</result>"
        seen.append(((req.system or ""), query_of(req)))
        return "<result>\ninformation\n</result>"

    fake(responder=responder)
    opt = GEPA(budget=40)
    f = f.opt(MORE, optimizer=opt)
    assert len(seen) == len(set(seen)) == opt.calls
    assert f.instructions == make_classifier().instructions            # nothing beat the written one: kept


def test_gepa_drops_a_proposal_that_copies_an_input_and_says_so(fake):
    from functai import GEPA
    long_rows = [{"user_query": f"Hello there, I would like to know about option number {i} please.", "result": "information"}
                 for i in range(6)] + [{"user_query": f"Please book the room number {i} for the whole of next week.",
                                        "result": "booking"} for i in range(6)]
    f = make_classifier()
    reflections = []

    def responder(req):
        if _reflecting(req):
            reflections.append(_last_user(req))
            m = re.search(r"\n  user_query: (.*)", _last_user(req))
            return f"<result>\nIf the message says '{m.group(1)}', answer booking.\n</result>"
        return "<result>\ninformation\n</result>"

    fake(responder=responder)
    opt = GEPA(budget=40)
    f = f.opt(long_rows, optimizer=opt)
    assert any(t["note"] == "copied an input: dropped" for t in opt.trials)
    assert any("dropped: it copied an input" in text for text in reflections[1:])   # the next reflection is told
    assert f.instructions == make_classifier().instructions


def test_gepa_prefers_the_shorter_of_equal_instructions():
    from functai.optimizers import GEPA, _Candidate
    pool = [_Candidate("a long written instruction", (), "written", [1, 0]),
            _Candidate("short", (0,), "reflect", [0, 1]),
            _Candidate("both, and wordy about it", (0, 1), "combine", [1, 1])]
    assert GEPA._frontier(pool) == {2: 2}                              # dominated candidates leave the frontier
    assert GEPA._pair(pool[:2], GEPA._frontier(pool[:2])) == (0, 1)


def test_gepa_refuses_a_module(fake):
    from functai import GEPA
    f = make_classifier()

    @module
    def route(user_query: str) -> str:
        return f(user_query)

    fake("<result>\ninformation\n</result>")
    with pytest.raises(TypeError, match="one AI function"):
        route.opt(MORE, optimizer=GEPA)
