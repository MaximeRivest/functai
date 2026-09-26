"""Evaluation and optimization, offline: the fake model answers by rules."""

import re

import pytest

import functai
from functai import (BootstrapFewShotWithRandomSearch, Evaluate, Example, InstructionSearch,
                     LabeledFewShot, _ai, ai, evaluate, module)

TRAIN = [
    Example(user_query="I need to reserve a room.", result="booking").with_inputs("user_query"),
    Example(user_query="How do I get there?", result="information").with_inputs("user_query"),
    Example(user_query="Cancel my reservation.", result="cancelation").with_inputs("user_query"),
    Example(user_query="Book me a suite.", result="booking").with_inputs("user_query"),
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
    f.opt(trainset=TRAIN)
    demos = f.demos
    boot = [d for d in demos if not isinstance(d, dict)]
    labeled = [d for d in demos if isinstance(d, dict)]
    assert len(boot) == 3 and len(labeled) == 1          # the wrong run was not kept; its label fills in
    assert labeled[0]["outputs"] == {"result": "information"}
    n = len(r.requests)
    f("Can I book for Tuesday?")
    req = r.requests[n]
    assert [m.role for m in req.messages] == ["user", "assistant"] * 4 + ["user"]
    f.undo_opt()
    assert f.demos == []
    assert len(f.programs()) == 1 and f.optimization_runs()[0]["optimizer"] == "BootstrapFewShot"


def test_labeled_few_shot(fake):
    f = make_classifier()
    fake()
    f.opt(trainset=TRAIN, optimizer=LabeledFewShot, k=2)
    assert len(f.demos) == 2 and all(isinstance(d, dict) for d in f.demos)


def test_examples_can_be_dicts_or_pairs(fake):
    f = make_classifier()
    fake()
    f.opt(trainset=[{"user_query": "a", "result": "booking"},
                    ({"user_query": "b"}, {"result": "information"})], optimizer=LabeledFewShot(k=5))
    assert {d["inputs"]["user_query"] for d in f.demos} == {"a", "b"}


def test_evaluate_scores_in_parallel(fake):
    f = make_classifier()
    fake(responder=lambda req: "<result>\n" + _label(query_of(req)) + "\n</result>")
    res = evaluate(f, TRAIN, functai.exact_match, num_threads=4)
    assert res.score == 100.0 and len(res.results) == 4
    res = Evaluate(devset=TRAIN, metric=lambda ex, pred: pred.result == "booking", num_threads=2)(f)
    assert res.score == 50.0


def test_evaluation_counts_errors_as_zero(fake):
    f = make_classifier()
    fake(responder=lambda req: RuntimeError("down") if "room" in query_of(req) else
         "<result>\n" + _label(query_of(req)) + "\n</result>")
    functai.configure(api_retries=0)
    res = evaluate(f, TRAIN, functai.exact_match)
    assert res.score == 75.0 and len(res.errors) == 1


def test_random_search_picks_the_best_candidate(fake):
    f = make_classifier()

    def responder(req):
        # right only when the prompt carries worked examples
        has_demos = len(req.messages) > 1
        return "<result>\n" + (_label(query_of(req)) if has_demos else "information") + "\n</result>"

    fake(responder=responder)
    opt = BootstrapFewShotWithRandomSearch(num_candidate_programs=2, max_labeled_demos=2)
    f.opt(trainset=TRAIN, optimizer=opt)
    assert f.demos and max(opt.candidates)[1] != "zero-shot"
    assert evaluate(f, TRAIN, functai.exact_match).score == 100.0


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
    f.opt(trainset=TRAIN, optimizer=opt)
    assert "Use the labels exactly" in f.instructions
    assert f.demos == []
    f.undo_opt()
    assert "Use the labels exactly" not in f.instructions


def test_a_teacher_model_bootstraps_the_demos(fake):
    f = make_classifier()

    def responder(req):
        smart = req.model == "gpt-4.1"
        return "<result>\n" + (_label(query_of(req)) if smart else "information") + "\n</result>"

    r = fake(responder=responder)
    f.opt(trainset=TRAIN, teacher_lm="gpt-4.1", max_labeled_demos=4)
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
    f.opt(n_synth=2, teacher_lm="gpt-4.1", optimizer=LabeledFewShot)
    assert sorted(d["inputs"]["user_query"] for d in f.demos) == ["reserve a table", "where is it"]


def test_dspy_optimizers_are_refused_with_a_way_forward(fake):
    class MIPROv2:
        def __init__(self, **kw): ...
    MIPROv2.__module__ = "dspy.teleprompt.mipro_optimizer_v2"
    f = make_classifier()
    fake()
    with pytest.raises(TypeError, match="InstructionSearch"):
        f.opt(trainset=TRAIN, optimizer=MIPROv2)


def test_save_and_load(fake, tmp_path):
    f = make_classifier()
    fake(responder=lambda req: "<result>\n" + _label(query_of(req)) + "\n</result>")
    f.opt(trainset=TRAIN, max_bootstrapped_demos=2, max_labeled_demos=3)
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
    trainset = [Example(claim="The Eiffel Tower is in Paris.", result=["fact: paris"]).with_inputs("claim")]

    def metric(example, prediction, trace=None):
        return 1.0 if prediction.result else 0.0

    research_hop.opt(trainset=trainset, metric=metric, call_defaults=dict(hops=1))
    assert len(generate_query.demos) == 1 and len(append_notes.demos) == 1
    n = len(r.requests)
    research_hop("Big Ben is in London.", 1)
    assert [m.role for m in r.requests[n].messages] == ["user", "assistant", "user"]
    research_hop.undo_opt()
    assert generate_query.demos == [] and append_notes.demos == []


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
