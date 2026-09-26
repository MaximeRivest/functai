"""Evaluation results are tables: offline, the fake model answers by rules."""

import dataclasses
import enum
import re

import dpyr
import polars as pl
import pytest
from dpyr import col

import functai
from functai import _ai, ai, compare, evaluate, module

ROWS = [
    {"user_query": "I need to reserve a room.", "result": "booking", "lang": "en"},
    {"user_query": "How do I get there?", "result": "information", "lang": "en"},
    {"user_query": "Annuler ma réservation.", "result": "cancelation", "lang": "fr"},
    {"user_query": "Book me a suite.", "result": "booking", "lang": "en"},
]


def query_of(req):
    text = "".join(p.text for p in req.messages[-1].parts)
    m = re.search(r"<user_query>\n(.*?)\n</user_query>", text, re.S)
    return m.group(1) if m else ""


def label(query):
    q = query.lower()
    return "booking" if ("reserve" in q or "book" in q) else "cancelation" if "cancel" in q else "information"


def make_classifier():
    @ai
    def classify_intent(user_query: str) -> str:
        """Classify user intent as 'booking', 'cancelation', or 'information'."""
        return _ai
    return classify_intent


def answer(req):
    return "<result>\n" + label(query_of(req)) + "\n</result>"


# ------------------------------------------------------------------ the score


def test_score_is_a_fraction_with_an_interval(fake):
    fake(responder=answer)                     # misses the French cancelation ("annuler")
    ev = evaluate(make_classifier(), ROWS)     # exact match: the data has a `result` column
    assert ev.metrics == ["exact_match"]
    assert ev.score == 0.75 and float(ev) == 0.75
    assert "exact_match 0.75 [0.30, 0.95]" in repr(ev)
    s = ev.summary.collect().to_dicts()[0]
    assert s["metric"] == "exact_match" and s["n"] == 4 and s["failed"] == 0
    assert s["low"] == pytest.approx(0.3006, abs=1e-4) and s["high"] == pytest.approx(0.9544, abs=1e-4)


def test_intervals_wilson_for_0_1_and_student_t_otherwise():
    from functai.evaluation import interval
    assert interval([1.0, 1.0, 1.0, 1.0])[1:] == (pytest.approx(0.5101, abs=1e-4), 1.0)
    mean, low, high = interval([0.5, 0.7, 0.9])
    assert mean == pytest.approx(0.7) and (low, high) == (pytest.approx(0.2031, abs=1e-4),
                                                          pytest.approx(1.1969, abs=1e-4))
    assert interval([0.4]) == (0.4, None, None)


def test_failed_rows_count_zero_in_the_score_and_are_null_in_the_table(fake):
    fake(responder=lambda req: RuntimeError("down") if "room" in query_of(req) else answer(req))
    functai.configure(api_retries=0)
    ev = evaluate(make_classifier(), ROWS)
    assert ev.score == 0.5 and len(ev.errors) == 1 and ev.errors[0][0] == 0
    t = ev.table.arrange(col.example).collect()
    assert t["exact_match"].to_list() == [None, 1.0, 0.0, 1.0]
    assert t["pred_result"][0] is None and "down" in t["error"][0]
    assert ev.summary.collect()["failed"].to_list() == [1]
    with pytest.raises(RuntimeError, match="1 rows failed"):
        evaluate(make_classifier(), ROWS, max_errors=0)


# ------------------------------------------------------------------ the table


def test_the_table_has_the_data_the_predictions_the_metrics_and_the_costs(fake):
    fake(responder=answer)
    ev = evaluate(make_classifier(), ROWS)
    t = ev.table
    assert t.columns == ["example", "user_query", "result", "lang", "pred_result", "exact_match",
                         "error", "seconds", "input_tokens", "output_tokens", "model", "run"]
    rows = t.collect().to_dicts()
    assert rows[0]["pred_result"] == "booking" and rows[0]["input_tokens"] == 3
    assert rows[0]["model"] == "gpt-4.1-mini" and rows[0]["run"] == ev.run
    by_lang = t.group_by(col.lang).summarize(acc=col.exact_match.mean()).arrange(col.lang).collect()
    assert by_lang.to_dicts() == [{"lang": "en", "acc": 1.0}, {"lang": "fr", "acc": 0.0}]


def test_data_can_be_any_table(fake, tmp_path):
    fake(responder=answer)
    path = tmp_path / "dev.parquet"
    pl.DataFrame(ROWS).write_parquet(path)
    for data in (str(path), pl.DataFrame(ROWS), dpyr.read(ROWS)):
        assert evaluate(make_classifier(), data).score == 0.75


def test_python_objects_stay_objects_for_the_call_and_become_text_in_the_table(fake):
    class Doc:
        def __init__(self, text):
            self.text = text

        def __str__(self):
            return self.text

    seen = []

    @ai
    def classify(doc) -> str:
        """Classify."""
        seen.append(doc)
        return _ai

    fake("<result>\nbooking\n</result>")
    ev = evaluate(classify, [{"doc": Doc("book it"), "result": "booking"}])
    assert isinstance(seen[0], Doc)
    assert ev.table.collect()["doc"].to_list() == ["book it"]


def test_structured_outputs_are_nested_columns(fake):
    class Tone(enum.Enum):
        HAPPY = "happy"

    @dataclasses.dataclass
    class Entity:
        name: str
        kind: str

    @ai
    def extract(text: str) -> list[Entity]:
        """The entities."""

    @ai
    def tone(text: str) -> Tone:
        """The tone."""

    fake(responder=lambda req: '<result>\n[{"name": "Paris", "kind": "city"}]\n</result>')
    t = evaluate(extract, [{"text": "I love Paris"}]).table
    assert t.schema["pred_result"] == dpyr.dtypes.nested("List(Struct(name: Str, kind: Str))")
    assert t.collect()["pred_result"].to_list() == [[{"name": "Paris", "kind": "city"}]]
    fake("<result>\nhappy\n</result>")
    assert tone.map([{"text": "yay"}]).collect()["pred_result"].to_list() == ["happy"]


def test_a_column_whose_types_mix_is_kept_as_json_text(fake):
    @ai
    def anything(text: str) -> object:
        """Anything."""

    fake('<result>\n4\n</result>', '<result>\n"four"\n</result>')
    with pytest.warns(UserWarning, match="pred_result"):
        t = evaluate(anything, [{"text": "a"}, {"text": "b"}]).table
    assert t.collect()["pred_result"].to_list() == ["4", '"four"']


def test_map_runs_a_function_over_a_table(fake):
    fake(responder=answer)
    out = make_classifier().map(pl.DataFrame(ROWS).drop("result"), num_threads=2)
    assert out.columns[:4] == ["example", "user_query", "lang", "pred_result"]
    assert out.filter(col.pred_result == "booking").shape[0] == 2


# ------------------------------------------------------------------ metrics


def test_metrics_are_callables_or_dpyr_expressions_and_get_their_own_columns(fake):
    fake(responder=answer)

    def is_booking(row, pred):
        return pred.result == "booking"

    ev = evaluate(make_classifier(), ROWS, {
        "correct": col.pred_result == col.result,
        "booking": is_booking,
        "short": lambda row, pred: len(pred.result) < 10,
    })
    assert ev.metrics == ["correct", "booking", "short"]
    assert ev.score == 0.75                                # the first metric
    summary = {r["metric"]: r["mean"] for r in ev.summary.collect().to_dicts()}
    assert summary == {"correct": 0.75, "booking": 0.5, "short": 0.5}
    assert ev.table.select(col.correct, col.booking).schema == {"correct": dpyr.FLOAT64,
                                                                  "booking": dpyr.FLOAT64}


def test_unnamed_metrics_are_named_after_their_function(fake):
    fake(responder=answer)
    ev = evaluate(make_classifier(), ROWS, [functai.exact_match, lambda row, pred: 1.0,
                                             col.pred_result == "booking"])
    assert ev.metrics == ["exact_match", "score", "score_2"]


def test_an_ai_judge_is_a_metric(fake):
    @ai
    def judge(row, prediction) -> float:
        """Between 0 and 1: how close the prediction is to the row's result."""

    def responder(req):
        return "<result>\n0.5\n</result>" if "how close" in (req.system or "") else answer(req)

    fake(responder=responder)
    ev = evaluate(make_classifier(), ROWS, judge)
    assert ev.metrics == ["judge"] and ev.score == 0.5


def test_a_metric_that_fails_on_a_row_is_that_rows_error(fake):
    fake(responder=answer)

    def picky(row, pred):
        if row["lang"] == "fr":
            raise KeyError("no french")
        return 1.0

    ev = evaluate(make_classifier(), ROWS, picky)
    assert ev.score == 0.75 and "metric picky: KeyError" in ev.errors[0][1]


def test_mistakes_are_caught_before_any_model_call(fake):
    r = fake(responder=answer)
    f = make_classifier()
    with pytest.raises(ValueError, match="no column named"):
        evaluate(f, [{"user_query": "a"}], functai.exact_match)
    with pytest.raises(TypeError, match=r"must take \(row, prediction\)"):
        evaluate(f, ROWS, lambda row, pred, trace: 1.0)
    with pytest.raises(ValueError, match=r"\['model'\]"):
        evaluate(f, [{"user_query": "a", "model": "x"}])
    assert r.requests == []


# ------------------------------------------------------------------ comparing and logging runs


def test_compare_pairs_the_examples(fake):
    f = make_classifier()
    fake(responder=lambda req: "<result>\ninformation\n</result>")
    before = evaluate(f, ROWS)
    fake(responder=answer)
    after = evaluate(f, ROWS)
    diff = compare(before, after).collect().to_dicts()[0]
    assert (diff["before"], diff["after"], diff["diff"]) == (0.25, 0.75, 0.5)
    assert (diff["better"], diff["worse"], diff["same"]) == (2, 0, 2)
    assert diff["low"] < 0.5 < diff["high"]
    with pytest.raises(ValueError, match="same examples"):
        compare(before, evaluate(f, ROWS[:2]))


def test_runs_are_logged_as_parquet_and_read_back_as_one_table(fake, tmp_path):
    f = make_classifier()
    fake(responder=answer)
    a = evaluate(f, ROWS, log=tmp_path)
    b = evaluate(f, ROWS, {"booked": col.pred_result == "booking"}, log=tmp_path)
    everything = functai.runs(tmp_path)
    assert everything.shape[0] == 8
    per_run = everything.group_by(col.run).summarize(n=dpyr.n()).collect()
    assert sorted(per_run["run"].to_list()) == sorted([a.run, b.run])
    assert {"exact_match", "booked"} <= set(everything.columns)       # lined up by name


# ------------------------------------------------------------------ modules


@ai
def shout(text: str) -> str:
    """Shout."""


@module
def shout_twice(text: str):
    return [shout(text), shout(text + "!")]


def test_a_module_is_evaluated_on_what_it_returns_and_costs_all_its_calls(fake):
    fake(responder=lambda req: "<result>\nHEY\n</result>")
    ev = evaluate(shout_twice, [{"text": "hey", "result": ["HEY", "HEY"]}])
    assert ev.score == 1.0
    row = ev.table.collect().to_dicts()[0]
    assert row["pred_result"] == ["HEY", "HEY"]
    assert row["input_tokens"] == 6 and row["output_tokens"] == 4        # both calls


# ------------------------------------------------------------------ without dpyr


def test_without_dpyr_the_score_works_and_tables_say_what_to_install(fake, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "dpyr", None)          # `import dpyr` now fails
    fake(responder=answer)
    f = make_classifier()
    ev = evaluate(f, ROWS)
    assert ev.score == 0.75 and "0.75" in repr(ev)
    with pytest.raises(ImportError, match=r"functai\[data\]"):
        ev.table
    with pytest.raises(ImportError, match=r"functai\[data\]"):
        evaluate(f, "dev.parquet")
    f.opt(trainset=ROWS)                                     # optimizing needs no tables
    assert f.demos


# ------------------------------------------------------------------ AI functions on columns


def test_an_ai_function_called_on_a_column_is_a_column(fake):
    r = fake(responder=answer)
    f = make_classifier()
    frame = dpyr.read(ROWS).mutate(intent=f(col.user_query))
    assert frame.schema["intent"] == dpyr.STR and r.requests == []      # nothing runs yet
    got = frame.collect()
    assert got["intent"].to_list() == ["booking", "information", "information", "booking"]
    assert len(r.requests) == 4
    frame.collect()
    dpyr.read(ROWS).mutate(intent=f(col.user_query)).filter(col.intent == "booking").collect()
    assert len(r.requests) == 4                                           # remembered


def test_several_inputs_columns_and_constants(fake):
    @ai
    def answer_in(question: str, context: str, language: str) -> str:
        """Answer from the context."""

    def responder(req):
        text = "".join(p.text for p in req.messages[-1].parts)
        lang = re.search(r"<language>\n(.*?)\n</language>", text, re.S).group(1)
        ctx = re.search(r"<context>\n(.*?)\n</context>", text, re.S).group(1)
        return f"<result>\n{lang}:{ctx}\n</result>"

    fake(responder=responder)
    t = dpyr.read([{"q": "a", "doc": "x"}, {"q": "b", "doc": "y"}])
    got = t.mutate(a=answer_in(col.q, context=col.doc, language="fr")).collect()
    assert got["a"].to_list() == ["fr:x", "fr:y"]


def test_the_column_keeps_the_prompt_it_was_written_with(fake):
    def responder(req):
        return "<result>\n" + ("B" if "Be terse" in (req.system or "") else "A") + "\n</result>"

    fake(responder=responder)
    f = make_classifier()
    before = dpyr.read(ROWS).mutate(x=f(col.user_query))
    f.instructions = "Be terse."
    after = dpyr.read(ROWS).mutate(x=f(col.user_query))
    assert before.collect()["x"].unique().to_list() == ["A"]
    assert after.collect()["x"].unique().to_list() == ["B"]


def test_structured_outputs_and_filters_on_columns(fake):
    @dataclasses.dataclass
    class Entity:
        name: str
        kind: str

    @ai
    def extract(text: str) -> list[Entity]:
        """The entities."""

    @ai
    def is_booking(user_query: str) -> bool:
        """Is it a booking?"""

    fake(responder=lambda req: '<result>\n[{"name": "Paris", "kind": "city"}]\n</result>'
         if "entities" in (req.system or "") else
         "<result>\n" + str(label(query_of(req)) == "booking").lower() + "\n</result>")
    t = dpyr.read(ROWS)
    assert t.filter(is_booking(col.user_query)).shape[0] == 2
    e = t.mutate(e=extract(col.user_query))
    assert e.schema["e"] == dpyr.dtypes.nested("List(Struct(name: Str, kind: Str))")


def test_failed_rows_raise_after_all_ran_or_become_null(fake):
    fake(responder=lambda req: RuntimeError("down") if "room" in query_of(req) else answer(req))
    functai.configure(api_retries=0)
    f = make_classifier()
    with pytest.raises(dpyr.DpyrError, match="failed on 1 of 4 rows"):
        dpyr.read(ROWS).mutate(x=f(col.user_query)).collect()
    with pytest.warns(UserWarning):
        got = dpyr.read(ROWS).mutate(x=f.vectorize(errors="null")(col.user_query)).collect()
    assert got["x"].to_list() == [None, "information", "information", "booking"]


def test_a_module_on_a_column(fake):
    @module
    def shout_both(text: str) -> list[str]:
        return [shout(text), shout(text + "!")]

    fake(responder=lambda req: "<result>\nHEY\n</result>")
    got = dpyr.read([{"t": "hey"}]).mutate(s=shout_both(col.t)).collect()
    assert got["s"].to_list() == [["HEY", "HEY"]]
    with pytest.raises(TypeError, match="no return annotation"):
        shout_twice(col.t)


def test_an_ai_judge_with_typed_parameters_gets_plain_data(fake):
    @ai
    def typed_judge(row: dict, prediction: dict) -> float:
        """Between 0 and 1: how close the prediction is to the row's result."""

    seen = []

    def responder(req):
        if "how close" in (req.system or ""):
            seen.append("".join(p.text for p in req.messages[-1].parts))
            return "<result>\n1.0\n</result>"
        return answer(req)

    fake(responder=responder)
    ev = evaluate(make_classifier(), ROWS[:1], typed_judge)
    assert ev.errors == [] and ev.score == 1.0
    assert '"result": "booking"' in seen[0]


def test_columns_work_in_modules_with_postponed_annotations(fake, tmp_path, monkeypatch):
    (tmp_path / "postponed.py").write_text(
        "from __future__ import annotations\n"
        "from functai import ai\n\n"
        "@ai\n"
        "def label(text: str) -> list[str]:\n"
        '    """Labels."""\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    import postponed
    fake(responder=lambda req: '<result>\n["a"]\n</result>')
    frame = dpyr.read([{"t": "x"}]).mutate(l=postponed.label(col.t))
    assert frame.schema["l"] == dpyr.dtypes.nested("List(Str)")
    assert frame.collect()["l"].to_list() == [["a"]]
