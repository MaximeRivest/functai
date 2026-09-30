"""Stage 5: rated turns as rows that keep their context; evaluating and optimizing on them; splitting by
conversation. The contract is contract/calls.md, *Rows that keep their context*."""

import json
import warnings

import pytest

import functai
from functai import ai, calllog, module

XML = "<result>\n{}\n</result>"
pytest.importorskip("dpyr")


def texts(request):
    return " ".join(getattr(p, "text", "") or "" for m in request.messages for p in m.parts)


@ai
def tutor(message: str) -> str:
    """Tutor a student in fractions."""


def _rows(table):
    return table.collect().to_dicts()


def test_a_rated_turn_is_a_row_that_keeps_its_earlier_turns(fake, tmp_path):
    r = fake(responder=lambda req: XML.format(f"after {len(req.messages)}"))
    functai.configure(log_calls=tmp_path)
    chat = tutor.conversation("alex")
    chat("Hi, I'm Alex.")
    p = chat.predict("What is my name?")
    functai.rate(p.call_id, "wrong", answer="Alex")
    [row] = _rows(functai.rated(tutor))
    assert row["message"] == "What is my name?" and row["result"] == "Alex" and row["conversation"] == "alex"
    assert row["earlier"] == [{"inputs": {"message": "Hi, I'm Alex."}, "outputs": {"result": "after 1"},
                               "steps": row["earlier"][0]["steps"],
                               "signature": row["earlier"][0]["signature"]}]
    # asked again with its own earlier turns; no conversation changes
    asked = len(r.requests)
    ev = functai.evaluate(tutor, _rows(functai.rated(tutor)))
    assert "Hi, I'm Alex." in texts(r.requests[asked]) and len(r.requests[asked].messages) == 3
    assert len(chat.turns) == 2
    again = max(calllog.read(tmp_path)[0], key=lambda c: c["started"])
    assert again["saw"] == [{"saw_of": p.call_id}] and "conversation" not in again
    assert ev.table is not None


def test_a_row_whose_earlier_turns_were_not_kept_is_left_out(fake, tmp_path):
    fake(responder=lambda req: XML.format("ok"))
    functai.configure(log_calls=tmp_path)
    chat = tutor.conversation()
    with functai.configure(log_content=False):
        chat("secret first message")
    p = chat.predict("second")
    functai.rate(p.call_id, "right")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="shown earlier turns the log cannot show again"):
            functai.rated(tutor)
    assert caught is not None


def test_rows_without_a_conversation_look_as_before(fake, tmp_path):
    fake(responder=lambda req: XML.format("ok"))
    functai.configure(log_calls=tmp_path)
    functai.rate(tutor.predict("alone").call_id, "right")
    [row] = _rows(functai.rated(tutor))
    assert "earlier" not in row and "conversation" not in row


def test_a_module_row_keeps_what_each_helper_was_shown(fake, tmp_path):
    @ai
    def answer(message: str) -> str:
        """Answer kindly."""

    @module
    def support(message: str) -> str:
        return answer(message)

    r = fake(responder=lambda req: XML.format(f"after {len(req.messages)}"))
    functai.configure(log_calls=tmp_path)
    chat = support.conversation(remembers={answer: "conversation"})
    chat("where is B-2210?")
    chat("and B-2211?")
    functai.rate(chat.turns[-1].id, "right")
    [row] = _rows(functai.rated(support))
    assert [h["program"] for h in row["helpers"]] == ["answer"]
    assert row["helpers"][0]["earlier"][0]["inputs"] == {"message": "where is B-2210?"}
    assert [t["inputs"] for t in row["earlier"]] == [{"message": "where is B-2210?"}]
    asked = len(r.requests)
    functai.evaluate(support, _rows(functai.rated(support)))
    assert "where is B-2210?" in texts(r.requests[asked])            # the helper's memory, as it was


def test_optimizers_measure_with_such_rows_but_never_show_them_as_examples(fake, tmp_path):
    fake(responder=lambda req: XML.format("ok"))
    functai.configure(log_calls=tmp_path)
    chat = tutor.conversation()
    chat("first")
    functai.rate(chat.predict("second").call_id, "right")
    rows = _rows(functai.rated(tutor)) + [{"message": "plain", "result": "ok"}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        better = functai.labeled_few_shot(tutor, rows, k=5)
    assert [d["inputs"]["message"] for d in better.demos] == ["plain"]
    assert any("never used as worked examples" in str(w.message) for w in caught)


def test_split_keeps_each_conversation_on_one_side():
    rows = [{"x": i, "conversation": f"c{i % 4}"} for i in range(20)] + [{"x": 99, "conversation": None}]
    train, test = functai.split(rows, test=0.25, seed=1)
    assert len(train) + len(test) == 21 and test
    assert not {r["conversation"] for r in train if r["conversation"]} & {r["conversation"] for r in test}
    with pytest.raises(ValueError):
        functai.split(rows, by="missing")


def test_a_record_keeps_its_steps_only_when_whole(fake, tmp_path):
    fake(responder=lambda req: XML.format("ok"))
    functai.configure(log_calls=tmp_path)
    chat = tutor.conversation()
    chat("kept")
    with functai.configure(log_content={"message": False}):
        tutor.conversation()("dropped")
    recs = {c.get("inputs", {}).get("message", "dropped"): c for c in calllog.read(tmp_path)[0]}
    assert recs["kept"]["steps"][0]["kind"] == "model" and "steps" not in recs["dropped"]


def test_a_call_continued_by_a_later_writer_is_its_record(tmp_path):
    base = {"functai_call": 2, "id": "01926a8e-6c1a-7b3e-9d2f-0a8c5e4b1f21", "started": "2026-09-30T00:00:00.000000Z"}
    day = tmp_path / "2026-09-30"
    day.mkdir()
    (day / "a.jsonl").write_text(json.dumps({**base, "outputs": None}) + "\n")
    (day / "b.jsonl").write_text(json.dumps({**base, "writer": 2, "outputs": {"result": "x"}}) + "\n")
    [call] = calllog.read(tmp_path)[0]
    assert call["writer"] == 2 and call["outputs"] == {"result": "x"}
