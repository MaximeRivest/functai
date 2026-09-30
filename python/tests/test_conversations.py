"""Stage 2 (and stage 3's memory): conversations, their stores, turns, branches, queues, leases.

Offline: a fake provider answers. The contract is contract/conversations.md."""

import json
import re
import threading
import time

import pytest

import functai
from functai import ai, calllog, module, stores
from functai.stores import MemoryConversations

XML = "<result>\n{}\n</result>"


def texts(request):
    return [getattr(p, "text", "") or "" for m in request.messages for p in m.parts]


@pytest.fixture
def echo(fake):
    """A model that answers with how many messages it was sent and the last one."""
    return fake(responder=lambda req: XML.format(f"{len(req.messages)}: "
                                                 f"{re.sub(r'<[^>]+>', '', texts(req)[-1]).strip()[-40:]}"))


@ai
def tutor(message: str) -> str:
    """Tutor a student in fractions."""


# ------------------------------------------------------------------ vignette 1


def test_a_tutor_that_remembers(echo, tmp_path):
    chat = tutor.conversation("alex", store=tmp_path)
    chat("Hi, I'm Alex.")
    chat("What is 1/2 + 1/3?")
    chat("Is it 2/5?")
    assert len(echo.requests[2].messages) == 5 and "Hi, I'm Alex." in texts(echo.requests[2])[0]
    assert [t.inputs["message"] for t in chat.turns[-1].saw] == ["Hi, I'm Alex.", "What is 1/2 + 1/3?"]
    before = len(echo.requests)
    request = chat.render("Why can't I just add the bottoms?")
    assert len(echo.requests) == before and len(request.messages) == 7      # the exact next request, nothing sent
    other = chat.continue_from(chat.turns[1])
    other("Is it 5/6?")
    assert (len(chat.turns), len(other.turns)) == (3, 3)                   # both paths are kept
    assert "2/5" not in " ".join(texts(echo.requests[-1]))                  # the branch never saw the other path
    tutor("alone")                                                          # the function itself is unchanged
    assert len(echo.requests[-1].messages) == 1
    # tomorrow, in another process: the same line opens it, at its head (the most recent turn)
    stores._folders.clear()
    again = tutor.conversation("alex", store=tmp_path)
    again("I'm back.")
    assert "Is it 5/6?" in " ".join(texts(echo.requests[-1])) and len(again.turns) == 4


def test_a_turn_knows_what_it_is(echo):
    chat = tutor.conversation()
    chat("one")
    t = chat.turns[-1]
    assert t.state == "done" and t.parent is None and t.call == t.id and t.model == "gpt-4.1-mini"
    assert t.result == "1: one" and t.outputs == {"result": "1: one"} and t.inputs == {"message": "one"}
    assert t.usage["total_tokens"] == 5
    chat("two")
    assert chat.turns[-1].parent == t.id and "2 turns" in repr(chat)


def test_a_turn_is_known_before_its_call_starts(fake):
    gate = threading.Event()

    def slow(req):
        gate.wait(5)
        return XML.format("done")

    fake(responder=slow)
    chat = tutor.conversation()
    s = chat.stream("hello")
    t = s.turn                                                              # known at once, saved already
    assert t.state == "running" and chat.turns[-1].id == t.id
    gate.set()
    assert s.result == "done" and chat.turn(t.id).state == "done"


# ------------------------------------------------------------------ what the model sees


def test_every_earlier_turn_is_shown_by_default_and_saw_does_not_grow(echo, tmp_path):
    chat = tutor.using(log_calls=tmp_path).conversation()
    for m in ("a", "b", "c", "d"):
        chat(m)
    recs = sorted(calllog.read(tmp_path)[0], key=lambda r: r["started"])
    assert [len(r["saw"]) for r in recs] == [0, 1, 2, 2]                    # saw_of the previous turn, then it
    assert calllog.saw(recs[3]["id"], recs) == [{"call": r["id"], "steps": True} for r in recs[:3]]
    assert len(echo.requests[3].messages) == 7


def test_a_rule_leaves_bulky_inputs_out_of_earlier_turns(fake, tmp_path):
    @ai
    def reader(document: str, question: str) -> str:
        """Answer the question from the document."""

    r = fake(responder=lambda req: XML.format("ok"))
    chat = reader.using(log_calls=tmp_path).conversation(context=functai.last_turns(5, without=["document"]))
    chat("BULKY-DOCUMENT", "first?")
    chat("SECOND-DOCUMENT", "second?")
    sent = " ".join(texts(r.requests[1]))
    assert "BULKY-DOCUMENT" not in sent and "first?" in sent and "SECOND-DOCUMENT" in sent
    last = max(calllog.read(tmp_path)[0], key=lambda c: c["started"])
    assert last["saw"] == [{"call": chat.turns[0].id, "without": ["document"]}]


def test_the_context_rule_is_checked(echo):
    with pytest.raises(ValueError):
        functai.last_turns(-1)
    with pytest.raises(TypeError):
        tutor.conversation(context=3)


# ------------------------------------------------------------------ the same send twice, two sends at once


def test_the_same_request_id_twice_is_one_turn(echo, tmp_path):
    chat = tutor.conversation("r", store=tmp_path)
    a = chat("hi", request_id="page-7")
    b = tutor.conversation("r", store=tmp_path)("hi", request_id="page-7")
    assert a == b and len(chat.all_turns()) == 1 and len(echo.requests) == 1


def test_two_sends_at_once_queue(fake, tmp_path):
    started = threading.Event()
    release = threading.Event()

    def slow(req):
        if not started.is_set():
            started.set()
            release.wait(5)
        return XML.format(f"{len(req.messages)}")

    fake(responder=slow)
    first = tutor.conversation("q", store=tmp_path).stream("first")
    assert started.wait(5)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("v", tutor.conversation("q", store=tmp_path)("second")))
    t.start()
    time.sleep(0.3)
    assert "v" not in out                                                   # waits behind the running turn
    release.set()
    first.result
    t.join(5)
    chat = tutor.conversation("q", store=tmp_path)
    assert [x.inputs["message"] for x in chat.turns] == ["first", "second"]
    assert out["v"] == "3"                                                  # the second saw the first


def test_a_conversation_may_refuse_a_second_send(fake):
    gate = threading.Event()
    fake(responder=lambda req: (gate.wait(5), XML.format("x"))[1])
    chat = tutor.conversation(sends="refuse")
    s = chat.stream("first")
    with pytest.raises(functai.ConversationError) as err:
        tutor.conversation(chat.id, sends="refuse")("second")
    assert err.value.code == "conversation-busy"
    gate.set()
    s.result


def test_branches_answer_in_parallel(fake):
    fake(responder=lambda req: XML.format(f"{len(req.messages)}"))
    chat = tutor.conversation()
    chat("question")
    q = chat.turns[-1]
    streams = [chat.continue_from(q).stream("again", lm=m) for m in ("gpt-4.1-mini", "gpt-4.1-nano")]
    assert [s.result for s in streams] == ["3", "3"]
    assert sorted(chat.turn(s.turn.id).model for s in streams) == ["gpt-4.1-mini", "gpt-4.1-nano"]
    assert all(chat.turn(s.turn.id).parent == q.id for s in streams)


# ------------------------------------------------------------------ stopping, leases


def test_a_turn_is_stopped_from_another_process(fake, tmp_path):
    gate = threading.Event()
    fake(responder=lambda req: (gate.wait(3), XML.format("late"))[1])
    s = tutor.conversation("s", store=tmp_path).stream("hi")
    elsewhere = tutor.conversation("s", store=tmp_path)
    elsewhere.stop(elsewhere.turns[-1])
    time.sleep(0.6)
    gate.set()
    with pytest.raises(functai.Cancelled):
        s.result
    assert elsewhere.turns[-1].state == "stopped"


def test_a_turn_whose_process_stopped_is_interrupted_and_is_never_a_parent(echo, tmp_path):
    chat = tutor.conversation("i", store=tmp_path)
    chat("done one")
    done = chat.turns[-1]
    dead = calllog.new_id()
    past = calllog._iso(time.time() - 60)
    chat.store.append("i", [
        {"functai_conversation": 1, "kind": "turn", "at": past, "turn": dead, "parent": done.id,
         "program": chat.turns[-1]._st.record["program"], "inputs": {"message": "lost"}},
        {"functai_conversation": 1, "kind": "lease", "at": past, "turn": dead, "holder": "gone:1:x",
         "until": past, "attempt": 1}])
    again = tutor.conversation("i", store=tmp_path)
    assert again.turn(dead).state == "interrupted"
    again("next")                                                          # continues from the last turn that ended
    assert again.turns[-1].parent == done.id
    assert again.turn(dead).resume() == "3: lost"                           # it can go on: its call again, same turn
    assert again.turn(dead).state == "done"


# ------------------------------------------------------------------ what a conversation refuses


def test_a_host_that_keeps_no_transcripts_refuses_a_stored_conversation(echo, tmp_path):
    with functai.configure(log_content={"message": False}):
        with pytest.raises(functai.ConversationError) as err:
            tutor.conversation("secret", store=tmp_path)
        assert err.value.code == "conversation-content"
        tutor.conversation("secret")("in memory is fine")                  # memory keeps nothing beyond the process


def test_a_field_with_no_json_form_is_refused(echo):
    from typing import Any

    @module
    def summarize(frame: Any) -> str:
        return "two rows"

    with pytest.raises(functai.ConversationError) as err:
        summarize.conversation()
    assert err.value.code == "conversation-opaque"


def test_a_bad_id_is_refused(echo):
    for bad in ("../x", "", "a/b", ".hidden", "x" * 201):
        with pytest.raises(functai.ConversationError) as err:
            tutor.conversation(bad)
        assert err.value.code == "conversation-id"


def test_wrong_inputs_are_refused_before_anything_is_kept(echo):
    chat = tutor.conversation()
    with pytest.raises(TypeError):
        chat()
    with pytest.raises(TypeError):
        chat("a", unknown=1)
    assert chat.all_turns() == []


# ------------------------------------------------------------------ vignette 9: three answers, then one


def test_a_merge_is_a_turn_the_next_one_sees(fake):
    r = fake(responder=lambda req: XML.format("merged" if "Combine" in str(req.system) else "an answer"))

    @ai
    def merge(answers: list[dict]) -> str:
        """Combine these answers into one."""

    chat = tutor.conversation()
    chat("q")
    q = chat.turns[-1]
    branches = [chat.continue_from(q).predict("again", lm=m).call_id for m in ("gpt-4.1-mini", "gpt-4.1-nano")]
    merged = chat.continue_from(q).merge(branches, merge)
    assert merged.result == "merged" and merged.reads == branches and merged.made_by["name"] == "merge"
    after = chat.continue_from(merged)
    after("thanks")
    assert "merged" in " ".join(texts(r.requests[-1]))                      # the next turn sees the merge
    assert after.turns[-2].id == merged.id


# ------------------------------------------------------------------ stores


def test_a_store_of_your_own_needs_two_methods(echo):
    class Mine:
        def __init__(self):
            self.records = {}

        def append(self, conversation, records, *, expect=None):
            log = self.records.setdefault(conversation, [])
            if expect is not None and expect != len(log):
                raise functai.ConversationError("store-conflict", "changed")
            for r in records:
                log.append({**r, "seq": len(log) + 1})
            return len(log)

        def read(self, conversation, after=0):
            return [dict(r) for r in self.records.get(conversation, [])[after:]]

    mine = Mine()
    chat = tutor.conversation("own", store=mine)
    chat("hi")
    chat("again")
    kinds = [r["kind"] for r in mine.records["own"]]
    assert kinds[:3] == ["program", "turn", "lease"] and kinds.count("ended") == 2
    assert chat.turns[-1].events() is not None and list(chat.turns[-1].events()) == []   # no event log: none


def test_a_folder_store_keeps_records_and_events(echo, tmp_path):
    chat = tutor.conversation("f", store=tmp_path)
    chat("hi")
    path = tmp_path / "conversations" / "f.jsonl"
    records = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["seq"] for r in records] == list(range(1, len(records) + 1))
    assert all(r["functai_conversation"] == 1 for r in records)
    events = list(chat.turns[-1].events())
    assert events[0]["kind"] == "started" and events[-1]["kind"] == "done"
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    with pytest.raises(functai.ConversationError):
        chat.store.append("f", [{"kind": "x"}], expect=1)                    # conditional: the one compare-and-set


def test_memory_conversations_are_one_per_process(echo):
    tutor.conversation("shared")("hi")
    assert len(tutor.conversation("shared").turns) == 1
    assert len(tutor.conversation("shared", store=MemoryConversations()).turns) == 0


def test_the_head_can_be_moved_for_everyone(echo):
    chat = tutor.conversation("h")
    chat("a")
    chat("b")
    first = chat.turns[0]
    chat.head = first
    assert tutor.conversation("h").head.id == first.id
    tutor.conversation("h")("c")
    assert [t.inputs["message"] for t in tutor.conversation("h").turns] == ["a", "c"]


# ------------------------------------------------------------------ stage 3: helpers inside a module


@ai
def topic(message: str) -> str:
    """What is the customer writing about? One word."""


@ai
def answer(message: str, topic: str) -> str:
    """Answer kindly."""


@ai
def handoff(conversation: list[dict[str, str]]) -> str:
    """Summarize this support conversation for the person who takes it over."""


@pytest.fixture
def support_model(fake):
    def respond(req):
        system = str(req.system)
        if "writing about" in system:
            return XML.format("shipping")
        if "Summarize" in system:
            return XML.format("summary")
        return XML.format(f"answer after {len(req.messages)} messages")
    return fake(responder=respond)


def test_helpers_remember_only_when_told(support_model):
    seen = []

    @module
    def support(message: str) -> str:
        seen.append(functai.earlier())
        return answer(message, topic(message))

    chat = support.conversation(remembers={answer: "conversation"})
    assert chat("where is B-2210?") == "answer after 1 messages"
    assert chat("and B-2211?") == "answer after 3 messages"               # answer remembers its own calls
    topics = [r for r in support_model.requests if "writing about" in str(r.system)]
    assert all(len(r.messages) == 1 for r in topics)                        # topic remembers nothing
    assert seen[-1] == [{"message": "where is B-2210?", "result": "answer after 1 messages"}]
    assert len(chat.render("next?", "shipping", call=answer).messages) == 5
    other = chat.continue_from(chat.turns[0])
    assert other("cancel it") == "answer after 3 messages"                  # its own branch only


def test_a_helper_may_remember_this_turn_only(support_model):
    @module
    def loop(message: str) -> str:
        answer(message, "a")
        return answer(message, "b")

    chat = loop.conversation(remembers={answer: "turn"})
    assert chat("x") == "answer after 3 messages"                          # its first call in this turn
    assert chat("y") == "answer after 3 messages"                          # nothing of the earlier turn


def test_earlier_is_data_a_helper_can_take(support_model):
    @module
    def support(message: str) -> str:
        if message == "person please":
            return handoff(functai.earlier())
        return answer(message, topic(message))

    chat = support.conversation()
    chat("hello")
    assert chat("person please") == "summary"
    sent = " ".join(texts(support_model.requests[-1]))
    assert "hello" in sent and "answer after 1 messages" in sent


def test_remembers_names_a_helper_the_module_calls(support_model):
    @module
    def support(message: str) -> str:
        return answer(message, "x")

    with pytest.raises(ValueError):
        support.conversation(remembers={handoff: "conversation"})
    with pytest.raises(ValueError):
        support.conversation(remembers={answer: "for ever"})
    with pytest.raises(TypeError):
        answer.conversation(remembers={answer: "turn"})


def test_a_remembering_program_inside_another_is_refused_unless_declared(support_model):
    inner = answer.conversation("inner")

    @module
    def outer(message: str) -> str:
        return inner(message, "x")

    with pytest.raises(functai.ConversationError) as err:
        outer.conversation("o1")("hi")
    assert err.value.code == "conversation-nested"
    assert outer.conversation("o2", remembers={inner: "own"})("hi") == "answer after 1 messages"
    assert len(inner.turns) == 1


def test_a_module_turn_records_what_it_was_shown(support_model, tmp_path):
    @module(log_calls=tmp_path)
    def support(message: str) -> str:
        return answer(message, topic(message))

    chat = support.conversation()
    chat("a")
    chat("b")
    chat("c")
    recs = {r["id"]: r for r in calllog.read(tmp_path)[0]}
    last = recs[chat.turns[-1].id]
    assert last["program"]["kind"] == "module" and last["conversation"]["id"] == chat.id
    assert calllog.saw(last["id"], recs) == [{"call": chat.turns[0].id}, {"call": chat.turns[1].id}]
    assert chat.turns[-1].tree().splitlines() == ["support", "├─ topic", "└─ answer"]
    assert [c.function for c in chat.turns[-1].find(topic)] == ["topic"]
