"""Streaming: the same call, watched while it is made (contract/streaming.md)."""

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import lm15
import pytest

import functai
from conftest import FakeRouter
from functai import _ai, ai, calllog, module
from functai.streaming import Cancelled, Done, Failed, Retry, Started, Text, Thinking, partial_json

CONTRACT = Path(__file__).resolve().parents[2] / "contract"      # the repository's, shared by every language
XML = "<result>\n{}\n</result>"


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for var in (calllog.ENV_FOLDER, calllog.ENV_CONTENT, calllog.ENV_CALLER):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def streaming(fake):
    """``r = streaming(*replies, responder=...)``: a fake model that streams its
    replies in pieces of 3 characters."""
    return fake


def answers(text):
    return lambda req: XML.format(text)


def kinds(events):
    return [e.kind for e in events]


@ai
def haiku(topic: str) -> str:
    """A haiku about the topic."""


@ai
def solve(problem: str) -> float:
    """Solve the word problem."""
    reasoning: str = _ai["Step by step."]
    return _ai


@dataclass
class Contact:
    name: str
    city: Optional[str]
    age: int


@ai
def contacts(text: str) -> List[Contact]:
    """Everyone the text mentions."""


# ------------------------------------------------------------------ the answer, as it is written


def test_iterating_gives_the_answer_as_it_is_written(streaming):
    r = streaming(responder=answers("Snow on the cedar,\nquiet as a held breath."))
    s = haiku.stream("the first snow")
    pieces = list(s)
    assert len(pieces) > 5 and all(len(p) <= 6 for p in pieces)         # pieces as they arrive
    assert "".join(pieces) == "Snow on the cedar,\nquiet as a held breath." == s.result == s.text
    assert s.done and list(s) == pieces                         # iterating again replays
    assert len(r.requests) == 1


def test_a_stream_is_the_same_call(streaming, tmp_path):
    streaming(responder=answers("Snow."))
    functai.configure(log_calls=tmp_path)
    plain = haiku("snow")
    s = haiku.stream("snow")
    assert s.result == plain and s.prediction.result == plain
    assert s.prediction.call_id == s.call_id
    first, second = [json.loads(line) for f in sorted(tmp_path.rglob("*.jsonl")) for line in f.read_text().splitlines()]
    assert second["id"] == s.call_id and second["outputs"] == first["outputs"] and second["error"] is None
    ex = second["exchanges"][0]
    assert ex["streamed"] is True and 0 <= ex["first_delta"] <= ex["seconds"]
    assert "streamed" not in first["exchanges"][0]
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((CONTRACT / "schema" / "call.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(second))


def test_every_field_is_shown_and_the_answer_is_marked(streaming):
    streaming(responder=lambda req: "<reasoning>\nTen pencils cost ten times $0.40.\n</reasoning>\n"
                                    "<result>\n4.0\n</result>")
    s = solve.stream("3 pencils cost $1.20. How much do 10 cost?")
    events = list(s.events())
    assert isinstance(events[0], Started) and events[0].inputs == {"problem": "3 pencils cost $1.20. How much do 10 cost?"}
    assert isinstance(events[-1], Done) and events[-1].value == 4.0 and events[-1].call == s.call_id
    texts = [e for e in events if isinstance(e, Text)]
    reasoning = "".join(e.text for e in texts if e.field == "reasoning")
    assert reasoning == "Ten pencils cost ten times $0.40." and not any(e.answer for e in texts if e.field == "reasoning")
    assert "".join(s) == "4.0" and s.result == 4.0
    assert [e.field for e in texts].index("result") > [e.field for e in texts].index("reasoning")   # in order
    assert s.fields == {"reasoning": "Ten pencils cost ten times $0.40.", "result": "4.0"}
    assert s.prediction.reasoning == reasoning
    assert str(texts[0]) == texts[0].text                       # print(event, end="") prints its text


def test_wrong_arguments_fail_at_once_not_in_the_background(streaming):
    streaming(responder=answers("x"))
    with pytest.raises(TypeError):
        haiku.stream()
    with pytest.raises(TypeError):
        haiku.stream("a", "b")
    with pytest.raises(TypeError):
        haiku.stream("a", all=True)                 # no such input


def test_a_failed_call_raises_at_the_end_of_iteration_and_from_result(streaming):
    streaming(responder=lambda req: "no tags here", )
    s = haiku.using(retries=0).stream("snow")
    with pytest.raises(Exception) as err:
        list(s)
    assert type(err.value).__name__ == "Refusal"
    with pytest.raises(type(err.value)):
        s.result
    seen = []
    with pytest.raises(type(err.value)):
        for event in s.events():
            seen.append(event)
    assert isinstance(seen[-1], Failed) and seen[-1].error is err.value
    assert s.done and "failed: Refusal" in repr(s)


# ------------------------------------------------------------------ retries


def test_an_unreadable_reply_is_asked_again_and_the_retry_is_shown(streaming):
    streaming('<result>\nwritten\n</result> <result>\ntwice\n</result>', XML.format("Once."))
    s = haiku.stream("snow")
    events = list(s.events())
    retry = [e for e in events if isinstance(e, Retry)]
    assert len(retry) == 1 and "could not be read" in retry[0].reason and retry[0].wait is None
    after = events[events.index(retry[0]) + 1:]
    assert "".join(e.text for e in after if isinstance(e, Text)) == "Once."
    assert s.result == "Once." and s.text == "Once."            # the answer so far starts again


def test_a_cut_reply_says_so(streaming):
    streaming(("<result>\nSnow on the ced", "length"), XML.format("Snow on the cedar."))
    s = haiku.stream("snow")
    [retry] = [e for e in s.events() if isinstance(e, Retry)]
    assert "cut off" in retry.reason and s.result == "Snow on the cedar."


class Flaky(FakeRouter):
    """Its first stream breaks after two pieces."""

    def __init__(self, reply):
        super().__init__(responder=lambda req: reply)
        self.broke = False

    def _events(self, response):
        for i, event in enumerate(super()._events(response)):
            if i == 8 and not self.broke:                    # after 'Snow on'
                self.broke = True
                busy = lm15.RateLimitError("slow down", provider="openai")
                busy.retry_after = 0.01
                raise busy
            yield event


def test_a_provider_error_mid_stream_is_sent_again(tmp_path):
    functai.configure(lm="gpt-4.1-mini", client=Flaky(XML.format("Snow on the cedar.")), log_calls=tmp_path)
    s = haiku.stream("snow")
    events = list(s.events())
    [retry] = [e for e in events if isinstance(e, Retry)]
    assert "RateLimitError" in retry.reason and retry.wait == 0.01
    assert [e for e in events[:events.index(retry)] if isinstance(e, Text)]          # text was shown, then voided
    assert s.result == "Snow on the cedar." == s.text
    [rec] = [json.loads(line) for f in tmp_path.rglob("*.jsonl") for line in f.read_text().splitlines()]
    assert [e.get("error", {}).get("type") for e in rec["exchanges"]] == ["RateLimitError", None]
    assert all(e["streamed"] for e in rec["exchanges"])


# ------------------------------------------------------------------ tools and calls inside calls


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"Sunny and 22C in {city}."


def test_tool_calls_and_results_are_shown_whole(streaming):
    @ai(tools=[get_weather])
    def assistant_(question: str) -> str:
        """Answer; use a tool when you need facts."""

    call = lm15.ToolCallPart(id="c1", name="get_weather", input={"city": "Montreal"})
    streaming([call], XML.format("Sunny, 22C."))
    s = assistant_.stream("Weather in Montreal?")
    events = list(s.events())
    assert kinds(events) == ["started", "tool_call", "tool_result"] + ["text"] * 4 + ["done"]
    tc, tr = events[1], events[2]
    assert (tc.id, tc.name, tc.input) == ("c1", "get_weather", {"city": "Montreal"})
    assert (tr.id, tr.output) == ("c1", "Sunny and 22C in Montreal.")
    assert s.result == "Sunny, 22C."


def test_a_tool_that_calls_an_ai_function_shows_its_call_inside(streaming):
    def which_team(message: str) -> str:
        """The team for a message."""
        return team(message)

    @ai
    def team(message: str) -> Literal["shipping", "billing"]:
        """Which team?"""

    @ai(tools=[which_team])
    def helper(question: str) -> str:
        """Answer the customer."""

    call = lm15.ToolCallPart(id="c1", name="which_team", input={"message": "charged twice"})
    streaming([call], XML.format("billing"), XML.format("Billing will help you."))
    s = helper.stream("Who helps me?")
    events = list(s.events())
    inner = [e for e in events if e.function == "team"]
    assert kinds(inner) == ["started", "text", "text", "text", "done"] and inner[0].parent == s.call_id
    assert "".join(s) == "Billing will help you."             # only the stream's own answer


def test_a_module_stream_shows_every_call_inside(streaming):
    @ai
    def draft(topic: str) -> str:
        """A paragraph about the topic."""

    @ai
    def shorten(text: str) -> str:
        """The text in at most twelve words."""

    @module
    def blurb(topic: str) -> str:
        return shorten(draft(topic))

    streaming(XML.format("A long paragraph about snow."), XML.format("Snow, briefly."))
    s = blurb.stream("snow")
    events = list(s.events())
    assert [(e.kind, e.function) for e in events if e.kind != "text"] == [
        ("started", "blurb"), ("started", "draft"), ("done", "draft"), ("started", "shorten"),
        ("done", "shorten"), ("done", "blurb")]
    assert "".join(s.text_of(shorten)) == "Snow, briefly."
    assert "".join(s.text_of(draft)) == "A long paragraph about snow."
    assert s.result == "Snow, briefly."
    done = [e for e in events if isinstance(e, Done)]
    assert done[-1].prediction is None and done[0].prediction.result == "A long paragraph about snow."
    with pytest.raises(TypeError, match="s.events()"):
        iter(s)
    with pytest.raises(TypeError, match="s.result"):
        s.prediction
    with pytest.raises(TypeError):
        blurb.stream()


def test_escalating_to_another_model_is_a_retry(monkeypatch):
    from functai import models
    monkeypatch.setattr(models, "JUDGMENT_ONLY", models.JUDGMENT_ONLY | {"typesafe"})

    def measured(req):
        if req.model == "openai:gpt-4.1":
            return XML.format("billing")
        dist = {"shipping": 0.6, "billing": 0.4}
        return lm15.Response(id=None, model=req.model, message=lm15.Message.assistant(
            [lm15.DataPart(value={"result": "shipping"}, probabilities={"result": dist},
                           method="provider_classification")]),
            finish_reason="stop", usage=lm15.Usage(input_tokens=10, output_tokens=0, total_tokens=10))

    @ai
    def team(message: str) -> Literal["shipping", "billing"]:
        """Which team?"""

    functai.configure(client=FakeRouter(responder=measured, provider="typesafe"))
    s = team.using(lm="typesafe:jev", escalate_to="openai:gpt-4.1", escalate_below=0.9).stream("charged twice")
    events = list(s.events())
    [retry] = [e for e in events if isinstance(e, Retry)]
    assert "60% sure" in retry.reason and "openai:gpt-4.1 answers instead" in retry.reason
    assert s.result == "billing" == s.text


def test_escalating_to_an_ai_function_makes_its_answer_the_streams(streaming):
    @ai(lm="openai:gpt-4.1")
    def careful(message: str) -> Literal["shipping", "billing"]:
        """Which team? Think hard."""

    from functai import models

    def reply(req):
        if req.model == "openai:gpt-4.1":
            return XML.format("billing")
        return lm15.Response(id=None, model=req.model, message=lm15.Message.assistant(
            [lm15.DataPart(value={"result": "shipping"}, probabilities={"result": {"shipping": 0.5, "billing": 0.5}},
                           method="provider_classification")]),
            finish_reason="stop", usage=lm15.Usage(input_tokens=1, output_tokens=0, total_tokens=1))

    @ai
    def quick(message: str) -> Literal["shipping", "billing"]:
        """Which team?"""

    r = FakeRouter(responder=reply, provider="typesafe")
    models_before = models.JUDGMENT_ONLY
    models.JUDGMENT_ONLY = models.JUDGMENT_ONLY | {"typesafe"}
    try:
        functai.configure(client=r, lm="typesafe:jev")
        s = quick.using(escalate_to=careful, escalate_below=0.9).stream("charged twice")
        assert "".join(s) == "shippingbilling"                 # the pieces as shown: a retry cannot unshow
        assert s.text == "billing" == s.result                 # the answer so far starts again
        child = [e for e in s.events() if isinstance(e, Started) and e.function == "careful"]
        assert child and child[0].parent == s.call_id
    finally:
        models.JUDGMENT_ONLY = models_before


# ------------------------------------------------------------------ whole replies


def test_a_cached_reply_is_shown_in_one_piece(streaming):
    r = streaming(responder=answers("Snow on the cedar."))
    cached = haiku.using(cache_replies=True)
    cached("snow")
    s = cached.stream("snow")
    assert list(s) == ["Snow on the cedar."] and len(r.requests) == 1


def test_a_client_that_cannot_stream_answers_whole():
    class Plain:
        def __init__(self):
            self.inner = FakeRouter(responder=answers("Snow."))
            self.resolve = self.inner.resolve

        def complete(self, request):
            return self.inner.complete(request)

    functai.configure(lm="gpt-4.1-mini", client=Plain())
    assert list(haiku.stream("snow")) == ["Snow."]


def test_a_model_that_refuses_to_stream_answers_whole(streaming):
    r = streaming(responder=answers("Snow."))
    r.stream = lambda request: (_ for _ in ()).throw(lm15.UnsupportedFeatureError("no streaming here"))
    s = haiku.stream("snow")
    assert list(s) == ["Snow."] and s.result == "Snow."


def test_a_layout_that_reads_replies_whole_shows_fields_at_the_end(streaming):
    streaming(responder=lambda req: '{"reasoning": "ten times 0.40", "result": 4.0}')
    s = solve.using(adapter="json").stream("10 pencils?")
    texts = [e for e in s.events() if isinstance(e, Text)]
    assert [(e.field, e.text) for e in texts] == [("reasoning", "ten times 0.40"), ("result", "4.0")]


def test_thinking_no_output_reads_is_shown_as_thinking(streaming):
    streaming(responder=lambda req: [lm15.ThinkingPart("Short poem, five-seven-five."), lm15.TextPart(XML.format("Snow."))])
    s = haiku.stream("snow")
    thinking = "".join(e.text for e in s.events() if isinstance(e, Thinking))
    assert thinking == "Short poem, five-seven-five." and s.result == "Snow."


# ------------------------------------------------------------------ watching while it is written


class Gate(FakeRouter):
    """Streams one piece each time the test says ``gate.step()``."""

    def __init__(self, reply):
        super().__init__(responder=lambda req: reply)
        self.go = queue.Queue()
        self.sent = threading.Event()
        self.closed = threading.Event()

    def _events(self, response):
        inner = super()._events(response)
        try:
            for event in inner:
                if event.type == "delta":
                    self.go.get(timeout=5)
                yield event
                if event.type == "delta":
                    self.sent.set()
        finally:
            self.closed.set()

    def step(self, n=1):
        for _ in range(n):
            self.sent.clear()
            self.go.put(None)
            assert self.sent.wait(5)
        time.sleep(0.05)


def test_the_answer_so_far_while_it_is_written():
    reply = XML.format('[{"name": "Ada Lovelace", "city": "London", "age": 36}, {"name": "Bo", "city": null, "age": 3}]')
    gate = Gate(reply)
    functai.configure(lm="gpt-4.1-mini", client=gate)
    s = contacts.stream("Ada and Bo")
    assert s.partial is None and s.text == "" and not s.done
    seen = []
    while not s.done:
        gate.go.put(None)
        time.sleep(0.005)
        if not s.done:
            seen.append(s.partial)
    final = [{"name": "Ada Lovelace", "city": "London", "age": 36}, {"name": "Bo", "city": None, "age": 3}]
    shown = [p for p in seen if p]
    assert len(shown) > 10
    for p in shown:                                        # every view so far agrees with the end
        assert len(p) <= 2
        for got, want in zip(p, final):
            assert set(got) <= set(want)
            assert got.get("name", "") == want["name"][:len(got.get("name", ""))]
            assert got.get("age", want["age"]) == want["age"]           # numbers only once complete
    assert {"name": "Ada Lov"} in shown or any(p[0].get("name", "").startswith("Ada L") for p in shown)
    assert s.result == [Contact("Ada Lovelace", "London", 36), Contact("Bo", None, 3)] == s.partial


def test_closing_cancels_the_call_and_the_log_says_so(tmp_path):
    gate = Gate(XML.format("Snow on the cedar, quiet as a held breath."))
    functai.configure(lm="gpt-4.1-mini", client=gate, log_calls=tmp_path)
    with haiku.stream("snow") as s:
        gate.step(7)                                           # "<result>\n" shows nothing, then pieces
        pieces = []
        for piece in s:
            pieces.append(piece)
            if len(pieces) == 2:
                s.close()
        assert len(pieces) >= 2                                # iteration ends quietly after close
    gate.go.put(None)
    assert gate.closed.wait(5)                                 # the provider stream was released
    with pytest.raises(Cancelled):
        s.result
    s._thread.join(5)
    [rec] = [json.loads(line) for f in tmp_path.rglob("*.jsonl") for line in f.read_text().splitlines()]
    assert rec["error"]["type"] == "Cancelled"
    assert rec["exchanges"][-1]["error"]["type"] == "Cancelled" and rec["exchanges"][-1]["streamed"]


def test_leaving_a_with_block_cancels_an_unfinished_call():
    gate = Gate(XML.format("Snow on the cedar."))
    functai.configure(lm="gpt-4.1-mini", client=gate)
    with haiku.stream("snow") as s:
        gate.step(1)
    gate.go.put(None)
    s._thread.join(5)
    with pytest.raises(Cancelled):
        s.result


def test_a_closed_module_stream_starts_no_new_call(streaming):
    started = threading.Event()
    release = threading.Event()

    def slow(x: str) -> str:
        started.set()
        release.wait(5)
        return x

    @ai
    def one(x: str) -> str:
        """Echo."""

    @module
    def pipeline(x: str) -> str:
        return one(slow(x))

    r = streaming(responder=answers("never"))
    s = pipeline.stream("x")
    assert started.wait(5)
    s.close()
    release.set()
    s._thread.join(5)
    with pytest.raises(Cancelled):
        s.result
    assert r.requests == []


def test_many_streams_at_once(streaming):
    import re
    streaming(responder=lambda req: XML.format(re.search(r"topic \d+", req.messages[-1].parts[0].text)[0][::-1]))
    streams = [haiku.stream(f"topic {i}") for i in range(20)]
    assert [s.result for s in streams] == [f"topic {i}"[::-1] for i in range(20)]
    assert all("".join(s) == s.result for s in streams)


def test_settings_of_the_caller_reach_the_stream(streaming):
    r = streaming(responder=answers("x"))
    with functai.configure(temperature=0.3):
        s = haiku.stream("snow")
    s.result
    assert r.requests[0].config.temperature == 0.3


# ------------------------------------------------------------------ async


def test_async_iteration_and_await(streaming):
    streaming(responder=answers("Snow on the cedar."))

    async def main():
        s = haiku.stream("snow")
        pieces = [p async for p in s]
        events = [e async for e in s.events()]
        return pieces, events, await s, await haiku.stream("more snow")

    pieces, events, value, again = asyncio.run(main())
    assert "".join(pieces) == value == again == "Snow on the cedar." and events[-1].kind == "done"


def test_async_consumer_cancelled_cancels_the_call():
    gate = Gate(XML.format("Snow on the cedar."))
    functai.configure(lm="gpt-4.1-mini", client=gate)
    holder = {}

    async def main():
        holder["s"] = s = haiku.stream("snow")
        task = asyncio.ensure_future(s._result_async())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    gate.go.put(None)
    holder["s"]._thread.join(5)
    with pytest.raises(Cancelled):
        holder["s"].result


# ------------------------------------------------------------------ JSON forms


def test_events_as_json_match_the_contract(streaming):
    jsonschema = pytest.importorskip("jsonschema")
    from referencing import Registry, Resource
    docs = [json.loads((CONTRACT / "schema" / f"{n}.schema.json").read_text()) for n in ("call", "event")]
    registry = Registry().with_resources([(d["$id"], Resource.from_contents(d)) for d in docs])
    validator = jsonschema.Draft202012Validator(docs[1], registry=registry)

    @ai(tools=[get_weather])
    def assistant_(question: str) -> Contact:
        """Answer."""

    call = lm15.ToolCallPart(id="c1", name="get_weather", input={"city": "Oslo"})
    streaming([call], "no tags", XML.format('{"name": "Ada", "city": "Oslo", "age": 36}'))
    s = assistant_.stream("Weather?")
    dicts = [e.to_dict() for e in s.events()]
    for d in dicts:
        errors = list(validator.iter_errors(d))
        assert not errors, (d, errors[0].message)
    assert {d["kind"] for d in dicts} == {"started", "tool_call", "tool_result", "retry", "text", "done"}
    assert dicts[-1]["value"] == {"name": "Ada", "city": "Oslo", "age": 36}
    json.dumps(dicts)
    failed = Failed(s.call_id, "x", ValueError("bad")).to_dict()
    assert failed["error"] == {"type": "ValueError", "message": "bad"} and not list(validator.iter_errors(failed))


# ------------------------------------------------------------------ the answer so far, read as JSON


@pytest.mark.parametrize("text, value", [
    ('', None),
    ('Snow', None),
    ('[', []),
    ('[{"na', [{}]),
    ('[{"name": "Ad', [{"name": "Ad"}]),
    ('[{"name": "Ada", "age": 3', [{"name": "Ada"}]),
    ('[{"name": "Ada", "age": 36', [{"name": "Ada"}]),
    ('[{"name": "Ada", "age": 36,', [{"name": "Ada", "age": 36}]),
    ('{"ok": tr', {}),
    ('{"ok": true, "none": null}', {"ok": True, "none": None}),
    ('{"q": "a \\"b\\" \\u00e9 \\', {"q": 'a "b" é '}),
    ('{"e": "\\ud83d', {"e": ""}),
    ('{"e": "\\ud83d\\ude00!"}', {"e": "😀!"}),
    ('```json\n{"a": [1, 2', {"a": [1]}),
    ('{"a": [1, 2]}\n```', {"a": [1, 2]}),
    ('{"n": -1.5e3}', {"n": -1500.0}),
])
def test_partial_json(text, value):
    assert partial_json(text) == value


def test_partial_json_refuses_what_is_not_json():
    with pytest.raises(ValueError):
        partial_json('{"a": nope}')
    with pytest.raises(ValueError):
        partial_json('{a: 1}')


def test_a_thinking_model_streams_its_thinking_as_the_reasoning(fake):
    fake(responder=lambda req: [lm15.ThinkingPart("Five, seven, five syllables."), lm15.TextPart(XML.format("Snow."))],
         provider="anthropic", lm="anthropic:claude-sonnet-4-5")
    s = haiku.using(module="cot").stream("snow")
    texts = [e for e in s.events() if isinstance(e, Text)]
    assert "".join(e.text for e in texts if e.field == "reasoning") == "Five, seven, five syllables."
    assert not any(isinstance(e, Thinking) for e in s.events())
    assert s.prediction.reasoning == "Five, seven, five syllables." and "".join(s) == "Snow."


def test_show_prints_the_call_as_it_is_written(streaming, capsys):
    call = lm15.ToolCallPart(id="c1", name="get_weather", input={"city": "Oslo"})

    @ai(tools=[get_weather])
    def assistant_(question: str) -> str:
        """Answer."""

    streaming("<reasoning>\nTen times 0.40.\n</reasoning>\n<result>\n4.0\n</result>",
              [call], XML.format("Sunny in Oslo."), "garbage", XML.format("Snow."))
    solve.stream("x").show()
    assistant_.stream("x").show()
    haiku.stream("x").show()
    assert capsys.readouterr().out == (
        "reasoning: Ten times 0.40.\nresult: 4.0\n"
        "→ get_weather(city='Oslo')\n← Sunny and 22C in Oslo.\nSunny in Oslo.\n"
        "[asked again: the reply could not be read (reply is missing pattern section(s): 'result'); asking again]\n"
        "Snow.\n")


def test_show_raises_what_the_call_raised(streaming, capsys):
    streaming("garbage")
    with pytest.raises(Exception, match="missing pattern"):
        haiku.using(retries=0).stream("x").show()


def test_a_failure_nobody_read_is_not_swallowed(streaming):
    import gc
    streaming("garbage")
    s = haiku.using(retries=0).stream("x")
    s.wait()
    with pytest.warns(RuntimeWarning, match="failed and nothing read its result"):
        del s
        gc.collect()
    streaming("garbage")
    seen = haiku.using(retries=0).stream("x").wait()
    with pytest.raises(Exception):
        seen.result
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        del seen
        gc.collect()
