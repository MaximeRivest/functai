"""Long agents and pipelines: tools that answer with pictures, notes from code,
the step hook and its built-in plugins (limits, keep_pictures), the call log
in a store of one's own, and results that come later.

Offline: a fake provider. The contract is contract/tools.md, streaming.md,
plugins.md, calls.md and conversations.md."""

import base64

import lm15
import pytest
from lm15.types import image

import functai
from functai import ai, module
from functai.stores import MemoryConversations

XML = "<result>\n{}\n</result>"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


def results(req):
    return [p for m in req.messages for p in m.parts if type(p).__name__ == "ToolResultPart"]


def script(*steps, answer=lambda req: "done"):
    """A model that asks for these tool calls, one per reply, then answers."""
    def respond(req):
        n = len(results(req))
        if n < len(steps) and req.tools:                             # offered no tool, the model answers
            name, args = steps[n]
            return [lm15.ToolCallPart(id=f"c{n + 1}", name=name, input=args)]
        return XML.format(answer(req))
    return respond


@functai.tool(effects="reads")
def look(page: str):
    """A screenshot of a page, with its title."""
    return [f"page {page}", image(data=PNG, media_type="image/png")]


@ai(tools=[look])
def browse(task: str) -> str:
    """Look at pages to do the task."""


# ------------------------------------------------------------------ pictures from tools


def test_a_tool_answers_with_a_picture(fake):
    r = fake(responder=script(("look", {"page": "home"})))
    s = browse.stream("find the price")
    events = list(s.events())
    assert s.result == "done"
    [part] = results(r.requests[-1])
    kinds = [type(c).__name__ for c in part.content]
    assert kinds == ["TextPart", "ImagePart"]                         # the model sees the picture itself
    assert part.content[1].data == PNG
    [res] = [e for e in events if e.kind == "tool_result"]
    assert res.output == "page home"                                  # the text, for readers that show text
    assert res.parts == [{"type": "text", "text": "page home"},
                         {"type": "image", "media_type": "image/png", "data": PNG}]
    assert res.to_dict()["parts"] == res.parts
    kept = res.to_dict(content=False, keep={"inputs": {"task": False}, "outputs": {"result": True}})
    assert "parts" not in kept and "output" not in kept


def test_a_picture_from_a_file_is_kept_whole(fake, tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\nfile")

    @functai.tool(effects="reads")
    def screenshot() -> object:
        """The screen."""
        return image(path=str(shot))

    @ai(tools=[screenshot])
    def watch(task: str) -> str:
        """Watch."""

    r = fake(responder=script(("screenshot", {})))
    assert watch("look") == "done"
    [part] = results(r.requests[-1])
    assert part.content[0].data == base64.b64encode(shot.read_bytes()).decode() and part.content[0].path is None


def test_a_text_or_a_value_is_text_as_before(fake):
    @functai.tool(effects="reads")
    def facts() -> dict:
        """Facts."""
        return {"a": 1, "list": ["x"]}

    @ai(tools=[facts])
    def ask(q: str) -> str:
        """Ask."""

    r = fake(responder=script(("facts", {})))
    s = ask.stream("q")
    events = list(s.events())
    [res] = [e for e in events if e.kind == "tool_result"]
    assert res.output == '{"a": 1, "list": ["x"]}' and res.parts is None
    assert "parts" not in res.to_dict()
    assert s.result == "done"


def test_a_plugin_sees_and_changes_a_picture_result(fake):
    seen = []
    p = functai.Plugin("caption")

    @p.tool_result
    def caption(t):
        seen.append(t.output)
        return functai.Change(output=[*t.output, "(the price is at the top right)"])

    r = fake(responder=script(("look", {"page": "home"})))
    assert browse.using(plugins=[p])("find") == "done"
    assert seen[0][1]["type"] == "image"
    [part] = results(r.requests[-1])
    assert [type(c).__name__ for c in part.content] == ["TextPart", "ImagePart", "TextPart"]


def test_a_turn_keeps_a_picture_result_and_reuses_it(fake):
    calls = []

    @functai.tool(effects="changes")
    def click(where: str):
        """Click, and see the screen."""
        calls.append(where)
        return ["clicked", image(data=PNG, media_type="image/png")]

    @ai(tools=[click])
    def agent(task: str) -> str:
        """Do it."""

    store = MemoryConversations()
    fake(responder=script(("click", {"where": "buy"})))
    chat = agent.conversation(store=store, approve=["click"])
    with pytest.raises(functai.Waiting):
        chat("buy it")
    turn = chat.head
    turn.approve()
    assert calls == ["buy"]
    recs = [r for r in store.read(chat.id, 0) if r.get("kind") == "tool" and r.get("state") == "done"]
    assert recs[0]["output"][1] == {"type": "image", "media_type": "image/png", "data": PNG}


# ------------------------------------------------------------------ the step hook


def test_the_step_hook_sees_each_step_and_can_wrap_up(fake):
    seen = []
    p = functai.Plugin("wrap")

    @p.step
    def wrap(s):
        seen.append((s.step, s.max_steps, [r.invocation for r in s.results], list(s.tools), s.spent.requests))
        if s.step == 3:
            return functai.Change(tools=[], reminder="Answer now.")

    r = fake(responder=script(*[("look", {"page": str(i)}) for i in range(10)]))
    assert browse.using(plugins=[p], max_steps=5)("find") == "done"
    assert seen == [(2, 5, [1], ["look"], 1), (3, 5, [1, 2], ["look"], 2)]
    last = r.requests[-1]
    assert not last.tools                                            # no tool offered: the model must answer
    texts = [c.text for c in results(last)[-1].content if type(c).__name__ == "TextPart"]
    assert texts[-1] == "Answer now."                                # the reminder follows the last result


def test_earlier_results_are_shown_otherwise_and_the_record_says_so(fake, tmp_path):
    p = functai.Plugin("forget", version="2.0.0")

    @p.step
    def forget(s):
        old = [r for r in s.results[:-1] if r.pictures]
        if old:
            return functai.Change(replace={r.invocation: "(an earlier screenshot)" for r in old})

    r = fake(responder=script(*[("look", {"page": str(i)}) for i in range(3)]))
    with functai.configure(log_calls=tmp_path):
        assert browse.using(plugins=[p])("find") == "done"
    pictures = [sum(type(c).__name__ == "ImagePart" for p_ in results(q) for c in p_.content) for q in r.requests]
    assert pictures == [0, 1, 1, 1]                                  # only the newest picture is sent
    rec = [x for x in functai.calllog.read(tmp_path)[0] if x["program"]["name"] == "browse"][0]
    changes = [c for c in rec["changes"] if c["hook"] == "step"]
    assert changes[0] == {"plugin": "forget", "version": "2.0.0", "hook": "step",
                          "change": {"replace": {"1": "(an earlier screenshot)"}}}
    assert all("request_hash" in ex for ex in rec["exchanges"])      # still replayable: nothing was rewritten
    tool_steps = [st for st in rec["steps"] if st["kind"] == "tool"]
    assert tool_steps[0]["output"][1]["type"] == "image"             # the turn keeps what the tool gave


def test_a_step_handler_stops_the_call(fake):
    p = functai.Plugin("brake")
    p.step(lambda s: functai.Change(stop="too many pages") if s.step >= 3 else None)
    fake(responder=script(*[("look", {"page": str(i)}) for i in range(10)]))
    with pytest.raises(functai.PluginError) as err:
        browse.using(plugins=[p])("find")
    assert err.value.code == "plugin-stopped" and "too many pages" in str(err.value)


def test_a_step_change_that_does_not_fit_is_refused(fake):
    p = functai.Plugin("bad")
    p.step(lambda s: functai.Change(tools=["fly"]))
    fake(responder=script(("look", {"page": "a"}), ("look", {"page": "b"})))
    with pytest.raises(functai.PluginError) as err:
        browse.using(plugins=[p])("find")
    assert err.value.code == "plugin-change"
    with pytest.raises(functai.PluginError):
        functai.Plugin("x").on("step", lambda s: functai.Change(sections=["no"]))  # registering is fine...
        p2 = functai.Plugin("y")
        p2.step(lambda s: functai.Change(sections=["no"]))
        fake(responder=script(("look", {"page": "a"})))
        browse.using(plugins=[p2])("find")                             # ...a field step does not take is not


# ------------------------------------------------------------------ limits and keep_pictures


def test_limits_wrap_up_at_the_last_step_instead_of_failing(fake):
    r = fake(responder=script(*[("look", {"page": str(i)}) for i in range(10)]))
    assert browse.using(plugins=[functai.limits(seconds=3600)], max_steps=3)("find") == "done"
    assert len(r.requests) == 3 and not r.requests[-1].tools
    last = [c.text for c in results(r.requests[-1])[-1].content if type(c).__name__ == "TextPart"][-1]
    assert last == functai.builtins.LAST_STEP


def test_a_token_limit_wraps_up_then_stops_the_next_call(fake):
    r = fake(responder=script(*[("look", {"page": str(i)}) for i in range(10)]))
    limit = functai.limits(tokens=1)                                  # reached after the first request
    assert browse.using(plugins=[limit])("find") == "done"
    assert len(r.requests) == 2 and not r.requests[-1].tools
    reminder = [c.text for c in results(r.requests[-1])[-1].content if type(c).__name__ == "TextPart"][-1]
    assert reminder.startswith("This task has reached its limit (1 tokens)")

    @module
    def twice(task: str) -> str:
        return browse(task) + browse(task)

    with pytest.raises(functai.PluginError) as err, functai.configure(plugins=[limit]):
        twice("find")
    assert err.value.code == "plugin-stopped" and "limit reached: 1 tokens" in str(err.value)


def test_a_cost_limit_needs_prices_and_counts_unpriced_models_as_reached(fake):
    with pytest.raises(ValueError):
        functai.limits(cost=1.0)
    with pytest.raises(ValueError):
        functai.limits()
    fake(responder=script(("look", {"page": "a"})))
    limit = functai.limits(cost=1.0, prices={"some-other-model": {"input_per_million": 1, "output_per_million": 2}})
    with pytest.raises(functai.PluginError) as err:
        browse.using(plugins=[limit])("find")                         # gpt-4.1-mini has no price: refused up front
    assert "no price is known for gpt-4.1-mini" in str(err.value)
    flat = {"input_per_million": 1, "output_per_million": 2, "cache_read_per_million": 1,
            "cache_write_per_million": 1}
    priced = functai.limits(cost=1000.0, prices={"gpt-4.1-mini": flat}, then="stop")
    assert browse.using(plugins=[priced])("find") == "done"


def test_spent_cost_is_exact_or_nothing():
    from functai.plugins import Spent
    s = Spent([("m", {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000,
                      "cache_read_tokens": 0, "cache_write_tokens": 0})], 1.0)
    assert s.cost({"m": {"input_per_million": 3, "output_per_million": 15}}) == 18.0
    s2 = Spent([("m", {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000})], 1.0)
    assert s2.cost({"m": {"input_per_million": 3, "output_per_million": 15}}) is None   # cache counts unknown
    assert s.cost({"other": {"input_per_million": 3, "output_per_million": 15}}) is None
    assert Spent([("m", {"output_tokens": 5})], 0).cost({"m": {"input_per_million": 3, "output_per_million": 15}}) is None
    assert s.tokens == 2_000_000 and Spent([], 0).tokens == 0


def test_keep_pictures_shows_only_the_newest(fake):
    r = fake(responder=script(*[("look", {"page": str(i)}) for i in range(5)]))
    assert browse.using(plugins=[functai.keep_pictures(2)])("find") == "done"
    counts = [sum(type(c).__name__ == "ImagePart" for p_ in results(q) for c in p_.content) for q in r.requests]
    assert counts == [0, 1, 2, 2, 2, 2]
    gone = [c.text for c in results(r.requests[-1])[0].content if type(c).__name__ == "TextPart"]
    assert gone == ["page 0", functai.builtins.PICTURE_GONE]


def test_keep_pictures_drops_in_batches_for_the_prompt_cache(fake):
    r = fake(responder=script(*[("look", {"page": str(i)}) for i in range(7)]))
    assert browse.using(plugins=[functai.keep_pictures(2, at_once=3)], max_steps=10)("find") == "done"
    counts = [sum(type(c).__name__ == "ImagePart" for p_ in results(q) for c in p_.content) for q in r.requests]
    assert counts == [0, 1, 2, 3, 4, 2, 3, 4]


# ------------------------------------------------------------------ the call log in a store of one's own


class ListStore:
    def __init__(self):
        self.lines = []

    def append(self, record):
        import json
        self.lines.append(json.dumps(record))                         # what a database would keep: JSON

    def read(self):
        import json
        return [json.loads(x) for x in self.lines]


def test_calls_go_to_a_store_and_are_read_back_rated(fake):
    store = ListStore()
    fake(XML.format("billing"))

    @ai
    def team(message: str) -> str:
        """Which team answers this message."""

    with functai.configure(log_calls=store):
        pred = team.predict("I was charged twice.")
        functai.rate(pred, "right")
    recs = store.read()
    assert [("functai_call" in r, "functai_rating" in r) for r in recs] == [(True, False), (False, True)]
    rows = functai.rated(team, folder=store)
    assert len(rows) == 1
    assert len(functai.calls(team, folder=store)) == 1
    with pytest.raises(TypeError):
        functai.prune_calls(folder=store)


def test_a_store_that_fails_never_stops_a_call(fake):
    class Broken:
        def append(self, record):
            raise ConnectionError("database away")

        def read(self):
            return []

    fake(XML.format("billing"))

    @ai
    def team(message: str) -> str:
        """Which team answers this message."""

    with functai.configure(log_calls=Broken()), pytest.warns(UserWarning, match="could not log"):
        assert team("x") == "billing"
    with pytest.raises(TypeError):
        functai.configure(log_calls=object())


def test_memory_calls_keeps_records_in_order(fake):
    mem = functai.MemoryCalls()
    fake(XML.format("a"), XML.format("b"))

    @ai
    def f(x: str) -> str:
        """F."""

    with functai.configure(log_calls=mem):
        f("1")
        f("2")
    assert [r["outputs"]["result"] for r in mem.read()] == ["a", "b"]


# ------------------------------------------------------------------ results that come later


jobs = []


@functai.tool(effects="changes")
def start_harvest(url: str):
    """Visit a site in a cloud browser; its report comes back later."""
    jobs.append(url)
    return functai.later(f"harvest:{len(jobs)}")


@functai.tool(effects="reads")
def weather(city: str) -> str:
    """Today's weather."""
    return f"sunny in {city}"


@ai(tools=[start_harvest, weather])
def researcher(task: str) -> str:
    """Research the site."""


def last_result_text(req):
    return "".join(getattr(c, "text", "") for c in results(req)[-1].content)


@pytest.fixture
def harvests():
    jobs.clear()
    yield jobs


def test_a_turn_waits_for_a_later_result_and_goes_on_when_it_is_delivered(fake, harvests):
    store = MemoryConversations()
    r = fake(responder=script(("start_harvest", {"url": "a.com"}), ("weather", {"city": "Oslo"}),
                              answer=lambda req: "report: " + "".join(
                                  getattr(c, "text", "") for c in results(req)[0].content)))
    chat = researcher.conversation("c1", store=store)
    with pytest.raises(functai.Waiting) as err:
        chat("look at a.com")
    assert [x["key"] for x in err.value.later] == ["harvest:1"] and not err.value.approvals
    assert chat.head.state == "waiting" and harvests == ["a.com"]
    asked = len(r.requests)
    # the webhook, in another process: open the conversation, deliver
    again = researcher.conversation("c1", store=store)
    assert again.deliver("harvest:1", ["3 pages", image(data=PNG, media_type="image/png")]) == \
        "report: 3 pages"
    assert harvests == ["a.com"]                                      # the tool did not run again
    assert len(r.requests) - asked == 2                               # no model answer paid twice
    assert again.deliver("harvest:1", "late duplicate") is None       # a webhook sent twice changes nothing
    kinds = [(x["kind"], x.get("state")) for x in store.read("c1", 0) if x["kind"] in ("tool", "delivered")]
    assert kinds == [("tool", "started"), ("tool", "later"), ("delivered", None), ("tool", "done"),
                     ("tool", "done")]


def test_a_result_delivered_before_the_turn_waits_is_used_at_once(fake, harvests):
    store = MemoryConversations()
    fake(responder=script(("start_harvest", {"url": "b.com"}),
                          answer=lambda req: "report: " + last_result_text(req)))
    chat = researcher.conversation("c2", store=store)
    chat.deliver("harvest:1", "fast", resume=False)                   # the webhook came first
    assert chat("look") == "report: fast"


def test_a_later_result_outside_a_turn_is_refused(fake, harvests):
    fake(responder=script(("start_harvest", {"url": "c.com"})))
    with pytest.raises(functai.ConversationError) as err:
        researcher("look")
    assert err.value.code == "later-unavailable" and harvests == ["c.com"]


def test_a_waiting_turn_cannot_resume_before_its_result(fake, harvests):
    store = MemoryConversations()
    fake(responder=script(("start_harvest", {"url": "d.com"})))
    chat = researcher.conversation("c3", store=store)
    with pytest.raises(functai.Waiting):
        chat("look")
    with pytest.raises(functai.ConversationError) as err:
        chat.head.resume()
    assert err.value.code == "turn-state" and "harvest:1" in str(err.value)
    assert chat.head.later[0]["key"] == "harvest:1"


def test_a_later_event_is_shown(fake, harvests):
    store = MemoryConversations()
    fake(responder=script(("start_harvest", {"url": "e.com"})))
    chat = researcher.conversation("c4", store=store)
    s = chat.stream("look")
    events = []
    with pytest.raises(functai.Waiting):
        for e in s.events():
            events.append(e)
    later = [e for e in events if e.kind == "later"]
    assert [(e.key, e.name, e.invocation) for e in later] == [("harvest:1", "start_harvest", 1)]


# ------------------------------------------------------------------ notes


def test_code_says_what_happened(fake):
    fake(XML.format("ok"))

    @ai
    def write(x: str) -> str:
        """Write."""

    @module
    def pipeline(url: str) -> str:
        functai.note("screen captured", {"screen": 4})
        return write(url)

    s = pipeline.stream("a.com")
    assert s.result == "ok"
    [n] = [e for e in s.events() if e.kind == "note"]
    assert (n.name, n.data, n.function) == ("screen captured", {"screen": 4}, "pipeline")
    kept = n.to_dict(content=False, keep={"inputs": {"url": False}, "outputs": {"result": True}})
    assert kept["name"] == "screen captured" and "data" not in kept
    functai.note("nothing watches")                                   # outside any call: nothing happens
    with pytest.raises(TypeError):
        functai.note("")


def test_what_these_write_passes_the_schemas(fake, harvests, tmp_path):
    import json as _json
    from contract_support import assert_valid, validator
    fake(responder=script(("look", {"page": "a"}), ("start_harvest", {"url": "a.com"})))

    @ai(tools=[look, start_harvest])
    def agent(task: str) -> str:
        """Do it."""

    functai.configure(log_calls=tmp_path / "log")
    p = functai.keep_pictures(0)
    chat = agent.conversation("schemas", store=tmp_path / "store", plugins=[p, functai.limits(seconds=3600)])
    with pytest.raises(functai.Waiting):
        chat("go")
    assert chat.deliver("harvest:1", ["report", image(data=PNG, media_type="image/png")]) == "done"
    conv, event, call = validator("conversation"), validator("event"), validator("call")
    for line in (tmp_path / "store" / "conversations" / "schemas.jsonl").read_text().splitlines():
        rec = _json.loads(line)
        assert_valid(conv, rec, rec["kind"])
    kinds = set()
    for path in (tmp_path / "store" / "trees").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            e = _json.loads(line)
            kinds.add(e["kind"])
            assert_valid(event, e, e["kind"])
    assert {"later", "tool_result"} <= kinds
    for rec in functai.calllog.read(tmp_path / "log")[0]:
        assert_valid(call, rec, rec["program"]["name"])
