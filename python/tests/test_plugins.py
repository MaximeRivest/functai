"""Plugins: hooks over tools, context and model choice, changes recorded as data, built-ins made with the
public hooks only. The contract is contract/plugins.md."""

import json
import re
import textwrap

import lm15
import pytest

import functai
from functai import Change, Plugin, PluginError, ai, calllog

XML = "<result>\n{}\n</result>"


def texts(request):
    return " ".join(getattr(p, "text", "") or "" for m in request.messages for p in m.parts)


def last_user(request):
    return re.sub(r"<[^>]+>", "", getattr(request.messages[-1].parts[0], "text", "")).strip()


@ai
def tutor(message: str) -> str:
    """Tutor a student."""


def records(folder):
    return sorted(calllog.read(folder)[0], key=lambda r: r["started"])


# ------------------------------------------------------------------ the plugin itself


def test_a_plugin_is_named_versioned_and_checked():
    ext = Plugin("modes", version="1.2.0")

    @ext.before_call
    def f(call):
        return None

    assert ext.describe() == {"name": "modes", "version": "1.2.0", "api": 1, "description": "", "hooks": ["before_call"]}
    for bad in ("Modes", "", "x y", 3):
        with pytest.raises(PluginError) as err:
            Plugin(bad)
        assert err.value.code == "plugin-name"
    with pytest.raises(PluginError) as err:
        Plugin("future", api=2)
    assert err.value.code == "plugin-api"
    with pytest.raises(PluginError) as err:
        ext.on("before_agent_start", f)                  # a typo, or another product's hook: refused, not ignored
    assert err.value.code == "plugin-hook"
    with pytest.raises(TypeError):
        functai.configure(plugins=ext)                # a list


def test_a_plugin_is_loaded_from_a_file(tmp_path, fake):
    path = tmp_path / "loud.py"
    path.write_text(textwrap.dedent('''
        import functai
        plugin = functai.Plugin("loud", version="1.0.0")

        @plugin.before_call
        def shout(call):
            return functai.Change(sections=["ANSWER IN CAPITALS."])
        '''))
    r = fake(responder=lambda req: XML.format("OK"))
    with functai.configure(plugins=[str(path)]):
        tutor("hi")
    assert "ANSWER IN CAPITALS." in str(r.requests[-1].system)
    with pytest.raises(PluginError):
        functai.load_plugin(tmp_path / "missing.py")
    (tmp_path / "empty.py").write_text("x = 1\n")
    with pytest.raises(PluginError) as err:
        functai.load_plugin(tmp_path / "empty.py")
    assert err.value.code == "plugin-load"


# ------------------------------------------------------------------ before_call: sections, model, settings, tools


def test_before_call_changes_are_data_and_recorded(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("ok"))
    ext = Plugin("modes", version="2.0.0")
    seen = []

    @ext.before_call
    def careful(call):
        seen.append((call.function, call.lm, call.path))
        return Change(sections=["Be careful."], lm="gpt-4.1-nano", settings={"temperature": 0.2})

    with functai.configure(plugins=[ext], log_calls=tmp_path):
        tutor("hi")
        rendered = tutor.render("again")
    assert seen[0] == ("tutor", "gpt-4.1-mini", "tutor")
    assert r.requests[0].model == "gpt-4.1-nano" and r.requests[0].config.temperature == 0.2
    assert "Be careful." in str(r.requests[0].system) and "Be careful." in str(rendered.system)
    [rec] = records(tmp_path)
    assert "sections" not in rec and rec["model"] == "gpt-4.1-nano"   # the host's shaping: in changes only
    assert rec["changes"] == [{"plugin": "modes", "version": "2.0.0", "hook": "before_call",
                               "change": {"sections": ["Be careful."], "lm": "gpt-4.1-nano",
                                          "settings": {"temperature": 0.2}}}]
    assert rec["program"]["version"] == tutor.version            # the program is unchanged: its version too


def test_before_call_may_replace_the_instruction(fake):
    r = fake(responder=lambda req: XML.format("ok"))
    mode = Plugin("mode")
    mode.before_call(lambda call: Change(instruction="You are a pirate tutor.", sections=["Say arr."]))
    with functai.configure(plugins=[mode]):
        tutor("hi")
    system = str(r.requests[0].system)
    assert "You are a pirate tutor." in system and "Say arr." in system and "Tutor a student." not in system


def test_asked_again_a_row_takes_its_conversation_not_the_host_s_old_shaping(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("ok"))
    functai.configure(log_calls=tmp_path)
    old_mode = Plugin("old-mode")
    old_mode.before_call(lambda call: Change(instruction="OLD HOST INSTRUCTION"))
    chat = tutor.conversation(plugins=[old_mode])
    chat("first")
    functai.rate(chat.predict("second").call_id, "right")
    rows = functai.rated(tutor).collect().to_dicts()
    asked = len(r.requests)
    functai.evaluate(tutor, rows)
    sent = r.requests[asked]
    assert "OLD HOST INSTRUCTION" not in str(sent.system) and "Tutor a student." in str(sent.system)
    assert "first" in texts(sent)                                   # its conversation is shown again


def test_before_call_offers_a_subset_of_tools(fake):
    @functai.tool(effects="reads")
    def look(x: str) -> str:
        """Look."""
        return "seen"

    @functai.tool(effects="changes")
    def write(x: str) -> str:
        """Write."""
        return "written"

    @ai(tools=[look, write])
    def agent(request: str) -> str:
        """Help."""

    r = fake(XML.format("done"))
    read_only = Plugin("read-only")

    @read_only.before_call
    def only_reading(call):
        return Change(tools=[t for t in call.all_tools if t == "look"])

    with functai.configure(plugins=[read_only]):
        agent("go")
    assert [t.name for t in r.requests[0].tools] == ["look"]
    bad = Plugin("bad")
    bad.before_call(lambda call: Change(tools=["rm"]))
    with functai.configure(plugins=[bad]):
        with pytest.raises(PluginError) as err:
            agent("go")
    assert err.value.code == "plugin-change" and "rm" in str(err.value)


def test_a_hook_may_change_only_what_it_can(fake):
    fake(responder=lambda req: XML.format("ok"))
    ext = Plugin("confused")
    ext.before_call(lambda call: Change(block="no"))
    with functai.configure(plugins=[ext]):
        with pytest.raises(PluginError) as err:
            tutor("hi")
    assert err.value.code == "plugin-change" and err.value.hook == "before_call"
    odd = Plugin("odd")
    odd.before_call(lambda call: "not a change")
    with functai.configure(plugins=[odd]):
        with pytest.raises(PluginError):
            tutor("hi")


def test_a_failing_transform_stops_the_call(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("ok"))
    ext = Plugin("redact")

    @ext.before_call
    def broken(call):
        raise RuntimeError("bug")

    with functai.configure(plugins=[ext], log_calls=tmp_path):
        with pytest.raises(PluginError) as err:
            tutor("secret")
    assert err.value.code == "plugin-failed" and err.value.plugin == "redact" and not r.requests
    [rec] = records(tmp_path)
    assert rec["error"]["code"] == "plugin-failed"


# ------------------------------------------------------------------ order and layers


def test_order_program_first_host_last_once_each(fake):
    fake(responder=lambda req: XML.format("ok"))
    order = []

    def ext(name):
        e = Plugin(name)
        e.before_call(lambda call: order.append((name, list(call.sections))) or Change(sections=[name]))
        return e

    host, block, own = ext("host"), ext("block"), ext("own")

    @ai(plugins=[own, host])
    def f(x: str) -> str:
        """F."""

    functai.configure(plugins=[host])
    with functai.configure(plugins=[block]):
        f("x")
    assert order == [("own", []), ("block", ["own"]), ("host", ["own", "block"])]   # host once, last
    order.clear()
    with functai.configure(program_plugins=False):
        f("x")
    assert [n for n, _ in order] == ["host"]                  # a host may refuse a program's own


# ------------------------------------------------------------------ tool_call, tool_result: approval is one of them


@functai.tool(effects="changes")
def send(to: str, text: str) -> str:
    """Send a message."""
    return f"sent to {to}"


@ai(tools=[send])
def clerk(request: str) -> str:
    """Help."""


def this_turn(req):
    """The tool results of the turn being asked (earlier turns of a conversation hold their own)."""
    asks = [i for i, m in enumerate(req.messages) if m.role == "user"
            and any(type(p).__name__ == "TextPart" for p in m.parts)]
    return [p for m in req.messages[asks[-1] if asks else 0:] for p in m.parts
            if type(p).__name__ == "ToolResultPart"]


def one_call(name="send", args=None):
    def respond(req):
        done = this_turn(req)
        if not done:
            return [lm15.ToolCallPart(id="c1", name=name, input=args or {"to": "ana", "text": "hi"})]
        c = done[-1].content
        return XML.format("".join(getattr(x, "text", "") for x in c) if isinstance(c, (list, tuple)) else str(c))
    return respond


def test_tool_call_changes_input_or_blocks_and_tool_result_rewrites(fake, tmp_path):
    fake(responder=one_call())
    policy = Plugin("policy", version="1.0.0")

    @policy.tool_call
    def redirect(tool):
        if tool.input["to"] == "ana":
            return Change(inputs={"to": "team"})

    @policy.tool_result
    def tag(result):
        return Change(output=result.output + " (checked)")

    with functai.configure(plugins=[policy], log_calls=tmp_path):
        assert clerk("message ana") == "sent to team (checked)"
    [rec] = records(tmp_path)
    assert [c["hook"] for c in rec["changes"]] == ["tool_call", "tool_result"]
    assert rec["changes"][0]["change"] == {"inputs": {"to": "team"}}

    stop = Plugin("stop")
    stop.tool_call(lambda tool: Change(block="not on weekends"))
    with functai.configure(plugins=[stop]):
        assert clerk("go") == "This call was blocked (stop): not on weekends"


def test_a_failing_tool_check_blocks_the_tool(fake):
    fake(responder=one_call())
    ran = []
    flaky = Plugin("flaky")

    @flaky.tool_call
    def boom(tool):
        raise KeyError("oops")

    @functai.tool(effects="changes")
    def act(to: str, text: str) -> str:
        """Act."""
        ran.append(1)
        return "done"

    @ai(tools=[act])
    def doer(request: str) -> str:
        """Do."""

    fake(responder=one_call("act"))
    with functai.configure(plugins=[flaky]):
        out = doer("go")
    assert ran == [] and "blocked (flaky)" in out


def test_any_plugin_can_ask_a_person_and_the_turn_waits(fake):
    fake(responder=one_call())
    big = Plugin("big-sends")

    @big.tool_call
    def ask_for_team(tool):
        if tool.input.get("to") == "everyone":
            tool.ask("this goes to everyone")

    chat = clerk.conversation(plugins=[big])
    assert chat("tell ana") == "sent to ana"                     # not asked
    fake(responder=one_call(args={"to": "everyone", "text": "hi"}))
    with pytest.raises(functai.Waiting) as err:
        chat("tell everyone")
    [a] = err.value.approvals
    assert (a.plugin, a.question, a.name) == ("big-sends", "this goes to everyone", "send")
    assert chat.turns[-1].approve(a) == "sent to everyone"


def test_approve_is_the_approval_extension_and_runs_last(fake):
    fake(responder=one_call())
    seen = []
    rewrite = Plugin("rewrite")

    @rewrite.tool_call
    def to_team(tool):
        return Change(inputs={"to": "team"})

    def ask(approval):
        seen.append(approval.input)
        return True

    with functai.configure(plugins=[rewrite]):
        assert clerk.using(approve=ask)("go") == "sent to team"
    assert seen == [{"to": "team", "text": "hi"}]                 # the rule saw what would run


# ------------------------------------------------------------------ the escape hatch


def test_replacing_a_request_is_allowed_and_marks_the_call(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("ok"))
    import dataclasses
    raw = Plugin("raw")

    @raw.request
    def cheaper(event):
        return dataclasses.replace(event.request, model="gpt-4.1-nano")

    with functai.configure(plugins=[raw], log_calls=tmp_path):
        tutor("hi")
    assert r.requests[0].model == "gpt-4.1-nano"
    [rec] = records(tmp_path)
    assert rec["replayable"] is False and "request_hash" not in rec["exchanges"][0]
    assert rec["changes"] == [{"plugin": "raw", "version": "0.0.0", "hook": "request",
                               "change": {"request": "replaced"}}]


# ------------------------------------------------------------------ conversations: turn_start, context, entries


def test_turn_start_and_context_change_a_turn_and_resuming_shows_the_same(fake, tmp_path):
    r = fake(responder=lambda req: XML.format(f"{len(req.messages)}"))
    ext = Plugin("tidy")

    @ext.turn_start
    def strip(event):
        return Change(inputs={"message": event.inputs["message"].strip()})

    @ext.context
    def only_last(event):
        return Change(keep=[t.id for t in event.turns[-1:]], sections=[f"{len(event.turns)} earlier turns"])

    chat = tutor.conversation("ctx", store=tmp_path, plugins=[ext])
    chat("  one  ")
    chat("two")
    chat("three")
    assert chat.turns[0].inputs == {"message": "one"}
    assert len(r.requests[2].messages) == 3 and "2 earlier turns" in str(r.requests[2].system)
    record = chat.turns[-1]._st.record
    assert record["context"] == {"turns": [chat.turns[1].id], "without": {}, "sections": ["2 earlier turns"]}
    assert [c["hook"] for c in record["changes"]] == ["turn_start", "context"]


def test_entries_follow_the_branch(fake):
    fake(responder=lambda req: XML.format("ok"))
    notes = Plugin("notes")

    @notes.turn_end
    def note(event):
        event.remember("seen", {"message": event.inputs["message"]})

    chat = tutor.conversation(plugins=[notes])
    chat("a")
    chat("b")
    other = chat.continue_from(chat.turns[0])
    other("c")
    assert [e["data"]["message"] for e in chat.entries("notes", "seen", branch=chat.turns[-1].id)] == ["a", "b"]
    assert [e["data"]["message"] for e in other.entries("notes", "seen")] == ["a", "c"]


def test_a_failing_turn_end_changes_nothing(fake):
    fake(responder=lambda req: XML.format("ok"))
    bad = Plugin("bad-end")
    bad.turn_end(lambda event: 1 / 0)
    chat = tutor.conversation(plugins=[bad])
    with pytest.warns(UserWarning, match="bad-end"):
        assert chat("hi") == "ok"
    assert chat.turns[-1].state == "done"


# ------------------------------------------------------------------ compaction (a built-in, public hooks only)


def test_compaction_summarizes_older_turns_per_branch(fake, tmp_path):
    def respond(req):
        if "summary of a conversation" in str(req.system) or "Summarize" in str(req.system):
            return XML.format("SUMMARY: " + last_user(req)[:40].replace("\n", " "))
        return XML.format(f"answer {len(req.messages)}")

    r = fake(responder=respond)
    functai.configure(log_calls=tmp_path)

    @ai
    def summarize(earlier_summary: str, new_turns: list[dict]) -> str:
        """Summarize the conversation so far."""

    chat = tutor.conversation("long", plugins=[functai.compaction(keep=2, every=2, summarize=summarize)])
    for i in range(5):
        chat(f"message {i}")
    # after turn 4 (4 open turns >= keep + every) the first two were folded
    [entry] = chat.entries("compaction", "summary")
    assert entry["data"]["turns"] == 2 and entry["data"]["through"] == chat.turns[1].id
    last = r.requests[-1]
    assert "SUMMARY" in str(last.system) and "message 0" not in texts(last) and "message 2" in texts(last)
    rec = next(c for c in records(tmp_path) if c["id"] == chat.turns[-1].id)
    assert rec["sections"][0].startswith("Earlier in this conversation (2 turns, summarized)")
    assert len(calllog.saw(rec["id"], records(tmp_path))) == 2       # only the turns after the summary
    other = chat.continue_from(chat.turns[0])                         # a branch from before the summary
    other("another path")
    assert "SUMMARY" not in str(r.requests[-1].system)


def test_a_rated_turn_is_asked_again_with_its_summary(fake, tmp_path):
    r = fake(responder=lambda req: XML.format("SUM" if "Summarize" in str(req.system) else "ok"))
    functai.configure(log_calls=tmp_path)

    @ai
    def summarize(earlier_summary: str, new_turns: list[dict]) -> str:
        """Summarize the conversation so far."""

    chat = tutor.conversation(plugins=[functai.compaction(keep=1, every=1, summarize=summarize)])
    for i in range(4):
        chat(f"m{i}")
    functai.rate(chat.turns[-1].id, "right")
    [row] = functai.rated(tutor).collect().to_dicts()
    assert row["sections"] and row["sections"][0].endswith("SUM")
    asked = len(r.requests)
    functai.evaluate(tutor, [row])
    assert "SUM" in str(r.requests[asked].system)                     # the same summary, nothing re-run
    assert len(r.requests) == asked + 1


# ------------------------------------------------------------------ delegation (a built-in)


def test_delegate_runs_a_program_in_its_own_conversation_per_branch(fake):
    def respond(req):
        if "Look things up" in str(req.system):
            return XML.format(f"found ({len(req.messages)} messages)")
        done = this_turn(req)
        if not done:
            return [lm15.ToolCallPart(id="c1", name="researcher", input={"question": "where?"})]
        c = done[-1].content
        return XML.format("".join(getattr(x, "text", "") for x in c) if isinstance(c, (list, tuple)) else str(c))

    fake(responder=respond)

    @ai
    def research(question: str) -> str:
        """Look things up in the notes."""

    researcher = functai.delegate(research, name="researcher")
    assert researcher.effects == "reads" and researcher.__name__ == "researcher"

    @ai(tools=[researcher])
    def assistant(request: str) -> str:
        """Help, asking the researcher when needed."""

    chat = assistant.conversation("main")
    assert chat("first") == "found (1 messages)"
    assert chat("second") == "found (3 messages)"                     # it remembers what it was asked before
    sub = research.conversation("main.researcher")
    assert len(sub.all_turns()) == 2
    other = chat.continue_from(chat.turns[0])
    assert other("elsewhere") == "found (3 messages)"                # this branch: one earlier delegation, not two
    assert research("plain") == "found (1 messages)"                  # the program itself is unchanged


def test_what_plugins_write_passes_the_schemas(fake, tmp_path):
    from contract_support import assert_valid, validator

    def respond(req):
        if "Summarize" in str(req.system):
            return XML.format("SUM")
        return one_call()(req)

    fake(responder=respond)
    functai.configure(log_calls=tmp_path / "log")
    ask = Plugin("ask-first")
    ask.tool_call(lambda tool: tool.ask("sure?") and None)
    ask.before_call(lambda call: Change(sections=["Be brief."]))
    ask.turn_start(lambda turn: Change(inputs={"request": turn.inputs["request"] + "!"}))

    @ai
    def summarize(earlier_summary: str, new_turns: list[dict]) -> str:
        """Summarize the conversation so far."""

    chat = clerk.conversation("schemas", store=tmp_path / "store",
                              plugins=[ask, functai.compaction(keep=1, every=1, summarize=summarize)])
    for i in range(3):
        with pytest.raises(functai.Waiting):
            chat(f"send {i}")
        chat.turns[-1].approve()
    conv, event, call = validator("conversation"), validator("event"), validator("call")
    lines = [json.loads(x) for x in (tmp_path / "store" / "conversations" / "schemas.jsonl").read_text().splitlines()]
    for rec in lines:
        assert_valid(conv, rec, rec["kind"])
    assert {"entry", "approval", "waiting"} <= {r["kind"] for r in lines}
    assert any("context" in r for r in lines if r["kind"] == "turn")
    for path in (tmp_path / "store" / "trees").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            assert_valid(event, e, e["kind"])
    recs = calllog.read(tmp_path / "log")[0]
    for rec in recs:
        assert_valid(call, rec, rec["program"]["name"])
    assert any(r.get("sections") for r in recs) and any(r.get("changes") for r in recs)


@pytest.mark.parametrize("adapter", [None, "xml", "chat", "json"])
def test_every_built_in_layout_sends_plugin_sections(fake, adapter):
    fake("unused")
    mark = Plugin("mark")
    mark.before_call(lambda call: Change(sections=["SECTION-MARK"]))
    fn = tutor.using(adapter=adapter) if adapter else tutor
    with functai.configure(plugins=[mark]):
        request = fn.render("hi")                    # exactly what a call would send
    assert "SECTION-MARK" in str(request)


def test_a_template_that_never_writes_the_instruction_refuses_a_changed_one(fake):
    from functai import system, user
    r = fake(responder=lambda req: "Paris")
    mark = Plugin("mark")
    mark.before_call(lambda call: Change(sections=["SECTION-MARK"]))

    @ai(template=[system("You answer capitals."), user("{country}")])
    def capital(country: str) -> str:
        """Capital."""

    with functai.configure(plugins=[mark]):
        with pytest.raises(PluginError) as err:
            capital("France")                         # never sent without what its record would claim
        assert err.value.code == "plugin-change" and "{instruction}" in str(err.value) and not r.requests
        with pytest.raises(PluginError):
            capital.render("France")
    assert capital("France") == "Paris"               # without plugins: the template as written

    @ai(template=[system("You answer capitals. {instruction}"), user("{country}")])
    def placed(country: str) -> str:
        """Capital."""

    with functai.configure(plugins=[mark]):
        placed("France")
    assert "SECTION-MARK" in str(r.requests[-1].system)
