"""What a program runs on (contract/calls.md, *Setups*; design/18): its
setup, beside its version; what it may reach, and the wall that refuses it;
what code spent; the saved folder's setup and what differs between two."""

import json
import subprocess
import sys

import lm15
import pytest

import functai
from conftest import FakeRouter
from functai import calllog

XML = "<result>\n{}\n</result>"


@functai.ai
def mood(text: str) -> str:
    """How does the writer feel?"""
    ...


def look_here(page: str) -> str:
    """Look at a page."""
    return f"page {page}: in stock"


def look_elsewhere(page: str) -> str:
    """Look at a page."""
    return f"page {page}: sold out"


here = functai.tool(look_here, name="look", effects="reads", needs={"domains": ["shop.example"]})
elsewhere = functai.tool(look_elsewhere, name="look", effects="reads", needs={"domains": ["shop.example"]})


def agent(tool):
    @functai.ai(tools=[tool])
    def stock(item: str) -> str:
        """Is the item in stock?"""
        ...
    return stock


def records(folder):
    return [json.loads(line) for f in sorted(folder.rglob("*.jsonl")) for line in f.read_text().splitlines()]


def test_another_model_is_another_setup_of_the_same_version():
    with functai.configure(lm="openai:gpt-4.1-mini"):
        a, va = mood.setup, mood.version
    with functai.configure(lm="gpt-4.1-mini"):                    # the same route
        assert mood.setup == a
    with functai.configure(lm="anthropic:claude-haiku-4-5"):
        assert mood.setup != a and mood.version == va
    with functai.configure(lm="openai:gpt-4.1-mini", temperature=0.2):
        assert mood.setup != a
    with functai.configure(lm="openai:gpt-4.1-mini", base_url="https://gw.example/v1", cache_replies=True,
                           api_retries=9):
        assert mood.setup == a                                    # where and how it is reached: not in it


@functai.module
def both(a: str) -> str:
    return mood(a)


def test_code_a_version_cannot_read_is_named_by_the_host():
    """calls.md, *Code the language cannot read*: code_id is in every setup worked out under it, a module's too."""
    with functai.configure(lm="openai:gpt-4.1-mini"):
        plain = (mood.setup, both.setup)
        assert "code_id" not in mood.setup_document()             # nothing said: setups as before
        with functai.configure(code_id="sha256:aaaa"):
            assert mood.setup_document()["code_id"] == "sha256:aaaa"
            assert both.setup_document()["code_id"] == "sha256:aaaa"
            assert mood.setup != plain[0] and both.setup != plain[1]
            version = mood.version
        assert mood.version == version                            # a setting, not the version
        with functai.configure(code_id="sha256:bbbb"):
            other = both.setup
        with functai.configure(code_id="sha256:aaaa"):
            assert both.setup != other
        assert (mood.setup, both.setup) == plain
    for bad in ["", "   ", "x" * 201, 7]:
        with pytest.raises(TypeError, match="code_id"):
            with functai.configure(code_id=bad):
                pass


def test_rewriting_a_tool_is_another_setup_of_the_same_version():
    a, b = agent(here), agent(elsewhere)
    assert a.version == b.version                                 # what the model is told is the same
    with functai.configure(lm="openai:gpt-4.1-mini"):
        assert a.setup != b.setup
        entry = a.setup_document()["tools"]["look"]
    assert entry["code"]["language"] == "python" and entry["code"]["code"].startswith("sha256:")


def test_an_ai_function_used_as_a_tool_counts_by_its_setup():
    @functai.ai(tools=[mood])
    def caring(message: str) -> str:
        """Answer kindly."""
        ...
    with functai.configure(lm="openai:gpt-4.1-mini"):
        assert caring.setup_document()["tools"]["mood"] == {"setup": mood.setup}


def test_escalation_names_the_backup():
    @functai.ai(escalate_to=mood)
    def careful(text: str) -> str:
        """How does the writer feel?"""
        ...
    with functai.configure(lm="openai:gpt-4.1-mini"):
        assert careful.setup_document()["escalate"] == {"to": {"setup": mood.setup}, "below": 0.9}
    with functai.configure(lm="openai:gpt-4.1-mini", escalate_to="anthropic:claude-sonnet-4-5", escalate_below=0.7):
        assert mood.setup_document()["escalate"] == {"to": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
                                                     "below": 0.7}


def test_plugins_count_with_their_settings():
    with functai.configure(lm="openai:gpt-4.1-mini", plugins=[functai.keep_pictures(2)]):
        a = mood.setup
        assert mood.setup_document()["plugins"] == [{"name": "keep-pictures", "version": "1.0.0",
                                                     "config": {"last": 2, "at_once": 1}}]
    with functai.configure(lm="openai:gpt-4.1-mini", plugins=[functai.keep_pictures(3)]):
        assert mood.setup != a


def test_reach_open_and_behind_a_wall():
    stock = agent(here)
    with functai.configure(lm="openai:gpt-4.1-mini"):
        assert stock.setup_document()["reach"] == "open"
    with functai.configure(lm="openai:gpt-4.1-mini", wall="srt"):
        assert stock.setup_document()["reach"] == {"domains": ["shop.example"], "secrets": []}
    with functai.configure(lm="openai:gpt-4.1-mini", wall={"name": "srt", "domains": ["*.example", "shop.example"],
                                                           "limits": {"seconds": 60}}):
        assert stock.setup_document()["reach"] == {"domains": ["*.example", "shop.example"], "secrets": [],
                                                   "limits": {"seconds": 60}}


def test_a_wall_that_does_not_let_through_what_it_needs_refuses_before_any_request(tmp_path):
    stock = agent(here)
    router = FakeRouter(responder=lambda req: XML.format("yes"))
    with functai.configure(lm="openai:gpt-4.1-mini", client=router, log_calls=tmp_path,
                           wall={"name": "srt", "domains": ["api.other.example"]}):
        with pytest.raises(functai.ReachError) as err:
            stock("socks")
    assert err.value.code == "needs-unmet" and err.value.missing == {"domains": ["shop.example"], "secrets": []}
    assert not router.requests                                    # nothing sent
    [rec] = records(tmp_path)
    assert rec["error"]["type"] == "ReachError" and rec["placement"] == {"wall": "srt"}
    with functai.configure(lm="openai:gpt-4.1-mini", client=router, wall={"name": "srt", "domains": ["*.example"]}):
        assert stock("socks") == "yes"                            # *.example lets shop.example through


def test_a_record_says_its_setup_and_where_it_ran(tmp_path):
    router = FakeRouter(responder=lambda req: XML.format("calm"))
    with functai.configure(lm="openai:gpt-4.1-mini", client=router, log_calls=tmp_path,
                           base_url="https://gw.example/v1", wall="srt"):
        mood("fine")
        setup = mood.setup
    [rec] = records(tmp_path)
    assert rec["program"]["setup"] == setup
    assert rec["placement"] == {"wall": "srt", "base_url": "https://gw.example/v1"}
    assert "spent" not in rec


def test_what_code_spent_is_recorded_and_counted_by_limits(tmp_path):
    @functai.tool(effects="reads")
    def browse(url: str) -> str:
        """Browse a page with a paid service."""
        functai.spend(0.04)
        functai.spend(0.01)
        return "a page"

    @functai.ai(tools=[browse])
    def walker(url: str) -> str:
        """Walk the site."""
        ...

    @functai.module
    def study(url: str) -> str:
        functai.spend(0.2)
        return walker(url)

    def reply(req):
        done = [p for m in req.messages for p in m.parts if type(p).__name__ == "ToolResultPart"]
        if not done and req.tools:
            return [lm15.ToolCallPart(id="c1", name="browse", input={"url": "https://a.example/"})]
        return XML.format("fine")
    router = FakeRouter(responder=reply)
    with functai.configure(lm="openai:gpt-4.1-mini", client=router, log_calls=tmp_path):
        assert study("https://a.example/") == "fine"
    by_name = {r["program"]["name"]: r for r in records(tmp_path)}
    assert by_name["walker"]["spent"] == {"usd": 0.05} and by_name["study"]["spent"] == {"usd": 0.2}
    # a cost limit counts what code spent, and needs no price for it
    with functai.configure(lm="openai:gpt-4.1-mini", client=router,
                           plugins=[functai.limits(cost=0.1, prices={"openai:gpt-4.1-mini": {
                               "input_per_million": 0, "output_per_million": 0}})]):
        with pytest.raises(functai.PluginError) as err:
            study("https://a.example/")
    assert err.value.code == "plugin-stopped" and "cost 0.1" in str(err.value)
    with pytest.raises(ValueError):
        functai.spend(-1)
    functai.spend(1.0)                                            # outside any call: nothing


def test_a_module_names_each_step_it_reaches():
    @functai.module(needs={"domains": ["api.browser-use.com"], "secrets": ["BU_KEY"]})
    def study(url: str) -> str:
        return mood(url)
    with functai.configure(lm="openai:gpt-4.1-mini", wall="srt"):
        doc = study.setup_document()
        assert doc["version"] == study.version
        assert list(doc["ai"].values()) == [mood.setup]
        assert doc["reach"] == {"domains": ["api.browser-use.com"], "secrets": ["BU_KEY"]}
    with pytest.raises(TypeError):
        functai.module(needs={"domains": ["not a host"]})(lambda url: url)


def test_a_saved_folder_keeps_its_setup_and_two_are_compared(tmp_path, monkeypatch):
    (tmp_path / "setup_shop.py").write_text('''
import functai

@functai.tool(effects="reads", needs={"domains": ["shop.example"]})
def look(page: str) -> str:
    """Look at a page."""
    return "in stock"

@functai.ai(tools=[look])
def stock(item: str) -> str:
    """Is the item in stock?"""
    ...

@functai.module
def answer(item: str) -> str:
    return stock(item)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    import setup_shop as shop
    with functai.configure(lm="openai:gpt-4.1-mini"):
        functai.save(shop.answer, tmp_path / "a")
    with functai.configure(lm="anthropic:claude-haiku-4-5", temperature=0.3):
        functai.save(shop.answer, tmp_path / "b")
    manifest = json.loads((tmp_path / "a" / "functai.json").read_text())
    node = manifest["nodes"]["setup_shop:answer"]
    assert node["version"] == shop.answer.version and node["setup"]["version"] == shop.answer.version
    [tool] = manifest["nodes"]["setup_shop:stock"]["ai"]["tools"]
    assert tool == {"node": "setup_shop:look", "effects": "reads", "needs": {"domains": ["shop.example"], "secrets": []}}
    with functai.configure(lm="openai:gpt-4.1-mini"):
        loaded = functai.load(tmp_path / "a", trust=True)
        assert loaded.version == shop.answer.version and loaded.setup == shop.answer.setup
        assert functai.setup_diff(tmp_path / "a", loaded) == []
    diff = {(d["program"], d["field"]): (d["before"], d["after"]) for d in functai.setup_diff(tmp_path / "a", tmp_path / "b")}
    assert diff[("setup_shop:stock", "model.provider")] == ("openai", "anthropic")
    assert diff[("setup_shop:stock", "config")] == (None, {"temperature": 0.3})
    out = subprocess.run([sys.executable, "-m", "functai", "setup-diff", str(tmp_path / "a"), str(tmp_path / "b")],
                         capture_output=True, text=True, check=True).stdout
    assert 'setup_shop:stock: model.model: "gpt-4.1-mini" -> "claude-haiku-4-5"' in out
    assert calllog.module_version(loaded) == shop.answer.version


def test_what_the_code_a_setup_reads_cannot_say_code_id_says():
    def made(answer, code_id=None):
        def look(page: str) -> str:
            """Look at a page."""
            return f"page {page}: {answer}"
        return functai.tool(look, effects="reads", code_id=code_id)
    with functai.configure(lm="openai:gpt-4.1-mini"):
        assert agent(made("in stock")).setup == agent(made("sold out")).setup        # one text, two behaviours
        assert agent(made("in stock", "a")).setup != agent(made("sold out", "b")).setup
        assert agent(made("x", "v3")).setup_document()["tools"]["look"]["code"]["id"] == "v3"
