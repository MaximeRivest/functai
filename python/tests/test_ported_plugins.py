"""Real Pi and Chattering extensions, ported as FunctAI plugins (examples/plugins): the plugin API covers what
they do. Offline: a fake provider."""

import subprocess
import sys
from pathlib import Path

import lm15

import functai
from functai import ai

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "plugins"))
import auto_inject  # noqa: E402
import bulky_budget  # noqa: E402
import checkpoints  # noqa: E402
import modes  # noqa: E402
import prompt_capture  # noqa: E402
import session_prompt  # noqa: E402

XML = "<result>\n{}\n</result>"


@functai.tool(effects="reads")
def read_file(path: str) -> str:
    """Read a file."""
    return "contents"


@functai.tool(effects="changes")
def write_file(path: str, text: str) -> str:
    """Write a file."""
    Path(path).write_text(text)
    return "written"


@ai(tools=[read_file, write_file])
def assistant(message: str) -> str:
    """You help with the project."""


def texts(request):
    return " ".join(getattr(p, "text", "") or "" for m in request.messages for p in m.parts)


def test_modes(fake):
    r = fake(XML.format("a"), XML.format("b"), XML.format("c"))
    chat = assistant.conversation(plugins=[modes.plugin([
        modes.Mode("review", "Review", appendix="Only point out problems.", tools=["read_file"]),
        modes.Mode("pirate", "Pirate", system_prompt="You are a pirate.")])])
    chat("hello")
    assert [t.name for t in r.requests[0].tools] == ["read_file", "write_file"]
    modes.switch(chat, "review")
    chat("look")
    assert [t.name for t in r.requests[1].tools] == ["read_file"]
    assert "Only point out problems." in str(r.requests[1].system) and "You help" in str(r.requests[1].system)
    other = chat.continue_from(chat.turns[0])           # the mode follows the branch: none on this one
    modes.switch(other, "pirate")
    other("ahoy")
    assert "You are a pirate." in str(r.requests[2].system) and "You help" not in str(r.requests[2].system)


def test_session_prompt(fake):
    r = fake(XML.format("a"), XML.format("b"))
    chat = assistant.conversation(plugins=[session_prompt.plugin])
    chat("one")
    session_prompt.set(chat, "Be terse.")
    chat("two")
    assert "Be terse." not in str(r.requests[0].system)
    assert str(r.requests[1].system).index("Be terse.") < str(r.requests[1].system).index("You help")


def test_auto_inject(fake, tmp_path):
    (tmp_path / "notes.md").write_text("THE NOTES")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("A = 1")
    r = fake(XML.format("ok"))
    chat = assistant.conversation(plugins=[auto_inject.plugin(tmp_path)])
    chat("read @notes.md and @src and @../outside")
    sent = texts(r.requests[0])
    assert "THE NOTES" in sent and "A = 1" in sent
    assert "THE NOTES" in chat.turns[0].inputs["message"]           # recorded: asked again with the same text


def test_prompt_capture(fake, tmp_path):
    fake(XML.format("ok"))
    chat = assistant.conversation("cap", plugins=[prompt_capture.plugin(tmp_path)])
    chat("hi")
    got = prompt_capture.last(tmp_path, "cap")
    assert "You help with the project." in got["pending"]
    assert "You help with the project." in prompt_capture.last(tmp_path, "assistant")["wire"]


def test_checkpoints(fake, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "start")
    target = repo / "file.txt"

    def respond(req):
        done = [p for m in req.messages for p in m.parts if type(p).__name__ == "ToolResultPart"]
        if not done:
            return [lm15.ToolCallPart(id="c1", name="read_file", input={"path": "x"}),
                    lm15.ToolCallPart(id="c2", name="write_file", input={"path": str(target), "text": "new"})]
        return XML.format("done")

    fake(responder=respond)
    chat = assistant.conversation(plugins=[checkpoints.plugin(repo)])
    chat("write it")
    [cp] = chat.entries("checkpoints", "checkpoint")
    assert cp["data"]["tool"] == "write_file" and cp["data"]["before"] != cp["data"]["after"]   # only the write
    assert target.read_text() == "new"
    checkpoints.undo(repo, cp["data"])
    assert not target.exists()                                       # undone: the file it created is gone


def test_bulky_budget(fake):
    @ai
    def viewer(question: str, document: str) -> str:
        """Answer about the document."""

    r = fake(responder=lambda req: XML.format("ok"))
    chat = viewer.conversation(plugins=[bulky_budget.plugin(["document"], high=250, low=120)])
    for i in range(4):
        chat(f"q{i}", f"DOC{i}-" + "x" * 100)
    last = texts(r.requests[-1])
    assert "DOC0" not in last and "DOC1" not in last and "DOC2" in last and "DOC3" in last
    assert chat.turns[-1]._st.record["context"]["without"] == {chat.turns[0].id: ["document"],
                                                               chat.turns[1].id: ["document"]}
