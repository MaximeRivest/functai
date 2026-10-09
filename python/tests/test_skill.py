"""Every Python example in the coding-agent skill's SKILL.md runs (offline: a fake
model answers), so the skill cannot drift from the code it teaches."""
import base64
import re
from pathlib import Path

import pytest

import functai

SKILL = Path(functai.__file__).parent / "skill" / "SKILL.md"
BLOCKS = re.findall(r"```python\n(.*?)```", SKILL.read_text(), re.S)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def answer(request):
    """A reply that fits whatever the request asks for."""
    system = request.system or ""
    if "<quotes>" in system:
        return '<quotes>\n["We used 12 plots."]\n</quotes>\n<result>\n{"units": 6, "analyses": [{"method": "anova"}]}\n</result>'
    if "circle" in system:
        return "<result>\ncircle\n</result>"
    if "shipping" in system:
        return "<result>\nbilling\n</result>"
    return "<result>\nNorth\n</result>"


@pytest.fixture
def world(tmp_path, monkeypatch, fake):
    """The files and names the examples assume: a project folder, its tables, a picture, a document."""
    monkeypatch.chdir(tmp_path)
    fake(responder=answer)
    (tmp_path / "out").mkdir()
    (tmp_path / "papers.csv").write_text("paper_id,methods\nP1,We used 12 plots.\nP2,Thirty mice.\n")
    (tmp_path / "tickets.csv").write_text("message,category\nRefund please,billing\nBox crushed,shipping\nCharged twice,billing\n")
    (tmp_path / "rows.csv").write_text("text,result\n" + "".join(f"Sentence {i}.,Plain {i}.\n" for i in range(12)))
    (tmp_path / "style_guide.md").write_text("Short sentences.")
    (tmp_path / "cat.png").write_bytes(PNG)
    (tmp_path / "report.md").write_text("# Report\n\nNorth sold most.")
    (tmp_path / "notes.md").write_text("Plan: ship Friday.")
    functai_ns = {"functai": functai}
    from typing import Literal

    from functai import ai

    @ai
    def team(message: str) -> Literal["shipping", "billing"]:
        """Which team?"""

    @ai
    def team_v2(message: str) -> Literal["shipping", "billing"]:
        """Which team? Refunds are billing."""

    @ai
    def simplify(text: str, style_guide: str) -> str:
        """Rewrite simply, following the style guide."""

    functai_ns.update(team=team, team_v2=team_v2, simplify=simplify, files=[tmp_path / "cat.png"])
    return functai_ns


def test_the_skill_has_its_examples():
    assert len(BLOCKS) >= 5


def test_the_skills_examples_need_nothing_functai_data_does_not_install():
    """`functai[data]` brings dpyr, not pandas (checked in a clean environment from
    PyPI, 2026-10-09): an example calling .to_pandas() fails for its readers."""
    for code in BLOCKS:
        assert ".to_pandas(" not in code and "import pandas" not in code, code[:80]


@pytest.mark.parametrize("i", range(len(BLOCKS)), ids=lambda i: BLOCKS[i].strip().splitlines()[0][:40])
def test_each_example_in_the_skill_runs(world, i):
    pytest.importorskip("dpyr")
    if "attachments" in BLOCKS[i]:
        pytest.importorskip("attachments")
    import runpy
    earlier = [b for b in BLOCKS[:i] if "def read_study" in b or "from dpyr import" in b]   # what a later one uses
    code = "\n\n".join([*earlier, BLOCKS[i]]).replace('att("report.pdf[images: true]")', 'att("report.md")').replace(
        'att("budget.xlsx")', 'att("notes.md")')   # no PDF/Excel readers needed offline; the same calls otherwise
    script = Path(f"example_{i}.py")             # a file, as an agent writes one: comments and _ai outputs need it
    script.write_text(code)
    runpy.run_path(str(script), init_globals=dict(world))
