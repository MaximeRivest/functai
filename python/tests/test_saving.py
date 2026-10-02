"""check / save / load / verify: programs written to real files, a fake model."""

import importlib
import itertools
import json
import shutil
import sys
import textwrap

import lm15
import pytest

import functai
from conftest import FakeRouter
from functai import saved

_ids = itertools.count()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """``mods = project({"name": source, ...})``: write modules (names made unique
    per test, so imports never collide) and import them."""
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))

    def make(files, data=None):
        n = next(_ids)
        names = {name: f"{name}_{n}" for name in files}
        for name, src in files.items():
            src = textwrap.dedent(src)
            for old, new in names.items():
                src = src.replace(f"from {old} import", f"from {new} import").replace(f"import {old}\n",
                                                                                        f"import {new} as {old}\n")
            (root / f"{names[name]}.py").write_text(src)
        for rel, text in (data or {}).items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        importlib.invalidate_caches()
        return {name: importlib.import_module(real) for name, real in names.items()}
    return make


XML = "<result>\n{}\n</result>"


def answer(text):
    return FakeRouter(responder=lambda req: XML.format(text))


PIPELINE = {
    "kinds": """
        import dataclasses
        import enum


        class Verdict(enum.Enum):
            TRUE = "true"
            FALSE = "false"


        @dataclasses.dataclass
        class Evidence:
            claim: str
            sources: list[str]
    """,
    "helpers": """
        import re

        import functai

        SPACES = re.compile(r"\\s+")


        def clean(text: str) -> str:
            stop = set(functai.file("data/stop.txt").read_text().split())
            return " ".join(w for w in SPACES.split(text) if w.lower() not in stop)
    """,
    "prog": """
        import json

        from functai import _ai, ai, module
        from helpers import clean
        from kinds import Evidence, Verdict

        DEFAULT = Verdict.FALSE


        def search(query: str) -> str:
            \"\"\"Search the web.\"\"\"
            return json.dumps({"q": clean(query)})


        @ai(tools=[search], temperature=0.2, examples=[("Paris is in France", "paris france")])
        def generate_query(claim: str) -> str:
            \"\"\"Write a search query for the claim.\"\"\"


        @ai(module="cot")
        def judge(evidence: Evidence) -> Verdict:
            \"\"\"Is the claim true, given the sources?\"\"\"
            verdict: Verdict = _ai
            return verdict


        @module
        def fact_check(claim: str) -> Verdict:
            query = generate_query(claim)
            decide = judge
            return decide(Evidence(claim=claim, sources=[search(query)])) or DEFAULT
    """,
}
DATA = {"data/stop.txt": "the\na\n"}


def pipeline_model():
    def responder(req):
        if "search query" in req.system:
            if not any(type(p).__name__ == "ToolResultPart" for m in req.messages for p in m.parts):
                return [lm15.ToolCallPart(id="c1", name="search", input={"query": "the capital of France"})]
            return XML.format("paris capital")
        return "<reasoning>\nok\n</reasoning>\n<verdict>\ntrue\n</verdict>"
    return FakeRouter(responder=responder)


# ------------------------------------------------------------------ check


def test_check_follows_the_whole_program(project):
    prog = project(PIPELINE, DATA)["prog"]
    report = functai.check(prog.fact_check)
    kinds = {n.name: n.kind for n in report.nodes.values()}
    assert kinds == {"fact_check": "module", "generate_query": "ai", "search": "function", "clean": "function",
                     "judge": "ai", "Evidence": "class", "Verdict": "class"}
    assert [n.files for n in report.nodes.values() if n.name == "clean"] == [["data/stop.txt"]]
    assert "functai" in report.requirements and "lm15" not in report.requirements   # lm15 comes through functai
    assert not report.errors, report
    text = repr(report)
    assert "tool search" in text and "DEFAULT = Verdict.FALSE" in text


def test_module_finds_ai_functions_under_other_names_and_through_helpers(project):
    prog = project(PIPELINE, DATA)["prog"]
    assert set(prog.fact_check.named_ai_functions()) == {"generate_query", "judge"}


def test_problems_name_the_offender_and_the_fix(project):
    mods = project({"bad": """
        import functai
        from functai import ai

        CACHE = {}
        COUNT = 0
        LOCK = __import__("threading").Lock()


        def remember(x: str) -> str:
            \"\"\"Remember.\"\"\"
            global COUNT
            COUNT += 1
            CACHE[x] = 1
            with LOCK:
                return x


        @ai(tools=[remember, lambda y: y])
        def untyped(text, n: int):
            \"\"\"Do it.\"\"\"


        def uses_local_import(x: str) -> str:
            import helpers_x
            return x


        @functai.module
        def run(text: str):
            return untyped(text, 1) + uses_local_import(text) + str(eval("1"))
    """})
    report = functai.check(mods["bad"].run)
    codes = {(p.code, p.where.split(":")[-1]) for p in report.problems}
    assert ("hidden-state", "remember") in codes                 # CACHE[x] = 1 and global COUNT
    assert ("unsaveable-value", "LOCK") in codes
    assert ("lambda", "untyped tools") in codes
    assert ("untyped-input", "untyped") in codes and ("untyped-output", "untyped") in codes
    assert ("untyped-output", "run") in codes
    assert ("dynamic-lookup", "run") in codes
    hidden = [p for p in report.problems if p.code == "hidden-state"]
    assert {("CACHE" in p.message) or ("COUNT" in p.message) for p in hidden} == {True}
    assert all(p.fix for p in report.problems)
    with pytest.raises(functai.Refused) as err:
        functai.save(mods["bad"].run, "/tmp/never")
    assert err.value.report is report or err.value.report.errors


def test_closures_and_nested_functions_are_saved_by_value(project, tmp_path):
    mods = project({"nested": """
        from functai import ai


        def make(prefix: str):
            @ai(temperature=0.1)
            def tag(text: str) -> str:
                \"\"\"Tag it.\"\"\"

            def run(text: str) -> str:
                return prefix + tag(text)
            return run

        run = make(">> ")
    """})
    report = functai.check(mods["nested"].run)
    assert report.ok, report
    functai.save(mods["nested"].run, tmp_path / "s")
    functai.configure(lm="gpt-4.1-mini", client=answer("x"))
    loaded = functai.load(tmp_path / "s", trust=True)
    assert loaded("hi") == ">> x"


def test_two_different_functions_with_one_name_conflict(project):
    mods = project({"twins": """
        from functai import ai


        def make(k: str):
            @ai
            def f(text: str) -> str:
                \"\"\"Do.\"\"\"
            return f

        a, b = make("a"), make("b")


        def both(text: str) -> str:
            return a(text) + b(text)
    """})
    codes = {p.code for p in functai.check(mods["twins"].both).errors}
    assert "name-conflict" in codes


# ------------------------------------------------------------------ save and load


def test_save_writes_a_readable_folder_and_load_runs_it(project, tmp_path, monkeypatch):
    mods = project(PIPELINE, DATA)
    prog = mods["prog"]
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "fact_check"
    functai.save(prog.fact_check, target)
    files = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert files == sorted([f"code/{mods[m].__name__}.py" for m in ("helpers", "kinds", "prog")]
                           + ["files/data/stop.txt", "functai.json", "requirements.lock", "requirements.txt"])
    code = (target / "code" / f"{prog.__name__}.py").read_text()
    assert "@ai" not in code and "def search(query: str) -> str:" in code
    assert code.index("DEFAULT = Verdict.FALSE") < code.index("def fact_check")
    manifest = json.loads((target / "functai.json").read_text())
    gq = manifest["nodes"][f"{prog.__name__}:generate_query"]["ai"]
    assert gq["config"] == {"temperature": 0.2} and gq["tools"] == [f"{prog.__name__}:search"]
    assert gq["state"]["demos"] == [{"inputs": {"claim": "Paris is in France"}, "outputs": {"result": "paris france"}}]

    with pytest.raises(PermissionError, match="trust=True"):
        functai.load(target)
    shutil.rmtree(tmp_path / "project")                        # the original code is gone
    for name in [m for m in sys.modules if m.startswith(("prog_", "helpers_", "kinds_"))]:
        del sys.modules[name]
    r = pipeline_model()
    functai.configure(lm="gpt-4.1-mini", client=r)
    fc = functai.load(target, trust=True)
    assert fc("Paris is the capital of France").value == "true"
    tool_result = [p for m in r.requests[1].messages for p in m.parts if type(p).__name__ == "ToolResultPart"]
    assert '"q": "capital of France"' in str(tool_result[0])   # the saved helper and data file ran


def test_save_is_all_or_nothing_and_does_not_overwrite_by_accident(project, tmp_path):
    prog = project(PIPELINE, DATA)["prog"]
    functai.save(prog.fact_check, tmp_path / "s")
    with pytest.raises(FileExistsError):
        functai.save(prog.fact_check, tmp_path / "s")
    functai.save(prog.fact_check, tmp_path / "s", overwrite=True)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".s.")]


def test_allow_saves_a_known_problem_as_a_constant(project, tmp_path):
    mods = project({"stateful": """
        from functai import ai

        SEEN = {"a": 1}


        @ai
        def f(text: str) -> str:
            \"\"\"Do.\"\"\"
            SEEN[text] = 1
            return _ai

        from functai import _ai
    """})
    with pytest.raises(functai.Refused):
        functai.save(mods["stateful"].f, tmp_path / "s")
    functai.save(mods["stateful"].f, tmp_path / "s", allow=["hidden-state"])
    manifest = json.loads((tmp_path / "s" / "functai.json").read_text())
    assert [p["code"] for p in manifest["allowed"]] == ["hidden-state"]
    assert "SEEN = {'a': 1}" in next((tmp_path / "s" / "code").glob("*.py")).read_text()


def test_templates_adapters_and_settings_come_back(project, tmp_path):
    mods = project({"layouts": """
        import lmcc
        from functai import ai, system, user

        QA = lmcc.adapter(messages=[
            lmcc.system("{instruction}\\n{% for f in outputs %}<{f.name}>\\n{f.value}\\n</{f.name}>\\n{% endfor %}"),
            lmcc.turns(), lmcc.user("Q: {x}")])


        @ai(template=[system("Pirate. {instruction}"), user("T: {x}")], max_tokens=50)
        def pirate(x: str) -> str:
            \"\"\"Say it.\"\"\"


        @ai(adapter=QA, teacher=pirate)
        def qa(x: str) -> str:
            \"\"\"Answer.\"\"\"


        def both(x: str) -> str:
            return pirate(x) + qa(x)
    """})
    functai.save(mods["layouts"].both, tmp_path / "s")
    r = FakeRouter(responder=lambda req: "arr" if req.system.startswith("Pirate") else XML.format("ok"))
    functai.configure(lm="gpt-4.1-mini", client=r)
    loaded = functai.load(tmp_path / "s", trust=True)
    assert loaded("hi") == "arrok"
    assert r.requests[0].messages[-1].parts[0].text == "T: hi" and r.requests[0].config.max_tokens == 50
    assert r.requests[1].messages[-1].parts[0].text == "Q: hi"
    qa = loaded.__globals__["qa"]
    assert qa._settings["teacher"] is loaded.__globals__["pirate"]


def test_optimized_state_is_saved(project, tmp_path):
    mods = project({"opt": """
        from functai import ai


        @ai
        def classify(text: str) -> str:
            \"\"\"Classify.\"\"\"
    """})
    f = mods["opt"].classify
    f.instructions = "Label it: yes or no."
    f.demos = [("a", "yes"), ("b", "no")]
    functai.save(f, tmp_path / "s")
    g = functai.load(tmp_path / "s", trust=True)
    assert g.instructions == "Label it: yes or no." and len(g.demos) == 2


# ------------------------------------------------------------------ what load refuses


def _resave_manifest(target, edit):
    path = target / "functai.json"
    m = json.loads(path.read_text())
    edit(m)
    path.write_text(json.dumps(m))


def test_load_refuses_edited_code_missing_packages_and_changed_prompts(project, tmp_path):
    prog = project(PIPELINE, DATA)["prog"]
    target = tmp_path / "s"
    functai.save(prog.fact_check, target)
    code = target / "code" / f"{prog.__name__}.py"
    rel = f"code/{prog.__name__}.py"
    code.write_text(code.read_text().replace("Write a search query", "Write a query"))
    with pytest.raises(saved.LoadRefused, match="was changed since it was saved"):
        functai.load(target, trust=True)

    # the same edit with the hash updated (someone edited both): the prompt differs
    _resave_manifest(target, lambda m: m["hashes"].__setitem__(rel, saved._sha256(code.read_bytes())))
    with pytest.raises(saved.LoadRefused, match="renders a different request"):
        functai.load(target, trust=True)
    with pytest.warns(UserWarning, match="different request"):
        functai.load(target, trust=True, check_env="warn")

    _resave_manifest(target, lambda m: m["requirements"].append("not-a-real-package==1.0"))
    with pytest.raises(saved.LoadRefused, match="not-a-real-package"):
        functai.load(target, trust=True)


# ------------------------------------------------------------------ verify


def test_recordings_replay_and_catch_behavior_changes(project, tmp_path):
    mods = project(PIPELINE, DATA)
    prog = mods["prog"]
    target = tmp_path / "s"
    functai.configure(lm="gpt-4.1-mini", client=pipeline_model())
    functai.save(prog.fact_check, target, record=[{"claim": "Paris is the capital of France"}])
    rec = json.loads((target / "recordings.json").read_text())
    assert len(rec["recordings"][0]["exchanges"]) == 3 and rec["recordings"][0]["output"] == {"json": "true"}
    functai.configure(client=None)
    assert functai.verify(target, trust=True, fresh=False).ok

    # the helper now behaves differently: the tool result, so the next request, changes
    rel = f"code/{mods['helpers'].__name__}.py"
    code = target / rel
    code.write_text(code.read_text().replace('" ".join', '"-".join'))
    _resave_manifest(target, lambda m: m["hashes"].__setitem__(rel, saved._sha256(code.read_bytes())))
    v = functai.verify(target, trust=True, fresh=False)
    assert not v.ok and "did not send when saved" in v.problems[0] and "capital-of-France" in v.problems[0]
    with pytest.raises(PermissionError):
        functai.verify(target)


def test_recordings_made_on_the_default_model_replay(project, tmp_path, monkeypatch):
    # no model configured: functai picks one from the logins it finds; the replay has none to pick from
    from functai import models
    prog = project(PIPELINE, DATA)["prog"]
    functai.configure(lm=None, client=pipeline_model())
    real = models.resolve
    monkeypatch.setattr(models, "resolve", lambda s: real({**s, "lm": "gpt-4.1-mini"}) if s.get("lm") is None
                        else real(s))                                     # as a default pick does
    functai.save(prog.fact_check, tmp_path / "s", record=[{"claim": "Paris is the capital of France"}])
    monkeypatch.setattr(models, "resolve", real)
    functai.configure(client=None)
    rec = json.loads((tmp_path / "s" / "recordings.json").read_text())
    assert rec["settings"]["lm"] == "gpt-4.1-mini"
    v = functai.verify(tmp_path / "s", trust=True, fresh=False)
    assert v.ok, v.problems


@pytest.mark.skipif(shutil.which("uv") is None, reason="fresh verification builds environments with uv")
def test_verify_in_a_fresh_environment(project, tmp_path):
    prog = project(PIPELINE, DATA)["prog"]
    functai.configure(lm="gpt-4.1-mini", client=pipeline_model())
    functai.save(prog.fact_check, tmp_path / "s", record=[{"claim": "Paris is the capital of France"}])
    v = functai.verify(tmp_path / "s", trust=True)
    assert v.ok and v.fresh, v.log[-2000:]


def test_file_resolves_next_to_the_calling_code(project, tmp_path):
    mods = project({"reader": """
        import functai


        def where():
            return functai.file("data/x.txt")
    """}, {"data/x.txt": "hi"})
    assert mods["reader"].where().read_text() == "hi"


def test_notebook_classes_have_their_source_and_field_comments(tmp_path):
    """IPython keeps cell source in linecache under names like <ipython-input-3-…>;
    inspect.getsource cannot find classes there. functai finds the most recent
    definition, and shows its field comments to the model, as it does from files."""
    import linecache
    from functai.docments import _class_field_docments, class_source
    ns = {"__name__": "__main__"}
    for i, src in enumerate(["import dataclasses\n@dataclasses.dataclass\nclass Row:\n    a: int  # old\n",
                             "import dataclasses\n@dataclasses.dataclass\nclass Row:\n    a: int  # the count\n"
                             "    b: str  # the label\n"]):
        name = f"<ipython-input-{i}-cafe{i}>"
        linecache.cache[name] = (len(src), None, src.splitlines(keepends=True), name)
        exec(compile(src, name, "exec"), ns)
    Row = ns["Row"]
    assert "# the label" in class_source(Row)
    assert _class_field_docments(Row) == {"a": "the count", "b": "the label"}


def test_an_ai_function_can_be_a_tool_of_another(project, tmp_path):
    mods = project({"tooly": """
        from functai import ai


        @ai
        def lookup(term: str) -> str:
            \"\"\"Define the term.\"\"\"


        @ai(tools=[lookup])
        def explain(text: str) -> str:
            \"\"\"Explain; look terms up.\"\"\"
    """})
    report = functai.check(mods["tooly"].explain)
    assert report.ok, report
    assert [n.name for n in report.nodes.values()] == ["explain", "lookup"]
    functai.save(mods["tooly"].explain, tmp_path / "s")
    loaded = functai.load(tmp_path / "s", trust=True)
    assert loaded._tools[0].__name__ == "lookup"


def test_a_class_defined_in_any_notebook_cell_has_its_source():
    # rat (like IPython) keeps each cell's code in linecache with no mtime
    import linecache
    from functai.docments import class_source
    code = "from dataclasses import dataclass\n\n@dataclass\nclass CellThing:\n    name: str\n"
    linecache.cache["<rat-cell-9999>"] = (len(code), None, code.splitlines(True), "<rat-cell-9999>")
    ns = {"__name__": "__main__"}
    exec(compile(code, "<rat-cell-9999>", "exec"), ns)
    assert class_source(ns["CellThing"]).startswith("@dataclass\nclass CellThing:")
