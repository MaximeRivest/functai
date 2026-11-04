"""FunctAI Studio, offline: the map read from code, runs tagged by example, judgements and
what they carry, what-ifs, comparisons, the code edit, the summary page, the command line
and the dashboard's API. A fake model answers (conftest's FakeRouter)."""

import http.client
import json
import shutil
import threading
from pathlib import Path

import pytest

import functai
from functai import calllog
from functai.studio import cli, running
from functai.studio.data import snapshot
from functai.studio.edit import set_model
from functai.studio.graph import build_map
from functai.studio.page import summary_html
from functai.studio.project import Project, StudioError, find_project

EXAMPLE = Path(functai.__file__).parent / "studio" / "examples" / "club"


def _user(request):
    return "".join(getattr(p, "text", "") or "" for m in request.messages for p in m.parts)


def club_model(request):
    """kind gets one message wrong on purpose (the little brother is "spam")."""
    system, user = request.system or "", _user(request)
    if "Function: kind" in system:
        if any(w in user for w in ("FREE", "WIN", "brother")):
            return "<result>\nspam\n</result>"
        if any(w in user for w in ("should", "Could we")):
            return "<result>\nidea\n</result>"
        return "<result>\nquestion\n</result>"
    if "Function: answer" in system:
        return "<result>\nFrom the FAQ.\n</result>"
    if "Function: reply" in system:
        return "<result>\nReply to: " + user.split("</message>")[0].split("<message>")[-1].strip()[:30] + "\n</result>"
    return "<result>\nok\n</result>"


@pytest.fixture
def club(tmp_path, monkeypatch, fake):
    folder = tmp_path / "club-inbox"
    shutil.copytree(EXAMPLE, folder, ignore=shutil.ignore_patterns("__pycache__", ".functai"))
    monkeypatch.chdir(folder)
    router = fake(responder=club_model)
    p = find_project("club.py")
    p.load()
    p.router = router
    return p


def _row(snap, n):
    return next(r for r in snap["rows"] if str(r["n"]) == str(n))


# ------------------------------------------------------------------ the project and the map

def test_finds_the_pipeline_and_its_steps(club):
    assert club.name == "club_inbox"
    assert list(club.steps()) == ["kind", "answer", "reply"]
    assert len(club.examples()) == 12
    assert (club.functai_dir / ".gitignore").exists() is False      # nothing written before a run
    with pytest.raises(StudioError, match="no FunctAI program named"):
        Project(club.file, "nope").load()


def test_the_map_follows_values_and_conditions(club):
    g = build_map(club)
    nodes = {n["id"]: n for n in g["nodes"]}
    edges = {(e["from"], e["to"]) for e in g["edges"]}
    assert g["flows"] and not g["notes"]
    assert {"in:message", "kind", "answer", "reply", "const:FAQ", "out"} <= set(nodes)
    assert ("kind", "answer") in edges                 # the condition on kind decides whether answer runs
    assert ("const:FAQ", "answer") in edges and ("answer", "reply") in edges and ("reply", "out") in edges
    assert nodes["answer"]["when"] == "kind == 'question'"
    spam = next(n for n in g["nodes"] if n["kind"] == "value" and n.get("returns"))
    assert spam["value"] == "" and spam["when"] == "kind == 'spam'" and (spam["id"], "out") in edges
    idea = next(n for n in g["nodes"] if n["kind"] == "value" and not n.get("returns"))
    assert idea["alt"] == ["answer"] and (idea["id"], "reply") in edges
    assert nodes["kind"]["choices"] == ["question", "idea", "spam"]
    # the same code draws the same picture
    assert build_map(club) == g


def test_one_ai_function_is_a_one_step_map(tmp_path, monkeypatch, fake):
    (tmp_path / "one.py").write_text('from functai import ai\n\n@ai\ndef title(text: str) -> str:\n    """A title."""\n')
    monkeypatch.chdir(tmp_path)
    p = find_project()
    g = build_map(p)
    assert [n["kind"] for n in g["nodes"]] == ["input", "step", "output"]


# ------------------------------------------------------------------ runs, judgements, scores

def test_runs_are_tagged_by_example_and_judged_by_people(club):
    res = running.run_examples(club, club.examples(), threads=3)
    assert res["failed"] == 0
    calls, _ = calllog.read(club.calls_dir)
    roots = [c for c in calls if not c["parent"]]
    assert len(roots) == 12 and all(c["caller"]["studio"]["key"] for c in roots)
    snap = snapshot(club)
    assert snap["scores"]["end"]["unchecked"] == 12 and snap["scores"]["end"]["checked"] == 0
    r8 = _row(snap, 8)                                       # "can my little brother join?" sorted as spam
    assert r8["steps"]["kind"]["value"] == "spam" and r8["end"]["value"] == "" and "answer" not in r8["steps"]
    functai.rate(r8["steps"]["kind"]["call"], "wrong", answer="question", by="maxime", folder=club.calls_dir)
    functai.rate(_row(snap, 1)["call"], "right", by="maxime", folder=club.calls_dir)
    snap = snapshot(club)
    r8 = _row(snap, 8)
    assert r8["steps"]["kind"]["judgement"]["source"] == "person"
    assert r8["state"] == "wrong" and r8["end"]["judgement"] == {
        "verdict": "wrong", "source": "step", "because": "kind", "why": "kind was judged wrong on the way"}
    assert snap["scores"]["end"]["right"] == 1 and snap["scores"]["end"]["wrong"] == 1
    assert snap["scores"]["steps"]["kind"] == {"right": 0, "checked": 1, "low": 0.0,
                                               "high": snap["scores"]["steps"]["kind"]["high"]}
    # a run again (from the cache, free) carries the judgements of the same answers
    running.run_examples(club, club.examples()[:8])
    again = snapshot(club)
    assert _row(again, 8)["call"] != r8["call"]
    assert _row(again, 8)["steps"]["kind"]["judgement"]["source"] == "carried"
    assert _row(again, 1)["state"] == "right" and _row(again, 1)["end"]["judgement"]["source"] == "carried"
    assert all(c["cached"] for c in _row(again, 1)["steps"].values())
    assert again["money"]["per_message_cents"] > 0
    assert any(a["what"].startswith("judged kind wrong") for a in again["activity"])


def test_a_choice_judged_wrong_makes_another_choice_wrong_but_text_is_never_guessed(club):
    running.run_examples(club, club.examples())
    snap = snapshot(club)
    # someone says #8's kind should be "question": a run that answers "idea" is wrong (a choice)
    functai.rate(_row(snap, 8)["steps"]["kind"]["call"], "wrong", answer="question", by="m", folder=club.calls_dir)
    from functai.studio.data import Judgements
    calls, ratings = calllog.read(club.calls_dir)
    j = Judgements(calls, ratings)
    k8 = _row(snap, 8)["steps"]["kind"]["call"]
    other = dict(next(c for c in calls if c["id"] == k8), id="x", outputs={"result": "idea"})
    assert j.of(other, exact=True)["verdict"] == "wrong"
    assert j.of(dict(other, outputs={"result": "question"}), exact=True)["verdict"] == "right"
    # free text that differs from a judged answer stays unjudged
    assert j.of(dict(other, outputs={"result": "something else"}), exact=False) is None


# ------------------------------------------------------------------ what-if, blame, compare

def test_why_gives_the_wrong_step_its_right_answer(club):
    running.run_examples(club, club.examples())
    snap = snapshot(club)
    r8 = _row(snap, 8)
    version = calllog.program_info(club.program)["version"]
    functai.rate(r8["steps"]["kind"]["call"], "wrong", answer="question", by="m", folder=club.calls_dir)
    # the person knows what the end should be: the reply the pipeline gives a question
    functai.rate(r8["call"], "wrong", answer="Reply to: can my little brother join? he", by="m",
                 folder=club.calls_dir)
    res = running.why(club, 8)
    assert res["blamed"] == "kind" and "“kind” is at fault" in res["say"]
    assert res["tried"][0]["steps"]["answer"] == "From the FAQ."
    # nothing of the what-if is left behind: the program, its version and the project's log
    assert club.program._globals["kind"] is club.steps()["kind"]
    assert calllog.program_info(club.program)["version"] == version
    assert len([c for c in calllog.read(club.calls_dir)[0] if not c["parent"]]) == 12
    assert any((club.studio_dir / "whatif").rglob("*.jsonl"))
    blame = running.blame(club)
    assert blame["counts"] == {"kind": 1} and blame["say"].startswith("Fix “kind” first")


def test_a_what_if_answer_nobody_judged_waits_for_a_person(club):
    running.run_examples(club, club.examples())
    r8 = _row(snapshot(club), 8)
    functai.rate(r8["steps"]["kind"]["call"], "wrong", answer="question", by="m", folder=club.calls_dir)
    res = running.why(club, 8)
    assert res["tried"][0]["result"] == "someone has to look" and res["blamed"] is None
    running.judge_whatif(club, r8["inputs"], res["tried"][0]["end"], "right", by="m")
    assert running.why(club, 8)["blamed"] == "kind"


def test_compare_measures_a_change_on_judged_examples(club, tmp_path):
    running.run_examples(club, club.examples())
    snap = snapshot(club)
    for r in snap["rows"]:
        if r["steps"]["kind"]["value"] == "spam" and "brother" in r["inputs"]["message"]:
            functai.rate(r["call"], "wrong", answer="Reply to: can my little brother join? he", by="m",
                         folder=club.calls_dir)
        else:
            functai.rate(r["call"], "right", by="m", folder=club.calls_dir)
    old = club.root / "club_before.py"
    shutil.copy(club.file, old)
    res = running.compare(club, before=str(old), use_all=True)
    assert res["settled"] == 12 and res["before_right"] == res["after_right"] == 11
    assert "Tested on 12 judged examples" in res["say"] and res["better"] == res["worse"] == 0
    # only examples kept aside, unless told otherwise
    held = running.compare(club, before=str(old))
    assert held["n"] == sum(club.held_back(e["key"]) for e in club.examples())


def test_trying_another_model_for_one_step_is_recorded_as_a_trial(club):
    res = running.run_examples(club, club.examples()[:2], overrides={"kind": "claude-sonnet-5-5"})
    calls, _ = calllog.read(club.calls_dir)
    kinds = [c for c in calls if c["program"]["name"] == "kind"]
    assert {c["model"] for c in kinds} == {"claude-sonnet-5-5"}
    assert all(c["caller"]["studio"]["trial"] == {"step": "kind", "lm": "claude-sonnet-5-5"} for c in kinds)
    assert kinds[0]["changes"][0]["change"] == {"lm": "claude-sonnet-5-5"}
    assert res["trial"] and not any(r["ran"] for r in snapshot(club)["rows"])   # trials are not the pipeline's answers


# ------------------------------------------------------------------ the code, the page

def test_set_model_changes_one_decorator_and_nothing_else(club):
    before = club.file.read_text()
    out = set_model(club, "reply", "claude-sonnet-5-5")
    assert out["before"] == "@ai" and out["after"] == '@ai(lm="claude-sonnet-5-5")'
    set_model(club, "reply", "gpt-6-luna")
    after = club.file.read_text()
    assert after.replace('@ai(lm="gpt-6-luna")\ndef reply', "@ai\ndef reply") == before
    club.file.write_text(before.replace('@ai\ndef kind', '@ai(temperature=0)\ndef kind'))
    set_model(club, "kind", "x-model")
    assert '@ai(lm="x-model", temperature=0)\ndef kind' in club.file.read_text()
    with pytest.raises(StudioError):
        set_model(club, "club_inbox", "x")
    with pytest.raises(StudioError):
        set_model(club, "kind", 'bad"name')


def test_the_summary_is_one_page_with_no_script(club):
    running.run_examples(club, club.examples())
    html = summary_html(snapshot(club))
    assert "<script" not in html and "<svg" in html
    for words in ("What it does", "How often it&#x27;s right", "Nobody has checked its answers yet",
                  "What it costs", "Examples, step by step"):
        assert words in html


# ------------------------------------------------------------------ the command line

def test_the_command_line_never_asks_and_states_cost_first(club, capsys):
    assert cli.main(["run", "--estimate", "--json"]) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["examples"] == 12 and est["requests"] == 36 and est["cents"] > 0
    assert cli.main(["run", "--budget", "0", "--json"]) == cli.EXIT_BUDGET
    assert json.loads(capsys.readouterr().out)["over_budget"] is True
    assert not club.calls_dir.exists() or not any(club.calls_dir.rglob("*.jsonl"))
    assert cli.main(["run", "--only", "1,8"]) == 0
    out = capsys.readouterr().out
    assert "Ran 2 examples" in out and "Only a person can say" in out
    assert cli.main(["rate", "#8/kind", "wrong", "--answer", "question"]) == cli.EXIT_ERROR
    assert "--by NAME is needed" in capsys.readouterr().err
    assert cli.main(["rate", "#8/kind", "wrong", "--answer", "question", "--by", "maxime"]) == 0
    assert cli.main(["show", "#8"]) == 0
    shown = capsys.readouterr().out
    assert "wrong by maxime" in shown and "should have said: question" in shown
    assert cli.main(["map", "--json"]) == 0 and json.loads(capsys.readouterr().out)["flows"] is True
    assert cli.main(["page"]) == 0 and (club.root / "club_inbox-summary.html").exists()


def test_example_makes_a_ready_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["example", "club", "mine"]) == 0
    assert (tmp_path / "mine" / "club.py").exists() and (tmp_path / "mine" / "examples" / "club_inbox.csv").exists()
    assert cli.main(["example", "club", "mine"]) == cli.EXIT_ERROR


def test_held_back_examples_are_stable(club):
    keys = [e["key"] for e in club.examples()]
    first = [club.held_back(k) for k in keys]
    assert first == [club.held_back(k) for k in keys] and 0 < sum(first) < len(keys)


# ------------------------------------------------------------------ the dashboard's API

@pytest.fixture
def server(club):
    import http.server

    from functai.studio.server import Studio, make_handler
    studio = Studio(club, club.root, key=None)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(studio))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield studio, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data.decode()


def test_the_dashboard_api(server, club):
    studio, port = server
    status, st = _req(port, "GET", "/api/state")
    assert status == 200 and st["program"]["name"] == "club_inbox" and st["facts"][0][0] == "What it does"
    assert _req(port, "GET", f"/api/state?since={st['etag']}")[1]["unchanged"] is True
    assert _req(port, "GET", "/")[0] == 200 and _req(port, "GET", "/ui/app.js")[0] == 200
    assert _req(port, "GET", "/ui/../server.py")[0] == 404
    # changes need the dashboard's header (other sites cannot add it)
    assert _req(port, "POST", "/api/settings", {"monthly": 50})[0] == 403
    h = {"X-Studio": "1", "Content-Type": "application/json"}
    assert _req(port, "POST", "/api/settings", {"monthly": 50}, h)[1]["monthly"] == 50
    running.run_examples(club, club.examples()[:3])
    st = _req(port, "GET", "/api/state")[1]
    call = _row(st, 1)["call"]
    assert _req(port, "POST", "/api/rate", {"call": call, "verdict": "right"}, h)[1]["by"] == club.person()
    assert _row(_req(port, "GET", "/api/state")[1], 1)["state"] == "right"
    status, rec = _req(port, "GET", f"/api/call?id={call}")
    assert status == 200 and rec["id"] == call
    status, page = _req(port, "GET", "/api/page")
    assert status == 200 and "in one page" not in page and "What it does" in page
    # a run above the person's limit waits for a yes
    _req(port, "POST", "/api/settings", {"cap_dollars": 0}, h)
    status, body = _req(port, "POST", "/api/run", {}, h)
    assert status == 409 and body["estimate"]["cents"] > 0


def test_a_studio_on_another_address_needs_its_key(club):
    import http.server

    from functai.studio.server import Studio, make_handler
    studio = Studio(club, club.root, key="secret")
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(studio))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        assert _req(port, "GET", "/api/state")[0] == 403
        assert _req(port, "GET", "/api/state?key=secret")[0] == 200
        assert _req(port, "GET", "/api/state", headers={"Cookie": "studio_key=secret"})[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------------------------ costs before every action, who pays, versions

def test_every_action_that_asks_a_model_has_a_cost_and_the_limit(server, club):
    studio, port = server
    h = {"X-Studio": "1", "Content-Type": "application/json"}
    running.run_examples(club, club.examples())
    snap = snapshot(club)
    r8 = _row(snap, 8)
    functai.rate(r8["steps"]["kind"]["call"], "wrong", answer="question", by="m", folder=club.calls_dir)
    st = _req(port, "GET", "/api/state")[1]
    assert st["estimates"]["why"]["8"] > 0 and st["estimates"]["blame"] == st["estimates"]["why"]["8"]
    assert st["estimates"]["not_run"] == 0 and st["estimates"]["per_example"] > 0
    est = _req(port, "POST", "/api/estimate", {"action": "compare", "lm": {"reply": "claude-sonnet-5-5"},
                                                "all": True}, h)[1]
    assert est["examples"] == 0 and est["cents"] == 0          # nothing judged at the end yet: nothing to compare
    _req(port, "POST", "/api/settings", {"cap_dollars": 0}, h)
    for path, body in (("/api/why", {"n": 8}), ("/api/blame", {}), ("/api/run", {})):
        status, out = _req(port, "POST", path, body, h)
        assert status == 409 and out["estimate"]["cents"] > 0, path
    assert _req(port, "POST", "/api/why", {"n": 8, "yes": True}, h)[0] == 200


def test_the_command_line_stops_what_ifs_and_comparisons_above_the_budget(club, capsys):
    running.run_examples(club, club.examples())
    r8 = _row(snapshot(club), 8)
    functai.rate(r8["steps"]["kind"]["call"], "wrong", answer="question", by="m", folder=club.calls_dir)
    capsys.readouterr()
    assert cli.main(["why", "#8", "--budget", "0"]) == cli.EXIT_BUDGET
    assert "Not run: Finding the step at fault would cost about" in capsys.readouterr().out
    assert cli.main(["blame", "--budget", "0", "--json"]) == cli.EXIT_BUDGET
    assert json.loads(capsys.readouterr().out)["over_budget"] is True
    assert not (club.studio_dir / "whatif").exists()
    assert cli.main(["why", "#8"]) == 0


@pytest.fixture
def creds(tmp_path, monkeypatch):
    """A credentials file of the test's own: no real key or sign-in is read or touched."""
    monkeypatch.setenv("LM15_CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
              "GROQ_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from functai import accounts
    from functai.studio import connect
    monkeypatch.setattr(accounts, "CLI_LOGINS", {})          # nor the Claude Code / Codex sign-ins of this machine
    connect._PROVIDER_CACHE.clear()
    return tmp_path / "credentials.json"


def test_who_pays_names_the_company_and_whether_it_is_connected(club, creds, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-shown")
    club.update_settings(lm="claude-haiku-5-5")
    set_model(club, "reply", "gpt-6-luna")
    club.load()
    acc = {a["company"]: a for a in snapshot(club)["money"]["accounts"]}
    assert acc["Anthropic"]["steps"] == ["kind", "answer"] and acc["Anthropic"]["found"] is True
    assert acc["Anthropic"]["pays"] == "your account, per use"
    assert acc["OpenAI"]["found"] is False and acc["OpenAI"]["steps"] == ["reply"]
    assert "not-shown" not in json.dumps(snapshot(club))


def test_subscriptions_cost_nothing_per_use_and_openrouter_has_prices():
    from functai.studio.data import price_of
    assert price_of("claude:claude-haiku-5-5")["free"] and price_of("chatgpt:gpt-6-luna")["input"] == 0
    assert price_of("openrouter:anthropic/claude-haiku-5.5")["input"] == price_of("claude-haiku-5-5")["input"]
    assert price_of("ollama:llama3.2")["free"]


def test_keys_are_saved_checked_and_signed_out(creds, monkeypatch):
    from lm15.login import Auth, Verification

    from functai.studio import connect
    checks = []
    monkeypatch.setattr(Auth, "verify", lambda self, provider, **kw: checks.append(provider) or Verification("valid"))
    assert connect.default_model(None) is None
    out = connect.add_key("anthropic", "sk-ant-0123456789abcdef")
    assert out == {"provider": "anthropic", "label": "Anthropic", "checked": True, "ends_with": "cdef"}
    assert checks == ["anthropic"]
    assert "0123456789" not in json.dumps(connect.connections()) and "sk-ant" in creds.read_text()
    assert connect.default_model(None) == "claude-haiku-5-5"
    assert connect.model_status("claude-haiku-5-5")["ready"] and not connect.model_status("gpt-6-luna")["ready"]
    assert [c["model"] for c in connect.catalogue()][:3] == ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
    with pytest.raises(StudioError, match="does not look like an API key"):
        connect.add_key("openai", "short")
    monkeypatch.setattr(Auth, "verify", lambda self, provider, **kw: Verification("rejected", detail="bad key"))
    with pytest.raises(StudioError, match="refused that key"):
        connect.add_key("openai", "sk-wrong-0123456789")
    assert not connect.model_status("gpt-6-luna")["ready"]          # a refused key is not kept
    assert connect.sign_out("anthropic")["say"].startswith("Signed out of Anthropic")
    assert connect.default_model(None) is None


def test_a_sign_in_is_driven_from_the_page(creds, monkeypatch):
    import time as _t

    from lm15.login import DeviceCodeNotice, TextPrompt

    from functai import accounts
    from functai.studio import connect

    def fake_login(provider, *, ui, say, **kw):
        ui.notify(DeviceCodeNotice("ABCD-1234", "https://example.com/device", 600, 5))
        answer = ui.prompt(TextPrompt("team", "Which team?"))
        say(f"Signed in to {provider} as {answer}.")
    monkeypatch.setattr(accounts, "login", fake_login)
    logins = connect.Logins()
    w = logins.start("copilot")
    assert w.provider == "github-copilot"
    for _ in range(50):
        if w.prompt:
            break
        _t.sleep(0.05)
    v = w.view()
    assert v["state"] == "waiting" and v["notices"][0]["user_code"] == "ABCD-1234" and v["prompt"]["label"] == "Which team?"
    with pytest.raises(StudioError, match="in progress"):
        logins.start("xai")
    w.answer("red")
    for _ in range(50):
        if w.state == "done":
            break
        _t.sleep(0.05)
    assert w.state == "done" and w.result == "Signed in to github-copilot as red."


def test_secrets_only_from_this_computer(server, club, creds):
    from functai.studio.server import post
    studio, port = server
    with pytest.raises(StudioError, match="accepted only from this computer"):
        post(studio, "/api/accounts/key", {"provider": "openai", "key": "sk-0123456789"}, local=False)
    st = _req(port, "GET", "/api/state")[1]
    assert {x["provider"] for x in st["accounts"]["can_add"]["keys"]} >= {"anthropic", "openai", "gemini"}
    assert st["accounts"]["can_add"]["subscriptions"][0]["provider"] == "claude-code"


def test_versions_keep_their_code_and_one_can_go_back(server, club):
    studio, port = server
    h = {"X-Studio": "1", "Content-Type": "application/json"}
    running.run_examples(club, club.examples()[:2])
    v1 = _req(port, "GET", "/api/state")[1]["program"]["versions"][-1]
    assert v1["current"] and v1["code_kept"] and v1["ran"] == 2
    original = club.file.read_text()
    club.file.write_text(original.replace("Two sentences at most.", "One sentence."))
    st = _req(port, "GET", "/api/state")[1]
    assert [v["label"] for v in st["program"]["versions"]] == [1, 2] and st["program"]["version"] == 2
    seen = _req(port, "GET", f"/api/version?hash={v1['hash']}")[1]
    assert seen["code"] == original and any(line.startswith("+") and "One sentence." in line for line in seen["diff"])
    assert _req(port, "POST", "/api/restore", {"hash": v1["hash"]}, h)[1]["ok"]
    assert club.file.read_text() == original
    st = _req(port, "GET", "/api/state")[1]
    assert st["program"]["version"] == 1 and all(v["code_kept"] for v in st["program"]["versions"])
    assert _req(port, "POST", "/api/restore", {"hash": "sha256:nope"}, h)[0] == 400


# ------------------------------------------------------------------ edits in the dashboard, written into the code

def test_edits_change_the_code_precisely_and_undo(club):
    from functai.studio import edit
    original = club.file.read_text()
    base = lambda: edit.sha(club.file.read_text())  # noqa: E731
    edit.set_model(club, "kind", "claude-sonnet-5-5", base=base())
    edit.set_instruction(club, "kind", "What kind of message is it?\nAsking who may come is a question.", base=base())
    edit.set_answer(club, "kind", "choices", ["question", "idea", "spam", "complaint"], base=base())
    edit.set_input(club, "reply", "message", "draft", base=base())
    edit.add_step(club, "answer", "shorten", "Make it one sentence.", base=base())
    edit.add_step(club, "(end)", "translate", "Translate it to French.", extra=["message"], base=base())
    club.load()
    code = club.file.read_text()
    assert '@ai(lm="claude-sonnet-5-5")\ndef kind(message: str) -> Literal["question", "idea", "spam", "complaint"]:' in code
    assert '    """What kind of message is it?\n    Asking who may come is a question.\n    """' in code
    assert "draft = shorten(answer(message, FAQ)) if" in code and "return translate(reply(draft, draft), message)" in code
    assert "def translate(result: str, message: str) -> str:" in code
    assert set(club.steps()) == {"kind", "shorten", "answer", "translate", "reply"}
    g = build_map(club)
    assert ("answer", "shorten") in {(e["from"], e["to"]) for e in g["edges"]}
    edit.remove_step(club, "shorten", base=base())
    assert "shorten" not in club.file.read_text()
    with pytest.raises(StudioError, match="cannot simply be taken out"):
        edit.remove_step(club, "kind", base=base())
    with pytest.raises(edit.Conflict):
        edit.set_instruction(club, "kind", "x", base="stale")
    with pytest.raises(StudioError, match="not be valid Python"):
        edit.set_source(club, "def (", base=base())
    with pytest.raises(StudioError, match="not available"):
        edit.set_input(club, "reply", "message", "nonsense", base=base())
    while edit.history(club):
        edit.undo(club)
    assert club.file.read_text() == original


def test_values_and_files_can_be_changed_and_undone(club):
    from functai.studio import edit
    g = build_map(club)
    idea = next(n for n in g["nodes"] if n["kind"] == "value" and not n.get("returns"))
    edit.set_value(club, idea["line"], idea["col"], "Great idea!", base=edit.sha(club.file.read_text()))
    assert 'else "Great idea!"' in club.file.read_text()
    faq = club.root / "club_faq.md"
    before = faq.read_text()
    assert edit.wiring(club)["files"] == {"FAQ": "club_faq.md"}
    assert edit.wiring(club)["labels"]["k"] == "what kind gave"
    edit.set_file(club, "FAQ", before + "- Fees: the regional competition costs $20 per team.\n")
    assert "$20" in faq.read_text()
    assert edit.undo(club)["file"] == "club_faq.md" and faq.read_text() == before
    club.file.write_text(club.file.read_text() + "\n# changed by hand\n")
    with pytest.raises(edit.Conflict):            # the last change is no longer the file's state
        edit.undo(club)


def test_review_shows_what_each_model_saw_and_keeps_notes(club):
    from functai.studio.data import example_detail
    running.run_examples(club, club.examples()[:2])
    d = example_detail(club, 2)
    assert [s["name"] for s in d["steps"]] == ["kind", "answer", "reply"]
    answer = d["steps"][1]
    assert answer["inputs"]["question"] == "Can I bring my cousin?" and "Robotics Club FAQ" in answer["inputs"]["faq"]
    assert "Answer using only the club FAQ" in answer["sent"]["system"]
    assert "<question>" in answer["sent"]["messages"][0]["text"] and answer["reply"]["text"]
    functai.rate(answer["call"], "wrong", answer="Yes, if they sign in.", note="too long", by="maxime",
                 folder=club.calls_dir)
    j = example_detail(club, 2)["steps"][1]["judgement"]
    assert j["verdict"] == "wrong" and j["note"] == "too long" and j["answer"] == "Yes, if they sign in."
    assert example_detail(club, 3)["ran"] is False


def test_the_dashboard_edits_through_the_api(server, club):
    studio, port = server
    h = {"X-Studio": "1", "Content-Type": "application/json"}
    st = _req(port, "GET", "/api/state")[1]
    assert st["wiring"]["files"] == {"FAQ": "club_faq.md"} and "Robotics Club" in st["files"]["FAQ"]
    status, out = _req(port, "POST", "/api/edit", {"op": "model", "step": "kind", "model": "gpt-6-luna",
                                                   "base": st["code"]["sha"]}, h)
    assert status == 200 and out["what"] == "kind now uses gpt-6-luna"
    assert _req(port, "POST", "/api/edit", {"op": "model", "step": "kind", "model": "x-model",
                                            "base": st["code"]["sha"]}, h)[0] == 400       # stale base
    st = _req(port, "GET", "/api/state")[1]
    assert st["steps"]["kind"]["lm"] == "gpt-6-luna" and st["edits"][-1]["what"] == "kind now uses gpt-6-luna"
    _req(port, "POST", "/api/edit", {"op": "models", "model": "claude-haiku-5-5", "base": st["code"]["sha"]}, h)
    assert {v["lm"] for v in _req(port, "GET", "/api/state")[1]["steps"].values()} == {"claude-haiku-5-5"}
    assert _req(port, "POST", "/api/undo", {}, h)[1]["undone"] == "every step now uses claude-haiku-5-5"
    running.run_examples(club, club.examples()[:1])
    status, d = _req(port, "GET", "/api/example?n=1")
    assert status == 200 and d["steps"][0]["name"] == "kind" and d["end"]["call"]


# ------------------------------------------------------------------ a new pipeline, examples, the home page

def test_a_plan_becomes_code_that_loads(tmp_path, monkeypatch, fake):
    from functai.studio import newpipe
    monkeypatch.chdir(tmp_path)
    plan = {"name": "Café reviews!", "description": "Reply to reviews.",
            "inputs": [{"name": "review", "about": "one review"}, {"name": "", "about": "dropped"}],
            "facts": [{"name": "menu", "about": "today's menu"}],
            "steps": [{"name": "sort", "instruction": "Praise, suggestion or complaint?", "takes": ["review"],
                       "answer": "choices", "choices": ["praise", "suggestion", "complaint"]},
                      {"name": "flag", "instruction": "Is it about health or safety?", "takes": ["review", "sort"],
                       "answer": "yes/no"},
                      {"name": "reply", "instruction": "Write a kind reply.", "takes": ["review", "sort", "menu", "nope"],
                       "answer": "text"},
                      {"name": "", "instruction": ""}],
            "examples": [{"review": "Great coffee!"}, {"review": ""}, {"review": "Cold soup."}],
            "model": "claude-haiku-5-5"}
    made = newpipe.write(tmp_path, plan)
    assert made == tmp_path / "caf_reviews" / "caf_reviews.py"
    code = made.read_text()
    assert 'def sort(review: str) -> Literal["praise", "suggestion", "complaint"]:' in code
    assert "def reply(review: str, sort: Literal[" in code and 'MENU = open("menu.md"' in code
    assert "reply_result = reply(review, sort_result, MENU)" in code
    assert (tmp_path / "caf_reviews" / "menu.md").read_text().startswith("(Write today's menu here")
    p = find_project(str(made))
    p.load()
    assert list(p.steps()) == ["sort", "flag", "reply"] and len(p.examples()) == 2
    g = build_map(p)
    assert g["flows"] and ("sort", "flag") in {(e["from"], e["to"]) for e in g["edges"]}
    with pytest.raises(StudioError, match="already exists"):
        newpipe.write(tmp_path, plan)
    with pytest.raises(StudioError, match="at least one step"):
        newpipe.check({"name": "x", "steps": [{"name": "a", "instruction": " "}]})


def test_the_ai_helper_drafts_a_plan_and_examples_as_data(tmp_path, monkeypatch, fake, creds):
    from functai.studio import connect, helper, newpipe
    monkeypatch.setattr(connect, "default_model", lambda explicit=None: "claude-haiku-5-5")
    plan = {"name": "notes_quiz", "description": "A quiz from notes.", "inputs": [{"name": "notes", "about": "notes"}],
            "facts": [], "steps": [{"name": "quiz", "instruction": "Write 5 questions.", "takes": ["notes"],
                                    "answer": "list of texts", "choices": []}],
            "examples": [{"notes": "Cells have a nucleus."}]}

    def answer(request):
        system = request.system or ""
        if "Function: plan_pipeline" in system:
            return "<result>\n" + json.dumps(plan) + "\n</result>"
        if "Function: write_examples" in system:
            return "<result>\n" + json.dumps([{"notes": "Mitochondria make energy."}, {"notes": ""}]) + "\n</result>"
        return "<result>\nok\n</result>"
    fake(responder=answer)
    got = helper.draft_plan("a quiz from my notes")
    assert got["steps"][0]["answer"] == "list of texts" and got["model"] == "claude-haiku-5-5"
    made = newpipe.write(tmp_path, got)
    p = find_project(str(made))
    p.load()
    assert helper.draft_examples(p, 2) == [{"notes": "Mitochondria make energy."}]


def test_examples_are_added_pasted_changed_removed_and_undone(club):
    from functai.studio import edit
    first = club.examples_path().read_text()
    edit.add_examples(club, [{"message": "Is there pizza?"}])
    assert club.examples()[-1]["inputs"] == {"message": "Is there pizza?"}
    with pytest.raises(StudioError, match="already there"):
        edit.add_examples(club, [{"message": "Is there pizza?"}])
    r = edit.paste_examples(club, "first line\n\nsecond, with a comma\nIs there pizza?\n")
    assert r["added"] == 2 and r["skipped"] == 1
    r = edit.paste_examples(club, "message\tignored\nfrom a spreadsheet\tx\n")
    assert club.examples()[-1]["inputs"] == {"message": "from a spreadsheet"}
    edit.update_example(club, 1, {"message": "When do we meet?"})
    assert club.examples()[0]["inputs"]["message"] == "When do we meet?"
    edit.delete_example(club, 2)
    assert len(club.examples()) == 12 + 4 - 1
    with pytest.raises(StudioError, match="at least one input"):
        edit.add_examples(club, [{"message": "  "}])
    while edit.history(club):
        edit.undo(club)
    assert club.examples_path().read_text() == first


def test_home_lists_pipelines_makes_one_and_opens_it(tmp_path, monkeypatch, fake, creds):
    import http.server

    from functai.studio.server import Studio, make_handler
    monkeypatch.chdir(tmp_path)
    shutil.copytree(EXAMPLE, tmp_path / "club-inbox", ignore=shutil.ignore_patterns("__pycache__", ".functai"))
    studio = Studio(None, tmp_path, key=None)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(studio))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port, h = httpd.server_address[1], {"X-Studio": "1", "Content-Type": "application/json"}
    try:
        st = _req(port, "GET", "/api/state")[1]
        assert st["project"] is None and [p["path"] for p in st["projects"]] == ["club-inbox/club.py"]
        assert st["projects"][0]["examples"] == 12 and st["accounts"]["connections"] == []
        out = _req(port, "POST", "/api/new", {"plan": {"name": "first", "steps": [
            {"name": "summary", "instruction": "Summarize it.", "takes": ["text"]}]}}, h)[1]
        assert out["made"] == "first/first.py"
        st = _req(port, "GET", "/api/state")[1]
        assert st["program"]["name"] == "first" and st["fill_in"] == []
        _req(port, "POST", "/api/examples", {"op": "paste", "text": "one\ntwo"}, h)
        assert len(_req(port, "GET", "/api/state")[1]["rows"]) == 2
        _req(port, "POST", "/api/projects/close", {}, h)
        st = _req(port, "GET", "/api/state")[1]
        assert st["project"] is None and {p["path"] for p in st["projects"]} == {"club-inbox/club.py", "first/first.py"}
        assert _req(port, "POST", "/api/projects/open", {"path": "../../etc"}, h)[0] == 400
        _req(port, "POST", "/api/projects/open", {"path": "club-inbox/club.py"}, h)
        assert _req(port, "GET", "/api/state")[1]["program"]["name"] == "club_inbox"
    finally:
        httpd.shutdown()
        httpd.server_close()
