"""Signing in from functai: a temporary lm15 credentials file, no network."""

import lm15
import pytest

import functai
from functai import _ai, accounts, ai, models


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(accounts, "CLI_LOGINS", {k: (name, tmp_path / f"no-{k}.json")
                                                 for k, (name, _p) in accounts.CLI_LOGINS.items()})
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return path


def test_a_saved_key_is_listed_and_used(store, capsys):
    record = functai.login("groq", key="gsk-test", auth=store)
    assert record.provider == "groq" and record.source == "saved key" and record.status == "ready"
    assert "Try: functai.configure(lm='groq:openai/gpt-oss-120b')" in capsys.readouterr().out
    rows = functai.logins(auth=store)
    assert [r.provider for r in rows] == ["groq"]
    assert "saved key" in repr(rows)

    router, model, route = models.resolve({"lm": "groq:openai/gpt-oss-120b", "auth": store})
    assert route.provider == "groq"
    assert router is models._shared(("auth", id(accounts.auth_for(store))), None)   # the saved-login router

    router, _m, _r = models.resolve({"lm": "gpt-4.1-mini", "auth": store})
    assert router is models._shared(("env",), None)                                    # nothing saved for openai
    router, _m, _r = models.resolve({"lm": "groq:x", "auth": False})
    assert router is models._shared(("env",), None)                                    # auth=False: never


def test_an_explicit_api_key_beats_the_saved_login(store):
    functai.login("groq", key="gsk-saved", auth=store)
    router, _m, _r = models.resolve({"lm": "groq:x", "auth": store, "api_key": "gsk-explicit"})
    assert router not in (models._shared(("env",), None), models._shared(("auth", id(accounts.auth_for(store))), None))


def test_already_signed_in_does_nothing_and_logout_forgets(store, capsys):
    functai.login("groq", key="gsk-1", auth=store)
    functai.login("groq", auth=store)
    assert "Already signed in" in capsys.readouterr().out
    functai.logout("groq", auth=store)
    assert functai.logins(auth=store) == []
    functai.logout("groq", auth=store)
    assert "Not signed in" in capsys.readouterr().out


def test_friendly_names():
    assert accounts.canonical("Claude") == "claude-code"
    assert accounts.canonical("chatgpt") == "openai-codex"
    assert accounts.canonical("copilot") == "github-copilot"
    assert models.model_string("claude:claude-sonnet-4-5") == "claude-code:claude-sonnet-4-5"
    assert models.model_string("copilot:gpt-4.1") == "github-copilot:gpt-4.1"
    assert models.model_string("groq:openai/gpt-oss-120b") == "groq:openai/gpt-oss-120b"
    with pytest.raises(ValueError, match="unknown provider"):
        functai.login("nope", key="x")


def test_a_cli_login_on_this_machine_is_the_default_way_in(store, tmp_path, monkeypatch):
    cli = tmp_path / "claude.json"
    cli.write_text("{}")
    monkeypatch.setitem(accounts.CLI_LOGINS, "claude-code", ("Claude Code CLI", cli))
    a = accounts.auth_for(store)
    assert accounts._choose_method(a, "claude-code", None) == ("external:claude-code-cli", False)
    cli.unlink()
    method, unverified = accounts._choose_method(a, "claude-code", None)
    assert method is None and unverified            # two browser logins remain: the picker asks, with a note
    assert accounts._choose_method(a, "github-copilot", None) == ("device", True)
    assert accounts._choose_method(a, "openai", None) == ("api_key", False)


def test_subscription_providers_get_their_api_s_abilities():
    assert models.capabilities("claude-code", "claude-sonnet-4-5") == models.capabilities("anthropic", "claude-sonnet-4-5")
    assert models.capabilities("openai-codex", "gpt-5.5") == models.capabilities("openai", "gpt-5.5")
    assert models.capabilities("github-copilot", "gpt-4.1")["native_function_calling"] is True


def test_codex_requests_leave_out_max_tokens(fake):
    @ai(max_tokens=100)
    def f(x: str) -> str:
        """Echo."""
        return _ai

    r = fake("<result>\nok\n</result>", lm="openai-codex:gpt-5.5")
    with pytest.warns(UserWarning, match="max_tokens"):
        assert f("x") == "ok"
    assert r.requests[0].config is None or r.requests[0].config.max_tokens is None


def test_a_missing_credential_says_how_to_sign_in(fake):
    @ai
    def f(x: str) -> str: ...

    fake(lm15.MissingCredentialError("no key", provider="github-copilot"), lm="copilot:gpt-4.1")
    with pytest.raises(functai.LoginRequired) as err:
        f("x")
    assert "functai.login('copilot')" in str(err.value) and err.value.provider == "github-copilot"

    fake(lm15.MissingCredentialError("no key", provider="openai", env_keys=("OPENAI_API_KEY",)))
    with pytest.raises(functai.LoginRequired, match=r"functai.login\('openai'\), set \$OPENAI_API_KEY"):
        f("y")


def test_a_spent_key_is_not_reported_as_a_missing_login(fake):
    @ai
    def f(x: str) -> str: ...
    fake(lm15.AuthError("Key limit exceeded", status=403), lm="openrouter:x")
    with pytest.raises(lm15.AuthError, match="limit exceeded"):
        f("x")
    fake(lm15.AuthError("invalid key", status=401), lm="openrouter:y")
    with pytest.raises(functai.LoginRequired):
        f("y")
