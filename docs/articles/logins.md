---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Signing in: subscriptions and keys

*Use your Claude, ChatGPT, Copilot or xAI subscription, or API keys, and see what is available right now.*

functai needs a way to pay for each call: an **API key** from a provider,
or a **subscription** you already have (Claude Pro/Max, ChatGPT Plus/Pro,
GitHub Copilot, xAI). Sign in once; every later session uses it.

## API keys

Set the provider's variable in your environment, and name a model:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

```{.python .no-run}
functai.configure(lm="gpt-4.1-mini")
```

Or save a key once, so you don't need the variable:

```{.python .no-run}
functai.login("openai")                 # asks for the key
functai.login("groq", key="gsk-...")
```

## Subscriptions

```{.python .no-run}
functai.login("claude")       # Claude Pro/Max
functai.login("chatgpt")      # ChatGPT Plus/Pro (the Codex backend)
functai.login("copilot")      # GitHub Copilot
functai.login("grok")         # xAI
functai.login("openrouter")   # approve in the browser; OpenRouter makes a key
functai.login()               # asks which
```

A browser opens when the machine has a screen; over SSH the link (or a
device code) is printed for you to open anywhere. Then name the model
with the account's prefix:

```{.python .no-run}
functai.configure(lm="claude:claude-sonnet-4-5")
functai.configure(lm="chatgpt:gpt-5.5")
functai.configure(lm="copilot:gpt-4.1")
```

**A Claude Code or Codex CLI already signed in on this machine is used
as is**: no new login, nothing copied. `functai.login("claude")` just
records that choice.

Whether a provider allows its subscription to be used this way, and how
it bills it, is the provider's decision. Some sign-ins (GitHub Copilot,
Kimi Code, the Claude and ChatGPT browser logins) are not yet certified by
lm15; functai says so before starting.

## What can I use right now?

`functai.logins()` lists everything available on this machine, and a
model to try with each. It only reads files and environment variables:

```
provider        how                            status                            try
──────────────  ─────────────────────────────  ────────────────────────────────  ────────────────────────
Claude          saved login                    ready until 2026-09-26 19:02 UTC  claude:claude-sonnet-4-5
GitHub Copilot  saved login                    ready until 2026-09-27 11:02 UTC  copilot:gpt-4.1
ChatGPT         Codex CLI                      found                             chatgpt:gpt-5.5
openai          environment ($OPENAI_API_KEY)  found                             gpt-4.1-mini
```

## Which credential a call uses

For each call, the first of these that exists:

1. an explicit `api_key=` (in `@ai`, `using` or `configure`);
2. the saved login or key for that provider;
3. the environment variable, or a CLI login found on the machine.

So a saved key beats the same provider's environment variable.

## When a login runs out

Logins renew themselves. One that expired and can't renew raises
`functai.LoginRequired`, with the command to type
(`functai.login('claude')`). functai **never switches to a paid key on
its own**. A missing key raises the same error, naming the variable to
set.

- `functai.login("claude")` when you are already signed in says so and
  does nothing; `again=True` signs in again.
- `functai.logout("claude")` forgets the saved login. An environment key
  or CLI login for that provider is used again afterwards (except xAI,
  which lm15 blocks on purpose).

## Where logins are kept

In lm15's credentials file, `~/.config/lm15/credentials.json` (or
`$LM15_CREDENTIALS_PATH`), shared with every tool built on lm15.
`configure(auth="path/to/file.json")` uses another file;
`configure(auth=False)` uses none.
