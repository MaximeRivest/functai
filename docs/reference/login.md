# login { #functai.login }

```{.python .no-run}
login(
    provider=None,
    *,
    key=None,
    method=None,
    again=False,
    open_browser=None,
    auth=None,
)
```

Sign in to a provider once; every later session uses it.

For subscriptions (Claude, ChatGPT, GitHub Copilot, xAI, Kimi) a browser
opens when this machine has a screen; over SSH a link or device code is
printed. A Claude Code or Codex CLI already signed in on this machine is
used as is. For API providers, the key is asked for, or given, and saved.
Already signed in: says so and does nothing.

## Parameters {.doc-section .doc-section-parameters}

| Name         | Type        | Description                                                                                                                                                                | Default   |
|--------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| provider     | str         | ``"claude"``, ``"chatgpt"``, ``"copilot"``, ``"grok"``, ``"kimi"``, ``"openrouter"``, or an API provider (``"openai"``, ``"anthropic"``, ``"groq"``...). None: asks which. | `None`    |
| key          | str         | An API key to save, instead of asking.                                                                                                                                     | `None`    |
| method       | str         | One of lm15's sign-in methods for the provider (see ``login_methods``).                                                                                                    | `None`    |
| again        | bool        | Sign in again even when already signed in.                                                                                                                                 | `False`   |
| open_browser | bool        | Default: when this machine has a display.                                                                                                                                  | `None`    |
| auth         | str or path | Another credentials file than lm15's.                                                                                                                                      | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                 |
|--------|--------|---------------------------------------------|
|        | Login  | What was saved, and a model to try with it. |

## See Also {.doc-section .doc-section-see-also}

- [`logins`](logins.md): everything usable right now.
- [`logout`](logout.md): forget a saved login.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
# not run: opens a browser to sign in
functai.login("claude")
functai.configure(lm="claude:claude-sonnet-4-5")

functai.login("groq", key="gsk-...")
```

```output
Already signed in: Login(Claude: saved login, ready, until 2026-09-27 03:33 UTC). Use functai.login('claude', again=True) to sign in again.
Saved the groq API key. Try: functai.configure(lm='groq:openai/gpt-oss-120b')
Login(groq: saved key, ready)
```