---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Models and settings

*Choose any model from any provider, and set it for the whole program, a block, a function, or one copy.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

An AI function says *what* you want. Which model answers it, and how, is a
setting, and settings can change without touching the function.

## If you set nothing

functai looks at what this machine can use (API keys in the environment,
saved keys, subscription logins) and picks a small, capable model,
preferring API keys, in this order: `gpt-4.1-mini` (OpenAI),
`claude-haiku-4-5` (Anthropic), `gemini-2.5-flash` (Google), Groq,
OpenRouter, then your Claude, ChatGPT or Copilot subscription. It says
which, once:

```
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
```

For anything you'll run twice, choose explicitly: the default depends on
the machine.

## Naming a model

A model is a string. The provider is found from the name, or given as a
prefix:

| model | provider | key it uses |
|---|---|---|
| `"gpt-4.1-mini"`, `"o4-mini"`, `"gpt-5"` | OpenAI | `OPENAI_API_KEY` |
| `"claude-haiku-4-5"`, `"claude-sonnet-4-5"` | Anthropic | `ANTHROPIC_API_KEY` |
| `"gemini-2.5-flash"` | Google | `GEMINI_API_KEY` |
| `"groq:openai/gpt-oss-120b"` | Groq | `GROQ_API_KEY` |
| `"openrouter:qwen/qwen3-32b"` | OpenRouter | `OPENROUTER_API_KEY` |
| `"ollama:qwen3:8b"` | a local Ollama | none |
| `"claude:claude-sonnet-4-5"`, `"chatgpt:gpt-5.5"` | your subscription | a [login](logins.md) |

The litellm spellings (`"openai/gpt-4o"`, `"anthropic/claude-sonnet-4-5"`)
work too. Requests go straight to each provider's API through
[lm15](https://github.com/lm15-dev/lm15-python); no provider SDK is
installed.

## Where settings come from

Settings are looked up at every call, from the most specific place to
the most general. The first place that sets a value wins:

1. **one copy:** `fn.using(lm="…")` returns a copy with other settings;
2. **the function:** `@ai(lm="…")`, or `fn.lm = "…"` later;
3. **a block:** `with functai.configure(lm="…"):`;
4. **the whole program:** `functai.configure(lm="…")`.

```python
@ai
def capital(country: str) -> str:
    """The country's capital city."""
    ...

capital("Australia")                       # from configure(): gpt-4.1-mini
```

```output
'Canberra'
```

```python
with functai.configure(lm="gpt-4.1-nano"):
    print(capital("Canada"))               # this block only
```

```output
Ottawa
```

```python
fast = capital.using(lm="gpt-4.1-nano", temperature=0)
fast("Brazil")                             # a copy; `capital` is unchanged
```

```output
'Brasília'
```

`functai.settings` reads the effective values:

```python
functai.settings.lm
```

```output
'gpt-4.1-mini'
```

A `with configure(...)` block applies to the current thread and to the
threads functai starts from it (for example in `evaluate(num_threads=8)`),
not to threads started elsewhere.

## Changing a function after the fact

Everything can be set when the function is defined, changed on it later,
or changed on a copy, which leaves the original alone:

| | at definition | afterwards | on a copy |
|---|---|---|---|
| model | `@ai(lm="gpt-4.1")` | `fn.lm = "gpt-4.1"` | `fn.using(lm="gpt-4.1")` |
| connection | `@ai(client=...)` | `fn.using(client=...)` | `fn.using(client=...)` |
| layout | `@ai(adapter="chat")` | `fn.adapter = "chat"` | `fn.using(adapter="chat")` |
| chat template | `@ai(template=[...])` | `fn.template = [...]` | `fn.using(template=[...])` |

A bad value (an unknown layout, a template that can't be read back, a
connection object where a model name belongs) is refused where you write
it, before anything changes.

## The model and the connection are separate

`lm=` names the model. `client=` is *how to reach it*, for the rare case
you build that yourself with lm15: a second account's key, a proxy, a
company gateway.

```{.python .no-run}
import lm15

@ai(lm="gpt-4.1-mini", client=lm15.OpenAILM(api_key=OTHER_KEY))
def f(text: str) -> str: ...
```

A model prefix naming another provider than the client's is refused.
Credentials and connections are never [saved](saving.md) with a
program: the machine that loads it uses its own.

## All settings

| setting | meaning |
|---|---|
| `lm` | the model |
| `api_key`, `base_url` | for the provider `lm` points to; an explicit key beats every other |
| `auth` | saved logins: on by default; a path for another credentials file; `False` for none |
| `client` | the lm15 connection, when you build it yourself (below) |
| `temperature`, `max_tokens`, `seed`, `top_p`, `stop`, … | sampling: any lm15 `Config` field |
| `adapter`, `template` | the [prompt format](layouts.md) |
| `module` | `"predict"` (default), `"cot"` ([reasoning first](outputs.md)), `"react"` |
| `tools`, `max_steps`, `tool_errors` | the [tool loop](tools.md) |
| `stateful`, `state_window` | [memory](memory.md) |
| `retries`, `api_retries`, `cache_replies` | [reliability](reliability.md) |
| `capabilities` | what the model can do, when you know better than functai's table |
| `optimizer`, `teacher`, `teacher_lm` | [optimization](improving.md) defaults |
| `debug` | print one line per call |

A misspelled setting is an error, not silently ignored:

```python
try:
    functai.configure(temprature=0)
except TypeError as error:
    print(error)
```

```output
configure: unknown setting(s) ['temprature']. functai settings: ['adapter', 'api_key', 'api_retries', 'auth', 'autocompile', 'autocompile_n', 'autogen_instructions', 'autoinstruct', 'base_url', 'cache_replies', 'capabilities', 'client', 'debug', 'escalate_below', 'escalate_to', 'include_fn_name_in_instructions', 'instruction_autorefine_calls', 'instruction_autorefine_max_examples', 'instruction_lm', 'lm', 'max_steps', 'module', 'on_unreadable', 'optimizer', 'retries', 'state_window', 'stateful', 'teacher', 'teacher_lm', 'tool_errors']; lm15 Config fields: ['cache', 'extensions', 'frequency_penalty', 'logprobs', 'max_tokens', 'presence_penalty', 'probabilities', 'reasoning', 'response_format', 'seed', 'service_tier', 'stop', 'store', 'temperature', 'tool_choice', 'top_k', 'top_p', 'user_id']
```

Settings some models refuse (OpenAI's reasoning models take no
`temperature`; the ChatGPT backend no `max_tokens`) are left out of those
requests with a one-time warning, so one `configure(...)` works across
models.
