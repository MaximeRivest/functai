---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# When the model gets it wrong

*Repairs, retries, provider errors, the reply cache, and layouts refused before they cost anything.*

Models make small mistakes of form, providers have bad minutes, and some
settings can't work together. functai handles each case in a set way, and
tells you when it did.

| what happens | what functai does | setting |
|---|---|---|
| a slightly misspelled reply (`<Result>` for `<result>`, `**answer**`) | reads it anyway, by one rule, and lists the repair in `prediction.repairs` | always on |
| a reply it can't read | asks again once, with a hint of what was expected | `retries=1`; `retries=0` raises `lmcc.Refusal` |
| a reply cut off at the token limit | sends it again with twice the budget when you set one; without one the reply already had the model's whole limit, so it raises at once, saying how much went to thinking | `max_tokens`, `retries` |
| a provider error: rate limit, 5xx, timeout | sends it again, waiting longer each time | `api_retries=3` |
| a login that expired | raises `LoginRequired` with the command to type; never switches to a paid key | – |
| a tool loop that doesn't finish | raises `StepLimit` | `max_steps=8` |
| an impossible layout (a JSON layout on a model without structured output; several outputs and no reply form) | refuses before any request is sent | – |
| an unknown setting | raises, instead of ignoring it | – |

## `Refusal`: when a reply can't be read

`lmcc.Refusal` carries a `.code` (what went wrong) and a `.hint` (what
was expected). The usual fixes, in order:

1. Look at the reply with `print(functai.phistory())`.
2. Make the output easier to write: a simpler type, a `Literal` instead of
   free text, a comment on the field.
3. Try the `"json"` layout on a model with structured output, where the
   provider enforces the form.

A reply cut off at the token limit (code `parse-truncated`) says why in its
hint: how many of the tokens went to thinking, what the limit was and
whether it can be raised, and what lm15 changed in your request. For
example, on Claude Opus with `reasoning="max"` and no `max_tokens`:

```text
the provider cut the reply at its length limit before field 'result'; the model
spent 128000 of its 128000 output tokens thinking; no max_tokens was set, and lm15
sent 128000, the most it knows this model to allow: lower the reasoning effort or
ask for less (lm15 adapted the request: config.reasoning.thinking_budget dropped: ...)
```

Here raising `max_tokens` can't help: Claude's newer models take no thinking
budget, so the model can think until the limit. Lower the effort (`xhigh`,
`high`) or ask for less.

## The reply cache

While you work in a notebook, re-running a cell calls the model again. To
answer identical requests from memory instead, turn on the cache:

```{.python .no-run}
functai.configure(cache_replies=True)
```

An identical request (same model, same messages, same settings) is then
answered from memory, so re-running a cell or an evaluation costs
nothing. `cache_replies="disk"` keeps the replies in a file instead,
across runs and processes (one request in flight for the same question,
however many processes ask), so a long run resumes by being run again
([Big tables](tables.md#long-runs)). `functai.clear_cache()` empties the
memory cache, `functai.clear_cache("disk")` the file.

A reply that could not be read is not kept, so asking again reaches the
model. The cache is off by default, because a cached answer hides how
much a model's answers vary: an identical request gets an identical
answer even when the model samples (the optimizers' teacher is never
answered from it, for that reason). `fn.using(replicate=1)` asks for a
second, independent answer to the same request (`replicate=2` a third),
cached under its own key.
