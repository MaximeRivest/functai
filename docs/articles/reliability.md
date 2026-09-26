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
| a reply cut off at the token limit | sends it again with twice the budget | automatic |
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

## The reply cache

While you work in a notebook, re-running a cell calls the model again. To
answer identical requests from memory instead, turn on the cache:

```{.python .no-run}
functai.configure(cache_replies=True)
```

An identical request (same model, same messages, same settings) is then
answered from memory, so re-running a cell or an evaluation costs
nothing. `functai.clear_cache()` empties it. It is off by default,
because a cached answer hides how much a model's answers vary.
