---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# From a prompt you already have

*Bring your OpenAI-style messages as they are. Get the same request, then types, tables and measurement.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

You have a prompt that works. It took a while to get right, and you don't
want a library rewriting it. Good: functai can send it **exactly as it
is**. Then, one step at a time and only if the numbers say so, you can
let functai take over the parts that are chores: parsing, retries, running
it over a table, measuring it.

## Your prompt today

Something like this, with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()
SYSTEM = ("You route customer messages for a homeware shop. "
          "Answer with one word: shipping, billing, product or account.")

def route(message):
    reply = client.chat.completions.create(
        model="gpt-4.1-mini", temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": message}])
    return reply.choices[0].message.content.strip().lower()
```

## The same messages, in functai

Paste the messages into `template=`, and turn each f-string hole into a
`{name}` that matches a parameter:

```python
SYSTEM = ("You route customer messages for a homeware shop. "
          "Answer with one word: shipping, billing, product or account.")

@ai(template=[
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "{message}"},
])
def route(message: str) -> str:
    """Route the message."""
```

Don't take our word that nothing was added. `render` builds the request
without sending it:

```python
request = route.render("The mug arrived in pieces.")
print("system:", request.system)
for m in request.messages:
    print(f"{m.role}:", m.parts[0].text)
```

```output
system: You route customer messages for a homeware shop. Answer with one word: shipping, billing, product or account.
user: The mug arrived in pieces.
```

Byte for byte what your code sent. (The docstring isn't in it: your
template decides what the model sees.) And it's still a function:

```python
route("The mug arrived in pieces.")
```

```output
'product'
```

## What you get without changing the prompt

**It runs on a table**, each distinct message once, several at a time:

```python
from dpyr import col

tickets = functai.datasets.tickets()
tickets.select(col.message).mutate(team=route(col.message)).slice_head(n=5)
```

```output
# dpyr dataframe · source: polars · showing 5 of 5 rows
┌─────────────────────────────────────────────────────────────────────┬──────────┐
│ message                                                             ┆ team     │
│ ---                                                                 ┆ ---      │
│ str                                                                 ┆ str      │
╞═════════════════════════════════════════════════════════════════════╪══════════╡
│ Hi, my order A-1042 still hasn't arrived and it's been three weeks. ┆ shipping │
│ The mug arrived in pieces.                                          ┆ product  │
│ I was charged twice for order B-2210, please fix this.              ┆ billing  │
│ How do I change the email on my account?                            ┆ account  │
│ The kettle lid doesn't close properly anymore after a month of use. ┆ product  │
└─────────────────────────────────────────────────────────────────────┴──────────┘
```

**It can be measured** against answers you trust:

```python
yours = functai.evaluate(route, tickets, expected="category", num_threads=8)
yours
```

```output
Evaluation(route, 80 examples: exact_match 0.76 [0.66, 0.84])
```

**Every call is inspectable**: `print(functai.phistory())` shows the
conversation, `route.render(...)` the request before it's sent.

**Provider errors are retried**, and any model works by changing
`lm=`: `route.using(lm="claude-haiku-4-5")`.

## Step 1: a return type instead of `.strip().lower()`

Your code cleans the reply by hand, and trusts that it is one of the four
words. Say so in the type instead:

```python
from typing import Literal

@ai(template=[
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "{message}"},
])
def route_typed(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Route the message."""

route_typed("THE MUG ARRIVED IN PIECES!!")
```

```output
'product'
```

The reply is read into the type: an answer that isn't one of the four is
repaired if it's close ("Shipping." → `"shipping"`), asked again once if
it isn't, and never slips through as a sentence.

## Step 2: let the function be the prompt

Your template carries two things: an instruction, and the list of
answers. The function can carry both, the instruction in the docstring
and the answers in the type. Then functai writes the messages:

```python
@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Route customer messages for a homeware shop to the team that answers them."""
    ...

team("The mug arrived in pieces.")
```

```output
'product'
```

What the model saw, and what it answered:

```python
print(functai.phistory())
```

```output
[2026-10-02T09:57:20] team → gpt-4.1-mini

System message:

Function: team

Route customer messages for a homeware shop to the team that answers them.

Reply in exactly this form:
<result>
one of: shipping, billing, product, account
</result>


User message:

<message>
The mug arrived in pieces.
</message>


Response:

<result>
product
</result>

(finish: stop; tokens in 65, out 9)
```

Is that as good as your hand-written prompt? Don't guess, compare, on the
same messages:

```python
functai.compare(yours, functai.evaluate(team, tickets, expected="category", num_threads=8))
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬────────┬──────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after  ┆ diff ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---    ┆ ---  ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64    ┆ f64  ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪════════╪══════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.7625 ┆ 0.8125 ┆ 0.05 ┆ -0.010298 ┆ 0.110298 ┆ 5      ┆ 1     ┆ 74   ┆ 80  │
└─────────────┴────────┴────────┴──────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```

If yours is better, keep your template: it is a first-class way to use
functai, not a beginner's mode. If they're the same, the function form is
easier to change, to optimize, and to extend with more outputs.

## Your OpenAI habits, in functai

| with the OpenAI SDK | in functai |
|---|---|
| `model="gpt-4.1-mini"` | `@ai(lm="gpt-4.1-mini")`, or `functai.configure(lm=...)` for all |
| `messages=[...]` with f-strings | `template=[...]` with `{name}` holes, or the docstring |
| few-shot user/assistant pairs | `@ai(examples=[...])`, or `functai.bootstrap_few_shot(fn, rows)` to pick them from data |
| `response_format` / JSON schema | the return type: a `Literal`, a dataclass, a pydantic model |
| parsing and retrying bad JSON | done for you; `retries=` |
| `tools=[{json schema}]` and the call loop | `tools=[a_python_function]`; the loop is run for you |
| `temperature=0` | `temperature=0` |
| a loop over rows, a thread pool | `fn(col.text)`, `fn.map(table)`, `evaluate(..., num_threads=8)` |
| "does the new prompt do better?" | `compare(evaluate(old, data), evaluate(new, data))` |

Templates can do more than paste: loops over inputs, blocks shown only
when an input has a value, a prefilled start of the answer. See
[Prompt formats and chat templates](layouts.md).

## Where next

- [Is it right?](accuracy.md): metrics beyond exact match, including a model as the judge.
- [Make it better](improving.md): let functai choose examples and try instructions.
- [Tools](tools.md): your Python functions, called by the model.
