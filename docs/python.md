---
rat:
  project: ../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# functai for Python

*Write a Python function. A language model does the work. You measure how well.*

An AI function is an ordinary Python function with `@ai` on top. Its name,
its docstring and its types say what you want; the `...` where the body
would be is the part the model writes. The answer comes back as the type
you asked for. Then you run it on a whole table, find out how often it is
right, and make it better.

```python
from typing import Literal
from dpyr import col
import functai
from functai import ai

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

tickets = functai.datasets.tickets()                 # 80 real-looking support messages
tickets.select(col.message).mutate(team=team(col.message)).slice_head(n=5)
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
# dpyr dataframe · source: polars · showing 5 of 5 rows
┌─────────────────────────────────────────────────────────────────────┬──────────┐
│ message                                                             ┆ team     │
│ ---                                                                 ┆ ---      │
│ str                                                                 ┆ str      │
╞═════════════════════════════════════════════════════════════════════╪══════════╡
│ Hi, my order A-1042 still hasn't arrived and it's been three weeks. ┆ shipping │
│ The mug arrived in pieces.                                          ┆ shipping │
│ I was charged twice for order B-2210, please fix this.              ┆ billing  │
│ How do I change the email on my account?                            ┆ account  │
│ The kettle lid doesn't close properly anymore after a month of use. ┆ product  │
└─────────────────────────────────────────────────────────────────────┴──────────┘
```

How often is it right? The data has the answers:

```python
ev = functai.evaluate(team, tickets, expected="category", num_threads=8)
ev
```

```output
Evaluation(team, 80 examples: exact_match 0.90 [0.81, 0.95])
```

The first number is how often it was right; the range in brackets says
how sure that number is. The misses turn out to be the shop's own rules,
which the model can't guess: write them in the docstring, measure again.
That loop (**write, run, measure, improve**) is what functai is for.

## Where do you start?

<div class="journey" markdown>

- **[I have a table of text](get-started.md)**
  Label, sort or score every row of a table, check it against answers you trust, and make it better. 20 minutes.
- **[I have notes or documents](articles/notes-to-data.md)**
  Field notes, reports, emails: pull out the facts as columns, following your protocol, and check every field.
- **[I have a prompt that works](articles/from-a-prompt.md)**
  Bring your OpenAI messages as they are. Get the same request, then types, tests and a table for free.

</div>

All three meet in the same place: [is it right?](articles/accuracy.md),
[make it better](articles/improving.md), [make it cheaper](articles/cheaper.md),
[ship it](articles/saving.md). To learn it in order, take the
[eight tutorials](tutorials/index.md), from a first function to decision
models and a model you own.

## Installation

```bash
pip install "functai[data]"
```

Python 3.11+. The `[data]` part brings tables (through
[dpyr](https://github.com/MaximeRivest/dpyr), which reads pandas and
polars data frames, CSV, parquet, Excel, databases, and more).

!!! note "Coming from 1.1?"
    [Upgrading](articles/upgrading.md) says what changed in 1.2.

functai needs a model. If you have an API key in your environment
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …) or a Claude,
ChatGPT or Copilot subscription, there is nothing to set up: functai
picks a small, capable model you can use and tells you which. To choose
yourself: `functai.configure(lm="claude-haiku-4-5")`. See
[Models](articles/models.md) and [Signing in](articles/logins.md).

## The words you'll use

- `@ai`: turns a typed function into a model call. The docstring is the instruction; `...` is the body the model writes.
- `fn(col.text)`: runs it on a whole column, each distinct value once, several at a time.
- `evaluate(fn, data, expected=...)`: how often it is right, with an honest range, and a table of every answer.
- `compare(before, after)`: did a change really help, or was it luck?
- `fn.opt(rows)`: an improved copy, with worked examples chosen from your rows; `functai.gepa(fn, rows, teacher=...)` rewrites the instruction from its mistakes instead.
- `save(program, folder)`: everything it needs, in a folder that runs anywhere; TypeScript, R and Julia load its AI functions too.

Every name, with its documentation, is in the [reference](reference/index.md);
what changed, release by release, in the [news](news.md).

## Getting help

Something doesn't work the way this site says? Please
[open an issue](https://github.com/maximerivest/functai/issues) with a
short example. `print(functai.phistory())` shows exactly what the model
was sent and what it answered, which is usually where the answer is.
