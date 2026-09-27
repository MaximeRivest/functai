---
rat:
  project: ../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Get started

*From a table of text to a measured, improved program, in about 20 minutes.*

You have a table with a column of text: support messages, survey answers, reviews, abstracts. You want a new column: a label, a score, a fact pulled out of each row. And you want to know how often it's right.

We'll use `tickets`, 80 messages sent to a small homeware shop, each already labelled with the team that should answer it. It ships with functai, so everything below runs as is (`pip install "functai[data]"`, and a model: see the [home page](index.md#installation)).

## One message

An AI function is an ordinary Python function with `@ai` on top. Its name and docstring say what you want; its return type says what shape the answer must have. It has no body: the model's answer is the return value.

```python
import functai
from functai import ai
from typing import Literal

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""

team("I was charged twice for order B-2210, please fix this.")
```

```output
Traceback (most recent call last):
  File "/home/maxime/.cache/rat/kernels/py@functai/python-kernel.py", line 847, in run_code
    result = eval(compile(expr, "<rat>", "eval"), namespace, namespace)
  File "<rat>", line 9, in <module>
  File "/home/maxime/Projects/functai/functai/core.py", line 758, in __call__
    object.__setattr__(second, "first", pred)
                   ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/maxime/Projects/functai/functai/core.py", line 108, in value
    self._ai_requested = True
  File "/home/maxime/Projects/functai/functai/core.py", line 102, in ensure_materialized
    self._value: Any = None
                 ^^^^^^^^^^
  File "/home/maxime/Projects/functai/functai/core.py", line 636, in _run
    recent = self.history[-window:] if window > 0 else self.history
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/maxime/Projects/functai/functai/core.py", line 686, in _call_model
    outs = [f.name for f in spec.signature.outputs if f.purpose != "tools.calls"]
                             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/maxime/Projects/functai/functai/core.py", line 554, in _plan_for
    return {"inputs": ins, "outputs": outs}
                       ^^^^^^^^^^^^^^^^^^^^
  File "/home/maxime/Projects/functai/functai/models.py", line 225, in resolve
    picked = accounts.default_model(settings.get("auth"))
    if picked is None:
RuntimeError: no model configured: call functai.configure(lm='gpt-4.1-mini') or pass lm=... to @ai (functai.logins() shows what you can use)
```

What the model saw, and what it answered:

```python
print(functai.phistory())
```

```output
(no model calls yet)
```

No model was named, so functai picked one this machine can use, and said so, once (the first line of the output). `phistory()` shows the conversation it sent: the docstring became the instruction, the parameter became a tagged input, and the return type became the allowed answers. You never write that prompt, but you can always read it.

## The whole table

`functai.datasets.tickets()` is a table (a [dpyr](https://github.com/MaximeRivest/dpyr) data frame).

```python
from dpyr import col, n

tickets = functai.datasets.tickets()
tickets
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌─────┬─────────────────────────────────────────────────────────────────────┬─────────┬──────────┬──────────┐
│ id  ┆ message                                                             ┆ channel ┆ category ┆ order_id │
│ --- ┆ ---                                                                 ┆ ---     ┆ ---      ┆ ---      │
│ i64 ┆ str                                                                 ┆ str     ┆ str      ┆ str      │
╞═════╪═════════════════════════════════════════════════════════════════════╪═════════╪══════════╪══════════╡
│ 1   ┆ Hi, my order A-1042 still hasn't arrived and it's been three weeks. ┆ email   ┆ shipping ┆ A-1042   │
│ 2   ┆ The mug arrived in pieces.                                          ┆ chat    ┆ shipping ┆ null     │
│ 3   ┆ I was charged twice for order B-2210, please fix this.              ┆ email   ┆ billing  ┆ B-2210   │
│ 4   ┆ How do I change the email on my account?                            ┆ chat    ┆ account  ┆ null     │
│ 5   ┆ The kettle lid doesn't close properly anymore after a month of use. ┆ email   ┆ product  ┆ null     │
│ 6   ┆ I'd like my money back for the toaster, it burns everything.        ┆ email   ┆ billing  ┆ null     │
│ 7   ┆ Tracking for C-3319 hasn't moved since Monday.                      ┆ chat    ┆ shipping ┆ C-3319   │
│ 8   ┆ I forgot my password and the reset email never comes.               ┆ chat    ┆ account  ┆ null     │
│ 9   ┆ Box was crushed and the lamp inside is cracked. Order D-4001.       ┆ email   ┆ shipping ┆ D-4001   │
│ 10  ┆ My coupon code SPRING10 didn't apply at checkout.                   ┆ chat    ┆ billing  ┆ null     │
└─────┴─────────────────────────────────────────────────────────────────────┴─────────┴──────────┴──────────┘
```

Called on a column instead of a value, `team` makes a new column. Each distinct message goes to the model once, eight at a time, and the answers are remembered for the session.

```python
routed = tickets.mutate(team=team(col.message))
routed.select(col.message, col.team)
```

```output
# dpyr dataframe · source: polars · showing 10 of 80 rows
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
│ I'd like my money back for the toaster, it burns everything.        ┆ billing  │
│ Tracking for C-3319 hasn't moved since Monday.                      ┆ shipping │
│ I forgot my password and the reset email never comes.               ┆ account  │
│ Box was crushed and the lamp inside is cracked. Order D-4001.       ┆ shipping │
│ My coupon code SPRING10 didn't apply at checkout.                   ┆ billing  │
└─────────────────────────────────────────────────────────────────────┴──────────┘
```

It's a column like any other: count, filter, group.

```python
routed.count(col.team)
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌──────────┬─────┐
│ team     ┆ n   │
│ ---      ┆ --- │
│ str      ┆ i64 │
╞══════════╪═════╡
│ account  ┆ 19  │
│ billing  ┆ 17  │
│ product  ┆ 22  │
│ shipping ┆ 22  │
└──────────┴─────┘
```

> **Your own data** `from dpyr import read`, then `read("messages.csv")`, `read("survey.parquet")`,
> `read("replies.xlsx")`, or `read(df)` for a pandas or polars data frame.
> Back out with `.to_pandas()` or `.to_polars()`. See [Big tables](articles/tables.md).

## Is it right?

Looking at a few rows tells you nothing about the other thousand. The table already has the right team for each message, in its `category` column, so let's count.

```python
before = functai.evaluate(team, tickets, expected="category", num_threads=8)
before
```

```output
Evaluation(team, 80 examples: exact_match 0.91 [0.83, 0.96])
```

The score is the share of messages it got right. The two numbers in brackets are the range the true score is probably in: with 80 messages, it can't be pinned down more tightly than that.

Every answer is in a table. The misses:

```python
misses = before.table.filter(col.exact_match == 0)
misses.select(col.message, col.category, col.pred_result)
```

```output
# dpyr dataframe · source: polars · showing 7 of 7 rows
┌─────────────────────────────────────────────────────────────────┬──────────┬─────────────┐
│ message                                                         ┆ category ┆ pred_result │
│ ---                                                             ┆ ---      ┆ ---         │
│ str                                                             ┆ str      ┆ str         │
╞═════════════════════════════════════════════════════════════════╪══════════╪═════════════╡
│ Refund the blender please, it stopped working after two days.   ┆ billing  ┆ product     │
│ Money back please, the knife set is not as sharp as advertised. ┆ billing  ┆ product     │
│ The duvet shrank in the wash, I'd like my money back.           ┆ billing  ┆ product     │
│ Refund please: the towels are much thinner than in the photos.  ┆ billing  ┆ product     │
│ The teapot spout was chipped when it arrived. Order b2610.      ┆ shipping ┆ product     │
│ Return the headphones and give me a refund please.              ┆ billing  ┆ shipping    │
│ I want to cancel my subscription and get this month refunded.   ┆ billing  ┆ account     │
└─────────────────────────────────────────────────────────────────┴──────────┴─────────────┘
```

Read them. They aren't random: something that **arrived broken** went to *product* when this shop sends it to *shipping* (the carrier pays), and **requests for money back** went to whatever they were about, when they all go to *billing*. Those are house rules. No model can guess them.

## Say the rules

The docstring is the instruction, so the rules go there, in plain words.

```python
@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?

    House rules:
    - Anything wrong with the delivery itself, including an item that
      arrived broken, is shipping: the carrier pays.
    - Any request for money back is billing, whatever it is about.
    """

after = functai.evaluate(team, tickets, expected="category", num_threads=8)
functai.compare(before, after)
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬───────┬────────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after ┆ diff   ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---   ┆ ---    ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64   ┆ f64    ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪═══════╪════════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.9125 ┆ 0.975 ┆ 0.0625 ┆ -0.002248 ┆ 0.127248 ┆ 6      ┆ 1     ┆ 73   ┆ 80  │
└─────────────┴────────┴───────┴────────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```

`compare` lines up the two runs message by message. `better` and `worse` count the messages that changed; `low` and `high` bound the improvement. When that range doesn't include 0, the change is real, not luck.

Here: 6 messages better, 1 worse, 6 points up, and the range just touches 0 (from −0.2 to +12.7 points). Almost certainly real, but 80 messages can't quite prove it. When it matters, the next step is more labelled messages, not more wording.

> **When the rule is hard to put into words** Sometimes you can show the rule but not say it. Then give functai
> examples and let it pick the most useful ones (and try better
> instructions): `team.opt(trainset=labelled_rows, expected="category")`.
> See [Make it better](articles/improving.md).

## Several answers at once

The messages often mention an order number, and the table has those too. Asking for a record gets both in one call. The comment on a field is guidance for the model:

```python
from dataclasses import dataclass

@dataclass
class Ticket:
    team: Literal["shipping", "billing", "product", "account"]
    order_id: str | None   # like A-1042: a capital letter, a dash, four digits; None if there is none

@ai
def triage(message: str) -> Ticket:
    """Triage this customer message for a homeware shop.

    House rules:
    - Anything wrong with the delivery itself, including an item that
      arrived broken, is shipping: the carrier pays.
    - Any request for money back is billing, whatever it is about.
    """

tickets.mutate(**triage.unpack(col.message)).select(col.message, col.team, col.order_id)
```

```output
# dpyr dataframe · source: polars · showing 10 of 80 rows
┌─────────────────────────────────────────────────────────────────────┬──────────┬──────────┐
│ message                                                             ┆ team     ┆ order_id │
│ ---                                                                 ┆ ---      ┆ ---      │
│ str                                                                 ┆ str      ┆ str      │
╞═════════════════════════════════════════════════════════════════════╪══════════╪══════════╡
│ Hi, my order A-1042 still hasn't arrived and it's been three weeks. ┆ shipping ┆ A-1042   │
│ The mug arrived in pieces.                                          ┆ shipping ┆ null     │
│ I was charged twice for order B-2210, please fix this.              ┆ billing  ┆ B-2210   │
│ How do I change the email on my account?                            ┆ account  ┆ null     │
│ The kettle lid doesn't close properly anymore after a month of use. ┆ product  ┆ null     │
│ I'd like my money back for the toaster, it burns everything.        ┆ billing  ┆ null     │
│ Tracking for C-3319 hasn't moved since Monday.                      ┆ shipping ┆ C-3319   │
│ I forgot my password and the reset email never comes.               ┆ account  ┆ null     │
│ Box was crushed and the lamp inside is cracked. Order D-4001.       ┆ shipping ┆ D-4001   │
│ My coupon code SPRING10 didn't apply at checkout.                   ┆ billing  ┆ null     │
└─────────────────────────────────────────────────────────────────────┴──────────┴──────────┘
```

`unpack` gives one column per field (still one model call per message). And each field is checked against its own column:

```python
triaged = functai.evaluate(triage, tickets, expected={"team": "category", "order_id": "order_id"},
                           num_threads=8)
triaged.summary
```

```output
# dpyr dataframe · source: polars · showing 3 of 3 rows
┌────────────────┬───────┬──────────┬──────────┬─────┬────────┐
│ metric         ┆ mean  ┆ low      ┆ high     ┆ n   ┆ failed │
│ ---            ┆ ---   ┆ ---      ┆ ---      ┆ --- ┆ ---    │
│ str            ┆ f64   ┆ f64      ┆ f64      ┆ i64 ┆ i64    │
╞════════════════╪═══════╪══════════╪══════════╪═════╪════════╡
│ exact_match    ┆ 0.975 ┆ 0.913356 ┆ 0.993117 ┆ 80  ┆ 0      │
│ team_match     ┆ 0.975 ┆ 0.913356 ┆ 0.993117 ┆ 80  ┆ 0      │
│ order_id_match ┆ 1.0   ┆ 0.954182 ┆ 1.0      ┆ 80  ┆ 0      │
└────────────────┴───────┴──────────┴──────────┴─────┴────────┘
```

`exact_match` counts messages where everything was right; the other rows score each field alone.

## Cheaper?

A smaller model costs less and answers faster. Is it good enough? Same data, same question:

```python
small = functai.evaluate(team.using(lm="gpt-4.1-nano"), tickets, expected="category", num_threads=8)
functai.compare(after, small)
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬────────┬─────────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after  ┆ diff    ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---    ┆ ---     ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64    ┆ f64     ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪════════╪═════════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.975  ┆ 0.9125 ┆ -0.0625 ┆ -0.136297 ┆ 0.011297 ┆ 2      ┆ 7     ┆ 71   ┆ 80  │
└─────────────┴────────┴────────┴─────────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```

The small model got 7 messages wrong that the bigger one got right, and 2 the other way round: 6 points worse, between −14 and +1. Probably worse, not proven; for routing, where a wrong team costs someone's time, the bigger model looks worth it. [Make it cheaper](articles/cheaper.md) compares four models side by side.

`using` makes a copy of the function with other settings; the original is untouched. The table of each run also has the tokens and seconds of every row, so the cost is a sum away: `after.table.summarize(tokens=col.input_tokens.sum())`.

## Keep it

A folder with everything the function needs: its code, the rules, the examples, and the exact packages.

```python
functai.save(team, "support_router/", overwrite=True)
```

```output
team  AI function (message: str → Literal['shipping', 'billing', 'product', 'account'])  [__main__]
└── Literal  (stdlib)

requirements: functai @ file:///home/maxime/Projects/functai

! local-install  requirements: installed from folders on this machine: functai (/home/maxime/Projects/functai)
    fix: the saved program loads where those folders exist; publish them, or install released versions, to load it anywhere
```

Anyone can load it with `functai.load("support_router/", trust=True)` and get the same function. See [Ship it](articles/saving.md).

## Where next

- **Your notes are messier than one label?** [Turn notes into data](articles/notes-to-data.md): several facts per note, a protocol to follow, a score per field.
- **You already have a prompt?** [From a prompt you already have](articles/from-a-prompt.md).
- **Going deeper on this path:** [Is it right?](articles/accuracy.md) · [Make it better](articles/improving.md) · [Make it cheaper](articles/cheaper.md) · [Big tables](articles/tables.md).

