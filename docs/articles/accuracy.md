---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Is it right?

*Measure a function on answers you trust: a score with its range, every answer in a table, and fair comparisons.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

A function that looks right on five rows can be wrong on one row in four.
The only way to know is to run it on rows whose right answers you already
have, and count. That is `functai.evaluate`.

## A score, and how sure it is

```python
from typing import Literal
from dpyr import col, n

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

tickets = functai.datasets.tickets()
ev = functai.evaluate(team, tickets, expected="category", num_threads=8)
ev
```

```output
Evaluation(team, 80 examples: exact_match 0.93 [0.85, 0.97])
```

- `expected="category"` says which column holds the right answers. (A
  column named like the output, `result`, is found without it.)
- The first number is the score: the share of rows it got right.
- The two in brackets are the **range**: the true score, on messages like
  these, is very likely between them. Read the range before the score.

## How many rows do I need?

The range narrows as you add rows. For a function that is right 80% of
the time:

```python
from dpyr import read
from functai.evaluation import interval

rows = []
for size in [10, 30, 100, 300, 1000]:
    right = round(0.8 * size)
    _, low, high = interval([1] * right + [0] * (size - right))
    rows.append({"rows": size, "score": "80%", "low": f"{low:.0%}", "high": f"{high:.0%}"})
read(rows)
```

```output
# dpyr dataframe · source: polars · showing 5 of 5 rows
┌──────┬───────┬─────┬──────┐
│ rows ┆ score ┆ low ┆ high │
│ ---  ┆ ---   ┆ --- ┆ ---  │
│ i64  ┆ str   ┆ str ┆ str  │
╞══════╪═══════╪═════╪══════╡
│ 10   ┆ 80%   ┆ 49% ┆ 94%  │
│ 30   ┆ 80%   ┆ 63% ┆ 90%  │
│ 100  ┆ 80%   ┆ 71% ┆ 87%  │
│ 300  ┆ 80%   ┆ 75% ┆ 84%  │
│ 1000 ┆ 80%   ┆ 77% ┆ 82%  │
└──────┴───────┴─────┴──────┘
```

With 30 rows, "80%" means anything from about 60% to 90%. With 100, you
can tell 80% from 70%. For most decisions, **50 to 200 carefully
labelled rows** is the sweet spot: an afternoon of work that tells you
whether to trust the next hundred thousand.

## Every answer, as a table

`ev.table` has one row per example: your columns, the prediction
(`pred_result`), the score, and the error, seconds and tokens of each
call.

```python
ev.table.filter(col.exact_match == 0).select(col.message, col.category, col.pred_result)
```

```output
# dpyr dataframe · source: polars · showing 6 of 6 rows
┌─────────────────────────────────────────────────────────────────┬──────────┬─────────────┐
│ message                                                         ┆ category ┆ pred_result │
│ ---                                                             ┆ ---      ┆ ---         │
│ str                                                             ┆ str      ┆ str         │
╞═════════════════════════════════════════════════════════════════╪══════════╪═════════════╡
│ Refund the blender please, it stopped working after two days.   ┆ billing  ┆ product     │
│ I want a refund for the chair, it wobbles no matter what I do.  ┆ billing  ┆ product     │
│ Money back please, the knife set is not as sharp as advertised. ┆ billing  ┆ product     │
│ The duvet shrank in the wash, I'd like my money back.           ┆ billing  ┆ product     │
│ Refund please: the towels are much thinner than in the photos.  ┆ billing  ┆ product     │
│ Return the headphones and give me a refund please.              ┆ billing  ┆ shipping    │
└─────────────────────────────────────────────────────────────────┴──────────┴─────────────┘
```

Group it like any table. By the right answer:

```python
ev.table.group_by(col.category).summarize(right=col.exact_match.mean(), rows=n())
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌──────────┬──────────┬──────┐
│ category ┆ right    ┆ rows │
│ ---      ┆ ---      ┆ ---  │
│ str      ┆ f64      ┆ i64  │
╞══════════╪══════════╪══════╡
│ account  ┆ 1.0      ┆ 18   │
│ billing  ┆ 0.727273 ┆ 22   │
│ product  ┆ 1.0      ┆ 18   │
│ shipping ┆ 1.0      ┆ 22   │
└──────────┴──────────┴──────┘
```

By where the message came from:

```python
ev.table.group_by(col.channel).summarize(right=col.exact_match.mean(), rows=n())
```

```output
# dpyr dataframe · source: polars · showing 2 of 2 rows
┌─────────┬───────┬──────┐
│ channel ┆ right ┆ rows │
│ ---     ┆ ---   ┆ ---  │
│ str     ┆ f64   ┆ i64  │
╞═════════╪═══════╪══════╡
│ chat    ┆ 0.9   ┆ 40   │
│ email   ┆ 0.95  ┆ 40   │
└─────────┴───────┴──────┘
```

`ev.write("run.parquet")` saves it; `ev.table.to_pandas()` hands it to
pandas.

## When "right" isn't one word

Exact match suits labels. For other answers, say what "right" means:

| what "right" means | the metric |
|---|---|
| equals the answer (case and spacing ignored) | nothing: the default |
| each field of a record equals its column | nothing: the default, one score per field ([example](notes-to-data.md#check-every-field)) |
| any rule you can code | a function `metric(row, prediction)` returning a number or `True`/`False` |
| any rule over the table's columns | a [dpyr](https://github.com/MaximeRivest/dpyr) expression, like `col.pred_result == col.category` |
| a judgment (tone, faithfulness, completeness) | another AI function, acting as the judge |

Several at once, as a dict; the first one is `ev.score`:

```python
@ai
def reply(message: str) -> str:
    """A short, friendly reply to the customer, saying what happens next."""
    ...

@ai
def judge(row: dict, prediction: dict) -> bool:
    """Does the reply address the customer's actual problem, politely, without promising a refund?"""
    ...

replies = functai.evaluate(reply, tickets.slice_head(n=12), {
    "judge": judge,
    "short": lambda row, prediction: len(prediction.result.split()) <= 60,
}, num_threads=8)
replies.summary
```

```output
# dpyr dataframe · source: polars · showing 2 of 2 rows
┌────────┬──────────┬──────────┬──────────┬─────┬────────┐
│ metric ┆ mean     ┆ low      ┆ high     ┆ n   ┆ failed │
│ ---    ┆ ---      ┆ ---      ┆ ---      ┆ --- ┆ ---    │
│ str    ┆ f64      ┆ f64      ┆ f64      ┆ i64 ┆ i64    │
╞════════╪══════════╪══════════╪══════════╪═════╪════════╡
│ judge  ┆ 0.416667 ┆ 0.19326  ┆ 0.680489 ┆ 12  ┆ 0      │
│ short  ┆ 1.0      ┆ 0.757506 ┆ 1.0      ┆ 12  ┆ 0      │
└────────┴──────────┴──────────┴──────────┴─────┴────────┘
```

A judge is itself a model, so it can be wrong: check a few of its
verdicts by hand (`replies.table.select(col.message, col.pred_result, col.judge)`)
before trusting its average.

A judge that must quote its evidence is easier to check, and
`functai.quotes_found` checks the quotes for you, for free: each must be
in the source word for word (spacing, case, curly quotes and dashes
aside). A judge that paraphrased, or made a sentence up, is caught:

```python
@ai
def supported(source: str, claim: str) -> bool:
    """Is the claim supported by the source?"""
    evidence: list[str] = _ai["sentences copied word for word from the source that support or contradict the claim"]
    return _ai

source = "The parcel left Leeds on Monday. It was delayed by snow near Carlisle and arrived on Friday."
verdict = supported.predict(source, "Snow slowed the parcel down.")
verdict.result, verdict.evidence, functai.quotes_found(source, verdict.evidence)
```

```output
(True, ['It was delayed by snow near Carlisle'], [True])
```

## Did a change help?

Evaluate both versions **on the same rows**, then compare. `compare`
pairs the rows, which detects a real change with far fewer rows than two
separate scores would.

```python
@ai
def team_v2(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message? An item that arrived
    broken is shipping; any request for money back is billing."""
    ...

change = functai.compare(ev, functai.evaluate(team_v2, tickets, expected="category", num_threads=8))
change
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬───────┬────────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after ┆ diff   ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---   ┆ ---    ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64   ┆ f64    ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪═══════╪════════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.925  ┆ 0.9   ┆ -0.025 ┆ -0.074761 ┆ 0.024761 ┆ 1      ┆ 3     ┆ 76   ┆ 80  │
└─────────────┴────────┴───────┴────────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```

`better` and `worse` count rows that changed. When `low` to `high`
doesn't include 0, the change is real. When it does, the honest next step
is more rows, not more tinkering.


Here the one-line rules moved 3 rows the right way and 2 the wrong way:
+1 point, somewhere between −4 and +7. That can't be told apart from luck.
[Get started](../get-started.md#say-the-rules) spells the same rules out
in full and measures a clear gain: wording matters, and only measuring
tells you which wording works.

## Keep a record

`evaluate(..., log="runs/")` writes each run to `runs/<name>.parquet`;
`functai.runs("runs/")` reads them all back as one table: every version
you tried, side by side, with its score.

## Don't grade on the homework

When you [improve a function from examples](improving.md), score it on
*other* rows. A score on the rows it learned from says how well it
memorized them, not how well it works.
