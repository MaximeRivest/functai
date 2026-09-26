---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Make it cheaper

*Compare models on your own data, count what each run costs, and cut tokens where it doesn't hurt.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

The best model for a job is the cheapest one that is right often enough,
**on your data**. Benchmarks can't tell you which that is; twenty minutes
with `evaluate` can.

## The same question, several models

`fn.using(lm=...)` is the same function on another model. Run each on the
same rows:

```python
from typing import Literal
from dpyr import col, read

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?

    House rules: anything wrong with the delivery itself, including an item
    that arrived broken, is shipping. Any request for money back is billing."""

tickets = functai.datasets.tickets()
results = []
for model in ["gpt-4.1-mini", "gpt-4.1-nano", "claude-haiku-4-5", "gemini-2.5-flash"]:
    ev = functai.evaluate(team.using(lm=model), tickets, expected="category", num_threads=8)
    summary = ev.summary.collect().to_dicts()[0]
    cost = ev.table.summarize(input_tokens=col.input_tokens.sum(), output_tokens=col.output_tokens.sum(),
                              seconds=col.seconds.mean()).collect().to_dicts()[0]
    results.append({"model": model, "right": ev.score, "low": summary["low"], "high": summary["high"], **cost})

models = read(results)
models
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌──────────────────┬────────┬──────────┬──────────┬──────────────┬───────────────┬──────────┐
│ model            ┆ right  ┆ low      ┆ high     ┆ input_tokens ┆ output_tokens ┆ seconds  │
│ ---              ┆ ---    ┆ ---      ┆ ---      ┆ ---          ┆ ---           ┆ ---      │
│ str              ┆ f64    ┆ f64      ┆ f64      ┆ i64          ┆ i64           ┆ f64      │
╞══════════════════╪════════╪══════════╪══════════╪══════════════╪═══════════════╪══════════╡
│ gpt-4.1-mini     ┆ 0.9875 ┆ 0.932537 ┆ 0.99779  ┆ 7488         ┆ 720           ┆ 0.711048 │
│ gpt-4.1-nano     ┆ 0.8875 ┆ 0.799818 ┆ 0.939673 ┆ 7791         ┆ 741           ┆ 0.71245  │
│ claude-haiku-4-5 ┆ 0.95   ┆ 0.878377 ┆ 0.980386 ┆ 7741         ┆ 720           ┆ 0.51464  │
│ gemini-2.5-flash ┆ 0.9875 ┆ 0.932537 ┆ 0.99779  ┆ 7411         ┆ 400           ┆ 0.862319 │
└──────────────────┴────────┴──────────┴──────────┴──────────────┴───────────────┴──────────┘
```

Read `right` together with `low` and `high`: two models whose ranges
overlap a lot may be equally good, and then the cheaper one wins. To be
sure about two of them, `compare` their evaluations: pairing the rows
settles it with fewer rows.

## From tokens to money

functai counts tokens exactly; prices are your provider's, and they
change, so bring your own (dollars per million tokens, input and output):

```python
prices = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40),
          "claude-haiku-4-5": (1.00, 5.00), "gemini-2.5-flash": (0.30, 2.50)}   # check yours

for r in results:
    p_in, p_out = prices[r["model"]]
    dollars = (r["input_tokens"] * p_in + r["output_tokens"] * p_out) / 1e6
    r["per_1000_messages"] = round(dollars / len(tickets.collect()) * 1000, 3)
read(results).select(col.model, col.right, col.per_1000_messages)
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌──────────────────┬────────┬───────────────────┐
│ model            ┆ right  ┆ per_1000_messages │
│ ---              ┆ ---    ┆ ---               │
│ str              ┆ f64    ┆ f64               │
╞══════════════════╪════════╪═══════════════════╡
│ gpt-4.1-mini     ┆ 0.9875 ┆ 0.052             │
│ gpt-4.1-nano     ┆ 0.8875 ┆ 0.013             │
│ claude-haiku-4-5 ┆ 0.95   ┆ 0.142             │
│ gemini-2.5-flash ┆ 0.9875 ┆ 0.04              │
└──────────────────┴────────┴───────────────────┘
```

(Those prices are illustrative, as listed by the providers in 2025;
check your own before deciding.)

## Where tokens go

- **Examples are sent with every call.** Ten examples of 50 words each is
  500 words on every row. [Optimization](improving.md) that picks 4 good
  ones is often cheaper *and* better than 16.
- **Reasoning is paid output.** An extra `reasoning` output, or
  `module="cot"`, can help hard cases; measure whether it does before
  paying for it on every row.
- **Short instructions are fine.** The rules that change answers are
  worth their words; politeness isn't.

## Don't pay twice

While you work in a notebook, re-running a cell calls the model again.
`functai.configure(cache_replies=True)` answers identical requests from
memory instead (same model, same prompt, same input), so re-running an
evaluation costs nothing. It's off by default because a cached answer
hides how much a model's answers vary.

On tables, each distinct input is sent once per session anyway: 10,000
rows with 2,000 distinct messages cost 2,000 calls.

## Very large volumes: your own small model

When a function runs millions of times, a small model can be *trained*
to answer it: thousands of rows a second on one GPU, for nothing per
call, with the unsure cases sent to a big model. See
[Bake it into a small model](baking.md).
