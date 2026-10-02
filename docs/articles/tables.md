---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Big tables

*Any data frame in, AI functions as columns, pandas or polars back out.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

functai works on tables through [dpyr](https://github.com/MaximeRivest/dpyr),
a small data frame library with dplyr's verbs (`mutate`, `filter`,
`group_by`, `summarize`, …) running on polars or duckdb. It is the one
door for every kind of table: whatever you have goes in with `read`, and
comes back out as what you use.

## Any table in

```python
import pandas as pd
from dpyr import read, col, n

df = pd.DataFrame({"review": [
    "Loved it, would buy again.",
    "Arrived late and the box was crushed.",
    "Loved it, would buy again.",
    "Does what it says. Nothing more.",
]})
reviews = read(df)
reviews
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌───────────────────────────────────────┐
│ review                                │
│ ---                                   │
│ str                                   │
╞═══════════════════════════════════════╡
│ Loved it, would buy again.            │
│ Arrived late and the box was crushed. │
│ Loved it, would buy again.            │
│ Does what it says. Nothing more.      │
└───────────────────────────────────────┘
```

`read` takes a pandas or polars data frame, an Arrow table, a Hugging
Face dataset, a list of dicts, a dict of columns, or a path: CSV, TSV,
parquet, JSON, Excel, a DuckDB or SQLite database, a URL.

## AI functions as columns

Called with a column (`col.review`) instead of a value, an AI function
becomes a column expression: use it in `mutate`, `filter`, anywhere a
column goes.

```python
from typing import Literal

@ai
def sentiment(review: str) -> Literal["positive", "neutral", "negative"]:
    """The customer's overall feeling about the product."""
    ...

@ai
def about_delivery(review: str) -> bool:
    """Is the review about the delivery rather than the product?"""
    ...

reviews.mutate(sentiment=sentiment(col.review), delivery=about_delivery(col.review))
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌───────────────────────────────────────┬───────────┬──────────┐
│ review                                ┆ sentiment ┆ delivery │
│ ---                                   ┆ ---       ┆ ---      │
│ str                                   ┆ str       ┆ bool     │
╞═══════════════════════════════════════╪═══════════╪══════════╡
│ Loved it, would buy again.            ┆ positive  ┆ false    │
│ Arrived late and the box was crushed. ┆ negative  ┆ true     │
│ Loved it, would buy again.            ┆ positive  ┆ false    │
│ Does what it says. Nothing more.      ┆ neutral   ┆ false    │
└───────────────────────────────────────┴───────────┴──────────┘
```

Columns and constants mix: `reply(col.message, tone="formal")`.

## Several columns from one call

When the answer is a record, `unpack` gives one column per field:

```python
from dataclasses import dataclass

@dataclass
class Review:
    sentiment: Literal["positive", "neutral", "negative"]
    topic: Literal["product", "delivery", "price", "service"]
    would_recommend: bool | None     # None when the review doesn't say

@ai
def read_review(review: str) -> Review:
    """What this product review says."""
    ...

reviews.mutate(**read_review.unpack(col.review))
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌───────────────────────────────────────┬───────────┬──────────┬─────────────────┐
│ review                                ┆ sentiment ┆ topic    ┆ would_recommend │
│ ---                                   ┆ ---       ┆ ---      ┆ ---             │
│ str                                   ┆ str       ┆ str      ┆ bool            │
╞═══════════════════════════════════════╪═══════════╪══════════╪═════════════════╡
│ Loved it, would buy again.            ┆ positive  ┆ product  ┆ true            │
│ Arrived late and the box was crushed. ┆ negative  ┆ delivery ┆ null            │
│ Loved it, would buy again.            ┆ positive  ┆ product  ┆ true            │
│ Does what it says. Nothing more.      ┆ neutral   ┆ product  ┆ null            │
└───────────────────────────────────────┴───────────┴──────────┴─────────────────┘
```

## How it runs

- **Each distinct input goes to the model once.** Rows 1 and 3 above
  cost one call. Answers are remembered for the session, so running the
  cell again is free.
- **Eight at a time** by default; providers' rate limits are waited out
  and retried.
- **Only what's shown.** A table displayed in a notebook computes the
  rows it shows; the whole column is computed when you save, collect or
  summarize it.
- **The prompt is fixed when the line is written.** Optimizing the
  function afterwards gives new columns new answers, and never mixes old
  and new ones in one column.
- **Failures don't lose work.** A row that fails raises after every other
  row has run; running again retries only the failures. To get nulls
  instead: `errors="null"`.

Options go through `vectorize` (or `unpack`):

```{.python .no-run}
reviews.mutate(sentiment=sentiment.vectorize(threads=16, errors="null")(col.review))
```

## Back out

```python
out = reviews.mutate(sentiment=sentiment(col.review))
out.to_pandas()
```

```output
review sentiment
0             Loved it, would buy again.  positive
1  Arrived late and the box was crushed.  negative
2             Loved it, would buy again.  positive
3       Does what it says. Nothing more.   neutral
```

`.to_polars()`, `.write_parquet("scored.parquet")`, `.write_csv(...)`
work the same way.

## The whole run, with its costs

`fn.map(table)` runs every row and returns the run itself: the
prediction, plus the error, seconds and tokens of each row. It is
[`evaluate`](accuracy.md) without the scoring.

```python
sentiment.map(reviews).select(col.review, col.pred_result, col.input_tokens, col.seconds)
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌───────────────────────────────────────┬─────────────┬──────────────┬──────────┐
│ review                                ┆ pred_result ┆ input_tokens ┆ seconds  │
│ ---                                   ┆ ---         ┆ ---          ┆ ---      │
│ str                                   ┆ str         ┆ i64          ┆ f64      │
╞═══════════════════════════════════════╪═════════════╪══════════════╪══════════╡
│ Loved it, would buy again.            ┆ positive    ┆ 57           ┆ 1.0967   │
│ Arrived late and the box was crushed. ┆ negative    ┆ 59           ┆ 0.897529 │
│ Loved it, would buy again.            ┆ positive    ┆ 57           ┆ 0.967973 │
│ Does what it says. Nothing more.      ┆ neutral     ┆ 58           ┆ 0.867175 │
└───────────────────────────────────────┴─────────────┴──────────────┴──────────┘
```

## Long runs

On ten thousand rows, `map` shows a progress line as rows finish (rows
done, errors, tokens, time left; `progress=False` hides it) and runs
`threads` rows at once. A run that long gets interrupted: a crash, a
laptop lid, a rate limit. Keep the model's replies on disk, and running
the same line again sends only what has no reply yet:

```python
import tempfile, time
functai.configure(cache_replies=tempfile.mkdtemp())   # in real use: cache_replies="disk"

def timed(run):
    start = time.perf_counter()
    table = run()
    return table, round(time.perf_counter() - start, 2)

first, first_seconds = timed(lambda: sentiment.map(reviews, threads=4, progress=False))
again, again_seconds = timed(lambda: sentiment.map(reviews, threads=4, progress=False))   # as after an interruption
first_seconds, again_seconds, list(again.pull(col.pred_result)) == list(first.pull(col.pred_result))
```

```output
(0.85, 0.01, True)
```

The second run cost nothing and took a fraction of the time: every
reply came from the file. `cache_replies="disk"` keeps them in one SQLite file in your cache
folder (`~/.cache/functai/replies.sqlite` on Linux), shared by every
process and notebook; a folder or a `.sqlite` path keeps them there
instead. A reply is kept only once it was read into the function's types,
so an interrupted run leaves nothing half-written, and a row that failed
is asked again. Two processes asking the same thing at once make one
request: the second waits for the first's reply.

A cached reply is the same reply: the model is not asked again, so it
cannot vary. To get another, independent answer to the same request (to
see how much answers vary, or to vote), ask for another replicate; it is
kept too:

```python
another = sentiment.using(replicate=1)("Does what it says. Nothing more.")
functai.configure(cache_replies=False)       # back to no cache, for what follows
another
```

```output
'neutral'
```

`functai.clear_cache("disk")` empties the file. A function whose
`log_content` keeps a field out of the log is never written to disk,
since the file holds whole requests and replies.
