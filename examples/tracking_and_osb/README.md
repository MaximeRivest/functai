# Tracking and observability: what was sent, what it cost, how it went


FunctAI keeps a record of every model call, and every evaluation can be
logged as a table. No server to run: the records are Python objects and
parquet files.

Rendered from [`main.qmd`](main.qmd); every output below is a real
reply.

``` python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)

from functai import ai, _ai

@ai
def headline(article: str) -> str:
    """A headline for the article, at most 8 words."""
    angle: str = _ai["The one fact a reader should remember."]
    return _ai
```

## One call

`all=True` returns everything the call produced: each output, the tokens
used across all model calls, and anything the reader had to forgive in
the reply.

``` python
p = headline("The city council voted 7-2 on Tuesday to turn the old rail yard "
             "into a 12-hectare park, with construction starting next spring.", all=True)
p.result, p.angle
```

    ('City council approves 12-hectare park project at old rail yard',
     'City council approves 12-hectare park project')

``` python
p.usage
```

    {'input_tokens': 98,
     'output_tokens': 39,
     'total_tokens': 137,
     'cache_read_tokens': 0,
     'cache_write_tokens': 0,
     'reasoning_tokens': 0}

## The last calls

`phistory` prints the conversation as it was sent; `inspect_history`
returns the records (lm15 requests and responses), newest last:

``` python
print(functai.phistory())
```

    [2026-09-26T13:16:51] headline → gpt-4.1-mini

    System message:

    Function: headline

    A headline for the article, at most 8 words.

    Output guidance:
    - angle: The one fact a reader should remember.

    Reply in exactly this form:
    <angle>
    ...
    </angle>
    <result>
    ...
    </result>


    User message:

    <article>
    The city council voted 7-2 on Tuesday to turn the old rail yard into a 12-hectare park, with construction starting next spring.
    </article>


    Response:

    <angle>
    City council approves 12-hectare park project
    </angle>
    <result>
    City council approves 12-hectare park project at old rail yard
    </result>

    (finish: stop; tokens in 98, out 39)

``` python
rec = functai.inspect_history(1)[0]
rec.function, rec.model, rec.cached, rec.response.finish_reason, rec.response.usage.total_tokens
```

    ('headline', 'gpt-4.1-mini', False, 'stop', 137)

`debug=True` prints one line per call as it happens:

``` python
with functai.configure(debug=True):
    headline("A local bakery has baked the same sourdough loaf every day since 1952.")
```

    [functai] headline: model=gpt-4.1-mini; adapter=functai_xml; outputs=['angle', 'result'] (primary=result); tokens={'input_tokens': 85, 'output_tokens': 41, 'total_tokens': 126, 'cache_read_tokens': 0, 'cache_write_tokens': 0, 'reasoning_tokens': 0}

## Evaluation runs, logged

`evaluate(..., log=folder)` writes each run’s table to the folder as
parquet: one row per example, with the prediction, the metrics, the
error if any, the time and the tokens.

``` python
import tempfile
runs_dir = tempfile.mkdtemp()

articles = [
    {"article": "The river flooded three villages overnight; no one was hurt."},
    {"article": "A 14-year-old won the national chess championship in 9 rounds."},
    {"article": "The museum returned 40 artifacts to their country of origin."},
]

def short(row, prediction):
    return len(prediction.result.split()) <= 8

functai.evaluate(headline, articles, short, log=runs_dir)
headline.temperature = 1.0
functai.evaluate(headline, articles, short, log=runs_dir)
```

    Evaluation(headline, 3 examples: short 1.00 [0.44, 1.00])

`functai.runs` reads every logged run back as one table, so runs compare
like any data (here with [dpyr](https://github.com/MaximeRivest/dpyr)):

``` python
from dpyr import col, n

log = functai.runs(runs_dir)
log.group_by(col.run).summarize(
    examples=n(),
    short=col.short.mean(),
    seconds=col.seconds.mean(),
    tokens=col.output_tokens.sum(),
)
```

    # dpyr dataframe · source: polars · showing 2 of 2 rows
    shape: (2, 5)
    ┌───────────────────────────────┬──────────┬───────┬──────────┬────────┐
    │ run                           ┆ examples ┆ short ┆ seconds  ┆ tokens │
    │ ---                           ┆ ---      ┆ ---   ┆ ---      ┆ ---    │
    │ str                           ┆ i64      ┆ f64   ┆ f64      ┆ i64    │
    ╞═══════════════════════════════╪══════════╪═══════╪══════════╪════════╡
    │ headline-20260926-131656-2499 ┆ 3        ┆ 1.0   ┆ 1.11198  ┆ 100    │
    │ headline-20260926-131658-d9db ┆ 3        ┆ 1.0   ┆ 0.828825 ┆ 100    │
    └───────────────────────────────┴──────────┴───────┴──────────┴────────┘

``` python
log.select(col.run, col.article, col.pred_result).slice_head(6)
```

    # dpyr dataframe · source: polars · showing 6 of 6 rows
    shape: (6, 3)
    ┌───────────────────────────────┬──────────────────────────────────────┬─────────────────────────────────────┐
    │ run                           ┆ article                              ┆ pred_result                         │
    │ ---                           ┆ ---                                  ┆ ---                                 │
    │ str                           ┆ str                                  ┆ str                                 │
    ╞═══════════════════════════════╪══════════════════════════════════════╪═════════════════════════════════════╡
    │ headline-20260926-131656-2499 ┆ The river flooded three villages     ┆ River floods three villages, no     │
    │                               ┆ overnig…                             ┆ injuries…                           │
    │ headline-20260926-131656-2499 ┆ A 14-year-old won the national chess ┆ 14-Year-Old Wins National Chess     │
    │                               ┆ cha…                                 ┆ Champion…                           │
    │ headline-20260926-131656-2499 ┆ The museum returned 40 artifacts to  ┆ Museum Returns 40 Artifacts to      │
    │                               ┆ thei…                                ┆ Country o…                          │
    │ headline-20260926-131658-d9db ┆ The river flooded three villages     ┆ Three villages flooded overnight;   │
    │                               ┆ overnig…                             ┆ no inj…                             │
    │ headline-20260926-131658-d9db ┆ A 14-year-old won the national chess ┆ 14-Year-Old Wins National Chess     │
    │                               ┆ cha…                                 ┆ Champion…                           │
    │ headline-20260926-131658-d9db ┆ The museum returned 40 artifacts to  ┆ Museum Returns 40 Artifacts to      │
    │                               ┆ thei…                                ┆ Origin Co…                          │
    └───────────────────────────────┴──────────────────────────────────────┴─────────────────────────────────────┘

The files are plain parquet: pandas, polars, duckdb or a BI tool read
them directly.

## Not paying twice

With `cache_replies=True`, an identical request is answered from memory
(handy while iterating on a notebook); `inspect_history` marks it:

``` python
with functai.configure(cache_replies=True):
    headline.temperature = 0
    headline("The river flooded three villages overnight; no one was hurt.")
    headline("The river flooded three villages overnight; no one was hurt.")
[r.cached for r in functai.inspect_history(2)]
```

    [False, True]
