---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# evaluate { #functai.evaluate }

```{.python .no-run}
evaluate(
    program,
    data,
    metric=None,
    *,
    expected=None,
    num_threads=1,
    max_errors=None,
    log=None,
    call_defaults=None,
    states=None,
    threads=None,
    progress=False,
)
```

Run a program on rows with known answers, and score it.

Every row runs (in parallel with ``num_threads``); a row that fails
keeps its error and counts 0. The score comes with a 95% interval, and
every answer is kept as a row of a table you can filter and group.

## Parameters {.doc-section .doc-section-parameters}

| Name          | Type                                                 | Description                                                                                                                                                                                                                                                                                                                                       | Default    |
|---------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| program       | AI function or module                                | What to evaluate.                                                                                                                                                                                                                                                                                                                                 | _required_ |
| data          | list of dict, or a table                             | The rows: a list of dicts, or anything ``dpyr.read()`` takes (a parquet or CSV path, a pandas or polars dataframe, a Hugging Face dataset). Columns named like the parameters are the inputs; a column named like an output (``result`` for the return value) is its expected answer, unless ``expected=`` names another; other columns are kept. | _required_ |
| expected      | str or dict                                          | The column holding the right answers, when it isn't named like the output: ``expected="category"``. A dict names a column per output: ``{"result": "category", "order_id": "order"}``. The default metric is then exact match against these columns.                                                                                              | `None`     |
| metric        | function, dpyr expression, AI function, list or dict | How to score a row: ``metric(row, prediction)`` returning a number or a bool, a dpyr expression over the table (``col.pred_result == col.result``), an AI function acting as a judge, or several of these in a list or a dict ``{name: metric}``. Default: exact match on the outputs the data has columns for (case and spacing ignored).        | `None`     |
| num_threads   | int                                                  | How many rows run at once (``threads`` is the same, by the name ``map`` and ``vectorize`` use).                                                                                                                                                                                                                                                   | `1`        |
| threads       | int                                                  | How many rows run at once (``threads`` is the same, by the name ``map`` and ``vectorize`` use).                                                                                                                                                                                                                                                   | `1`        |
| progress      | bool                                                 | A line on stderr, updated as rows finish (None: when stderr is a terminal or in a notebook). Off by default here.                                                                                                                                                                                                                                 | `False`    |
| max_errors    | int                                                  | Stop and raise when more rows than this fail.                                                                                                                                                                                                                                                                                                     | `None`     |
| log           | folder                                               | Write the run's table to ``<log>/<run>.parquet``; ``runs(log)`` reads every logged run back.                                                                                                                                                                                                                                                      | `None`     |
| call_defaults | dict                                                 | Arguments the rows don't have, for every call (a module's options).                                                                                                                                                                                                                                                                               | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type       | Description                                                                                                           |
|--------|------------|-----------------------------------------------------------------------------------------------------------------------|
|        | Evaluation | ``.score`` (the first metric's mean), ``.summary`` (each metric with its interval), ``.table`` (one row per example). |

## See Also {.doc-section .doc-section-see-also}

- [`compare`](compare.md): two evaluations of the same rows, paired.
- [`Evaluation`](Evaluation.md): what this returns.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
from typing import Literal
from dpyr import col

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

tickets = functai.datasets.tickets().slice_head(n=20)
ev = evaluate(team, tickets, expected="category", num_threads=8)
ev
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
Evaluation(team, 20 examples: exact_match 0.90 [0.70, 0.97])
```

```python
ev.table.filter(col.exact_match == 0).select(col.message, col.category, col.pred_result)
```

```output
# dpyr dataframe · source: polars · showing 2 of 2 rows
┌────────────────────────────────────────────────────────────────────┬──────────┬─────────────┐
│ message                                                            ┆ category ┆ pred_result │
│ ---                                                                ┆ ---      ┆ ---         │
│ str                                                                ┆ str      ┆ str         │
╞════════════════════════════════════════════════════════════════════╪══════════╪═════════════╡
│ You sent me a blue rug but I ordered the green one (order A-1187). ┆ shipping ┆ product     │
│ Refund the blender please, it stopped working after two days.      ┆ billing  ┆ product     │
└────────────────────────────────────────────────────────────────────┴──────────┴─────────────┘
```