---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# compare { #functai.compare }

```{.python .no-run}
compare(before, after)
```

Compare two evaluations of the same rows, row by row.

Pairing the rows detects a real change with far fewer examples than two
separate scores would: a row both versions got right says nothing, a row
only one got right says a lot.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type       | Description                                                                           | Default    |
|--------|------------|---------------------------------------------------------------------------------------|------------|
| before | Evaluation | Two evaluations of the same rows (typically two versions of a prompt, or two models). | _required_ |
| after  | Evaluation | Two evaluations of the same rows (typically two versions of a prompt, or two models). | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type           | Description                                                                                                                                                                                                                                                               |
|--------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | dpyr dataframe | One row per metric both have: ``before`` and ``after`` (the means), ``diff`` with its 95% interval ``low`` to ``high`` (a paired t interval), and how many rows got ``better``, ``worse`` or stayed the ``same``. When the interval includes 0, the change could be luck. |

## See Also {.doc-section .doc-section-see-also}

- [`evaluate`](evaluate.md): produces the evaluations.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
from typing import Literal

rows = [
    {"message": "The mug arrived in pieces.", "result": "shipping"},
    {"message": "Box was crushed and the lamp inside is cracked.", "result": "shipping"},
    {"message": "I want my money back for the toaster.", "result": "billing"},
    {"message": "Please refund the blender, it stopped working.", "result": "billing"},
    {"message": "Toaster burns one side of the bread.", "result": "product"},
]

@ai
def category(message: str) -> Literal["shipping", "billing", "product"]:
    """The support category of the message."""
    ...

@ai
def category_v2(message: str) -> Literal["shipping", "billing", "product"]:
    """The support category of the message. An item that arrived broken is
    shipping; any request for money back is billing."""
    ...

compare(evaluate(category, rows, num_threads=5), evaluate(category_v2, rows, num_threads=5))
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬───────┬──────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after ┆ diff ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---   ┆ ---  ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64   ┆ f64  ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪═══════╪══════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.2    ┆ 0.8   ┆ 0.6  ┆ -0.079978 ┆ 1.279978 ┆ 3      ┆ 0     ┆ 2    ┆ 5   │
└─────────────┴────────┴───────┴──────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```