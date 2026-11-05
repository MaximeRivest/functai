---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# compare { #functai.compare }

```{.python .no-run}
compare(before, after, *, margin=None)
```

Compare two evaluations case by case: is ``after`` better than ``before``?

The two are paired on the cases both answered (by case key: the rows'
inputs, whatever their order), which detects a real change with far
fewer cases than two separate scores: a case both versions got right says
nothing, a case only one got right says a lot. For right-or-wrong
metrics the range is the MOVER-Wilson interval, which keeps its 95%
where the paired t interval falls to a few percent (with no case that
changed it says "exactly no difference"); otherwise the paired t interval.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type       | Description                                                                                                                                            | Default    |
|--------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| before | Evaluation | Two evaluations (typically two versions of a prompt, or two models) on the same cases, or overlapping ones.                                            | _required_ |
| after  | Evaluation | Two evaluations (typically two versions of a prompt, or two models) on the same cases, or overlapping ones.                                            | _required_ |
| margin | float      | How much worse is still acceptable: the outcome may then be ``same``, ``not_worse`` or ``not_better`` as well as ``better``, ``worse`` or ``unclear``. | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type           | Description                                                                                                                                                                                                                                                                                                                                                                      |
|--------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | dpyr dataframe | One row per metric both have: ``before`` and ``after`` (the means on the shared cases), ``diff`` with its 95% interval ``low`` to ``high``, how many cases got ``better``, ``worse`` or stayed the ``same``, ``n``, ``method``, ``outcome``, ``fragility`` (the fewest cases whose removal would erase the difference), ``only_before`` and ``only_after`` (cases only one has). |

## See Also {.doc-section .doc-section-see-also}

- [`evaluate`](evaluate.md): produces the evaluations.
- `functai.compare_setups`: the same from the call log.

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