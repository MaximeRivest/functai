---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# rated { #functai.rated }

```{.python .no-run}
rated(program, *, folder=None, by=None, since=None, any_file=False)
```

The calls people rated, as rows with known answers.

Each row is a call someone judged: its inputs under their names, and
the right answer under the output's name (the answer it gave, when it
was marked right; the correction, when it was marked wrong and
corrected). Hand it to ``evaluate`` or ``.opt`` as it is.

A call marked wrong without the right answer is left out: it says what
the answer isn't, not what it is. So are calls made when the program
had other inputs or outputs (another signature), and calls logged
without their values; a warning says how many.

Rated calls are the ones people chose to look at, not a fair sample:
a score on them says how the program does on those. To measure how
often it is right, rate a random draw (``rate(..., sample=...)``) and
keep the rows of that draw (``col.sample == "..."``).

## Parameters {.doc-section .doc-section-parameters}

| Name     | Type                                | Description                                                                                                                                                                                            | Default    |
|----------|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| program  | AI function, module or name         | Whose calls. Given the function itself, only calls with its current signature are used (calls from earlier versions of the prompt are kept: their inputs and outputs are the same).                    | _required_ |
| folder   | str or path                         | The log folder (default as for ``calls``).                                                                                                                                                             | `None`     |
| by       | str                                 | Only this person's ratings. Default: everyone's; when people disagree, the latest rating is used and ``disputed`` is true.                                                                             | `None`     |
| since    | (date, datetime, timedelta or text) | Only calls and ratings from then on.                                                                                                                                                                   | `None`     |
| any_file | bool                                | A program defined in a notebook or a script is known by its file too, so two notebooks' ``summarize`` are two programs. ``True`` takes its calls from any file (a notebook that was moved or renamed). | `False`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type           | Description                                                                                                                                                                                            |
|--------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | dpyr dataframe | The inputs, the right answer (named like the output), then ``rating`` (``right`` or ``wrong``), ``rated_by``, ``origin`` (``review`` or ``edit``), ``disputed``, ``sample``, ``version`` and ``call``. |

## See Also {.doc-section .doc-section-see-also}

- [`rate`](rate.md): judge a call.
- [`calls`](calls.md): every logged call.
- [`evaluate`](evaluate.md): score a program on these rows.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
import tempfile
from typing import Literal

@ai
def team(message: str) -> Literal["shipping", "billing", "product"]:
    """Which team should answer this customer message?"""
    ...

with functai.configure(log_calls=tempfile.mkdtemp()):
    a = team.predict("My parcel never came.")
    b = team.predict("The chair arrived with a snapped leg.")
    functai.rate(a, "right")
    functai.rate(b, "wrong", answer="shipping")    # broken on the way: the carrier's fault
    rows = functai.rated(team)
rows.select("message", "result", "rating")
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
# dpyr dataframe · source: polars · showing 2 of 2 rows
┌───────────────────────────────────────┬──────────┬────────┐
│ message                               ┆ result   ┆ rating │
│ ---                                   ┆ ---      ┆ ---    │
│ str                                   ┆ str      ┆ str    │
╞═══════════════════════════════════════╪══════════╪════════╡
│ My parcel never came.                 ┆ shipping ┆ right  │
│ The chair arrived with a snapped leg. ┆ shipping ┆ wrong  │
└───────────────────────────────────────┴──────────┴────────┘
```