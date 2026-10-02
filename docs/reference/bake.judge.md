---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# bake.judge { #functai.bake.judge }

```{.python .no-run}
bake.judge(baked, fn, rows=None, **options)
```

Measure a generative student on test rows: ``functai.evaluate`` on the
function running on the baked model, plus what only a student has.

``bake`` judges the student when it ends (``report=True``); ``judge``
does it again later, on other rows or with another metric. A head
model's report is made when it is baked (``baked.report``).

## Parameters {.doc-section .doc-section-parameters}

| Name            | Type                     | Description                                                                                                                                                                                                                                                                                                                              | Default    |
|-----------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| baked           | Baked                    | The student.                                                                                                                                                                                                                                                                                                                             | _required_ |
| fn              | AI function, or dict     | The function and ``rows``, or ``{function: rows}`` for a student of several functions.                                                                                                                                                                                                                                                   | _required_ |
| rows            | list of dict, or a table | Test rows with the right answers (never rows it trained on).                                                                                                                                                                                                                                                                             | `None`     |
| metric          | optional                 | Anything ``evaluate`` takes: a function, a dpyr expression, an AI judge, a dict per output. Without one, a function whose outputs are all finite (``Literal``, ``Enum``, ``bool``) is scored by exact match; any other is not scored (exact match on open text measures nothing): the report then gives readability and samples to read. | _required_ |
| compare_teacher | bool                     | Also run the teacher on the same rows, to compare (it costs a teacher pass). Default False.                                                                                                                                                                                                                                              | _required_ |
| by              | str                      | A column of the rows (a tag, a source) to score each of its values apart.                                                                                                                                                                                                                                                                | _required_ |
| samples         | int                      | How many answers the report shows to read (default 5).                                                                                                                                                                                                                                                                                   | _required_ |
| save            | bool                     | Write the report into the model's ``baked.json``.                                                                                                                                                                                                                                                                                        | _required_ |
| num_threads     | int                      | Rows judged at once (default 16).                                                                                                                                                                                                                                                                                                        | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                                                                                                                |
|--------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | report | Printed: per function, the score with its 95% range (per group with ``by``), readability (the share of replies its layout reads back into the output types), speed, the cost of a call next to the teacher's, and samples. |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: it needs a baked model
report = functai.bake.judge(baked, summarize, test_rows, metric=faithful)
print(report)
```