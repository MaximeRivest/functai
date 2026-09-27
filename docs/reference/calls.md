---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# calls { #functai.calls }

```{.python .no-run}
calls(program=None, *, folder=None, since=None)
```

Every logged call, as a table.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type                                | Description                                                                                                                                                                                                  | Default   |
|---------|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| program | AI function, module or name         | Only this program's calls. Its inputs then get a column each, and its outputs ``pred_<output>`` columns, as in ``evaluate``'s table. Default: every program's, with ``inputs`` and ``outputs`` as JSON text. | `None`    |
| folder  | str or path                         | The log folder. Default: the one calls are logged to here, else the default one (``~/.local/share/functai/calls``).                                                                                          | `None`    |
| since   | (date, datetime, timedelta or text) | Only calls from then on: ``"2026-09-20"``, ``"7d"``, ``"12h"``.                                                                                                                                              | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|--------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | dpyr dataframe | One row per call, oldest first: the inputs and ``pred_<output>`` (for one program; else ``program``, ``inputs`` and ``outputs``), ``rating`` (the latest judgement: ``right``, ``wrong``, or null), ``error``, ``started``, ``seconds``, ``input_tokens``, ``output_tokens``, ``reasoning_tokens``, ``total_tokens`` (Gemini's ``output_tokens`` leave its hidden reasoning out; a cost is ``total_tokens - input_tokens`` at the output price), ``model``, ``version``, ``purpose`` (``use``, or ``evaluation`` and ``optimization`` for calls that answered known questions), ``caller`` (who called, as JSON), ``call`` (its id) and ``parent`` (the call it ran in). |

## See Also {.doc-section .doc-section-see-also}

- [`rated`](rated.md): the rated calls as rows with known answers.
- [`rate`](rate.md): judge a call.

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
    team("My parcel never came.")
    team("I was charged twice.")
    table = functai.calls(team)
table.select("message", "pred_result", "rating", "model", "seconds")
```