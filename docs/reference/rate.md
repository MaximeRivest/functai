---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# rate { #functai.rate }

```{.python .no-run}
rate(
    call,
    verdict=_NOTHING,
    *,
    answer=_NOTHING,
    outputs=None,
    note=None,
    reasons=(),
    by=None,
    origin=None,
    sample=None,
    folder=None,
)
```

Say whether a call's answer is right, and if not, what it should have been.

"Right" means correct for this input, not "nice". A correction becomes a
row of data: ``rated`` gives it back with the inputs, for ``evaluate``
and ``.opt``. The rating is written to the call log, next to the call.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type                                        | Description                                                                                                                                                                                                                                                                                                                        | Default    |
|---------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| call    | Prediction, call id, or row                 | The call: what ``fn.predict(...)`` returned, its ``call_id``, or a row of ``calls()`` or ``rated()``.                                                                                                                                                                                                                              | _required_ |
| verdict | (\'right\', \'wrong\', True, False or None) | Is the answer right? None withdraws your earlier rating. May be left out when ``answer`` is given (then it is ``"wrong"``).                                                                                                                                                                                                        | `_NOTHING` |
| answer  | optional                                    | The right answer, in the answer's type (a label, a number, a dataclass...).                                                                                                                                                                                                                                                        | `_NOTHING` |
| outputs | dict                                        | Right values for other named outputs: ``{"priority": 2}``.                                                                                                                                                                                                                                                                         | `None`     |
| note    | str                                         | Why, in a sentence.                                                                                                                                                                                                                                                                                                                | `None`     |
| reasons | list of str                                 | Short tags: ``["wrong category"]``.                                                                                                                                                                                                                                                                                                | `()`       |
| by      | str                                         | Who is judging: a person. Default: the caller's ``user`` (``configure(caller={"user": ...})``). One person's later rating of a call replaces their earlier one. With no person named, the rating is made under this computer's account, which may be shared: it is kept on its own, and never replaces nor is replaced by another. | `None`     |
| origin  | str                                         | ``"review"`` (default: someone judged the answer) or ``"edit"`` (someone changed the output while using it).                                                                                                                                                                                                                       | `None`     |
| sample  | str                                         | The id of a random draw of calls this rating is part of: only a random draw measures how often a program is right.                                                                                                                                                                                                                 | `None`     |
| folder  | str or path                                 | The log folder. Default: the one calls are logged to here.                                                                                                                                                                                                                                                                         | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description             |
|--------|--------|-------------------------|
|        | dict   | The rating, as written. |

## See Also {.doc-section .doc-section-see-also}

- [`rated`](rated.md): the ratings as rows with known answers.
- [`calls`](calls.md): every logged call.

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
    p = team.predict("I was charged twice for one order.")
    rating = functai.rate(p, "right")
rating["verdict"]
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
'right'
```