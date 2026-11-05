---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Criterion { #functai.Criterion }

```{.python .no-run}
Criterion(
    name,
    kind='verdict',
    answers=None,
    scale=None,
    items=None,
    aggregate='mean',
    question=None,
    judge=None,
)
```

A question an answer is judged by: its name, its kind, and how its values count.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type                                              | Description                                                                                                                                                                                                                                                                                                                                                                                                                    | Default     |
|-----------|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| name      | str                                               | Letters, digits and ``_``: ``"real_problem"``. ``"correct"`` (a verdict on the whole answer) is what format-1 ratings mean.                                                                                                                                                                                                                                                                                                    | _required_  |
| kind      | (\'verdict\', \'label\', \'score\' or \'choice\') | A verdict is ``"right"`` or ``"wrong"``; a label one of ``answers``; a score a number (on ``scale``, when given); a choice a pick between two answers (``"first"``, ``"second"`` or ``"tie"`` from a judge).                                                                                                                                                                                                                   | `'verdict'` |
| answers   | dict                                              | A label's answers and the number each counts for, from 0 to 1, or None (it counts for nothing: "unsure").                                                                                                                                                                                                                                                                                                                      | `None`      |
| scale     | (lowest, highest)                                 | A score's range; it counts as ``(value - lowest) / (highest - lowest)``.                                                                                                                                                                                                                                                                                                                                                       | `None`      |
| items     | str                                               | A JSON Pointer to a list in the answer (``"/insights"``): each item of it is judged.                                                                                                                                                                                                                                                                                                                                           | `None`      |
| aggregate | \'mean\' or \'sum\'                               | How a result's items add up: the share of good items, or how many.                                                                                                                                                                                                                                                                                                                                                             | `'mean'`    |
| question  | str                                               | The words people are asked.                                                                                                                                                                                                                                                                                                                                                                                                    | `None`      |
| judge     | AI function, module or function                   | Who judges automatically (``evaluate(criteria=...)``, ``judge``). It is given by name what it asks for: the case's inputs, ``answer``, ``outputs``, ``expected``, ``item``, ``inputs``, and for a pair ``first`` and ``second``. Its answer (or its answer's ``value`` field) is the judgement: True or False for a verdict, an answer for a label, a number for a score, ``"first"``, ``"second"`` or ``"tie"`` for a choice. | `None`      |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
real_problem = functai.Criterion("real_problem", "label", items="/insights",
                                 answers={"yes": 1, "no": 0, "unsure": None},
                                 question="Is this a real problem?")
real_problem.bounded
```

## Attributes

| Name | Description |
| --- | --- |
| `bounded` | Its values lie between 0 and 1 (not a count of items, nor a score without a scale). |

## Methods

| Name | Description |
| --- | --- |
| [data](#functai.Criterion.data) | The criterion as the contract writes it. |

### data { #functai.Criterion.data }

```{.python .no-run}
Criterion.data()
```

The criterion as the contract writes it.