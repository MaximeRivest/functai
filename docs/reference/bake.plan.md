---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# bake.plan { #functai.bake.plan }

```{.python .no-run}
bake.plan(what, data=None, **options)
```

What a generative bake would do, decided, with nothing spent.

The same as ``bake(what, data, method="sft", plan_only=True, **options)``
(and as ``fn.bake(rows, plan_only=True)`` when the function needs a
generative student). Printed, the plan says the rows and how many a
teacher must answer (and what that costs), the tokens per pass, the
student, where it would train with the time or the price of each place
set up, the training settings, missing speed-up kernels, and the run's
folder.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type   | Description                                                                                                  | Default    |
|-----------|--------|--------------------------------------------------------------------------------------------------------------|------------|
| what      | Any    | As for ``bake``: an AI function and its rows, ``{fn: rows}``, a ``@module`` and its inputs, or ``Examples``. | _required_ |
| data      | Any    | As for ``bake``: an AI function and its rows, ``{fn: rows}``, a ``@module`` and its inputs, or ``Examples``. | _required_ |
| **options |        | Anything ``bake`` takes (``student=``, ``where=``, ``teacher=``, ``fixed=``, training settings...).          | `{}`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                     |
|--------|--------|---------------------------------------------------------------------------------------------------------------------------------|
|        | Plan   | ``print(plan)`` to read it, ``plan.using(where="tinker")`` for the same bake with one setting changed, ``plan.run()`` to do it. |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: a plan reads the rows and the machine (the guide Bake it into a small model shows one)
plan = functai.bake.plan(summarize, rows)
print(plan)
baked = plan.using(where="tinker").run()
```