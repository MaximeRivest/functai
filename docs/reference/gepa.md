---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# gepa { #functai.gepa }

```{.python .no-run}
gepa(
    fn,
    data,
    *,
    teacher=None,
    expected=None,
    selection=None,
    budget=300,
    minibatch=4,
    metric=None,
    feedback=None,
    num_threads=8,
    seed=0,
)
```

An improved copy whose instruction a ``teacher`` model rewrote from the
function's mistakes (``GEPA``; design/04-gepa.md).

Half the rows (or ``selection``) choose and are never shown to the teacher;
the other half give feedback. ``better.trials`` is the search. Its scores on
the choosing rows flatter the one chosen: measure it on rows it never saw.
The optimizer class ``GEPA`` takes the same options, for
``fn.opt(rows, optimizer=GEPA(...))`` and for ``@module``s' AI functions one at a time.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
better = functai.gepa(team.using(lm="gpt-5.4-nano"), train, teacher="gpt-6-sol", expected="category")
better.instructions
functai.evaluate(better, test, expected="category")
```