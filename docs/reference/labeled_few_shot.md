---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# labeled_few_shot { #functai.labeled_few_shot }

```{.python .no-run}
labeled_few_shot(fn, data, *, k=16, expected=None, sample=True, seed=0)
```

An improved copy: up to ``k`` rows with known answers become worked examples.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
from typing import Literal

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

taught = functai.labeled_few_shot(team, functai.datasets.tickets(), k=3, expected="category")
taught.state()
```

```output
instruction: (written from the code)
examples: 3
  1. message='Refund please: the towels are much thinner than in the photos.'  →  result='billing'
  2. message='I was charged for an order I cancelled (A-1401).'  →  result='billing'
  3. message="I'd like my money back for the toaster, it burns everything."  →  result='billing'
```

No model is called: the rows are the examples.