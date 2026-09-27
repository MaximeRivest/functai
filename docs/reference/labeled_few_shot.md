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
taught = functai.labeled_few_shot(team, train, k=8, expected="category")
```