---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# bootstrap_few_shot { #functai.bootstrap_few_shot }

```{.python .no-run}
bootstrap_few_shot(
    fn,
    data,
    *,
    teacher=None,
    expected=None,
    metric=None,
    max_bootstrapped=4,
    max_labeled=16,
    num_threads=8,
    seed=0,
)
```

An improved copy: the function (or a stronger ``teacher`` model) runs on
rows with known answers, and the runs that were right become worked
examples, whole (reasoning and tool calls included); labeled rows fill the rest.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: the teacher answers every row (Make it better runs one)
taught = functai.bootstrap_few_shot(team, train, teacher="gpt-6-sol", expected="category")
```