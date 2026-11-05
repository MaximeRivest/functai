---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# draw { #functai.draw }

```{.python .no-run}
draw(ids, n, seed)
```

A random draw of ``n`` ids, the same in every language.

Ranked by the hash of the seed and each id: uniform, reproducible, and
stable (drawing from more ids keeps each earlier pick that still ranks).
Rate the drawn calls with ``sample=seed``: only ratings of a random draw
measure a program without the bias of choosing what to judge.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type                                          | Description                                      | Default    |
|--------|-----------------------------------------------|--------------------------------------------------|------------|
| ids    | list of str (or predictions, or call records) |                                                  | _required_ |
| n      | int                                           |                                                  | _required_ |
| seed   | str                                           | Names the draw; becomes the ratings' ``sample``. | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description   |
|--------|-------------|---------------|
|        | list of str |               |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
functai.draw([f"call-{i}" for i in range(100)], 5, seed="review-2026-11-05")
```