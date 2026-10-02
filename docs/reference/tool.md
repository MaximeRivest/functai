---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# tool { #functai.tool }

```{.python .no-run}
tool(fn=None, *, effects=None, name=None, description=None)
```

Make a function a tool that says what it does to the world.

## Parameters {.doc-section .doc-section-parameters}

| Name        | Type                     | Description                                                                                                                                                                                                                                                                       | Default   |
|-------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| effects     | \'reads\' or \'changes\' | ``"reads"``: it only looks (a search, reading a file); it is never asked about by the ``"changes"`` rule, and a required journal does not wait before it. ``"changes"``: it changes something (writes, sends, pays). Left out: unknown, which every rule treats as ``"changes"``. | `None`    |
| name        | str                      | What the model is told, instead of the function's name and docstring.                                                                                                                                                                                                             | `None`    |
| description | str                      | What the model is told, instead of the function's name and docstring.                                                                                                                                                                                                             | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                          |
|--------|--------|------------------------------------------------------|
|        | Tool   | The function, callable as before, with ``.effects``. |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@functai.tool(effects="reads")
def order_status(order: str) -> str:
    """Where an order is."""
    return "in Leeds"

order_status.effects
```