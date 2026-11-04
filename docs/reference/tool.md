---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# tool { #functai.tool }

```{.python .no-run}
tool(
    fn=None,
    *,
    effects=None,
    name=None,
    description=None,
    notes=None,
    needs=None,
    code_id=None,
)
```

Make a function a tool that says what it does to the world.

## Parameters {.doc-section .doc-section-parameters}

| Name        | Type                     | Description                                                                                                                                                                                                                                                                                                   | Default   |
|-------------|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| effects     | \'reads\' or \'changes\' | ``"reads"``: it only looks (a search, reading a file); it is never asked about by the ``"changes"`` rule, and a required journal does not wait before it. ``"changes"``: it changes something (writes, sends, pays). Left out: unknown, which every rule treats as ``"changes"``.                             | `None`    |
| name        | str                      | What the model is told, instead of the function's name and docstring.                                                                                                                                                                                                                                         | `None`    |
| description | str                      | What the model is told, instead of the function's name and docstring.                                                                                                                                                                                                                                         | `None`    |
| notes       | dict                     | The notes its code makes (``functai.note(name, data)``), by name: a type, or ``functai.Note(type, latest=..., outside=...)``. Every AI function that has the tool declares them, and each note it makes is checked against them (``InterfaceError``, code ``interface-note``).                                | `None`    |
| needs       | dict                     | What its code reaches: ``{"domains": ["api.browser-use.com"], "secrets": ["BROWSER_USE_API_KEY"]}`` (names, never values). Behind a wall that lets through less (the ``wall`` setting), the function that has it is refused before its first call (``ReachError``); the reach counts in the function's setup. | `None`    |
| code_id     | str                      | What identifies its code beyond the code a setup reads of it: a revision (``"v3"``) or a bundle's hash. Give it when its behaviour comes from values it captured (a tool made by a factory) or from code outside what Python can follow.                                                                      | `None`    |

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

```output
'reads'
```