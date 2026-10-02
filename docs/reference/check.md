---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# check { #functai.check }

```{.python .no-run}
check(program, *, include=(), requires=())
```

List everything a program depends on, and what would stop a clean save.

Follows every name the code reaches: AI functions and modules (also
through helper functions), their tools, your own functions and classes
(in files or notebook cells), the types in the signatures, constants,
and data files read through ``functai.file``. Reads code only: runs
nothing and calls no model.

## Parameters {.doc-section .doc-section-parameters}

| Name     | Type                             | Description                                                                                                                | Default    |
|----------|----------------------------------|----------------------------------------------------------------------------------------------------------------------------|------------|
| program  | AI function, module, or function | The program's entry point.                                                                                                 | _required_ |
| include  | list of str                      | Modules or package prefixes to save as code even though they are installed (your own project, installed in editable mode). | `()`       |
| requires | list of str                      | Requirements to add by hand (``"numpy>=2"``), for what the code reaches in ways that reading it can't see.                 | `()`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                           |
|--------|--------|---------------------------------------------------------------------------------------------------------------------------------------|
|        | Report | Displays as a tree of dependencies, the requirements, and each problem with its fix. ``report.ok`` is True when nothing stops a save. |

## See Also {.doc-section .doc-section-see-also}

- [`save`](save.md): save the program, once ``check`` is clean.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
ORDERS = {"A-1042": "stuck at carrier"}

def lookup_order(order_id: str) -> str:
    """The order's shipping status."""
    return ORDERS.get(order_id, "no such order")

@ai(tools=[lookup_order])
def reply(message: str) -> str:
    """A short reply to the customer. Check the order first."""
    ...

check(reply)
```

```output
reply  AI function (message: str → str)  [__main__]
└── tool lookup_order  function  [__main__]
    └── ORDERS = {'A-1042': 'stuck at carrier'}

requirements: functai @ file:///home/maxime/Projects/functai/python

! local-install  requirements: installed from folders on this machine: functai (/home/maxime/Projects/functai/python), lmcc (/home/maxime/Projects/lmcc/python)
    fix: the saved program loads where those folders exist; publish them, or install released versions, to load it anywhere
```