---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# split_of { #functai.split_of }

```{.python .no-run}
split_of(row, program=None, *, share=0.2, salt='')
```

Whether a row is for training or held out, the same in every language.

The row's own ``split`` (``"train"`` or ``"held_out"``) when it has one;
else held out when the hash of its key (its conversation's id, so every
turn of a conversation is on one side) falls below ``share``. Adding rows
never moves an old one.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type                                        | Description                                                          | Default    |
|---------|---------------------------------------------|----------------------------------------------------------------------|------------|
| row     | dict                                        |                                                                      | _required_ |
| program | AI function, module, or list of input names | Whose inputs make the key (required unless the row has a ``split``). | `None`     |
| share   | float                                       | The share held out, from 0 to 1.                                     | `0.2`      |
| salt    | str                                         | Another salt, another split of the same rows.                        | `''`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                      | Description   |
|--------|---------------------------|---------------|
|        | \'train\' or \'held_out\' |               |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
[functai.split_of({"message": m}, ["message"]) for m in ("a", "b", "c", "d", "e")]
```