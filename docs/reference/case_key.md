---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# case_key { #functai.case_key }

```{.python .no-run}
case_key(program, row)
```

The key of the case a row is for a program.

Its values of the program's inputs (and a conversation's earlier turns,
when the row keeps them) decide it; nothing else of the row does, so
correcting its answer keeps its key. A call ``evaluate`` makes on the row
carries it (``case`` in its record).

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type                                            | Description   | Default    |
|---------|-------------------------------------------------|---------------|------------|
| program | AI function or module, or a list of input names |               | _required_ |
| row     | dict                                            |               | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description     |
|--------|--------|-----------------|
|        | str    | ``"sha256:…"``. |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
functai.case_key(["message"], {"message": "Charged twice.", "result": "billing"})
```