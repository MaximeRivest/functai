---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# all_turns { #functai.all_turns }

```{.python .no-run}
all_turns(without=())
```

Show the model every earlier turn of the conversation (the default).

Running out of the model's context, and being told, is better than a
model that silently misses what was said; so a conversation shows every
earlier turn unless told otherwise.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type        | Description                                                                                                                              | Default   |
|---------|-------------|------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| without | list of str | Inputs or outputs left out of every *earlier* turn (a long document the model already answered about). The current turn is always whole. | `()`      |

## Returns {.doc-section .doc-section-returns}

| Name   | Type    | Description                           |
|--------|---------|---------------------------------------|
|        | Context | For ``fn.conversation(context=...)``. |

## See Also {.doc-section .doc-section-see-also}

- [`last_turns`](last_turns.md): only the most recent turns.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: part of a program (the Memory guide runs whole ones)
qa = reader.conversation(context=functai.all_turns(without=["document"]))
```