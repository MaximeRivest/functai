---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# last_turns { #functai.last_turns }

```{.python .no-run}
last_turns(n, *, without=())
```

Show the model only the last ``n`` earlier turns.

The turns left out are still kept, and each turn's record says which
earlier turns it was shown (``turn.saw``), so an answer can be asked
again exactly as it was.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type        | Description                                       | Default    |
|---------|-------------|---------------------------------------------------|------------|
| n       | int         | How many earlier turns are shown (0: none).       | _required_ |
| without | list of str | Inputs or outputs left out of every earlier turn. | `()`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type    | Description                           |
|--------|---------|---------------------------------------|
|        | Context | For ``fn.conversation(context=...)``. |

## See Also {.doc-section .doc-section-see-also}

- [`all_turns`](all_turns.md): every earlier turn (the default).
- [`compaction`](compaction.md): older turns folded into a summary instead of dropped.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: part of a program (the Memory guide runs whole ones)
qa = reader.conversation(context=functai.last_turns(10, without=["document"]))
```