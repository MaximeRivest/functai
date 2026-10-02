---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# save { #functai.save }

```{.python .no-run}
save(
    program,
    path,
    *,
    include=(),
    requires=(),
    allow=(),
    examples=(),
    record=(),
    overwrite=False,
    weights='copy',
)
```

Save a program to a folder, with everything it depends on.

The folder holds the code the program reaches, each AI function's
settings, instruction and demos, data files read with ``functai.file``,
and pinned requirements. It is written whole or not at all. Keys and
connections are never saved.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type                             | Description                                                                                                                                                                                          | Default    |
|-----------|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| program   | AI function, module, or function | The program's entry point.                                                                                                                                                                           | _required_ |
| path      | str or path                      | The folder to write.                                                                                                                                                                                 | _required_ |
| include   | list of str                      | As for ``check``.                                                                                                                                                                                    | `()`       |
| requires  | list of str                      | As for ``check``.                                                                                                                                                                                    | `()`       |
| allow     | list of str                      | Problems to accept on purpose, like ``["hidden-state"]`` to save a global's current value.                                                                                                           | `()`       |
| record    | list of dict                     | Inputs to run the program on now, for real (this calls the model). The replies are recorded, and ``verify`` replays them to prove the saved program gives the same results, without calling a model. | `()`       |
| examples  | list of dict                     | Inputs to fingerprint the rendered requests of, besides the demos.                                                                                                                                   | `()`       |
| overwrite | bool                             | Replace an existing folder.                                                                                                                                                                          | `False`    |
| weights   | str                              | Baked weights: ``"copy"`` them into the folder (default) or ``"reference"`` them.                                                                                                                    | `'copy'`   |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                             |
|--------|--------|-----------------------------------------|
|        | Report | The ``check`` report of what was saved. |

## Raises {.doc-section .doc-section-raises}

| Name   | Type    | Description                                    |
|--------|---------|------------------------------------------------|
|        | Refused | While ``check`` finds errors, with the report. |

## See Also {.doc-section .doc-section-see-also}

- [`check`](check.md): what would be saved.
- [`verify`](verify.md): prove the folder runs in a fresh environment.
- [`load`](load.md): read it back.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
import tempfile, os

@ai
def capital(country: str) -> str:
    """The country's capital city."""
    ...

folder = os.path.join(tempfile.mkdtemp(), "capital")
save(capital, folder, record=[{"country": "Kenya"}])
sorted(os.listdir(folder))
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
['code', 'functai.json', 'recordings.json', 'requirements.lock', 'requirements.txt']
```