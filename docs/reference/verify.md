---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# verify { #functai.verify }

```{.python .no-run}
verify(path, *, trust=False, fresh=True, python=None, timeout=900)
```

Prove a saved program runs somewhere else, without calling a model.

Builds a new environment with ``uv`` from the folder's lock file alone,
loads the program there from an empty directory (so nothing from your
project can leak in), then checks that every AI function renders
byte-identical requests, and that each recording (``save(record=...)``)
replays to the same result.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type        | Description                                                                                                                      | Default    |
|---------|-------------|----------------------------------------------------------------------------------------------------------------------------------|------------|
| path    | str or path | The saved folder.                                                                                                                | _required_ |
| trust   | bool        | Must be True: verifying runs the saved code.                                                                                     | `False`    |
| fresh   | bool        | Build a new environment (default). ``False`` checks in this one: quicker, and blind to packages installed here but not declared. | `True`     |
| python  | str         | The Python version or interpreter for the new environment.                                                                       | `None`     |
| timeout | float       | Seconds before giving up.                                                                                                        | `900`      |

## Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description                                                         |
|--------|--------------|---------------------------------------------------------------------|
|        | Verification | Displays what was checked; ``.ok`` is True when everything matched. |

## See Also {.doc-section .doc-section-see-also}

- [`save`](save.md): write the folder.
- [`load`](load.md): use it.

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
verify(folder, trust=True, fresh=False)
```