---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# load { #functai.load }

```{.python .no-run}
load(path, *, trust=False, check_env='refuse')
```

Load a saved program, ready to call.

Checks before running anything: the files' hashes (catching accidental
edits) and the packages this environment has. After loading, checks that
every AI function renders the same requests as when it was saved.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type        | Description                                                                                                                               | Default    |
|-----------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------|
| path      | str or path | The saved folder.                                                                                                                         | _required_ |
| trust     | bool        | Must be True: loading runs the saved code. The hashes catch accidents, not someone who edits both the code and ``functai.json``.          | `False`    |
| check_env | str         | ``"refuse"`` (default) refuses on version or request differences; ``"warn"`` loads anyway, with warnings. Missing packages always refuse. | `'refuse'` |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                  | Description                   |
|--------|-----------------------|-------------------------------|
|        | AI function or module | The program, as it was saved. |

## Raises {.doc-section .doc-section-raises}

| Name   | Type        | Description                               |
|--------|-------------|-------------------------------------------|
|        | LoadRefused | When a check fails, saying which and why. |

## See Also {.doc-section .doc-section-see-also}

- [`save`](save.md): write the folder.
- [`verify`](verify.md): prove it runs in a fresh environment.

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

folder = os.path.join(tempfile.mkdtemp(), "capital")
save(capital, folder)
loaded = load(folder, trust=True)
loaded("Peru")
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
'Lima'
```