---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# phistory { #functai.phistory }

```{.python .no-run}
phistory(n=1)
```

The last model calls, as readable text: what was sent, what came back.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type   | Description                       | Default   |
|--------|--------|-----------------------------------|-----------|
| n      | int    | How many calls, most recent last. | `1`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                           |
|--------|--------|---------------------------------------------------------------------------------------------------------------------------------------|
|        | text   | Every message of each call, the reply, the finish reason and the tokens. Displays as is at a notebook prompt; ``print`` it elsewhere. |

## See Also {.doc-section .doc-section-see-also}

- [`inspect_history`](inspect_history.md): the same calls as lm15 request and response objects.
- [`FunctAIFunc.render`](FunctAIFunc.md#functai.FunctAIFunc.render): the request a call would send, without sending it.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@ai
def capital(country: str) -> str:
    """The country's capital city."""
    ...

capital("Japan")
print(phistory())
```