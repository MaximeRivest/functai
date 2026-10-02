---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# clear_cache { #functai.clear_cache }

```{.python .no-run}
clear_cache(which=None)
```

Forget the replies the reply cache kept, so the next identical requests
reach the model again.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type     | Description                                                                                                                                                                                                                               | Default   |
|--------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| which  | optional | None (default): the cache in this process's memory (``cache_replies=True``). ``"disk"``: the shared file ``cache_replies="disk"`` uses (in the user's cache folder, ``functai/replies.sqlite``). A folder or ``.sqlite`` path: that file. | `None`    |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: it deletes the replies kept on this machine
functai.clear_cache()            # this process's memory
functai.clear_cache("disk")      # the replies kept across runs
```