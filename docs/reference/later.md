---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# later { #functai.later }

```{.python .no-run}
later(key)
```

What a tool returns when its result comes later: it started something
that ends elsewhere (a job, a cloud browser, a person's reply) and whoever
learns the result delivers it under ``key``.

Only in a conversation's turn: the turn runs the step's other tool calls,
then stops, saved, and waits (``Waiting``); ``chat.deliver(key, result)``,
from any process, gives the tool its result and the turn goes on. The
tool does not run again. Anywhere else nothing could wait for it, and the
call refuses ``later-unavailable``.

``key`` names one result in its conversation (put an id in it:
``"harvest:8f2c"``). A result delivered before the turn waits is used at
once.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@functai.tool(effects="changes")
def start_harvest(url: str) -> str:
    """Visit a site in a cloud browser; returns what it saw."""
    job = harvests.start(url, webhook="https://example.com/done")
    return functai.later(f"harvest:{job.id}")

# in the webhook, any process:
chat.deliver(f"harvest:{job_id}", report)
```