---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# bake.run { #functai.bake.run }

```{.python .no-run}
bake.run(folder)
```

A bake run, from its folder: reattach to it after a restart.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type        | Description                                                           | Default    |
|--------|-------------|-----------------------------------------------------------------------|------------|
| folder | str or path | The run's folder (``run.folder``; listed by ``functai.bake.runs()``). | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description   |
|--------|--------|---------------|
|        | Run    |               |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: it needs a run on this machine
run = functai.bake.run("~/.cache/functai/bakes/summarize-qwen3.5-2b-here-7f3a2c91d0e4")
run.metrics()            # the loss curve so far
baked = run.wait()       # the model, when it is done
```