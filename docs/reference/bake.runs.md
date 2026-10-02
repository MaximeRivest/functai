# bake.runs { #functai.bake.runs }

```{.python .no-run}
bake.runs(home=None)
```

Every bake run on this machine, newest first.

A run is a folder (and, while it trains, a process of its own) under
``~/.cache/functai/bakes`` (``$XDG_CACHE_HOME/functai/bakes``), so runs
started from a notebook that has since closed are listed too.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type        | Description             | Default   |
|--------|-------------|-------------------------|-----------|
| home   | str or path | Another folder of runs. | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description                                                                                                                   |
|--------|-------------|-------------------------------------------------------------------------------------------------------------------------------|
|        | list of Run | Each with ``state`` (``running``, ``stopped``, ``done``, ``failed``...), ``metrics()``, ``wait()``, ``stop()``, ``resume()``. |