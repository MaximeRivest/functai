# bake.load { #functai.bake.load }

```{.python .no-run}
bake.load(path, *, device=None)
```

A baked model, from its folder.

Every file's hash is checked against ``baked.json``: a folder changed
since baking is refused (``BakeError``). A folder baked by FunctAI 1.1
(``baked.json`` format 1) is refused too, with the advice to bake again.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type        | Description                                                                                  | Default    |
|--------|-------------|----------------------------------------------------------------------------------------------|------------|
| path   | str or path | The folder (``baked.path``, or the ``path=`` given to ``bake``).                             | _required_ |
| device | str         | ``"cuda"``, ``"cuda:1"``, ``"mps"`` or ``"cpu"`` (default: the best one its weights fit on). | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                       |
|--------|--------|-----------------------------------|
|        | Baked  | ``fn.using(lm=baked)`` to use it. |