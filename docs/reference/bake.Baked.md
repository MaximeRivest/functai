# bake.Baked { #functai.bake.Baked }

```{.python .no-run}
bake.Baked(path, *, device=None, check=True)
```

A model trained for one AI function (see the module docstring).

## Attributes

| Name | Description |
| --- | --- |
| `layout` | The lmcc adapter artifact the model reads its inputs through. |

## Methods

| Name | Description |
| --- | --- |
| [predict](#functai.bake.Baked.predict) | Answers for many rows at once (the fast path for big tables): per row, |
| [probabilities](#functai.bake.Baked.probabilities) | Per text, the probability of every answer, per field (texts as the layout writes them). |
| [save](#functai.bake.Baked.save) | Copy the model to ``path``; returns it loaded from there. |
| [serve](#functai.bake.Baked.serve) | Serve a generative student with vLLM and send calls there (see ``sft.serve``). |
| [stop](#functai.bake.Baked.stop) | Stop the vLLM server ``serve()`` started; calls run in-process again. |
| [texts](#functai.bake.Baked.texts) | The input text the model reads for each row of inputs (written by its layout). |

### predict { #functai.bake.Baked.predict }

```{.python .no-run}
bake.Baked.predict(rows)
```

Answers for many rows at once (the fast path for big tables): per row,
each field's answer, its probability, and the full distribution.

### probabilities { #functai.bake.Baked.probabilities }

```{.python .no-run}
bake.Baked.probabilities(texts)
```

Per text, the probability of every answer, per field (texts as the layout writes them).

### save { #functai.bake.Baked.save }

```{.python .no-run}
bake.Baked.save(path, *, overwrite=False)
```

Copy the model to ``path``; returns it loaded from there.

### serve { #functai.bake.Baked.serve }

```{.python .no-run}
bake.Baked.serve(**options)
```

Serve a generative student with vLLM and send calls there (see ``sft.serve``).

### stop { #functai.bake.Baked.stop }

```{.python .no-run}
bake.Baked.stop()
```

Stop the vLLM server ``serve()`` started; calls run in-process again.

### texts { #functai.bake.Baked.texts }

```{.python .no-run}
bake.Baked.texts(rows)
```

The input text the model reads for each row of inputs (written by its layout).