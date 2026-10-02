# bake.Baked { #functai.bake.Baked }

```{.python .no-run}
bake.Baked(path, *, device=None, check=True)
```

A model trained for one or more AI functions (see the module docstring).

## Attributes

| Name | Description |
| --- | --- |
| `layout` | The lmcc adapter artifact its function's calls are written in (one function). |
| `signature` | What the student reads (one function). |

## Methods

| Name | Description |
| --- | --- |
| [call_signature](#functai.bake.Baked.call_signature) | The signature ``fn``'s calls bind on this model: what the student reads. |
| [download](#functai.bake.Baked.download) | Bring weights trained on a service here (merged into a standard |
| [entry_for](#functai.bake.Baked.entry_for) | The entry for ``fn`` (refused when the model was not trained for it, |
| [on](#functai.bake.Baked.on) | Run on ``where``: ``"transformers"`` (in this process), ``"vllm"`` (a |
| [predict](#functai.bake.Baked.predict) | A head model's answers for many rows at once (the fast path for big |
| [probabilities](#functai.bake.Baked.probabilities) | Per text, the probability of every answer, per field (texts as the layout writes them). |
| [reduce](#functai.bake.Baked.reduce) | (spec, inputs) as the student reads them (fixed and derived inputs left |
| [requirements](#functai.bake.Baked.requirements) | The packages running it needs here. |
| [save](#functai.bake.Baked.save) | Copy the model to ``path``; returns it loaded from there. |
| [serve](#functai.bake.Baked.serve) | Serve with vLLM and send calls there; returns the endpoint. |
| [stop](#functai.bake.Baked.stop) | Stop a server this model started (``serve()``/``on("vllm")``); calls run in-process again. |
| [texts](#functai.bake.Baked.texts) | The input text the model reads for each row of inputs (written by its layout). |

### call_signature { #functai.bake.Baked.call_signature }

```{.python .no-run}
bake.Baked.call_signature(fn, spec)
```

The signature ``fn``'s calls bind on this model: what the student reads.

### download { #functai.bake.Baked.download }

```{.python .no-run}
bake.Baked.download(path=None)
```

Bring weights trained on a service here (merged into a standard
folder). Returns the model, loaded from its folder.

### entry_for { #functai.bake.Baked.entry_for }

```{.python .no-run}
bake.Baked.entry_for(fn, spec)
```

The entry for ``fn`` (refused when the model was not trained for it,
or when it changed since).

### on { #functai.bake.Baked.on }

```{.python .no-run}
bake.Baked.on(where=None, **options)
```

Run on ``where``: ``"transformers"`` (in this process), ``"vllm"`` (a
server started here), ``"tinker"``, or an OpenAI-compatible URL. Returns
the model.

### predict { #functai.bake.Baked.predict }

```{.python .no-run}
bake.Baked.predict(rows)
```

A head model's answers for many rows at once (the fast path for big
tables): per row, each field's answer, its probability, and the full
distribution.

### probabilities { #functai.bake.Baked.probabilities }

```{.python .no-run}
bake.Baked.probabilities(texts)
```

Per text, the probability of every answer, per field (texts as the layout writes them).

### reduce { #functai.bake.Baked.reduce }

```{.python .no-run}
bake.Baked.reduce(fn, spec, inputs, *, check=True)
```

(spec, inputs) as the student reads them (fixed and derived inputs left
out, after checking their values).

### requirements { #functai.bake.Baked.requirements }

```{.python .no-run}
bake.Baked.requirements()
```

The packages running it needs here.

### save { #functai.bake.Baked.save }

```{.python .no-run}
bake.Baked.save(path, *, overwrite=False)
```

Copy the model to ``path``; returns it loaded from there.

### serve { #functai.bake.Baked.serve }

```{.python .no-run}
bake.Baked.serve(**options)
```

Serve with vLLM and send calls there; returns the endpoint.

### stop { #functai.bake.Baked.stop }

```{.python .no-run}
bake.Baked.stop()
```

Stop a server this model started (``serve()``/``on("vllm")``); calls run in-process again.

### texts { #functai.bake.Baked.texts }

```{.python .no-run}
bake.Baked.texts(rows)
```

The input text the model reads for each row of inputs (written by its layout).