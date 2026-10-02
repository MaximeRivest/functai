# bake.Baked { #functai.bake.Baked }

```{.python .no-run}
bake.Baked(path, *, device=None, check=True)
```

A model trained to answer one or more AI functions: what ``bake``
returns, and what ``functai.bake.load(folder)`` reads back.

Use it as a model: ``fast = summarize.using(lm=baked)`` is the same
function, answered by the weights (``fast("...")``, ``fast.map(rows)``,
``functai.evaluate(fast, rows)``). Calls send exactly the tokens the
student was trained on.

Two kinds (``kind``): a **head** (``"head"``) answers functions whose
every output has a fixed set of answers, with a probability for each
(``probabilities``, ``predict`` for many rows at once); a **generative
student** (``"generative"``, trained with ``method="sft"``) writes its answer like any chat model, and runs
in this process, on a vLLM server, on Tinker, or at any
OpenAI-compatible address (``on``).

On disk it is a folder: ``baked.json`` (what it answers: per function,
its signature, its layout and the inputs left out; the weights' form and
base model; the chat template; the run that made it; the report; file
hashes, checked when loading), ``model/`` (Hugging Face weights, merged
when trained with LoRA: a standard folder for vLLM, TGI or
transformers), ``adapter/`` (the LoRA adapter alone), ``tokenizer/``,
and ``heads.safetensors`` for a head of several outputs. Weights trained
on a service and not brought here yet are a ``tinker://`` address
(``download()`` brings them here).

## Attributes

| Name | Description |
| --- | --- |
| `capabilities` | What its calls can do (tools, streaming...), as lmcc lays out requests for it. |
| `device` | Where it runs in this process (``cuda``, ``mps`` or ``cpu``): the ``device`` given to |
| `endpoint` | The address calls are sent to when it runs on a server (``on("vllm")``, a URL); else None. |
| `fingerprint` | The fingerprint of the signature it was trained to read (one function). |
| `functions` | The names of the AI functions it answers. |
| `layout` | The lmcc adapter artifact its function's calls are written in (one function). |
| `model` | The model name its calls are logged under: ``baked:<name>``. |
| `name` | The model's name (by default the function's, or the functions' joined by ``+``). |
| `provider` | The provider name its calls are logged under. |
| `report` | How it did on its test rows when it was baked (or after ``judge(..., save=True)``); None |
| `runner` | What answers its calls now (in this process by default; see ``on``). |
| `signature` | What the student reads (one function). |
| `student` | The base model it was trained from (a Hugging Face id). |
| `template` | The chat template it was trained with (its text and hash): calls must write the same. |

## Methods

| Name | Description |
| --- | --- |
| [call_signature](#functai.bake.Baked.call_signature) | The signature ``fn``'s calls bind on this model: what the student reads. |
| [complete](#functai.bake.Baked.complete) | Answer one lm15 request (what FunctAI calls; to call the function on it, use |
| [download](#functai.bake.Baked.download) | Bring weights trained on a service here (merged into a standard |
| [entry_for](#functai.bake.Baked.entry_for) | The entry for ``fn`` (refused when the model was not trained for it, |
| [on](#functai.bake.Baked.on) | Run on ``where``: ``"transformers"`` (in this process), ``"vllm"`` (a |
| [predict](#functai.bake.Baked.predict) | A head model's answers for many rows at once (the fast path for big |
| [probabilities](#functai.bake.Baked.probabilities) | Per text, the probability of every answer, per field (texts as the layout writes them). |
| [reduce](#functai.bake.Baked.reduce) | (spec, inputs) as the student reads them (fixed and derived inputs left |
| [requirements](#functai.bake.Baked.requirements) | The packages running it needs here. |
| [resolve](#functai.bake.Baked.resolve) | Where a call named ``model`` goes (used by FunctAI when it routes a call). |
| [save](#functai.bake.Baked.save) | Copy the model to ``path``; returns it loaded from there. |
| [serve](#functai.bake.Baked.serve) | Serve with vLLM and send calls there; returns the endpoint. |
| [size](#functai.bake.Baked.size) | Its folder's size on disk, in bytes. |
| [stop](#functai.bake.Baked.stop) | Stop a server this model started (``serve()``/``on("vllm")``); calls run in-process again. |
| [texts](#functai.bake.Baked.texts) | The input text the model reads for each row of inputs (written by its layout). |
| [tokenizer](#functai.bake.Baked.tokenizer) | Its tokenizer, checked to carry the chat template it was trained with. |

### call_signature { #functai.bake.Baked.call_signature }

```{.python .no-run}
bake.Baked.call_signature(fn, spec)
```

The signature ``fn``'s calls bind on this model: what the student reads.

### complete { #functai.bake.Baked.complete }

```{.python .no-run}
bake.Baked.complete(request)
```

Answer one lm15 request (what FunctAI calls; to call the function on it, use
``fn.using(lm=baked)``).

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

### resolve { #functai.bake.Baked.resolve }

```{.python .no-run}
bake.Baked.resolve(model)
```

Where a call named ``model`` goes (used by FunctAI when it routes a call).

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

### size { #functai.bake.Baked.size }

```{.python .no-run}
bake.Baked.size()
```

Its folder's size on disk, in bytes.

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

### tokenizer { #functai.bake.Baked.tokenizer }

```{.python .no-run}
bake.Baked.tokenizer()
```

Its tokenizer, checked to carry the chat template it was trained with.