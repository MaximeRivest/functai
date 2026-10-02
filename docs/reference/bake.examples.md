# bake.examples { #functai.bake.examples }

`bake.examples`

From an AI function and rows of data to training examples.

The input a trained model reads is written by lmcc, with the same layout the
function uses when the trained model answers a call (``Baked`` pins it). So
training text and call-time text come from one writer and cannot drift.

A head model answers *finite* outputs: ``Literal``, ``Enum`` and ``bool``.
Each becomes a list of answer keys, spelled as lmcc spells them in a reply
("true", an enum's value), which are also the keys of the probabilities
functai returns (``prediction.probabilities[field][key]``).

## Classes

| Name | Description |
| --- | --- |
| [BakeError](#functai.bake.examples.BakeError) | A bake, or a baked model, cannot do what was asked; the message says |
| [HeadField](#functai.bake.examples.HeadField) | One finite output: its answer keys in a fixed order, and their JSON values. |

### BakeError { #functai.bake.examples.BakeError }

```{.python .no-run}
bake.examples.BakeError()
```

A bake, or a baked model, cannot do what was asked; the message says
what to do instead.

Raised, before anything is spent when it can be known then, for: rows
without the inputs or answers a function needs; an answer that is not
one of a head's choices; a student without a chat template; a function
changed since its model was baked (bake it again); a call whose ``fixed``
input has another value than the one it was trained with; a baked folder
whose files changed, or that FunctAI 1.1 wrote (``baked.json`` format
1: bake it again); a model trained elsewhere whose chat template writes
other tokens than its examples (``adopt``); no place set up to train.
A ``ValueError``.

### HeadField { #functai.bake.examples.HeadField }

```{.python .no-run}
bake.examples.HeadField(name, keys, values)
```

One finite output: its answer keys in a fixed order, and their JSON values.

## Functions

| Name | Description |
| --- | --- |
| [answer_key](#functai.bake.examples.answer_key) | How an answer is keyed in probabilities: "true"/"false", an enum's value as text. |
| [distribution](#functai.bake.examples.distribution) | A teacher's distribution over this field's keys, renormalized over the keys |
| [head_fields](#functai.bake.examples.head_fields) | The function's outputs as head fields; refuses outputs a classifier cannot give. |
| [head_layout](#functai.bake.examples.head_layout) | What a head model reads: the inputs alone (one input bare, several in tags); |
| [head_signature](#functai.bake.examples.head_signature) | The signature a head model serves: the plain inputs and the finite outputs. |
| [request_text](#functai.bake.examples.request_text) | The text a trained model reads from an lm15 request: its last user message. |
| [row_inputs](#functai.bake.examples.row_inputs) | A row's inputs for ``fn``, each optional one it lacks with its default. |

### answer_key { #functai.bake.examples.answer_key }

```{.python .no-run}
bake.examples.answer_key(value)
```

How an answer is keyed in probabilities: "true"/"false", an enum's value as text.

### distribution { #functai.bake.examples.distribution }

```{.python .no-run}
bake.examples.distribution(field, probs)
```

A teacher's distribution over this field's keys, renormalized over the keys
it knows (mass on other text is dropped, as the records' logprob targets did).

### head_fields { #functai.bake.examples.head_fields }

```{.python .no-run}
bake.examples.head_fields(spec)
```

The function's outputs as head fields; refuses outputs a classifier cannot give.

### head_layout { #functai.bake.examples.head_layout }

```{.python .no-run}
bake.examples.head_layout(signature)
```

What a head model reads: the inputs alone (one input bare, several in tags);
no instruction. The reply is a JSON object the model's answers fill.

### head_signature { #functai.bake.examples.head_signature }

```{.python .no-run}
bake.examples.head_signature(spec, fields)
```

The signature a head model serves: the plain inputs and the finite outputs.

### request_text { #functai.bake.examples.request_text }

```{.python .no-run}
bake.examples.request_text(request)
```

The text a trained model reads from an lm15 request: its last user message.

### row_inputs { #functai.bake.examples.row_inputs }

```{.python .no-run}
bake.examples.row_inputs(fn, row)
```

A row's inputs for ``fn``, each optional one it lacks with its default.