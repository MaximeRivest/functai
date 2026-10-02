# Baked models

A baked model is weights trained to answer one or more AI functions. This
document fixes what every language must agree on: **what a generative
student is trained on** (so a model trained by one language, or by any
tool from examples one language wrote, is called correctly by every
other), and **the folder** (`baked.json`). Training itself is not part of
the contract: weights need not match between languages or trainers.

## The examples

A generative student learns one function's calls as chat conversations.
For a function and a row (inputs, and the outputs to learn):

1. **The student's signature** is the function's (as
   [functions.md](functions.md) builds it), without the inputs the bake
   leaves out (*fixed* and *derived*, below), and without the `reasoning`
   output unless the bake keeps it (`reasoning: true` and `module: "cot"`).
   The instruction is unchanged.
2. **The student's layout** is the function's layout (its adapter or
   template; [layouts/](layouts/)) with replies written from values
   (`replay: "values"`). Its capabilities are `{"instruct": true}`: no
   native transport, so everything the layout writes is text.
3. **No worked examples.** A student is trained and called without the
   function's demos.
4. **The prompt** is the request the layout renders for the row's inputs
   (text inputs given other values written as in functions.md) with no
   earlier turns, as chat messages: the request's `system` text as a
   `system` message, then each message, a `developer` message as a
   `system` one, each message's text parts joined.
5. **The reply** is the text of the assistant message the layout writes
   for the row's outputs (render the same call with that example as the
   one earlier turn, and take its assistant message).

The conversation is the prompt then `{"role": "assistant", "content":
<reply>}`; the loss is on the reply only. A row's `weight` (default 1) is
how many times it counts.

Turning messages into tokens is the student's own chat template's rule
(with `enable_thinking: false` when the template takes it). An
implementation that writes tokens records the template's SHA-256 and its
keyword arguments; a model trained elsewhere is adopted only when its
template writes the same tokens for the examples.

### Fixed and derived inputs

- A **fixed** input has one value in every row. It is left out of the
  student's signature; `baked.json` keeps `"sha256:"` of its value's
  canonical JSON ([calls.md](calls.md) §canonical; a text input's value
  as the call writes it). A call whose value hashes differently is
  refused before any request (`baked-fixed`).
- A **derived** input is decided by another input the student still
  reads (`{"guidance": "section"}`). It is left out too; `baked.json`
  keeps `{"from": <source>, "values": {<hash of a source value>: <hash of
  its value>}}` from the rows. Two rows whose source values hash the same
  and whose values differ refuse the bake. A call whose source value is
  not in the table, or whose value is not the one the table gives, is
  refused (`baked-derived`).

The cases (`cases/baked/`) pin the student's signature, the hashes and
the conversations for five definitions.

### The examples table

`functai.bake.examples` (and its equivalents) write one row per
conversation: `function`, `messages`, and when a student is named
`input_ids` (the tokens) and `answer_start` (where the reply starts), with
`prompt_tokens`, `answer_tokens`, `tag`, `weight`, `row_id`, `split`
(`"train"` or `"validation"`) and `source` (`"data"`, `"teacher"`,
`"program"`). Saved as Parquet (what made them, below, in the schema's
metadata under `functai`) or JSON lines (a `.meta.json` file beside). The
metadata is `{"functai_examples": 1, "student", "template", "functions":
[<entry>...]}`, entries as in `baked.json`.

## The folder

```
baked.json      format 2 (schema/baked.schema.json)
model/          Hugging Face weights (safetensors): merged when trained with LoRA
adapter/        the LoRA adapter, when there is one
tokenizer/      the student's tokenizer and chat template
```

`baked.json`:

| key | |
|---|---|
| `functai_baked` | `2` |
| `kind` | `"head"` (a classifier, finite outputs) or `"generative"` |
| `name`, `student` | the model's name; the base model (Hugging Face id or folder) |
| `functions` | one entry per function it answers: `name`, `fingerprint` (lmcc's, of the function's full signature), `signature` (what the student reads), `layout` (the adapter artifact), `outputs`, `reasoning`, `fixed`, `derived`, `capabilities` |
| `weights` | `{"form": "merged", "path": "model"}`, `{"form": "lora", "base", "adapter"}`, or `{"form": "remote", "service", "uri", "base"}` (weights on a service, e.g. `tinker://…`) |
| `template` | `sha256`, `kwargs`, `end` (what the template writes after a reply), `stop_token_ids` (generative) |
| `generation` | `max_new_tokens`, `max_model_len`, from the data (generative) |
| `head` | `architecture`, `fields`, `max_length`, `temperatures` (head) |
| `hashes`, `sizes` | every other file's SHA-256 and size |

A function runs on a baked model through its entry: by name, or (a model
of one function) by signature; its full signature must have the entry's
fingerprint, else it is refused (the function changed since baking).
Format 1 folders (one function, no `functions` list) are refused with
the advice to bake again.
