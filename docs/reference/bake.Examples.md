# bake.Examples { #functai.bake.Examples }

```{.python .no-run}
bake.Examples(rows, *, student=None, template=None, entries=None, info=None)
```

The training conversations for a student, made by
``functai.bake.examples``: what FunctAI will send the student, in a form
every trainer reads.

One row per example: ``function`` (the AI function it trains);
``messages`` (the chat as the function's layout writes it, the reply
last: TRL, Axolotl, Unsloth and prime-rl read it as is); ``prompt`` and
``completion`` (the same chat split at the reply, written on ``save``);
with ``student=``, ``input_ids`` (the exact tokens under the student's
chat template: what training sees and what a call sends),
``answer_start`` (where the reply starts: the loss is on the tokens from
there on), ``prompt_tokens`` and ``answer_tokens``; ``tag`` and
``weight`` (from your columns); ``row_id`` and ``split`` (``train`` or
``validation``); ``source`` (``data``, ``teacher`` or ``program``).

```{.python .no-run}
ex = functai.bake.examples(summarize, rows, student="Qwen/Qwen3.5-4B")
ex.stats()                                   # counts and token lengths
ex = ex.filter(lambda r: r["prompt_tokens"] < 16_000)
ex.save("train.parquet")                     # or .jsonl / .jsonl.gz; ex.to_hf() for datasets
baked = functai.bake.bake(ex)                # or train elsewhere, then functai.bake.adopt(...)
```

## Attributes

| Name | Description |
| --- | --- |
| `functions` | The names of the AI functions the examples train. |
| `table` | The examples as a dpyr table (``pip install "functai[data]"``). |
| `tokenized` | Whether every example carries its tokens (``input_ids``: made with ``student=``). |

## Methods

| Name | Description |
| --- | --- |
| [filter](#functai.bake.Examples.filter) | The examples ``keep(row)`` is true for: ``ex.filter(lambda r: r["prompt_tokens"] < 16_000)``. |
| [from_records](#functai.bake.Examples.from_records) | Examples from plain rows (each needs ``function`` and ``messages``), as ``records()`` gives |
| [load](#functai.bake.Examples.load) | Examples written by ``save`` (or by a bake run). |
| [records](#functai.bake.Examples.records) | Plain rows for files and tables: ``prompt`` and ``completion`` added, |
| [save](#functai.bake.Examples.save) | Write the examples: ``.parquet``, ``.jsonl`` or ``.jsonl.gz``. What |
| [split](#functai.bake.Examples.split) | The examples of one split: ``ex.split("train")`` or ``ex.split("validation")``. |
| [stats](#functai.bake.Examples.stats) | Counts and token lengths (lengths need tokens: ``student=``). |
| [to_hf](#functai.bake.Examples.to_hf) | A Hugging Face ``datasets.Dataset``. |

### filter { #functai.bake.Examples.filter }

```{.python .no-run}
bake.Examples.filter(keep)
```

The examples ``keep(row)`` is true for: ``ex.filter(lambda r: r["prompt_tokens"] < 16_000)``.

### from_records { #functai.bake.Examples.from_records }

```{.python .no-run}
bake.Examples.from_records(recs, meta=None)
```

Examples from plain rows (each needs ``function`` and ``messages``), as ``records()`` gives
them; ``meta`` carries the student, template and layouts when known.

### load { #functai.bake.Examples.load }

```{.python .no-run}
bake.Examples.load(path)
```

Examples written by ``save`` (or by a bake run).

### records { #functai.bake.Examples.records }

```{.python .no-run}
bake.Examples.records(columns=None)
```

Plain rows for files and tables: ``prompt`` and ``completion`` added,
token ids as lists.

### save { #functai.bake.Examples.save }

```{.python .no-run}
bake.Examples.save(path, *, columns=None)
```

Write the examples: ``.parquet``, ``.jsonl`` or ``.jsonl.gz``. What
made them (the functions' layouts, the student's template) goes in the
Parquet file's metadata, or a ``.meta.json`` file beside a JSONL one.

### split { #functai.bake.Examples.split }

```{.python .no-run}
bake.Examples.split(name)
```

The examples of one split: ``ex.split("train")`` or ``ex.split("validation")``.

### stats { #functai.bake.Examples.stats }

```{.python .no-run}
bake.Examples.stats()
```

Counts and token lengths (lengths need tokens: ``student=``).

### to_hf { #functai.bake.Examples.to_hf }

```{.python .no-run}
bake.Examples.to_hf(columns=None)
```

A Hugging Face ``datasets.Dataset``.