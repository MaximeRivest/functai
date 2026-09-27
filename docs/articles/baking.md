---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Bake it into a small model

*Train a small model that answers an AI function, run the same function on it, and send only the unsure cases to a big model.*

A function answered by a big model can be **baked**: a small model is
trained to answer it, and the same function then runs on those weights,
on your own hardware, thousands of times faster and for nothing per
call. The function does not change; what executes it does.

Needs `pip install "functai[bake]"` and, in practice, a GPU.

> **Recorded, not re-run** Training takes a GPU and minutes, so the code on this page is not re-run
> when the site is built. The outputs are from a real run on the banking77
> dataset (77 intents, 13,000 labeled questions) on one RTX 3090.

## Bake

```{.python .no-run}
from typing import Literal
from functai import ai

@ai
def intent(text: str) -> Literal["card_arrival", "card_delivery_estimate", ...]:   # 77 intents
    """The customer's intent."""
    ...

baked = intent.bake(rows, student="jhu-clsp/ettin-encoder-17m")   # rows have a "result" label
print(baked.report)
```

```
Baked intent: jhu-clsp/ettin-encoder-17m (16.9M parameters)
  trained on 8,994 rows (the data's labels), validated on 999, tested on 3,076 labeled rows
  training: 6 passes (best 5), 31 s on cuda:1 (bf16), inputs up to 56 tokens

  on the test rows            student
  accuracy                    90.8% (89.8%–91.8%)
  top-3                       97.1%
  calibration error (ECE)     0.011 (was 0.048; temperature 1.58)

  answering only when sure:  most confident share → accuracy (confidence at the cut)
      50% → 99.6%  (≥ 0.99)
      80% → 98.0%  (≥ 0.89)
      90% → 95.9%  (≥ 0.65)
     100% → 90.8%  (≥ 0.17)
    for 95% accuracy: escalate_below=0.58 keeps 92% of rows

  speed on cuda:1: 18,707 rows/s batched (tokenizing included), 3.7 ms for one row
```

Then run the same function on the baked weights:

```{.python .no-run}
fast = intent.using(lm=baked)
fast("my card still hasn't arrived")          # 'card_arrival'
fast.predict("...").probabilities           # {'result': {'card_arrival': 0.93, ...}}
```

## Two kinds of student

| | `method="head"` (default) | `method="sft"` |
|---|---|---|
| for | a fixed set of answers: `Literal`, `Enum`, `bool`, a dataclass of them | any output: text, numbers, records |
| model | an encoder (Ettin, ModernBERT), or a decoder with a new answer layer | a small chat model (Qwen3.5-0.8B) |
| reads | the input alone, no prompt | the function's full prompt, in its layout |
| gives | a calibrated probability for every answer | the reply, read back into your types |
| speed (one 3090) | 16,000–43,000 rows/s | 12 rows/s in-process, 82 rows/s with `baked.serve()` (vLLM) |

## Where the labels come from

- **Your data.** A column named like an output is a label; a
  `<output>__probs` column (`{answer: probability}`) is a soft label.
- **A teacher.** `teacher="jev-latest"`, or any model or AI function,
  labels the rows. `labels="teacher"` relabels every training row while
  keeping your labels for testing, which measures what the teacher is
  worth.

The report always says which labels the numbers rest on, and says it
loudly when a student is capped by its teacher. The strongest finding of
the experiments behind this feature: a small model trained on **human**
labels (91.5%) beat every teacher's labels (77–82%).

```
on the test rows            student                 teacher (jev-latest)
accuracy                    75.2% (71.2%–78.8%)     80.4%
teacher labels: 2,000 rows from jev-latest in 6 s (1,852 tokens a row, $0.09)
note: result: the student (75.2%) is below its teacher (80.4%); trained on teacher
      labels, it can at best match it.
```

## Escalation: small model first, big model when unsure

Because the confidence is measured, unsure answers can go to a bigger
model:

```{.python .no-run}
cut = baked.report.threshold(0.95)["threshold"]
safe = intent.using(lm=baked, escalate_to="claude-opus-5.5", escalate_below=cut)
p = safe.predict("...")
p.escalated, p.first.confidence          # True, 0.41 when Opus answered
```

On 500 banking77 test questions, a teacher-labelled student alone scored
75.2%; sending its unsure two thirds to Claude Opus 5.5 scored 90.2%,
against 92% for Opus on everything.

## Details that matter

- A function whose inputs, outputs or answers changed since baking is
  refused.
- Training follows what measured best: the whole model, AdamW, warmup
  then cosine, early stopping on held-out rows, and a fitted temperature
  so confidences mean what they say. Everything is overridable.
- The GPU with the most free memory is used; memory held by other
  programs is never taken. A chat model that doesn't fit is trained as a
  LoRA adapter, merged into the saved weights.
- A baked model is a folder: `baked.save(folder)`,
  `functai.bake.load(folder)`. [`functai.save(program)`](saving.md) copies
  a program's baked weights along, and `verify` checks they answer the
  same in a fresh environment.

## Prime Intellect

For reinforcement learning or distillation on Prime Intellect's hosted
training, `functai.bake.prime.env_package(fn, rows, folder, name=...)`
writes a verifiers environment holding the saved program and its rows,
and `functai.bake.prime.config(fn, env=..., model=..., loss="rl")`
writes the `prime train` configuration. Needs `pip install "functai[prime]"`.
