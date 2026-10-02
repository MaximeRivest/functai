---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data,bake]"]
---

# 7. A model you own

*A bank gets thousands of questions a day in 77 kinds. By the end you
will have trained a small model that answers the same AI function on
your own computer, for nothing per call and thousands of times faster;
measured it against the language model it replaces; and set it to hand
only the questions it's unsure about to the language model.*

**Can you skip this one?** If you can answer these, jump to
[tutorial 8](08-living-with-it.md). The answers are at the bottom.

1. What changes in an AI function when you bake it, and what doesn't?
2. You have no labelled rows. Where do a small model's labels come from,
   and what limits how good it can get?
3. How do you keep a cheap model's accuracy high without paying for a
   big model on every question?

## What you need

This is the one tutorial that trains weights. It needs functai's `bake`
extra (PyTorch and Hugging Face transformers, a few gigabytes) and, in
practice, a GPU: the training below takes about a minute on one RTX 3090
and much longer on a laptop's CPU. It downloads a 17-million-parameter
model (about 70 MB) and a public dataset. About 25 cents of model calls.

```{.python .no-run}
pip install "functai[data,bake]"
```

```python
import tempfile
from typing import Literal

from dpyr import col, n, read

import functai
from functai import ai

log_folder = tempfile.mkdtemp()
functai.configure(lm="gpt-6-luna", log_calls=log_folder)
```

```output
configure(lm='gpt-6-luna', log_calls='/tmp/tmp0e_w_vy1')
```

## The questions

[banking77](https://github.com/PolyAI-LDN/task-specific-datasets) is a
public dataset of 13,000 real questions to a bank, each labelled with
one of 77 intents by people. It's a fair stand-in for any
high-volume classification job: support tickets, forms, logs.

```python
url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
train = read(url + "train.csv")
test = read(url + "test.csv")
train
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌──────────────────────────────────────────────────────────────┬──────────────┐
│ text                                                         ┆ category     │
│ ---                                                          ┆ ---          │
│ str                                                          ┆ str          │
╞══════════════════════════════════════════════════════════════╪══════════════╡
│ I am still waiting on my card?                               ┆ card_arrival │
│ What can I do if my card still hasn't arrived after 2 weeks? ┆ card_arrival │
│ I have been waiting over a week. Is the card still coming?   ┆ card_arrival │
│ Can I track my card while it is in the process of delivery?  ┆ card_arrival │
│ How do I know if I will get my card, or if it is lost?       ┆ card_arrival │
│ When did you send me my new card?                            ┆ card_arrival │
│ Do you have info about the card on delivery?                 ┆ card_arrival │
│ What do I do if I still have not received my new card?       ┆ card_arrival │
│ Does the package with my card have tracking?                 ┆ card_arrival │
│ I ordered my card but it still isn't here                    ┆ card_arrival │
└──────────────────────────────────────────────────────────────┴──────────────┘
```

```python
intents = sorted(set(train.pull(col.category)))
len(intents), intents[:8]
```

```output
(77, ['Refund_not_showing_up', 'activate_my_card', 'age_limit', 'apple_pay_or_google_pay', 'atm_support', 'automatic_top_up', 'balance_not_updated_after_bank_transfer', 'balance_not_updated_after_cheque_or_cash_deposit'])
```

Seventy-seven answers is too many to type, so build the `Literal` from
the list. The function is as short as ever:

```python
Intent = Literal[tuple(intents)]

@ai
def intent(text: str) -> Intent:
    """What the bank's customer wants."""
    ...
```

We'll measure everything on the same 300 test questions (with the
answers in a column named like the output, `result`), and keep the rest
of the test set for the training reports.

```python
labelled_test = test.rename(result=col.category)
measure_on = labelled_test.slice_sample(n=300, seed=7)
```

## The language model, measured

```python
ev_luna = functai.evaluate(intent, measure_on, num_threads=16)
ev_luna
```

```output
Evaluation(intent, 300 examples: exact_match 0.81 [0.77, 0.85])
```

```python
ev_luna.table.summarize(seconds_each=col.seconds.median(),
                        dollars_per_1000=(1000 * (col.input_tokens * 0.10 + (col.total_tokens - col.input_tokens) * 0.50) / 1e6).mean())
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌──────────────┬──────────────────┐
│ seconds_each ┆ dollars_per_1000 │
│ ---          ┆ ---              │
│ f64          ┆ f64              │
╞══════════════╪══════════════════╡
│ 1.15583      ┆ 0.075929         │
└──────────────┴──────────────────┘
```

Good, and cheap per question. But a bank that gets a million questions
a month pays for a million calls, waits a second for each, and sends
every customer's words to another company. For a job this narrow, there
is another way.

## Bake it

**Baking** trains a small model to answer an AI function, then runs the
same function on those weights. The function doesn't change (same name,
same types, same answers); what executes it does. The student here is
Ettin, a 17-million-parameter encoder: it reads the question and gives a
probability for each of the 77 answers.

First, with the people's labels, all 10,000 training questions:

```python
people = intent.bake(train.rename(result=col.category), test=labelled_test.slice_head(n=1000), log=False)
print(people.report)
```

```output
Baked intent: jhu-clsp/ettin-encoder-17m (16.9M parameters)
  trained on 9,003 rows (the data's labels), validated on 1,000, tested on 1,000 labeled rows
  training: 6 passes (best 6), 36 s on cuda:1 (bf16), inputs up to 56 tokens

  on the test rows            student                 
  accuracy                    91.1% (89.2%–92.7%)     
  top-3                       96.7%
  calibration error (ECE)     0.025 (was 0.050; temperature 1.74)

  answering only when sure:  most confident share → accuracy (confidence at the cut)
      50% → 99.8%  (≥ 0.98)
      80% → 98.0%  (≥ 0.86)
      90% → 95.3%  (≥ 0.65)
     100% → 91.1%  (≥ 0.15)
    for 95% accuracy: escalate_below=0.61 keeps 91% of rows

  speed on cuda:1: 16,122 rows/s batched (tokenizing included), 4.8 ms for one row
```

Read the report top to bottom: what it was trained on and how long it
took, its accuracy on the test rows with an interval, its **calibration
error** (how far its confidences are from the truth; lower is better,
and a fitted temperature corrects it), the accuracy you get if it only
answers when it's sure, and its speed.

Now run the same function on the baked weights. `using(lm=...)` takes a
baked model like any model name:

```python
fast = intent.using(lm=people)

p = fast.predict("my card still hasn't arrived after two weeks")
p.result, round(p.confidence, 3)
```

```output
('card_arrival', 0.99)
```

On the same 300 questions:

```python
ev_fast = functai.evaluate(fast, measure_on, num_threads=16)
ev_fast
```

```output
Evaluation(intent, 300 examples: exact_match 0.89 [0.85, 0.92])
```

```python
functai.compare(ev_luna, ev_fast)
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬──────────┬──────────┬──────┬──────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before   ┆ after    ┆ diff ┆ low      ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---      ┆ ---      ┆ ---  ┆ ---      ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64      ┆ f64      ┆ f64  ┆ f64      ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪══════════╪══════════╪══════╪══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.813333 ┆ 0.893333 ┆ 0.08 ┆ 0.032581 ┆ 0.127419 ┆ 39     ┆ 15    ┆ 246  ┆ 300 │
└─────────────┴──────────┴──────────┴──────┴──────────┴──────────┴────────┴───────┴──────┴─────┘
```

The paired comparison says whether your own model is clearly worse,
clearly better, or indistinguishable from the language model on these
questions. On banking77 the small model trained on people's labels comes
out ahead, and that isn't a fluke: 77 intents drawn where this bank's
labellers drew them are exactly what a language model has to guess, and
exactly what labels teach. And it costs nothing per call, runs on your
hardware, and answers in milliseconds.

## No labels? A teacher

Most jobs don't start with 10,000 labelled rows. Then the language model
can be the **teacher**: it labels unlabelled questions once, and the
student learns from its labels. Here, 2,000 training questions with the
labels thrown away:

```python
unlabelled = train.select(col.text).slice_sample(n=2000, seed=1)

taught = intent.bake(unlabelled, teacher="gpt-6-luna", test=labelled_test.slice_head(n=500),
                     compare_teacher=True, prices={"teacher": (0.10, 0.50)}, log=False)
print(taught.report)
```

```output
Baked intent: jhu-clsp/ettin-encoder-17m (16.9M parameters)
  trained on 1,800 rows (teacher (hard)), validated on 200, tested on 500 labeled rows
  training: 30 passes (best 28), 37 s on cuda:1 (bf16), inputs up to 56 tokens

  on the test rows            student                 teacher (gpt-6-luna)
  accuracy                    72.6% (68.5%–76.3%)     87.6%
  top-3                       87.2%
  calibration error (ECE)     0.078 (was 0.190; temperature 1.99)
  agrees with the teacher     77.2%

  answering only when sure:  most confident share → accuracy (confidence at the cut)
      50% → 92.8%  (≥ 0.88)
      80% → 84.2%  (≥ 0.56)
      90% → 78.4%  (≥ 0.39)
     100% → 72.6%  (≥ 0.16)
    for 95% accuracy: escalate_below=0.94 keeps 38% of rows

  speed on cuda:1: 11,977 rows/s batched (tokenizing included), 4.7 ms for one row
  teacher labels: 2,000 rows from gpt-6-luna in 172 s (508 tokens a row, $0.15)
  break-even in time against the teacher: after 2,430 rows

  note: result: the student (72.6%) is below its teacher (87.6%); trained on teacher labels, it can at best match it. Human labels lifted the same kind of student from 77% to 91.5% on banking77: label more rows by hand, or use a stronger teacher
```

With `compare_teacher=True` (it costs a teacher pass over the test rows,
so it is off unless asked), the report shows the teacher next to the
student, on the same test rows, and what the labels cost. It also says the thing to remember: a
student trained on a teacher's labels can at best match the teacher,
and usually lands a little below it. People's labels, when you have
them, are worth more than any teacher's.

## Small model first, language model when unsure

The student's confidence is calibrated, so you can use it the way
tutorial 6 used Jev's: answer when sure, and send the rest to the
language model. The report tells you where to cut for a target accuracy:

```python
cut = people.report.threshold(0.95)
cut
```

```output
{'threshold': 0.6097557957281937, 'share': 0.914, 'accuracy': 0.9507658643326039}
```

```python
careful = intent.using(lm=people, escalate_to="gpt-6-luna", escalate_below=cut["threshold"])

ev_careful = functai.evaluate(careful, measure_on, num_threads=16)
escalated = sum(bool(p and p.escalated) for p in ev_careful.predictions)
ev_careful, f"{escalated} of {len(ev_careful)} asked gpt-6-luna"
```

```output
(Evaluation(intent, 300 examples: exact_match 0.91 [0.87, 0.93]), '28 of 300 asked gpt-6-luna')
```

```python
read([{"setup": name, **ev.summary.collect().to_dicts()[0]} for name, ev in [
    ("gpt-6-luna on everything", ev_luna),
    ("your model alone", ev_fast),
    ("your model, gpt-6-luna when unsure", ev_careful),
]]).select(col.setup, col.mean, col.low, col.high)
```

```output
# dpyr dataframe · source: polars · showing 3 of 3 rows
┌────────────────────────────────────┬──────────┬──────────┬──────────┐
│ setup                              ┆ mean     ┆ low      ┆ high     │
│ ---                                ┆ ---      ┆ ---      ┆ ---      │
│ str                                ┆ f64      ┆ f64      ┆ f64      │
╞════════════════════════════════════╪══════════╪══════════╪══════════╡
│ gpt-6-luna on everything           ┆ 0.813333 ┆ 0.765381 ┆ 0.853363 │
│ your model alone                   ┆ 0.893333 ┆ 0.853297 ┆ 0.923424 │
│ your model, gpt-6-luna when unsure ┆ 0.906667 ┆ 0.868415 ┆ 0.934636 │
└────────────────────────────────────┴──────────┴──────────┴──────────┘
```

Most questions are answered by your model, for free; the language model
sees only the ones your model was unsure of. Compare the accuracy and
the number of paid calls: that's the trade you tune with
`escalate_below`.

## Keep it

A baked model is a folder: the weights, the tokenizer, and what it was
trained to answer. Load it back anywhere functai runs:

```python
import os
folder = os.path.join(tempfile.mkdtemp(), "intent-model")
people.save(folder)

from functai.bake import load
reloaded = load(folder)
intent.using(lm=reloaded)("How do I top up with Apple Pay?")
```

```output
'apple_pay_or_google_pay'
```

A function whose inputs, outputs or answers change is refused by the
baked model it no longer matches, so a stale model can't silently answer
the wrong question. `functai.save()` of a program copies its baked
weights along (tutorial 8).

## What it cost

```python
functai.calls(folder=log_folder).summarize(
    calls=n(), dollars=((col.input_tokens * 0.10 + (col.total_tokens - col.input_tokens) * 0.50) / 1e6).sum())
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌───────┬─────────┐
│ calls ┆ dollars │
│ ---   ┆ ---     │
│ i64   ┆ f64     │
╞═══════╪═════════╡
│ 3402  ┆ 0.21506 │
└───────┴─────────┘
```

The training itself cost only electricity: a minute of GPU.

## Your turn

1. Bake with `teacher="jev-latest"` instead. Jev gives a probability per
   answer, which the student learns from as a soft label. Is the student
   better than with `gpt-6-luna`'s labels? Cheaper?
2. Try `escalate_below` at 0.5, 0.8 and 0.95 of `people`'s confidence.
   Plot paid calls against accuracy.
3. Bake with only 500 of the people's labels. How much accuracy does the
   data buy? (That curve tells you how much labelling is worth.)

## What you learned

- `fn.bake(rows)` trains a small model to answer an AI function;
  `fn.using(lm=baked)` runs the same function on it, on your hardware,
  for nothing per call.
- The report measures the student on held-out rows: accuracy with an
  interval, calibration, accuracy when answering only when sure, speed.
- Without labels, a `teacher=` model labels the rows once; the student
  can at best match its teacher. People's labels are worth more.
- A calibrated student can escalate: `escalate_to=` a language model,
  below a confidence the report suggests.
- A baked model is a folder: `save()`, `functai.bake.load()`.

**Answers to the check at the top.** (1) What executes it: the name,
types, docstring and answers stay; the weights answering change. (2) From
a teacher (a language model, or Jev) that labels them once; the student
can at best match the teacher. (3) Answer with the small model when it's
sure, and escalate the rest (`escalate_to=`, `escalate_below=`).

**Next:** [8. Living with it](08-living-with-it.md): tools, the call
log, people's corrections, versions, and saving a whole program.
