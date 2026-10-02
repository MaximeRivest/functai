---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]"]
---

# 4. Making it better without fooling yourself

*A refund desk, a function that decides, and three ways to improve it:
write the rules down, show it examples, let a stronger model teach it.
By the end you will know which helped, by how much, and how sure you can
be, because you will have kept one set of rows aside and looked at it
only once.*

**Can you skip this one?** If you can answer these, jump to
[tutorial 5](05-choosing-a-model.md). The answers are at the bottom.

1. You try five versions of a docstring and keep the one that scores
   best on your test rows. Why is its score too optimistic?
2. Two versions are right on 36 and 38 of the same 40 rows. What should
   you count to know if the difference is real?
3. What does `functai.labeled_few_shot` change in a function, and what
   does it leave alone?

## The refund desk

The homeware shop from tutorial 1 gets refund requests too. For each one,
someone decides: **approve** or **deny**. `functai.datasets.refunds()`
has 120 of them: what the customer wrote, what the order system knows
(price, days since delivery, whether it was a final-sale item), and the
decision the shop's rules give.

```python
import random
import tempfile
from typing import Literal

from dpyr import col, n, read

import functai
from functai import ai

log_folder = tempfile.mkdtemp()
functai.configure(lm="gpt-6-luna", log_calls=log_folder)

refunds = functai.datasets.refunds()
refunds.select(col.message, col.price, col.days_since_delivery, col.final_sale, col.decision)
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌─────────────────────────────────────────────────────────────────────────────────────────────┬────────┬─────────────────────┬────────────┬──────────┐
│ message                                                                                     ┆ price  ┆ days_since_delivery ┆ final_sale ┆ decision │
│ ---                                                                                         ┆ ---    ┆ ---                 ┆ ---        ┆ ---      │
│ str                                                                                         ┆ f64    ┆ i64                 ┆ bool       ┆ str      │
╞═════════════════════════════════════════════════════════════════════════════════════════════╪════════╪═════════════════════╪════════════╪══════════╡
│ Hiya, just a heads up - the wall clock arrived over 9 weeks ago now and it's totally the w… ┆ 28.51  ┆ 66                  ┆ false      ┆ deny     │
│ Ordered the grey rug, got sent green instead 4 weeks ago now, want a refund.                ┆ 39.22  ┆ 28                  ┆ false      ┆ approve  │
│ Hiya, so I'd only made about three batches of cookies with it when it just went quiet mid-… ┆ 9.38   ┆ 9                   ┆ false      ┆ approve  │
│ Hi, I bought a planter from you about 8 weeks ago and it was fine to begin with, but now i… ┆ 139.47 ┆ 57                  ┆ false      ┆ approve  │
│ I'm sorry to even ask this after so long, it's been about a year since these arrived, I've… ┆ 75.9   ┆ 368                 ┆ false      ┆ deny     │
│ I need a refund for the chair I got about two and a half weeks ago. It was fine at first b… ┆ 388.71 ┆ 17                  ┆ false      ┆ approve  │
│ Hello, I ordered a casserole dish about four weeks ago, and after taking it out of the box… ┆ 458.27 ┆ 26                  ┆ false      ┆ approve  │
│ It's been four weeks since this arrived and I still can't pour a cup of tea without the li… ┆ 11.63  ┆ 28                  ┆ true       ┆ approve  │
│ I'm writing about the pair of pillows I got about 3-4 weeks ago. They were fine at first b… ┆ 128.65 ┆ 25                  ┆ false      ┆ approve  │
│ Hi! So this ended up still boxed up untouched in my hallway since it arrived about three m… ┆ 50.05  ┆ 94                  ┆ false      ┆ deny     │
└─────────────────────────────────────────────────────────────────────────────────────────────┴────────┴─────────────────────┴────────────┴──────────┘
```

```python
refunds.filter(col.id == 5).pull(col.message)[0]
```

```output
"I'm sorry to even ask this after so long, it's been about a year since these arrived, I've been using them regularly, but the colour just never really settled in with the rest of my bathroom and I keep reaching for a different set instead, so I was hoping a refund might still be possible?\n\nPriya"
```

A function that decides takes the message *and* the facts. Every input
is a typed parameter:

```python
Decision = Literal["approve", "deny"]

@ai
def refund(message: str, price: float, days_since_delivery: int, final_sale: bool) -> Decision:
    """Should the shop refund this request?"""
    ...
```

## Three piles of rows, used for three different things

Every change you try is a guess. To know if a guess helped, you measure
it. The trap is measuring every guess on the same rows and keeping the
winner: you end up choosing the version that got lucky on those rows,
and its score flatters it. (There's a simulation of exactly this below.)

So split the 120 requests three ways, with the same mix of decisions in
each:

- **examples** (about 40): rows the function may learn from (worked
  examples);
- **dev** (about 40): rows you measure your guesses on, as often as you
  like;
- **test** (about 40): rows you look at **once**, at the very end, for
  the version you chose.

```python
rng = random.Random(2026)
piles = {"examples": [], "dev": [], "test": []}
for decision in ["approve", "deny"]:
    ids = refunds.filter(col.decision == decision).pull(col.id)
    rng.shuffle(ids)
    for i, id_ in enumerate(ids):
        piles[["examples", "dev", "test"][i % 3]].append(id_)

examples = refunds.filter(col.id.is_in(piles["examples"]))
dev = refunds.filter(col.id.is_in(piles["dev"]))
test = refunds.filter(col.id.is_in(piles["test"]))
{name: len(ids) for name, ids in piles.items()}
```

```output
{'examples': 41, 'dev': 40, 'test': 39}
```

## Where we start

```python
ev_plain = functai.evaluate(refund, dev, expected="decision", num_threads=8)
ev_plain
```

```output
Evaluation(refund, 40 examples: exact_match 0.90 [0.77, 0.96])
```

The model has never seen the shop's rules, yet it's right most of the
time: it knows what refund policies usually say. What's it getting
wrong? Look at the dev rows (never the test rows):

```python
(ev_plain.table.filter(col.exact_match == 0)
    .select(col.decision, col.pred_result, col.days_since_delivery, col.final_sale, col.state, col.message))
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌──────────┬─────────────┬─────────────────────┬────────────┬────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
│ decision ┆ pred_result ┆ days_since_delivery ┆ final_sale ┆ state  ┆ message                                                                                 │
│ ---      ┆ ---         ┆ ---                 ┆ ---        ┆ ---    ┆ ---                                                                                     │
│ str      ┆ str         ┆ i64                 ┆ bool       ┆ str    ┆ str                                                                                     │
╞══════════╪═════════════╪═════════════════════╪════════════╪════════╪═════════════════════════════════════════════════════════════════════════════════════════╡
│ deny     ┆ approve     ┆ 6                   ┆ false      ┆ used   ┆ hi, i got the wool throw about a week ago and its fine and all but the colour just      │
│          ┆             ┆                     ┆            ┆        ┆ doesnt …                                                                                │
│ approve  ┆ deny        ┆ 71                  ┆ false      ┆ faulty ┆ Since about ten weeks in, it's been pooling water underneath after each morning's use,  │
│          ┆             ┆                     ┆            ┆        ┆ so …                                                                                    │
│ deny     ┆ approve     ┆ 20                  ┆ false      ┆ used   ┆ I got this about three weeks ago and I've been cooking with it since, but honestly it's │
│          ┆             ┆                     ┆            ┆        ┆ ju…                                                                                     │
│ approve  ┆ deny        ┆ 11                  ┆ true       ┆ faulty ┆ Subject: Refund Request                                                                 │
│          ┆             ┆                     ┆            ┆        ┆                                                                                         │
│          ┆             ┆                     ┆            ┆        ┆ Dear Sir/Madam,                                                                         │
│          ┆             ┆                     ┆            ┆        ┆                                                                                         │
│          ┆             ┆                     ┆            ┆        ┆ I am writing regarding the shoe rack delivered t…                                       │
└──────────┴─────────────┴─────────────────────┴────────────┴────────┴─────────────────────────────────────────────────────────────────────────────────────────┘
```

(`state` is the item's true condition, which the shop's staff recorded.
The function doesn't see it; we do, to understand the mistakes.)

Look at the kind of mistake. Two approve a used item the customer
simply no longer wants, as if any return within a few weeks were fine;
one denies a fault reported after ten weeks, as if a 30-day window
applied to everything. The model assumed the most common policy, and
this shop's differs both ways: nothing for a used item, and far more
generous with damage and faults. Nothing told the model so.

## 1. Write the rules down

The rules are in the docstring of `functai.datasets.refunds`. They're
the shop's policy, not something we invented by staring at the mistakes,
which matters: rules written to fix particular dev rows would be fitted
to those rows.

```python
@ai
def refund_rules(message: str, price: float, days_since_delivery: int, final_sale: bool) -> Decision:
    """Should the shop refund this request? Follow the refund rules exactly:

    - Damaged on arrival, or the wrong item (or part of the order missing): refund within 60 days
      of delivery, final sale or not.
    - Faulty (it failed in normal use): refund within 365 days, final sale or not.
    - Unopened, or opened but not used, and no longer wanted: refund within 30 days, never for a
      final-sale item.
    - Used and no longer wanted: no refund.
    """
    ...

ev_rules = functai.evaluate(refund_rules, dev, expected="decision", num_threads=8)
ev_rules
```

```output
Evaluation(refund_rules, 40 examples: exact_match 0.97 [0.87, 1.00])
```

## 2. Show it worked examples

A new colleague learns from rules, and also from seeing past cases.
`functai.labeled_few_shot` picks rows with known answers and puts them in
front of every question as solved examples. It returns an improved copy;
`refund_rules` stays as it was:

```python
refund_shown = functai.labeled_few_shot(refund_rules, examples, k=8, expected="decision")

ev_shown = functai.evaluate(refund_shown, dev, expected="decision", num_threads=8)
ev_shown
```

```output
Evaluation(refund_rules, 40 examples: exact_match 0.97 [0.87, 1.00])
```

Each version has its own `version`, a fingerprint of everything it sends
besides the inputs (the instruction, the layout, the examples). The call
log files every call under it, so later you can tell which version gave
which answer:

```python
{"plain": refund.version, "rules": refund_rules.version, "shown": refund_shown.version}
```

```output
{'plain': 'sha256:519d420f6444c4c02c12f66c84f6fd9b69a757874b3756c913f2cd18b5432e4b', 'rules': 'sha256:bd8215cefb14540523a56f100b01d410a7927a398aa330139e1bacee3233b2cf', 'shown': 'sha256:b6ac834d612056cd83c1a761b5ce9525c96c9dab337f5137866484477d82df2d'}
```

## 3. Let a stronger model teach

Labelled rows show the answer, not the thinking. `functai.bootstrap_few_shot` runs
a **teacher** on the example rows, keeps the runs whose answer was right,
and uses those as the worked examples. Here the teacher is `gpt-6-sol`,
OpenAI's larger current model: twenty times the price per token of
`gpt-6-luna`, but it only answers a handful of rows, once.

```python
refund_taught = functai.bootstrap_few_shot(refund_rules, examples, expected="decision",
                                           teacher="gpt-6-sol", max_bootstrapped=4, max_labeled=4)

ev_taught = functai.evaluate(refund_taught, dev, expected="decision", num_threads=8)
ev_taught
```

```output
Evaluation(refund_rules, 40 examples: exact_match 0.97 [0.87, 1.00])
```

(A fourth lever, `@ai(module="cot")`, asks the model to reason before
answering. Current models like `gpt-6-luna` already think before they
answer, so it changes little here; it helps older and smaller models
that don't.)

## Which helped?

All four on the dev rows, with their intervals:

```python
read([{"version": name, **ev.summary.collect().to_dicts()[0]} for name, ev in [
    ("1. no rules", ev_plain), ("2. the rules", ev_rules),
    ("3. rules + 8 examples", ev_shown), ("4. rules + taught by gpt-6-sol", ev_taught),
]]).select(col.version, col.mean, col.low, col.high)
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌────────────────────────────────┬───────┬──────────┬──────────┐
│ version                        ┆ mean  ┆ low      ┆ high     │
│ ---                            ┆ ---   ┆ ---      ┆ ---      │
│ str                            ┆ f64   ┆ f64      ┆ f64      │
╞════════════════════════════════╪═══════╪══════════╪══════════╡
│ 1. no rules                    ┆ 0.9   ┆ 0.769482 ┆ 0.96042  │
│ 2. the rules                   ┆ 0.975 ┆ 0.871186 ┆ 0.995573 │
│ 3. rules + 8 examples          ┆ 0.975 ┆ 0.871186 ┆ 0.995573 │
│ 4. rules + taught by gpt-6-sol ┆ 0.975 ┆ 0.871186 ┆ 0.995573 │
└────────────────────────────────┴───────┴──────────┴──────────┘
```

The intervals are wide: forty rows can't separate versions a few points
apart. But these versions were run on the *same* forty rows, and that
gives a much sharper test. Only the rows where two versions **disagree**
carry information about which is better. `functai.compare()` counts them
and puts an interval on the difference:

```python
functai.compare(ev_plain, ev_rules)
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬───────┬───────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after ┆ diff  ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---   ┆ ---   ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64   ┆ f64   ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪═══════╪═══════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.9    ┆ 0.975 ┆ 0.075 ┆ -0.010308 ┆ 0.160308 ┆ 3      ┆ 0     ┆ 37   ┆ 40  │
└─────────────┴────────┴───────┴───────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```

`better` is how many rows the rules got right that the plain version got
wrong; `worse`, the other way round; `same`, the rows where they agreed.
The interval is on the difference in accuracy. When it clears zero, the
change is very unlikely to be luck; when it includes zero, as it will
when only a row or two changed, forty rows can't tell. The honest
summary is then "every disagreement went the rules' way; on forty rows
that's suggestive, not proof". Knowing *why* it helped settles the rest:
the rules are the shop's policy, not a guess.

```python
functai.compare(ev_rules, ev_taught)
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬───────┬──────┬─────┬──────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after ┆ diff ┆ low ┆ high ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---   ┆ ---  ┆ --- ┆ ---  ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64   ┆ f64  ┆ f64 ┆ f64  ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪═══════╪══════╪═════╪══════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.975  ┆ 0.975 ┆ 0.0  ┆ 0.0 ┆ 0.0  ┆ 0      ┆ 0     ┆ 40   ┆ 40  │
└─────────────┴────────┴───────┴──────┴─────┴──────┴────────┴───────┴──────┴─────┘
```

Once the rules are in, examples and a teacher have little left to fix on
these rows: look at `better` and `worse`. That's a result too.

## The trap, simulated

Why not just pick the best dev score and report it? Suppose you tried
five versions that are all, truly, right 90% of the time, and scored
each on 40 rows. Simulate it, for free:

```python
rng = random.Random(1)
best_of_five = [max(sum(rng.random() < 0.9 for _ in range(40)) for _ in range(5)) / 40
                for _ in range(10_000)]
sum(best_of_five) / len(best_of_five)
```

```output
0.951675
```

Every version is 90%, yet the winner scores about 95% on average, just
by being the luckiest of five. The more versions you try on the same
rows, the bigger the flattery. That's why the test rows exist.

## Once, at the end

Choose on dev. When versions tie, choose the **simplest**: here, the
rules alone. The teacher's version was right on one more row, which
forty rows can't tell from luck, and worked examples make every call
longer and so dearer, forever. Now, once, the test rows:

```python
ev_final = functai.evaluate(refund_rules, test, expected="decision", num_threads=8)
ev_final
```

```output
Evaluation(refund_rules, 39 examples: exact_match 0.97 [0.87, 1.00])
```

That is the number to report. When it's lower than on dev, as it often
is, that's the flattery leaving, not a failure.

## What it cost

```python
prices = read([   # dollars per million tokens, 2026-09-27
    {"model": "gpt-6-luna", "input": 0.10, "output": 0.50},
    {"model": "gpt-6-sol", "input": 2.00, "output": 10.00},
])

functai.calls(folder=log_folder).left_join(prices, on=col.model).group_by(col.model).summarize(
    calls=n(), dollars=((col.input_tokens * col.input + (col.total_tokens - col.input_tokens) * col.output) / 1e6).sum())
```

```output
# dpyr dataframe · source: polars · showing 2 of 2 rows
┌────────────┬───────┬───────────┐
│ model      ┆ calls ┆ dollars   │
│ ---        ┆ ---   ┆ ---       │
│ str        ┆ i64   ┆ f64       │
╞════════════╪═══════╪═══════════╡
│ gpt-6-luna ┆ 199   ┆ 0.0141581 │
│ gpt-6-sol  ┆ 8     ┆ 0.006538  │
└────────────┴───────┴───────────┘
```

Worked examples make every question longer (each call now carries four
to eight solved cases), so they cost more per call. On a small model
that's still cents. Weigh it anyway: it's paid on every call, forever.

## Your turn

1. Try `functai.labeled_few_shot(..., k=16)`. Measure it on dev and `compare()` it with
   the rules-only version.
2. Add one sentence to the rules that you think would fix a dev mistake.
   Is it policy, or is it fitted to that row? How could you tell?
3. `print(refund_taught.render("x", 1.0, 1, False).system)` and look at
   `refund_taught.render("x", 1.0, 1, False).messages`: find the taught
   examples, and count how much longer the request is than
   `refund_rules`'s.

## What you learned

- Split once, before you start: rows to learn from, rows to choose on,
  rows to test once.
- Writing the rules down is the most direct improvement, when the rules
  are yours to write.
- `functai.labeled_few_shot` adds solved examples;
  `functai.bootstrap_few_shot(teacher=...)` adds a teacher's runs that
  were right. Each returns an improved copy, with a new `version`.
- On the same rows, compare versions by their disagreements
  (`functai.compare()`), not by eyeballing two intervals.
- When versions tie, keep the simplest and cheapest.
- Picking the best of several on the same rows flatters the winner. The
  test rows, used once, give the honest number.

**Answers to the check at the top.** (1) It was chosen for being the
luckiest on those rows, so part of its score is luck that won't come
back: the winner's curse. (2) The rows where they disagree: how many one
got right and the other wrong, each way (`functai.compare()`). (3) It
returns a copy whose requests carry worked examples; the instruction,
inputs, outputs and model stay as they were, and the function you passed
in is unchanged.

**Next:** [5. Choosing a model](05-choosing-a-model.md) puts eight
current models through the same test and weighs accuracy against cost
and speed.
