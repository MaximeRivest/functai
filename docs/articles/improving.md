---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Make it better

*Rules in the docstring, examples, optimizers, bigger models: in order of cost, each one measured.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

When a function isn't right often enough, there's a ladder of things to
try, from free to expensive. Climb it in order, and [measure](accuracy.md)
each step on rows the function didn't learn from.

| step | costs | good when |
|---|---|---|
| 1. say the rule in the docstring | nothing | you can put the rule into words |
| 2. a tighter type | nothing | answers drift out of the allowed set, or have the wrong shape |
| 3. examples, by hand | a few tokens per call | the rule is easier to show than to say |
| 4. examples chosen from your data (`functai.bootstrap_few_shot`) | one run over the data | you have labelled rows |
| 5. an optimizer that also rewrites the instruction | many runs | steps 1 to 4 plateau |
| 6. a bigger model | every call | nothing else moved it |

## Two sets of rows

Learn from some rows, judge on the others. Here, the first 40 tickets to
learn from and the last 40 to judge:

```python
from typing import Literal
from dpyr import col

tickets = functai.datasets.tickets()
learn = tickets.filter(col.id <= 40)
judge_on = tickets.filter(col.id > 40)

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

start = functai.evaluate(team, judge_on, expected="category", num_threads=8)
start
```

```output
Evaluation(team, 40 examples: exact_match 0.90 [0.77, 0.96])
```

## 1. Say the rule

Read the misses first. If you can say what they have in common, say it
in the docstring: it's free, exact, and anyone can read it later. (This
is the step [Get started](../get-started.md#say-the-rules) takes.)

## 2. A tighter type

A `Literal` or an `Enum` restricts the answer to a set; a dataclass gives
it named parts; `int | None` allows "no answer" instead of a guess. The
type is part of the prompt, and the reply is checked against it. See
[Types](types.md).

## 3. Examples, by hand

A few worked examples, shown before every question:

```python
@ai(examples=[
    ("The vase came smashed.", "shipping"),
    ("Money back please, the chair wobbles.", "billing"),
])
def team_ex(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

functai.compare(start, functai.evaluate(team_ex, judge_on, expected="category", num_threads=8))
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬───────┬────────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after ┆ diff   ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---   ┆ ---    ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64   ┆ f64    ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪═══════╪════════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.9    ┆ 0.875 ┆ -0.025 ┆ -0.139223 ┆ 0.089223 ┆ 2      ┆ 3     ┆ 35   ┆ 40  │
└─────────────┴────────┴───────┴────────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```

Pairs of `(input, answer)`, or rows like `{"message": ..., "result": ...}`.

## 4. Examples chosen from your data

`functai.bootstrap_few_shot` runs the function on your labelled rows and
keeps up to 4 runs that got the right answer as worked examples (with
their reasoning and tool calls, if any), then adds labelled rows up to 16
examples. It returns an improved copy: `team` itself is unchanged, so you
can measure the two side by side. Your code, types and prompt format are
never touched.

```python
taught = functai.bootstrap_few_shot(team, learn, expected="category")
taught.state()
```

```output
instruction: (written from the code)
examples: 16
  1. message="Hi, my order A-1042 still hasn't arrived and it's been three weeks."  →  result='shipping'
  2. message='The mug arrived in pieces.'  →  result='shipping'
  3. message='I was charged twice for order B-2210, please fix this.'  →  result='billing'
  4. message='How do I change the email on my account?'  →  result='account'
  5. message="When will order B-2417 ship? It says 'processing' for a week."  →  result='shipping'
  6. message='Can you merge my two accounts? I signed up twice by mistake.'  →  result='account'
  7. message='Two of the six plates were broken on arrival, order D-4120.'  →  result='shipping'
  8. message='I want a refund for the chair, it wobbles no matter what I do.'  →  result='billing'
  9. message="My coupon code SPRING10 didn't apply at checkout."  →  result='billing'
  10. message='The glass carafe was shattered when I opened the package.'  →  result='shipping'
  11. message='The non-stick coating is peeling off my frying pan.'  →  result='product'
  12. message='The vase arrived with a big crack down the side.'  →  result='shipping'
  13. message="Order c3319 was delivered to my neighbour's address instead of mine."  →  result='shipping'
  14. message='i cant log in it says my account is locked??'  →  result='account'
  15. message='Is the cutting board safe to use for raw meat?'  →  result='product'
  16. message='Does the stand mixer come with a dough hook?'  →  result='product'
```

```python
optimized = functai.compare(start, functai.evaluate(taught, judge_on, expected="category", num_threads=8))
optimized
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬────────┬───────┬───────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before ┆ after ┆ diff  ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---    ┆ ---   ┆ ---   ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64    ┆ f64   ┆ f64   ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪════════╪═══════╪═══════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.9    ┆ 0.975 ┆ 0.075 ┆ -0.036904 ┆ 0.186904 ┆ 4      ┆ 1     ┆ 35   ┆ 40  │
└─────────────┴────────┴───────┴───────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```


On these 40 judging rows: +5 points (3 rows better, 1 worse), somewhere
between −5 and +15. That range includes 0, so with 40 rows this can't be
told apart from luck; more judging rows would settle it. Measuring is
what keeps you from shipping a change that only looked better.

`team` is still the function you started with; `taught.optimization_runs()`
says how the copy was made. To keep the result, [save the
function](saving.md) (or just its examples: `taught.save("team.json")`).

## 5. Instructions, rewritten and tried

`InstructionSearch` asks a model to propose instructions from your code
and a few rows, tries each (with sets of examples) on small batches, and
keeps the one that scores best on the judging rows. It costs many runs;
use it when steps 1 to 4 have stopped helping.

```{.python .no-run}
from functai import InstructionSearch

search = InstructionSearch(num_candidates=6, num_trials=12)
searched = team.opt(learn, valset=judge_on, expected="category", optimizer=search)
searched.trials      # every try: which instruction, which examples, its score
```

The [translator example](../examples/optimizing_translator.md) runs it
end to end, with a model as the judge.

`GEPA` rewrites the instruction from the function's own mistakes instead:
a `teacher` model reads its answers on a few rows with feedback in words
("wrong: the right answer is billing"), writes a better instruction, and
the best of what it writes is kept, chosen on rows it is never shown.
It is at its best when a small, cheap model runs the function and a large
one writes its instruction once:

```{.python .no-run}
small = team.using(lm="gpt-5.4-nano")                  # the model that will run it
better = functai.gepa(small, learn, expected="category",
                      teacher="gpt-6-sol", budget=300)  # the model that writes its instruction
better.instructions
better.trials        # every instruction tried: its parent, how it was made, its score, its length
```

Its score on the rows it chose with flatters (it is the best of many
there): measure it on rows it never saw. How it differs from the paper's
GEPA, and why, is in `design/04-gepa.md`.

## 6. A bigger model, or a teacher

```{.python .no-run}
team.using(lm="gpt-4.1")                            # a bigger model for every call
functai.bootstrap_few_shot(team, learn, expected="category",
                           teacher="gpt-4.1")       # or: a big model writes the examples once,
                                                    # and the small one uses them from then on
```

The second line is often the better deal: you pay for the big model
once, during optimization, not on every call. [Make it cheaper](cheaper.md)
shows how to check.

## The optimizers

By name, the common cases: `functai.labeled_few_shot(fn, rows, k=8)`,
`functai.bootstrap_few_shot(fn, rows, teacher=...)`, `functai.gepa(fn,
rows, teacher=...)`. Each returns an improved copy. For the others, or
their every option, `fn.opt(rows, optimizer=...)` (it uses
`BootstrapFewShot` unless told otherwise) also returns a copy:

| `optimizer=` | what it does |
|---|---|
| `LabeledFewShot(k=16)` | your labelled rows, as examples |
| `BootstrapFewShot(...)` (default) | runs the function; runs that were right become examples, reasoning and tool calls included |
| `BootstrapFewShotWithRandomSearch(num_candidate_programs=8)` | several sets of examples, keeps the best on `valset` |
| `InstructionSearch(num_candidates=6, num_trials=12)` | proposed instructions × sets of examples, keeps the best on `valset` |
| `GEPA(budget=300, teacher=...)` | a teacher rewrites the instruction from the mistakes, with feedback in words; keeps the best on rows it never shows |

Every option is in the [reference](../reference/index.md#optimizers).
