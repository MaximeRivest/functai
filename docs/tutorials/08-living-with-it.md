---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]"]
---

# 8. Living with it

*A function in a notebook is an experiment. A function the shop relies
on needs more: it has to look things up instead of guessing, leave a
record of every answer, learn from the people who correct it, and travel
to wherever it's needed. By the end you will have done all four.*

**Can you skip this one?** If you can answer these, you've finished the
series. The answers are at the bottom.

1. When a model "calls a tool", who runs the code?
2. Where does a correction a person makes to an answer end up, and how
   do you use it?
3. What does a function's `version` identify, and what doesn't change
   it?

## Setting up

```python
import os
import random
import tempfile
from typing import Literal

from dpyr import col, n, read

import functai
from functai import ai

log_folder = tempfile.mkdtemp()
functai.configure(lm="gpt-6-luna", log_calls=log_folder)

tickets = functai.datasets.tickets()
```

In real use you'd write `functai.configure(log_calls=True)` (or set
`FUNCTAI_LOG_CALLS=1`), and calls would go to one folder on your
machine, the same one functai in R and TypeScript use. We keep this
tutorial's calls in a folder of their own.

## Looking things up instead of guessing

Customers ask where their order is. The model can't know: the answer is
in the shop's order system, which here is a dictionary:

```python
ORDERS = {
    "A-1042": ("in transit", "held at the Montreal depot since September 8"),
    "C-3319": ("delivered", "left with a neighbour at 14 Elm Street on September 20"),
    "A-1299": ("delivered", "the courier's photo shows it at the side door, September 22"),
    "B-2417": ("processing", "waiting for stock; ships October 2"),
    "A-1350": ("in transit", "delayed by the carrier; new estimate September 30"),
    "C-3480": ("processing", "not shipped yet, so the address can still be changed"),
}
```

A **tool** is an ordinary Python function the model may ask for. Its
name, docstring and typed parameters tell the model what it does, just
like an AI function's:

```python
looked_up = []

def lookup_order(order: str) -> str:
    """Look up an order's delivery status by its number: a letter, a dash and four digits, like A-1042."""
    looked_up.append(order)                   # so we can see what the model asked for
    if order.upper() not in ORDERS:
        return "there is no order with that number"
    status, note = ORDERS[order.upper()]
    return f"{status}: {note}"
```

Get the mental model right, because it's easy to get wrong. The model
does **not** run your function. It replies, in effect, "please run
`lookup_order` with `A-1042`". functai runs your Python function, in your
process, sends the result back, and the model then writes its answer, or
asks for another lookup. The model's skill is knowing *when* to look
something up and *what* to ask for.

```python
@ai(tools=[lookup_order])
def where_is(message: str) -> str:
    """Tell the customer where their order is, in one or two friendly sentences. Use only what the
    order system says; never guess a date. If the message has no order number, ask for it."""
    ...

asked = tickets.filter(col.order_id.is_in(list(ORDERS))).pull(col.message) + [
    "Where is my order Z-9999?", "Hi, where's my parcel? It's been ages."]

for message in asked:
    print(message, "\n  ->", where_is(message), "\n")
looked_up
```

```output
Hi, my order A-1042 still hasn't arrived and it's been three weeks. 
  -> Your order is in transit and has been held at the Montreal depot since September 8. I’m sorry it hasn’t arrived yet. 

Tracking for C-3319 hasn't moved since Monday. 
  -> Your order was delivered and left with a neighbour at 14 Elm Street on September 20. 

Order c3319 was delivered to my neighbour's address instead of mine. 
  -> Could you please provide your order number in the format of a letter, a dash, and four digits (for example, A-1042)? 

My parcel says delivered but it's not here. Order A-1299. 
  -> The courier’s photo shows your parcel at the side door. The order system lists it as delivered on September 22. 

When will order B-2417 ship? It says 'processing' for a week. 
  -> Your order is processing and waiting for stock. The system says it ships October 2. 

Order #A-1350 is two weeks late. Any news? 
  -> Your order is in transit and delayed by the carrier. The new estimated delivery date is September 30. 

Can I change the delivery address for order C-3480? It hasn't shipped yet. 
  -> Yes, since order C-3480 hasn’t shipped yet, the delivery address can still be changed. 

Where is my order Z-9999? 
  -> I couldn’t find an order with number Z-9999. Please check the number and send it again. 

Hi, where's my parcel? It's been ages. 
  -> I’m sorry it’s taking so long. Could you share your order number so I can check where your parcel is? 

['A-1042', 'C-3319', 'A-1299', 'B-2417', 'A-1350', 'C-3480', 'Z-9999']
```

Read the replies against `ORDERS`, and `looked_up` against the messages.
The unknown order should be reported, not invented; the message with no
number should get a question, because there was nothing to look up;
every date should come from the table.

Now look closely at the message that writes its number as "c3319". Did
the model look up `C-3319`, or ask the customer for a number they had
already given? Models read a tool's description literally: we said "a
letter, a dash and four digits", and "c3319" has no dash. If it asked
again, the fix is one sentence in the docstring ("customers often leave
out the dash or write the letter in lower case"), or one line of Python
in the tool (normalize the number before looking it up), then reading
the replies again. This is the everyday work of living with an AI
function: read what it did, and fix the words or the code.

## The log: every answer, on the record

Every call so far is a line in the log folder. `functai.calls()` reads
them all, or one function's (with a column per input and output):

```python
functai.calls(folder=log_folder).count(col.program, col.model)
```

```output
# dpyr dataframe · groups: program · source: polars · showing 1 of 1 rows
┌──────────┬────────────┬─────┐
│ program  ┆ model      ┆ n   │
│ ---      ┆ ---        ┆ --- │
│ str      ┆ str        ┆ i64 │
╞══════════╪════════════╪═════╡
│ where_is ┆ gpt-6-luna ┆ 9   │
└──────────┴────────────┴─────┘
```

```python
functai.calls(where_is, folder=log_folder).select(col.message, col.seconds, col.input_tokens, col.total_tokens)
```

```output
# dpyr dataframe · source: polars · showing 9 of 9 rows
┌────────────────────────────────────────────────────────────────────────────┬──────────┬──────────────┬──────────────┐
│ message                                                                    ┆ seconds  ┆ input_tokens ┆ total_tokens │
│ ---                                                                        ┆ ---      ┆ ---          ┆ ---          │
│ str                                                                        ┆ f64      ┆ i64          ┆ i64          │
╞════════════════════════════════════════════════════════════════════════════╪══════════╪══════════════╪══════════════╡
│ Hi, my order A-1042 still hasn't arrived and it's been three weeks.        ┆ 4.045424 ┆ 340          ┆ 398          │
│ Tracking for C-3319 hasn't moved since Monday.                             ┆ 3.189276 ┆ 332          ┆ 409          │
│ Order c3319 was delivered to my neighbour's address instead of mine.       ┆ 1.737133 ┆ 146          ┆ 227          │
│ My parcel says delivered but it's not here. Order A-1299.                  ┆ 3.290793 ┆ 341          ┆ 396          │
│ When will order B-2417 ship? It says 'processing' for a week.              ┆ 3.302832 ┆ 340          ┆ 389          │
│ Order #A-1350 is two weeks late. Any news?                                 ┆ 2.893913 ┆ 335          ┆ 406          │
│ Can I change the delivery address for order C-3480? It hasn't shipped yet. ┆ 3.081853 ┆ 343          ┆ 449          │
│ Where is my order Z-9999?                                                  ┆ 2.892411 ┆ 319          ┆ 393          │
│ Hi, where's my parcel? It's been ages.                                     ┆ 1.784169 ┆ 142          ┆ 190          │
└────────────────────────────────────────────────────────────────────────────┴──────────┴──────────────┴──────────────┘
```

Each call knows its function's name, its **version**, the model, the
time it took, its tokens and, since the log keeps content by default,
its inputs and outputs (`@ai(log_content=False)` keeps only their sizes,
for private data). That's the raw material for everything below.

## People correct it; corrections become data

Here's the loop that keeps a function honest after it ships. Someone
reviews a sample of real answers, says which were right, and corrects
the wrong ones. Let's play the reviewer, using the right answers we
happen to have:

```python
@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

sample = tickets.slice_sample(n=30, seed=8)
sample.mutate(guess=team(col.message)).count(col.guess)       # 30 real calls, logged as use
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌──────────┬─────┐
│ guess    ┆ n   │
│ ---      ┆ --- │
│ str      ┆ i64 │
╞══════════╪═════╡
│ account  ┆ 8   │
│ billing  ┆ 6   │
│ product  ┆ 8   │
│ shipping ┆ 8   │
└──────────┴─────┘
```

Each logged call can be rated by passing its row (or its id, or what
`fn.predict(...)` returned) to `functai.rate()`. A wrong one carries
the right answer:

```python
truth = dict(zip(tickets.pull(col.message), tickets.pull(col.category)))

for call in functai.calls(team, folder=log_folder).collect().to_dicts():
    right = truth[call["message"]]
    if call["pred_result"] == right:
        functai.rate(call, "right")
    else:
        functai.rate(call, "wrong", answer=right, note="the shop's house rules")
```

`functai.rated()` turns the reviews back into rows with known answers,
typed like the function's inputs and output:

```python
reviewed = functai.rated(team, folder=log_folder)
reviewed.select(col.message, col.result, col.rating)
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌────────────────────────────────────────────────────────────────────────────┬──────────┬────────┐
│ message                                                                    ┆ result   ┆ rating │
│ ---                                                                        ┆ ---      ┆ ---    │
│ str                                                                        ┆ str      ┆ str    │
╞════════════════════════════════════════════════════════════════════════════╪══════════╪════════╡
│ Can I change the password without the old one?                             ┆ account  ┆ right  │
│ The kettle lid doesn't close properly anymore after a month of use.        ┆ product  ┆ right  │
│ Please close my account, I'm moving abroad.                                ┆ account  ┆ right  │
│ The glass carafe was shattered when I opened the package.                  ┆ shipping ┆ right  │
│ I changed my email and now I can't log in with either.                     ┆ account  ┆ right  │
│ The frying pan arrived with a big dent in it.                              ┆ shipping ┆ right  │
│ Order #A-1350 is two weeks late. Any news?                                 ┆ shipping ┆ right  │
│ Can I change the delivery address for order C-3480? It hasn't shipped yet. ┆ shipping ┆ right  │
│ I'd like my money back for the toaster, it burns everything.               ┆ billing  ┆ right  │
│ The teapot spout was chipped when it arrived. Order b2610.                 ┆ shipping ┆ right  │
└────────────────────────────────────────────────────────────────────────────┴──────────┴────────┘
```

Those rows are an evaluation set that grows by itself as people review.
Any new version of the function is measured on it:

```python
@ai
def team_rules(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message? House rules: anything wrong with the delivery
    itself (late, lost, wrong item, missing, broken on arrival) is shipping; anything about money,
    including every request for money back, is billing; problems in use and product questions are
    product; sign-in, passwords, profile and personal data are account."""
    ...

functai.compare(functai.evaluate(team, reviewed, num_threads=8),
                functai.evaluate(team_rules, reviewed, num_threads=8))
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌─────────────┬──────────┬───────┬──────────┬───────────┬────────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before   ┆ after ┆ diff     ┆ low       ┆ high   ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---      ┆ ---   ┆ ---      ┆ ---       ┆ ---    ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64      ┆ f64   ┆ f64      ┆ f64       ┆ f64    ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪══════════╪═══════╪══════════╪═══════════╪════════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.966667 ┆ 1.0   ┆ 0.033333 ┆ -0.034833 ┆ 0.1015 ┆ 1      ┆ 0     ┆ 29   ┆ 30  │
└─────────────┴──────────┴───────┴──────────┴───────────┴────────┴────────┴───────┴──────┴─────┘
```

They can also teach it: `team.opt(reviewed)` gives a copy whose worked
examples are the corrections (tutorial 4). The ratings live in the
same folder as the calls, in the same format across languages: a
correction made from R or TypeScript shows up in Python's `rated()`, and
the other way round.

## Versions

Every call in the log carries the version of the function that made it:

```python
{"team": team.version, "team_rules": team_rules.version}
```

```output
{'team': 'sha256:99ae724adb8e3da04da85dddbeec5dc1feed95478e91e3c027342bba47f6013c', 'team_rules': 'sha256:8b7512187f0a31e48cb7bde813c8ae0aaa93b384f25df0c2c0bfecd5e0ed3626'}
```

```python
functai.calls(team, folder=log_folder).count(col.version, col.purpose)
```

```output
# dpyr dataframe · groups: version · source: polars · showing 2 of 2 rows
┌─────────────────────────────────────────────────────────────────────────┬────────────┬─────┐
│ version                                                                 ┆ purpose    ┆ n   │
│ ---                                                                     ┆ ---        ┆ --- │
│ str                                                                     ┆ str        ┆ i64 │
╞═════════════════════════════════════════════════════════════════════════╪════════════╪═════╡
│ sha256:99ae724adb8e3da04da85dddbeec5dc1feed95478e91e3c027342bba47f6013c ┆ evaluation ┆ 30  │
│ sha256:99ae724adb8e3da04da85dddbeec5dc1feed95478e91e3c027342bba47f6013c ┆ use        ┆ 30  │
└─────────────────────────────────────────────────────────────────────────┴────────────┴─────┘
```

A version is a fingerprint of everything the function sends besides its
inputs: the instruction, the layout, the worked examples. Change a word
of the docstring and the version changes. Change the *model* and it
doesn't: the model is a setting, so you can compare models on one
version. And the same function written in R or TypeScript has the same
version, so their calls and ratings add up. `purpose` tells real use
apart from the calls evaluations and optimizers made, so a dashboard
never mistakes a test for traffic.

## Saving it

A function is worth keeping once you've measured it. `functai.check()`
lists what it depends on (the model, the code, the packages), and says
what would stop it from being saved:

```python
functai.check(team_rules)
```

```output
team_rules  AI function (message: str → Literal['shipping', 'billing', 'product', 'account'])  [__main__]
└── Literal  (stdlib)

requirements: functai @ file:///home/maxime/Projects/functai/python

! local-install  requirements: installed from folders on this machine: functai (/home/maxime/Projects/functai/python), lmcc (/home/maxime/Projects/lmcc/python)
    fix: the saved program loads where those folders exist; publish them, or install released versions, to load it anywhere
```

(If it warns that functai or lmcc were installed from folders on this
machine, that's a development setup talking: the folder would only load
where those folders exist. Installed from PyPI, the warning goes away.)

`functai.save()` writes it as a folder:

```python
folder = os.path.join(tempfile.mkdtemp(), "team_rules")
functai.save(team_rules, folder)
sorted(os.listdir(folder))
```

```output
['code', 'functai.json', 'requirements.lock', 'requirements.txt']
```

`functai.verify()` loads the folder in a clean state and checks it still
sends exactly what it sent when it was saved; `functai.load()` gives the
function back:

```python
functai.verify(folder, trust=True, fresh=False)
```

```output
verified in this environment
```

```python
loaded = functai.load(folder, trust=True)
loaded("The courier left my package in the rain and the box is soaked.")
```

```output
'shipping'
```

Python can save more than one AI function: a program of your own code
around several of them (`@functai.module`), with their tools, baked
weights and dependencies, verified in a fresh environment. R and
TypeScript load the folders of plain AI functions Python saves, and
refuse, with the reason, what only Python can run.

## What it cost

```python
functai.calls(folder=log_folder).summarize(
    calls=n(), dollars=((col.input_tokens * 0.10 + (col.total_tokens - col.input_tokens) * 0.50) / 1e6).sum())
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
┌───────┬──────────┐
│ calls ┆ dollars  │
│ ---   ┆ ---      │
│ i64   ┆ f64      │
╞═══════╪══════════╡
│ 100   ┆ 0.002883 │
└───────┴──────────┘
```

## Your turn

1. Add a second tool, `cancel_order(order)`, that only works when the
   status is "processing", and a function that handles cancellation
   requests. What does the model do for an order already in transit?
2. Rate five of `where_is`'s replies (`functai.calls(where_is)` gives
   their rows). Which would you mark wrong, and why? What would you add
   to its docstring?
3. Run `taught = team.opt(reviewed)` and evaluate it on `reviewed`
   again. Why is that evaluation too kind, and what would you evaluate
   it on instead?

## What you learned

- A tool is a Python function the model may ask for (`@ai(tools=[...])`);
  functai runs it and sends back the result. The model decides when,
  and with what.
- The call log records every call: `functai.calls()` reads it as a
  table.
- `functai.rate()` records people's verdicts and corrections;
  `functai.rated()` turns them into an evaluation set that grows with
  use.
- `fn.version` names what the function sends; models don't change it,
  and languages share it.
- `functai.check()`, `save()`, `verify()` and `load()` keep a function,
  checking it still sends exactly what it did.

**Answers to the check at the top.** (1) Your program does: the model
asks, functai runs the Python function and sends the result back. (2) In
the log folder, next to the calls; `functai.rated(fn)` gives them back
as rows with the right answers, ready for `evaluate()` and `.opt()` (or `functai.gepa`).
(3) Everything the function sends besides its inputs (instruction,
layout, worked examples). The model, and the language it was written
in, don't change it.

## The whole series, in one page

You have now done, in Python, the whole life of an AI function:

1. **Write it**: a name, a docstring, typed inputs and a typed answer
   (`@ai`); read what the model reads (`render()`).
2. **Get typed answers**: literals, optional numbers, dataclasses,
   lists; several fields at once (tutorial 2).
3. **Measure it**: `evaluate()`, intervals, baselines, confusion
   matrices, `compare()` (tutorial 3).
4. **Improve it without fooling yourself**: rules, examples, a teacher;
   three piles of rows; paired comparisons (tutorial 4).
5. **Choose the model**: accuracy, cost and speed on your rows, and a
   rule written before the chart (tutorial 5).
6. **Decide with it**: costs of mistakes, the model reading and Python
   ruling, calibrated probabilities from Jev, the cheapest action,
   escalation (tutorial 6).
7. **Own it**: bake a small model, run the same function on it, escalate
   when unsure (tutorial 7).
8. **Live with it**: tools, the log, ratings, versions, saving (here).

Three habits carry through all of it: look at what the model reads,
measure with intervals on rows you didn't tune on, and count the cost in
dollars.
