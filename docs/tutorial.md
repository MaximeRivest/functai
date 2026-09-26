# FunctAI tutorial: from one function to a tested program


This tutorial builds a small support desk, one step at a time: read a
customer’s message, triage it, look up their order, draft a reply, then
measure how well it works and improve it. Every cell below ran to
produce this page; the outputs are real model replies.

You need `pip install "functai[data]"` and a key for a model provider
(here `OPENAI_API_KEY`), or a subscription login:
`functai.login("claude")`.

## 1. A first AI function

The function definition is the prompt. The docstring is the instruction,
the parameters are the inputs, the return type is the output.

``` python
import functai
from functai import ai, _ai

functai.configure(lm="gpt-4.1-mini", temperature=0)

@ai
def summarize(message: str) -> str:
    """Summarize the customer's message in one short sentence."""

summarize("Hi, I ordered a kettle three weeks ago (order A-1042) and it still "
          "hasn't arrived. Tracking has said 'label created' for 20 days. "
          "Can you tell me what's going on?")
```

    "Customer is inquiring about the delayed delivery of their kettle order A-1042, which has been stuck at 'label created' status for 20 days."

The body can be empty (a docstring, `...`, or `return _ai`): the model’s
answer is the return value.

## 2. Types are the contract

Return a type and you get that type back, checked. Dataclasses, `Enum`,
`Literal`, lists, `Optional` and pydantic models all work, as outputs
and as inputs. Comments on fields become guidance for the model.

``` python
from dataclasses import dataclass
from enum import Enum
from typing import Literal

class Priority(Enum):
    LOW = "low"
    NORMAL = "normal"
    URGENT = "urgent"

@dataclass
class Triage:
    category: Literal["shipping", "billing", "product", "account"]
    priority: Priority
    order_id: str | None   # as written in the message, None if there is none

@ai
def triage(message: str) -> Triage:
    """Triage a customer support message."""

ticket = triage("I was charged twice for order B-2210, please refund one of them ASAP!")
ticket
```

    Triage(category='billing', priority=<Priority.URGENT: 'urgent'>, order_id='B-2210')

``` python
ticket.priority, ticket.order_id
```

    (<Priority.URGENT: 'urgent'>, 'B-2210')

## 3. Thinking before answering

A variable assigned from `_ai` is one more output, written before the
return value. Name it `reasoning` and the model thinks first. `all=True`
returns every output, plus the tokens the call used.

``` python
@ai
def is_urgent(message: str) -> bool:
    """Does this message need a reply within the hour?"""
    reasoning: str = _ai["Which words or facts show how urgent it is."]
    return _ai

p = is_urgent("Our whole team is locked out of the account and we have a demo in 30 minutes.",
              all=True)
p.reasoning
```

    'The message states that the whole team is locked out of the account and there is a demo scheduled in 30 minutes. This indicates an urgent issue that needs immediate attention to avoid missing the demo.'

``` python
p.result, p.usage["input_tokens"], p.usage["output_tokens"]
```

    (True, 91, 56)

`functai.phistory()` shows the exact conversation that was sent:

``` python
print(functai.phistory())
```

    [2026-09-26T13:13:14] is_urgent → gpt-4.1-mini

    System message:

    Function: is_urgent

    Does this message need a reply within the hour?

    Output guidance:
    - reasoning: Which words or facts show how urgent it is.

    Reply in exactly this form:
    <reasoning>
    ...
    </reasoning>
    <result>
    (boolean)
    </result>


    User message:

    <message>
    Our whole team is locked out of the account and we have a demo in 30 minutes.
    </message>


    Response:

    <reasoning>
    The message states that the whole team is locked out of the account and there is a demo scheduled in 30 minutes. This indicates an urgent issue that needs immediate attention to avoid missing the demo.
    </reasoning>
    <result>
    true
    </result>

    (finish: stop; tokens in 91, out 56)

## 4. Tools

Tools are plain typed Python functions. The model calls them as needed,
and FunctAI runs them and hands back the results, until the model
answers.

``` python
ORDERS = {
    "A-1042": {"item": "kettle", "status": "stuck at carrier", "shipped": "2026-09-02"},
    "B-2210": {"item": "toaster", "status": "delivered", "charged": 2},
}

def lookup_order(order_id: str) -> dict:
    """The order's item, shipping status and number of charges."""
    return ORDERS.get(order_id, {"error": f"no order {order_id}"})

@ai(tools=[lookup_order])
def draft_reply(message: str) -> str:
    """Draft a short, friendly reply to the customer. Check the order first
    when the message mentions one, and say what we will do next."""

print(draft_reply("Where is my order A-1042?? It's been weeks."))
```

    Thank you for reaching out about your order A-1042. I see that your kettle is currently stuck with the carrier, which is why it has been delayed. We will contact the carrier to expedite the delivery and keep you updated on the progress. We appreciate your patience!

## 5. A program of several AI functions

A `@module` is ordinary Python that calls AI functions. It can be
evaluated, optimized and saved as one program.

``` python
from functai import module

@dataclass
class Handled:
    triage: Triage
    reply: str

@module
def handle(message: str) -> Handled:
    t = triage(message)
    reply = draft_reply(message) if t.priority is not Priority.LOW else "Thanks, we'll get back to you."
    return Handled(t, reply)

handle("My password reset email never comes, I can't log in at all.")
```

    Handled(triage=Triage(category='account', priority=<Priority.URGENT: 'urgent'>, order_id=None), reply="I'm sorry to hear you're having trouble with the password reset email. Let's get this sorted out for you. Please check your spam or junk folder just in case the email ended up there. If you still don't see it, let me know, and I can help you further with resetting your password.")

## 6. Measuring it

The category is the part of the triage that decides who handles a
message, so let’s give it its own function and find out how well it
works.

``` python
@ai
def category(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """The support category of the message."""
```

To know whether it works, run it on messages with known answers. Data is
rows: a list of dicts, or any table (a parquet or CSV file, a pandas or
polars dataframe). Columns named like the parameters are the inputs; the
others are the expected answers.

Our desk has house rules the model can’t guess: an item that arrives
broken is a **shipping** claim (the carrier pays), and anything about
getting money back is **billing**, whatever it’s about.

``` python
from dpyr import col

dev = [
    {"message": "The mug arrived in pieces.", "category": "shipping"},
    {"message": "I want my money back for the toaster, it's useless.", "category": "billing"},
    {"message": "The kettle's lid doesn't close properly.", "category": "product"},
    {"message": "How do I change the email on my profile?", "category": "account"},
    {"message": "Box was crushed and the lamp inside is cracked.", "category": "shipping"},
    {"message": "Please refund the blender, it stopped working.", "category": "billing"},
    {"message": "Toaster burns one side of the bread.", "category": "product"},
    {"message": "Please delete my account and my data.", "category": "account"},
    {"message": "Screen was shattered when I opened the package.", "category": "shipping"},
    {"message": "Return the headphones and give me a refund please.", "category": "billing"},
    {"message": "Tracking hasn't moved in a week.", "category": "shipping"},
    {"message": "My coupon code didn't apply at checkout.", "category": "billing"},
]

right = col.pred_result == col.category

ev = functai.evaluate(category, dev, {"right": right}, num_threads=8)
ev
```

    Evaluation(category, 12 examples: right 0.58 [0.32, 0.81])

The metric here is a [dpyr](https://github.com/MaximeRivest/dpyr) column
expression: the prediction (`pred_result`) equals the expected category.
A metric can also be a Python function `metric(row, prediction)`, or
another AI function acting as a judge.

The score comes with a 95% interval: with 12 examples, it is wide. The
details are a table, one row per example, ready for the usual data tools
(dpyr verbs here, or `.collect()` for a polars dataframe). The misses:

``` python
ev.table.filter(col.right == 0).select(col.message, col.category, col.pred_result)
```

    # dpyr dataframe · source: polars · showing 5 of 5 rows
    shape: (5, 3)
    ┌───────────────────────────────────────────┬──────────┬─────────────┐
    │ message                                   ┆ category ┆ pred_result │
    │ ---                                       ┆ ---      ┆ ---         │
    │ str                                       ┆ str      ┆ str         │
    ╞═══════════════════════════════════════════╪══════════╪═════════════╡
    │ The mug arrived in pieces.                ┆ shipping ┆ product     │
    │ Box was crushed and the lamp inside is c… ┆ shipping ┆ product     │
    │ Please refund the blender, it stopped wo… ┆ billing  ┆ product     │
    │ Screen was shattered when I opened the p… ┆ shipping ┆ product     │
    │ Return the headphones and give me a refu… ┆ billing  ┆ product     │
    └───────────────────────────────────────────┴──────────┴─────────────┘

``` python
ev.table.group_by(col.category).summarize(accuracy=col.right.mean())
```

    # dpyr dataframe · source: polars · showing 4 of 4 rows
    shape: (4, 2)
    ┌──────────┬──────────┐
    │ category ┆ accuracy │
    │ ---      ┆ ---      │
    │ str      ┆ f64      │
    ╞══════════╪══════════╡
    │ account  ┆ 1.0      │
    │ billing  ┆ 0.5      │
    │ product  ┆ 1.0      │
    │ shipping ┆ 0.25     │
    └──────────┴──────────┘

## 7. Improving it

An optimizer changes what the function sends besides the inputs: its
instruction and its worked examples (“demos”). It never touches your
code or types. `BootstrapFewShot` runs the function on training rows and
keeps the runs your metric accepts as demos.

``` python
train = [
    {"message": "The vase came smashed.", "category": "shipping"},
    {"message": "Frying pan arrived with a big dent.", "category": "shipping"},
    {"message": "I'd like a refund for the kettle, the lid is loose.", "category": "billing"},
    {"message": "Money back please, the chair wobbles.", "category": "billing"},
    {"message": "The handle came off after two uses.", "category": "product"},
    {"message": "I can't turn on two-factor authentication.", "category": "account"},
    {"message": "Wrong item in my box.", "category": "shipping"},
    {"message": "Charged in USD instead of CAD.", "category": "billing"},
]

before = ev
category.opt(trainset=train, metric=right)
after = functai.evaluate(category, dev, {"right": right}, num_threads=8)
functai.compare(before, after)
```

    # dpyr dataframe · source: polars · showing 1 of 1 rows
    shape: (1, 10)
    ┌────────┬──────────┬──────────┬──────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
    │ metric ┆ before   ┆ after    ┆ diff ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
    │ ---    ┆ ---      ┆ ---      ┆ ---  ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
    │ str    ┆ f64      ┆ f64      ┆ f64  ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
    ╞════════╪══════════╪══════════╪══════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
    │ right  ┆ 0.583333 ┆ 0.583333 ┆ 0.0  ┆ -0.270924 ┆ 0.270924 ┆ 1      ┆ 1     ┆ 10   ┆ 12  │
    └────────┴──────────┴──────────┴──────┴───────────┴──────────┴────────┴───────┴──────┴─────┘

`compare` pairs the examples: `better`, `worse` and `same` count
examples that changed, and `low`/`high` is the 95% interval of `diff`.
On so few examples it is wide; when it includes 0, the change could be
luck, and the honest next step is more examples, not more optimizing.
The demos the optimizer chose are in `category.demos`, and
`category.undo_opt()` goes back.

When you *can* state the rule, state it: the docstring is the
instruction.

``` python
@ai
def category_with_rules(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """The support category of the message.
    House rules: an item that arrived broken is shipping (the carrier pays),
    and any request for money back is billing."""

with_rules = functai.evaluate(category_with_rules, dev, {"right": right}, num_threads=8)
functai.compare(before, with_rules)
```

    # dpyr dataframe · source: polars · showing 1 of 1 rows
    shape: (1, 10)
    ┌────────┬──────────┬───────┬──────────┬──────────┬──────────┬────────┬───────┬──────┬─────┐
    │ metric ┆ before   ┆ after ┆ diff     ┆ low      ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
    │ ---    ┆ ---      ┆ ---   ┆ ---      ┆ ---      ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
    │ str    ┆ f64      ┆ f64   ┆ f64      ┆ f64      ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
    ╞════════╪══════════╪═══════╪══════════╪══════════╪══════════╪════════╪═══════╪══════╪═════╡
    │ right  ┆ 0.583333 ┆ 1.0   ┆ 0.416667 ┆ 0.089494 ┆ 0.743839 ┆ 5      ┆ 0     ┆ 7    ┆ 12  │
    └────────┴──────────┴───────┴──────────┴──────────┴──────────┴────────┴───────┴──────┴─────┘

Optimizers earn their keep when the rule is hard to put into words, and
you have examples of it.

## 8. Running on a table

Called with a column instead of a value, an AI function becomes a column
expression. Each distinct message goes to the model once, several at a
time, and answers are remembered for the session.

``` python
from dpyr import read, n

inbox = read([
    {"id": 1, "message": "Where's my parcel? Order A-1042."},
    {"id": 2, "message": "I was charged twice!"},
    {"id": 3, "message": "The toaster smells like plastic."},
    {"id": 4, "message": "Where's my parcel? Order A-1042."},
])

(inbox
    .mutate(category=category(col.message), urgent=is_urgent(col.message))
    .group_by(col.category)
    .summarize(n=n(), urgent=col.urgent.sum()))
```

    # dpyr dataframe · source: polars · showing 3 of 3 rows
    shape: (3, 3)
    ┌──────────┬─────┬────────┐
    │ category ┆ n   ┆ urgent │
    │ ---      ┆ --- ┆ ---    │
    │ str      ┆ i64 ┆ i64    │
    ╞══════════╪═════╪════════╡
    │ billing  ┆ 1   ┆ 1      │
    │ product  ┆ 1   ┆ 1      │
    │ shipping ┆ 2   ┆ 0      │
    └──────────┴─────┴────────┘

## 9. Shipping it

`functai.check` lists everything the program depends on: its AI
functions, tools, types, constants and packages, and anything that would
stop it from being saved.

``` python
functai.check(handle)
```

    handle  @module  [__main__]
    ├── triage  AI function (message: str → Triage)  [__main__]
    │   └── Triage  class  [__main__]
    │       ├── Priority  class  [__main__]
    │       │   └── Enum  (stdlib)
    │       ├── Literal  (stdlib)
    │       └── dataclass  (stdlib)
    ├── Priority  (see above)
    ├── draft_reply  AI function (message: str → str)  [__main__]
    │   └── tool lookup_order  function  [__main__]
    │       └── ORDERS = {'A-1042': {'item': 'kettle', 'status...
    └── Handled  class  [__main__]
        ├── Triage  (see above)
        └── dataclass  (stdlib)

    requirements: functai==1.0.1
    no problems: ready to save

`functai.save(handle, "support_desk/")` writes it all to a folder you
can commit, `functai.verify("support_desk/", trust=True)` proves it
works in a fresh environment, and `functai.load(...)` brings it back.

## Where next

- The [README](../README.md) is the reference: every setting, layouts
  and chat templates, logins, evaluation and optimization options.
- The [examples](../examples/) go deeper on one topic each.
