---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# datasets.tickets { #functai.datasets.tickets }

```{.python .no-run}
datasets.tickets()
```

Customer support messages to a small homeware shop, with the team each belongs to.

80 rows. Each message goes to one of four teams; the shop has house rules
about which one, and about how order numbers are written.

| column | what it holds |
|---|---|
| ``id`` | the message's number |
| ``message`` | what the customer wrote |
| ``channel`` | ``"email"`` or ``"chat"`` |
| ``category`` | the team: ``"shipping"``, ``"billing"``, ``"product"`` or ``"account"`` |
| ``order_id`` | the order number, like ``"A-1042"`` (a letter, a dash, four digits); missing when there is none |

## Notes {.doc-section .doc-section-notes}

The house rules, which the labels follow:

- Anything wrong with the delivery itself (late, lost, sent to the wrong
  place, the wrong item, something missing, or **broken when it
  arrived**) is **shipping**: the carrier pays.
- Anything about money (charges, invoices, coupons, cards, and **every
  request for money back**, whatever the reason) is **billing**.
- Problems that appear while using a product, and questions about
  products, are **product**.
- Signing in, passwords, profile details, personal data and emails from
  the shop are **account**.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
tickets = functai.datasets.tickets()
tickets
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌─────┬─────────────────────────────────────────────────────────────────────┬─────────┬──────────┬──────────┐
│ id  ┆ message                                                             ┆ channel ┆ category ┆ order_id │
│ --- ┆ ---                                                                 ┆ ---     ┆ ---      ┆ ---      │
│ i64 ┆ str                                                                 ┆ str     ┆ str      ┆ str      │
╞═════╪═════════════════════════════════════════════════════════════════════╪═════════╪══════════╪══════════╡
│ 1   ┆ Hi, my order A-1042 still hasn't arrived and it's been three weeks. ┆ email   ┆ shipping ┆ A-1042   │
│ 2   ┆ The mug arrived in pieces.                                          ┆ chat    ┆ shipping ┆ null     │
│ 3   ┆ I was charged twice for order B-2210, please fix this.              ┆ email   ┆ billing  ┆ B-2210   │
│ 4   ┆ How do I change the email on my account?                            ┆ chat    ┆ account  ┆ null     │
│ 5   ┆ The kettle lid doesn't close properly anymore after a month of use. ┆ email   ┆ product  ┆ null     │
│ 6   ┆ I'd like my money back for the toaster, it burns everything.        ┆ email   ┆ billing  ┆ null     │
│ 7   ┆ Tracking for C-3319 hasn't moved since Monday.                      ┆ chat    ┆ shipping ┆ C-3319   │
│ 8   ┆ I forgot my password and the reset email never comes.               ┆ chat    ┆ account  ┆ null     │
│ 9   ┆ Box was crushed and the lamp inside is cracked. Order D-4001.       ┆ email   ┆ shipping ┆ D-4001   │
│ 10  ┆ My coupon code SPRING10 didn't apply at checkout.                   ┆ chat    ┆ billing  ┆ null     │
└─────┴─────────────────────────────────────────────────────────────────────┴─────────┴──────────┴──────────┘
```