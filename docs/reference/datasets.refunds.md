---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# datasets.refunds { #functai.datasets.refunds }

```{.python .no-run}
datasets.refunds()
```

Refund requests to the homeware shop of :func:`tickets`, with the decision its rules give.

120 rows. What the customer wrote, what the order system knows, what
state the item is in, and whether the shop's refund rules say to pay.

| column | what it holds |
|---|---|
| ``id`` | the request's number |
| ``message`` | what the customer wrote |
| ``item`` | what they bought |
| ``price`` | what they paid, in dollars |
| ``days_since_delivery`` | from the order system, not the message |
| ``final_sale`` | bought on final sale (clearance) |
| ``state`` | ``"unopened"``, ``"opened_unused"``, ``"used"`` (works, no longer wanted), ``"damaged"`` (on arrival), ``"wrong_item"`` or ``"faulty"`` (failed in normal use) |
| ``decision`` | ``"approve"`` or ``"deny"``: what the rules below give |

## Notes {.doc-section .doc-section-notes}

The refund rules, which ``decision`` follows exactly:

- Damaged on arrival, or the wrong item (or part of the order missing):
  a refund within **60 days** of delivery, final sale or not.
- Faulty (it failed in normal use): a refund within **365 days**, final
  sale or not.
- Unopened, or opened but not used, and no longer wanted: a refund
  within **30 days**, and **never for a final-sale item**.
- Used and no longer wanted: **no refund**.

The messages were written by a language model from each row's facts,
in varied tones and lengths; some say what happened only indirectly.
The facts, and so the decisions, were drawn first (``r/data-raw/refunds.R``).

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
refunds = functai.datasets.refunds()
refunds
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌─────┬───────────────────────────────────────────────────────────┬─────────────────────┬────────┬─────────────────────┬────────────┬───────────────┬──────────┐
│ id  ┆ message                                                   ┆ item                ┆ price  ┆ days_since_delivery ┆ final_sale ┆ state         ┆ decision │
│ --- ┆ ---                                                       ┆ ---                 ┆ ---    ┆ ---                 ┆ ---        ┆ ---           ┆ ---      │
│ i64 ┆ str                                                       ┆ str                 ┆ f64    ┆ i64                 ┆ bool       ┆ str           ┆ str      │
╞═════╪═══════════════════════════════════════════════════════════╪═════════════════════╪════════╪═════════════════════╪════════════╪═══════════════╪══════════╡
│ 1   ┆ Hiya, just a heads up - the wall clock arrived over 9     ┆ wall clock          ┆ 28.51  ┆ 66                  ┆ false      ┆ wrong_item    ┆ deny     │
│     ┆ weeks ago now and it's totally the w…                     ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 2   ┆ Ordered the grey rug, got sent green instead 4 weeks ago  ┆ floor rug           ┆ 39.22  ┆ 28                  ┆ false      ┆ wrong_item    ┆ approve  │
│     ┆ now, want a refund.                                       ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 3   ┆ Hiya, so I'd only made about three batches of cookies     ┆ stand mixer         ┆ 9.38   ┆ 9                   ┆ false      ┆ faulty        ┆ approve  │
│     ┆ with it when it just went quiet mid-…                     ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 4   ┆ Hi, I bought a planter from you about 8 weeks ago and it  ┆ ceramic planter     ┆ 139.47 ┆ 57                  ┆ false      ┆ faulty        ┆ approve  │
│     ┆ was fine to begin with, but now i…                        ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 5   ┆ I'm sorry to even ask this after so long, it's been about ┆ bath towels         ┆ 75.9   ┆ 368                 ┆ false      ┆ used          ┆ deny     │
│     ┆ a year since these arrived, I've…                         ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 6   ┆ I need a refund for the chair I got about two and a half  ┆ desk chair          ┆ 388.71 ┆ 17                  ┆ false      ┆ faulty        ┆ approve  │
│     ┆ weeks ago. It was fine at first b…                        ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 7   ┆ Hello, I ordered a casserole dish about four weeks ago,   ┆ cast-iron casserole ┆ 458.27 ┆ 26                  ┆ false      ┆ opened_unused ┆ approve  │
│     ┆ and after taking it out of the box…                       ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 8   ┆ It's been four weeks since this arrived and I still can't ┆ teapot              ┆ 11.63  ┆ 28                  ┆ true       ┆ wrong_item    ┆ approve  │
│     ┆ pour a cup of tea without the li…                         ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 9   ┆ I'm writing about the pair of pillows I got about 3-4     ┆ pillow pair         ┆ 128.65 ┆ 25                  ┆ false      ┆ faulty        ┆ approve  │
│     ┆ weeks ago. They were fine at first b…                     ┆                     ┆        ┆                     ┆            ┆               ┆          │
│ 10  ┆ Hi! So this ended up still boxed up untouched in my       ┆ ceramic planter     ┆ 50.05  ┆ 94                  ┆ false      ┆ unopened      ┆ deny     │
│     ┆ hallway since it arrived about three m…                   ┆                     ┆        ┆                     ┆            ┆               ┆          │
└─────┴───────────────────────────────────────────────────────────┴─────────────────────┴────────┴─────────────────────┴────────────┴───────────────┴──────────┘
```