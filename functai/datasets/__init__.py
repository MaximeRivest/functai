"""Small labelled datasets to learn functai with.

Each one is a table of text with the right answers already filled in, so
every example in the documentation runs as is, and "how accurate is it?"
has an answer from the first minute. They were written for functai's
documentation; the answers follow the rules given in each docstring, and
those rules are exactly what a model can't guess on its own.

Needs ``pip install "functai[data]"`` (the tables are dpyr dataframes:
``.to_pandas()`` and ``.to_polars()`` convert them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

__all__ = ["field_notes", "tickets"]


def _read(name: str) -> Any:
    try:
        import dpyr
    except ImportError:
        raise ImportError(f'functai.datasets.{name}() is a table: pip install "functai[data]"') from None
    return dpyr.read(str(HERE / f"{name}.csv"))


def tickets() -> Any:
    """Customer support messages to a small homeware shop, with the team each belongs to.

    80 rows. Each message goes to one of four teams; the shop has house rules
    about which one, and about how order numbers are written.

    | column | what it holds |
    |---|---|
    | ``id`` | the message's number |
    | ``message`` | what the customer wrote |
    | ``channel`` | ``"email"`` or ``"chat"`` |
    | ``category`` | the team: ``"shipping"``, ``"billing"``, ``"product"`` or ``"account"`` |
    | ``order_id`` | the order number, like ``"A-1042"`` (a letter, a dash, four digits); missing when there is none |

    Notes
    -----
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

    Examples
    --------
    ```python
    tickets = functai.datasets.tickets()
    tickets
    ```
    """
    return _read("tickets")


def field_notes() -> Any:
    """Bird survey notes written by volunteers, with the species, count and behaviour of each.

    60 rows, one bird species per note, from four sites between April and May
    2026. The labels follow the survey's counting protocol below.

    | column | what it holds |
    |---|---|
    | ``id`` | the note's number |
    | ``site`` | ``"Marsh boardwalk"``, ``"North field"``, ``"Creek trail"`` or ``"Old orchard"`` |
    | ``date`` | ``YYYY-MM-DD`` |
    | ``note`` | what the volunteer wrote |
    | ``species`` | the checklist's common name (American robin, black-capped chickadee, blue jay, northern cardinal, mallard, Canada goose, great blue heron, red-tailed hawk, downy woodpecker, song sparrow, American crow, barn swallow), or ``"other"`` |
    | ``count`` | how many birds; missing when the note gives no number |
    | ``behaviour`` | ``"feeding"``, ``"nesting"``, ``"flying"``, ``"resting"`` or ``"calling"`` |

    Notes
    -----
    The protocol, which the labels follow:

    - **Species**: the checklist's name. Nicknames count ("robin", "heron",
      "red-tail", "downy"); a species not on the checklist is ``"other"``.
    - **Count**: every bird seen or heard, young included. One bird named
      without a number ("a blue jay") is 1; "a pair" or "a couple" is 2; an approximate number ("about 40", "~25", "maybe 6") is
      that number; no number ("a few", "several", "a flock", "lots") is
      missing: **never guess**.
    - **Behaviour**: singing, calling and drumming are *calling*; building,
      sitting on a nest or bringing food to young are *nesting*; perched,
      swimming, roosting or standing still are *resting*.

    Examples
    --------
    ```python
    notes = functai.datasets.field_notes()
    notes
    ```
    """
    return _read("field_notes")
