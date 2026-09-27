---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# datasets.field_notes { #functai.datasets.field_notes }

```{.python .no-run}
datasets.field_notes()
```

Bird survey notes written by volunteers, with the species, count and behaviour of each.

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

## Notes {.doc-section .doc-section-notes}

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

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
notes = functai.datasets.field_notes()
notes
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌─────┬─────────────────┬────────────┬────────────────────────────────────────────────────────────────────────────┬────────────────────────┬───────┬───────────┐
│ id  ┆ site            ┆ date       ┆ note                                                                       ┆ species                ┆ count ┆ behaviour │
│ --- ┆ ---             ┆ ---        ┆ ---                                                                        ┆ ---                    ┆ ---   ┆ ---       │
│ i64 ┆ str             ┆ str        ┆ str                                                                        ┆ str                    ┆ i64   ┆ str       │
╞═════╪═════════════════╪════════════╪════════════════════════════════════════════════════════════════════════════╪════════════════════════╪═══════╪═══════════╡
│ 1   ┆ Marsh boardwalk ┆ 2026-04-12 ┆ Great blue heron standing in the shallows, stabbing at fish. Caught one.   ┆ great blue heron       ┆ 1     ┆ feeding   │
│ 2   ┆ North field     ┆ 2026-04-12 ┆ A pair of robins pulling worms on the lawn.                                ┆ American robin         ┆ 2     ┆ feeding   │
│ 3   ┆ Creek trail     ┆ 2026-04-12 ┆ Heard a chickadee calling 'chick-a-dee-dee' from the pines, didn't see it. ┆ black-capped chickadee ┆ 1     ┆ calling   │
│ 4   ┆ Old orchard     ┆ 2026-04-13 ┆ Downy woodpecker drumming on a dead branch.                                ┆ downy woodpecker       ┆ 1     ┆ calling   │
│ 5   ┆ Marsh boardwalk ┆ 2026-04-13 ┆ About 40 Canada geese flying over in a V, heading north.                   ┆ Canada goose           ┆ 40    ┆ flying    │
│ 6   ┆ North field     ┆ 2026-04-13 ┆ Red-tailed hawk perched on the fence post, just sitting there for ten      ┆ red-tailed hawk        ┆ 1     ┆ resting   │
│     ┆                 ┆            ┆ minutes.                                                                   ┆                        ┆       ┆           │
│ 7   ┆ Creek trail     ┆ 2026-04-14 ┆ 3 blue jays squabbling at the feeder over peanuts.                         ┆ blue jay               ┆ 3     ┆ feeding   │
│ 8   ┆ Old orchard     ┆ 2026-04-14 ┆ Male cardinal singing from the top of the apple tree.                      ┆ northern cardinal      ┆ 1     ┆ calling   │
│ 9   ┆ Marsh boardwalk ┆ 2026-04-15 ┆ Mallards, a few of them, dabbling near the reeds.                          ┆ mallard                ┆ null  ┆ feeding   │
│ 10  ┆ North field     ┆ 2026-04-15 ┆ Barn swallows swooping low over the grass, maybe 6.                        ┆ barn swallow           ┆ 6     ┆ flying    │
└─────┴─────────────────┴────────────┴────────────────────────────────────────────────────────────────────────────┴────────────────────────┴───────┴───────────┘
```