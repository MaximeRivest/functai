---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Turn notes into data

*Field notes, reports, emails: pull the facts out as columns, following your protocol, and check every field.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

Notes written by people hold the facts you need, in words: "a pair of
robins pulling worms", "about 40 geese heading north". A language model
can read them the way a trained volunteer would, and write each fact into
its own column. Then the notes are data: you can count, filter and plot
them.

What makes this work is not the model. It's **your protocol**: the rules
a careful person in your field follows when turning a note into a
record. This page shows how to write that protocol down, and how to
check, field by field, that it is being followed.

## The notes

`field_notes` holds 60 notes from a bird survey, with what an expert
recorded for each one: the species, how many birds, and what they were
doing.

```python
from dpyr import col, n, desc

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

## Say what a record looks like

A record is a small class. Each field has a type, and the types are the
first rules: a species must come from the checklist, a behaviour from the
survey's five, and a count may be missing (`int | None`). A comment on a
field tells the model what goes there.

```python
from dataclasses import dataclass
from typing import Literal

Species = Literal["American robin", "black-capped chickadee", "blue jay", "northern cardinal",
                  "mallard", "Canada goose", "great blue heron", "red-tailed hawk",
                  "downy woodpecker", "song sparrow", "American crow", "barn swallow", "other"]

@dataclass
class Sighting:
    species: Species
    count: int | None     # how many birds
    behaviour: Literal["feeding", "nesting", "flying", "resting", "calling"]

@ai
def sighting(note: str) -> Sighting:
    """The bird observation in this survey note."""
```

```python
sighting("Mallard hen with 9 ducklings swimming along the edge.")
```

```output
Sighting(species='mallard', count=10, behaviour='resting')
```

What the model saw, and what it answered:

```python
print(functai.phistory())
```

```output
[2026-09-26T17:38:27] sighting → gpt-4.1-mini

System message:

Function: sighting

The bird observation in this survey note.

Sighting fields:
- Sighting.count: how many birds

Reply in exactly this form:
<result>
JSON matching this schema: {"type": "object", "properties": {"species": {"enum": ["American robin", "black-capped chickadee", "blue jay", "northern cardinal", "mallard", "Canada goose", "great blue heron", "red-tailed hawk", "downy woodpecker", "song sparrow", "American crow", "barn swallow", "other"], "type": "string"}, "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "behaviour": {"enum": ["feeding", "nesting", "flying", "resting", "calling"], "type": "string"}}, "required": ["species", "count", "behaviour"]}
</result>


User message:

<note>
Mallard hen with 9 ducklings swimming along the edge.
</note>


Response:

<result>
{"species": "mallard", "count": 10, "behaviour": "resting"}
</result>

(finish: stop; tokens in 224, out 28)
```

## Every note, as columns

`unpack` turns each field into its own column (one model call per note):

```python
records = notes.select(col.site, col.note).mutate(**sighting.unpack(col.note))
records
```

```output
# dpyr dataframe · source: polars · showing 10 of 60 rows
┌─────────────────┬────────────────────────────────────────────────────────────────────────────────┬────────────────────────┬───────┬───────────┐
│ site            ┆ note                                                                           ┆ species                ┆ count ┆ behaviour │
│ ---             ┆ ---                                                                            ┆ ---                    ┆ ---   ┆ ---       │
│ str             ┆ str                                                                            ┆ str                    ┆ i64   ┆ str       │
╞═════════════════╪════════════════════════════════════════════════════════════════════════════════╪════════════════════════╪═══════╪═══════════╡
│ Marsh boardwalk ┆ Great blue heron standing in the shallows, stabbing at fish. Caught one.       ┆ great blue heron       ┆ null  ┆ feeding   │
│ North field     ┆ A pair of robins pulling worms on the lawn.                                    ┆ American robin         ┆ 2     ┆ feeding   │
│ Creek trail     ┆ Heard a chickadee calling 'chick-a-dee-dee' from the pines, didn't see it.     ┆ black-capped chickadee ┆ null  ┆ calling   │
│ Old orchard     ┆ Downy woodpecker drumming on a dead branch.                                    ┆ downy woodpecker       ┆ null  ┆ calling   │
│ Marsh boardwalk ┆ About 40 Canada geese flying over in a V, heading north.                       ┆ Canada goose           ┆ 40    ┆ flying    │
│ North field     ┆ Red-tailed hawk perched on the fence post, just sitting there for ten minutes. ┆ red-tailed hawk        ┆ null  ┆ resting   │
│ Creek trail     ┆ 3 blue jays squabbling at the feeder over peanuts.                             ┆ blue jay               ┆ 3     ┆ feeding   │
│ Old orchard     ┆ Male cardinal singing from the top of the apple tree.                          ┆ northern cardinal      ┆ 1     ┆ calling   │
│ Marsh boardwalk ┆ Mallards, a few of them, dabbling near the reeds.                              ┆ mallard                ┆ null  ┆ feeding   │
│ North field     ┆ Barn swallows swooping low over the grass, maybe 6.                            ┆ barn swallow           ┆ 6     ┆ flying    │
└─────────────────┴────────────────────────────────────────────────────────────────────────────────┴────────────────────────┴───────┴───────────┘
```

And now it's data. Birds per species and site:

```python
(records
    .group_by(col.species)
    .summarize(notes=n(), birds=col.count.sum())
    .arrange(desc(col.birds)))
```

```output
# dpyr dataframe · source: polars · showing 10 of 13 rows
┌────────────────────────┬───────┬───────┐
│ species                ┆ notes ┆ birds │
│ ---                    ┆ ---   ┆ ---   │
│ str                    ┆ i64   ┆ i64   │
╞════════════════════════╪═══════╪═══════╡
│ Canada goose           ┆ 5     ┆ 60    │
│ barn swallow           ┆ 4     ┆ 47    │
│ mallard                ┆ 4     ┆ 12    │
│ American robin         ┆ 5     ┆ 10    │
│ blue jay               ┆ 5     ┆ 10    │
│ northern cardinal      ┆ 4     ┆ 6     │
│ black-capped chickadee ┆ 5     ┆ 4     │
│ great blue heron       ┆ 5     ┆ 4     │
│ American crow          ┆ 4     ┆ 3     │
│ downy woodpecker       ┆ 5     ┆ 2     │
└────────────────────────┴───────┴───────┘
```

## Check every field

Before trusting those totals, check them against the expert's records.
The notes table has columns named like the record's fields (`species`,
`count`, `behaviour`), so `evaluate` compares each one:

```python
before = functai.evaluate(sighting, notes, num_threads=8)
before.summary
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌─────────────────┬──────────┬──────────┬──────────┬─────┬────────┐
│ metric          ┆ mean     ┆ low      ┆ high     ┆ n   ┆ failed │
│ ---             ┆ ---      ┆ ---      ┆ ---      ┆ --- ┆ ---    │
│ str             ┆ f64      ┆ f64      ┆ f64      ┆ i64 ┆ i64    │
╞═════════════════╪══════════╪══════════╪══════════╪═════╪════════╡
│ exact_match     ┆ 0.6      ┆ 0.473661 ┆ 0.714305 ┆ 60  ┆ 0      │
│ species_match   ┆ 1.0      ┆ 0.939828 ┆ 1.0      ┆ 60  ┆ 0      │
│ count_match     ┆ 0.616667 ┆ 0.490176 ┆ 0.729117 ┆ 60  ┆ 0      │
│ behaviour_match ┆ 0.933333 ┆ 0.840746 ┆ 0.973771 ┆ 60  ┆ 0      │
└─────────────────┴──────────┴──────────┴──────────┴─────┴────────┘
```

`exact_match` is the share of notes where all three fields were right;
the next rows score each field alone. The species are nearly all right.
The counts and behaviours are where it disagrees with the expert. Look:

```python
before.table.filter(col.count_match == 0).select(col.note, col.count, col.pred_count)
```

```output
# dpyr dataframe · source: polars · showing 10 of ? rows
┌──────────────────────────────────────────────────────────────────────────────┬───────┬────────────┐
│ note                                                                         ┆ count ┆ pred_count │
│ ---                                                                          ┆ ---   ┆ ---        │
│ str                                                                          ┆ i64   ┆ i64        │
╞══════════════════════════════════════════════════════════════════════════════╪═══════╪════════════╡
│ Heard a chickadee calling 'chick-a-dee-dee' from the pines, didn't see it.   ┆ 1     ┆ null       │
│ Downy woodpecker drumming on a dead branch.                                  ┆ 1     ┆ null       │
│ Song sparrow singing on a shrub by the bridge.                               ┆ 1     ┆ null       │
│ Turkey vulture circling high over the marsh.                                 ┆ 1     ┆ null       │
│ Robin carrying mud and grass into the hedge. Nest in progress!               ┆ 1     ┆ null       │
│ Heron flew across the pond and landed out of sight.                          ┆ 1     ┆ null       │
│ Canada goose sitting on eggs on the island, mate standing guard next to her. ┆ 2     ┆ 1          │
│ Blue jay screaming its alarm call, probably at a cat.                        ┆ 1     ┆ null       │
│ hawk (red tail seen clearly) soaring in circles                              ┆ 1     ┆ null       │
│ Chickadee pecking at birch catkins.                                          ┆ 1     ┆ null       │
└──────────────────────────────────────────────────────────────────────────────┴───────┴────────────┘
```

```python
before.table.filter(col.behaviour_match == 0).select(col.note, col.behaviour, col.pred_behaviour)
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌──────────────────────────────────────────────────────────────────────┬───────────┬────────────────┐
│ note                                                                 ┆ behaviour ┆ pred_behaviour │
│ ---                                                                  ┆ ---       ┆ ---            │
│ str                                                                  ┆ str       ┆ str            │
╞══════════════════════════════════════════════════════════════════════╪═══════════╪════════════════╡
│ Mallard hen with 9 ducklings swimming along the edge.                ┆ resting   ┆ feeding        │
│ Osprey hovering then diving into the pond!                           ┆ feeding   ┆ flying         │
│ Song sparrow carrying a caterpillar into the thicket (nest nearby?). ┆ nesting   ┆ feeding        │
│ Robin feeding worms to 3 chicks in the nest by the bridge.           ┆ nesting   ┆ feeding        │
└──────────────────────────────────────────────────────────────────────┴───────────┴────────────────┘
```

These aren't mistakes of reading. They're choices the survey made that
the model doesn't know: that one bird mentioned is a count of 1 even with
no number written, that a bird only *heard* still counts, that "a few"
must stay blank rather than be guessed, that a bird bringing food to its
chicks is *nesting*, not *feeding*, and that swimming is *resting*.

## Write the protocol down

The docstring is the instruction: write the protocol there, the way you
would brief a new volunteer.

```python
@ai
def sighting(note: str) -> Sighting:
    """The bird observation in this survey note, recorded by the survey's protocol.

    Species: the checklist's name. Nicknames count ("robin", "heron",
    "red-tail", "downy"); a species not on the checklist is "other".

    Count: every bird seen or heard, young included. One bird named without
    a number ("a blue jay", "Downy woodpecker drumming") is 1. "A pair" or
    "a couple" is 2. An approximate number ("about 40", "~25", "maybe 6") is
    that number. When the note gives no number ("a few", "several", "a flock",
    "lots"), the count is None: never guess.

    Behaviour: singing, calling and drumming are calling. Building, sitting
    on a nest, or bringing food to young is nesting. Perched, swimming,
    roosting or standing still is resting.
    """

after = functai.evaluate(sighting, notes, num_threads=8)
functai.compare(before, after)
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌─────────────────┬──────────┬──────────┬──────────┬───────────┬──────────┬────────┬───────┬──────┬─────┐
│ metric          ┆ before   ┆ after    ┆ diff     ┆ low       ┆ high     ┆ better ┆ worse ┆ same ┆ n   │
│ ---             ┆ ---      ┆ ---      ┆ ---      ┆ ---       ┆ ---      ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str             ┆ f64      ┆ f64      ┆ f64      ┆ f64       ┆ f64      ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════════╪══════════╪══════════╪══════════╪═══════════╪══════════╪════════╪═══════╪══════╪═════╡
│ exact_match     ┆ 0.6      ┆ 0.916667 ┆ 0.316667 ┆ 0.17807   ┆ 0.455263 ┆ 21     ┆ 2     ┆ 37   ┆ 60  │
│ species_match   ┆ 1.0      ┆ 1.0      ┆ 0.0      ┆ 0.0       ┆ 0.0      ┆ 0      ┆ 0     ┆ 60   ┆ 60  │
│ count_match     ┆ 0.616667 ┆ 0.983333 ┆ 0.366667 ┆ 0.24113   ┆ 0.492203 ┆ 22     ┆ 0     ┆ 38   ┆ 60  │
│ behaviour_match ┆ 0.933333 ┆ 0.933333 ┆ 0.0      ┆ -0.067262 ┆ 0.067262 ┆ 2      ┆ 2     ┆ 56   ┆ 60  │
└─────────────────┴──────────┴──────────┴──────────┴───────────┴──────────┴────────┴───────┴──────┴─────┘
```

One row per field: which improved, by how much, and whether the change is
bigger than luck (`low` to `high`). What still disagrees:

```python
after.table.filter(col.exact_match == 0).select(col.note, col.species, col.pred_species,
                                                 col.count, col.pred_count,
                                                 col.behaviour, col.pred_behaviour)
```

```output
# dpyr dataframe · source: polars · showing 5 of 5 rows
┌──────────────────────────────────────────────────────────────────────┬────────────────┬────────────────┬───────┬────────────┬───────────┬────────────────┐
│ note                                                                 ┆ species        ┆ pred_species   ┆ count ┆ pred_count ┆ behaviour ┆ pred_behaviour │
│ ---                                                                  ┆ ---            ┆ ---            ┆ ---   ┆ ---        ┆ ---       ┆ ---            │
│ str                                                                  ┆ str            ┆ str            ┆ i64   ┆ i64        ┆ str       ┆ str            │
╞══════════════════════════════════════════════════════════════════════╪════════════════╪════════════════╪═══════╪════════════╪═══════════╪════════════════╡
│ A dozen barn swallows skimming the creek for insects.                ┆ barn swallow   ┆ barn swallow   ┆ 12    ┆ 12         ┆ feeding   ┆ flying         │
│ Osprey hovering then diving into the pond!                           ┆ other          ┆ other          ┆ 1     ┆ 1          ┆ feeding   ┆ flying         │
│ Several Canada geese flying low over the field, honking.             ┆ Canada goose   ┆ Canada goose   ┆ null  ┆ null       ┆ flying    ┆ calling        │
│ Song sparrow carrying a caterpillar into the thicket (nest nearby?). ┆ song sparrow   ┆ song sparrow   ┆ 1     ┆ 1          ┆ nesting   ┆ feeding        │
│ Robin feeding worms to 3 chicks in the nest by the bridge.           ┆ American robin ┆ American robin ┆ 4     ┆ 3          ┆ nesting   ┆ nesting        │
└──────────────────────────────────────────────────────────────────────┴────────────────┴────────────────┴───────┴────────────┴───────────┴────────────────┘
```

Some of what's left may be the protocol being unclear rather than the
model being wrong. That is worth knowing too: if two careful people could
read a note two ways, so can a model. Tighten the wording, or accept
that note as ambiguous.

## Missing is not zero

The count is `int | None`, and the protocol says *never guess*. A note
like "a few mallards" gets no count, not a made-up 3:

```python
records = notes.select(col.note).mutate(**sighting.unpack(col.note))
records.filter(col.count.is_na()).select(col.note, col.count)
```

```output
# dpyr dataframe · source: polars · showing 7 of 7 rows
┌────────────────────────────────────────────────────────────────────┬───────┐
│ note                                                               ┆ count │
│ ---                                                                ┆ ---   │
│ str                                                                ┆ i64   │
╞════════════════════════════════════════════════════════════════════╪═══════╡
│ Mallards, a few of them, dabbling near the reeds.                  ┆ null  │
│ Crows, a whole noisy flock, going to roost in the oaks.            ┆ null  │
│ Red-winged blackbirds everywhere on the cattails, singing.         ┆ null  │
│ Several song sparrows hopping in the brush pile, picking at seeds. ┆ null  │
│ Lots of crows mobbing a hawk and cawing like crazy.                ┆ null  │
│ Several Canada geese flying low over the field, honking.           ┆ null  │
│ Chickadees, a few, flying from the hedge into the woods.           ┆ null  │
└────────────────────────────────────────────────────────────────────┴───────┘
```

In an analysis that difference matters: a blank count is left out of a
sum, where a guess would quietly change it.

## Your own notes

- **Your records**: any dataclass or pydantic model. Fields can be
  numbers, dates, lists (`list[str]` for several species), other records.
  See [Types](types.md).
- **Your notes**: `read("notes.csv")`, a spreadsheet, a pandas data
  frame; see [Big tables](tables.md). Longer documents work the same way.
- **Your answers**: label 30 to 50 notes yourself, carefully, in columns
  named like the fields (or point `expected={"species": "sp_code"}` at
  yours). That small table is what tells you whether to trust the other
  thousand.
- **Hard cases**: when you can show the rule but not say it, give
  examples: `sighting.opt(trainset=labelled_notes)`. See [Make it better](improving.md).
