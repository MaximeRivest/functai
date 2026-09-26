---
rat:
  python:
    dependencies: ["-e .[data]"]
---

# Programs of several AI functions: a multi-hop fact checker


A `@module` is a plain Python function that calls AI functions: loops,
ifs and ordinary helpers included. It is called, evaluated, optimized
and saved as one program. This one checks a claim by searching a small
library in hops, taking notes, then deciding.

Every output below is a real reply. This page is a notebook: open it in
Chattering and run it, or run it all with `python tools/docs.py run examples/modules/README.md`.

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)

from functai import ai, module
```

## A search engine (plain Python)

```python
LIBRARY = [
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "Gustave Eiffel's company designed and built the Eiffel Tower.",
    "The Statue of Liberty's internal frame was designed by Gustave Eiffel.",
    "The Statue of Liberty was a gift from France to the United States in 1886.",
    "Mount Everest, at 8,849 m, is the highest mountain above sea level.",
    "K2 is the second-highest mountain on Earth, at 8,611 m.",
    "The Great Wall of China is not visible to the naked eye from orbit.",
    "Marie Curie won Nobel Prizes in both physics (1903) and chemistry (1911).",
]

def search(query: str, k: int = 2) -> list[str]:
    """The k passages sharing the most words with the query."""
    words = set(query.lower().split())
    return sorted(LIBRARY, key=lambda p: -len(words & set(p.lower().split())))[:k]
```

## Three AI functions and the program

```python
from typing import Literal

Verdict = Literal["supported", "refuted", "not enough info"]

@ai
def next_query(claim: str, notes: list[str]) -> str:
    """A short search query for the fact still missing to check the claim."""

@ai
def take_notes(claim: str, notes: list[str], passages: list[str]) -> list[str]:
    """The notes, plus what the passages say that bears on the claim."""

@ai
def decide(claim: str, notes: list[str]) -> Verdict:
    """Is the claim supported or refuted by the notes?"""

@module
def check_claim(claim: str, hops: int = 2) -> Verdict:
    notes: list[str] = []
    for _ in range(hops):
        notes = take_notes(claim, notes, search(next_query(claim, notes)))
    return decide(claim, notes)

check_claim("The man who designed the Eiffel Tower also worked on the Statue of Liberty.")
```

```output
'supported'
```

## Evaluating the whole program

Rows name the module’s inputs (`claim`) and the expected result. The
metric sees what the module returned; the table’s tokens add up every
model call the program made.

```python
dev = [
    {"claim": "The Eiffel Tower was finished before the Statue of Liberty was given to the US.", "result": "refuted"},
    {"claim": "K2 is taller than Mount Everest.", "result": "refuted"},
    {"claim": "Marie Curie won two Nobel Prizes in different sciences.", "result": "supported"},
    {"claim": "The Great Wall of China can be seen from orbit with the naked eye.", "result": "refuted"},
    {"claim": "Gustave Eiffel's company built a tower completed in 1889.", "result": "supported"},
    {"claim": "The Eiffel Tower was built for a World's Fair.", "result": "supported"},
]

ev = functai.evaluate(check_claim, dev, num_threads=6)
ev
```

```output
Evaluation(check_claim, 6 examples: exact_match 0.83 [0.44, 0.97])
```

```python
from dpyr import col

ev.table.select(col.claim, col.result, col.pred_result, col.input_tokens, col.seconds)
```

```output
# dpyr dataframe · source: polars · showing 6 of 6 rows
shape: (6, 5)
┌────────────────────────────────────────────────┬───────────┬─────────────────┬──────────────┬──────────┐
│ claim                                          ┆ result    ┆ pred_result     ┆ input_tokens ┆ seconds  │
│ ---                                            ┆ ---       ┆ ---             ┆ ---          ┆ ---      │
│ str                                            ┆ str       ┆ str             ┆ i64          ┆ f64      │
╞════════════════════════════════════════════════╪═══════════╪═════════════════╪══════════════╪══════════╡
│ The Eiffel Tower was finished before the Stat… ┆ refuted   ┆ refuted         ┆ 660          ┆ 4.083946 │
│ K2 is taller than Mount Everest.               ┆ refuted   ┆ not enough info ┆ 461          ┆ 3.939941 │
│ Marie Curie won two Nobel Prizes in different… ┆ supported ┆ supported       ┆ 624          ┆ 3.643245 │
│ The Great Wall of China can be seen from orbi… ┆ refuted   ┆ refuted         ┆ 583          ┆ 3.918292 │
│ Gustave Eiffel's company built a tower comple… ┆ supported ┆ supported       ┆ 590          ┆ 3.78339  │
│ The Eiffel Tower was built for a World's Fair… ┆ supported ┆ supported       ┆ 594          ┆ 3.712875 │
└────────────────────────────────────────────────┴───────────┴─────────────────┴──────────────┴──────────┘
```

## Optimizing it

Optimizing a module tunes every AI function it calls, against the one
metric on the program’s output. A run the metric accepts becomes a
worked example (a “demo”) for each AI function it went through.

```python
train = [
    {"claim": "Mount Everest is higher than 8,000 m.", "result": "supported"},
    {"claim": "The Statue of Liberty was a gift from Germany.", "result": "refuted"},
    {"claim": "Marie Curie's chemistry Nobel came before her physics one.", "result": "refuted"},
    {"claim": "Eiffel's company also designed part of the Statue of Liberty.", "result": "supported"},
]

check_claim.opt(trainset=train)
{fn.__name__: len(fn.demos) for fn in check_claim.ai_functions()}
```

```output
{'take_notes': 4, 'next_query': 4, 'decide': 3}
```

```python
after = functai.evaluate(check_claim, dev, num_threads=6)
functai.compare(ev, after)
```

```output
# dpyr dataframe · source: polars · showing 1 of 1 rows
shape: (1, 10)
┌─────────────┬──────────┬──────────┬──────┬─────┬──────┬────────┬───────┬──────┬─────┐
│ metric      ┆ before   ┆ after    ┆ diff ┆ low ┆ high ┆ better ┆ worse ┆ same ┆ n   │
│ ---         ┆ ---      ┆ ---      ┆ ---  ┆ --- ┆ ---  ┆ ---    ┆ ---   ┆ ---  ┆ --- │
│ str         ┆ f64      ┆ f64      ┆ f64  ┆ f64 ┆ f64  ┆ i64    ┆ i64   ┆ i64  ┆ i64 │
╞═════════════╪══════════╪══════════╪══════╪═════╪══════╪════════╪═══════╪══════╪═════╡
│ exact_match ┆ 0.833333 ┆ 0.833333 ┆ 0.0  ┆ 0.0 ┆ 0.0  ┆ 0      ┆ 0     ┆ 6    ┆ 6   │
└─────────────┴──────────┴──────────┴──────┴─────┴──────┴────────┴───────┴──────┴─────┘
```

`check_claim.undo_opt()` reverts every function it tuned;
`check_claim.save("check_claim.json")` keeps their instructions and
demos.

## On a table

A module with a return type works on columns like an AI function:

```python
from dpyr import read

read(dev).mutate(verdict=check_claim(col.claim)).select(col.claim, col.verdict)
```

```output
# dpyr dataframe · source: polars · showing 6 of 6 rows
shape: (6, 2)
┌────────────────────────────────────────────────┬─────────────────┐
│ claim                                          ┆ verdict         │
│ ---                                            ┆ ---             │
│ str                                            ┆ str             │
╞════════════════════════════════════════════════╪═════════════════╡
│ The Eiffel Tower was finished before the Stat… ┆ refuted         │
│ K2 is taller than Mount Everest.               ┆ not enough info │
│ Marie Curie won two Nobel Prizes in different… ┆ supported       │
│ The Great Wall of China can be seen from orbi… ┆ refuted         │
│ Gustave Eiffel's company built a tower comple… ┆ supported       │
│ The Eiffel Tower was built for a World's Fair… ┆ supported       │
└────────────────────────────────────────────────┴─────────────────┘
```
