---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Multi-step programs

*@module: ordinary Python that calls AI functions, measured, optimized and saved as one program.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

Real tasks rarely fit in one model call. You search, then take notes,
then decide; you draft, then check. In functai the program that ties
those calls together is plain Python: loops, ifs, helpers. Put `@module`
on it and it becomes one program you can evaluate, optimize and save.

## A research loop

Two AI functions and an ordinary search function:

```python
@ai
def next_query(claim: str, notes: list[str]) -> str:
    """A search query that would help check the claim, given the notes so far."""

@ai
def take_notes(claim: str, notes: list[str], documents: list[str]) -> list[str]:
    """The notes, extended with what the documents say about the claim."""

LIBRARY = {
    "eiffel": "The Eiffel Tower is a wrought-iron tower in Paris, completed in 1889.",
    "k2": "K2 is the second-highest mountain on Earth, on the China–Pakistan border.",
    "everest": "Mount Everest lies on the border between Nepal and China.",
}

def search(query: str) -> list[str]:
    """Your retriever: a vector store, a search API, ..."""
    words = query.lower().split()
    return [text for key, text in LIBRARY.items() if any(key in w for w in words)] or ["No results."]
```

The program calls them in a loop, and a third AI function decides:

```python
from typing import Literal
from functai import module

@ai
def verdict(claim: str, notes: list[str]) -> Literal["true", "false", "unknown"]:
    """Is the claim true, according to the notes?"""

@module
def fact_check(claim: str, hops: int = 2) -> Literal["true", "false", "unknown"]:
    notes: list[str] = []
    for _ in range(hops):
        query = next_query(claim, notes)
        notes = take_notes(claim, notes, search(query))
    return verdict(claim, notes)

fact_check("K2 is in Nepal.")
```

```output
'false'
```

`phistory(5)` shows the calls it made, in order:

```python
print(functai.phistory(5)[:1500], "…")
```

```output
[2026-09-26T17:38:15] next_query → gpt-4.1-mini

System message:

Function: next_query

A search query that would help check the claim, given the notes so far.

Reply in exactly this form:
<result>
...
</result>


User message:

<claim>
K2 is in Nepal.
</claim>
<notes>
[]
</notes>


Response:

<result>
Is K2 located in Nepal?
</result>

(finish: stop; tokens in 64, out 14)

────────────────────────────────────────────────────────────

[2026-09-26T17:38:16] take_notes → gpt-4.1-mini

System message:

Function: take_notes

The notes, extended with what the documents say about the claim.

Reply in exactly this form:
<result>
JSON matching this schema: {"type": "array", "items": {"type": "string"}}
</result>


User message:

<claim>
K2 is in Nepal.
</claim>
<notes>
[]
</notes>
<documents>
[
  "K2 is the second-highest mountain on Earth, on the China–Pakistan border."
]
</documents>


Response:

<result>
["K2 is in Nepal is incorrect because K2 is located on the China–Pakistan border, not in Nepal."]
</result>

(finish: stop; tokens in 109, out 31)

────────────────────────────────────────────────────────────

[2026-09-26T17:38:17] next_query → gpt-4.1-mini

System message:

Function: next_query

A search query that would help check the claim, given the notes so far.

Reply in exactly this form:
<result>
...
</result>


User message:

<claim>
K2 is in Nepal.
</claim>
<notes>
[
  "K2 is in Nepal is incorrect because K2 is located on the China–Pakistan border, not in Nepal."
]
</not …
```

## Measured and improved as one

A module is evaluated like a single function. The metric sees what the
module returned (`pred_result` in the table), and the tokens of every
inner call are added up per row.

```python
claims = [
    {"claim": "The Eiffel Tower is in Paris.", "result": "true"},
    {"claim": "K2 is in Nepal.", "result": "false"},
    {"claim": "Mount Everest is on the border of Nepal.", "result": "true"},
    {"claim": "The Eiffel Tower was finished in 1920.", "result": "false"},
]

ev = functai.evaluate(fact_check, claims, num_threads=4)
ev
```

```output
Evaluation(fact_check, 4 examples: exact_match 1.00 [0.51, 1.00])
```

Optimizing a module improves each AI function inside it: every run the
metric accepts gives a worked example to each function it went through.

```{.python .no-run}
fact_check.opt(trainset=claims, call_defaults=dict(hops=1))
```

## Why a module and not a plain function?

A plain Python function that calls AI functions works; you just can't
treat it as one thing. `@module` adds:

- `evaluate(program, data)` and `program.opt(...)` for the whole program;
- a typed signature (`claim: str → Literal[...]`), so it can be saved,
  run on a table, and checked;
- `functai.check(program)` follows every function, tool and constant it
  reaches, so [saving](saving.md) takes all of it.
