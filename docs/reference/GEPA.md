# GEPA { #functai.GEPA }

```{.python .no-run}
GEPA(
    metric=None,
    *,
    budget=300,
    minibatch=4,
    teacher=None,
    feedback=None,
    num_threads=8,
    seed=0,
)
```

Rewrite the instruction from the function's mistakes: GEPA (Agrawal et
al., 2025), with functai's changes (design/04-gepa.md).

Half the rows (or ``valset``) select, and are never shown; the other half
give feedback. A pool of instructions is scored row by row on the selection
rows; a parent is picked from its Pareto frontier, run on ``minibatch``
feedback rows, and a ``teacher`` model reads its answers with feedback in
words and writes a new instruction. A child that does better on the
minibatch is scored on the selection rows and joins the pool. Every fourth
step, two frontier candidates that win different rows are combined. The
reflection sees what was tried and failed; a proposal that copies an input
is dropped; ties go to the shorter instruction; a row is run once per
instruction.

``budget`` counts calls of the function; ``trials`` holds every candidate
afterwards (its selection score is optimistic: it was chosen on those rows).
``feedback(row, prediction, error) -> str`` gives the words (default: right,
or the right answer, or the error).