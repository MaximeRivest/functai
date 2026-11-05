# judge_pairs { #functai.judge_pairs }

```{.python .no-run}
judge_pairs(
    judge_,
    pairs,
    criterion=preferred,
    *,
    both_orders=True,
    folder=None,
    records=None,
    sample=None,
    threads=1,
)
```

Have a judge pick the better of two answers to the same case, in both orders.

Judges favour a position; asked both ways, a judge's preference for the
first (or second) shown cancels out, and ``positions`` measures it. The
judge is given ``first`` and ``second`` (the answers), the case's inputs,
and ``first_outputs``, ``second_outputs``; it answers ``"first"``,
``"second"`` or ``"tie"``.

## Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description                                         |
|--------|--------------|-----------------------------------------------------|
|        | list of dict | The ratings written (``calls`` in the order shown). |