# agreement { #functai.agreement }

```{.python .no-run}
agreement(
    program,
    criterion,
    a='people',
    b=None,
    *,
    drawn=False,
    folder=None,
    records=None,
    since=None,
)
```

Can a judge stand in for people: two sources on the targets both judged.

Right-or-wrong values and labels: ``agree`` (with its range) and Cohen's
``kappa``; scores: ``mad`` (mean absolute difference) and ``r``; picks:
the same winner. Items of one answer are counted as targets of their own:
this describes the judge, it is not a test.