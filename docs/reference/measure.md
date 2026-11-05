# measure { #functai.measure }

```{.python .no-run}
measure(
    program,
    criterion=correct,
    *,
    source='people',
    drawn=False,
    by='setup',
    folder=None,
    records=None,
    since=None,
)
```

Every result's value under a criterion and source, and each group's
value by case: ``{"results": [...], "cases": {group: {case: value}}}``.