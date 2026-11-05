# self_agreement { #functai.self_agreement }

```{.python .no-run}
self_agreement(
    program,
    criterion=correct,
    *,
    source='people',
    by='setup',
    folder=None,
    records=None,
    since=None,
)
```

How much each setup agrees with itself, over cases it answered two or
more times: the mean pairwise agreement with its range, and ``spread``
(the typical difference between repeats). ``{group: {...}}``.