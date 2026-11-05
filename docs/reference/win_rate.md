# win_rate { #functai.win_rate }

```{.python .no-run}
win_rate(
    program,
    a,
    b,
    criterion=preferred,
    *,
    source='people',
    drawn=False,
    margin=None,
    by='setup',
    folder=None,
    records=None,
    since=None,
)
```

How often setup ``b`` wins side by side against ``a`` (ties count half), with its 95% range.

A failed answer loses to any answer. Returns ``rate``, ``low``, ``high``,
``n`` (cases), ``outcome`` (against one half), ``fragility``.