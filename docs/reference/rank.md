# rank { #functai.rank }

```{.python .no-run}
rank(
    program,
    criterion=preferred,
    *,
    source='people',
    drawn=False,
    by='setup',
    groups=None,
    folder=None,
    records=None,
    since=None,
)
```

Rank every setup of a program from side-by-side picks (Bradley–Terry).

Each case and pair of setups that met is one outcome (the later setup's
share of the picks; a failed answer loses). Returns each setup's
strength (the first's 0) with its range, the order, and ``fragility``: at
most how many cases would change which setup leads.