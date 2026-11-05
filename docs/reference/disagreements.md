# disagreements { #functai.disagreements }

```{.python .no-run}
disagreements(
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

Cases worth a look, or worth adding to the examples: those on which
setups' values differ, then those on which one setup's repeats differ,
the largest spread first. Each ``{"case", "kind", "spread", "values"}``.