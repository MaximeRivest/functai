# compare_setups { #functai.compare_setups }

```{.python .no-run}
compare_setups(
    program,
    a,
    b,
    criterion=correct,
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

Is setup ``b`` better than ``a``, on the cases both answered.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type                                                                       | Description                             | Default    |
|-----------|----------------------------------------------------------------------------|-----------------------------------------|------------|
| program   | AI function, module, or name                                               |                                         | _required_ |
| a         | setup ids (or versions with ``by=\"version\"``), or programs (their setup) |                                         | _required_ |
| b         | setup ids (or versions with ``by=\"version\"``), or programs (their setup) |                                         | _required_ |
| criterion | Criterion                                                                  |                                         | `correct`  |
| source    | \"people\", a judge (program or function), or a judge\'s id                |                                         | `'people'` |
| drawn     | bool                                                                       | Only people's ratings of a random draw. | `False`    |
| margin    | float                                                                      | How much worse is still acceptable.     | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                                  |
|--------|--------|----------------------------------------------------------------------------------------------------------------------------------------------|
|        | dict   | ``diff`` (b - a), ``low``, ``high``, ``method``, ``outcome``, ``n``, ``better``, ``worse``, ``same``, ``fragility``, ``only_a``, ``only_b``. |