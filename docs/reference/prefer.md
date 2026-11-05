# prefer { #functai.prefer }

```{.python .no-run}
prefer(
    a,
    b,
    winner,
    *,
    criterion=preferred,
    by=None,
    blind=None,
    sample=None,
    note=None,
    shown=None,
    folder=None,
)
```

Say which of two answers to the same case is better.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type                                         | Description                                                                                        | Default    |
|--------|----------------------------------------------|----------------------------------------------------------------------------------------------------|------------|
| a      | predictions, call ids or call records        | In the order they were shown (or give ``shown``).                                                  | _required_ |
| b      | predictions, call ids or call records        | In the order they were shown (or give ``shown``).                                                  | _required_ |
| winner | one of them, ``\"tie\"``, or None (withdraw) |                                                                                                    | _required_ |
| blind  | bool                                         | You were not told which version made which (show them in a random order: ``functai.shown_order``). | `None`     |
| sample | str                                          | The random draw this pair is part of.                                                              | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description         |
|--------|--------|---------------------|
|        | dict   | The rating written. |