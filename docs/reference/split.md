# split { #functai.split }

```{.python .no-run}
split(rows, *, by='conversation', test=0.2, seed=0)
```

Two tables, with every group of rows on one side: ``train, test =
functai.split(functai.rated(tutor))``.

Turns of one conversation depend on each other: a test row whose
conversation is also in the training rows measures memory, not the
program. A row with no group (``conversation`` null) is a group of its own.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type                  | Description                                                                                     | Default          |
|--------|-----------------------|-------------------------------------------------------------------------------------------------|------------------|
| rows   | table or list of dict |                                                                                                 | _required_       |
| by     | str                   | The column that names each row's group (default ``conversation``).                              | `'conversation'` |
| test   | float                 | The share of groups in the test side (at least one group each side when there are two or more). | `0.2`            |
| seed   | int                   |                                                                                                 | `0`              |

## Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                           |
|--------|---------------|-------------------------------------------------------|
|        | (train, test) | Of the same kind as ``rows`` (dpyr tables, or lists). |