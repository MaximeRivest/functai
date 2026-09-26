# runs { #functai.runs }

```{.python .no-run}
runs(folder)
```

Every evaluation logged in a folder, as one table.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type        | Description                                        | Default    |
|--------|-------------|----------------------------------------------------|------------|
| folder | str or path | The folder given to ``evaluate(..., log=folder)``. | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type           | Description                                                                                                         |
|--------|----------------|---------------------------------------------------------------------------------------------------------------------|
|        | dpyr dataframe | The rows of every run, with a ``run`` column. Runs with different columns line up by name; missing values are null. |

## See Also {.doc-section .doc-section-see-also}

evaluate : ``log=`` writes the runs.