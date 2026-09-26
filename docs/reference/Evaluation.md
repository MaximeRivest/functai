# Evaluation { #functai.Evaluation }

```{.python .no-run}
Evaluation(target, rows, runs, metrics, values, run_id)
```

The result of ``evaluate``: a score, its uncertainty, and every answer.

``ev.scores(metric)`` gives each row's value for a metric (the first by
default); ``ev.write("run.parquet")`` saves the table.

## Attributes {.doc-section .doc-section-attributes}

| Name        | Type           | Description                                                                                                                                                        |
|-------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| score       | float          | The first metric's mean, from 0 to 1. A failed row counts 0.                                                                                                       |
| summary     | dpyr dataframe | One row per metric: ``mean``, the 95% interval ``low`` to ``high``, ``n``, and how many rows ``failed``.                                                           |
| table       | dpyr dataframe | One row per example: the data, ``pred_<output>`` for each output, each metric, ``error``, ``seconds``, ``input_tokens``, ``output_tokens``, ``model`` and ``run``. |
| predictions | list           | Each row's ``Prediction`` (None where it failed), with its turns, tokens and repairs.                                                                              |
| errors      | list           | ``(row number, message)`` for each row that failed.                                                                                                                |

## See Also {.doc-section .doc-section-see-also}

- [`evaluate`](evaluate.md): what produces it.
- [`compare`](compare.md): two of them, paired.

## Methods

| Name | Description |
| --- | --- |
| [scores](#functai.Evaluation.scores) | One metric's value per row, a failed row counting 0. |
| [write](#functai.Evaluation.write) | Save the table (``.parquet`` keeps every type; also .jsonl, .arrow, ...). |

### scores { #functai.Evaluation.scores }

```{.python .no-run}
Evaluation.scores(metric=None)
```

One metric's value per row, a failed row counting 0.

### write { #functai.Evaluation.write }

```{.python .no-run}
Evaluation.write(path)
```

Save the table (``.parquet`` keeps every type; also .jsonl, .arrow, ...).