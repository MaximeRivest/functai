# Evaluation { #functai.Evaluation }

```{.python .no-run}
Evaluation(
    target,
    rows,
    runs,
    metrics,
    values,
    run_id,
    jobs=None,
    keys=None,
    kinds=None,
)
```

The result of ``evaluate``: a score, its uncertainty, and every answer.

``ev.scores(metric)`` gives each row's value for a metric (the first by
default; a row asked several times, the mean of its repeats);
``ev.write("run.parquet")`` saves the table.

## Attributes {.doc-section .doc-section-attributes}

| Name        | Type           | Description                                                                                                                                                                                                                                                                                                                                                       |
|-------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| score       | float          | The first metric's mean over cases, from 0 to 1. A failed row counts 0.                                                                                                                                                                                                                                                                                           |
| summary     | dpyr dataframe | One row per metric: ``mean``, the 95% interval ``low`` to ``high`` (over cases), ``n`` (cases), and how many runs ``failed``.                                                                                                                                                                                                                                     |
| table       | dpyr dataframe | One row per run (``repeats`` runs per example): the data, ``pred_<output>`` for each output, each metric, ``error``, ``seconds``, ``input_tokens``, ``output_tokens``, ``reasoning_tokens``, ``total_tokens``, ``model``, ``run``, ``case_key``, ``repeat``, ``call_id`` (the call that answered: its trace in the call log) and ``reused`` (taken from the log). |
| predictions | list           | Each run's ``Prediction`` (None where it failed), with its turns, tokens and repairs.                                                                                                                                                                                                                                                                             |
| errors      | list           | ``(row number, message)`` for each run that failed.                                                                                                                                                                                                                                                                                                               |
| cases       | list of str    | Each row's case key (``functai.case_key``).                                                                                                                                                                                                                                                                                                                       |

## See Also {.doc-section .doc-section-see-also}

- [`evaluate`](evaluate.md): what produces it.
- [`compare`](compare.md): two of them, paired by case.

## Methods

| Name | Description |
| --- | --- |
| [by_case](#functai.Evaluation.by_case) | ``{case key: value}``: a metric's value per case (rows of one case, and repeats, averaged). |
| [scores](#functai.Evaluation.scores) | One metric's value per row (a failed row counting 0; the mean of a row's repeats). |
| [write](#functai.Evaluation.write) | Save the table (``.parquet`` keeps every type; also .jsonl, .arrow, ...). |

### by_case { #functai.Evaluation.by_case }

```{.python .no-run}
Evaluation.by_case(metric=None)
```

``{case key: value}``: a metric's value per case (rows of one case, and repeats, averaged).

### scores { #functai.Evaluation.scores }

```{.python .no-run}
Evaluation.scores(metric=None)
```

One metric's value per row (a failed row counting 0; the mean of a row's repeats).

### write { #functai.Evaluation.write }

```{.python .no-run}
Evaluation.write(path)
```

Save the table (``.parquet`` keeps every type; also .jsonl, .arrow, ...).