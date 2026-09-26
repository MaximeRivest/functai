# bake.BakeReport { #functai.bake.BakeReport }

```{.python .no-run}
bake.BakeReport(
    function,
    student,
    parameters,
    device,
    truth,
    label_source,
    teacher,
    rows,
    training,
    fields,
    coverage,
    confidence=list(),
    correct=list(),
    speed=dict(),
    labeling=dict(),
    breakeven=dict(),
    notes=list(),
    max_length=0,
    truncated=0,
)
```



## Attributes

| Name | Description |
| --- | --- |
| `accuracy` | What baking measured: accuracy with its interval, calibration, speed, and where to escalate. |

## Methods

| Name | Description |
| --- | --- |
| [threshold](#functai.bake.BakeReport.threshold) | The confidence cut at which the model's own answers reach ``accuracy`` on |

### threshold { #functai.bake.BakeReport.threshold }

```{.python .no-run}
bake.BakeReport.threshold(accuracy=0.95, min_rows=20)
```

The confidence cut at which the model's own answers reach ``accuracy`` on
the test rows: ``{"threshold", "share", "accuracy"}`` (use it as
``escalate_below``), or None when no cut reaches it.