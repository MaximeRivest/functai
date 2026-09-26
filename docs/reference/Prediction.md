# Prediction { #functai.Prediction }

```{.python .no-run}
Prediction(
    values,
    *,
    turn=None,
    response=None,
    responses=(),
    repairs=(),
    attempts=1,
    probabilities=None,
    measured_by=None,
)
```

Everything one call produced.

- the outputs, by name (``pred.result``, ``pred.reasoning``, ``dict(pred)``)
- ``pred.turn``: the lmcc turn (inputs, every model and tool step, outputs)
- ``pred.response`` / ``pred.responses``: the lm15 responses
- ``pred.usage``: tokens summed over every model call
- ``pred.repairs``: what the reader forgave in the reply

## Attributes

| Name | Description |
| --- | --- |
| `confidence` | How sure the model was: the probability it gave its own answer, the lowest |