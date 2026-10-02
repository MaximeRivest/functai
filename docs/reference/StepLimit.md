# StepLimit { #functai.StepLimit }

```{.python .no-run}
StepLimit(message, turn)
```

A function with tools asked the model ``max_steps`` times (8 by
default) and still had no answer.

``err.turn`` is the exchange so far (every tool call and result), to see
what the model kept doing. Raise the limit with
``@ai(tools=[...], max_steps=20)`` (or ``fn.using(max_steps=20)``), or
make the instruction say when to stop looking things up.