# Optimizer { #functai.Optimizer }

```{.python .no-run}
Optimizer()
```

The base class of optimizers: subclass it to write your own.

An optimizer has one method, ``compile(program, *, trainset, valset=None)``:
given an AI function or a ``@module`` and rows with known answers (a list
of dicts), it
returns the new state of each AI function it improves, as
``{fn: ProgramState(instructions=..., demos=...)}``. It never changes the
functions itself: ``fn.opt(rows, optimizer=MyOptimizer())`` builds the
improved copy from what it returns. ``metric`` (``metric(row,
prediction)``) is set from ``fn.opt(metric=...)`` when the optimizer has
none.

```{.python .no-run}
class FirstRows(functai.Optimizer):          # the first three rows become worked examples
    def compile(self, program, *, trainset, valset=None):
        demos = tuple({"inputs": {"message": r["message"]}, "outputs": {"result": r["team"]}}
                      for r in trainset[:3])
        return {program: functai.ProgramState(demos=demos)}

better = team.opt(rows, optimizer=FirstRows())
```