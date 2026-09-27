---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Upgrading

## From 1.1

Four changes, each so that the API says what it does and a type checker
can follow it:

| 1.1 | now |
|---|---|
| `fn(x, all=True)` | `fn.predict(x)` (an input may now be called `all`) |
| `fn.opt(trainset=rows)` changes `fn`; `fn.undo_opt()` reverts | `better = fn.opt(rows)` is an improved copy; `fn` is unchanged, so `evaluate(fn, ...)` and `evaluate(better, ...)` compare side by side |
| `fn.opt(trainset=rows, optimizer=LabeledFewShot(k=8))` | also by name: `functai.labeled_few_shot(fn, rows, k=8)`, `functai.bootstrap_few_shot(fn, rows, teacher=...)`, `functai.gepa(fn, rows, teacher=...)` |
| `module.opt(...)` changes the AI functions it calls | an improved copy of the module; the AI functions themselves are unchanged (`better.state()` shows what the copy runs with) |
| a body that is only a docstring | still works; write `...` after the docstring so Pyright (VS Code) accepts the function, or `return _ai`, which mypy accepts too |
| sync calls only | `await fn.acall(x)`, `await fn.apredict(x)`, and `@ai async def` for a function whose body is the model call |

`fn.programs()` and `fn.latest_program()` are gone (`fn.optimization_runs()`
says how a copy was made; `fn.trials` is a search's candidates), and
`from functai import *` no longer brings helpers such as `flexiclass` or
`sig2str` (they are `functai.flexiclass`, ...).

## From 0.x

*What changed in 1.0, and how to move code over.*

1.0 keeps the way you write functions (`@ai`, `_ai`, `configure`,
`all=True`, `stateful`, `tools`, `module="cot"`, `.opt`, `undo_opt`,
`@module`, `phistory`). What runs underneath is new: prompts are laid out
by [lmcc](https://github.com/MaximeRivest/lmcc) and sent by
[lm15](https://github.com/lm15-dev/lm15-python). Data, metrics and
optimizers are new too.

| 0.x | 1.0 |
|---|---|
| `configure(lm=<an LM object>)` | `configure(lm="gpt-4.1")`: a model name (litellm spellings work) |
| training data as `Example(...).with_inputs(...)` | a dict per row, or a table; columns named like the parameters are the inputs |
| `metric(example, pred, trace=None)` | `metric(row, prediction)`, or a dpyr expression |
| optimizer classes from other libraries | `functai.BootstrapFewShot`, `functai.InstructionSearch`, … |
| an evaluator returning a percentage | `functai.evaluate(program, data, metric)`: `.score` (0 to 1, with an interval), `.table` |
| `adapter="json"`, `adapter="chat"` | same names, now lmcc layouts |
| custom adapter classes | `template=[system(...), turns(), user(...)]`, or an `lmcc.Adapter` |
| tools switched the program to an agent module | tools run in a tool loop; the prompt does not change |
| `fn.signature` | an lmcc `SignatureCore` |
| exporting the program to another framework | removed; `fn.state()`, `fn.save(path)`, [`functai.save`](saving.md) |
| `stateful` history in a separate object | lmcc turns in `fn.history` |

## Behaviour changes

- **Automatic instruction writing is opt-in.** In 0.x every new function
  asked the model to rewrite its own instruction (`autoinstruct`), and
  the first calls refined it again. That spent money at import time and
  made prompts change by themselves. Now `@ai(autoinstruct=True)` or
  `@ai(instruction_autorefine_calls=2)` turns them on; they run at the
  first call, not at definition.
- Prompts are laid out by lmcc, so their text differs from 0.x. Re-run
  your evaluations after upgrading.
- Unknown settings raise instead of being ignored.
