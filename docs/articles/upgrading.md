---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Upgrading

## From 1.1 to 1.2

*What changes when you move from FunctAI 1.1 to 1.2.*

```python
import functai
from functai import ai, module
```

Most of what is new adds to the API: conversations, tools that ask
first, serving, plugins, a new bake. These are the changes that need you
to change code, then the ones that change what your code does without
any change.

### Code to change

| 1.1 | now |
|---|---|
| `@ai(stateful=True)`, `state_window=n`, `fn.history`, `fn.reset()`, `module.history` | a conversation: `chat = fn.conversation("alex")`, called like the function; `context=functai.last_turns(n)` for a window; `chat.turns` for the history; a new conversation (another id) to start over. The function itself remembers nothing ([Memory](memory.md)) |
| `fn(x, all=True)` | `fn.predict(x)` (an input may now be called `all`) |
| `fn.opt(trainset=rows)` changes `fn`; `fn.undo_opt()` reverts | `better = fn.opt(rows)` is an improved copy; `fn` is unchanged, so `evaluate(fn, ...)` and `evaluate(better, ...)` compare side by side |
| `fn.programs()`, `fn.latest_program()` | `fn.optimization_runs()` says how a copy was made; `fn.trials` holds a search's candidates |
| `fn.opt(trainset=rows, optimizer=LabeledFewShot(k=8))` | also by name: `functai.labeled_few_shot(fn, rows, k=8)`, `functai.bootstrap_few_shot(fn, rows, teacher=...)`, `functai.gepa(fn, rows, teacher=...)` |
| `module.opt(...)` changes the AI functions it calls | an improved copy of the module; the AI functions themselves are unchanged (`better.state()` shows what the copy runs with) |
| `from functai import *` brings helpers (`flexiclass`, `sig2str`, `settings`...) | it brings the API only; helpers are `functai.flexiclass`, `functai.settings`, ... |
| a body that is only a docstring | still works; write `...` after the docstring so Pyright (VS Code) accepts the function, or `return _ai`, which mypy accepts too |
| `def f(x, /)` (positional-only) in `@ai` or `@module` | refused when defined: a program's inputs are given by name (a row of data, another language). Remove the `/` |
| a `@module` annotated with a type defined further down | refused when defined: define the type first |
| `fn.map(rows, num_threads=8)` | `threads=8` (`num_threads` still works) |
| sync calls only | `await fn.acall(x)`, `await fn.apredict(x)`, and `@ai async def` for a function whose body is the model call |

Settings that no longer exist are refused by name (`TypeError: ... unknown
setting(s) ['stateful']`), so nothing is silently ignored.

**Baked models.** Bake was rebuilt ([Bake it into a small model](baking.md)):

- **A model baked with 1.1 is refused**, a head included (`baked.json`
  format 1): bake it again. A saved program that names one cannot load
  until then.
- `fn.bake(rows)` now decides what to train (`method="auto"`): a head when
  every output has a fixed set of answers, as before; for any other
  function, a generative student, where 1.1 refused. That needs
  `pip install "functai[bake]"` and a GPU here, or a training service you
  have set up (`"functai[tinker]"`); `plan_only=True` says what it would
  do and what it would cost, and spends nothing.
- The teacher is no longer run again on the test rows by default
  (`compare_teacher=True` to compare).

### Inputs are checked, and converted when the meaning is clear

Every call, of an AI function or a module, now binds each input to its
declared type before anything else runs. A value whose meaning is clear
is converted; anything else is refused with `functai.InterfaceError`
(code `interface-input`), and nothing is sent:

```python
@module
def double(n: int) -> int:
    return n * 2

double("5"), double(5.0)
```

```output
(10, 10)
```

```python
try:
    double("five")
except functai.InterfaceError as error:
    print(error.code, error.field, "·", error)
```

```output
interface-input n · double: input 'n': "five" does not bind to {"type":"integer"}
```

- Text takes a number (`42` is `"42"`), `True`/`False`, a list or a record
  (as JSON), or a value with a text of its own (a date, a data frame);
  `<object at 0x…>` is refused. An integer takes `5.0` and `"5"`; a number
  takes `"2.5"`; a boolean only `True` or `False`.
- A record input keeps only the members it declares.
- A missing value (`None`, `NaN`, pandas' `NA`) for an optional input is
  that input left out: it gets its default. That is what a table's empty
  cell now means.
- Outputs are checked, never converted, and a record holds only its
  fields: a model's reply with another member does not fit, and is
  asked again.
- `InterfaceError` is a `TypeError` and a `ValueError`, so an `except
  TypeError` you already have still catches it.

### What changes without a code change

- **Versions.** Every function with a default, and every module, gets a
  new `version` once (a default now counts by what is written, a module's
  version includes its interface). The call log's
  [Versions](call-log.md#versions) shows a new one. Ratings are matched by
  signature, which defaults do not change, so they still apply.
- **Ratings made with 1.1 in a notebook.** `rated(fn)` now tells two
  notebooks' functions of the same name apart by the notebook's file. A
  call logged by 1.1 from a notebook names another file (the kernel's
  cell file), so some of your earlier ratings may be missing from
  `rated(fn)`: `functai.rated(fn, any_file=True)` takes them from every
  file.
- **Who rated.** A rating made without `by=` is recorded under the
  computer's account, and kept on its own: on a shared account, one
  person's rating no longer replaces another's (a disagreement is marked
  `disputed`).
- **A reply cut off at the token limit** is sent again with twice
  `max_tokens` only when you set `max_tokens`. Without it, the reply
  already had the model's whole limit, so it fails at once, and the error
  says how many tokens went to thinking and what to change.
- **Models that only run at temperature 1** (GPT-6; Claude Opus, Sonnet,
  Fable and Mythos 5) no longer fail when you set `temperature=0`: the
  setting is left out of their requests, with one warning.
- **The reply cache** never keeps a reply that could not be read, and the
  optimizers' teacher is never answered from it.
- **A tool that returns a record** (a dataclass, a pydantic model) is
  shown to the model as JSON, not as Python's `str()`.
- **Error messages quote the value at fault** (cut after 80
  characters), unless `log_content` keeps that field out of the log.
- **The call log is format 2.** FunctAI reads both formats; if you parse
  the JSON lines yourself, records have new fields (`program.interface`,
  `saw`, `request_hash`...): [`contract/calls.md`](https://github.com/maximerivest/functai/blob/master/contract/calls.md)
  says each.

## From 0.x

*What changed in 1.0, and how to move code over.*

1.0 kept the way you write functions (`@ai`, `_ai`, `configure`,
`all=True`, `stateful`, `tools`, `module="cot"`, `.opt`, `undo_opt`,
`@module`, `phistory`); some of these have changed since (see *From 1.1*
above). What runs underneath is new: prompts are laid out
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
| `stateful` history in a separate object | lmcc turns in `fn.history` (since replaced by conversations) |

### Behaviour changes in 1.0

- **Automatic instruction writing is opt-in.** In 0.x every new function
  asked the model to rewrite its own instruction (`autoinstruct`), and
  the first calls refined it again. That spent money at import time and
  made prompts change by themselves. Now `@ai(autoinstruct=True)` or
  `@ai(instruction_autorefine_calls=2)` turns them on; they run at the
  first call, not at definition.
- Prompts are laid out by lmcc, so their text differs from 0.x. Re-run
  your evaluations after upgrading.
- Unknown settings raise instead of being ignored.
