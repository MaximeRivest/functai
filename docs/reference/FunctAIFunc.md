---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# FunctAIFunc { #functai.FunctAIFunc }

```{.python .no-run}
FunctAIFunc(
    fn,
    *,
    tools=None,
    template=None,
    messages=None,
    module_kwargs=None,
    examples=None,
    requires=None,
    **cfg,
)
```

A typed Python function whose body is a model call. Build with ``@ai``.

## Attributes

| Name | Description |
| --- | --- |
| `instructions` | The instruction the model gets: an optimized one, or the one written from the code. |
| `signature` | The lmcc signature: inputs, outputs, instruction. |
| `version` | The function's version: a fingerprint of what it sends besides its inputs. |

## Methods

| Name | Description |
| --- | --- |
| [bake](#functai.FunctAIFunc.bake) | Train weights that answer this function; returns the baked model. |
| [explain](#functai.FunctAIFunc.explain) | How calls are laid out for the current model: adapter, reader, transports, formats. |
| [freeze](#functai.FunctAIFunc.freeze) | Stop further automatic instruction refinement. |
| [latest_program](#functai.FunctAIFunc.latest_program) | The state in use (``fresh=True``: the unoptimized one). |
| [map](#functai.FunctAIFunc.map) | Run on every row of a table, and return the run table. |
| [opt](#functai.FunctAIFunc.opt) | Improve the instruction and worked examples from examples, in place. |
| [plan](#functai.FunctAIFunc.plan) | The lmcc plan for the current model: ``.explain()``, ``.describe()``, ``.render(...)``. |
| [programs](#functai.FunctAIFunc.programs) | Every state optimization produced for this function, oldest first. |
| [render](#functai.FunctAIFunc.render) | The exact request the next call would send, without sending it. |
| [reset](#functai.FunctAIFunc.reset) | Forget the conversation (stateful functions). |
| [save](#functai.FunctAIFunc.save) | Write the instruction and demos to a JSON file (``load`` reads it back). |
| [state](#functai.FunctAIFunc.state) | The instruction and demos in use. |
| [stream](#functai.FunctAIFunc.stream) | Call the function and watch the answer being written. |
| [undo_opt](#functai.FunctAIFunc.undo_opt) | Revert the last optimizations. |
| [unpack](#functai.FunctAIFunc.unpack) | One column per field of the answer, to spread into a table. |
| [using](#functai.FunctAIFunc.using) | A copy of this function with other settings or another layout. |
| [vectorize](#functai.FunctAIFunc.vectorize) | This function as a column expression, with options. |

### bake { #functai.FunctAIFunc.bake }

```{.python .no-run}
FunctAIFunc.bake(data, **options)
```

Train weights that answer this function; returns the baked model.
``fast = fn.using(lm=baked)`` runs the function on them. See
``functai.bake.bake`` for the options (student, teacher, labels, test, ...).

### explain { #functai.FunctAIFunc.explain }

```{.python .no-run}
FunctAIFunc.explain()
```

How calls are laid out for the current model: adapter, reader, transports, formats.

### freeze { #functai.FunctAIFunc.freeze }

```{.python .no-run}
FunctAIFunc.freeze()
```

Stop further automatic instruction refinement.

### latest_program { #functai.FunctAIFunc.latest_program }

```{.python .no-run}
FunctAIFunc.latest_program(fresh=False)
```

The state in use (``fresh=True``: the unoptimized one).

### map { #functai.FunctAIFunc.map }

```{.python .no-run}
FunctAIFunc.map(data, *, num_threads=1)
```

Run on every row of a table, and return the run table.

``evaluate`` without the scoring: the rows, the predictions
(``pred_<output>``), and each row's ``error``, ``seconds``, tokens and
``model``. Needs ``pip install "functai[data]"``.

#### Parameters {.doc-section .doc-section-parameters}

| Name        | Type                     | Description                                                                       | Default    |
|-------------|--------------------------|-----------------------------------------------------------------------------------|------------|
| data        | list of dict, or a table | Anything ``dpyr.read()`` takes; columns named like the parameters are the inputs. | _required_ |
| num_threads | int                      | How many rows run at once.                                                        | `1`        |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type           | Description   |
|--------|----------------|---------------|
|        | dpyr dataframe |               |

#### See Also {.doc-section .doc-section-see-also}

- `FunctAIFunc.vectorize`: the function as a column expression.

#### Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@ai
def capital(country: str) -> str:
    """The country's capital city."""

capital.map([{"country": "Norway"}, {"country": "Ghana"}], num_threads=2)
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
# dpyr dataframe · source: polars · showing 2 of 2 rows
┌─────────┬─────────┬─────────────┬───────┬──────────┬──────────────┬───────────────┬─────────────────────────┬──────────────────────────────┐
│ example ┆ country ┆ pred_result ┆ error ┆ seconds  ┆ input_tokens ┆ output_tokens ┆ model                   ┆ run                          │
│ ---     ┆ ---     ┆ ---         ┆ ---   ┆ ---      ┆ ---          ┆ ---           ┆ ---                     ┆ ---                          │
│ i64     ┆ str     ┆ str         ┆ null  ┆ f64      ┆ i64          ┆ i64           ┆ str                     ┆ str                          │
╞═════════╪═════════╪═════════════╪═══════╪══════════╪══════════════╪═══════════════╪═════════════════════════╪══════════════════════════════╡
│ 0       ┆ Norway  ┆ Oslo        ┆ null  ┆ 0.860826 ┆ 42           ┆ 10            ┆ gpt-4.1-mini-2025-04-14 ┆ capital-20260926-232006-1857 │
│ 1       ┆ Ghana   ┆ Accra       ┆ null  ┆ 0.700394 ┆ 42           ┆ 10            ┆ gpt-4.1-mini-2025-04-14 ┆ capital-20260926-232006-1857 │
└─────────┴─────────┴─────────────┴───────┴──────────┴──────────────┴───────────────┴─────────────────────────┴──────────────────────────────┘
```

### opt { #functai.FunctAIFunc.opt }

```{.python .no-run}
FunctAIFunc.opt(trainset=None, optimizer=None, metric=None, valset=None, **opts)
```

Improve the instruction and worked examples from examples, in place.

Only what the function sends besides its inputs changes: the
instruction and the demos. Code, types and layout are never touched.
``undo_opt()`` reverts.

#### Parameters {.doc-section .doc-section-parameters}

| Name       | Type                        | Description                                                                                                  | Default    |
|------------|-----------------------------|--------------------------------------------------------------------------------------------------------------|------------|
| trainset   | list of dict, or a table    | Rows as for ``evaluate``: columns named like the parameters are the inputs, the others the expected outputs. | `None`     |
| expected   | str or dict                 | The column holding the right answers, as for ``evaluate``: ``expected="category"``.                          | _required_ |
| optimizer  | optimizer class or instance | Default ``BootstrapFewShot``. See the Optimizers section.                                                    | `None`     |
| metric     | function or dpyr expression | As for ``evaluate``. Default: exact match on the expected outputs.                                           | `None`     |
| valset     | list of dict, or a table    | Rows for optimizers that choose between candidates.                                                          | `None`     |
| teacher_lm | str                         | A stronger model that runs the examples; its good runs become demos.                                         | _required_ |
| teacher    | AI function                 | Or a teacher function.                                                                                       | _required_ |
| n_synth    | int                         | With a teacher: first write this many training rows.                                                         | _required_ |
| **opts     |                             | Passed to the optimizer.                                                                                     | `{}`       |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description                   |
|--------|-------------|-------------------------------|
|        | FunctAIFunc | The same function, optimized. |

#### See Also {.doc-section .doc-section-see-also}

- [`evaluate`](evaluate.md): measure before and after.
- `FunctAIFunc.undo_opt`: revert.

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
from typing import Literal

@ai
def category(message: str) -> Literal["shipping", "billing", "product"]:
    """The support category of the message."""

train = [
    {"message": "The vase came smashed.", "result": "shipping"},
    {"message": "Money back please, the chair wobbles.", "result": "billing"},
    {"message": "The handle came off after two uses.", "result": "product"},
]
category.opt(trainset=train)
[d.inputs["message"] for d in category.demos]
```

### plan { #functai.FunctAIFunc.plan }

```{.python .no-run}
FunctAIFunc.plan()
```

The lmcc plan for the current model: ``.explain()``, ``.describe()``, ``.render(...)``.

### programs { #functai.FunctAIFunc.programs }

```{.python .no-run}
FunctAIFunc.programs()
```

Every state optimization produced for this function, oldest first.

### render { #functai.FunctAIFunc.render }

```{.python .no-run}
FunctAIFunc.render(*args, **kwargs)
```

The exact request the next call would send, without sending it.

#### Parameters {.doc-section .doc-section-parameters}

| Name     | Type   | Description                                     | Default   |
|----------|--------|-------------------------------------------------|-----------|
| *args    |        | The call's inputs, as for calling the function. | `()`      |
| **kwargs |        | The call's inputs, as for calling the function. | `()`      |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description                                                      |
|--------|--------------|------------------------------------------------------------------|
|        | lm15.Request | ``.system``, ``.messages``, ``.tools``, ``.config``, ``.model``. |

#### See Also {.doc-section .doc-section-see-also}

- [`phistory`](phistory.md): what was actually sent.

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
@ai
def capital(country: str) -> str:
    """The country's capital city."""

request = capital.render("Chile")
print(request.system)
print(request.messages[0].parts[0].text)
```

### reset { #functai.FunctAIFunc.reset }

```{.python .no-run}
FunctAIFunc.reset()
```

Forget the conversation (stateful functions).

### save { #functai.FunctAIFunc.save }

```{.python .no-run}
FunctAIFunc.save(path)
```

Write the instruction and demos to a JSON file (``load`` reads it back).

### state { #functai.FunctAIFunc.state }

```{.python .no-run}
FunctAIFunc.state()
```

The instruction and demos in use.

### stream { #functai.FunctAIFunc.stream }

```{.python .no-run}
FunctAIFunc.stream(*args, **kwargs)
```

Call the function and watch the answer being written.

The call starts at once, in the background, and is the same call as
``fn(...)``: the same retries, tools and call log line, the same value
in the end. Iterate the stream for the answer's text as it arrives.

#### Parameters {.doc-section .doc-section-parameters}

| Name     | Type   | Description                                     | Default   |
|----------|--------|-------------------------------------------------|-----------|
| *args    |        | The call's inputs, as for calling the function. | `()`      |
| **kwargs |        | The call's inputs, as for calling the function. | `()`      |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                                                                                                                                                                                                                               |
|--------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | Stream | ``for piece in s`` (or ``async for``): the answer's text, piece by piece. ``s.result``: the value (waits); ``await s`` in async code. ``s.events()``: everything, with the reasoning, tool calls and retries. ``s.text``, ``s.partial``: the answer so far. ``s.close()``: stop the call. |

#### See Also {.doc-section .doc-section-see-also}

- [`Stream`](Stream.md): what this returns.

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
@ai
def haiku(topic: str) -> str:
    """A haiku about the topic."""

for piece in haiku.stream("the first snow"):
    print(piece, end="", flush=True)
```

Everything the model writes, reasoning first:

```{.python .no-run}
@ai
def solve(problem: str) -> float:
    """Solve the word problem."""
    reasoning: str = _ai["Step by step."]
    return _ai

s = solve.stream("3 pencils cost $1.20. How much do 10 cost?")
for event in s.events():
    if event.kind == "text":
        print(event.text, end="", flush=True)
s.result
```

### undo_opt { #functai.FunctAIFunc.undo_opt }

```{.python .no-run}
FunctAIFunc.undo_opt(steps=1)
```

Revert the last optimizations.

#### Parameters {.doc-section .doc-section-parameters}

| Name   | Type   | Description                       | Default   |
|--------|--------|-----------------------------------|-----------|
| steps  | int    | How many optimizations to revert. | `1`       |

### unpack { #functai.FunctAIFunc.unpack }

```{.python .no-run}
FunctAIFunc.unpack(*args, threads=None, errors='raise', prefix='', **kwargs)
```

One column per field of the answer, to spread into a table.

For a function whose answer is a record (a dataclass, a pydantic
model, a TypedDict): ``table.mutate(**fn.unpack(col.text))`` adds
one column per field. Each distinct row still costs one model call.

#### Parameters {.doc-section .doc-section-parameters}

| Name     | Type   | Description                                                        | Default   |
|----------|--------|--------------------------------------------------------------------|-----------|
| *args    | Any    | The inputs, as columns (``col.note``) or constants.                | `()`      |
| **kwargs | Any    | The inputs, as columns (``col.note``) or constants.                | `()`      |
| threads  | int    | How many rows run at once (default 8).                             | `None`    |
| errors   | str    | ``"raise"`` (default) or ``"null"``, as for ``vectorize``.         | `'raise'` |
| prefix   | str    | Put before each column's name (``prefix="ai_"`` → ``ai_species``). | `''`      |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                            |
|--------|--------|--------------------------------------------------------|
|        | dict   | ``{field: column expression}``, for ``mutate(**...)``. |

#### See Also {.doc-section .doc-section-see-also}

- `FunctAIFunc.vectorize`: the whole answer as one column.

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
from dataclasses import dataclass
from dpyr import read, col

@dataclass
class Contact:
    name: str
    city: str | None   # None when the text does not say

@ai
def contact(text: str) -> Contact:
    """The person the text is about."""

people = read([{"text": "Ada Lovelace wrote to us from London."},
               {"text": "Grace Hopper called."}])
people.mutate(**contact.unpack(col.text))
```

### using { #functai.FunctAIFunc.using }

```{.python .no-run}
FunctAIFunc.using(template=_KEEP, **settings)
```

A copy of this function with other settings or another layout.

The copy starts with the same instruction and demos; the original is
untouched. A setting given as None is no longer set by the copy: it
comes from ``configure`` or the defaults. An adapter replaces the
template, and a template replaces the adapter.

#### Parameters {.doc-section .doc-section-parameters}

| Name       | Type   | Description                                                                               | Default   |
|------------|--------|-------------------------------------------------------------------------------------------|-----------|
| template   | list   | A chat template for the copy.                                                             | `_KEEP`   |
| **settings |        | Any setting ``@ai`` takes: ``lm``, ``temperature``, ``adapter``, ``client``, ``tools``... | `{}`      |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description   |
|--------|-------------|---------------|
|        | FunctAIFunc | The copy.     |

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
@ai
def capital(country: str) -> str:
    """The country's capital city."""

capital.using(lm="gpt-4.1-nano")("Chile")
```

### vectorize { #functai.FunctAIFunc.vectorize }

```{.python .no-run}
FunctAIFunc.vectorize(dtype=None, threads=None, errors='raise')
```

This function as a column expression, with options.

Calling an AI function on a dpyr column (``fn(col.text)``) is the same
with the defaults. Each distinct input is sent once; answers are
remembered for the session; the prompt in use now is the one the column
is computed with.

#### Parameters {.doc-section .doc-section-parameters}

| Name    | Type     | Description                                                                                                                  | Default   |
|---------|----------|------------------------------------------------------------------------------------------------------------------------------|-----------|
| dtype   | optional | The column type. Default: the return annotation (text when there is none).                                                   | `None`    |
| threads | int      | How many rows run at once (default 8).                                                                                       | `None`    |
| errors  | str      | ``"raise"`` (default): raise after every row ran; running again retries only the failures. ``"null"``: a failed row is null. | `'raise'` |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type     | Description                                                 |
|--------|----------|-------------------------------------------------------------|
|        | function | Call it on columns: ``fn.vectorize(threads=16)(col.text)``. |

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
from dpyr import read, col

@ai
def capital(country: str) -> str:
    """The country's capital city."""

read([{"country": "Norway"}, {"country": "Ghana"}]).mutate(
    capital=capital.vectorize(threads=2)(col.country))
```