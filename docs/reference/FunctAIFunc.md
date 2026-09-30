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

Calling it runs the model and gives the answer, typed as the function's
return type; ``predict`` gives every output; ``acall`` and ``apredict`` are
the same in async code (an ``async def`` AI function is awaited directly).

## Attributes

| Name | Description |
| --- | --- |
| `instructions` | The instruction the model gets: an optimized one, or the one written from the code. |
| `interface` | What a caller gives and gets, as data (contract/programs.md): the |
| `signature` | The lmcc signature: inputs, outputs, instruction. |
| `trials` | What the search that made this copy tried (``GEPA``'s candidates, |
| `version` | The function's version: a fingerprint of what it sends besides its inputs. |

## Methods

| Name | Description |
| --- | --- |
| [acall](#functai.FunctAIFunc.acall) | ``await fn.acall(...)``: the answer, in async code. The call runs in a |
| [apredict](#functai.FunctAIFunc.apredict) | ``await fn.apredict(...)``: ``predict`` in async code. |
| [bake](#functai.FunctAIFunc.bake) | Train weights that answer this function; returns the baked model. |
| [conversation](#functai.FunctAIFunc.conversation) | A conversation with this function: each call a turn that sees the |
| [explain](#functai.FunctAIFunc.explain) | How calls are laid out for the current model: adapter, reader, transports, formats. |
| [freeze](#functai.FunctAIFunc.freeze) | Stop further automatic instruction refinement. |
| [map](#functai.FunctAIFunc.map) | Run on every row of a table, and return the run table. |
| [opt](#functai.FunctAIFunc.opt) | An improved copy: its instruction and worked examples chosen from rows |
| [optimization_runs](#functai.FunctAIFunc.optimization_runs) | How this function was improved, oldest first: the optimizer, the |
| [plan](#functai.FunctAIFunc.plan) | The lmcc plan for the current model: ``.explain()``, ``.describe()``, ``.render(...)``. |
| [predict](#functai.FunctAIFunc.predict) | The call, with everything it produced: every output (``p.result``, |
| [render](#functai.FunctAIFunc.render) | The exact request the next call would send, without sending it. |
| [save](#functai.FunctAIFunc.save) | Write the instruction and demos to a JSON file (``load`` reads it back). |
| [state](#functai.FunctAIFunc.state) | The instruction and demos in use. |
| [stream](#functai.FunctAIFunc.stream) | Call the function and watch the answer being written. |
| [unpack](#functai.FunctAIFunc.unpack) | One column per field of the answer, to spread into a table. |
| [using](#functai.FunctAIFunc.using) | A copy of this function with other settings or another layout. |
| [vectorize](#functai.FunctAIFunc.vectorize) | This function as a column expression, with options. |

### acall { #functai.FunctAIFunc.acall }

```{.python .no-run}
FunctAIFunc.acall(*args, **kwargs)
```

``await fn.acall(...)``: the answer, in async code. The call runs in a
worker thread, so the event loop is free while the model answers.

### apredict { #functai.FunctAIFunc.apredict }

```{.python .no-run}
FunctAIFunc.apredict(*args, **kwargs)
```

``await fn.apredict(...)``: ``predict`` in async code.

### bake { #functai.FunctAIFunc.bake }

```{.python .no-run}
FunctAIFunc.bake(data, **options)
```

Train weights that answer this function; returns the baked model.
``fast = fn.using(lm=baked)`` runs the function on them. See
``functai.bake.bake`` for the options (student, teacher, labels, test, ...).

### conversation { #functai.FunctAIFunc.conversation }

```{.python .no-run}
FunctAIFunc.conversation(
    id=None,
    *,
    store=None,
    context=None,
    earlier_without=(),
    sends='queue',
    **settings,
)
```

A conversation with this function: each call a turn that sees the
earlier ones, kept in ``store``; the function itself is unchanged.

#### Parameters {.doc-section .doc-section-parameters}

| Name            | Type                         | Description                                                                                                                                          | Default   |
|-----------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| id              | str                          | The conversation's id; the same id in the same store opens it again (tomorrow, in another process). Default: a new one.                              | `None`    |
| store           | None, True, folder, or store | None: this process's memory. A folder (or True: the default one) keeps it across runs. Any object with ``append`` and ``read`` (``functai.stores``). | `None`    |
| context         | optional                     | Which earlier turns the model sees: every one (default), or ``functai.last_turns(10)``; ``without=[...]`` leaves bulky inputs out of earlier turns.  | `None`    |
| earlier_without | list of str                  | Outputs this function now writes that earlier turns lack (``reasoning`` after turning on ``module="cot"``).                                          | `()`      |
| sends           | str                          | Two sends at once: ``"queue"`` (default), ``"refuse"`` or ``"branch"``.                                                                              | `'queue'` |
| **settings      | Any                          | Settings for every turn (``approve``, ``lm``...).                                                                                                    | `{}`      |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type         | Description                                                                                                         |
|--------|--------------|---------------------------------------------------------------------------------------------------------------------|
|        | Conversation | Called like the function. ``chat.turns``, ``chat.render(...)``, ``chat.continue_from(turn)``, ``chat.stream(...)``. |

#### Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@ai
def tutor(message: str) -> str:
    """Tutor a student in fractions, one small step at a time."""
    ...

chat = tutor.conversation("alex")
chat("Hi, I'm Alex.")
chat("What is 1/2 + 1/3?")
[t.inputs["message"] for t in chat.turns[-1].saw]
```

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

### map { #functai.FunctAIFunc.map }

```{.python .no-run}
FunctAIFunc.map(data, *, threads=None, num_threads=None, progress=None)
```

Run on every row of a table, and return the run table.

``evaluate`` without the scoring: the rows, the predictions
(``pred_<output>``), and each row's ``error``, ``seconds``, tokens and
``model``. A row that fails keeps its error; the others go on. Needs
``pip install "functai[data]"``.

For long runs, keep replies on disk (``functai.configure(
cache_replies="disk")``): running ``map`` again after an interruption,
or to retry the rows that failed, sends only what has no kept reply.

#### Parameters {.doc-section .doc-section-parameters}

| Name     | Type                     | Description                                                                                                              | Default    |
|----------|--------------------------|--------------------------------------------------------------------------------------------------------------------------|------------|
| data     | list of dict, or a table | Anything ``dpyr.read()`` takes; columns named like the parameters are the inputs.                                        | _required_ |
| threads  | int                      | How many rows run at once (default 1). ``num_threads`` is the same.                                                      | `None`     |
| progress | bool                     | A line on stderr, updated as rows finish: rows done, errors, tokens, time left. Default: on in a terminal or a notebook. | `None`     |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type           | Description   |
|--------|----------------|---------------|
|        | dpyr dataframe |               |

#### See Also {.doc-section .doc-section-see-also}

- `FunctAIFunc.vectorize`: the function as a column expression.

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
@ai
def capital(country: str) -> str:
    """The country's capital city."""
    ...

capital.map([{"country": "Norway"}, {"country": "Ghana"}], threads=2, progress=False)
```

### opt { #functai.FunctAIFunc.opt }

```{.python .no-run}
FunctAIFunc.opt(data=None, *, optimizer=None, metric=None, valset=None, **opts)
```

An improved copy: its instruction and worked examples chosen from rows
with known answers. This function is unchanged.

Only what the function sends besides its inputs changes: the
instruction and the demos. Code, types and layout are never touched.
``functai.labeled_few_shot``, ``functai.bootstrap_few_shot`` and
``functai.gepa`` are the common cases, by name.

#### Parameters {.doc-section .doc-section-parameters}

| Name       | Type                        | Description                                                                                                  | Default    |
|------------|-----------------------------|--------------------------------------------------------------------------------------------------------------|------------|
| data       | list of dict, or a table    | Rows as for ``evaluate``: columns named like the parameters are the inputs, the others the expected outputs. | `None`     |
| expected   | str or dict                 | The column holding the right answers, as for ``evaluate``: ``expected="category"``.                          | _required_ |
| optimizer  | optimizer class or instance | Default ``BootstrapFewShot``. See the Optimizers section.                                                    | `None`     |
| metric     | function or dpyr expression | As for ``evaluate``. Default: exact match on the expected outputs.                                           | `None`     |
| valset     | list of dict, or a table    | Rows for optimizers that choose between candidates.                                                          | `None`     |
| teacher_lm | str                         | A stronger model that runs the examples; its good runs become demos.                                         | _required_ |
| teacher    | AI function                 | Or a teacher function.                                                                                       | _required_ |
| n_synth    | int                         | With a teacher: first write this many training rows.                                                         | _required_ |
| **opts     |                             | Passed to the optimizer.                                                                                     | `{}`       |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description                              |
|--------|-------------|------------------------------------------|
|        | FunctAIFunc | The improved copy, with its own version. |

#### See Also {.doc-section .doc-section-see-also}

- [`evaluate`](evaluate.md): measure before and after.

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
from typing import Literal

@ai
def category(message: str) -> Literal["shipping", "billing", "product"]:
    """The support category of the message."""
    ...

train = [
    {"message": "The vase came smashed.", "result": "shipping"},
    {"message": "Money back please, the chair wobbles.", "result": "billing"},
    {"message": "The handle came off after two uses.", "result": "product"},
]
taught = category.opt(train)
[d.inputs["message"] for d in taught.demos]
```

### optimization_runs { #functai.FunctAIFunc.optimization_runs }

```{.python .no-run}
FunctAIFunc.optimization_runs()
```

How this function was improved, oldest first: the optimizer, the
examples, what changed (and, for a search, its ``trials``).

### plan { #functai.FunctAIFunc.plan }

```{.python .no-run}
FunctAIFunc.plan()
```

The lmcc plan for the current model: ``.explain()``, ``.describe()``, ``.render(...)``.

### predict { #functai.FunctAIFunc.predict }

```{.python .no-run}
FunctAIFunc.predict(*args, **kwargs)
```

The call, with everything it produced: every output (``p.result``,
``p.reasoning``...), the tokens, the model's replies, the call's id.

#### Examples {.doc-section .doc-section-examples}

```{.python .no-run}
@ai
def solve(question: str) -> float:
    """Solve the word problem."""
    reasoning: str = _ai     # step by step
    return _ai

p = solve.predict("3 pencils cost $1.20. How much do 10 cost?")
p.result, p.reasoning
```

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
    ...

request = capital.render("Chile")
print(request.system)
print(request.messages[0].parts[0].text)
```

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
    ...

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
    ...

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
    ...

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
    ...

read([{"country": "Norway"}, {"country": "Ghana"}]).mutate(
    capital=capital.vectorize(threads=2)(col.country))
```