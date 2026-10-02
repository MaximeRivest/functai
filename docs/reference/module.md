---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# module { #functai.module }

`module`

``@module``: a plain Python function that calls @ai functions, optimized as one program.

    @module
    def research(claim: str, hops: int = 2) -> list[str]:
        facts = []
        for _ in range(hops):
            query = generate_query(claim, facts)
            facts = append_notes(claim, facts, search(query))
        return facts

    better = research.opt(rows, metric=...)   # a copy with generate_query and append_notes tuned together
    research.interface                        # what it takes and gives, as data (checked on every call)

The metric sees ``Prediction(result=<what the module returned>)``.

## Classes

| Name | Description |
| --- | --- |
| [FunctAIModule](#functai.module.FunctAIModule) | A Python function that calls AI functions, as one program: called, |

### FunctAIModule { #functai.module.FunctAIModule }

```{.python .no-run}
module.FunctAIModule(
    fn,
    *,
    requires=(),
    interface=None,
    outputs=None,
    answer_from=None,
    _namespace=None,
    _output_fields=None,
    **settings,
)
```

A Python function that calls AI functions, as one program: called,
streamed, evaluated, optimized and saved as a whole. Build with ``@module``.

Its ``interface`` (what it takes and gives, as data) is derived and
checked when it is defined, and every call is checked against it: its
inputs before its code runs, its outputs when it returns
(``InterfaceError``, which is also a ``TypeError``).

#### Attributes

| Name | Description |
| --- | --- |
| `interface` | What the module takes and gives, as data (contract/programs.md). |
| `version` | The module's version: a fingerprint of its code and its AI functions. |

#### Methods

| Name | Description |
| --- | --- |
| [conversation](#functai.module.FunctAIModule.conversation) | A conversation with this module: each call a turn, kept in ``store``; |
| [load](#functai.module.FunctAIModule.load) | A copy running with the states a ``save`` wrote. |
| [map](#functai.module.FunctAIModule.map) | Run on every row of a table; returns the rows with ``pred_result`` |
| [named_ai_functions](#functai.module.FunctAIModule.named_ai_functions) | Every @ai function this module reaches: called by name, under another |
| [opt](#functai.module.FunctAIModule.opt) | An improved copy: every @ai function this module calls tuned against |
| [save](#functai.module.FunctAIModule.save) | Every AI function's instruction and demos, in one JSON file. |
| [state](#functai.module.FunctAIModule.state) | The instruction and demos each AI function runs with in this module, by name. |
| [stream](#functai.module.FunctAIModule.stream) | Call the module and watch every AI function it calls, as it works. |
| [vectorize](#functai.module.FunctAIModule.vectorize) | This module as a dpyr row function (see ``FunctAIFunc.vectorize``); |

##### conversation { #functai.module.FunctAIModule.conversation }

```{.python .no-run}
module.FunctAIModule.conversation(
    id=None,
    *,
    store=None,
    context=None,
    remembers=None,
    sends='queue',
    **settings,
)
```

A conversation with this module: each call a turn, kept in ``store``;
``remembers={helper: "conversation"}`` gives a helper its own earlier
calls (helpers remember nothing otherwise); ``functai.earlier()`` in
its code is the conversation so far. See ``functai.conversations.Conversation``.

##### load { #functai.module.FunctAIModule.load }

```{.python .no-run}
module.FunctAIModule.load(path)
```

A copy running with the states a ``save`` wrote.

##### map { #functai.module.FunctAIModule.map }

```{.python .no-run}
module.FunctAIModule.map(
    data,
    *,
    threads=None,
    num_threads=None,
    call_defaults=None,
    progress=None,
)
```

Run on every row of a table; returns the rows with ``pred_result``
(what the module returned) as a dpyr dataframe. See ``FunctAIFunc.map``.

##### named_ai_functions { #functai.module.FunctAIModule.named_ai_functions }

```{.python .no-run}
module.FunctAIModule.named_ai_functions()
```

Every @ai function this module reaches: called by name, under another
name, or through helper functions (looked up when asked, so functions
defined after the module are found). Keys are function names, qualified by
module when two share a name.

##### opt { #functai.module.FunctAIModule.opt }

```{.python .no-run}
module.FunctAIModule.opt(
    data,
    *,
    metric=None,
    optimizer=None,
    call_defaults=None,
    valset=None,
    expected=None,
    **optimizer_kwargs,
)
```

An improved copy: every @ai function this module calls tuned against
one metric on the module's output. This module and its functions are
unchanged. ``call_defaults`` fill module arguments the rows lack.

##### save { #functai.module.FunctAIModule.save }

```{.python .no-run}
module.FunctAIModule.save(path)
```

Every AI function's instruction and demos, in one JSON file.

##### state { #functai.module.FunctAIModule.state }

```{.python .no-run}
module.FunctAIModule.state()
```

The instruction and demos each AI function runs with in this module, by name.

##### stream { #functai.module.FunctAIModule.stream }

```{.python .no-run}
module.FunctAIModule.stream(*args, **kwargs)
```

Call the module and watch every AI function it calls, as it works.

The call starts at once, in the background, and is the same call as
``module(...)``. ``s.events()`` shows each call inside it (started,
its text as it is written, tool calls, retries, done);
``s.text_of(fn)`` one AI function's answer as it is written;
``s.result`` what the module returned (waits). See ``Stream``.

##### vectorize { #functai.module.FunctAIModule.vectorize }

```{.python .no-run}
module.FunctAIModule.vectorize(dtype=None, threads=None, errors='raise')
```

This module as a dpyr row function (see ``FunctAIFunc.vectorize``);
its column type is the module's return annotation, or ``dtype``.

## Functions

| Name | Description |
| --- | --- |
| [module](#functai.module.module) | Make a Python function that calls AI functions into one program. |

### module { #functai.module.module }

```{.python .no-run}
module.module(
    fn=None,
    /,
    *,
    requires=(),
    interface=None,
    outputs=None,
    answer_from=None,
    **settings,
)
```

Make a Python function that calls AI functions into one program.

The body is ordinary Python: loops, ifs, helpers, several AI functions.
As a module it can be evaluated, optimized (each AI function inside
learns from the runs the metric accepts), run on a table, and saved as
one program. Use it bare (``@module``) or with requirements.

Its interface (``blurb.interface``) is derived from the function and
checked when the module is defined (a type its annotations name must be
defined by then), and every call is checked against it: inputs before
the code runs, outputs when it returns (``InterfaceError``, a
``TypeError``: a missing or unknown input names the field). ``Any``,
``object`` or no annotation is an opaque field (any value, never
checked, for data frames and the like); ``functai.JSON`` is any JSON
value; a parameter with a default is optional.

#### Parameters {.doc-section .doc-section-parameters}

| Name        | Type        | Description                                                                                                                                                                        | Default    |
|-------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| requires    | list of str | Packages the program needs that functai cannot see from the code (``["numpy>=2"]``), for ``functai.save``.                                                                         | `()`       |
| outputs     | dict        | Several outputs, by name and type: ``outputs={"team": str, "minutes": int, "result": Reply}``; the code returns a dict of them. The last is the answer.                            | `None`     |
| interface   | dict        | The whole interface as data (contract/programs.md), instead of deriving it; the code is then called with the inputs by keyword.                                                    | `None`     |
| answer_from | AI function | The AI function whose answer, as it is written, is this module's answer: a view that shows only the module's boundary (a served program's caller) shows that text as the module's. | `None`     |
| log_calls   |             | The call log and receiver settings, for this module's calls (as for ``@ai``). ``log_content={"transcript": False}`` keeps an input out of the log.                                 | _required_ |
| log_content |             | The call log and receiver settings, for this module's calls (as for ``@ai``). ``log_content={"transcript": False}`` keeps an input out of the log.                                 | _required_ |
| caller      |             | The call log and receiver settings, for this module's calls (as for ``@ai``). ``log_content={"transcript": False}`` keeps an input out of the log.                                 | _required_ |
| observers   |             | The call log and receiver settings, for this module's calls (as for ``@ai``). ``log_content={"transcript": False}`` keeps an input out of the log.                                 | _required_ |
| journal     |             | The call log and receiver settings, for this module's calls (as for ``@ai``). ``log_content={"transcript": False}`` keeps an input out of the log.                                 | _required_ |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                                                   |
|--------|---------------|-------------------------------------------------------------------------------|
|        | FunctAIModule | Called like the original. The return annotation is the program's output type. |

#### See Also {.doc-section .doc-section-see-also}

- [`evaluate`](evaluate.md): measure the program on rows with known answers.
- [`save`](save.md): save it with everything it depends on.

#### Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@ai
def draft(topic: str) -> str:
    """A two-sentence paragraph about the topic."""
    ...

@ai
def shorten(text: str) -> str:
    """The text in at most twelve words."""
    ...

@module
def blurb(topic: str) -> str:
    return shorten(draft(topic))

blurb("why paired comparisons need fewer examples")
```