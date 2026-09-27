# module { #functai.module }

`module`

``@module``: a plain Python function that calls @ai functions, optimized as one program.

    @module
    def research(claim: str, hops: int = 2):
        facts = []
        for _ in range(hops):
            query = generate_query(claim, facts)
            facts = append_notes(claim, facts, search(query))
        return facts

    research.opt(trainset=..., metric=...)     # tunes generate_query and append_notes together

The metric sees ``Prediction(result=<what the module returned>)``.

## Classes

| Name | Description |
| --- | --- |
| [FunctAIModule](#functai.module.FunctAIModule) | Callable wrapper for an orchestrator function that calls @ai functions. |

### FunctAIModule { #functai.module.FunctAIModule }

```{.python .no-run}
module.FunctAIModule(fn, *, requires=())
```

Callable wrapper for an orchestrator function that calls @ai functions.

#### Attributes

| Name | Description |
| --- | --- |
| `version` | The module's version: a fingerprint of its code and its AI functions. |

#### Methods

| Name | Description |
| --- | --- |
| [map](#functai.module.FunctAIModule.map) | Run on every row of a table; returns the rows with ``pred_result`` |
| [named_ai_functions](#functai.module.FunctAIModule.named_ai_functions) | Every @ai function this module reaches: called by name, under another |
| [opt](#functai.module.FunctAIModule.opt) | Tune every @ai function this module calls, against one metric on the |
| [save](#functai.module.FunctAIModule.save) | Every AI function's instruction and demos, in one JSON file. |
| [stream](#functai.module.FunctAIModule.stream) | Call the module and watch every AI function it calls, as it works. |
| [vectorize](#functai.module.FunctAIModule.vectorize) | This module as a dpyr row function (see ``FunctAIFunc.vectorize``); |

##### map { #functai.module.FunctAIModule.map }

```{.python .no-run}
module.FunctAIModule.map(data, *, num_threads=1, call_defaults=None)
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
    trainset,
    metric=None,
    optimizer=None,
    call_defaults=None,
    valset=None,
    expected=None,
    **optimizer_kwargs,
)
```

Tune every @ai function this module calls, against one metric on the
module's output. ``call_defaults`` fill module arguments the examples lack.

##### save { #functai.module.FunctAIModule.save }

```{.python .no-run}
module.FunctAIModule.save(path)
```

Every AI function's instruction and demos, in one JSON file.

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
module.module(fn=None, *, requires=())
```

Make a Python function that calls AI functions into one program.

The body is ordinary Python: loops, ifs, helpers, several AI functions.
As a module it can be evaluated, optimized (each AI function inside
learns from the runs the metric accepts), run on a table, and saved as
one program. Use it bare (``@module``) or with requirements.

#### Parameters {.doc-section .doc-section-parameters}

| Name     | Type        | Description                                                                                                | Default   |
|----------|-------------|------------------------------------------------------------------------------------------------------------|-----------|
| requires | list of str | Packages the program needs that functai cannot see from the code (``["numpy>=2"]``), for ``functai.save``. | `()`      |

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

@ai
def shorten(text: str) -> str:
    """The text in at most twelve words."""

@module
def blurb(topic: str) -> str:
    return shorten(draft(topic))

blurb("why paired comparisons need fewer examples")
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
'Paired comparisons simplify decisions, needing fewer examples by focusing on two items.'
```