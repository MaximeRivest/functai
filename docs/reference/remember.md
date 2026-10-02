---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# remember { #functai.remember }

```{.python .no-run}
remember(mode='conversation', *, steps=False)
```

What an AI function called inside a module's conversation remembers.

In a module's conversation, the module's turns remember each other, but
the AI functions it calls (its helpers) start fresh at every call unless
the conversation says otherwise, in ``remembers={helper: ...}``. A
plain ``"conversation"`` or ``"turn"`` there is the same as
``remember("conversation")`` or ``remember("turn")``; ``remember`` is
for ``steps``.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type                         | Description                                                                                                                                                             | Default          |
|--------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| mode   | \'conversation\' or \'turn\' | ``"conversation"``: the helper is shown its own earlier calls on this branch, in earlier turns and in this one. ``"turn"``: only its earlier calls in the current turn. | `'conversation'` |
| steps  | bool                         | Also show the tool calls and results of those earlier calls (by default only their inputs and answers).                                                                 | `False`          |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                           |
|--------|--------|-------------------------------------------------------|
|        | Memory | For ``module.conversation(remembers={helper: ...})``. |

## See Also {.doc-section .doc-section-see-also}

- [`earlier`](earlier.md): the conversation so far, as data a helper takes as an input.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: part of a program (the Memory guide runs a whole one)
@module
def support(message: str) -> str:
    return answer(message, topic(message))

# answer sees its own earlier answers; topic remembers nothing
chat = support.conversation("ana", remembers={answer: functai.remember("conversation", steps=True)})
```