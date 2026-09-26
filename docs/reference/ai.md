# ai { #functai.ai }

```{.python .no-run}
ai(_fn=None, **cfg)
```

Turn a typed Python function into an AI function.

The function's parts are the prompt: its name is the task, the docstring
the instruction, the parameters the inputs, the return type the output
(and the type the reply is read back into). Comments on parameters,
fields and the return line are guidance. A body that is only a
docstring, ``...`` or ``return _ai`` means "the model's answer is the
return value"; otherwise ``_ai`` stands for the model's answer inside
the body. Use it bare (``@ai``) or with settings (``@ai(lm=...)``).

## Parameters {.doc-section .doc-section-parameters}

| Name        | Type                | Description                                                                                                                                                                        | Default    |
|-------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| lm          | str                 | The model: ``"gpt-4.1-mini"``, ``"claude-haiku-4-5"``, ``"groq:openai/gpt-oss-120b"``, ``"claude:claude-sonnet-4-5"`` (a subscription)... Default: the one set with ``configure``. | _required_ |
| temperature | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                          | _required_ |
| max_tokens  | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                          | _required_ |
| seed        | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                          | _required_ |
| top_p       | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                          | _required_ |
| stop        | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                          | _required_ |
| module      | str                 | ``"predict"`` (default), ``"cot"`` (reasoning before the answer: the model's thinking channel when it has one), or ``"react"``.                                                    | _required_ |
| tools       | list of functions   | Typed Python functions the model may call. A call then runs the tool loop: at most ``max_steps`` model calls (default 8).                                                          | _required_ |
| stateful    | bool                | Remember the conversation between calls (the last ``state_window`` turns, default 5).                                                                                              | _required_ |
| adapter     | str or lmcc.Adapter | The prompt layout: ``"xml"`` (default), ``"chat"``, ``"json"``, or an lmcc adapter.                                                                                                | _required_ |
| template    | list                | A chat template, ``[system(...), turns(), user(...)]``: write the conversation yourself. Replaces ``adapter``.                                                                     | _required_ |
| examples    | list                | Worked examples shown before the question: pairs ``("input", "output")`` or rows ``{"text": ..., "result": ...}``.                                                                 | _required_ |
| retries     | int                 | How many times an unreadable reply is asked again (default 1).                                                                                                                     | _required_ |
| api_retries | int                 | How many times a provider error is re-sent (default 3).                                                                                                                            | _required_ |
| **settings  |                     | Any other setting ``configure`` takes (``api_key``, ``client``, ``cache_replies``, ``teacher``, ``optimizer``, ``debug``...). An unknown setting is an error.                      | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description                                                                                                              |
|--------|-------------|--------------------------------------------------------------------------------------------------------------------------|
|        | FunctAIFunc | The AI function. Call it like the original; ``all=True`` returns a ``Prediction`` with every output and the tokens used. |

## See Also {.doc-section .doc-section-see-also}

- [`configure`](configure.md): settings for every function at once.
- [`module`](module.md): a Python function that calls several AI functions, as one program.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@ai
def sentiment(text: str) -> str:
    """Is the text 'positive', 'negative' or 'neutral'?"""

sentiment("The update broke my favourite feature.")
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
'negative'
```

``_ai`` in the body: ``reasoning: str = _ai`` is one more output, written
before the answer (its comment describes it); a bare ``_ai`` is the
answer, and plain Python runs on it.

```python
@ai
def solve(question: str) -> float:
    """Solve the word problem."""
    reasoning: str = _ai     # step by step, the calculation
    return round(_ai, 2)

p = solve("3 pencils cost $1.20. How much do 10 cost?", all=True)
p.result, p.reasoning
```

```output
(4.0, 'First, find the cost of one pencil by dividing the total cost by the number of pencils:\n$1.20 ÷ 3 = $0.40 per pencil.\n\nNext, find the cost of 10 pencils by multiplying the cost per pencil by 10:\n$0.40 × 10 = $4.00.')
```

Settings in the decorator:

```python
@ai(lm="gpt-4.1-nano", temperature=0)
def headline(article: str) -> str:
    """A headline of at most eight words."""

headline("The council voted to turn the old rail yard into a park with a pool.")
```

```output
'Council Approves Rail Yard Turned Park with Pool'
```