---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# ai { #functai.ai }

```{.python .no-run}
ai(_fn=None, /, **cfg)
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

| Name        | Type                | Description                                                                                                                                                                                                                                                                                                                                                                                            | Default    |
|-------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| lm          | str                 | The model: ``"gpt-4.1-mini"``, ``"claude-haiku-4-5"``, ``"groq:openai/gpt-oss-120b"``, ``"claude:claude-sonnet-4-5"`` (a subscription)... Default: the one set with ``configure``.                                                                                                                                                                                                                     | _required_ |
| temperature | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                                                                                                                                                                                                                                              | _required_ |
| max_tokens  | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                                                                                                                                                                                                                                              | _required_ |
| seed        | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                                                                                                                                                                                                                                              | _required_ |
| top_p       | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                                                                                                                                                                                                                                              | _required_ |
| stop        | optional            | Sampling settings; any lm15 ``Config`` field is accepted.                                                                                                                                                                                                                                                                                                                                              | _required_ |
| module      | str                 | ``"predict"`` (default), ``"cot"`` (reasoning before the answer: the model's thinking channel when it has one), or ``"react"``.                                                                                                                                                                                                                                                                        | _required_ |
| tools       | list of functions   | Typed Python functions the model may call. A call then runs the tool loop: at most ``max_steps`` model calls (default 8).                                                                                                                                                                                                                                                                              | _required_ |
| approve     | function or rule    | Ask before a tool runs (``functai.tool(effects=...)`` says what each does): a function given each ``Approval`` (``True``, ``False``, or a reason to refuse), or a rule (``"changes"``, ``"all"``, a list of tool names). See ``functai.tool``.                                                                                                                                                         | _required_ |
| adapter     | str or lmcc.Adapter | The prompt layout: ``"xml"`` (default), ``"chat"``, ``"json"``, or an lmcc adapter.                                                                                                                                                                                                                                                                                                                    | _required_ |
| template    | list                | A chat template, ``[system(...), turns(), user(...)]``: write the conversation yourself. Replaces ``adapter``.                                                                                                                                                                                                                                                                                         | _required_ |
| examples    | list                | Worked examples shown before the question: pairs ``("input", "output")`` or rows ``{"text": ..., "result": ...}``.                                                                                                                                                                                                                                                                                     | _required_ |
| retries     | int                 | How many times an unreadable reply is asked again (default 1).                                                                                                                                                                                                                                                                                                                                         | _required_ |
| api_retries | int                 | How many times a provider error is re-sent (default 3).                                                                                                                                                                                                                                                                                                                                                | _required_ |
| log_calls   | bool or folder      | Keep this function's calls in the call log (see ``functai.calls``); ``False`` keeps them out, whatever ``configure`` says.                                                                                                                                                                                                                                                                             | _required_ |
| log_content | bool or dict        | ``False``: log only sizes, times and tokens, never the values (for a function that sees secrets). ``{"transcript": False}``: every value but that input's; ``{"*": False, "question": True}``: only the question's. It only removes: a host's ``configure`` or block that drops a value wins over the function's own ``True``. A name the function has no field for is an error (``LogContentError``). | _required_ |
| observers   | list                | Functions (or lists) given each event of this function's calls as they happen, in the form a log keeps (``functai.eventlog``), beside the host's observers.                                                                                                                                                                                                                                            | _required_ |
| journal     | store or Journal    | Where the call tree's events are kept while it runs (a ``functai.MemoryStore``, or ``functai.Journal(store, required=True)``); only where the host sets none.                                                                                                                                                                                                                                          | _required_ |
| **settings  |                     | Any other setting ``configure`` takes (``api_key``, ``client``, ``cache_replies``, ``teacher``, ``optimizer``, ``debug``...). An unknown setting is an error.                                                                                                                                                                                                                                          | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description                                                                                                                                                                                                               |
|--------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | FunctAIFunc | The AI function. Call it like the original; ``fn.predict(...)`` returns a ``Prediction`` with every output and the tokens used; ``await fn.acall(...)`` in async code (an ``async def`` AI function is awaited directly). |

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
    ...

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

p = solve.predict("3 pencils cost $1.20. How much do 10 cost?")
p.result, p.reasoning
```

```output
(4.0, 'First, find the cost of one pencil by dividing the total cost by the number of pencils: $1.20 ÷ 3 = $0.40 per pencil.\n\nNext, find the cost of 10 pencils by multiplying the cost per pencil by 10: $0.40 × 10 = $4.00.')
```

Settings in the decorator:

```python
@ai(lm="gpt-4.1-nano", temperature=0)
def headline(article: str) -> str:
    """A headline of at most eight words."""
    ...

headline("The council voted to turn the old rail yard into a park with a pool.")
```

```output
'Council Converts Rail Yard into Park with Pool'
```