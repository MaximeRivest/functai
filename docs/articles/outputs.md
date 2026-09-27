---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Reasoning and several answers

*Ask the model to think first, return several values, and get everything a call produced with predict.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

A function returns one value, but a model can write more than one thing
before it answers: its reasoning, a draft, a confidence. In functai each
of those is a variable assigned from `_ai`.

## One more output

Assign `_ai` to a variable and it becomes an output with that name; a
comment on the line describes it to the model. Outputs are written in
the order you declare them, and the answer comes last, so an output
declared first is written *before* the answer.

```python
@ai
def solve(question: str) -> float:
    """Solve the word problem."""
    reasoning: str = _ai    # step by step, the calculation that gives the answer
    return _ai

solve("A train travels 120 miles in 2 hours, then 90 miles in 1.5 hours. "
      "What is its average speed in miles per hour?")
```

```output
60.0
```

The call returned only the answer. The reasoning was written, and used by
the model, but not returned.

## Everything: `predict`

`fn.predict(...)` makes the same call and gives a `Prediction` with every
output by name, plus what the call cost:

```python
p = solve.predict("A train travels 120 miles in 2 hours, then 90 miles in 1.5 hours. "
                  "What is its average speed in miles per hour?")
p.reasoning
```

```output
'First, find the total distance traveled by the train:\n120 miles + 90 miles = 210 miles\n\nNext, find the total time taken:\n2 hours + 1.5 hours = 3.5 hours\n\nNow, calculate the average speed by dividing the total distance by the total time:\nAverage speed = Total distance / Total time = 210 miles / 3.5 hours = 60 miles per hour'
```

```python
p.result, p.usage["input_tokens"], p.usage["output_tokens"]
```

```output
(60.0, 98, 103)
```

`dict(p)` gives the outputs as a dictionary, and `p.turn` the whole
exchange (every model and tool step).

## Reasoning without declaring it: `module="cot"`

`@ai(module="cot")` asks for reasoning before the answer without changing
the function. Models with a thinking channel of their own (OpenAI
o-series and GPT-5, Claude 4, Gemini 2.5) use it; the others write a
`reasoning` section first.

```python
@ai(module="cot")
def solve_cot(question: str) -> float:
    """Solve the word problem."""
    ...

solve_cot("If 3 pencils cost $1.20, how much do 10 pencils cost?")
```

```output
4.0
```

## Several values back

To return several values, declare each as an output and return them
together:

```python
@ai
def critique_and_improve(text: str) -> tuple[str, str]:
    """Criticize the text constructively, then improve it."""
    critique: str = _ai     # what is unclear or rude, in one sentence
    return critique, _ai

critique, improved = critique_and_improve("U should fix this asap, it's broken.")
print(critique)
print(improved)
```

```output
The text is too informal and vague, lacking specific details and polite tone which can come across as rude.
The text is too informal and vague, lacking specific details and polite tone which can come across as rude.
```

Or return a record (a dataclass), which names each part; see
[Types](types.md).

Here `critique` is a named output and the bare `_ai` is the answer (the
improved text): a bare `_ai` is always the answer.

## Using an output in the body

The body runs after the model answers, so any output can be checked or
combined in plain Python before it is returned:

```python
from typing import Literal

@ai
def confident_label(text: str) -> str:
    """Is the review positive or negative?"""
    label: Literal["positive", "negative"] = _ai
    confidence: float = _ai   # how sure you are, from 0 to 1
    return label if confidence >= 0.7 else "unsure"

confident_label("It was fine, I guess. Not what I expected.")
```

```output
'negative'
```

## Descriptions in brackets

Where a comment won't fit (a long description, a line already
commented), the description can go in brackets: `critique: str =
_ai["What is unclear or rude, in one sentence."]`. It means exactly the
same thing.
