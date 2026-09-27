---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# How a function becomes a prompt

*Each part of a Python function becomes part of the prompt. Learn the mapping once and you can predict what the model sees.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

An AI function is an ordinary Python function with `@ai` on top. You never
write a prompt: every part of the function already says something, and
functai turns each part into the matching part of the prompt. Learn this
mapping once and you can predict what the model sees from any function.

| part of the function | becomes | |
|---|---|---|
| its name | the task's name | `def triage(...)` |
| the docstring | the instruction | `"""Triage a support message."""` |
| each parameter | an input, with its name and type | `message: str` |
| the return type | the output, and the type the reply is read back into | `-> Ticket` |
| a comment on a parameter, a field or the return line | guidance for that value | `message: str,  # as the customer wrote it` |
| a variable assigned from `_ai` | one more output, written before the answer | `reasoning: str = _ai   # think first` |
| the rest of the body | ordinary Python, run on the model's answer | `return score.clamp(0, 1)` |
| `@ai(...)` options | how it runs: model, layout, tools, memory | `@ai(lm="claude-haiku-4-5")` |

## The smallest AI function

A name, typed parameters, a docstring and a return type are enough. The
body can be empty: a docstring alone, `...`, or `return _ai` all mean
"the model's answer is the return value".

```python
@ai
def sentiment(text: str) -> str:
    """Is the text 'positive', 'negative' or 'neutral'?"""
    ...

sentiment("The update broke my favourite feature.")
```

```output
'negative'
```

`phistory()` shows exactly what was sent and what came back:

```python
print(functai.phistory())
```

```output
[2026-09-26T17:36:40] sentiment → gpt-4.1-mini

System message:

Function: sentiment

Is the text 'positive', 'negative' or 'neutral'?

Reply in exactly this form:
<result>
...
</result>


User message:

<text>
The update broke my favourite feature.
</text>


Response:

<result>
negative
</result>

(finish: stop; tokens in 55, out 9)
```

The docstring is the instruction, the parameter became a tagged input,
and the return became an output the reply must fill (`result`, since a
return value has no name of its own).

## Names are part of the prompt

The model reads your names, so name things the way you would for a
colleague. Compare the function above with this one:

```python
@ai
def f(x: str) -> str:
    """Classify."""
    ...

f("The update broke my favourite feature.")
```

```output
'Complaint'
```

It still answers, but it had to guess what you meant. A good name and a
good docstring are the cheapest improvement you will ever make.

## Types are the output contract

The return type says what shape the answer must have, and the reply is
read back into that type. Ask for a `Literal` and you get one of its
values, never a sentence around it:

```python
from typing import Literal

@ai
def sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """The sentiment of the text."""
    ...

sentiment("The update broke my favourite feature.")
```

```output
'negative'
```

Every common type works, as input and as output: numbers, lists,
dictionaries, `Enum`, `Optional`, dataclasses, pydantic models. See
[Types](types.md).

## Comments are guidance

A comment on a parameter or on the return line becomes guidance for that
value. It is the natural place for units, formats and edge cases:

```python
@ai
def translate(
    text: str,        # English, informal
    register: str,    # "formal" or "casual"
) -> str:             # French, about the same length as the input
    """Translate the text to French."""

translate("hey, can u send me the file asap?", register="formal")
```

```output
"Bonjour, pourriez-vous m'envoyer le fichier dès que possible, s'il vous plaît ?"
```

The same works on the fields of a class; see [Types](types.md).

## `_ai` is the model's answer

Inside the body, `_ai` stands for what the model will answer. Use it like
the value it stands for, and plain Python runs on the answer before it's
returned:

```python
@ai
def sentiment_score(text: str) -> float:
    """How positive the text is, from 0.0 (very negative) to 1.0 (very positive)."""
    return max(0.0, min(1.0, _ai))

sentiment_score("Honestly the best purchase I made this year.")
```

```output
0.95
```

Whatever the model says, the score stays between 0 and 1.

Assign `_ai` to a variable and the variable becomes **one more output**,
named after it and written *before* the answer. A comment on that line
describes it. That's how you ask a model to think first:

```python
@ai
def is_urgent(message: str) -> bool:
    """Does this message need a reply within the hour?"""
    reasoning: str = _ai     # which words or facts show how urgent it is
    return _ai

p = is_urgent.predict("Our whole team is locked out and we have a demo in 30 minutes.")
p.reasoning, p.result
```

```output
('The message states that the whole team is locked out and there is a demo scheduled in 30 minutes. This indicates an immediate problem that could impact an important event happening very soon, requiring urgent assistance.', True)
```

What the model saw, and what it answered:

```python
print(functai.phistory())
```

```output
[2026-09-26T17:36:44] is_urgent → gpt-4.1-mini

System message:

Function: is_urgent

Does this message need a reply within the hour?

Output guidance:
- reasoning: which words or facts show how urgent it is

Reply in exactly this form:
<reasoning>
...
</reasoning>
<result>
(boolean)
</result>


User message:

<message>
Our whole team is locked out and we have a demo in 30 minutes.
</message>


Response:

<reasoning>
The message states that the whole team is locked out and there is a demo scheduled in 30 minutes. This indicates an immediate problem that could impact an important event happening very soon, requiring urgent assistance.
</reasoning>
<result>
true
</result>

(finish: stop; tokens in 88, out 57)
```

`fn.predict(...)` returns every output, not just the return value. The rule is
simple: **a bare `_ai` is always the answer; `name = _ai` is another
output.** (The description can also go in brackets,
`_ai["which words show urgency"]`, where a comment won't fit.) More in
[Reasoning and several answers](outputs.md).

## Options say how it runs

What the function *means* lives in its code. How it *runs* goes in
`@ai(...)`: which model, which layout, which tools, whether it remembers
the conversation.

```python
@ai(lm="gpt-4.1-nano", temperature=0)
def headline(article: str) -> str:
    """A headline of at most eight words."""
    ...

headline("The city council voted on Tuesday to turn the old rail yard into a park with a public pool.")
```

```output
'City Council Approves Old Rail Yard Park Plan'
```

Settings can also be given for the whole program with `configure()`, for
a block with `with configure(...):`, or for one copy with
`fn.using(...)`. See [Models and settings](models.md).

## What happens when you call it

1. functai reads the function once, when it is decorated: inputs,
   outputs, their types and comments, the instruction.
2. At each call, [lmcc](https://github.com/MaximeRivest/lmcc) lays the
   call out as messages for the model you chose (tags by default).
3. [lm15](https://github.com/lm15-dev/lm15-python) sends it to the provider.
4. lmcc reads the reply back into your types. A slightly misspelled reply
   is repaired; an unreadable one is asked again once.
5. The rest of your body runs on the answer, and its return value is
   what you get.

To see step 2 without sending anything, ask for the request itself:

```python
request = sentiment.render("Great tacos, loud music.")
print("--- system")
print(request.system)
for message in request.messages:
    print(f"--- {message.role}")
    print(message.parts[0].text)
```

```output
--- system
Function: sentiment

The sentiment of the text.

Reply in exactly this form:
<result>
one of: positive, negative, neutral
</result>

--- user
<text>
Great tacos, loud music.
</text>
```
