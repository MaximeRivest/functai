---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Prompt formats and chat templates

*How values are written into the prompt and read back, and how to write the conversation yourself.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

You don't need this page to use functai: the default format works for
every model. Read it when you want the prompt to look a particular way,
because a model was trained on a format, or because you are bringing an
existing prompt over (then start with [From a prompt you already
have](from-a-prompt.md)).

## The default layout

The instruction and the form of the reply go in the system message; the
inputs go in tags in the user message; earlier turns (worked examples,
memory) go as messages in between.

```python
@ai
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""

summarize("Foundation models are now mature enough to be used in real applications, "
          "provided they are measured like any other component.")
print(functai.phistory())
```

```output
[2026-09-26T17:38:05] summarize → gpt-4.1-mini

System message:

Function: summarize

Summarize the text in one sentence.

Reply in exactly this form:
<result>
...
</result>


User message:

<text>
Foundation models are now mature enough to be used in real applications, provided they are measured like any other component.
</text>


Response:

<result>
Foundation models have reached a level of maturity suitable for real applications, as long as they are evaluated like any other component.
</result>

(finish: stop; tokens in 65, out 31)
```

The same form that is shown to the model is used to read its reply, so
the prompt and the parser can never drift apart.

## Shipped layouts

| `adapter=` | layout |
|---|---|
| `None` or `"xml"` | tagged sections (above) |
| `"chat"` | `[[ ## name ## ]]` sections, ending with `[[ ## completed ## ]]` |
| `"json"` | one JSON object, enforced by the provider (models with structured output) |
| an `lmcc.Adapter` | any lmcc layout, including one loaded from a JSON file |

```python
summarize.using(adapter="chat")("Short prompts are cheaper, but clear ones are better.")
print(functai.phistory())
```

```output
[2026-09-26T17:38:06] summarize → gpt-4.1-mini

System message:

Function: summarize

Summarize the text in one sentence.

Respond with the corresponding output fields, each under its header, then end with [[ ## completed ## ]]:

[[ ## result ## ]]
...

[[ ## completed ## ]]

User message:

[[ ## text ## ]]
Short prompts are cheaper, but clear ones are better.



Response:

[[ ## result ## ]]
Clear prompts are more effective than short ones, despite the higher cost of short prompts.

[[ ## completed ## ]]

(finish: stop; tokens in 72, out 28)
```

## Writing the conversation yourself

A chat template lists the messages, with the function's values in
braces: `{instruction}`, and each input by name.

```python
from functai import system, user, turns, assistant

@ai(template=[
    system("You are a helpful pirate. {instruction}"),
    user("Text: {text}"),
])
def pirate_summary(text: str) -> str:
    """Summarize in ten words."""

pirate_summary("Foundation models are now mature enough to be used in real-world applications.")
```

```output
'Foundation models mature, enabling practical use in real-world applications.'
```

With one output and no reply form in the template, the whole reply is the
value. With several outputs, spell the form in the template; **the same
form is the parser**:

```python
@ai(template=[
    system("{instruction}\n\nAnswer in this form:\n"
           "{% for f in outputs %}{f.name}: {f.value}\n{% endfor %}"),
    turns(),
    user("Review: {review}"),
])
def rate(review: str) -> int:
    """Rate the review from 1 to 5 stars."""
    verdict: str = _ai["One short sentence."]
    return _ai

dict(rate("Great tacos, loud music. I'll be back.", all=True))
```

```output
{'verdict': 'Positive and concise review with a clear intention to return.', 'result': 4}
```

```python
print(functai.phistory())
```

```output
[2026-09-26T17:38:08] rate → gpt-4.1-mini

System message:

Function: rate

Rate the review from 1 to 5 stars.

Output guidance:
- verdict: One short sentence.

Answer in this form:
verdict: ...
result: (integer)


User message:

Review: Great tacos, loud music. I'll be back.

Response:

verdict: Positive and concise review with a clear intention to return.  
result: 4

(finish: stop; tokens in 62, out 20)
```

## Template details

- `turns()` marks where worked examples and earlier conversation go.
  Without it they go right before the last user message.
- `{% if context %}…{% endif %}` shows a block only when an input has a
  value; `{% for f in inputs %}` and `{% for f in outputs %}` loop over
  the fields.
- A last `assistant("<answer>")` is a **prefill**: sent to models that can
  continue it, and read as the start of the reply either way.
- OpenAI-style dictionaries work too:
  `template=[{"role": "system", "content": "..."}, ...]`.
- A template that can't be read back (several outputs and no reply form)
  is refused when the function is defined, before any model is called.

Templates use lmcc's template language; its
[documentation](https://github.com/MaximeRivest/lmcc) covers the rest.
