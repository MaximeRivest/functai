---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Inspecting calls

*See exactly what was sent, what came back, what it cost, and what would be sent, without sending it.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

When an answer surprises you, look at the conversation before changing
anything. Nearly every problem is visible there: an instruction that
says something else than you meant, an input that arrived empty, a
reply in the wrong form.

```python
@ai
def summarize(text: str, focus: str = "key points") -> str:
    """Summarize the text in one sentence, concentrating on the focus."""
    ...

summarize("FunctAI lets developers write typed functions whose body is a model call, "
          "so they can concentrate on logic instead of prompt strings.", focus="benefits")
```

```output
'FunctAI benefits developers by allowing them to write typed functions with model calls as the body, enabling them to focus on logic rather than crafting prompt strings.'
```

## The last calls: `phistory`

```python
print(functai.phistory())
```

```output
[2026-10-02T09:57:59] summarize → gpt-4.1-mini

System message:

Function: summarize

Summarize the text in one sentence, concentrating on the focus.

Reply in exactly this form:
<result>
...
</result>


User message:

<text>
FunctAI lets developers write typed functions whose body is a model call, so they can concentrate on logic instead of prompt strings.
</text>
<focus>
benefits
</focus>


Response:

<result>
FunctAI benefits developers by allowing them to write typed functions with model calls as the body, enabling them to focus on logic rather than crafting prompt strings.
</result>

(finish: stop; tokens in 83, out 38)
```

`phistory(3)` shows the last three calls. `functai.inspect_history(3)`
returns them as objects (lm15's `Request` and `Response`), to inspect in
code.

## Before sending: `render`

`fn.render(...)` builds the exact request the call would send, without
sending it:

```python
request = summarize.render("Some text.")
print(request.system)
```

```output
Function: summarize

Summarize the text in one sentence, concentrating on the focus.

Reply in exactly this form:
<result>
...
</result>
```

## The layout: `explain` and `signature_text`

```python
functai.signature_text(summarize)
```

```output
'Signature: summarize | Doc: Function: summarize\n\nSummarize the text in one sentence, concentrating on the focus. | Inputs: text:str, focus:str | Outputs: result*'
```

```python
print(summarize.explain())
```

```output
adapter: functai_xml
reader: derived
input  text                 kernel-scalar (kernel)
input  focus                kernel-scalar (kernel)
output result               kernel-scalar (kernel)
```

## What a call cost

`fn.predict(...)` returns the tokens, summed over every model call it made
(tool loops included):

```python
p = summarize.predict("Short text.")
p.usage
```

```output
{'input_tokens': 60, 'output_tokens': 14, 'total_tokens': 74, 'cache_read_tokens': 0, 'cache_write_tokens': 0, 'reasoning_tokens': 0}
```

For a whole run, `evaluate` and `fn.map` put the tokens and time of every
row in their table; see [Evaluation](accuracy.md).

## One line per call

`functai.configure(debug=True)` prints a line for each call as it
happens: which function, which model, how long, how many tokens.
