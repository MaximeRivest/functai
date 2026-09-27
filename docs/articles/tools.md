---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Tools

*Give an AI function plain Python functions to call: lookups, searches, calculations.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

A model only knows what is in its prompt. To let it look things up or
compute exactly, give it **tools**: ordinary typed Python functions. When
a function has tools, a call becomes a loop. The model asks for a tool,
functai runs it and gives back the result, and so on until the model
answers.

## A first tool

A tool is a typed function with a docstring. Its name, parameters and
docstring are what the model reads, exactly as for an AI function.

```python
ORDERS = {
    "A-1042": {"item": "kettle", "status": "stuck at carrier", "shipped": "2026-09-02"},
    "B-2210": {"item": "toaster", "status": "delivered"},
}

def lookup_order(order_id: str) -> dict:
    """The order's item and shipping status."""
    return ORDERS.get(order_id, {"error": f"no order {order_id}"})

@ai(tools=[lookup_order])
def answer(question: str) -> str:
    """Answer the customer's question. Look the order up first."""
    ...

answer("Where is my order A-1042?")
```

```output
'Your order A-1042, which is a kettle, is currently stuck at the carrier. It was shipped on September 2, 2026.'
```

`phistory()` shows the whole loop: the tool call, its result, the answer.

```python
print(functai.phistory())
```

```output
[2026-09-26T17:39:19] answer → gpt-4.1-mini

System message:

Function: answer

Answer the customer's question. Look the order up first.

Reply in exactly this form:
<result>
...
</result>


User message:

<question>
Where is my order A-1042?
</question>


Assistant message:

[tool call lookup_order({"order_id": "A-1042"})]

Tool message:

[tool result call_PQpBupN0krbUb3NnQles5Qxz] {"item": "kettle", "status": "stuck at carrier", "shipped": "2026-09-02"}

Tools: lookup_order

Response:

<result>
Your order A-1042, which is a kettle, is currently stuck at the carrier. It was shipped on September 2, 2026.
</result>

(finish: stop; tokens in 139, out 39)
```

## Several tools

```python
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression, like '(15 * 23) + 10'."""
    return eval(expression, {"__builtins__": {}})   # a demo: never eval untrusted text

def today() -> str:
    """Today's date, as YYYY-MM-DD."""
    return "2026-09-26"

@ai(tools=[lookup_order, calculate, today])
def assistant(question: str) -> str:
    """Answer the question, using the tools for facts and arithmetic."""
    ...

assistant("How many days ago was order A-1042 shipped?")
```

```output
'Order A-1042 was shipped 24 days ago.'
```

## How the loop behaves

- **Native or text.** Models with native tool calling use it; others get
  the same tools as text, in the same layout. Adding a tool never changes
  how the rest of the prompt is written.
- **At most `max_steps` model calls** (8 by default). A loop that runs out
  raises `functai.StepLimit`: `@ai(tools=[...], max_steps=20)` to allow
  more.
- **A tool that raises** is reported to the model, which can try again
  differently. `@ai(tool_errors="raise")` stops the call instead.
- **All of it is recorded**: `fn.predict(...)` gives the whole exchange in
  `p.turn`, and the tokens of every step are added up in `p.usage`.

```python
p = answer.predict("Has my toaster, order B-2210, been delivered?")
p.result, p.usage["input_tokens"]
```

```output
('Yes, your toaster from order B-2210 has been delivered.', 220)
```

## Tools and optimization

A tool loop is a normal AI function: it can be [evaluated](accuracy.md)
and [optimized](improving.md). The worked examples an optimizer picks
keep their tool calls, so the model sees how a good run used the tools.
