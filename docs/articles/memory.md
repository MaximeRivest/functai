---
rat:
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Memory

*Functions that remember the conversation: stateful=True.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

By default an AI function remembers nothing: each call starts fresh,
which is what you want for extraction, classification, and anything you
evaluate. For a conversation, ask it to remember with `stateful=True`.

```python
@ai(stateful=True)
def chat(message: str) -> str:
    """A friendly assistant. Keep answers short."""

chat("Hello, my name is Alex and I live in Montréal.")
```

```output
'Hello Alex! How can I assist you today?'
```

```python
chat("What's my name, and what's a good thing to do in my city in winter?")
```

```output
'Your name is Alex. In Montréal during winter, a great activity is visiting the Montréal en Lumière festival or enjoying ice skating at Parc La Fontaine.'
```

## What is kept

Each call is kept as one turn in `chat.history`, and the last
`state_window` turns (5 by default) are sent with the next call, written
in the function's own layout.

```python
len(chat.history)
```

```output
2
```

`chat.reset()` forgets everything:

```python
chat.reset()
chat("What's my name?")
```

```output
"I don't know your name yet. What should I call you?"
```

## Settings

- `@ai(stateful=True, state_window=20)` keeps more turns. Every kept turn
  is sent again with each call, so a longer window costs more tokens.
- Memory lives in the function, in this process. It is not saved by
  `fn.save()` or `functai.save()`.
- Memory and tools combine: a stateful function with tools remembers the
  tool results too.
