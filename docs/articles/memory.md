---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Memory

*Conversations: calls that remember each other.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

An AI function remembers nothing: each call starts fresh, which is what
you want for extraction, classification, and anything you evaluate. For
a conversation, open one. The function is unchanged; the memory is the
conversation's.

```python
@ai
def chat(message: str) -> str:
    """A friendly assistant. Keep answers short."""
    ...

alex = chat.conversation("alex")
alex("Hello, my name is Alex and I live in Montréal.")
```

```output
'Hello Alex! How can I assist you today?'
```

```python
alex("What's my name, and what's a good thing to do in my city in winter?")
```

```output
"Your name is Alex. In Montréal during winter, a great activity is visiting the Montréal Botanical Garden's winter light displays or enjoying ice skating at Parc La Fontaine."
```

## What each answer was based on

Every call is a turn. A turn knows which earlier turns it was shown, and
the call log records it (`saw`), so an answer can be asked again later
exactly as it was:

```python
[t.inputs["message"] for t in alex.turns[-1].saw]
```

```output
['Hello, my name is Alex and I live in Montréal.']
```

`render` shows the exact request the next turn would send, and sends
nothing:

```python
print(alex.render("And in summer?").messages[0].parts[0].text)
```

```output
<message>
Hello, my name is Alex and I live in Montréal.
</message>
```

## Trying another path

Nothing is ever deleted. To try another path, continue from an earlier
turn: the new turn is a branch, and both paths are kept.

```python
other = alex.continue_from(alex.turns[0])
other("What's the weather like there in July?")
len(alex.turns), len(other.turns)
```

```output
(2, 2)
```

## Keeping it

Without a store, a conversation lives in this process. With one, the same
line opens it again tomorrow, in another notebook or another process, at
its most recent turn:

```python
import tempfile
folder = tempfile.mkdtemp()

tutor = chat.conversation("alex", store=folder)
tutor("Remember: my favourite colour is green.")

again = chat.conversation("alex", store=folder)     # tomorrow
again("What's my favourite colour?")
```

```output
'Your favourite colour is green.'
```

A store is a folder (files, locked so several processes can share it), or
any object with `append` and `read` (see `functai.stores`).

## What the model sees

Every earlier turn, by default: running out of the model's context and
being told is better than a model silently missing what was said. To send
fewer, keep the last turns, and leave bulky inputs out of earlier ones:

```python
@ai
def reader(document: str, question: str) -> str:
    """Answer the question from the document."""
    ...

qa = reader.conversation(context=functai.last_turns(10, without=["document"]))
```

## Settings

- A turn may use another model: `alex("Why?", lm="claude-sonnet-4-5")`;
  each turn records which model answered.
- Turning reasoning on changes what the function writes: a conversation
  whose earlier turns have no `reasoning` refuses, and says how to go on
  (`earlier_without=["reasoning"]`).
- Two sends at once queue: the second continues from the first
  (`sends="refuse"` or `"branch"` otherwise). The same `request_id` sent
  twice (a double click) is one turn.
- Inside a module, the AI functions it calls remember nothing unless the
  conversation says so: `support.conversation(id, remembers={answer:
  "conversation"})`. `functai.earlier()` is the conversation so far, as
  data, for a helper that takes it as an input.
- A conversation kept in a store keeps what was said. When
  `log_content` says a field may not be kept, a stored conversation
  refuses to start rather than forget.
