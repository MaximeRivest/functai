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
'Your name is Alex. In Montréal during winter, a great activity is visiting the Montréal en Lumière festival or enjoying ice skating at Parc La Fontaine.'
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

## Several answers, then one

Branches can be tried side by side, by different models, then joined:
`merge` asks another AI function to make one answer from them, and
records it as a turn of its own, which the next turn sees.

```python
@ai
def tutor(message: str) -> str:
    """Tutor a student in fractions. Two short sentences at most."""
    ...

@ai
def combine(answers: list[dict]) -> str:
    """Combine these tutors' answers into one: the clearest explanation, two sentences at most."""
    ...

lesson = tutor.conversation("lesson")
lesson("Why is 1/3 bigger than 1/4?")
start = lesson.turns[0]
tries = [lesson.continue_from(start).predict("Explain it with a picture in words.", lm=m).call_id
         for m in ("gpt-4.1-mini", "gpt-4.1-nano", "claude-haiku-4-5")]
merged = lesson.continue_from(start).merge(tries, combine)
merged.result
```

```output
'Imagine a pizza cut into 3 equal slices and the same pizza cut into 4 equal slices; each slice from the 3-slice pizza (1/3) is bigger than each slice from the 4-slice pizza (1/4). So, 1/3 of the pizza is larger than 1/4 of the pizza.'
```

```python
lesson.continue_from(merged)("Say it in one sentence.")
```

```output
'1/3 is bigger than 1/4 because dividing something into fewer parts makes each part larger.'
```

`merged.reads` names the branches it was made from, and `merged.made_by`
the function that made it: rating the merged answer rates `combine`.
`combine` is given the branches' answers (with their model and inputs)
when it has one input; name its inputs to give it something else.

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

A folder is a `functai.FolderStore`: one file per conversation, appended
to under a lock, so several processes (a web server's workers, a notebook)
can share it, and written to disk before each step returns. Without a
store, a conversation lives in this process's memory
(`functai.MemoryConversations`). Any object with two methods is a store
too, a database table say: `append(conversation, records, *, expect=None)`
adds records at the end, all or none (only if the conversation holds
`expect` records, when given), and `read(conversation, after=0)` gives
the records after a position, in order. `help(functai.stores)` has the
whole protocol.

## Stopping a turn

A turn can be stopped from anywhere: this process, or another one that
opened the same store (a "stop" button on a web page). It ends `stopped`,
and its stream raises `functai.Cancelled`:

```python
import threading, time

@ai
def story(topic: str) -> str:
    """A long story, at least 800 words."""
    ...

stories = story.conversation("stories")
s = stories.stream("a lighthouse keeper")
threading.Timer(2, lambda: stories.stop(s.turn)).start()
try:
    s.result
except functai.Cancelled:
    pass
stories.turns[-1].state
```

```output
'stopped'
```

A turn is `running`, `waiting` (for a person to approve a tool call: see
[Tools that change things](tools.md#tools-that-change-things)), `done`,
`failed`, `stopped`, `abandoned`, or `interrupted` when its process died
without ending it ([it can be resumed](tools.md#when-the-process-dies)).
Each turn also keeps its `inputs`, `outputs`, `model`, `usage` (tokens,
summed over every call inside it) and `error`.

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
- A conversation kept in a store keeps what was said. When
  `log_content` says a field may not be kept, a stored conversation
  refuses to start rather than forget.

## Helpers inside a module

A [module](modules.md)'s conversation remembers the module's turns. The
AI functions it calls (its helpers) still start fresh at every call,
unless the conversation says which ones remember: a classifier should
judge each message alone, while the one that writes the reply should see
its earlier replies.

```python
from typing import Literal
from functai import module

@ai
def topic(message: str) -> Literal["billing", "shipping", "other"]:
    """What the customer is writing about."""
    ...

@ai
def reply(message: str, topic: str) -> str:
    """Answer the customer in one short sentence, using what they told you earlier."""
    ...

@module
def support(message: str) -> str:
    return reply(message, topic(message))

desk = support.conversation("lee", remembers={reply: "conversation"})
desk("Hi, I'm Lee. My parcel B-2210 is late.")
desk("Which parcel was I asking about?")
```

```output
'You were asking about parcel B-2210.'
```

`reply` saw its earlier call; `topic` did not. `"turn"` remembers only
the calls made earlier in the same turn (a helper called in a loop), and
`functai.remember("conversation", steps=True)` also shows the tool calls
of those earlier calls.

A helper that needs the whole conversation as data, a hand-off summary
for a person, say, takes it as an input: `functai.earlier()` is the
conversation so far, one row per earlier turn.

```python
@ai
def handoff(conversation: list[dict[str, str]]) -> str:
    """Summarize this support conversation in one sentence, for the person who takes it over."""
    ...

@module
def support_desk(message: str) -> str:
    if "person" in message.lower():
        return handoff(functai.earlier())
    return reply(message, topic(message))

lee = support_desk.conversation("lee-2")
lee("My parcel B-2210 is late.")
lee("I'd like to talk to a person.")
```

```output
'Customer reports that parcel B-2210 is late and needs an update on its delivery status.'
```
