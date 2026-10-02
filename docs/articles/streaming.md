---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Watch it being written

*Show the answer as the model writes it: text, reasoning, tool calls, records filling in, in a notebook, a script or a web app.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

A long answer takes seconds to write. Calling the function waits for all
of it; `.stream(...)` makes the same call and lets you watch it being
written.

## Print it as it arrives

```python
@ai
def story(topic: str) -> str:
    """A four-sentence story about the topic."""
    ...

story.stream("a lighthouse keeper's cat").show()
```

```output
Every evening, the lighthouse keeper's cat would curl up by the warm lantern, watching the waves crash below. One stormy night, the cat's sharp eyes noticed a ship struggling against the fierce winds. It meowed loudly, alerting the keeper to adjust the light just in time to guide the vessel safely to shore. From that day on, the cat was known as the lighthouse's silent guardian, watching over both sea and land.
```

`.show()` prints the answer as it arrives and waits for the end. In a
notebook, the words appear one after another. To do something else with
each piece, iterate:

```python
pieces = []
for piece in story.stream("a kettle that sings"):
    pieces.append(piece)               # or print(piece, end="", flush=True)
len(pieces), "".join(pieces)[:60]
```

```output
(81, 'Every morning, the old kettle on the stove would sing a chee')
```

## The same call

A stream is not a different kind of call. It asks the model for the same
thing, reads the answer the same way, retries and uses tools the same
way, and ends with the same value:

```python
s = story.stream("a kettle that sings")
s.result
```

```output
'Every morning, the old kettle on the stove would sing a cheerful tune as it boiled water. Its melodic whistle brought joy to the entire kitchen, waking everyone with a smile. One day, the family discovered that the kettle’s song changed with the weather, humming softly on rainy days and loudly on sunny ones. From then on, the singing kettle became their magical weather forecast and a beloved member of the household.'
```

`s.result` waits for the end and gives what `story(...)` returns (or
raises what it raises). `s.prediction` is what `predict` returns, with
the tokens and every message. With the [call log](call-log.md) on, the
call gets the same line, which also records how long the first word
took.

The call starts as soon as `.stream(...)` returns, in the background: you
can start several and read them as they come.

## Reasoning, then the answer

A function that writes a reasoning before its answer shows both.
`.show()` labels them:

```python
@ai
def solve(problem: str) -> float:
    """Solve the word problem."""
    reasoning: str = _ai["Step by step, briefly."]
    return _ai

solve.stream("3 pencils cost $1.20. How much do 10 cost?").show()
```

```output
reasoning: First, find the cost of one pencil by dividing the total cost by the number of pencils: $1.20 ÷ 3 = $0.40 per pencil. Then, multiply the cost per pencil by 10 to find the cost of 10 pencils: $0.40 × 10 = $4.00.
result: 4.00
```

Iterating the stream gives the answer's text only. `s.events()` gives
everything, in order, as small objects with a `kind`:

```python
s = solve.stream("A train leaves at 3:40 and arrives at 5:15. How long is the trip, in minutes?")
pieces = {}
for event in s.events():
    if event.kind == "text":
        pieces.setdefault(event.field, []).append(event.text)
    else:
        print(event.kind)
{field: (len(texts), texts[:6]) for field, texts in pieces.items()}      # how many pieces, the first ones
```

```output
started
request
done
{'reasoning': (49, ['The', ' train', ' leaves', ' at', ' 3', ':']), 'result': (1, ['95'])}
```

| `kind` | what happened |
|---|---|
| `started` | a call began (`inputs`) |
| `text` | a piece of an output (`field`, `text`; `answer` when it is the answer) |
| `thinking` | a piece of a thinking model's own thinking |
| `tool_call`, `tool_result` | the model used a tool, and what the tool said |
| `retry` | the model is asked again (`reason`) |
| `done`, `failed` | a call ended (`value`, or `error`) |

## Tools

Tool calls are shown when the model has finished asking, and their
results when the tool ran:

```python
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"Oslo": "Light snow, -2C.", "Lima": "Cloudy, 18C."}.get(city, "Unknown.")

@ai(tools=[get_weather])
def assistant(question: str) -> str:
    """Answer; use a tool when you need facts."""
    ...

assistant.stream("Should I pack a coat for Oslo or for Lima?").show()
```

```output
→ get_weather(city='Oslo')
← Light snow, -2C.
→ get_weather(city='Lima')
← Cloudy, 18C.
Oslo currently has light snow and a temperature of -2°C, so you should definitely pack a coat for Oslo. Lima, on the other hand, is cloudy with a mild temperature of 18°C, so a coat is not necessary there.
```

## Records and lists, filling in

When the answer is a record or a list, `s.partial` is the answer so far,
read from what has been written: the complete values, and each text as
far as it goes. A number appears only once it is complete, so what you
see is never a wrong number, only an unfinished list:

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    city: str | None   # None when the text does not say

@ai
def people(text: str) -> list[Person]:
    """Everyone the text mentions."""
    ...

s = people.stream("Ada wrote from London to Grace in Arlington; Alan answered from Manchester, and Kurt said nothing.")
seen = []
for piece in s:
    if s.partial and s.partial not in seen:
        seen.append(s.partial)
for view in seen[::3]:
    print(view)
s.result
```

```output
[{}]
[{'name': 'Ada', 'city': 'London'}, {}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Arlington'}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Arlington'}, {'name': 'Alan'}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Arlington'}, {'name': 'Alan', 'city': 'Manchester'}, {}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Arlington'}, {'name': 'Alan', 'city': 'Manchester'}, {'name': 'Kurt', 'city': None}]
[Person(name='Ada', city='London'), Person(name='Grace', city='Arlington'), Person(name='Alan', city='Manchester'), Person(name='Kurt', city=None)]
```

`s.partial` is plain data, provisional; `s.result` is the checked,
typed answer. `s.text` is the answer's text so far, and `s.fields` every
output's.

## When the model is asked again

If a reply cannot be read (a choice outside the allowed ones, a reply cut
off), FunctAI asks the model again, as a normal call does. The stream
shows a `retry` event, and the answer starts again: `s.text` and
`s.partial` restart with it, and `.show()` says so. Pieces already
handed to a `for` loop cannot be taken back, so `"".join(pieces)` is the
text as shown; the answer is `s.result`.

## Modules

A module's stream shows every AI function it calls, as it calls them:

```python
from functai import module

@ai
def draft(topic: str) -> str:
    """A paragraph about the topic."""
    ...

@ai
def shorten(text: str) -> str:
    """The text in at most twelve words."""
    ...

@module
def blurb(topic: str) -> str:
    return shorten(draft(topic))

blurb.stream("tide pools").show()
```

```output
▸ draft
Tide pools are fascinating coastal ecosystems found in the rocky intertidal zones where seawater collects during low tide. These pools serve as temporary habitats for a diverse array of marine life, including sea stars, anemones, crabs, and small fish. The unique conditions of tide pools, such as fluctuating water levels, temperature, and salinity, create a challenging environment that supports specially adapted organisms. Exploring tide pools offers valuable insights into marine biodiversity and the delicate balance of coastal ecosystems.
▸ shorten
Tide pools are diverse, temporary coastal habitats with unique marine life.
```

`s.text_of(shorten)` gives one function's answer as it is written, and
`s.result` what the module returned.

## Stopping

Closing a stream stops the call: the model stops at its next piece, and
the call ends with `functai.Cancelled` (in the call log too). Leaving a
`with` block closes it:

```python
with story.stream("an endless staircase") as s:
    for i, piece in enumerate(s):
        if i == 10:
            break                          # the with block closes it
s
```

```output
<Stream story: closing>
```

Ctrl-C while watching stops it too. A stream you never close runs to its
end, like a call. The provider may still bill the words it wrote before
it stopped.

## In async code, and in a web app

Streams work with `async for`, and `await s` gives the result without
blocking the event loop:

```python
import asyncio

async def main():
    s = story.stream("a kettle that sings")
    n = 0
    async for piece in s:
        n += 1
    return n, (await s)[:40]

asyncio.run(main())        # in Jupyter, where an event loop is already running: await main()
```

```output
(82, 'Every morning, the old kettle on the sto')
```

Each event has `.to_dict()`, plain JSON data, to send to a browser (as
server-sent events, or over a websocket). If the consumer goes away (the
browser tab closes and the server cancels its task), the call stops. The
events are a written contract
([`contract/streaming.md`](https://github.com/maximerivest/functai/blob/master/contract/streaming.md)),
the same for FunctAI in other languages.

## What a caller may see

The events above are the **full** view: everything, values included. Two
narrower views are made from them, event by event:

- **kept**: what may be written down, as the `log_content` settings allow
  (a field kept out of the log is kept out of these events too);
- **outside**: what someone who only sees the program's boundary may see,
  a customer of a [served program](serving.md), say: the program's answer
  and its text as it is written, approvals addressed to them, and the end.
  Never a helper's answer, a tool call or its result, a model's thinking,
  why a request was retried, or an error's message.

A module's answer is usually one of its helpers' answers. `answer_from`
says which, so the outside view shows that text being written, as the
module's own:

```python
from typing import Literal

@ai
def topic(message: str) -> Literal["billing", "shipping", "other"]:
    """What the message is about."""
    ...

@ai
def answer(message: str, topic: str) -> str:
    """Answer the customer in one short sentence."""
    ...

@module(answer_from=answer)
def support(message: str) -> str:
    return answer(message, topic(message))

s = support.stream("Where is my parcel B-2210?")
s.result
full = [(e.kind, e.function) for e in s.events() if e.kind != "text"]
outside = [(e["kind"], e["function"]) for e in s.events(view="outside") if e["kind"] != "text"]
full, outside
```

```output
([('started', 'support'), ('started', 'topic'), ('request', 'topic'), ('done', 'topic'), ('started', 'answer'), ('request', 'answer'), ('done', 'answer'), ('done', 'support')], [('started', 'support'), ('request', 'support'), ('request', 'support'), ('done', 'support')])
```

A view's events are the contract's JSON, ready to send to a browser.

## Every event, kept as it happens

A stream is one call, watched by whoever made it. To see every call a
process makes (a dashboard, an audit trail), give **observers**: each
gets every event of every call tree, in the kept view, as plain dicts.
A list collects them; a function is called with each, in a thread of its
own, so a slow observer never slows a call:

```python
seen = []
with functai.configure(observers=[seen]):
    support("I was charged twice for order B-2210.")
len(seen), sorted({e["kind"] for e in seen})
```

```output
(31, ['done', 'request', 'started', 'text'])
```

`functai.flush()` waits until every function observer has caught up,
and every best-effort journal has written what it was given (at exit,
FunctAI waits for them at most two seconds).

A **journal** keeps whole call trees in a store while they run, so
another process can follow a call, or find what a crashed one did.
`functai.MemoryStore` is a store in this process's memory; any object
that keeps events by the contract's rules is one too (`functai.Store`
says what it must do):

```python
store = functai.MemoryStore()
with functai.configure(journal=store):
    support("The kettle lid doesn't close.")
functai.flush()                          # a best-effort journal writes in the background
tree = store.trees()[-1]
[e["kind"] for e in store.read(tree, None) if e["kind"] != "text"]
```

```output
['started', 'started', 'request', 'done', 'started', 'request', 'done', 'done']
```

A journal is **best effort** by default: if the store fails, FunctAI
warns once and the call goes on. `functai.Journal(store, required=True)`
makes the call wait until the store has confirmed what happened so far,
at three moments: when it starts, before each tool runs, and before it
returns. If the store does not confirm, the call stops before the code or
the tool runs (`JournalError`, `journal-barrier`), or the caller is told
the end was not confirmed (`journal-end`, which still holds the call's
outcome). A program cannot replace or remove the journal its host set
(`journal-policy`).

A reader that follows a log, live or later, is a `functai.Follower`: it
takes events in any order and from any source (duplicates, a writer that
took over after a crash), and keeps each tree's state:

```python
follower = functai.Follower()
for event in store.read(tree, None):
    follower.receive(event)
follower.state(tree)["finished"], [c["fields"] for c in follower.state(tree)["calls"].values()]
```

```output
(True, [{}, {'result': 'other'}, {'result': 'Please check if there is any obstruction or misalignment preventing the lid from closing properly.'}])
```

## What streams, and what doesn't

- Every provider that lm15 can stream from streams. A model or client that
  can't (a baked model, a judgment-only provider) answers whole, and the
  stream shows the answer in one piece.
- A reply from the reply cache is shown in one piece.
- With `adapter="json"`, each output appears whole at the end of the
  reply: the JSON reader does not yet read a reply in pieces.
