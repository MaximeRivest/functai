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

story.stream("a lighthouse keeper's cat").show()
```

```output
Every evening, the lighthouse keeper's cat would perch on the windowsill, watching the waves crash against the rocks below. One stormy night, the cat's sharp eyes noticed a ship struggling in the dark, and it began meowing loudly to alert the keeper. Guided by the cat's urgent calls, the keeper lit the beacon just in time to guide the ship safely to shore. From that day on, the cat was hailed as the lighthouse's silent guardian, forever watching over the sea.
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
(80, 'Every morning, the old kettle on the stove would sing a chee')
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
'Every morning, the old kettle on the stove would sing a cheerful tune as it boiled water. Its melodic whistle brought joy to the entire kitchen, waking everyone with a smile. One day, the family discovered that the kettle’s song changed with the weather, humming softly on rainy days and loudly on sunny ones. From then on, the singing kettle became their magical weather forecast and a beloved morning companion.'
```

`s.result` waits for the end and gives what `story(...)` returns (or
raises what it raises). `s.prediction` is what `all=True` returns, with
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
reasoning: First, find the cost of one pencil by dividing the total cost by the number of pencils: $1.20 ÷ 3 = $0.40 per pencil.  
Then, multiply the cost per pencil by 10 to find the cost of 10 pencils: $0.40 × 10 = $4.00.
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
done
{'reasoning': (54, ['The', ' train', ' leaves', ' at', ' 3', ':']), 'result': (1, ['95'])}
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

assistant.stream("Should I pack a coat for Oslo or for Lima?").show()
```

```output
→ get_weather(city='Oslo')
← Light snow, -2C.
→ get_weather(city='Lima')
← Cloudy, 18C.
You should pack a coat for Oslo, where the weather is light snow and around -2°C. In Lima, the weather is cloudy and much warmer at about 18°C, so a coat is not necessary there.
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
[{'name': 'Ada', 'city': ''}]
[{'name': 'Ada', 'city': 'London'}, {'name': ''}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Ar'}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Arlington'}, {'name': ''}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Arlington'}, {'name': 'Alan', 'city': 'Manchester'}]
[{'name': 'Ada', 'city': 'London'}, {'name': 'Grace', 'city': 'Arlington'}, {'name': 'Alan', 'city': 'Manchester'}, {'name': 'K'}]
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

@ai
def shorten(text: str) -> str:
    """The text in at most twelve words."""

@module
def blurb(topic: str) -> str:
    return shorten(draft(topic))

blurb.stream("tide pools").show()
```

```output
▸ draft
Tide pools are fascinating coastal ecosystems found in the rocky intertidal zones where seawater collects during low tide. These pools create unique habitats that support a diverse array of marine life, including sea stars, anemones, crabs, and small fish. The organisms living in tide pools have adapted to survive the fluctuating conditions of temperature, salinity, and oxygen levels caused by the changing tides. Tide pools offer valuable opportunities for scientific study and environmental education, allowing people to observe marine biodiversity up close and understand the delicate balance of coastal ecosystems.
▸ shorten
Tide pools are unique coastal habitats supporting diverse marine life.
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

## What streams, and what doesn't

- Every provider that lm15 can stream from streams. A model or client that
  can't (a baked model, a judgment-only provider) answers whole, and the
  stream shows the answer in one piece.
- A reply from the reply cache is shown in one piece.
- With `adapter="json"`, each output appears whole at the end of the
  reply: the JSON reader does not yet read a reply in pieces.
