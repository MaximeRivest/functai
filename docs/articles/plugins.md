---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Plugins

*Change what programs do, without changing the programs, and keep every change on record.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai
```

A plugin is a few functions FunctAI calls at fixed moments: when a turn
starts, when its earlier turns are chosen, before an AI function is asked,
before a tool runs, after it ran, when a turn ends. Each returns a change
as data, never rewritten messages, so the call's record says what changed,
and an answer someone rated can be asked again exactly as it was.

## A mode

```python
@ai
def assistant(message: str) -> str:
    """You help a student with fractions."""
    ...

brief = functai.Plugin("brief", version="1.0.0")

@brief.before_call
def keep_it_short(call):
    return functai.Change(sections=["Answer in one sentence."])

chat = assistant.conversation("ana", plugins=[brief])
chat("What is a fraction?")
```

```output
'A fraction is a way to represent a part of a whole, written as one number (the numerator) over another number (the denominator).'
```

The function is unchanged: called on its own, it answers as it always did.

## A long conversation, kept short

`functai.compaction` folds older turns into a summary once there are many,
and shows the model the summary and the recent turns. Each branch has its
own summary.

```python
long = assistant.conversation("long", plugins=[functai.compaction(keep=2, every=2)])
for q in ["What is 1/2?", "And 1/3?", "Which is bigger?", "By how much?", "Show me with pizza."]:
    long(q)
print(long.entries("compaction", "summary")[-1]["data"]["text"])
```

```output
The user asked about the fractions 1/2 and 1/3. It was explained that 1/2 represents one part out of two equal parts of a whole and is equivalent to 0.5 in decimal form. For 1/3, it was explained that it represents one part out of three equal parts of a whole and is approximately equal to 0.333 in decimal form. No further questions or open points remain.
```

## Asking first

Any plugin can ask a person before a tool runs. In a conversation the
turn waits, saved, and goes on when someone answers, from any process.

```python
@functai.tool(effects="changes")
def send(to: str, text: str) -> str:
    """Send a message."""
    return f"sent to {to}"

everyone = functai.Plugin("big-sends")

@everyone.tool_call
def ask_for_everyone(tool):
    if tool.input.get("to") == "everyone":
        tool.ask("This goes to everyone.")
```

`approve="changes"` is the same mechanism: the built-in `approval` plugin.

## What is on record

A call's record lists every change (`changes`: which plugin, its version,
the hook, what it changed). Evaluating rated answers shows each its
earlier turns and the summary it was shown; how a host shaped the call (a
mode, a model) comes from the plugins of the process that evaluates, so an
improved instruction is what gets measured.

See `python/examples/plugins/` for six real extensions written as plugins.
