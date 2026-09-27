# Stream { #functai.Stream }

```{.python .no-run}
Stream(program, args, kwargs)
```

One call of an AI function or a module, watched while it is made.

Made by ``fn.stream(...)``; the call starts at once, in the background.
It is the same call as ``fn(...)``: the same retries, tools and call log
line, and the same value in the end.

Iterate it (``for piece in s``, or ``async for``) for the answer's text
as the model writes it; ``s.events()`` for everything that happens,
in this call and every call inside it; ``s.show()`` to print it as it
is written. ``s.result`` waits for the call and returns (or raises)
what ``fn(...)`` would; ``await s`` does the same in async code.
Iterating twice replays from the start.

When the model is asked again (an unreadable reply, a provider error,
an escalation), the pieces already shown cannot be taken back: the
next ones are the new answer, and a ``Retry`` event says so. So
``"".join(s)`` is the text as shown; the answer is ``s.result``, and
``s.text`` is always the answer so far.

Closing the stream (``s.close()``, or the end of a ``with`` block)
cancels the call if it is still running: ``s.result`` then raises
``Cancelled``. A stream never closed runs to its end.

## Attributes {.doc-section .doc-section-attributes}

| Name    | Type     | Description                                                                                                                                                                                             |
|---------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text    | str      | The answer's text so far (in the latest request to the model: it starts again when the model is asked again).                                                                                           |
| fields  | dict     | Every output's text so far, by name.                                                                                                                                                                    |
| partial | optional | The answer so far as a value: the text for a text answer, the JSON read so far for a record or a list (provisional: complete values, and strings as far as written), None for other answers until done. |
| done    | bool     | Whether the call has ended.                                                                                                                                                                             |
| call_id | str      | The call's id in the call log (waits for the call to start).                                                                                                                                            |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
@ai
def haiku(topic: str) -> str:
    """A haiku about the topic."""

for piece in haiku.stream("autumn rain"):
    print(piece, end="", flush=True)
```

```output
functai: no model chosen, so using gpt-4.1-mini (environment ($OPENAI_API_KEY)). Choose one with functai.configure(lm=...).
Soft autumn rain falls,  
Whispering through amber leaves,  
Nature’s gentle breath.
```

## Methods

| Name | Description |
| --- | --- |
| [close](#functai.Stream.close) | Stop: cancel the call if it is still running (at the model's next |
| [events](#functai.Stream.events) | Every event of the call and of the calls inside it, in order: |
| [show](#functai.Stream.show) | Print the call as it is written, and wait for its end. |
| [text_of](#functai.Stream.text_of) | The answer's text of every call of ``fn`` inside this stream, as it |
| [wait](#functai.Stream.wait) | Wait for the call to end (at most ``timeout`` seconds); returns the stream. |

### close { #functai.Stream.close }

```{.python .no-run}
Stream.close()
```

Stop: cancel the call if it is still running (at the model's next
piece of text; the provider may bill what it already generated).

### events { #functai.Stream.events }

```{.python .no-run}
Stream.events()
```

Every event of the call and of the calls inside it, in order:
``Started``, ``Text``, ``Thinking``, ``ToolCall``, ``ToolResult``,
``Retry``, ``Done``, ``Failed`` (``functai.streaming``). Works with
``for`` and ``async for``; each event has ``.kind`` and ``.to_dict()``.

### show { #functai.Stream.show }

```{.python .no-run}
Stream.show(file=None)
```

Print the call as it is written, and wait for its end.

The answer's text as it arrives; when the function writes several
outputs (a reasoning, then the answer), each is labelled; tool calls
and their results get a line each; a retry says why. For a module,
each AI function it calls, with its answer. In a notebook, the text
appears as it is written. Raises what the call raised.

### text_of { #functai.Stream.text_of }

```{.python .no-run}
Stream.text_of(fn)
```

The answer's text of every call of ``fn`` inside this stream, as it
is written (for a module's stream: ``s.text_of(summarize)``).

### wait { #functai.Stream.wait }

```{.python .no-run}
Stream.wait(timeout=None)
```

Wait for the call to end (at most ``timeout`` seconds); returns the stream.