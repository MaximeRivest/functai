# conversations.Turn { #functai.conversations.Turn }

```{.python .no-run}
conversations.Turn(conv, state)
```

One turn of a conversation, as its records say now.

``id`` (also ``call``: the id of the turn's call in the call log),
``parent`` (the turn it continues, or None), ``inputs``, ``outputs``,
``result`` (the answer), ``state`` (``running``, ``waiting``,
``interrupted``, ``done``, ``failed``, ``stopped``, ``abandoned``),
``saw`` (the earlier turns it was shown), ``model``, ``error``,
``waiting`` (the approvals it waits for), ``unfinished`` (tools that may
have run when it stopped), ``usage`` (tokens, summed over the calls
inside it). Another output is an attribute: ``turn.reasoning``.

## Attributes

| Name | Description |
| --- | --- |
| `conversation` | The id of the conversation it belongs to. |
| `error` | How a failed turn failed (``{"type", "code", "message"}``); None otherwise. |
| `id` | The turn's id, which is also the id of its call in the call log (``call``). |
| `inputs` | The turn's inputs, by name, as they were recorded (a copy). |
| `made_by` | A merge: the program that made it (its rating goes there). |
| `model` | The model that answered (or, before the end, the one the turn was asked to use). |
| `outputs` | Every output by name (``result``, ``reasoning``...), once the turn is done; ``{}`` before. |
| `parent` | The id of the turn this one continues (None for a conversation's first turn). |
| `reads` | A merge: the turns it was made from. |
| `request_id` | The ``request_id`` the turn was sent with, if any: the same id sent again is this turn, not a new one. |
| `result` | The answer (as the program's code returned it), typed; None until done. |
| `saw` | The earlier turns this turn was shown, in order. |
| `state` | Where the turn is: ``running``, ``waiting`` (for a person's answer), ``interrupted`` (its |
| `unfinished` | Tools that started and have no result: the process stopped while they ran, so they |
| `usage` | Tokens, summed over every model call inside the turn (a module's helpers and tool loops |
| `waiting` | The tool calls the turn waits for a person to approve (``functai.Approval``); ``[]`` |

## Methods

| Name | Description |
| --- | --- |
| [abandon](#functai.conversations.Turn.abandon) | End a turn that waits or was interrupted, without going on: ``abandoned``. |
| [approve](#functai.conversations.Turn.approve) | Say yes to an approval the turn waits for (``turn.waiting[0]``, its |
| [calls](#functai.conversations.Turn.calls) | The calls inside the turn, as its kept log says: ``{"call", |
| [deny](#functai.conversations.Turn.deny) | Say no: the model is told the person did not allow it (and why). |
| [events](#functai.conversations.Turn.events) | The turn's events from its store (the kept form, or a view made from |
| [find](#functai.conversations.Turn.find) | The calls of ``program`` inside this turn (to rate one: ``functai.rate(turn.find(fn)[0], ...)``). |
| [resume](#functai.conversations.Turn.resume) | Go on with a turn that waits (every approval answered) or was |
| [stop](#functai.conversations.Turn.stop) | Stop the turn wherever it runs: it ends ``stopped`` within a second. |
| [tree](#functai.conversations.Turn.tree) | The calls inside the turn, as an indented tree. |
| [wait](#functai.conversations.Turn.wait) | Wait until the turn is no longer running; returns it as it is then. |

### abandon { #functai.conversations.Turn.abandon }

```{.python .no-run}
conversations.Turn.abandon()
```

End a turn that waits or was interrupted, without going on: ``abandoned``.

### approve { #functai.conversations.Turn.approve }

```{.python .no-run}
conversations.Turn.approve(approval=None, *, by=None, resume=True)
```

Say yes to an approval the turn waits for (``turn.waiting[0]``, its
invocation number, or None for the only one). When nothing else waits,
the turn goes on here (``resume=False``: later, ``turn.resume()``);
returns its result.

### calls { #functai.conversations.Turn.calls }

```{.python .no-run}
conversations.Turn.calls()
```

The calls inside the turn, as its kept log says: ``{"call",
"parent", "function", "invocation", "ended"}`` each, in the order they
started.

### deny { #functai.conversations.Turn.deny }

```{.python .no-run}
conversations.Turn.deny(approval=None, reason=None, *, by=None, resume=True)
```

Say no: the model is told the person did not allow it (and why).

### events { #functai.conversations.Turn.events }

```{.python .no-run}
conversations.Turn.events(after=None, *, view='kept', timeout=None)
```

The turn's events from its store (the kept form, or a view made from
it: ``view="outside"``), after the event named by ``after`` (a
position), those kept so far and then each as it is kept, until its
last. What another process, a page after a reload, reads.

### find { #functai.conversations.Turn.find }

```{.python .no-run}
conversations.Turn.find(program)
```

The calls of ``program`` inside this turn (to rate one: ``functai.rate(turn.find(fn)[0], ...)``).

### resume { #functai.conversations.Turn.resume }

```{.python .no-run}
conversations.Turn.resume(results=None, rerun=())
```

Go on with a turn that waits (every approval answered) or was
interrupted (its process stopped), in this process: its program runs
again with the same inputs and earlier turns, each model reply it had
and each tool result it kept are reused, and it goes on from where it
stopped. A tool that started and has no result may have run:
``results={invocation: output}`` says what it returned, ``rerun=[invocation]``
runs it again. Returns the turn's answer (or raises ``Waiting`` again).

### stop { #functai.conversations.Turn.stop }

```{.python .no-run}
conversations.Turn.stop()
```

Stop the turn wherever it runs: it ends ``stopped`` within a second.

### tree { #functai.conversations.Turn.tree }

```{.python .no-run}
conversations.Turn.tree()
```

The calls inside the turn, as an indented tree.

### wait { #functai.conversations.Turn.wait }

```{.python .no-run}
conversations.Turn.wait(timeout=None)
```

Wait until the turn is no longer running; returns it as it is then.