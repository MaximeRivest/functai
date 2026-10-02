# conversations.Conversation { #functai.conversations.Conversation }

```{.python .no-run}
conversations.Conversation(
    program,
    id=None,
    *,
    store=None,
    context=None,
    earlier_without=(),
    remembers=None,
    sends='queue',
    _head=_FOLLOW,
    **settings,
)
```

A program's conversation: its turns, kept in a store, called like the
program. Made by ``fn.conversation(...)`` (an AI function or a module).

Called with the program's inputs, it makes one turn and returns its
answer; ``predict`` gives the whole call, ``stream`` watches it. Other
keyword arguments that are settings, not inputs, apply to that turn
only: ``chat("Why?", lm="claude-sonnet-4-5")``. ``request_id=`` makes a
send that is repeated (a double click) one turn.

## Parameters {.doc-section .doc-section-parameters}

| Name            | Type                         | Description                                                                                                                                                                                                                                | Default    |
|-----------------|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| program         | AI function or module        | What answers.                                                                                                                                                                                                                              | _required_ |
| id              | str                          | The conversation's id (letters, digits, ``.``, ``_``, ``-``). The same id in the same store opens the same conversation. Default: a new one.                                                                                               | `None`     |
| store           | None, True, folder, or store | Where it is kept: None (this process's memory), a folder, True (the default folder), or any object with ``append`` and ``read`` (``functai.stores``).                                                                                      | `None`     |
| context         | Context                      | Which earlier turns are shown: ``functai.all_turns()`` (default) or ``functai.last_turns(10)``, each with ``without=[...]``.                                                                                                               | `None`     |
| earlier_without | list of str                  | Outputs the program now writes that earlier turns lack (reasoning turned on, a first tool): earlier turns are shown without them.                                                                                                          | `()`       |
| remembers       | dict                         | A module's helpers' memory: ``{answer: "conversation"}``, ``"turn"``, ``functai.remember("conversation", steps=True)``, or ``"own"`` for a conversation used inside it. Helpers remember nothing otherwise.                                | `None`     |
| sends           | str                          | Two sends at once: ``"queue"`` (default: the second waits and continues from the first), ``"refuse"`` (``ConversationError`` ``conversation-busy``), or ``"branch"`` (the second continues from the last finished turn, beside the first). | `'queue'`  |
| **settings      | Any                          | Settings for every turn (``approve``, ``lm``, ...).                                                                                                                                                                                        | `{}`       |

## Attributes

| Name | Description |
| --- | --- |
| `head` | The turn this conversation continues from next (None: none yet). |
| `turns` | The turns from the first to the head, in order. |

## Methods

| Name | Description |
| --- | --- |
| [all_turns](#functai.conversations.Conversation.all_turns) | Every turn of every branch, in the order they were made. |
| [continue_from](#functai.conversations.Conversation.continue_from) | This conversation, continuing after ``turn`` (a Turn, its id, or its |
| [merge](#functai.conversations.Conversation.merge) | A turn after this conversation's head made from several branches by |
| [predict](#functai.conversations.Conversation.predict) | One turn of an AI function: the whole call (``p.turn`` is the lmcc |
| [render](#functai.conversations.Conversation.render) | The exact request the next turn would send (nothing is sent or |
| [stop](#functai.conversations.Conversation.stop) | Stop a running turn, wherever it runs. |
| [stream](#functai.conversations.Conversation.stream) | One turn, watched while it is made (a ``Stream``, with ``.turn``, |
| [turn](#functai.conversations.Conversation.turn) | One turn, by its id (or a Turn). |

### all_turns { #functai.conversations.Conversation.all_turns }

```{.python .no-run}
conversations.Conversation.all_turns()
```

Every turn of every branch, in the order they were made.

### continue_from { #functai.conversations.Conversation.continue_from }

```{.python .no-run}
conversations.Conversation.continue_from(turn)
```

This conversation, continuing after ``turn`` (a Turn, its id, or its
index in ``turns``): the next turn is a new branch. Nothing is deleted.

### merge { #functai.conversations.Conversation.merge }

```{.python .no-run}
conversations.Conversation.merge(branches, fn, **inputs)
```

A turn after this conversation's head made from several branches by
another AI function (``fn``): its answer becomes this program's
answer, recorded as a turn (``made_by`` fn, ``reads`` the branches), so
the next turn sees it. ``inputs``: ``fn``'s, by name; when ``fn`` has
one input and none is given, it is given the branches' answers
(``[{"model", <inputs>…, <outputs>…}]``).

### predict { #functai.conversations.Conversation.predict }

```{.python .no-run}
conversations.Conversation.predict(*args, request_id=None, **kwargs)
```

One turn of an AI function: the whole call (``p.turn`` is the lmcc
turn; ``p.call_id`` the turn's id).

### render { #functai.conversations.Conversation.render }

```{.python .no-run}
conversations.Conversation.render(*args, call=None, **kwargs)
```

The exact request the next turn would send (nothing is sent or
recorded). ``call=helper``: the request that helper would get, with
the memory the conversation gives it (inputs: the helper's).

### stop { #functai.conversations.Conversation.stop }

```{.python .no-run}
conversations.Conversation.stop(turn)
```

Stop a running turn, wherever it runs.

### stream { #functai.conversations.Conversation.stream }

```{.python .no-run}
conversations.Conversation.stream(*args, request_id=None, **kwargs)
```

One turn, watched while it is made (a ``Stream``, with ``.turn``,
known at once: the turn is saved before the model is asked).

### turn { #functai.conversations.Conversation.turn }

```{.python .no-run}
conversations.Conversation.turn(turn)
```

One turn, by its id (or a Turn).