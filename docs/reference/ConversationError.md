# ConversationError { #functai.ConversationError }

```{.python .no-run}
ConversationError(code, message, *, turn=None)
```

A conversation, or one of its turns, refused what was asked
(contract/conversations.md). ``code`` is one of:

- ``conversation-id``: an id that is not 1 to 200 letters, digits, ``.``,
  ``_`` or ``-``;
- ``conversation-content``: a store that keeps records, for a program a
  ``log_content`` layer drops a field of (the host said never keep it);
- ``conversation-signature``: the program now writes an output the
  earlier turns lack (name it in ``earlier_without``), or its inputs or
  outputs changed in a way earlier turns cannot be shown with;
- ``conversation-opaque``: the program has a field whose values may have
  no JSON form, which a conversation cannot keep as data;
- ``conversation-busy``: ``sends="refuse"``, and a turn is running (or
  one waits for a person's answer);
- ``conversation-nested``: a conversation used inside another one's turn
  that did not declare it (``remembers={chat: "own"}``);
- ``turn-unknown``, ``turn-state`` (the turn cannot do that now),
  ``turn-unfinished`` (resuming needs to know what a tool that may have
  run returned);
- ``store-conflict``: a conditional append found the conversation
  changed (FunctAI reads again and retries).

``turn`` names the turn at fault, when there is one.