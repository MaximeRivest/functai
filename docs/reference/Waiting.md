# Waiting { #functai.Waiting }

```{.python .no-run}
Waiting(message, *, turn=None, approvals=(), later=())
```

A turn stopped to wait (code ``turn-waiting``): for a person's answer
(a tool call its ``approve`` rule asks about, with no function to ask),
or for results that come later (tools that returned ``functai.later``).
``turn`` is the turn (``functai.conversations.Turn``), ``approvals`` the
questions (``turn.approve(...)`` or ``turn.deny(...)``) and ``later`` the
results awaited (each a dict with its ``key``: ``chat.deliver(key,
result)``), from any process that opens the conversation; once nothing
waits, the turn goes on.