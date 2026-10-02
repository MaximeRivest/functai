# Waiting { #functai.Waiting }

```{.python .no-run}
Waiting(message, *, turn=None, approvals=())
```

A turn stopped to wait for a person's answer (code ``turn-waiting``):
a tool call its ``approve`` rule asks about, with no function to ask.
``turn`` is the turn (``functai.conversations.Turn``) and ``approvals``
what waits: ``turn.approve(...)`` or ``turn.deny(...)``, from any process
that opens the conversation, answers and resumes it.