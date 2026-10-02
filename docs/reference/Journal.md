# Journal { #functai.Journal }

```{.python .no-run}
Journal(store, *, required=False, retries=2, backoff=0.05, timeout=30.0)
```

Where a call tree's kept log is written while it runs: a store, and how
surely (``configure(journal=Journal(store, required=True))``).

Best effort (the default) never makes a call wait: if the store fails,
FunctAI warns once and the log is kept as far as it could be. A required
journal makes the call wait until every event so far is confirmed at
three barriers: when it starts, before each tool runs, and before it
returns. If the start or a tool is not confirmed, the call stops with
``JournalError`` (``journal-barrier``); if its end is not, the caller gets
``JournalError`` (``journal-end``) holding what the call did.

``retries``: how many times an append is sent again, in a row, when no
answer came, waiting ``backoff`` seconds before the first resend and
twice as long before each next one. ``timeout``: the longest a barrier
waits for an answer; past it, no answer came (``"unknown"``: the store
may still keep what it was sent). A store should also time out its own
I/O. See ``Store`` for what a store answers.

A bare store is a best-effort journal: ``configure(journal=store)``.
Two journals are the same when their store and mode are; when a program
names its host's journal again, the host's timing applies.