# MemoryStore { #functai.MemoryStore }

```{.python .no-run}
MemoryStore()
```

A store kept in this process's memory, by the rules every store keeps
(streaming.md, *The rules a store keeps*): each claim and each append is
one step per log, and an event that does not pass the event schema is
refused ``event-malformed``. A journal for tests, notebooks and one
process; logs are lost when it ends.

``append(event)`` and ``extend(events)`` (a batch, kept whole or not at
all) answer ``"kept"`` or ``"duplicate"``, or raise ``EventRefused``;
``claim(tree)`` gives a later writer its number and the last kept event's
position; ``read(tree, after)`` gives the kept events after a position.

A journal's writer sends what waits as one batch, every event through
``extend`` (a lone one as a batch of one), except to a subclass that
overrides ``append`` and not ``extend`` (a transport, failures for a
test): that one is sent every event alone, through its ``append``.
``batches = False`` also sends every event alone.

## Methods

| Name | Description |
| --- | --- |
| [claim](#functai.MemoryStore.claim) | A later writer claims an unfinished log: it gets a writer number, one |
| [extend](#functai.MemoryStore.extend) | A batch of one log's events, checked each as if appended alone, in |
| [read](#functai.MemoryStore.read) | The kept events of a log after a position (all of them after None). |
| [writer_of](#functai.MemoryStore.writer_of) | The last writer number this store gave for a log (1 before any claim). |

### claim { #functai.MemoryStore.claim }

```{.python .no-run}
MemoryStore.claim(tree)
```

A later writer claims an unfinished log: it gets a writer number, one
more than any given for this log, and the position of the last kept
event, after which it numbers. Every earlier writer is fenced.

### extend { #functai.MemoryStore.extend }

```{.python .no-run}
MemoryStore.extend(events)
```

A batch of one log's events, checked each as if appended alone, in
order, kept whole or not at all: ``"duplicate"`` when every event is
one, ``"kept"`` when every event is kept or a duplicate; else nothing is
kept, and the refusal names the first event refused (``err.event``).

### read { #functai.MemoryStore.read }

```{.python .no-run}
MemoryStore.read(tree, after=None)
```

The kept events of a log after a position (all of them after None).
Never fenced: a fenced writer learns what was kept by reading.

### writer_of { #functai.MemoryStore.writer_of }

```{.python .no-run}
MemoryStore.writer_of(tree)
```

The last writer number this store gave for a log (1 before any claim).