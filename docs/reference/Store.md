# Store { #functai.Store }

```{.python .no-run}
Store()
```

What a journal needs of a store (streaming.md, *The rules a store keeps*).

``append(event)`` keeps one event and answers ``"kept"`` or
``"duplicate"``, or raises ``EventRefused`` (``event-malformed``,
``event-conflict``, ``event-gap``, ``event-after-end``, ``event-start``).
Any other answer is taken as a refusal; any other exception means no
answer came (the event may have been kept: it is sent again).
``read(tree, after)`` gives the kept events after a position (all of them
after None), or raises ``EventRefused("event-unknown")``.

A store may also have ``extend(events)`` (a batch of one log, kept whole
or not at all), and the writer then sends every event through it, what
waits as one batch, when ``extend`` is defined where ``append`` is or
further down (a subclass that changes only ``append`` is sent every
event alone, through it; one that changes only ``extend``, every event
through it, a lone one as a batch of one); ``batches = True`` or
``False`` says so outright. And ``claim(tree)`` (a later writer's claim, which fences earlier writers;
``JournalError.settle(claim=True)`` needs it). A store times out its own
I/O; a journal also stops waiting at its barriers after its ``timeout``.