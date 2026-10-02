# EventRefused { #functai.EventRefused }

```{.python .no-run}
EventRefused(code, message='', *, event=None)
```

A store's refusal of an append, a claim or a read (contract/streaming.md,
*The rules a store keeps*): ``event-malformed``, ``event-conflict``,
``event-gap``, ``event-after-end``, ``event-start`` or ``event-unknown``.
``event`` is the position of the event refused in a batch, when there is
one. Any other exception a store raises means no answer came.