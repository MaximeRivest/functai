# SawError { #functai.SawError }

```{.python .no-run}
SawError(code, call, message)
```

What a call saw cannot be known, or shown again (contract/calls.md,
*Saw*). ``code`` is one of ``not-recorded``, ``missing-call``,
``unknown-key``, ``saw-cycle``, ``not-kept``, ``turn-invalid``; ``call``
is the call whose record says so.