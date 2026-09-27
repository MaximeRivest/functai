# Cancelled { #functai.Cancelled }

```{.python .no-run}
Cancelled()
```

The stream was closed before its call ended.

A ``BaseException``, like ``asyncio.CancelledError`` and for the same
reason: code inside the call (a tool, a module's ``except Exception``)
must not catch it and carry on.