# flush { #functai.flush }

```{.python .no-run}
flush(timeout=5.0)
```

Wait (at most ``timeout`` seconds; None: for ever) until every observer
has been given the events made so far, and every best-effort journal has
tried to keep them. True when all of it was done in time.

Observers that are lists get each event as it is made, so they are
complete when a call returns; a function observer runs in a thread of
its own, and may still be catching up.