# ReachError { #functai.ReachError }

```{.python .no-run}
ReachError(message, missing)
```

A program needs to reach what the wall it runs behind does not let
through (code ``"needs-unmet"``: contract/calls.md, *Reach*), refused
before its first call. ``missing`` is ``{"domains": [...], "secrets":
[...]}``, what it needs and is not let through.