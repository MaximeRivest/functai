# Report { #functai.Report }

```{.python .no-run}
Report(entry, nodes, bindings, requirements, problems, packages, declared)
```

What ``functai.check`` found: everything a program depends on, and what
would stop a clean save.

Printed, it is a readable summary. ``ok`` (also its truth value) is True
when nothing is an error; ``errors`` and ``warnings`` list each
``Problem``; ``requirements`` maps each package the program needs to
its version; ``nodes`` are the programs reached (``"module:name"``).

## Attributes

| Name | Description |
| --- | --- |
| `errors` | The problems that stop ``save`` (``Refused``) unless allowed. |
| `ok` | Whether the program can be saved cleanly (no errors). |
| `warnings` | The problems worth knowing that do not stop ``save``. |