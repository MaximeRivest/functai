# Refused { #functai.Refused }

```{.python .no-run}
Refused(report)
```

``functai.save`` found problems that keep the program from being saved
cleanly, and saved nothing.

``err.report`` is the whole ``Report`` (``err.report.errors`` lists
each ``Problem``, with what to do). Fix them, or, for a deliberate
exception, ``save(..., allow=[code, ...])`` records it in the folder.