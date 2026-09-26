# Verification { #functai.Verification }

```{.python .no-run}
Verification(ok, fresh, problems, log='')
```

What ``verify`` found.

What ``verify`` found. ``ok`` means: a fresh environment built from the
saved requirements alone loaded the program, every AI function rendered
exactly the requests it rendered when saved, and each recording
(``save(record=...)``) produced the same result against its recorded replies.