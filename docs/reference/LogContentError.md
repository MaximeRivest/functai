# LogContentError { #functai.LogContentError }

```{.python .no-run}
LogContentError(field, message)
```

A ``log_content`` map that cannot be honoured (code
``"log-content-field"``): a key that is neither a field name nor ``"*"``,
or, in a program's own settings, a name the program has no field for (a
misspelling would otherwise write the very value it meant to keep out).
``field`` is the key.