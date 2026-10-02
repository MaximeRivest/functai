# ServeError { #functai.ServeError }

```{.python .no-run}
ServeError(code, message)
```

A program cannot be served as asked (code ``serve-opaque``: an input or
output with no JSON form cannot cross HTTP; ``serve-keys``: listening
beyond this machine with no keys).