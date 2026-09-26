# Problem { #functai.Problem }

```{.python .no-run}
Problem(code, where, message, fix, severity='error')
```

One thing that keeps a program from being saved cleanly.

``code`` is stable (match on it, or pass it to ``save(allow=...)``);
``where`` names the offender as ``module:name``; ``fix`` says what to do.