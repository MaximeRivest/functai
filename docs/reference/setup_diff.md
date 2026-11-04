# setup_diff { #functai.setup_diff }

```{.python .no-run}
setup_diff(a, b)
```

What differs between two setups, step by step: each ``{"program", "field",
"before", "after"}``, ``field`` a dotted path into the setup document
(``model.model``, ``config.temperature``, ``plugins``). ``a`` and ``b``:
programs here, or saved folders (their ``functai.json``).