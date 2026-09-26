# parse_docstring { #functai.parse_docstring }

```{.python .no-run}
parse_docstring(sym)
```

Split a numpy-style docstring into its parts.


Parse a subset of numpy-style docstrings:

##   Parameters {.doc-section .doc-section---parameters}

  name : type
      description...

##   Returns {.doc-section .doc-section---returns}

  type
      description...
Returns dict with 'param:<name>' and 'return' keys when found.