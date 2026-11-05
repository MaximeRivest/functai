# as_case { #functai.as_case }

```{.python .no-run}
as_case(call, *, rating=None, share=0.2, salt='', folder=None, records=None)
```

A logged call as a row to add to a program's examples.

The call's inputs; its answer when it was right, or the correction of a
``wrong`` rating (given, or the latest one of the call's); and its
``split``, held out by the rule every language shares (``split_of``).

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                      |
|--------|--------|--------------------------------------------------|
|        | dict   | A row: inputs, the answer when known, ``split``. |