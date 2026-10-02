# inspect_history { #functai.inspect_history }

```{.python .no-run}
inspect_history(n=1)
```

The last ``n`` requests FunctAI sent (or answered from its cache), oldest first.

Each is a record with ``function``, ``model``, ``request`` and
``response`` (lm15's objects: exactly what went to the provider and what
came back; ``response`` is None when the provider raised), ``cached``,
``error`` and ``timestamp``. The last 500 are kept, in this process only;
``phistory()`` prints them as readable text. For every call, kept on
disk: ``configure(log_calls=True)`` and ``functai.calls()``.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type   | Description                         | Default   |
|--------|--------|-------------------------------------|-----------|
| n      | int    | How many (default 1: the last one). | `1`       |

## Returns {.doc-section .doc-section-returns}

| Name   | Type               | Description   |
|--------|--------------------|---------------|
|        | list of CallRecord |               |