# Outcome { #functai.Outcome }

```{.python .no-run}
Outcome(value=None, error=None)
```

What a call's program did: its ``value``, or its ``error``.

``get()`` gives the value, or raises the error. A ``JournalError``
(``journal-end``) holds the outcome the journal could not confirm.

## Attributes

| Name | Description |
| --- | --- |
| `failed` | Whether the call raised (``error`` is set) rather than returned. |

## Methods

| Name | Description |
| --- | --- |
| [get](#functai.Outcome.get) | The value, or raise the error. |

### get { #functai.Outcome.get }

```{.python .no-run}
Outcome.get()
```

The value, or raise the error.