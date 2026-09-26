# configure { #functai.configure }

```{.python .no-run}
configure(**overrides)
```

Set defaults for every AI function: the model, sampling, layout, and more.

Called plainly, the settings apply to the whole program. Used in a
``with`` block, they apply inside the block only, in this thread and in
the threads functai starts from it (``evaluate(num_threads=8)``).

Settings are looked up at every call, most specific first:
``fn.using(...)``, then the function's own (``@ai(...)``), then a
``with configure(...)`` block, then ``configure(...)``.

## Parameters {.doc-section .doc-section-parameters}

| Name       | Type   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                 | Default    |
|------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| **settings |        | Any setting ``@ai`` takes: ``lm``, ``temperature``, ``max_tokens``, ``api_key``, ``base_url``, ``auth``, ``client``, ``adapter``, ``module``, ``tools``, ``max_steps``, ``stateful``, ``retries``, ``api_retries``, ``cache_replies``, ``teacher_lm``, ``debug``... An unknown setting raises ``TypeError``. The call log: ``log_calls`` (``True``, or a folder: keep every call), ``log_content`` and ``caller`` (who is calling, a dict). | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type      | Description                                                                |
|--------|-----------|----------------------------------------------------------------------------|
|        | configure | Usable as a context manager, to undo the settings at the end of the block. |

## See Also {.doc-section .doc-section-see-also}

- [`ai`](ai.md): settings for one function.
- [`FunctAIFunc.using`](FunctAIFunc.md#functai.FunctAIFunc.using): a copy of one function with other settings.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
functai.configure(lm="gpt-4.1-mini", temperature=0)
functai.settings.lm
```

```output
'gpt-4.1-mini'
```

For one block only:

```python
@ai
def capital(country: str) -> str:
    """The country's capital city."""

with functai.configure(lm="gpt-4.1-nano"):
    print(capital("Canada"))
```

```output
Ottawa
```