---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# quotes_found { #functai.quotes_found }

```{.python .no-run}
quotes_found(text, quotes)
```

Whether each quote is in the text, word for word.

White space, case, curly and straight quotes, dashes and a quote's own
surrounding quotation marks and final punctuation do not count; any
other difference does (a changed word, a paraphrase, an invented
sentence). Deterministic, and costs nothing.

## Parameters {.doc-section .doc-section-parameters}

| Name   | Type               | Description                             | Default    |
|--------|--------------------|-----------------------------------------|------------|
| text   | str                | The source the quotes should come from. | _required_ |
| quotes | str or list of str | One quote, or several.                  | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                 | Description                                                           |
|--------|----------------------|-----------------------------------------------------------------------|
|        | bool or list of bool | For one quote, whether it is found; for a list, one answer per quote. |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
source = "The parcel left Leeds on Monday. It was delayed by snow."
functai.quotes_found(source, ["“It was delayed by snow”", "It was lost"])
```