---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# keep_pictures { #functai.keep_pictures }

```{.python .no-run}
keep_pictures(last=3, *, at_once=1, name='keep-pictures')
```

Show a long tool loop only its newest pictures: before each step, the
pictures tool results gave earlier than the newest ``last`` are replaced
by a short text (the results' own text stays). A browsing agent that
takes a screenshot at every click otherwise sends every screenshot again
with every request.

``at_once``: drop old pictures only once this many are due, all
together. A provider's prompt cache keeps the start of a request only
while it stays the same, and every drop changes it: with caching on,
``at_once=5`` drops five pictures every fifth step instead of one at
each. The turn and the call log keep every picture as the tool gave it;
each request in the call log holds what it sent.

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
agent = browse.using(plugins=[functai.keep_pictures(3)], max_steps=200)
```