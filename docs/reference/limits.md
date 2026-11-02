---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# limits { #functai.limits }

```{.python .no-run}
limits(
    tokens=None,
    cost=None,
    seconds=None,
    prices=None,
    then='answer',
    name='limits',
)
```

Stop a call tree (an agent, a pipeline) when it has used enough: tokens,
money or time. Nothing is limited by default; this plugin is opt-in.

Counted over the whole tree a call starts (every AI function inside a
module, every tool's own calls), from the requests a provider answered
in this process (replies from the reply cache, and those a resumed turn
reuses, cost nothing now and are not counted). Checked before each AI
function's call and before each step of a tool loop, so one request may
go past a limit; the next does not start.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type                   | Description                                                                                                                                                                                                                                                                                                                                    | Default    |
|---------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| tokens  | int                    | Every token read and written (``total_tokens``), summed.                                                                                                                                                                                                                                                                                       | `None`     |
| cost    | float                  | The list price of the requests, from ``prices`` (needed with it). Exact or nothing: a request to a model with no price, or whose counts do not settle its price, counts as reaching the limit.                                                                                                                                                 | `None`     |
| seconds | float                  | Since the tree's outermost call started.                                                                                                                                                                                                                                                                                                       | `None`     |
| prices  | dict                   | A model's name (as you ask for it) → an lm15 ``InferencePricing`` (or a dict of its fields: ``{"input_per_million": 3, "output_per_million": 15, "cache_read_per_million": 0.3, "cache_write_per_million": 3.75}``). FunctAI ships no price list: prices change, and a wrong price would mislead.                                              | `None`     |
| then    | \'answer\' or \'stop\' | ``"answer"`` (the default): a tool loop that reaches a limit is offered no more tools and told to answer with what it has; so is a loop at its last step (``max_steps``), instead of failing with ``StepLimit``. ``"stop"``: the call stops. Either way a call that would start past a limit stops (``PluginError``, code ``plugin-stopped``). | `'answer'` |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
from lm15.models import InferencePricing
prices = {"claude-opus-5-5": InferencePricing(input_per_million=5, output_per_million=25)}
with functai.configure(plugins=[functai.limits(cost=3.0, seconds=600, prices=prices)]):
    report = concept_test(url)
```