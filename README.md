# FunctAI

DSPy-powered function decorators that turn typed Python into single-call LLM programs.

Highlights
- Single-call `@magic` functions with `step()`/`final()` markers
- Per-function adapters (`adapter="json" | "chat" | dspy.Adapter`)
- Pass LM by string (`lm="gpt-4.1"`) or LM instance; DSPy resolves providers
- Works with DSPy modules: `Predict`, `ChainOfThought`, `ReAct` (with tools)
- Structured outputs: plain types, lists/dicts, dataclasses, namedtuples
- Batch with `parallel(...)`, compile with `optimize(...)`

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Configure LM

You can let FunctAI/DSPy resolve the provider by passing a model string:

```python
@magic(lm="gpt-4.1")
def echo(text: str) -> str:
    "It returns the input text."
    ...
```

This is equivalent to setting `mod.lm = dspy.LM("gpt-4.1")`. You can also pass a provider-specific LM:

```python
import dspy
@magic(lm=dspy.LM(model="gpt-4.1"))
def echo(text: str) -> str: ...
```

Alternatively, configure globally:

```python
import dspy
dspy.settings.configure(lm=dspy.LM("gpt-4.1"))
```

Ensure provider environment variables are set (e.g., `OPENAI_API_KEY`).

## Quick Start

```python
from functai import magic

@magic(adapter="json")
def classify(text: str) -> str:
    """Return 'positive' or 'negative'."""
    ...

result = classify("This library is amazing!")  # → "positive"
```

## Markers and Lazy Outputs

Use `step()` and `final()` markers to define intermediate and final outputs. FunctAI builds a DSPy signature from your function and makes a single LLM call when a marked value is first used.

```python
from functai import magic, step, final
from typing import List

@magic(adapter="json", lm="gpt-4.1")
def analyze(text: str) -> dict:
    _sentiment: str = step(desc="Determine sentiment")
    _keywords: List[str] = step(desc="Extract keywords")
    summary: dict = final(desc="Combine analysis")
    return summary

res = analyze("FunctAI makes AI programming fun and easy!")
```

Lazy proxies materialize on first use. You can:
- Access directly: `str(res)` or `dict(res)`
- Force materialize: `res.value`
- Get raw DSPy output: `analyze(..., _prediction=True)` (returns `dspy.Prediction`)

Tip: Return eager values by doing `return summary.value`.

Unannotated markers are supported too:

```python
@magic(lm="gpt-4.1")
def fn(x: str) -> str:
    tmp = step(desc="Intermediate")  # type Any
    out = final(desc="Final")         # type Any
    return out
```

You can also return `final(...)` directly without naming a variable. In that case, the output is named `result` by default and typed from your function’s return annotation:

```python
from functai import magic, step, final
from typing import List

@magic(adapter="json", lm="gpt-4.1")
def analyze(text: str) -> dict:
    _sentiment: str = step("Determine sentiment")
    _keywords: List[str] = step("Extract keywords")
    return final("Combine analysis")  # final output name defaults to 'result'

res = analyze("FunctAI makes AI programming fun and easy!")
```

## Structured Outputs

You can return dataclasses or namedtuples. Fields become model outputs and are reconstructed after the call.

```python
from dataclasses import dataclass
from typing import List
from functai import magic, step, final

@dataclass
class Analysis:
    sentiment: str
    confidence: float
    keywords: List[str]

@magic(adapter="json", lm="gpt-4.1")
def analyze_text(text: str) -> Analysis:
    _sentiment: str = step("Determine sentiment")
    _confidence: float = step("Confidence score between 0 and 1")
    _keywords: List[str] = step("Important keywords")
    result: Analysis = final("Complete analysis")
    return result
```

## Choosing Modules and Adapters

`module` accepts a string (`"predict"`, `"cot"`, `"react"`), a `dspy.Module` subclass, or an instance. All `module_kwargs` are forwarded to the module constructor. For ReAct, pass tools via `tools=[...]` or `module_kwargs={"tools": [...]}`.

```python
import dspy
from functai import magic, final

def get_weather(city: str) -> str:
    return "sunny"

@magic(lm="gpt-4.1", module=dspy.ReAct, tools=[get_weather])
def agent(question: str) -> str:
    answer: str = final("Answer the question")
    return answer

@magic(module="cot", lm="gpt-4.1")
def derive(text: str) -> str:
    proof: str = final("Show your reasoning")
    return proof
```

ReAct notes
- ReAct builds two subprograms: an agent (react) and an extractor (extract). Step/Final fields are produced by the extract stage.
- If a module only returns `result`, FunctAI maps your `final()` to that `result` automatically.

Adapters
- `adapter="json"` → `dspy.JSONAdapter()`
- `adapter="chat"` → `dspy.ChatAdapter()`
- Custom adapters are supported (class or instance). A two-step adapter requires `adapter_kwargs`.

## Batch and Optimize

Parallel batch over the underlying module:

```python
from functai import parallel

rows = parallel(classify, inputs=[{"text": "great"}, {"text": "bad"}])
```

Compile with an optimizer (e.g., BootstrapFewShot). The per-function adapter and LM are preserved:

```python
from functai import optimize

trainset = [("I love it", "positive"), ("Terrible UX", "negative")]
compiled_classify = optimize(classify, trainset=trainset)
compiled_classify("So good!")
```

## Prediction Mode

Every `@magic` function accepts `_prediction=True` to return the raw `dspy.Prediction`:

```python
pred = agent("What is the surprise?", _prediction=True)
print(pred.result)       # the final answer
print(pred.reasoning)    # when available (e.g., ReAct extract stage)
print(pred.trajectory)   # full tool-use trace for ReAct
```

## Linting Tips

Some linters (e.g., Ruff F841) flag variables assigned but not used. This is common with `step()` markers that are consumed by the LLM, not Python. Two options:

1) Prefix with underscores (sanitized when building the signature)

```python
_sentiment: str = step("Determine sentiment")
```

2) Mark as used with `use(...)`

```python
from functai import use
sentiment: str = step("Determine sentiment")
use(sentiment)
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Format and lint
uv run ruff format .
uv run ruff check .
```
