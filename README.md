# FunctAI

DSPy-powered function decorators for AI-enhanced programming.

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

```python
from functai import magic

@magic(adapter="json")
def classify(text: str) -> str:
    """Return 'positive' or 'negative'."""
    ...

result = classify("This library is amazing!")
# Returns: "positive"
```

## Advanced Usage

Use `step()` and `final()` markers for complex multi-step operations:

```python
from functai import magic, step, final
from typing import List

@magic(adapter="json", lm="gpt-4.1")  # pass model string; DSPy resolves provider
def analyze(text: str) -> dict:
    # Tip: prefix with '_' to avoid linter warnings; underscores are stripped from model fields
    _sentiment: str = step(desc="Determine sentiment")
    _keywords: List[str] = step(desc="Extract keywords")
    summary: dict = final(desc="Combine analysis")
    return summary

result = analyze("FunctAI makes AI programming fun and easy!")
# Returns: {'main_idea': 'FunctAI simplifies and makes AI programming enjoyable.', 'tone': 'Enthusiastic and positive', 'focus': 'Ease and enjoyment of AI programming with FunctAI'}
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Format code
uv run ruff format .

# Lint
uv run ruff check .
```

## Examples

See the `examples/` directory for more usage examples.

## Linting Tips

Some linters (e.g., Ruff F841) flag variables assigned but not used. This is common with `step()` markers that are consumed by the LLM, not Python. You have two clean options:

1) Prefix with underscores (sanitized in signature)

```python
from functai import magic, step, final
from typing import List

@magic(adapter="json", lm="gpt-4.1")
def analyze(text: str) -> dict:
    _sentiment: str = step(desc="Determine sentiment")
    _keywords: List[str] = step(desc="Extract keywords")
    summary: dict = final(desc="Combine analysis")
    return summary
```

Notes:
- Leading underscores are stripped when building the DSPy signature, so model fields remain `sentiment` and `keywords`.

2) Mark as used with `use(...)`

```python
from functai import magic, step, final, use
from typing import List

@magic(adapter="json", lm="gpt-4.1")
def analyze(text: str) -> dict:
    sentiment: str = step(desc="Determine sentiment")
    keywords: List[str] = step(desc="Extract keywords")
    use(sentiment, keywords)  # mark as used for linters
    summary: dict = final(desc="Combine analysis")
    return summary
```
