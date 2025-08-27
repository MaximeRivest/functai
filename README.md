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

@magic(adapter="json")
def analyze(text: str) -> dict:
    sentiment: str = step(desc="Determine sentiment")
    keywords: List[str] = step(desc="Extract keywords")
    summary: dict = final(desc="Combine analysis")
    return summary
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