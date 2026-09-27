# functai

**Write a Python function. A language model does the work. You measure how well.**

functai turns a typed Python function into a call to a language model.
The function's name, docstring and types say what you want; the answer
comes back as the type you asked for. Then you run it on a whole table,
find out how often it is right, and make it better.

```python
from typing import Literal
from dpyr import col
import functai
from functai import ai

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""

team("I was charged twice for order B-2210, please fix this.")    # 'billing'

tickets = functai.datasets.tickets()                              # 80 labelled support messages
tickets.mutate(team=team(col.message))                            # a new column, one call per message
functai.evaluate(team, tickets, expected="category")              # how often it's right, with a range
```

## Installation

```bash
pip install "functai[data]"      # Python 3.11+
```

With an API key in your environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, …) or a Claude, ChatGPT or Copilot subscription, there's
nothing to set up: functai picks a small model you can use and tells you
which. To choose: `functai.configure(lm="claude-haiku-4-5")`.

## Documentation

**[maximerivest.github.io/functai](https://maximerivest.github.io/functai/)**, with three ways in:

- **[I have a table of text](https://maximerivest.github.io/functai/get-started.html)**: label, sort or score every row, check it, make it better.
- **[I have notes or documents](https://maximerivest.github.io/functai/articles/notes-to-data.html)**: pull the facts out as columns, following your protocol.
- **[I have a prompt that works](https://maximerivest.github.io/functai/articles/from-a-prompt.html)**: send it exactly as it is, then add types, tables and tests.

The [examples](examples/) each solve one problem end to end.

## Built on

[lm15](https://github.com/lm15-dev/lm15-python) (every provider, no SDKs),
[lmcc](https://github.com/MaximeRivest/lmcc) (how values are written into
prompts and read back) and [dpyr](https://github.com/MaximeRivest/dpyr)
(tables).

## Development

From git: `pip install "functai @ git+https://github.com/MaximeRivest/functai#subdirectory=python"`.

This folder is the Python package; the repository around it holds the
contract every language's FunctAI follows and the documentation site. In
this folder: `uv sync --all-groups`, then `uv run pytest` (offline, a fake
provider). The documentation runs against real models:
`.venv/bin/python tests/docs_live.py --render` (costs cents).
