# functai

**Write a function. A language model does the work. You measure how well.**

functai turns a typed function into a call to a language model. The
function's name, description and types say what you want; the answer comes
back as the type you asked for. Then you run it on a whole table, find out
how often it is right, and make it better.

```python
from typing import Literal
from functai import ai

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""

team("I was charged twice for order B-2210, please fix this.")    # 'billing'
```

## Languages

| Language | Folder | Install | Status |
|---|---|---|---|
| Python | [`python/`](python/) | `pip install "functai[data]"` | released ([PyPI](https://pypi.org/project/functai/)) |
| TypeScript / JavaScript | [`ts/`](ts/) | not on npm yet | 0.1.0: passes the whole contract; see [ts/README.md](ts/README.md) |
| R | [`r/`](r/) | not on CRAN yet | 0.1.0: passes the whole contract, pairs with the tidyverse; see [r/README.md](r/README.md) |
| Julia | [`julia/`](julia/) | not registered yet | 0.1.0: passes the whole contract; `@ai` functions with Julia types, broadcasting, formulas and MLJ; see [julia/README.md](julia/README.md) |

Every language follows the same [contract](contract/): the same function
has the same version everywhere, a call logged in one can be rated in
another, and a function improved and saved in one runs in another.
[design/01-many-languages.md](design/01-many-languages.md) is the plan.

## Documentation

**[maximerivest.github.io/functai](https://maximerivest.github.io/functai/)**

## This repository

| Path | What it holds |
|---|---|
| [`contract/`](contract/) | what every implementation must agree on: formats, schemas, cases |
| [`python/`](python/) | the Python package, its tests and examples |
| [`ts/`](ts/) | the TypeScript package and its tests |
| [`r/`](r/) | the R package and its tests |
| [`julia/`](julia/) | the Julia package (`FunctAI.jl`) and its tests |
| [`docs/`](docs/), [`tools/`](tools/) | the website: runnable notebooks, and the tool that runs and builds them; `tools/crosslang.py` checks the four languages against each other |
| [`design/`](design/) | design notes |
| [`check`](check) | one command: every implementation against the contract |

MIT licensed.
