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
| TypeScript / JavaScript | `ts/` | | next |
| R | `r/` | | planned |
| Julia | `julia/` | | planned |

Every language follows the same [contract](contract/): a call logged in one
can be rated in another, and a function improved in one runs in another.
[design/01-many-languages.md](design/01-many-languages.md) is the plan.

## Documentation

**[maximerivest.github.io/functai](https://maximerivest.github.io/functai/)**

## This repository

| Path | What it holds |
|---|---|
| [`contract/`](contract/) | what every implementation must agree on: formats, schemas, cases |
| [`python/`](python/) | the Python package, its tests and examples |
| [`docs/`](docs/), [`tools/`](tools/) | the website: runnable notebooks, and the tool that runs and builds them |
| [`design/`](design/) | design notes |
| [`check`](check) | one command: every implementation against the contract |

MIT licensed.
