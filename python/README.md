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
    ...

team("I was charged twice for order B-2210, please fix this.")    # 'billing'

tickets = functai.datasets.tickets()                              # 80 labelled support messages
tickets.mutate(team=team(col.message))                            # a new column, one call per message
functai.evaluate(team, tickets, expected="category")              # how often it's right, with a range
```

## Installation

```bash
pip install "functai[data]"      # Python 3.11+
```

That is 1.1.0. The documentation follows the next release, whose API
changed where [Upgrading](https://maximerivest.github.io/functai/articles/upgrading.html)
says; to use it now, install from GitHub:

```bash
pip install "functai[data] @ git+https://github.com/MaximeRivest/functai#subdirectory=python"
```

With an API key in your environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, …) or a Claude, ChatGPT or Copilot subscription, there's
nothing to set up: functai picks a small model you can use and tells you
which. To choose: `functai.configure(lm="claude-haiku-4-5")`.

## Programs you can describe, logs you can keep

Every program says what it takes and gives, as data, and a `@module`
checks every call against it:

```python
@functai.module
def support(message: str, tone: str = "kind") -> str:
    ...

support.interface          # {"description", "inputs": [...], "outputs": [...]}: served, saved, described
support(3)                 # InterfaceError (interface-input): 3 does not fit {"type":"string"}
functai.describe("saved/") # what a saved program takes and gives, without loading it
```

The call log keeps what you allow, field by field, and a host's rule holds
for every program it runs (`log_content` only ever removes):

```python
with functai.configure(log_calls=True, log_content={"transcript": False}):
    summarize(transcript)  # the record has every value but the transcript
```

A call tree's events can be watched and kept while it runs, and read
again elsewhere:

```python
seen = []
store = functai.MemoryStore()
functai.configure(observers=[seen], journal=functai.Journal(store, required=True))
```

## Documentation

**[maximerivest.github.io/functai](https://maximerivest.github.io/functai/python.html)**, with three ways in:

- **[I have a table of text](https://maximerivest.github.io/functai/get-started.html)**: label, sort or score every row, check it, make it better.
- **[I have notes or documents](https://maximerivest.github.io/functai/articles/notes-to-data.html)**: pull the facts out as columns, following your protocol.
- **[I have a prompt that works](https://maximerivest.github.io/functai/articles/from-a-prompt.html)**: send it exactly as it is, then add types, tables and tests.

Then [eight tutorials](https://maximerivest.github.io/functai/tutorials/index.html),
from a first function to decision models and a model you own; the
[examples](https://maximerivest.github.io/functai/examples/index.html), each
solving one problem end to end; and the
[reference](https://maximerivest.github.io/functai/reference/index.html).

functai also exists in [TypeScript, R and
Julia](https://maximerivest.github.io/functai/#what-each-language-has): the
same function has the same version in each, and a function saved here loads
there.

## Built on

[lm15](https://github.com/lm15-dev/lm15-python) (every provider, no SDKs),
[lmcc](https://github.com/MaximeRivest/lmcc) (how values are written into
prompts and read back) and [dpyr](https://github.com/MaximeRivest/dpyr)
(tables).

## Development

This folder is the Python package; the [repository around
it](https://github.com/MaximeRivest/functai) holds the contract every
language's FunctAI follows, the other languages, and the website. In this
folder: `uv sync --all-extras --all-groups`, then `uv run pytest` (offline,
a fake provider). Against real models (costs cents, needs model keys):
`.venv/bin/python tests/docs_live.py` runs this README's code, and
`--render` also every page of the website.
