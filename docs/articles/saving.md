---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Ship it: save, verify, load

*Find everything a program depends on, save it to a folder, prove it runs in a fresh environment, and load it back.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

A functai program is code plus a contract: its inputs and outputs are
typed, and to run it somewhere else, everything it depends on must come
along. The AI functions and their optimized prompts, the tools, your
helper functions and classes, constants, data files, and the exact
packages. functai reads the code to find all of it.

The steps are always the same:

1. `check(program)` shows what it depends on, and what would stop a
   clean save;
2. `save(program, folder)` writes it all to a folder you can commit;
3. `verify(folder)` proves it works in a brand-new environment;
4. `load(folder)` brings it back, anywhere.

## A program to save

```python
from dataclasses import dataclass
from enum import Enum
from functai import module

class Priority(Enum):
    LOW = "low"
    URGENT = "urgent"

@dataclass
class Handled:
    priority: Priority
    reply: str

ORDERS = {"A-1042": "stuck at carrier", "B-2210": "delivered"}

def lookup_order(order_id: str) -> str:
    """The order's shipping status."""
    return ORDERS.get(order_id, "no such order")

@ai
def priority(message: str) -> Priority:
    """How urgently the message needs an answer."""
    ...

@ai(tools=[lookup_order])
def draft_reply(message: str) -> str:
    """A short, friendly reply. Check the order first when one is mentioned."""
    ...

@module
def handle(message: str) -> Handled:
    return Handled(priority(message), draft_reply(message))
```

## Check

```python
functai.check(handle)
```

```output
handle  @module  [__main__]
├── Handled  class  [__main__]
│   ├── Priority  class  [__main__]
│   │   └── Enum  (stdlib)
│   └── dataclass  (stdlib)
├── priority  AI function (message: str → Priority)  [__main__]
│   └── Priority  (see above)
└── draft_reply  AI function (message: str → str)  [__main__]
    └── tool lookup_order  function  [__main__]
        └── ORDERS = {'A-1042': 'stuck at carrier', 'B-221...

requirements: functai @ file:///home/maxime/Projects/functai/python

! local-install  requirements: installed from folders on this machine: functai (/home/maxime/Projects/functai/python)
    fix: the saved program loads where those folders exist; publish them, or install released versions, to load it anywhere
```

> **Why this page shows `local-install`** This site is built from functai's development copy, installed from a
> folder, so `check` warns that the saved program needs that folder. It
> is the warning you would get for any package of yours installed that
> way. With functai installed from PyPI, the requirement reads
> `functai==<version>` and there is no warning.

`check` follows every name the code reaches: AI functions and modules
(also through helper functions), their tools, your own functions and
classes (in files or notebook cells), the types in signatures, constants,
and data files read through `functai.file("...")`.

| found | saved as |
|---|---|
| an AI function or `@module` | its code, settings, instruction and examples |
| a function or class from your code | its source, verbatim |
| a name from an installed package | a pinned requirement |
| the standard library | nothing |
| a constant (numbers, text, lists, dicts, enum members, patterns) | its value |
| a file read with `functai.file("data/x.txt")` | a copy |

And what stops a clean save, each with its fix:

| problem | example | fix |
|---|---|---|
| `hidden-state` | a tool writes `CACHE[q] = ...` into a global | pass it in and return it, or `save(allow=["hidden-state"])` to save its current value |
| `untyped-input`, `untyped-output` | `def f(text):` | annotate it |
| `unsaveable-value` | a global client, lock or open file | create it inside the function, or pass it in |
| `lambda`, `no-source`, `name-conflict` | a lambda tool; two nested `def f` | a named `def`; distinct names |
| `local-import-inside` | `import helpers` inside a function | import it at the top |

## Save

```python
import tempfile, os
folder = os.path.join(tempfile.mkdtemp(), "support_desk")

functai.save(handle, folder, record=[{"message": "Where is order A-1042?"}])
for root, dirs, files in sorted(os.walk(folder)):
    for f in sorted(files):
        print(os.path.relpath(os.path.join(root, f), folder))
```

```output
functai.json
recordings.json
requirements.lock
requirements.txt
code/main.py
```

The folder is plain files, readable and diffable:

| file | holds |
|---|---|
| `functai.json` | each AI function's settings, instruction, examples, signature and fingerprints; file hashes |
| `code/*.py` | the code the program reaches (a notebook becomes `code/main.py`) |
| `files/` | data files read with `functai.file(...)` |
| `requirements.txt`, `requirements.lock` | the packages, pinned; and everything they pull in |
| `recordings.json` | model replies recorded with `save(record=...)`, for `verify` |

`save` refuses while `check` finds errors, and writes the folder whole or
not at all. **Keys and connections are never saved**: the loading machine
uses its own.

## Verify

`verify` is the proof. It builds a new environment with `uv` from the
lock file alone, loads the program there from an empty folder (so nothing
from your project can leak in), and checks that every AI function sends
byte-identical requests, and that each recording replays to the same
result. No model is called.

```python
functai.verify(folder, trust=True, fresh=False)
```

```output
verified in this environment
```

(`fresh=False` checks in the current environment, which is quicker and
weaker; the default builds the fresh one, about a second once uv's cache
is warm.)

## Load

```python
loaded = functai.load(folder, trust=True)
loaded("My toaster order B-2210 never arrived??")
```

```output
Handled(priority=<Priority.URGENT: 'urgent'>, reply='Your toaster order B-2210 has been delivered. Please check if it might have been received by someone else at your address or left in a safe place. Let me know if you need any further assistance!')
```

`load` checks before it runs anything: file hashes (catching accidental
edits), missing packages, and afterwards that every AI function still
sends the requests it sent when saved.

> **`trust=True` means "run this code"** `load` and `verify` run the saved Python code, so they require
> `trust=True`. The hashes catch accidents, not someone who edits both the
> code and `functai.json`. Load only folders you would run as code.

## What reading code can't see

Names looked up while running (`getattr`, `importlib`, `eval`), functions
passed in as arguments, and files not read through `functai.file`.
`check` points at them. `@ai(requires=["numpy>=2"])` or
`save(requires=[...])` declares packages by hand;
`save(include=["myproject"])` saves an editable-installed project as
code; and `verify` with recordings catches anything still missing.
