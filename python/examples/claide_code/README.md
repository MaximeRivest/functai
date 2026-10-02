---
rat:
  python:
    dependencies: ["-e .[data]"]
---

# A terminal assistant: tools and memory


A small assistant that can look around a project with a few shell
commands, and remembers the conversation. Two things do it: a tool that
says it only reads (`functai.tool(effects="reads")`), and a conversation.

Every output below is a real reply. This page is a notebook: open it in
Chattering and run it, or run it all with `python/.venv/bin/python tools/docs.py run python/examples/claide_code/README.md`.

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)

from functai import ai
```

## The tool

A tool is a typed Python function with a docstring; the model sees its
name, parameters and docstring. This one runs a command, but only from
an allow-list of read-only commands: a model should never get a free
shell.

```python
import shlex
import subprocess

READ_ONLY = {"pwd", "ls", "whoami", "cat", "head", "wc", "git"}

def run(command: str) -> str:
    """Run a read-only shell command (pwd, ls, whoami, cat, head, wc, or
    git log/status/diff) and return what it printed."""
    argv = shlex.split(command)
    if not argv or argv[0] not in READ_ONLY or (argv[0] == "git" and argv[1:2] not in (["log"], ["status"], ["diff"])):
        return f"refused: {command!r} is not a read-only command"
    done = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    return (done.stdout + done.stderr)[-4000:]
```

## The assistant

```python
@ai(tools=[functai.tool(run, effects="reads")])
def assistant(message: str) -> str:
    """You help a developer understand the project in the current folder.
    Use the tool to look before you answer. Be brief."""
    ...

chat = assistant.conversation()
print(chat("Which project is this folder part of, and what changed in it recently?"))
```

```output
This folder is part of the "functai" project, a Python package for the FunctAI framework. The project involves conversations, tools, and serving programs with a focus on AI-assisted tutoring and decision models.

Recently, there was a very large update with 130 files changed, including 16,940 insertions and 690 deletions. Major changes include extensive additions to the conversations module, serving, stores, replies, and tests, as well as updates to documentation and various Python source files.
```

The model decided which commands to run. `functai.phistory()` shows the
whole exchange, tool calls and results included (here, the last model
call of the loop):

```python
print(functai.phistory())
```

````output
[2026-10-01T20:06:39] assistant → gpt-4.1-mini

System message:

Function: assistant

You help a developer understand the project in the current folder.
Use the tool to look before you answer. Be brief.

Reply in exactly this form:
<result>
...
</result>


User message:

<message>
Which project is this folder part of, and what changed in it recently?
</message>


Assistant message:

[tool call run({"command": "cat README.md"})]
[tool call run({"command": "git log -1 --stat"})]

Tool message:

[tool result call_oFxBWfrswu6beY2aUieCrsjU]  a refusal, and any other
exception means no answer came (the event is sent again, with a growing
pause). A barrier waits at most the journal's `timeout`; a store should
time out its own I/O too. A store with `extend(events)` is sent every
event through it, what waits as one batch; a subclass that overrides
only `append` is sent every event through its `append`. At exit, FunctAI waits at most two
seconds for observers and journals to catch up. In a process forked
inside a call, calls start a tree of their own.

## Conversations, tools that ask first, serving

A conversation is a program's calls that remember each other. The
function is unchanged; the memory is the conversation's, kept where you
say, and nothing in it is ever deleted:

```python
chat = tutor.conversation("alex", store="tutoring/")   # the same line tomorrow opens it again
chat("Hi, I'm Alex.")
chat("What is 1/2 + 1/3?")
chat.turns[-1].saw                                      # what that answer was based on
chat.render("Why can't I add the bottoms?")             # the next request, nothing sent
other = chat.continue_from(chat.turns[0])               # a branch
```

A tool says what it does to the world, and a person can be asked before it
runs: at once, or later, from any process (the turn waits, saved, and goes
on without paying for a model answer twice):

```python
@functai.tool(effects="changes")
def refund(order: str, amount: float) -> str: ...

chat = assistant.conversation(customer, store=STORE, approve="changes")
try:
    chat("Refund my late parcel, please.")
except functai.Waiting as w:
    ...                                                 # later, anywhere: w.turn.approve(w.approvals[0])
```

A program is served with `functai serve saved/ --keys keys.txt` (or
`functai.serve(program)`), to callers who see only its boundary; on
their side, `functai.remote(url, key=...)` is a program again. Replies
can be kept on disk (`configure(cache_replies="disk")`), so a long
`fn.map(rows, threads=8)` resumes by being run again. Rated turns become
rows that keep their earlier turns (`functai.rated`), for `evaluate` and
the optimizers.

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


Tool message:

[tool result call_hDMBz1pFV5pVIEFv0AFs1JOM]     |   40 +-
 docs/reference/all_turns.md                        |    8 +
 docs/reference/clear_cache.md                      |    5 +-
 docs/reference/configure.md                        |    6 +-
 docs/reference/conversations.Conversation.md       |  132 ++
 docs/reference/conversations.Turn.md               |  134 ++
 docs/reference/earlier.md                          |   23 +
 docs/reference/evaluate.md                         |    6 +-
 docs/reference/index.md                            |   50 +-
 docs/reference/last_turns.md                       |    8 +
 docs/reference/load.md                             |    7 +-
 docs/reference/module.md                           |   89 +-
 docs/reference/prune_calls.md                      |   30 +
 docs/reference/quotes_found.md                     |   44 +
 docs/reference/rate.md                             |   24 +-
 docs/reference/rated.md                            |   15 +-
 docs/reference/remember.md                         |    7 +
 docs/reference/remote.md                           |  100 +
 docs/reference/serve.md                            |   43 +
 docs/reference/split.md                            |   27 +
 docs/reference/tool.md                             |   44 +
 python/CHANGELOG.md                                |   63 +
 python/README.md                                   |   38 +
 python/examples/README.md                          |    2 +-
 python/examples/claide_code/README.md              |   24 +-
 python/functai/__init__.py                         |   21 +-
 python/functai/__main__.py                         |   80 +-
 python/functai/bake/sft.py                         |    2 +-
 python/functai/calllog.py                          |  414 ++++-
 python/functai/config.py                           |   25 +-
 python/functai/conversations.py                    | 1938 ++++++++++++++++++++
 python/functai/core.py                             |  251 +--
 python/functai/engine.py                           |  193 +-
 python/functai/errors.py                           |   62 +-
 python/functai/evaluation.py                       |  101 +-
 python/functai/eventlog.py                         |  132 +-
 python/functai/judges.py                           |   78 +
 python/functai/meta.py                             |    4 +-
 python/functai/module.py                           |   55 +-
 python/functai/optimizers.py                       |   17 +
 python/functai/remote.py                           |  208 +++
 python/functai/replies.py                          |  399 ++++
 python/functai/saved.py                            |    2 +-
 python/functai/schema/call.schema.json             |   47 +-
 python/functai/schema/conversation.schema.json     |  178 ++
 python/functai/schema/event.schema.json            |   37 +
 python/functai/serving.py                          |  591 ++++++
 python/functai/stores.py                           |  447 +++++
 python/functai/streaming.py                        |  107 +-
 python/functai/tools.py                            |  290 +++
 python/functai/views.py                            |  148 ++
 python/functai_verifiers/__init__.py               |   10 +-
 python/pyproject.toml                              |    3 +
 python/tests/test_contract_stages.py               |  120 ++
 python/tests/test_conversations.py                 |  452 +++++
 python/tests/test_event_logs.py                    |   15 +-
 python/tests/test_failure_paths.py                 |  189 +-
 python/tests/test_layouts.py                       |   24 +-
 python/tests/test_learning.py                      |  139 ++
 python/tests/test_long_runs.py                     |  163 ++
 python/tests/test_serving.py                       |  238 +++
 python/tests/test_tools.py                         |  331 ++++
 python/uv.lock                                     |    2 +-
 tools/reference.yml                                |   32 +
 130 files changed, 16940 insertions(+), 690 deletions(-)


Tools: run

Response:

<result>
This folder is part of the "functai" project, a Python package for the FunctAI framework. The project involves conversations, tools, and serving programs with a focus on AI-assisted tutoring and decision models.

Recently, there was a very large update with 130 files changed, including 16,940 insertions and 690 deletions. Major changes include extensive additions to the conversations module, serving, stores, replies, and tests, as well as updates to documentation and various Python source files.
</result>

(finish: stop; tokens in 2062, out 110)
````

## Memory

A conversation shows each call its earlier turns (every one, by default;
`context=functai.last_turns(5)` keeps fewer), tool calls and results
included:

```python
print(chat("Without running anything: what was the most recent change about?"))
```

```output
The most recent change appears to be a large-scale update or refactor involving many parts of the project. It includes extensive additions and improvements to the conversations module, serving infrastructure, data stores, replies handling, and testing. Documentation files were also updated. This suggests a major enhancement or expansion of the core functionality and robustness of the FunctAI framework.
```

A command outside the allow-list comes back as a tool result saying it
was refused, which the model reads and works around. Asking for
something destructive:

```python
print(chat("Run `rm main.qmd` for me."))
```

```output
I cannot run commands that modify or delete files.
```

`chat.turns` holds the conversation; `assistant.conversation()` starts
a new one (`assistant` itself remembers nothing).

```python
len(chat.turns)
```

```output
3
```
