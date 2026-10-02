# delegate { #functai.delegate }

```{.python .no-run}
delegate(program, *, name=None, description=None, remember=True, effects=None)
```

Another program as a tool: an assistant hands part of its work to it.

Asked inside a conversation's turn, the program answers in a
conversation of its own (``<conversation>.<name>``, in the same store),
which follows the branch of the turn that asked: asked again later on
that branch, it remembers what it was asked before; on another branch,
it does not. Its calls are in the asking turn's call tree, under the tool
call. Outside a conversation, or with ``remember=False``, it is called
plainly.

## Parameters {.doc-section .doc-section-parameters}

| Name        | Type                     | Description                                                                                                              | Default    |
|-------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------|------------|
| program     | AI function or module    | What does the work. Its inputs are the tool's.                                                                           | _required_ |
| name        | str                      | The tool's name (default: the program's).                                                                                | `None`     |
| description | str                      | What the model is told the tool does (default: the program's docstring).                                                 | `None`     |
| remember    | bool                     | Keep a conversation with it, per branch (default True).                                                                  | `True`     |
| effects     | \'reads\' or \'changes\' | What it does to the world (default: "reads" when every tool it has only reads, else unknown, which counts as "changes"). | `None`     |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description               |
|--------|--------|---------------------------|
|        | Tool   | For ``@ai(tools=[...])``. |