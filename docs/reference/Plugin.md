---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Plugin { #functai.Plugin }

```{.python .no-run}
Plugin(name, *, version='0.0.0', api=API, description='')
```

A named, versioned set of hooks.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type   | Description                                                                                              | Default    |
|---------|--------|----------------------------------------------------------------------------------------------------------|------------|
| name    | str    | Lower-case letters, digits, ``_`` and ``-``: how its changes and entries are named in records.           | _required_ |
| version | str    | Its own version, recorded with every change it makes.                                                    | `'0.0.0'`  |
| api     | int    | The plugin API it was written for (1). A version this FunctAI does not implement refuses ``plugin-api``. | `API`      |
| Hooks   |        |                                                                                                          | _required_ |

## Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```python
guard = functai.Plugin("no-deletes", version="1.0.0")

@guard.tool_call
def refuse_deletes(tool):
    if tool.name == "delete_file":
        return functai.Change(block="deleting files is not allowed here")

functai.configure(plugins=[guard])
```

## Methods

| Name | Description |
| --- | --- |
| [before_call](#functai.Plugin.before_call) | An AI function is about to be asked: sections, model, settings, tools. |
| [context](#functai.Plugin.context) | Which earlier turns a turn is shown, and sections about them. |
| [describe](#functai.Plugin.describe) | Its manifest, as data: name, version, api, the hooks it uses. |
| [on](#functai.Plugin.on) | Register ``fn`` for ``hook`` (as a decorator when ``fn`` is left out). |
| [request](#functai.Plugin.request) | The escape hatch: rewrite the provider request (the call is then not replayable). |
| [tool_call](#functai.Plugin.tool_call) | A tool is about to run: change its input, block it, or ask a person. |
| [tool_result](#functai.Plugin.tool_result) | A tool ran: change what the model is shown. |
| [turn_end](#functai.Plugin.turn_end) | A turn ended (hears only; may keep entries in its conversation). |
| [turn_start](#functai.Plugin.turn_start) | A turn is about to be recorded: may replace its inputs. |

### before_call { #functai.Plugin.before_call }

```{.python .no-run}
Plugin.before_call(fn)
```

An AI function is about to be asked: sections, model, settings, tools.

### context { #functai.Plugin.context }

```{.python .no-run}
Plugin.context(fn)
```

Which earlier turns a turn is shown, and sections about them.

### describe { #functai.Plugin.describe }

```{.python .no-run}
Plugin.describe()
```

Its manifest, as data: name, version, api, the hooks it uses.

### on { #functai.Plugin.on }

```{.python .no-run}
Plugin.on(hook, fn=None)
```

Register ``fn`` for ``hook`` (as a decorator when ``fn`` is left out).

### request { #functai.Plugin.request }

```{.python .no-run}
Plugin.request(fn)
```

The escape hatch: rewrite the provider request (the call is then not replayable).

### tool_call { #functai.Plugin.tool_call }

```{.python .no-run}
Plugin.tool_call(fn)
```

A tool is about to run: change its input, block it, or ask a person.

### tool_result { #functai.Plugin.tool_result }

```{.python .no-run}
Plugin.tool_result(fn)
```

A tool ran: change what the model is shown.

### turn_end { #functai.Plugin.turn_end }

```{.python .no-run}
Plugin.turn_end(fn)
```

A turn ended (hears only; may keep entries in its conversation).

### turn_start { #functai.Plugin.turn_start }

```{.python .no-run}
Plugin.turn_start(fn)
```

A turn is about to be recorded: may replace its inputs.