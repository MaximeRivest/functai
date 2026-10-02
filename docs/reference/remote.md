---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# remote { #functai.remote }

`remote`

A program served elsewhere, used like a local one (contract/serving.md).

    team = functai.remote("https://lambda.example/programs/team", key=os.environ["TEAM_KEY"])
    team("I was charged twice for order B-2210.")       # 'billing'
    team.map(tickets, threads=8)                          # a table, as with a local program
    functai.evaluate(team, rows)

Its interface is the server's (``GET /interface``): its inputs are bound
and checked here before anything is sent, and what comes back is checked
against its outputs. Each call is logged here (``program.kind`` ``"remote"``,
``program.remote`` the URL) and there; the server's record names this
call as its ``parent``, so the two logs make one call tree.

## Classes

| Name | Description |
| --- | --- |
| [RemoteError](#functai.remote.RemoteError) | The server answered with an error: ``code`` is its code (or |
| [RemoteProgram](#functai.remote.RemoteProgram) | A program served elsewhere, called like a local one. Built by |
| [RemoteStream](#functai.remote.RemoteStream) | A served call's events (the outside view), read as they arrive: |

### RemoteError { #functai.remote.RemoteError }

```{.python .no-run}
remote.RemoteError(code, message, *, status=0, type='')
```

The server answered with an error: ``code`` is its code (or
``remote-<status>``), ``status`` the HTTP status, ``type`` its error's type.

### RemoteProgram { #functai.remote.RemoteProgram }

```{.python .no-run}
remote.RemoteProgram(url, *, key=None, timeout=120.0)
```

A program served elsewhere, called like a local one. Built by
``functai.remote(url, key=...)``.

Its inputs and outputs are the served program's (read from the server's
``/interface``): inputs are checked here before anything is sent
(``InterfaceError``). It is called, mapped over a table (``.map``),
evaluated (``functai.evaluate``) and streamed (``.stream``: the
server's outside view) as a local program is. Each call is logged here,
as a program of kind ``"remote"``, and on the server, whose record names
this call as its parent.

``url``: where it is served; ``version``: the served program's version.
A served program's conversations are kept by its server: talk to them
over HTTP (``POST <url>/conversations/<id>/turns``). The tokens a call
used are in the server's log, not here.

#### Attributes

| Name | Description |
| --- | --- |
| `version` | The served program's version, as the server says. |

#### Methods

| Name | Description |
| --- | --- |
| [stream](#functai.remote.RemoteProgram.stream) | The call, watched: the server's outside view, as the contract's JSON |

##### stream { #functai.remote.RemoteProgram.stream }

```{.python .no-run}
remote.RemoteProgram.stream(*args, **kwargs)
```

The call, watched: the server's outside view, as the contract's JSON
events (not logged here).

### RemoteStream { #functai.remote.RemoteStream }

```{.python .no-run}
remote.RemoteStream(resp, answer)
```

A served call's events (the outside view), read as they arrive:
iterate for the answer's text, ``events()`` for every event, ``result``
for the value.

## Functions

| Name | Description |
| --- | --- |
| [remote](#functai.remote.remote) | A program served elsewhere (``functai serve``), used like a local one. |

### remote { #functai.remote.remote }

```{.python .no-run}
remote.remote(url, *, key=None, timeout=120.0)
```

A program served elsewhere (``functai serve``), used like a local one.

#### Parameters {.doc-section .doc-section-parameters}

| Name    | Type   | Description                                               | Default    |
|---------|--------|-----------------------------------------------------------|------------|
| url     | str    | Where it is served (the URL its ``/interface`` is under). | _required_ |
| key     | str    | The key the server asks for.                              | `None`     |
| timeout | float  | Seconds to wait for one answer.                           | `120.0`    |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type          | Description                                                                                                                       |
|--------|---------------|-----------------------------------------------------------------------------------------------------------------------------------|
|        | RemoteProgram | Called with the program's inputs; ``.map``, ``evaluate``, ``.stream`` work as for a local program; its calls are logged here too. |

#### Raises {.doc-section .doc-section-raises}

| Name   | Type        | Description                                                                                                                                                      |
|--------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | RemoteError | The server refused (``code`` ``remote-401`` for a wrong key...) or answered something that is not a FunctAI program. ``from functai.remote import RemoteError``. |

#### See Also {.doc-section .doc-section-see-also}

- [`serve`](serve.md): serve a program.

#### Examples {.doc-section .doc-section-examples}

```python
import functai
from functai import *
```

```{.python .no-run}
# not run: it needs a served program (the guide Serve it over HTTP runs one)
team = functai.remote("https://example.org/team", key=os.environ["TEAM_KEY"])
team("I was charged twice for order B-2210.")
team.map(tickets, threads=8)
```