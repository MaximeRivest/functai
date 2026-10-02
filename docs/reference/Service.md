# Service { #functai.Service }

```{.python .no-run}
Service(
    program,
    *,
    keys=None,
    store=None,
    lm=None,
    approvals='owner',
    trust=False,
)
```

A program as an HTTP service, independent of any server: ``handle``
answers one request; ``serve`` runs the standard library's threaded
server; ``asgi`` is an ASGI app (FastAPI's ``app.mount``, uvicorn).

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type                                 | Description                                                                                                                                                                        | Default    |
|-----------|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| program   | AI function, module, or saved folder | What is served (a folder is loaded with ``functai.load``).                                                                                                                         | _required_ |
| keys      | (list, str or file)                  | Bearer keys a caller must send (a file: one per line). None: no key, which ``serve`` allows only on 127.0.0.1.                                                                     | `None`     |
| store     | optional                             | Where conversations are kept (as ``fn.conversation(store=...)``); None keeps them in this process's memory.                                                                        | `None`     |
| lm        | str                                  | The model every call uses, bound when serving starts.                                                                                                                              | `None`     |
| approvals | \'owner\' or \'caller\'              | Who answers a tool call the program's ``approve`` rule asks about: the owner (in their own process; the caller sees the turn waiting), or the caller, through the approvals route. | `'owner'`  |
| trust     | bool                                 | For a saved folder that runs code: load it (``functai.load(trust=True)``).                                                                                                         | `False`    |

## Attributes

| Name | Description |
| --- | --- |
| `asgi` | An ASGI app: ``app.mount("/team", service.asgi)`` in FastAPI, or |

## Methods

| Name | Description |
| --- | --- |
| [handle](#functai.Service.handle) | Answer one request (``headers`` with lower-case names). |
| [serve](#functai.Service.serve) | Run the standard library's threaded HTTP server. Without keys it |

### handle { #functai.Service.handle }

```{.python .no-run}
Service.handle(method, path, headers, body=b'')
```

Answer one request (``headers`` with lower-case names).

### serve { #functai.Service.serve }

```{.python .no-run}
Service.serve(host='127.0.0.1', port=8080, *, block=True)
```

Run the standard library's threaded HTTP server. Without keys it
listens only on this machine (127.0.0.1, ::1, localhost).