# serve { #functai.serve }

```{.python .no-run}
serve(
    program,
    *,
    host='127.0.0.1',
    port=8080,
    keys=None,
    store=None,
    lm=None,
    approvals='owner',
    trust=False,
    block=True,
)
```

Serve a program over HTTP: its interface, calls, streams and
conversations, to callers who see only its boundary.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type                                               | Description                                                                                 | Default       |
|-----------|----------------------------------------------------|---------------------------------------------------------------------------------------------|---------------|
| program   | AI function, module, or saved folder               |                                                                                             | _required_    |
| host      | where to listen. Without ``keys``, only 127.0.0.1. |                                                                                             | `'127.0.0.1'` |
| port      | where to listen. Without ``keys``, only 127.0.0.1. |                                                                                             | `'127.0.0.1'` |
| keys      | list, str or file of keys                          | Callers send ``Authorization: Bearer <key>``.                                               | `None`        |
| store     | optional                                           | Where conversations are kept (default: this process's memory).                              | `None`        |
| lm        | str                                                | The model every call uses.                                                                  | `None`        |
| approvals | \'owner\' or \'caller\'                            |                                                                                             | `'owner'`     |
| block     | bool                                               | False: serve in a background thread and return the server (``server.shutdown()`` stops it). | `True`        |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                                             | Description   |
|--------|--------------------------------------------------|---------------|
|        | the server (``http.server.ThreadingHTTPServer``) |               |

## See Also {.doc-section .doc-section-see-also}

remote : a served program, used like a local one.
Service : the same, to mount in another app (``Service(...).asgi``).