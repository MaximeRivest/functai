"""A program served elsewhere, used like a local one (contract/serving.md).

    team = functai.remote("https://lambda.example/programs/team", key=os.environ["TEAM_KEY"])
    team("I was charged twice for order B-2210.")       # 'billing'
    team.map(tickets, threads=8)                          # a table, as with a local program
    functai.evaluate(team, rows)

Its interface is the server's (``GET /interface``): its inputs are bound
and checked here before anything is sent, and what comes back is checked
against its outputs. Each call is logged here (``program.kind`` ``"remote"``,
``program.remote`` the URL) and there; the server's record names this
call as its ``parent``, so the two logs make one call tree.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, Optional

from . import calllog
from .errors import FunctAIError, InterfaceError
from .module import FunctAIModule


class RemoteError(FunctAIError):
    """The server answered with an error: ``code`` is its code (or
    ``remote-<status>``), ``status`` the HTTP status, ``type`` its error's type."""

    def __init__(self, code: str, message: str, *, status: int = 0, type: str = ""):
        super().__init__(code, message)
        self.status = status
        self.type = type


def _request(url: str, key: Optional[str], data: Any = None, *, timeout: float, parent: Optional[str] = None,
             stream: bool = False) -> Any:
    headers = {"accept": "text/event-stream" if stream else "application/json"}
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        headers["content-type"] = "application/json"
    if key:
        headers["authorization"] = f"Bearer {key}"
    if parent:
        headers["functai-parent"] = parent
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if data is not None else "GET")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            err = (json.loads(exc.read().decode("utf-8") or "{}") or {}).get("error") or {}
        except ValueError:
            err = {}
        code = err.get("code") or f"remote-{exc.code}"
        text = err.get("message") or f"{url} answered {exc.code} ({err.get('type') or exc.reason})"
        if code in ("interface-input", "interface-output"):
            raise InterfaceError(code, err.get("field"), text) from None
        raise RemoteError(code, text, status=exc.code, type=err.get("type") or "") from None
    if stream:
        return resp
    with resp:
        return json.loads(resp.read().decode("utf-8"))


class RemoteProgram(FunctAIModule):
    """A program served elsewhere, called like a local one. Built by
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
    """

    def __init__(self, url: str, *, key: Optional[str] = None, timeout: float = 120.0):
        self._url = url.rstrip("/")
        self._key = key
        self._timeout = float(timeout)
        described = _request(self._url + "/interface", key, timeout=timeout)
        if described.get("functai_interface") != 1:
            raise RemoteError("remote-format", f"{url} does not describe a FunctAI program this version reads "
                                               f"(functai_interface {described.get('functai_interface')!r})")
        self._described = described
        remote = self

        def call(**inputs: Any) -> Any:
            current = calllog.current()
            got = _request(remote._url + "/call", remote._key, {"inputs": {k: calllog.to_json(v)[0]
                                                                           for k, v in inputs.items()}},
                           timeout=remote._timeout, parent=current.id if current is not None else None)
            outputs = got.get("outputs") or {}
            names = [f["name"] for f in remote.interface["outputs"]]
            return outputs.get(names[-1]) if len(names) == 1 else {n: outputs.get(n) for n in names}

        call.__name__ = described["name"]
        call.__qualname__ = described["name"]
        call.__doc__ = described["interface"].get("description") or ""
        call.__module__ = "functai.remote"
        super().__init__(call, interface=described["interface"])
        self.__name__ = described["name"]

    def _check_definition(self) -> None:
        """The server's interface, checked as its kind's: an AI function's
        shapes may carry lmcc's keywords, which a module's check refuses."""
        from . import interface as _interface
        _interface.check(self._interface, ai=self._described.get("kind") == "ai", program=self.__name__)

    @property
    def url(self) -> str:
        return self._url

    @property
    def version(self) -> str:
        """The served program's version, as the server says."""
        return self._described["version"]

    def ai_functions(self) -> list:
        return []

    def named_ai_functions(self) -> Dict[str, Any]:
        return {}

    def stream(self, *args: Any, **kwargs: Any) -> "RemoteStream":
        """The call, watched: the server's outside view, as the contract's JSON
        events (not logged here)."""
        given = self._given(args, kwargs)
        from . import interface as _interface
        bound = _interface.bind_inputs(self.interface, given, program=self.__name__)
        resp = _request(self._url + "/stream", self._key, {"inputs": {k: calllog.to_json(v)[0]
                                                                       for k, v in bound.items()}},
                        timeout=self._timeout, stream=True)
        return RemoteStream(resp, self.interface["outputs"][-1]["name"])

    def conversation(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("a remote program's conversations are kept by its server: POST "
                        f"{self._url}/conversations/<id>/turns")

    def __repr__(self) -> str:
        return f"<RemoteProgram {self.__name__} at {self._url}>"


class RemoteStream:
    """A served call's events (the outside view), read as they arrive:
    iterate for the answer's text, ``events()`` for every event, ``result``
    for the value."""

    def __init__(self, resp: Any, answer: str):
        self._resp = resp
        self._answer = answer
        self._events: list = []
        self._done = False
        self._value: Any = None
        self._error: Optional[Dict[str, Any]] = None

    def _read(self) -> Iterator[Dict[str, Any]]:
        data: list = []
        for raw in self._resp:
            line = raw.decode("utf-8").rstrip("\n").rstrip("\r")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                event = json.loads("\n".join(data))
                data = []
                self._events.append(event)
                if event.get("kind") == "done" and event.get("call") == event.get("tree"):
                    self._value, self._done = event.get("value"), True
                elif event.get("kind") == "failed" and event.get("call") == event.get("tree"):
                    self._error, self._done = event.get("error") or {}, True
                yield event
        self._done = True

    def events(self) -> Iterator[Dict[str, Any]]:
        yield from self._events
        if not self._done:
            yield from self._read()

    def __iter__(self) -> Iterator[str]:
        for e in self.events():
            if e.get("kind") == "text" and e.get("answer"):
                yield e["text"]

    @property
    def result(self) -> Any:
        for _ in self.events():
            pass
        if self._error is not None:
            raise RemoteError(self._error.get("code") or "remote-failed",
                              f"the served call failed ({self._error.get('type')})", type=self._error.get("type", ""))
        return self._value

    def close(self) -> None:
        try:
            self._resp.close()
        except Exception:  # noqa: BLE001
            pass


def remote(url: str, *, key: Optional[str] = None, timeout: float = 120.0) -> RemoteProgram:
    '''A program served elsewhere (``functai serve``), used like a local one.

    Parameters
    ----------
    url : str
        Where it is served (the URL its ``/interface`` is under).
    key : str, optional
        The key the server asks for.
    timeout : float
        Seconds to wait for one answer.

    Returns
    -------
    RemoteProgram
        Called with the program's inputs; ``.map``, ``evaluate``, ``.stream``
        work as for a local program; its calls are logged here too.

    Raises
    ------
    RemoteError
        The server refused (``code`` ``remote-401`` for a wrong key...) or
        answered something that is not a FunctAI program. ``from
        functai.remote import RemoteError``.

    See Also
    --------
    serve : serve a program.

    Examples
    --------
    ```python
    # not run: it needs a served program (the guide Serve it over HTTP runs one)
    team = functai.remote("https://example.org/team", key=os.environ["TEAM_KEY"])
    team("I was charged twice for order B-2210.")
    team.map(tickets, threads=8)
    ```
    '''
    return RemoteProgram(url, key=key, timeout=timeout)


__all__ = ["remote", "RemoteProgram", "RemoteStream", "RemoteError"]
