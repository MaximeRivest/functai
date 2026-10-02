"""Serving a program over HTTP (contract/serving.md).

    functai.serve(team, port=8080, keys="keys.txt")           # the standard library's server
    app = functai.Service(team, keys=["…"]).asgi               # or mounted in FastAPI, run by uvicorn

    functai serve team/ --lm gpt-4.1-mini --keys keys.txt --port 8080     # the same, from a saved folder

What a caller sees is the program's boundary (the ``outside`` view): its
answer, its answer's text as it is written, approvals addressed to it, and
its end; never a helper's answer, a tool's input or output, a thinking, or
an error's message. The owner watches everything in their own log.

Routes (JSON in, JSON out; ``Authorization: Bearer <key>`` when the service
has keys):

- ``GET /interface``: the program, described (``{"functai_interface": 1, …}``);
- ``GET /openapi.json``: the same as OpenAPI 3.1; ``GET /``: a form;
- ``POST /call`` ``{"inputs": {…}}``: ``{"call", "outputs", "value"}``;
- ``POST /stream`` ``{"inputs": {…}}``: Server-Sent Events of the outside view;
- ``POST /conversations/<id>/turns`` ``{"inputs", "request_id"?, "after"?, "wait"?}``:
  a turn, saved before it runs (``{"turn", "parent", "state"}``);
- ``GET /conversations/<id>/turns``, ``GET /conversations/<id>/turns/<turn>``;
- ``GET /conversations/<id>/turns/<turn>/events``: its events (SSE; a
  reconnect's ``Last-Event-ID`` is ``<writer>-<seq>``);
- ``POST /conversations/<id>/turns/<turn>/stop``;
- ``POST /conversations/<id>/turns/<turn>/approvals/<invocation>``
  ``{"verdict": "yes"|"no", "reason"?}`` (only when approvals go to callers).

A caller that is itself a FunctAI call sends ``FunctAI-Parent: <its call
id>``: the served call's record names it as its ``parent``, so the two logs
make one call tree.
"""

from __future__ import annotations

import asyncio
import hmac
import html
import json
import os
import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Union

from . import calllog
from .errors import ConversationError, FunctAIError, InterfaceError, ServeError

FORMAT = 1
MAX_BODY = 16 * 1024 * 1024          # bytes a request may send
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@dataclass
class Response:
    """What a route answers: a status, headers, and a body (bytes) or a
    stream of chunks (Server-Sent Events)."""
    status: int
    body: Union[bytes, Iterator[bytes]] = b""
    headers: Dict[str, str] = field(default_factory=dict)


def _json(status: int, data: Any) -> Response:
    return Response(status, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
                    {"content-type": "application/json"})


def _error(status: int, exc: BaseException, *, message: bool = False) -> Response:
    err: Dict[str, Any] = {"type": type(exc).__name__}
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        err["code"] = code
    if isinstance(exc, InterfaceError) and exc.field is not None:
        err["field"] = exc.field
    if message:
        err["message"] = calllog.safe_str(exc)
    return _json(status, {"error": err})


def _keys(keys: Any) -> List[str]:
    """Keys from a list, one key, or a file of one key per line (``#``
    comments and blank lines skipped)."""
    if keys is None:
        return []
    if isinstance(keys, (str, os.PathLike)) and Path(os.fspath(keys)).expanduser().is_file():
        lines = Path(os.fspath(keys)).expanduser().read_text().splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if isinstance(keys, str):
        return [keys]
    return [str(k) for k in keys]


def position_id(event: Mapping[str, Any]) -> str:
    """An event's position as an SSE id: ``<writer>-<seq>``."""
    return f"{event['writer']}-{event['seq']}"


def parse_position(text: Optional[str]) -> Optional[Dict[str, int]]:
    """``<writer>-<seq>`` back to a position (None when absent or not one)."""
    m = re.fullmatch(r"\s*(\d+)-(\d+)\s*", text or "")
    return {"writer": int(m.group(1)), "seq": int(m.group(2))} if m else None


def _sse(event: Mapping[str, Any]) -> bytes:
    return (f"id: {position_id(event)}\nevent: {event['kind']}\n"
            f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n").encode("utf-8")


class Service:
    '''A program as an HTTP service, independent of any server: ``handle``
    answers one request; ``serve`` runs the standard library's threaded
    server; ``asgi`` is an ASGI app (FastAPI's ``app.mount``, uvicorn).

    Parameters
    ----------
    program : AI function, module, or saved folder
        What is served (a folder is loaded with ``functai.load``).
    keys : list, str or file, optional
        Bearer keys a caller must send (a file: one per line). None: no
        key, which ``serve`` allows only on 127.0.0.1.
    store : optional
        Where conversations are kept (as ``fn.conversation(store=...)``);
        None keeps them in this process's memory.
    lm : str, optional
        The model every call uses, bound when serving starts.
    approvals : "owner" or "caller"
        Who answers a tool call the program's ``approve`` rule asks about:
        the owner (in their own process; the caller sees the turn waiting),
        or the caller, through the approvals route.
    trust : bool
        For a saved folder that runs code: load it (``functai.load(trust=True)``).
    '''

    def __init__(self, program: Any, *, keys: Any = None, store: Any = None, lm: Optional[str] = None,
                 approvals: str = "owner", trust: bool = False):
        if isinstance(program, (str, os.PathLike)):
            from .saved import load
            program = load(program, trust=trust)
        from .core import FunctAIFunc
        from .module import FunctAIModule
        if not isinstance(program, (FunctAIFunc, FunctAIModule)):
            raise TypeError(f"serve an AI function, a module, or a saved folder; not {type(program).__name__}")
        iface = program.interface
        opaque = [f["name"] for d in ("inputs", "outputs") for f in iface[d] if f.get("opaque")]
        if opaque:
            raise ServeError("serve-opaque", f"{program.__name__} cannot be served: {', '.join(opaque)} may hold "
                                             f"values with no JSON form, and only JSON crosses HTTP. Give "
                                             f"{'them' if len(opaque) > 1 else 'it'} a type")
        if approvals not in ("owner", "caller"):
            raise ValueError(f"approvals go to 'owner' or 'caller', not {approvals!r}")
        self.program = program if lm is None else _with_lm(program, lm)
        self.keys = _keys(keys)
        self.store = store
        self.lm = lm
        self.approvals = approvals
        self._conversations: Dict[str, Any] = {}
        self._lock = threading.Lock()

    # ----- describing it

    def describe(self) -> Dict[str, Any]:
        info = calllog.program_info(self.program)
        return {"functai_interface": FORMAT, "name": self.program.__name__, "kind": info["kind"],
                "version": info["version"], "interface": self.program.interface, "answer": info["answer"],
                "model": self.lm, "approvals": self.approvals}

    def openapi(self) -> Dict[str, Any]:
        iface = self.program.interface
        ins = {"type": "object", "properties": {f["name"]: f["shape"] for f in iface["inputs"]},
               "required": [f["name"] for f in iface["inputs"] if not f.get("optional")]}
        outs = {"type": "object", "properties": {f["name"]: f["shape"] for f in iface["outputs"]}}
        body = {"required": True, "content": {"application/json": {"schema": {
            "type": "object", "properties": {"inputs": ins}, "required": ["inputs"]}}}}
        return {"openapi": "3.1.0",
                "info": {"title": self.program.__name__, "version": self.describe()["version"],
                         "description": iface.get("description") or ""},
                "paths": {"/call": {"post": {"requestBody": body, "responses": {"200": {"description": "the answer",
                          "content": {"application/json": {"schema": {"type": "object", "properties": {
                              "call": {"type": "string"}, "outputs": outs, "value": {}}}}}}}}},
                          "/stream": {"post": {"requestBody": body, "responses": {"200": {
                              "description": "Server-Sent Events", "content": {"text/event-stream": {}}}}}}},
                "components": {"securitySchemes": {"key": {"type": "http", "scheme": "bearer"}}},
                "security": [{"key": []}] if self.keys else []}

    def form(self) -> str:
        iface = self.program.interface
        rows = "".join(f'<label>{html.escape(f["name"])}<br><textarea name="{html.escape(f["name"])}" rows="3">'
                       f'</textarea></label><br>' for f in iface["inputs"])
        return (f"<!doctype html><meta charset=utf-8><title>{html.escape(self.program.__name__)}</title>"
                f"<style>body{{font:16px system-ui;max-width:40em;margin:2em auto}}textarea{{width:100%}}</style>"
                f"<h1>{html.escape(self.program.__name__)}</h1><p>{html.escape(iface.get('description') or '')}</p>"
                f"<form id=f>{rows}<label>key <input name=__key type=password></label> <button>Ask</button></form>"
                f"<pre id=out></pre><script>"
                "f.onsubmit=async e=>{e.preventDefault();const d=new FormData(f),inputs={};"
                "for(const[k,v]of d)if(k!='__key'){try{inputs[k]=JSON.parse(v)}catch{inputs[k]=v}}"
                "const r=await fetch('call',{method:'POST',headers:{'content-type':'application/json',"
                "authorization:'Bearer '+d.get('__key')},body:JSON.stringify({inputs})});"
                "out.textContent=JSON.stringify(await r.json(),null,2)}</script>")

    # ----- answering

    def _authorized(self, headers: Mapping[str, str]) -> bool:
        if not self.keys:
            return True
        got = headers.get("authorization", "")
        if not got.lower().startswith("bearer "):
            return False
        given = got[7:].strip().encode("utf-8")
        return any(hmac.compare_digest(given, k.encode("utf-8")) for k in self.keys)

    def handle(self, method: str, path: str, headers: Mapping[str, str], body: bytes = b"") -> Response:
        """Answer one request (``headers`` with lower-case names)."""
        path = "/" + path.split("?", 1)[0].strip("/")
        headers = {k.lower(): v for k, v in headers.items()}
        try:
            if method == "GET" and path == "/":            # a form, which asks for the key itself
                return Response(200, self.form().encode("utf-8"), {"content-type": "text/html; charset=utf-8"})
            if not self._authorized(headers):
                return _json(401, {"error": {"type": "Unauthorized"}})
            if method == "GET" and path == "/interface":
                return _json(200, self.describe())
            if method == "GET" and path == "/openapi.json":
                return _json(200, self.openapi())
            data = json.loads(body.decode("utf-8") or "{}") if body else {}
            if not isinstance(data, dict):
                return _json(400, {"error": {"type": "BadRequest", "message": "the body is a JSON object"}})
            parent = headers.get("functai-parent")
            parent = parent if parent and _UUID.match(parent) else None
            parts = [p for p in path.split("/") if p]
            if method == "POST" and path == "/call":
                return self._call(data, parent)
            if method == "POST" and path == "/stream":
                return self._stream(data, parent)
            if parts and parts[0] == "conversations" and len(parts) >= 3 and parts[2] == "turns":
                return self._conversation(method, parts[1], parts[3:], data, headers, parent)
            return _json(404, {"error": {"type": "NotFound"}})
        except json.JSONDecodeError:
            return _json(400, {"error": {"type": "BadRequest", "message": "the body is not JSON"}})
        except InterfaceError as exc:
            return _error(422, exc, message=exc.code == "interface-input")
        except ConversationError as exc:
            status = {"turn-unknown": 404, "conversation-id": 400, "conversation-busy": 409}.get(exc.code, 409)
            return _error(status, exc, message=True)
        except FunctAIError as exc:
            return _error(409, exc)
        except Exception as exc:  # noqa: BLE001 — the caller sees the type, never the message (outside view)
            return _error(500, exc)

    def _inputs(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        inputs = data.get("inputs", {})
        if not isinstance(inputs, dict):
            raise InterfaceError("interface-input", None, "inputs is a JSON object of the program's inputs")
        return inputs

    def _context(self, parent: Optional[str]) -> Any:
        from .config import scoped
        from .conversations import _APPROVALS_TO
        caller = {**calllog.caller_of({}), "kind": "api"}
        return _Scope(scoped(caller=caller), _APPROVALS_TO, self.approvals, parent)

    def _call(self, data: Mapping[str, Any], parent: Optional[str]) -> Response:
        from .core import FunctAIFunc
        inputs = self._inputs(data)
        with self._context(parent):
            if isinstance(self.program, FunctAIFunc):
                p = self.program.predict(**inputs)
                value = p.get(self.program._spec().main)
                call_id = p.call_id
                outputs = {k: v for k, v in p.items() if k in {f["name"] for f in self.program.interface["outputs"]}}
            else:
                holder: List[str] = []
                value = _run_module(self.program, inputs, holder)
                call_id = holder[0] if holder else None
                from .interface import outputs_of
                ok, outputs = outputs_of(self.program.interface, value)
                if not ok:
                    outputs = {self.program.interface["outputs"][-1]["name"]: value}
        return _json(200, {"call": call_id, "outputs": {k: calllog.to_json(v)[0] for k, v in outputs.items()},
                           "value": calllog.to_json(value)[0]})

    def _stream(self, data: Mapping[str, Any], parent: Optional[str]) -> Response:
        inputs = self._inputs(data)
        with self._context(parent):
            s = self.program.stream(**inputs)
        return Response(200, self._events(s.events(view="outside"), s), {"content-type": "text/event-stream",
                                                                          "cache-control": "no-cache"})

    def _events(self, events: Iterable[Mapping[str, Any]], s: Any = None) -> Iterator[bytes]:
        try:
            for e in events:
                yield _sse(e)
        except BaseException:  # noqa: BLE001 — the call's own end is in the events; a reader gone stops it
            pass
        finally:
            if s is not None and not getattr(s, "done", True):
                s.close()

    # ----- conversations

    def conversation(self, cid: str) -> Any:
        with self._lock:
            chat = self._conversations.get(cid)
        if chat is None:
            chat = self.program.conversation(cid, store=self.store)
        return chat

    def _turn_json(self, t: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {"turn": t.id, "parent": t.parent, "state": t.state, "inputs": t.inputs}
        if t.state == "done":
            out["outputs"] = {k: v for k, v in t.outputs.items()
                              if k in {f["name"] for f in self.program.interface["outputs"]}}
            out["value"] = calllog.to_json(t.result)[0]
        if t.state == "waiting" and self.approvals == "caller":
            out["waiting"] = [{"invocation": a.invocation, "name": a.name, "input": calllog.to_json(a.input)[0],
                               "path": a.path} for a in t.waiting]
        if t.error:
            out["error"] = {k: v for k, v in t.error.items() if k in ("type", "code")}
        return out

    def _conversation(self, method: str, cid: str, rest: List[str], data: Mapping[str, Any],
                      headers: Mapping[str, str], parent: Optional[str]) -> Response:
        chat = self.conversation(cid)
        if method == "POST" and not rest:
            inputs = self._inputs(data)
            if data.get("after"):
                chat = chat.continue_from(data["after"])
            with self._context(parent):
                s = chat.stream(request_id=data.get("request_id"), **inputs)
            if data.get("wait"):
                from .streaming import Cancelled
                try:
                    s.result
                except (Exception, Cancelled):  # noqa: BLE001 — the turn's state says how it went
                    pass
            t = s.turn
            return _json(201, {**self._turn_json(t), "conversation": cid})
        if method == "GET" and not rest:
            return _json(200, {"conversation": cid, "turns": [self._turn_json(t) for t in chat.turns]})
        tid = rest[0]
        turn = chat.turn(tid)
        if method == "GET" and len(rest) == 1:
            return _json(200, self._turn_json(turn))
        if method == "GET" and rest[1:] == ["events"]:
            after = parse_position(headers.get("last-event-id"))
            return Response(200, self._events(turn.events(after, view="outside")),
                            {"content-type": "text/event-stream", "cache-control": "no-cache"})
        if method == "POST" and rest[1:] == ["stop"]:
            turn.stop()
            return _json(202, {"turn": tid, "stopping": True})
        if method == "POST" and len(rest) == 3 and rest[1] == "approvals":
            if self.approvals != "caller":
                return _json(403, {"error": {"type": "Forbidden", "message": "approvals go to the owner"}})
            verdict = data.get("verdict")
            if verdict not in ("yes", "no"):
                return _json(400, {"error": {"type": "BadRequest", "message": "verdict is 'yes' or 'no'"}})
            inv = int(rest[2])
            with self._context(parent):
                if verdict == "yes":
                    turn.approve(inv, resume=False)
                else:
                    turn.deny(inv, data.get("reason"), resume=False)
                again = chat.turn(tid)
                if not again._st.unanswered():
                    import contextvars
                    ctx = contextvars.copy_context()           # the caller's scope: approvals stay addressed to it
                    threading.Thread(target=ctx.run, args=(_quietly, again.resume), daemon=True).start()
            return _json(202, {"turn": tid, "verdict": verdict})
        return _json(404, {"error": {"type": "NotFound"}})

    # ----- front ends

    def serve(self, host: str = "127.0.0.1", port: int = 8080, *, block: bool = True) -> Any:
        """Run the standard library's threaded HTTP server. Without keys it
        listens only on this machine (127.0.0.1, ::1, localhost)."""
        if not self.keys and host not in ("127.0.0.1", "::1", "localhost"):
            raise ServeError("serve-keys", f"serving on {host} lets anyone on the network call "
                                           f"{self.program.__name__} and spend your model budget: give keys "
                                           f"(keys='keys.txt'), or serve on 127.0.0.1")
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _do(self) -> None:
                try:
                    n = int(self.headers.get("content-length") or 0)
                except ValueError:
                    n = -1
                if n < 0 or n > MAX_BODY:
                    self.send_response(413 if n > MAX_BODY else 400)
                    self.send_header("content-length", "0")
                    self.send_header("connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    return
                body = self.rfile.read(n) if n else b""
                r = service.handle(self.command, self.path, {k: v for k, v in self.headers.items()}, body)
                self.send_response(r.status)
                for k, v in r.headers.items():
                    self.send_header(k, v)
                if isinstance(r.body, (bytes, bytearray)):
                    self.send_header("content-length", str(len(r.body)))
                    self.end_headers()
                    self.wfile.write(r.body)
                    return
                self.send_header("connection", "close")
                self.end_headers()
                try:
                    for chunk in r.body:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    close = getattr(r.body, "close", None)
                    if callable(close):
                        close()
                self.close_connection = True

            do_GET = do_POST = _do

            def log_message(self, fmt: str, *args: Any) -> None:
                return None

        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        if not block:
            threading.Thread(target=server.serve_forever, daemon=True, name="functai-serve").start()
            return server
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return server

    @property
    def asgi(self) -> Callable[..., Any]:
        """An ASGI app: ``app.mount("/team", service.asgi)`` in FastAPI, or
        ``uvicorn.run(service.asgi)``. Each request is answered in a worker
        thread; events stream as they come."""
        service = self

        async def app(scope: Dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
            if scope["type"] == "lifespan":
                while True:
                    msg = await receive()
                    if msg["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif msg["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            if scope["type"] != "http":
                return
            body = b""
            while True:
                msg = await receive()
                body += msg.get("body", b"")
                if len(body) > MAX_BODY:
                    await send({"type": "http.response.start", "status": 413, "headers": []})
                    await send({"type": "http.response.body", "body": b""})
                    return
                if not msg.get("more_body"):
                    break
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            path = scope.get("path", "/")
            root = scope.get("root_path", "")
            if root and path.startswith(root):
                path = path[len(root):] or "/"
            r = await asyncio.to_thread(service.handle, scope["method"], path, headers, body)
            await send({"type": "http.response.start", "status": r.status,
                        "headers": [(k.encode(), v.encode()) for k, v in r.headers.items()]})
            if isinstance(r.body, (bytes, bytearray)):
                await send({"type": "http.response.body", "body": bytes(r.body)})
                return
            chunks: "queue.Queue[Any]" = queue.Queue()
            done = object()

            def pump() -> None:
                try:
                    for c in r.body:
                        chunks.put(c)
                finally:
                    chunks.put(done)

            threading.Thread(target=pump, daemon=True).start()
            try:
                while True:
                    c = await asyncio.to_thread(chunks.get)
                    if c is done:
                        break
                    await send({"type": "http.response.body", "body": c, "more_body": True})
                await send({"type": "http.response.body", "body": b""})
            except (asyncio.CancelledError, OSError):
                close = getattr(r.body, "close", None)
                if callable(close):
                    close()
                raise

        return app


class _Scope:
    """The settings a served call runs under: its caller, where approvals go,
    the remote caller's call as its parent."""

    def __init__(self, scoped: Any, var: Any, to: str, parent: Optional[str]):
        self.scoped, self.var, self.to, self.parent = scoped, var, to, parent

    def __enter__(self) -> "_Scope":
        self.scoped.__enter__()
        self.t1 = self.var.set(self.to)
        self.t2 = calllog.REMOTE_PARENT.set([self.parent] if self.parent else None)
        return self

    def __exit__(self, *exc: Any) -> None:
        calllog.REMOTE_PARENT.reset(self.t2)
        self.var.reset(self.t1)
        self.scoped.__exit__(*exc)


def _quietly(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except BaseException:  # noqa: BLE001 — the turn's records say how it went
        pass


def _run_module(program: Any, inputs: Dict[str, Any], holder: List[str]) -> Any:
    """Call a module and learn its call's id (a module returns only its value)."""
    s = program.stream(**inputs)
    value = s.result
    holder.append(s.call_id)
    return value


def _with_lm(program: Any, lm: str) -> Any:
    """The program with its model bound (an AI function's copy; a module runs
    every call under it)."""
    from .core import FunctAIFunc
    if isinstance(program, FunctAIFunc):
        return program.using(lm=lm)
    from .module import FunctAIModule
    bound = object.__new__(FunctAIModule)
    bound.__dict__.update(program.__dict__)
    original = bound._invoke_original

    def invoke(*args: Any, **kwargs: Any) -> Any:
        from .config import scoped
        with scoped(lm=lm):
            return original(*args, **kwargs)

    bound._invoke_original = invoke
    return bound


def serve(program: Any, *, host: str = "127.0.0.1", port: int = 8080, keys: Any = None, store: Any = None,
          lm: Optional[str] = None, approvals: str = "owner", trust: bool = False, block: bool = True) -> Any:
    '''Serve a program over HTTP: its interface, calls, streams and
    conversations, to callers who see only its boundary.

    Parameters
    ----------
    program : AI function, module, or saved folder
    host, port : where to listen. Without ``keys``, only 127.0.0.1.
    keys : list, str or file of keys, optional
        Callers send ``Authorization: Bearer <key>``.
    store : optional
        Where conversations are kept (default: this process's memory).
    lm : str, optional
        The model every call uses.
    approvals : "owner" or "caller"
    block : bool
        False: serve in a background thread and return the server
        (``server.shutdown()`` stops it).

    Returns
    -------
    the server (``http.server.ThreadingHTTPServer``)

    See Also
    --------
    remote : a served program, used like a local one.
    Service : the same, to mount in another app (``Service(...).asgi``).
    '''
    return Service(program, keys=keys, store=store, lm=lm, approvals=approvals, trust=trust).serve(
        host, port, block=block)


__all__ = ["Service", "serve", "Response", "position_id", "parse_position"]
