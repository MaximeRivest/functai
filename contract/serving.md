# Serving a program (format 1)

A program served over HTTP is called, streamed and conversed with by
callers who see only its **boundary** (the outside view,
[streaming.md](streaming.md), *Views*); a served program is a program
again on the caller's side (`remote`), logged there and here as one call
tree (Python `python/functai/serving.py`, `remote.py`).

## What cannot be served

- A program with an opaque field ([programs.md](programs.md)): only JSON
  crosses HTTP (`serve-opaque`).
- Listening beyond the machine (any address but 127.0.0.1, `::1`,
  `localhost`) with no keys (`serve-keys`): anyone could spend the
  owner's model budget.

## Keys

With keys, every route but `GET /` (a form, which asks for the key)
needs `Authorization: Bearer <key>`; a missing or wrong key is `401`.
Keys are compared in constant time.

## Routes

JSON in, JSON out; errors are `{"error": {"type", "code"?, "field"?}}`,
with a `message` only when it quotes the caller's own input
(`interface-input`): an error's message may quote what the caller must
not see.

| route | answer |
|---|---|
| `GET /interface` | `{"functai_interface": 1, "name", "kind", "version", "interface", "answer", "model", "approvals"}`: the program's interface with a format number (it is served outside a saved manifest) |
| `GET /openapi.json` | the same, as OpenAPI 3.1 |
| `POST /call` `{"inputs": {…}}` | `{"call", "outputs", "value"}`; a refused input `422` |
| `POST /stream` `{"inputs": {…}}` | Server-Sent Events of the outside view: `id: <writer>-<seq>`, `event: <kind>`, `data: <the event>` |
| `POST /conversations/<id>/turns` `{"inputs", "request_id"?, "after"?, "wait"?}` | `201` `{"turn", "parent", "state", "conversation", …}`: the turn is recorded before it runs; `after` continues from that turn; `wait` answers once it ended or waits |
| `GET /conversations/<id>/turns` | the branch's turns: `{"turn", "parent", "state", "inputs", "outputs"?, "value"?}` |
| `GET /conversations/<id>/turns/<turn>` | one turn; `waiting` lists its approvals when they go to callers |
| `GET /conversations/<id>/turns/<turn>/events` | its events (outside view, from the store's kept form), SSE; a reconnect's `Last-Event-ID: <writer>-<seq>` resumes after that event |
| `POST /conversations/<id>/turns/<turn>/stop` | `202`: the turn stops wherever it runs |
| `POST /conversations/<id>/turns/<turn>/approvals/<invocation>` `{"verdict": "yes"\|"no", "reason"?}` | `202`, when approvals go to callers (else `403`); the turn resumes once nothing else waits |

**Approvals** go to the owner (the default: the caller sees the turn
waiting, and the owner answers from their own process) or to the caller
(`approvals="caller"`: approval events are addressed `to: "caller"`, shown
in the outside view, and answered through the route).

## One call tree across two logs

A caller that is a FunctAI call sends `FunctAI-Parent: <its call id>`.
The served call records it as its `parent` (a parent in another log, as
[calls.md](calls.md) allows). The caller's record is a program of kind
`remote` with `program.remote` (where it is served) and the served
program's version. So a reader holding both logs sees one tree.

## A remote program

Built from `GET /interface`: its inputs are bound and checked on the
caller's side before anything is sent ([programs.md](programs.md)), its
answer is the served call's `outputs`; it maps over tables and evaluates
as a local program does.
