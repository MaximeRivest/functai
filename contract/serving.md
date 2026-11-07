# Serving a program (format 1)

A program served over HTTP is called, streamed and conversed with by
callers who see, unless its owner chooses otherwise, only its
**boundary** (the outside view, [streaming.md](streaming.md), *Views*);
a served program is a program again on the caller's side (`remote`),
logged there and here as one call tree (Python
`python/functai/serving.py`, `remote.py`).

## What callers see

The owner chooses, when serving, the view its callers are given (`view`):

| view | a caller sees |
|---|---|
| `outside` (the default) | the program's boundary: its answer, its answer's text as it is written, approvals addressed to it, the notes it declares `outside`, its end; an error's type and code, never its message |
| `kept` | what the owner's log keeps (`log_content`): with content whole, the default, every event and value of the tree; an error's type and code |
| `full` | every event and value, as the owner sees them: thinking, tool calls and results, helpers' calls, the model's stream (`piece`), the outputs FunctAI adds (`reasoning`), an error's message |
| the owner's own | what that view shows (streaming.md, *Views*); an error's type and code |

`outside` is the default because a served program's callers are often
not its owner (a page in a customer's browser, another team): what the
model thought, a tool's result, the program's prompt or an error's
message can hold what they must not see. Choosing another view is the
owner's decision, for callers they trust as themselves.

A view shapes what is sent, never what runs: the call, its log and its
record are the same whatever the view. A turn's events are read from the
conversation's store, which keeps the kept form (streaming.md, *Where
each form can be read*): there, `full` gives the kept form. `/stream`
gives the view live, from the process that runs the call.

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
(`interface-input`) or the view is `full`: an error's message may quote
what the caller must not see.

| route | answer |
|---|---|
| `GET /interface` | `{"functai_interface": 1, "name", "kind", "version", "interface", "answer", "model", "approvals", "view"}`: the program's interface with a format number (it is served outside a saved manifest); `view` the callers' (`outside`, `kept`, `full`, or `own` for the owner's own) |
| `GET /openapi.json` | the same, as OpenAPI 3.1 |
| `POST /call` `{"inputs": {…}}` | `{"call", "outputs", "value"}`: `outputs` the interface's, and with the `full` view also those FunctAI adds (`reasoning`, `calls`); a refused input `422` |
| `POST /stream` `{"inputs": {…}}` | Server-Sent Events of the callers' view: `id: <writer>-<seq>`, `event: <kind>`, `data: <the event>` |
| `POST /conversations/<id>/turns` `{"inputs", "request_id"?, "after"?, "wait"?}` | `201` `{"turn", "parent", "state", "conversation", …}`: the turn is recorded before it runs; `after` continues from that turn; `wait` answers once it ended or waits |
| `GET /conversations/<id>/turns` | the branch's turns: `{"turn", "parent", "state", "inputs", "outputs"?, "value"?}` (`outputs` as `/call` gives them) |
| `GET /conversations/<id>/turns/<turn>` | one turn; `waiting` lists its approvals when they go to callers |
| `GET /conversations/<id>/turns/<turn>/events` | its events (the callers' view, made from the store's kept form), SSE; a reconnect's `Last-Event-ID: <writer>-<seq>` resumes after that event |
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
