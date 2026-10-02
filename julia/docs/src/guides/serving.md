# Serving, and programs served elsewhere

A program served over HTTP is called, streamed and conversed with by callers who see only its boundary (the outside view); a served program is a program again on the caller's side ([`remote`](@ref)), logged there and here as one call tree (contract/serving.md). Python's, Julia's and any other FunctAI's servers and callers speak the same routes.

## Serving

```julia
server = serve(team; port = 8080, keys = "keys.txt", block = false)   # HTTP.jl's server, on a task
close(server)                                                          # stops it
serve("saved/team"; lm = "gpt-4.1-mini", keys = ["k-1"])              # a saved folder (from any language), blocking
```

- **Keys**: callers send `Authorization: Bearer <key>` (a list, one key, or a file of one per line), compared in constant time. Without keys it listens only on this machine: any other address refuses `serve-keys`, since anyone could spend your model budget.
- **What cannot be served**: a program with an untyped (opaque) input or output (`serve-opaque`): only JSON crosses HTTP.
- `store`: where served conversations are kept (as `conversation(…; store)`); `lm`: the model every call uses; `approvals = :caller` lets the caller answer the program's approvals (`:owner`, the default, answers them in the owner's own process).

| route | answer |
|---|---|
| `GET /interface` | `{"functai_interface": 1, "name", "kind", "version", "interface", "answer", "model", "approvals"}` |
| `GET /openapi.json` | the same, as OpenAPI 3.1; `GET /` is a form that asks for the key |
| `POST /call` `{"inputs": {…}}` | `{"call", "outputs", "value"}`; a refused input is `422` (`interface-input`, naming the field) |
| `POST /stream` | Server-Sent Events of the outside view (`id: <writer>-<seq>`) |
| `POST /conversations/<id>/turns` `{"inputs", "request_id"?, "after"?, "wait"?}` | `201`: the turn, saved before it runs |
| `GET /conversations/<id>/turns`, `…/turns/<turn>`, `…/turns/<turn>/events` | the branch's turns, one turn, its events (a reconnect's `Last-Event-ID` resumes) |
| `POST …/turns/<turn>/stop`, `…/turns/<turn>/approvals/<invocation>` `{"verdict", "reason"?}` | stop it wherever it runs; answer an approval (`approvals = :caller`) |

Errors are `{"error": {"type", "code"?, "field"?}}`, with a message only when it quotes the caller's own input: an error's message may quote what the caller must not see.

**The outside view** shows a caller the program's `started` (without its file and line), its answer's text as it is written, the approvals addressed to it, and its `done`, or its `failed` with the error's type and code and no message: never a helper's answer, a tool's input or output, a thinking, or why a request was retried. A program whose answer is one of its AI function's shows that function's text as it is written: `@program answer_from = reply function support(…)`. `FunctAI.outside(events)` is the same view of any log, and `eachevent(turn; view = :outside)` of a turn's.

`FunctAI.Service(program; keys, …)` is the same service without a server: `FunctAI.handle(service, method, path, headers, body)` answers one request, for mounting in another HTTP framework.

## A program served elsewhere

```julia
team = remote("https://example.org/team"; key = ENV["TEAM_KEY"])
team("I was charged twice for order B-2210.")
team.(tickets.message)                        # a column, concurrently
evaluate(team, labelled)                      # as a local program
for piece in stream(team, "Where is my parcel?")   # the server's outside view, as it is written
    print(piece)
end
```

Its interface is the server's: its inputs are bound and checked here before anything is sent (`InterfaceError`), whatever language serves it. Each call is logged here (`program.kind = "remote"`, `program.remote` the URL, the served program's version) and there, and the server's record names this call as its `parent` (the `FunctAI-Parent` header), so a reader holding both logs sees one tree. A served program's conversations are kept by its server: talk to them over HTTP. A wrong key, or a server that answers an error, throws `FunctAI.RemoteError` with the server's code (`remote-401`).
