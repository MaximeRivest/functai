---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Serve it over HTTP

*A program as a web service, for callers who see only what it answers; and a served program used from Python like a local one.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, module
from typing import Literal
```

Any AI function or `@module` can be served: its inputs and outputs become
an HTTP API (JSON in, JSON out, with an OpenAPI description), its
conversations become routes, and people who call it see its answer and
nothing of how it was made. On their side, `functai.remote` turns the
address back into a program.

## Serve

```python
@ai
def team(message: str) -> Literal["billing", "shipping", "technical", "other"]:
    """Which team should answer this customer message?"""
    ...

KEY = "a-long-random-key"                 # in real use: one per caller, kept in a file
server = functai.serve(team, port=0, keys=[KEY], block=False)
url = f"http://127.0.0.1:{server.server_port}"
url
```

```output
'http://127.0.0.1:44749'
```

`block=False` serves from a background thread and returns the server,
which is what a notebook needs; in a script, `functai.serve(team,
port=8080, keys="keys.txt")` serves until you stop it. `port=0` lets the
system pick a free port.

From a terminal, a [saved](saving.md) program is served without writing
any code:

```{.bash .no-run}
functai serve team/ --lm gpt-4.1-mini --keys keys.txt --port 8080
```

`--lm` binds the model every call uses. A program saved from Python runs
its author's code when it is loaded, so it is served only with `--trust`,
after you read that code (as `functai.load(..., trust=True)`); a folder
written as data alone (an AI function saved from another language) needs
none. `--store folder/` keeps
conversations on disk; `--host 0.0.0.0` listens beyond this machine,
which needs keys.

## Use it from Python

```python
team_there = functai.remote(url, key=KEY)
team_there("I was charged twice for order B-2210.")
```

```output
'billing'
```

`team_there` is a program again: its inputs and outputs are the served
program's, read from the server, and checked here before anything is
sent. It maps over a table and is evaluated like a local program:

```python
tickets = functai.datasets.tickets()
team_there.map(tickets.slice_head(n=4), threads=4, progress=False).select("message", "pred_result")
```

```output
# dpyr dataframe · source: polars · showing 4 of 4 rows
┌─────────────────────────────────────────────────────────────────────┬─────────────┐
│ message                                                             ┆ pred_result │
│ ---                                                                 ┆ ---         │
│ str                                                                 ┆ str         │
╞═════════════════════════════════════════════════════════════════════╪═════════════╡
│ Hi, my order A-1042 still hasn't arrived and it's been three weeks. ┆ shipping    │
│ The mug arrived in pieces.                                          ┆ shipping    │
│ I was charged twice for order B-2210, please fix this.              ┆ billing     │
│ How do I change the email on my account?                            ┆ technical   │
└─────────────────────────────────────────────────────────────────────┴─────────────┘
```

The tokens a call used are counted in the server's log, not here: the
caller does not see them.

## Use it from anything

The API is plain HTTP, so any language calls it. Each request carries
the key as `Authorization: Bearer <key>`:

```{.bash .no-run}
curl -s http://127.0.0.1:8080/call \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"inputs": {"message": "My parcel is late"}}'
```

The same request from Python's standard library:

```python
import json, urllib.request, urllib.error

def http(method, path, data=None, key=KEY):
    request = urllib.request.Request(url + path, method=method,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")

http("POST", "/call", {"inputs": {"message": "My parcel is late"}})
```

```output
(200, {'call': '01a0fcea-a6a3-7023-b63c-b5b993ef635f', 'outputs': {'result': 'shipping'}, 'value': 'shipping'})
```

`call` is the call's id in the server's call log: a caller who reports a
wrong answer can name it.

| route | what it does |
|---|---|
| `GET /interface` | the program described: its name, version, inputs and outputs |
| `GET /openapi.json` | the same, as OpenAPI 3.1 (for client generators and API tools) |
| `GET /` | a form in the browser, to try it |
| `POST /call` `{"inputs": {...}}` | one call: `{"call", "outputs", "value"}` |
| `POST /stream` `{"inputs": {...}}` | the same call, as Server-Sent Events while it is written |
| `POST /conversations/<id>/turns` | a turn of a conversation (below) |
| `GET /conversations/<id>/turns[/<turn>]` | the turns, or one turn |
| `GET /conversations/<id>/turns/<turn>/events` | a turn's events (Server-Sent Events; a reconnect resumes) |
| `POST /conversations/<id>/turns/<turn>/stop` | stop a running turn |
| `POST /conversations/<id>/turns/<turn>/approvals/<n>` | answer an approval, when callers may (below) |

A wrong input is refused before any model is asked, with the field and
the reason (`422`). A wrong key gets `401`:

```python
http("POST", "/call", {"inputs": {}}), http("GET", "/interface", key="wrong")[0]
```

```output
((422, {'error': {'type': 'InterfaceError', 'code': 'interface-input', 'field': 'message', 'message': "team: input 'message' is required"}}), 401)
```

## What a caller sees

A caller sees the program's boundary: its answer, the answer's text as
it is written, approvals addressed to them, and the end. Never what a
helper inside answered, a tool's input or output, a model's thinking,
why a request was retried, or an error's message (which might quote
something they must not see: only its type and code). You, the owner,
see everything in your own [call log](call-log.md).

```python
@ai
def topic(message: str) -> Literal["billing", "shipping", "other"]:
    """What the message is about."""
    ...

@ai
def answer(message: str, topic: str) -> str:
    """Answer the customer in one short sentence."""
    ...

@module(answer_from=answer)              # answer's text, shown as support's while it is written
def support(message: str) -> str:
    return answer(message, topic(message))

support_server = functai.serve(support, port=0, keys=[KEY], block=False)
support_there = functai.remote(f"http://127.0.0.1:{support_server.server_port}", key=KEY)

s = support_there.stream("Where is my parcel B-2210?")
s.result, sorted({event["kind"] for event in s.events()}), {event.get("function") for event in s.events()}
```

```output
('Your parcel B-2210 is currently in transit and should arrive within 3-5 business days.', ['done', 'request', 'started', 'text'], {'support'})
```

Only `support` appears: `topic` and `answer` are inside it. Locally,
`support.stream(...).events(view="outside")` shows exactly what a caller
gets (see [Watch it being written](streaming.md#what-a-caller-may-see)).

## Conversations

A conversation is addressed by an id; each `POST` is a turn, recorded
before it runs. `"wait": true` answers once the turn has ended (without
it, the answer comes at once, with the turn's id, and the turn runs on):

```python
@ai
def tutor(message: str) -> str:
    """Tutor a student in fractions. One short sentence."""
    ...

import tempfile
folder = tempfile.mkdtemp()               # where conversations are kept
tutor_server = functai.serve(tutor, port=0, keys=[KEY], store=folder, block=False)
url = f"http://127.0.0.1:{tutor_server.server_port}"

http("POST", "/conversations/ana/turns", {"inputs": {"message": "Hi, I'm Ana. What is 1/2 + 1/4?"}, "wait": True})[1]["value"]
```

```output
'To add 1/2 and 1/4, first find a common denominator (4), then convert 1/2 to 2/4 and add: 2/4 + 1/4 = 3/4.'
```

```python
http("POST", "/conversations/ana/turns", {"inputs": {"message": "What's my name?"}, "wait": True})[1]["value"]
```

```output
'Your name is Ana.'
```

The conversation is kept in `store`, so you can open it in your own
process, read it, and rate its answers:

```python
[t.inputs["message"] for t in tutor.conversation("ana", store=folder).turns]
```

```output
["Hi, I'm Ana. What is 1/2 + 1/4?", "What's my name?"]
```

A `request_id` you choose makes a send safe to repeat: the same id sent
twice (a double click, a retry after a timeout) is one turn. `after`
continues from an earlier turn, by its id: a branch, as
`chat.continue_from(turn)` makes one.

## Tools that ask first

A served program's [tools that change things](tools.md#tools-that-change-things)
can wait for a person. By default the owner answers (`approvals="owner"`):
the caller sees the turn waiting, and you approve it from your own
process. Here the owner's process is this notebook:

```python
@functai.tool(effects="changes")
def refund(order_id: str, amount: float) -> str:
    """Refund an amount to the customer's card."""
    return f"refunded {amount} for {order_id}"

@ai(tools=[refund], approve="changes")
def desk(message: str) -> str:
    """Help the customer. Refund lost orders in full; order A-1042 cost 39.00."""
    ...

desk_server = functai.serve(desk, port=0, keys=[KEY], store=folder, block=False)
url = f"http://127.0.0.1:{desk_server.server_port}"

status, turn = http("POST", "/conversations/sam/turns", {"inputs": {"message": "Order A-1042 was lost. Refund it, please."}, "wait": True})
turn["state"], turn["turn"] == desk.conversation("sam", store=folder).turns[-1].id
```

```output
('waiting', True)
```

The owner sees what waits, and answers:

```python
waiting = desk.conversation("sam", store=folder).turns[-1]
waiting.waiting[0].name, waiting.waiting[0].input
```

```output
('refund', {'order_id': 'A-1042', 'amount': 39.0})
```

```python
waiting.approve(by="owner")
```

```output
'The full amount of $39.00 for order A-1042 has been refunded due to the lost order. If you need any further assistance, please let me know.'
```

The caller reads the turn again and finds it done:

```python
http("GET", f"/conversations/sam/turns/{turn['turn']}")[1]["state"]
```

```output
'done'
```

With `approvals="caller"`, the approval is the caller's instead: it is
listed in the turn (`waiting`, with each tool call's `invocation`, name
and input), and answered with `POST
.../turns/<turn>/approvals/<invocation>` `{"verdict": "yes"}` (or `"no"`,
with a `reason` the model is told). A plain `POST /call` has nobody to
wait for, so a tool that needs approval refuses (`409`,
`approval-required`) before it runs.

## One call tree across two logs

When a FunctAI program calls a served one, its call is sent along
(`FunctAI-Parent`), and the server's record names it as its parent. Read
together, your log and the server's are one call tree:

```python
@module
def route(message: str) -> str:
    return "to " + team_there(message)

log = tempfile.mkdtemp()
functai.configure(log_calls=log)        # for this process: the server's threads too
route("The app crashes when I log in")
functai.configure(log_calls=False)
functai.calls(folder=log).select("program", "call", "parent")
```

```output
# dpyr dataframe · source: polars · showing 3 of 3 rows
┌─────────┬──────────────────────────────────────┬──────────────────────────────────────┐
│ program ┆ call                                 ┆ parent                               │
│ ---     ┆ ---                                  ┆ ---                                  │
│ str     ┆ str                                  ┆ str                                  │
╞═════════╪══════════════════════════════════════╪══════════════════════════════════════╡
│ route   ┆ 01a0fcea-e647-7056-9909-60e62477f1b3 ┆ null                                 │
│ team    ┆ 01a0fcea-e9fd-73ff-ab27-70db836bcc3c ┆ 01a0fcea-e647-7056-9909-60e62477f1b3 │
│ team    ┆ 01a0fcea-e9fe-77cc-b26d-a692ee16c71a ┆ 01a0fcea-e9fd-73ff-ab27-70db836bcc3c │
└─────────┴──────────────────────────────────────┴──────────────────────────────────────┘
```

The server runs in this notebook, so both sides log to one folder:
`route`, then `team` as you called it (a remote program), then `team` as
the server ran it. (A `with functai.configure(...)` block would not reach
the server: it applies to the threads started inside it, and the
server's are not.)

## In a web app

`functai.Service(program, ...)` is the same service without a server of
its own: `.asgi` is an ASGI app, to mount in FastAPI or Starlette, or to
run with uvicorn.

```{.python .no-run}
from fastapi import FastAPI

app = FastAPI()
app.mount("/team", functai.Service(team, lm="gpt-4.1-mini", keys=["..."]).asgi)
# uvicorn app:app  →  POST /team/call, GET /team/interface, ...
```

## Safe by default

Without keys, `serve` listens only on this machine. Any other address
needs them, because anyone who reaches the service spends your model
budget:

```python
try:
    functai.serve(team, host="0.0.0.0", port=0, block=False)
except functai.ServeError as error:
    print(error.code, "·", error)
```

```output
serve-keys · serving on 0.0.0.0 lets anyone on the network call team and spend your model budget: give keys (keys='keys.txt'), or serve on 127.0.0.1
```

- **Keys** are compared in constant time. A key file has one key per
  line; give each caller their own, so one can be withdrawn.
- **HTTPS.** The server speaks plain HTTP, and keys travel in a header:
  beyond this machine or a private network, put it behind a proxy that
  terminates TLS (Caddy, nginx, a cloud load balancer).
- **Only JSON crosses.** A program with an input or output that may have
  no JSON form (`Any`, or no annotation, on a `@module`) is refused with
  `serve-opaque`: give it a type, or `functai.JSON` for "any JSON".
- **One model, chosen by you.** `lm=` (or `--lm`) binds the model when
  serving starts: a caller cannot choose another.

```python
for s in (server, support_server, tutor_server, desk_server):
    s.shutdown()
```
