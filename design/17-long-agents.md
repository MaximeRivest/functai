# 17 — Long agents and pipelines

*Status, 2026-11-02: built in Python and TypeScript on the branch
`pipelines` (from `lm15-2`), with the contract and its cases
(`contract/cases/agents/`). R and Julia do not have it yet: they never read
`agents/` (contract/README.md, *Which cases each language passes*).*

## Why

A company rebuilding its AI backend on FunctAI (a pipeline of steps, some
of them browsing agents that click through a site for 150 to 300 steps)
listed seven things FunctAI lacked. Each was checked against the code and
the contract, then judged against what FunctAI is for: typed programs whose
every call can be **run, recorded, replayed, rated and improved**. A
request that makes that loop work for more kinds of program belongs here;
one that would pull FunctAI into another product's job does not.

| asked for | decision |
|---|---|
| a tool that answers with a picture | built: *What a tool returns* |
| steering a long tool loop (keep the last pictures, wrap up near a limit) | built: the `step` hook, `keep_pictures` |
| an event for "something happened" in plain code | built: `note` |
| spending limits | built as an opt-in plugin, `limits`, with prices the host gives |
| the call log somewhere other than a folder | built: a store of one's own |
| a run that waits for a webhook | built for tools: `later(key)` and `deliver` |
| recording plain code so a resumed run does not repeat it | **not built**, on purpose (below) |

## What was built

**Pictures from tools** (tools.md, *What a tool returns*). A tool returns
text, a value (JSON, as before), an lm15 part, or a list holding one. The
result is lmcc's tool step: text, or a list of lm15 parts. A part read from
a file is given its bytes, so the record is whole when another process
resumes the turn. The `tool_result` event keeps `output` as text (its text
parts) for every reader that exists, and adds `parts` only when there is
more than text: no format number changes, and old form makers drop the new
key, as streaming.md says they must.

**The `step` hook** (plugins.md, *The step hook*). Before each request after
the first in a tool loop: the tools offered (`[]` makes the model answer), a
`reminder` added after the step's last result, earlier results shown
otherwise (`replace`), or `stop`. Until now the only way was the `request`
hook, which marks the call not replayable. These changes are data, recorded
in the call's `changes`, and the request is still rendered by lmcc from the
turn, so its `request_hash` stays. The turn, the events and the
conversation records keep what the tools gave; only what the model is shown
changes. A reminder is a text part at the end of the last result rather than
a user message: every provider accepts it after tool results, and lmcc's turn
has no place for a message in the middle of a call.

**`keep_pictures(last, at_once)`.** Pictures before the newest `last` become
a short text. `at_once` drops several at a time, because every change to an
earlier result breaks a provider's prompt cache from that point on
(Anthropic's own computer-use loop does the same).

**`limits(tokens, cost, seconds, prices, then)`.** Opt-in; nothing is limited
by default. What a tree spent is given to `before_call` and `step` handlers
(`Spent`); the plugin is built on them, so anyone can write another.
Exact or nothing: a count that is not known (a model without a price, a
request whose cache counts are missing while its cache rates differ) counts
as reached, because a money guard that silently stops guarding is worse than
one that stops early. With `then="answer"` a loop at a limit, or at its last
step, is told to answer instead of failing with `StepLimit`. FunctAI ships
no price list: prices change, and a stale one would mislead quietly.

**`note(name, data)`.** A new event kind for plain code's progress ("screen 4
captured"). Not a value: the call log does not keep it. Its kept form keeps
the name and drops the data when the call's content is not whole.

**A store of one's own for the call log** (calls.md). `log_calls` may be any
object with `append(record)` and `read()`. Every reader takes it where it
takes a folder. TypeScript waits for an async `append` before the call
returns, so a serverless function that ends with its call has logged it.
`MemoryCalls` for tests.

**Results that come later** (tools.md, *A result that comes later*). A tool
returns `later(key)`; in a conversation's turn the step's other tools still
run, then the turn waits, saved; `chat.deliver(key, result)` from any
process records the result and resumes the turn. It reuses everything
approvals built (waiting records, leases, replay by site and invocation),
and its safety comes from the tool records: the tool ran once and is never
run again. A `delivered` record belongs to the conversation, not the turn,
so a webhook faster than the turn's own bookkeeping is not lost; a turn that
finds its results delivered after writing `waiting` resumes itself, and the
lease's conditional append lets only one process go on.

## What was not built, and why

**Recording plain code inside a module** (Temporal's activities, Inngest's
`step.run`). Recording a step is easy; resuming a stopped run after the code
changed is the hard part, and it is most of what workflow engines are (their
versioning APIs exist for it). FunctAI would grow half an engine. The
pipeline asked for already has an orchestrator: the app's own state
machine, webhooks and reconciler. Each step is a FunctAI program; the app
runs the steps. `caller` ties a run's calls together in the log, and each
step's `version` is its recipe. Inside one agent, work that must not repeat
is a tool that `changes` things, and work that ends elsewhere is `later`.

**A waiting module body** (`wait_for(key)` in plain code). It would invite
exactly the bug above: code before the wait runs again on resume. Waiting
is offered where the record makes it safe: a tool.

**A price list in FunctAI or lm15.** See `limits`. Python's bake reads
models.dev for its estimates (dated, cached); a host may build `prices`
from it.

## Trade-offs, stated

- **Call log size.** Each exchange keeps the whole request it sent. A
  300-step browsing loop with screenshots writes every screenshot it still
  shows, once per request: with `keep_pictures(3)`, about 3 pictures per
  request. Storing each picture once (by its hash) would need a new call
  record format; not done.
- **Events carry picture bytes** when content is whole (`parts`). A host
  that does not want them in its journal uses a view, or `log_content`.
- **Spending counts this process's tree.** Replies from the reply cache, and
  those a resumed turn reuses, are not counted again. A limit across
  several separate calls (a run of several steps) is the host's: pass what
  is left (`cost=12 - spent_so_far`).
- **Limits are checked between requests**: one request may go past one.
- **A resumed turn runs its `step` handlers again**: a time limit may
  decide otherwise, and then a new request is sent (it costs, never
  misleads).
- **R and Julia** have none of this yet. Their harnesses do not read
  `agents/`; the cross-language check does not exercise it. The usual
  catch-up branch follows (as stages 1.2 to 5 did).
- **TypeScript ratings to an async store** are not awaited by `rate()`
  (it stays synchronous); `flushCalls()` waits for them.
