# Tools that ask first (format 1)

A tool says what it does to the world, and a person can be asked before
it runs: at once (a function the program is given), or later, from any
process (the turn waits, saved). A turn that stopped (to wait, or because
its process died) goes on without paying for a model answer twice and
without running a tool twice (Python `python/functai/tools.py`,
`engine.py`). `cases/tools/` pin the rules and the refusal's text.

## Effects

A tool declares its **effects**: `reads` (it only looks) or `changes` (it
changes something: writes, sends, pays). A tool that declares nothing has
unknown effects, which every rule treats as `changes`: forgetting to
declare is safe. An AI function used as a tool reads, unless one of its
own tools changes things or declares nothing.

## Approval

The setting `approve` (in every layer where settings are: a program, a
block, the process, a conversation, one turn) is:

- absent: no tool call is asked about (the default);
- a function: asked about each tool call the rule `changes` selects, at
  once; it answers yes (`true`), no (`false`), or no with a reason (a
  text);
- a rule, asked about by a person later: `changes` (tools that change
  things or declare nothing), `all`, or a list of entries, each a tool's
  name, an approval path, or a path's ending (`answer/refund`).

A tool call's **approval path** is the names of the calls from the
outermost one to the call that asked, then the tool's
(`support/answer/refund`): names, not ids, so a host writes rules before
any call exists. Its **invocation** is its number among the tool calls of
the call that asked (1, 2, … across every step: lmcc's ids repeat across
replies); `tool_call` and `tool_result` events carry it, and so does every
call a tool makes (its `started` event and its record: `invocation`).

When a person is asked, the call's log shows `approval` (the tool call's
`id`, `invocation`, `name`, `input`, `effects`, `path`, and `to`: whom it is
addressed to, `owner` or `caller`), then, once answered, `approved`
(`verdict` `yes` or `no`, `by`, `reason`). The kept form drops an
approval's `input` and an answer's `reason` when the call's content is
not whole ([streaming.md](streaming.md)).

**A refusal is an answer the model sees**: the tool's result is `The
person did not allow this call.`, then ` Reason: ` and the reason when
one was given; the model may try something else.

**Who answers a rule.**

- In a conversation's turn: the turn stops and **waits** (a `waiting`
  record, [conversations.md](conversations.md)); its call and every call
  it is inside stop without ending (no `failed` event, no record: the
  log stays unfinished); the caller gets `Waiting` (`turn-waiting`). Any
  process that opens the conversation answers (`approval` record) and
  resumes it.
- On a stream without a conversation: the call waits in its process for
  the stream's answer.
- A plain call has nobody to ask: it refuses `approval-required`, before
  the tool runs.

## Keeping what may have happened

In a turn, a tool that changes things (or declares nothing) is recorded
`started` before it runs, and every tool `done` with its result after:
flushed when the store is durable. A required journal
([streaming.md](streaming.md)) waits for the tool call before a tool that
changes things runs, and only then: a tool that only reads runs without
waiting.

## Resuming

A turn that `waits` (every approval answered) or was `interrupted` goes on
in the process that claims it (a `lease` record, appended conditionally,
with `attempt` one more). Its program runs again, with the same inputs
and the same earlier turns:

- a request it made before gets the reply its `reply` records keep, in
  order, instead of being sent (no model answer is paid for twice); a
  module's code, run again, makes the same requests when it is
  deterministic given its inputs and replies (when it is not, a new
  request is sent: it costs, never misleads);
- a tool call finds its result in the `tool` records by the calling
  call's `site` and its `invocation`: it does not run again;
- a tool whose last record is `started` **may have run**: the turn does
  not go on until a person says what it returned (`given`, with its
  output) or asks it to run again (`rerun`); it refuses `turn-unfinished`
  otherwise. A tool is never run again on its own.
- an approval finds its answer in the `approval` records.

The log goes on as a later writer's ([streaming.md](streaming.md),
*Continuing a log*): the process claims the turn's log from the store and
numbers on from its last kept event. What the turn does again while
replaying is not shown again: its events are held back until it does
something it had not done (a request with no kept reply, a tool that
runs, an approval asked or newly answered, its end); then the calls
still open that it started while replaying are shown starting (the
outermost call's start is in the log already), and everything after as
it happens. Its request numbers go on from the log's.

The call's record is written again, with the same `id` and `writer` (the
later writer's number); a reader takes, for one id, the record of the
highest writer ([calls.md](calls.md)).

A turn that waits or was interrupted may instead be **abandoned** (its
`ended` record says `abandoned`).
