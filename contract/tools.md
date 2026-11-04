# Tools that ask first (format 1)

A tool says what it does to the world, and a person can be asked before
it runs: at once (a function the program is given), or later, from any
process (the turn waits, saved). A tool answers with text or with
pictures, sounds and documents, now or later. A turn that stopped (to
wait, or because its process died) goes on without paying for a model
answer twice and without running a tool twice (Python
`python/functai/tools.py`, `engine.py`). `cases/tools/` pin the rules
and the refusal's text; `cases/agents/` what a tool returns and a turn
that waits for a later result.

## What a tool returns

What a tool returns becomes its **result**, the tool step the model is
shown (lmcc's tool step: lm15 parts):

- a text: that text;
- a part (a picture, a sound, a video, a document: lm15's `image`,
  `audio`, `video`, `document` parts), or a list holding at least one: a
  **list of parts**, in order: each text a text part, each part as it is,
  each other value a text part of its JSON. A part read from a file is
  given its bytes (`data`), so the result is whole wherever it is read
  again; a part by web address stays one;
- anything else: its JSON, as text (as before 2026-11-02).

In data (a conversation's `tool` record, a `tool_result` hook's `output`,
a result given for a tool that may have run, a result delivered later) a
result is a text or a list of lm15 parts in lm15's JSON form
(`{"type": "image", "media_type": "image/png", "data": "<base64>"}`).
Other part kinds (a tool call, thinking) are refused. A provider that
cannot take a picture in a tool's result is lm15's to adapt, and its
response says what it changed. Each exchange of the call log holds the
request it sent, pictures included ([calls.md](calls.md)).

## Effects

A tool declares its **effects**: `reads` (it only looks) or `changes` (it
changes something: writes, sends, pays). A tool that declares nothing has
unknown effects, which every rule treats as `changes`: forgetting to
declare is safe. An AI function used as a tool reads, unless one of its
own tools changes things or declares nothing.

## Notes a tool makes

A tool may declare the notes its code makes (Python
`@functai.tool(notes={...})`, TypeScript `tool(name, { notes }, run)`),
as a module does ([programs.md](programs.md), *Notes*). They are the
interface's `notes` of every AI function that has the tool, and are
checked as that function's: a browsing agent's `page captured`, made by
its screenshot tool, is typed for whoever shows the agent's progress.

## What a tool needs

A tool may say what its code reaches (Python `@functai.tool(needs={...})`,
TypeScript `tool(name, { needs }, run)`): `{"domains": [...], "secrets":
[...]}`, host names (`*.` for every name under one) and the names of the
secrets it uses, never their values. Behind a wall that lets through less,
the function that has it is refused before its first call; the wall the
program may reach counts in its setup ([calls.md](calls.md), *Setups*,
*Reach*). A module says the same of its own code (`needs`).

## Approval

Approval is a plugin ([plugins.md](plugins.md)): the built-in `approval`
plugin's `tool_call` hook, which runs after every other plugin's, on the
input they left. Any plugin's `tool_call` may ask a person the same way
(*Who answers*), with a question of its own; a tool call may then be asked
about by several plugins, each question addressed by the plugin's name.

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
`id`, `invocation`, `name`, `input`, `effects`, `path`, `to`: whom it is
addressed to, `owner` or `caller`, `plugin`: which plugin asks, and
`question`, its reason, when it gives one), then, once answered,
`approved` (`verdict` `yes` or `no`, `by`, `reason`, `plugin`). The kept
form drops an approval's `input` and `question` and an answer's `reason`
when the call's content is not whole ([streaming.md](streaming.md)).

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
- an approval finds its answer in the `approval` records (by site,
  invocation and plugin).
- a tool's result is recorded as the model was shown it, after the
  `tool_result` hooks: replayed, it is shown again as it was, with no hook
  run again.
- a tool whose result comes later and was delivered is given it (*A result
  that comes later*).

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

## A result that comes later

A tool may start something that ends elsewhere, minutes later (a job, a
cloud browser, a person's reply by mail), and say its result comes
**later**: it returns `later(key)` (Python `functai.later("harvest:8f2c")`).
`key` names that one result in its conversation (a text; put an id in
it).

- **Only in a conversation's turn.** Anywhere else nothing could wait for
  it: the call refuses `later-unavailable` (the tool ran; what it started
  may be running).
- **It ran.** A `tool` record `later` (`site`, `invocation`, `id`, `name`,
  `input`, `effects`, `key`) is written when it returns; it is never run
  again. The log shows a `later` event (`id`, `invocation`, `name`, `key`).
- **The step goes on.** Every other tool call of the same model reply runs.
  Then, for each later result, a `delivered` record with its key settles
  it: when every one is delivered, the call goes on with them now; else
  the turn **waits**, as for an approval ([conversations.md](conversations.md)):
  a `waiting` record whose `later` lists them (`site`, `invocation`, `id`,
  `name`, `key`, `path`), and the call and every call it is inside stop
  without ending; the caller gets `Waiting` (`turn-waiting`) with what it
  waits for.
- **Delivering.** Any process that opens the conversation delivers a
  result (Python `chat.deliver(key, result)`): `result` is what the tool
  would have returned, turned into a result as *What a tool returns*
  says, and kept in a `delivered` record (`key`, `output`, `by`?) that
  belongs to no turn, so a result delivered before the tool said it would
  come later is used when it does. The first `delivered` record of a key
  is the one kept: delivering again changes nothing (a webhook may be sent
  twice). Then a turn that waits and has nothing left to wait for (every
  approval answered, every result delivered) goes on in the delivering
  process (unless asked not to), and the delivery returns its answer.
- **Going on.** The turn resumes as any waiting turn does (*Resuming*):
  the tool's result is its delivery, given to the `tool_result` hooks,
  recorded `done` and shown (`tool_result`). A turn that still waits for
  a result refuses to resume (`turn-state`).
- **No lost wake-up.** A turn that found a result undelivered writes its
  `waiting` record, then reads the conversation again: when every result
  it waits for is delivered by then, it goes on itself. Two processes may
  both try: the lease's conditional append lets one go on, and the other
  is told `turn-state`, which a delivery takes as "already going on".

This is the one way FunctAI waits for the world. Code of a module that
starts work outside a tool (and so outside these records) runs again when
the turn resumes: make such work a tool that changes things, or keep it
in the host's own workflow.
