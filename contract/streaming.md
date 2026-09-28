# Streaming (format 2)

A stream is **the same call, watched while it is made**. It asks the
model for the same thing, retries the same way, runs the same tools,
writes the same line to the call log, and ends with the same value (or
the same error) as calling the program. Streaming adds a view, never a
second behaviour.

This document is the contract for that view: the events a call tree
makes, in what order, their JSON form, what of them may be kept, and how
they are read again later, by the same process or another one. The
Python implementation is `python/functai/streaming.py`;
`schema/event.schema.json` checks an event's JSON form; `cases/events/`
pins replay, following, the kept form, the rules a store keeps and a
writer keeping a log in a journal.

## Words

- **Call tree**: a call and every call made inside it (a module's steps,
  a tool that calls an AI function, an escalation), as one process runs
  it. Its **outermost call** is the one the process started it with.
- **Log**: the events of one call tree, numbered in the order they
  happen. Its id is its outermost call's id.
- **Writer**: the process that numbers a log's events. A log has one at
  a time: the process running the tree. When it stops before the log's
  end, a later writer may continue the log (*Continuing a log*). Writers
  are numbered: `1` for the first, one more for each later one.
- **Stream**: one call, watched from its start: the events of that call
  and of every call inside it. A stream shows part of a log (all of it
  when it watches the outermost call).
- **Request**: one request to a model. A call makes one, or several:
  retries, tool steps, escalation.
- **Answer**: the output that is the call's answer (`program.answer` in the
  call log). Other outputs (a `reasoning` field) are shown too, as their
  own fields.
- **Form**: which events of a log, and which of their values, a reader
  gets. The **whole** log has every event with every value; it exists only
  in the process running the tree. The **kept** form is what may be kept
  outside it (*The kept form*). A **view** is what one kind of reader may
  see (*Views*).

## Events

Every event says which log it is in and where (`tree`, `writer`, `seq`,
`after`), when it was numbered (`at`), the call it is about (`call`, the
call log's id) and that call's program (`function`, the program's name).

| key | meaning |
|---|---|
| `functai_event` | the format, `2`. Its presence says the object is an event of this format. |
| `kind` | what happened (the table below). |
| `tree` | the log's id: the id of the call tree's outermost call. |
| `writer` | the number of the writer that numbered the event: `1`, or the number a later writer was given (*Continuing a log*). |
| `seq` | the event's place in its log: `1` for the first event, then one more for each event its writer numbers, with no gaps in the whole log. A later writer numbers on from the last *kept* event, so it may use again a number an earlier writer used for an event that was never kept. `(tree, writer, seq)` names one event, in every form. |
| `after` | the `seq` of the event before this one in the form being read (`0` for the first). In the whole log it is `seq − 1`; in other forms it skips what the form leaves out. |
| `at` | when the writer numbered the event: RFC 3339 UTC with exactly six fraction digits, the call log's time format, read from the writer's clock and never less than the `at` before it in the log (a clock set back repeats the last time). Order is `seq`'s, never `at`'s. |
| `call` | the id of the call it is about (a UUIDv7, as in the call log). |
| `function` | that call's program's name. |

| kind | when | fields |
|---|---|---|
| `started` | a call begins | `parent` (the call it runs in, or null), `root` (the call log's `root`), `program` (the call log's `program` object), `inputs` (JSON values), `content` (`true`; `false` in a form that left values out, then with `omitted`: *The kept form*), `saw` (the call log's `saw`: [calls.md](calls.md)) |
| `request` | the call begins a request to a model | `request` (its number: `1` for the call's first request, one more for each further one), `model` (the model asked, as the call log's exchange names it, or null) |
| `text` | a piece of an output's text is written | `field`, `answer` (true when the field is the answer), `text` |
| `thinking` | a piece of the model's own thinking that no output reads | `text` |
| `tool_call` | the model asked for a tool, and the request is complete | `id` (the call's id, as lmcc names it: the provider's, or one lmcc assigned), `name`, `input` |
| `tool_result` | the tool ran | `id`, `name`, `output` (text, as the model sees it) |
| `retry` | the model is asked again for this call's answer | `reason` (a sentence), `wait` (seconds before asking, or null) |
| `done` | the call ended with a value | `value`: what the call returned, as JSON (an AI function's answer, as its code returned it; a module's output, or its outputs by name when it has several). A value with no JSON form is described as in the call log. |
| `failed` | the call ended with an error | `error` `{"type", "message", "code"?}` as in the call log |

A later stage may add kinds (an approval, stage 4) and keys; what a
reader does with what it does not know is in *Formats*.

Laws:

1. **Order.** A call's events come in the order they happened: `started`,
   then its `request`, `text`, `thinking`, `tool_call`, `tool_result` and
   `retry` events and the events of the calls inside it, then exactly one
   `done` or `failed`. A log's first event is its outermost call's
   `started`; its last is that call's `done` or `failed`, and nothing
   follows it. When calls inside run at the same time, their events are
   interleaved in the order the writer received them, and `seq` numbers
   that one order. (A log whose writer stops before its end is
   unfinished: *Keeping a log while it is written*.)
2. **Text is exact.** Within one request, the concatenation of a field's
   `text` pieces is that field's raw text in the reply (lmcc kernel §8: a
   piece is shown only once no later byte can change it). Pieces are never
   revised. What they mean as a typed value is known only at `done`.
3. **A field's text is its latest request's.** A call's `request` event
   empties every field of the call: the text of a field so far is the
   concatenation, in order, of its pieces since the call's latest
   `request`. A `retry` empties them at once too, before the `request`
   that follows it (a retry may wait first). Tool results do not empty
   anything themselves: the `request` after them does.
4. **Escalation is a retry.** When a first model is unsure and another
   answers (`escalate_to`), the call shows `retry`. When the other is a
   model, a `request` to it follows; when it is an AI function, its call
   is a child of this one, and its answer is this call's answer.
5. **Tool calls are whole.** A `tool_call` is shown once the model has
   finished asking (its input complete), never piece by piece.
6. **Nothing is invented.** A reply that arrives whole (from the reply
   cache, or from a model that cannot stream) is shown as one `text` piece
   per field. A layout whose reader cannot read a reply in pieces (lmcc
   `plan.describe()["streaming"]["mode"] == "buffered"`, such as
   `adapter="json"`) shows its fields at the end of the request.
7. **One happening, one event.** A tree has one log. A stream opened on a
   call inside it (Python `fn.stream()` in a module's body) shows the
   same events, with the same `tree`, `writer` and `seq`: it starts at
   that call's `started` (its `after` is `0`), and each `after` skips the
   events of calls outside it.
8. **A request is an exchange.** A call's *n*th `request` event is its
   call record's *n*th exchange: failed attempts and replies from the
   cache included (a cached reply is a request answered at once).
   Retries inside the transport (lm15 sending again after a dropped
   connection) are neither.

## Replaying

Reading a form's events in order, from its first, and applying laws 1 to
4 gives, after each event, what a person watching that form live saw
then: which calls have started and ended, each field's text so far. So a
log read back from where it was kept shows what was seen live.

A reader keeps, for each call it has seen start: whether it ended, and
each field's text so far. `started` adds the call with no fields; `request`
and `retry` empty every field of their call; `text` appends to its field;
`done` and `failed` mark the call ended. `thinking`, `tool_call` and
`tool_result` are shown as they come and change no field. An event of a
kind the reader does not know changes nothing. The log is **finished**
once its outermost call has ended. `cases/events/replay-*.json` pin this.

**Resuming.** A reader that has events up to one it names by `writer`
and `seq` asks its source for the events after that one, in the same
form, and gets them in order. It goes on from the state it had there: the
events after it alone do not give the text so far. A source that does
not have that event in that form (a store never kept it: a piece of a
field it does not keep, an event a writer numbered but never had kept)
refuses `event-unknown`; the reader then starts again from the beginning,
in a form the source can give (*Where each form can be read*). It never
waits for a chain the source cannot give.

## Following a log

A reader that follows a form live (a page, another process) keeps, for
each tree it follows, the last event it has (its `writer` and `seq`;
`0` and `0` at first), and for each event it receives, in this order:

- an event whose format it does not know: it cannot know what the
  event's numbers mean, and stops following that source (it may start
  again from a source it can read);
- `writer` less than its last's: **stale**, dropped (a writer the log has
  left behind, whose event came late);
- the same `writer` and a `seq` not greater than its last: a
  **duplicate**, dropped (a writer or a transport may send an event
  again);
- `after` equal to its last's `seq`: the **next** event; it takes it;
- a later `writer` and an `after` less than its last's `seq`: the log was
  continued by a later writer from an earlier point (*Continuing a log*).
  The reader drops what it has after `after` (it **rewinds**: the state
  it had at `after`, or, when it did not keep that, the form read again
  from the beginning), then takes the event. When it does not have
  `after` itself, it is not reading the form it thinks, and starts again
  from the beginning;
- anything else: events were **lost**. It reads the same form again after
  its last event (*Resuming*).

It takes the place of an event of a kind it does not know (the next
event's `after` may name it) and applies nothing for it. `after` makes
this work in every form, though only the whole log has no gaps in `seq`.
`cases/events/follow-*.json` pin it.

## The kept form

What is kept of a log outside its process follows each call's
`log_content`, as the call's line in the call log does ([calls.md](calls.md),
*Content*): a value the call log would not keep is not kept here either,
nor anything that tells its size piece by piece. When a call's content is
whole, its events are kept as they are. When it is not, its events are
kept thus:

| kind | kept when the call's content is not whole |
|---|---|
| `started` | `content: false`; `inputs` holds only the inputs kept (absent when none is); `omitted` names the fields not kept, as the call record does |
| `request` | as is (it holds no value) |
| `text` | a field that is kept: as is. Any other: not kept at all |
| `thinking` | not kept |
| `tool_call` | `id` and `name`, no `input`; `content: false` |
| `tool_result` | `id` and `name`, no `output`; `content: false` |
| `retry` | `wait`, no `reason`; `content: false` |
| `done` | `value` when every output it holds is kept (an AI function's answer; a module's outputs); else no `value`, and `content: false` |
| `failed` | no `error.message`; `content: false` |

The kept form keeps each event's `writer` and `seq`, and sets `after` to
the `seq` of the kept event before it. So a watcher in another process
sees the tree's shape (who called what, when, how many requests) and the
values the log keeps, and nothing of the rest: neither its text nor the
length of each piece (a sequence of piece lengths is enough to guess much
of a reply's text). The call record's `sizes` gives each field's total.

What makes a form (the kept form, a view, an observer's feed) keeps only
what it knows: an event of a kind it does not know is left out, and so
is a key it does not know on an event it knows, whatever the content,
since it cannot know whether they hold a value. The writer knows its own
events, so its kept form keeps them. `cases/events/kept-*.json` pin this.

## Views

A view is what one kind of reader may see of a log: the owner sees the
whole log; a caller who sees only a program's boundary sees less (stage
3 names the views and who chooses them). A view:

- keeps each event's `writer` and `seq`, and sets `after` to the event
  before it in the view;
- shows a call's `started` before any other event of that call it shows,
  and its `done` or `failed` when it shows the call;
- shows every `request` and `retry` of a call whose text it shows (it may
  leave out the retry's `reason`, with `content: false`): they hold no
  value, and without them its reader would keep text the call voided;
- may leave out events and values; it never alters a value it shows and
  never invents one; like the kept form, it leaves out what it does not
  know.

Whether a view may show an event of a call inside a module as the
module's own (a child's field as the module's answer while it is written,
or an approval three calls down as the boundary's) is stage 3's: until
then, a view that shows only the module shows its `started`, then its
`done`.

**Retention is not disclosure.** The kept form is about what may be
*kept*. Sending events live to a reader the host allows (a page, a
worker's parent process) keeps nothing, and follows the view the host
chooses for that reader, not `log_content`. The whole form leaves the
process only that way: through the stream its caller watches, which
decides where it goes.

## Where each form can be read

The process running a tree can give any form of its log, while it runs.
A store keeps the kept form, so it can give the kept form and views made
from it, and no other. A reader of another form (the whole log, or a view
made from it that shows values the kept form lacks) can resume it only
from the process that runs the tree. From a store, it resumes only after
an event the store has; otherwise the store refuses `event-unknown` and
the reader starts again from the beginning in a form the store gives,
dropping what it had (values that were shown live and never kept are
then gone, as retention said they would be).

## Keeping a log while it is written

FunctAI gives a log's events to two kinds of receiver, each set like
`log_calls` (a program's settings, a block, `configure`), whether or not
anyone watches the call. A call made with either is watched, by it; its
behaviour does not change (laws above; the call log is the same).

**Observers** receive the kept form of every event of every call in their
scope, in order, as it happens (their `after` skips what is outside their
scope; a reader of an observer's feed keeps a last event per tree): a log
for watching (a socket, a telemetry exporter, a page). An observer never
gets in the way: if it fails, the implementation warns once and stops
giving it events. A slow observer does not slow the call: an
implementation may hand events to it from another thread, in order, and
may drop them (the observer then sees a loss: *Following a log*).

**Journals** keep logs for watching: a folder, a store. A journal keeps
the kept form of the whole log of every tree whose outermost call starts
in its scope; the journal a tree has is decided when its outermost call
starts. A call inside a tree does not start another log: a journal set
only around calls inside a tree (a block in a module's body) keeps
nothing of it, and the implementation warns once (a required journal
refuses `journal-scope` when that call starts, before it runs).

A journal is the watching log, under `log_content`. It is not a
conversation's memory (what later turns are shown: stage 2's
conversation store, under its own setting), nor the checkpoint a waiting
turn resumes from (the model's steps as lmcc keeps them, provider
thinking and tool outputs as parts, what each tool was given: stage 4's,
under its own retention). A kept `tool_call` has no `input` when content
is not whole, and a kept `thinking` is gone: resuming a model from the
events alone would be a guess. Each of the three is its own record.

- **Appending.** The writer appends each event in order, alone or in
  batches (a batch is kept whole or not at all), with the `after` the
  journal checks (*The rules a store keeps*): an append says where it
  goes. The journal **answers** an append once it has kept it as surely
  as it says it keeps things (a store declares that: stage 2). An event
  is **confirmed** when the journal answers kept or duplicate; **refused**
  when it answers anything else; **unanswered** when no answer comes. An
  unanswered append may have been kept: the writer cannot tell. It keeps
  every event not confirmed and sends it again (a resend of a kept event
  is a duplicate, which confirms it). After a refusal it appends nothing
  more to that log: `event-conflict` means another writer has it; the
  others are faults.
- **Best effort** (the default): the call never waits. A journal that
  fails does not stop the call; the implementation warns once, and keeps
  sending; if it gives up, the log stays kept up to its last confirmed
  event, unfinished. Every event is shown to readers as it is made.
- **Required** (the host asks for it): the call waits, until every event
  up to it is confirmed, at three **barriers**: after its outermost
  call's `started` (before any code or request runs); after each
  `tool_call` (before that tool runs); and after the outermost call's
  `done` or `failed` (before the call returns or raises). Calls inside
  the tree have no start or end barrier of their own: the tree's are
  enough, since appends are in order.

**A required journal that does not confirm.** A call's **outcome** is what
its program did: its value, or its error. It is decided before the
journal is asked, and nothing the journal answers changes it: the `done`
or `failed` event records it, and so does the call record.

1. At the start or before a tool, when an event up to the barrier is
   refused or stays unanswered (after the writer's own resends), the
   call does not go past the barrier: the code or tool does not run, and
   the outcome is the error `JournalError` with code `journal-barrier`.
   Its `failed` event is made as usual, and the writer sends what is not
   confirmed, then it (not after a refusal).
2. At the end, the outermost call's `done` or `failed` is confirmed: the
   call returns its value, or raises its error, as without a journal.
3. At the end, it is not confirmed: the call raises `JournalError` with
   code `journal-end`, holding the outcome (Python `err.outcome`) and
   saying `journal`: `"refused"` (the journal answered that it did not
   keep it) or `"unknown"` (no answer: it may be kept). The call record
   keeps the outcome (its outputs, or its error), and says `journal` with
   the same word. It never records a failure the call did not have.
4. With a required journal, the log's last event is given to readers
   (the stream, observers) only once it is confirmed; every other event
   is given as it is made. So a reader never sees an end that a store
   does not hold. When the end is not confirmed, the stream's reader
   sees no end (its result raises the `JournalError`), as when a writer
   stops.

The caller settles `"unknown"` by reading the log from the journal: if
it is finished, the outcome was kept. The writer may go on sending the
end after the call has raised (a duplicate if it was kept).
`cases/events/journal-*.json` pin this, one step at a time.

A log whose outermost call's `done` or `failed` is kept is **finished**.
Until then it is **unfinished**: its writer may still be running, or may
have stopped.

## Continuing a log

A later writer may continue an unfinished log (a process that resumes a
waiting turn after a restart: stage 4):

1. It **claims** the log from the store: the store gives it a writer
   number, one more than any it gave for that log, and names the last
   kept event. From then on the store refuses every event of an earlier
   writer (`event-conflict`): the earlier writer is **fenced**, even
   when it is still running and its event would fit the chain. Each
   claim fences every earlier one, so two writers never hold one number.
2. It reads the kept log up to that event, and numbers on from it: its
   first event's `seq` is one more than that event's, with its own
   `writer`, and `after` that event's `seq`. The numbers an earlier
   writer used after it (for events never kept) are used again, for
   other events: `writer` tells them apart.
3. Each call's next request number is one more than the highest in the
   kept log's `request` events of that call; `at` is never less than the
   last kept event's.

Readers following the log across the change rewind (*Following a log*).
Who may claim a log (a lease), when a writer counts as stopped, what the
later writer does with calls that were running (a tool that may have
run), and how a log is ended by someone other than a writer are stage
2's and stage 4's.

## The rules a store keeps

Anything that keeps logs for others to read keeps them by these rules.

**Claiming** a log for a later writer: refused `event-unknown` when the
store has no event of that log, `event-after-end` when the log is
finished; otherwise the store gives the next writer number and names the
last kept event, as *Continuing a log* says.

**Appending** the events of one log, in order, checking each (a batch is
kept whole or not at all), in this order:

| appending an event | result |
|---|---|
| not an event of this format (it does not pass the schema), `tree` not the log it is appended to, or `seq` not greater than `after` | refused `event-malformed` |
| a `seq` already kept, and the same event (equal canonical JSON) | nothing changes (`duplicate`): a writer may send an event again after a failure |
| a `seq` already kept, and another event | refused `event-conflict`: two writers are writing one log |
| in a log that has events, a `writer` other than the log's (the last number given; `1` before any claim) | refused `event-conflict`: an earlier writer, fenced, or a number the store never gave |
| any event after the log's last (its outermost call's `done` or `failed`) | refused `event-after-end` |
| `after` greater than the last kept `seq` (`0` for a new log) | refused `event-gap`: events are missing before it |
| `after` less than the last kept `seq` | refused `event-conflict`: the log went on another way |
| a first event that is not the log's outermost call's `started` from writer `1` (`call` not the `tree`, another kind, another writer) | refused `event-start` |
| otherwise | kept |

**Reading** a log after an event named by `writer` and `seq` gives the
kept events after it, in order; after `0`, all of them. A store that
does not have that event refuses `event-unknown` (*Resuming*).

`cases/events/store-*.json` pin these rules.

## Closing

Closing a stream before its call ended cancels the call: the request in
progress is stopped at its next piece, no new request or call starts,
and the call ends with the error `Cancelled` (in the call log too).
Stopping a request does not promise the provider stops generating or
billing at once. A stream nobody closes runs to its end, like a call.
Only the process running a call can close its stream; a reader of a kept
log that stops reading (a page closed) changes nothing. Stopping a call
from another process is stage 2's.

## The answer so far

Implementations offer the answer so far, read from its text (law 3): for
a text answer, the text; for a record or a list, the JSON read so far
(complete values, and the text of a string still being written; a number,
`true`, `false` or `null` only once complete); for any other answer,
nothing until `done`. It is provisional: `done` has the typed, checked
value.

## In the call log

A streamed request's exchange has `"streamed": true` and `first_delta`,
the seconds from sending the request to its first piece of content (the
wait a person feels). A cancelled call's `error` is `{"type":
"Cancelled"}`.

## Formats

Format 2 (2026-09-28) added `functai_event`, `tree`, `writer`, `seq`,
`after`, `at`, the `request` kind, `started`'s `root`, `program`,
`content`, `omitted` and `saw`, the kept form, views, journals and the
rules a store keeps; a `tool_call`'s `id` is never null. Format 1 events
(no `functai_event` key) were only ever shown inside the process that
made them, never kept; the schema still accepts them, so that a reader
meeting one knows it.

**What a later writer may add, and what a reader does with it.** A later
stage may add kinds of event and keys of an event, in format 2, when they
never change what the kinds and keys above mean (an approval, a tool
invocation's id on `started`). The schema accepts them. A reader that
replays or follows skips what it does not know (it takes an unknown
kind's place and applies nothing). What makes a form leaves it out
(*The kept form*): a reader may ignore, but a maker must not pass on what
it cannot judge. A change that would alter what a known kind or key means
(a new kind that empties fields, say) is a new format, and a reader
stops at an event whose format it does not know.
