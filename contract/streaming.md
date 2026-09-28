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
pins replay, following, the kept form, the rules a store keeps, a
writer keeping a log in a journal, and which receivers a tree gets.

## Words

- **Call tree**: a call and every call made inside it (a module's steps,
  a tool that calls an AI function, an escalation), as one process runs
  it. Its **outermost call** is the one the process started it with.
- **Log**: the events of one call tree, numbered in the order they
  happen. Its id is its outermost call's id.
- **Writer**: the process that numbers a log's events. A log has one at
  a time: the process running the tree. When it stops before the log's
  end, a later writer may continue the log (*Continuing a log*). Writers
  are numbered: `1` for the first; a later one gets its number from the
  store it claims the log from.
- **Position**: an event of a log, named by the writer that numbered it
  and its `seq`: `{"writer": 1, "seq": 57}`. With the log's `tree`, it
  names one event in every form. Two positions are the same only when
  both numbers are.
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
| `seq` | the event's place in its log: `1` for the first event, then one more for each event its writer numbers, with no gaps in the whole log. A later writer numbers on from the last *kept* event, so it may use again a number an earlier writer used for an event that was never kept: a `seq` alone does not name an event, its position (`writer` and `seq`) does. |
| `after` | the position of the event before this one in the form being read, or `null` for the form's first event. In the whole log it names the event numbered just before; in other forms it skips what the form leaves out. It is compared as a whole, never by its `seq` alone. |
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
   that call's `started` (its `after` is `null`), and each `after` skips
   the events of calls outside it.
8. **A request is an exchange.** The `request` events a writer makes
   for a call are, in order, the exchanges of the call record it writes:
   failed attempts and replies from the cache included (a cached reply is
   a request answered at once). With one writer, a call's *n*th `request`
   is its record's *n*th exchange. Retries inside the transport (lm15
   sending again after a dropped connection) are neither. (A call
   continued by a later writer: *Continuing a log*.)

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

**Resuming.** A reader that has events up to one it names by its
position asks its source for the events after that one, in the form it
has been reading, and gets them in order. It goes on from the state it
had there: the events after it alone do not give the text so far. A
source that does not have that event in that form (a store never kept
it: a piece of a field it does not keep, an event a writer numbered but
never had kept) refuses `event-unknown`; the reader then starts again
from the beginning, dropping what it has, in a form the source can give
(*Where each form can be read*). It never waits for a chain the source
cannot give. A reader that goes on in **another form** (the whole log's
reader, from a store that keeps the kept form) starts again from the
beginning, even when the source has its last event: a position names a
place in the log, not what the reader was shown before it, and the new
form would never update, nor void, a field only the old one showed.

## Following a log

A reader that follows a form live (a page, another process) keeps, for
each tree it follows, the positions of the events it has (the last is
its **last**; none at first), and for each event it receives, in this
order:

- an event whose format it does not know: it cannot know what the
  event's numbers mean, and stops following that source (it may start
  again from a source it can read);
- `writer` less than its last's: **stale**, dropped (a writer the log has
  left behind, whose event came late);
- the same `writer` as its last and a `seq` not greater than its last's:
  a **duplicate**, dropped (a writer or a transport may send an event
  again);
- `after` the position of its last event (`null` when it has none): the
  **next** event; it takes it;
- a later `writer`, and an `after` that names an event it has (or is
  `null`): the log was continued by a later writer from that event
  (*Continuing a log*). The reader drops what it has after that event
  (it **rewinds**: the state it had there, or, when it did not keep
  that, the form read again from the beginning), then takes the event;
- anything else: events were **lost**. It reads the same form again after
  its last event (*Resuming*).

Every comparison is of positions, writer and `seq` together. A reader
that compared a `seq` alone would take another writer's event as the
next, or rewind to an event of the wrong writer, and end with a state no
form of the log has; nor does a later `writer` alone say that the events
it has come before the one it receives (a claim may give a number no
event carries, and a reader may miss every event of a writer). When it
cannot tell, it has lost events, and the source settles it.

It takes the place of an event of a kind it does not know (the next
event's `after` may name it) and applies nothing for it. `after` makes
this work in every form, though only the whole log has no gaps in `seq`.
`cases/events/follow-*.json` pin it, `follow-09` to `13` for readers that
miss events where a later writer took over.

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
| `failed` | `error` with only `type` and `code`; `content: false` |

The kept form keeps each event's `writer` and `seq`, and sets `after` to
the position of the kept event before it. So a watcher in another process
sees the tree's shape (who called what, when, how many requests) and the
values the log keeps, and nothing of the rest: neither its text nor the
length of each piece (a sequence of piece lengths is enough to guess much
of a reply's text). The call record's `sizes` gives each field's total.

What makes a form (the kept form, a view, an observer's feed) keeps only
what it knows: an event of a kind it does not know is left out, and so
is a key it does not know on an event it knows, whatever the content,
since it cannot know whether they hold a value. The same holds inside
the objects this contract defines that an event carries: an `error`
keeps only `type`, `message` and `code` (`message` only when the call's
content is whole), a `program` only the keys the call log names, and a
`saw` entry with a key it does not know becomes `{}` (an entry no reader
knows, holding nothing: what the call saw stays not known, and nothing
of the entry is passed on). Values (`inputs`, `value`, a tool's `input`
and `output`) are the program's, kept or left out whole by the table
above. The writer knows its own events, so its kept form keeps them.
`cases/events/kept-*.json` pin this.

## Views

A view is what one kind of reader may see of a log: the owner sees the
whole log; a caller who sees only a program's boundary sees less (stage
3 names the views and who chooses them). A view:

- keeps each event's `writer` and `seq`, and sets `after` to the
  position of the event before it in the view;
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
from the process that runs the tree. Going to a store, it is going on in
another form: it starts again from the beginning in a form the store
gives, dropping what it had, whether or not the store has its last event
(values that were shown live and never kept are then gone, as retention
said they would be). `follow-10` pins it.

A later writer has the events of the writers before it only as they were
kept (it read them from the store: *Continuing a log*). So its process
gives them only in the kept form, and views made from it: every form it
gives is the kept log up to its claim, then its own events. A reader of
another form that asks it for the events after an earlier writer's event
is refused `event-unknown`, and starts again. A reader that follows live
across the change needs nothing more: the rules of *Following a log*
keep what it has up to the event the later writer names, all of which
happened, and drop the rest.

## Keeping a log while it is written

FunctAI gives a log's events to two kinds of receiver, set where
`log_calls` is (a program's settings, a block, `configure`), whether or
not anyone watches the call. A call made with either is watched, by it;
its behaviour does not change (laws above; the call log is the same).

How the layers combine is policy, and policy is the host's: a program
cannot remove what a host set around it.

- **Observers add up.** A call's observers are those of every layer
  around it: a program's own observer is given events beside the host's,
  never instead of them.
- **One journal per tree, and a required one holds.** A tree has one
  journal (its claims and fencing are that store's). The closest layer
  that sets one (or sets none) decides, as for every setting, except
  that a **required** journal cannot be replaced, weakened to best
  effort, or removed by a closer layer (a program's own setting inside a
  host's block or `configure`): the tree's outermost call is then
  refused `journal-policy` when it starts, before it runs. A closer layer
  may name the same required journal, or replace a best-effort one.
  `cases/events/receivers-01-layers.json` pins both.

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
  sending; if it gives up, the log is kept at least up to its last
  confirmed event, and may hold more (an unanswered append may have been
  kept, the end included: `journal-11`). Every event is shown to readers
  as it is made.
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
   code `journal-end`, holding the outcome (Python `err.outcome`), naming
   the event that records it by its position (`err.event`), and saying
   `journal`: `"refused"` (the journal answered that it did not keep it)
   or `"unknown"` (no answer: it may be kept). The call record keeps the
   outcome (its outputs, or its error), and says `journal` with the same
   word. It never records a failure the call did not have.
4. With a required journal, the log's last event is given to readers
   (the stream, observers) only once it is confirmed; every other event
   is given as it is made. So a reader never sees an end that a store
   does not hold. When the end is not confirmed, the stream's reader
   sees no end (its result raises the `JournalError`), as when a writer
   stops.

The caller settles `"unknown"` by reading the log from the journal and
looking for the event `err.event` names, by its position: when the log
holds it, the outcome was kept; when it does not and the log is
finished, another writer ended the log (it claimed it meanwhile) and the
caller's outcome was not kept: the log's end is that writer's, not the
caller's; when it does not and the log is unfinished, it was not kept
(yet: the writer may go on sending the end after the call has raised,
and is answered `duplicate` if it was kept, `event-conflict` if it has
been fenced). A finished log alone does not say the caller's outcome was
kept. `cases/events/journal-*.json` pin this, one step at a time
(`journal-13`: another writer ends the log).

A log whose outermost call's `done` or `failed` is kept is **finished**.
Until then it is **unfinished**: its writer may still be running, or may
have stopped.

## Continuing a log

A later writer may continue an unfinished log (a process that resumes a
waiting turn after a restart: stage 4):

1. It **claims** the log from the store: the store gives it a writer
   number, one more than any it gave for that log, and the position of
   the last kept event. From then on the store refuses every event of an
   earlier writer (`event-conflict`): the earlier writer is **fenced**,
   even when it is still running and its event would fit the chain, and
   even when it sends again an event the store holds. Each claim fences
   every earlier one, so two writers never hold one number.
2. It reads the kept log up to that event, and numbers on from it: its
   first event's `seq` is one more than that event's, with its own
   `writer` (the number its claim gave, never one worked out from the
   events: a claim may give a number no event carries), and `after` that
   event's position. The numbers an earlier writer used after it (for
   events never kept) are used again, for other events: `writer` tells
   them apart.
3. Each call's next request number is one more than the highest in the
   kept log's `request` events of that call; `at` is never less than the
   last kept event's. It never adds text to a request it did not make: a
   call it goes on with that was in a request makes a new `request` (or
   a `retry`) before any `text` of its own (law 2).

Every form of the log, from then on, is the kept log up to the claimed
event, then the later writer's events (*Where each form can be read*).
Readers following the log across the change rewind to the event it
names, or find they lost events (*Following a log*). The call record the
later writer writes holds its own exchanges (law 8); how a call's record
joins what each writer did, when the earlier writer also wrote a line
for it, is stage 4's. Who may claim a log (a lease), when a writer
counts as stopped, what the later writer does with calls that were
running (a tool that may have run), and how a log is ended by someone
other than a writer are stage 2's and stage 4's.

## The rules a store keeps

Anything that keeps logs for others to read keeps them by these rules.

**One step at a time, per log.** Each claim, and each append (a batch
whole), takes effect as one step, in one order with every other claim
and append of the same log: an append's checks and its write are one
step (a compare-and-set on the log's last position and its writer
number), and so are a claim's new number and its answer. A number given
by a claim is kept as surely as the events are (a store says how surely:
stage 2). A store that cannot do this (a folder on a shared disk without
locks, an object store without conditional writes) may keep logs only
where one process ever writes each one: it gives no claims, and cannot
be a journal that a later writer continues from.

**Claiming** a log for a later writer: refused `event-unknown` when the
store has no event of that log, `event-after-end` when the log is
finished; otherwise the store gives the next writer number and the
position of the last kept event (`{"writer": 2, "after": {"writer": 1,
"seq": 2}}`), as *Continuing a log* says.

**Appending** the events of one log, in order, checking each (a batch is
kept whole or not at all), in this order:

| appending an event | result |
|---|---|
| not an event of this format (it does not pass the schema), `tree` not the log it is appended to, `seq` not greater than `after`'s, or `after` naming a later writer than its own | refused `event-malformed` |
| in a log that has events, a `writer` other than the log's (the last number given; `1` before any claim) | refused `event-conflict`: an earlier writer, fenced (even when it sends again an event the store holds: a fenced writer never passes a barrier), or a number the store never gave |
| a `seq` already kept, and the same event (equal canonical JSON) | nothing changes (`duplicate`): a writer may send an event again after a failure |
| a `seq` already kept, and another event | refused `event-conflict`: two writers are writing one log |
| any event after the log's last (its outermost call's `done` or `failed`) | refused `event-after-end` |
| `after` not the position of the last kept event (`null` for a new log), and its `seq` greater than the last kept `seq` (`null`'s is `0`, and so is a new log's) | refused `event-gap`: events are missing before it |
| `after` not the position of the last kept event, otherwise | refused `event-conflict`: the log went on another way |
| a first event that is not the log's outermost call's `started` from writer `1` (`call` not the `tree`, another kind, another writer) | refused `event-start` |
| otherwise | kept |

**Reading** a log after an event named by its position gives the kept
events after it, in order; after `null`, all of them. A store that does
not have that event refuses `event-unknown` (*Resuming*). Reading is
never fenced: a fenced writer learns what was kept by reading.

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
`after` (a position), `at`, the `request` kind, `started`'s `root`, `program`,
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
