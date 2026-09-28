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
pins replay, following, the kept form and the rules a store keeps.

## Words

- **Call tree**: a call and every call made inside it (a module's steps,
  a tool that calls an AI function, an escalation), as one process runs
  it. Its **outermost call** is the one the process started it with.
- **Log**: the events of one call tree, numbered in the order they
  happen. Its id is its outermost call's id.
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

Every event says which log it is in and where (`tree`, `seq`, `after`),
when it was numbered (`at`), the call it is about (`call`, the call log's
id) and that call's program (`function`, the program's name).

| key | meaning |
|---|---|
| `functai_event` | the format, `2`. Its presence says the object is an event of this format. |
| `kind` | what happened (the table below). |
| `tree` | the log's id: the id of the call tree's outermost call. |
| `seq` | the event's place in its log: `1` for the first event, then one more for each event, with no gaps in the whole log. `(tree, seq)` names one event, in every form. |
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

Laws:

1. **Order.** A call's events come in the order they happened: `started`,
   then its `request`, `text`, `thinking`, `tool_call`, `tool_result` and
   `retry` events and the events of the calls inside it, then exactly one
   `done` or `failed`. A log's first event is its outermost call's
   `started`; its last is that call's `done` or `failed`, and nothing
   follows it. When calls inside run at the same time, their events are
   interleaved in the order the writer received them, and `seq` numbers
   that one order.
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
   same events, with the same `tree` and `seq`: it starts at that call's
   `started`, and its `after` skips the events of calls outside it.

## Replaying

Reading a form's events in order, from its first, and applying laws 1 to
4 gives, after each event, what a person watching that form live saw
then: which calls have started and ended, each field's text so far. So a
log read back from where it was kept shows what was seen live.

A reader keeps, for each call it has seen start: whether it ended, and
each field's text so far. `started` adds the call with no fields; `request`
and `retry` empty every field of their call; `text` appends to its field;
`done` and `failed` mark the call ended. `thinking`, `tool_call` and
`tool_result` are shown as they come and change no field. The log is
**finished** once its outermost call has ended. `cases/events/replay-*.json`
pin this.

**Resuming.** A reader that has events up to `seq` N asks for the events
after N, in the same form, and gets those with `seq` greater than N, in
order. It goes on from the state it had at N: the events after N alone do
not give the text so far.

## Following a log

A reader that follows a form live (a page, another process) keeps the
`seq` of the last event it has, starting from `0`, and for each event it
receives:

- `seq` not greater than its last: a **duplicate**, dropped (a writer or a
  transport may send an event again);
- `after` equal to its last: the next event; it takes it;
- anything else: events were **lost**. It reads the same form again after
  its last `seq`.

`after` makes this work in every form, though only the whole log has no
gaps in `seq`. `cases/events/follow-*.json` pin it.

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

The kept form keeps each event's `seq`, and sets `after` to the `seq` of
the kept event before it. So a watcher in another process sees the tree's
shape (who called what, when, how many requests) and the values the log
keeps, and nothing of the rest: neither its text nor the length of each
piece (a sequence of piece lengths is enough to guess much of a reply's
text). The call record's `sizes` gives each field's total.
`cases/events/kept-*.json` pin this.

## Views

A view is what one kind of reader may see of a log: the owner sees the
whole log; a caller who sees only a program's boundary sees less (stage
3 names the views and who chooses them). A view:

- keeps each event's `seq`, and sets `after` to the event before it in
  the view;
- shows a call's `started` before any other event of that call it shows,
  and its `done` or `failed` when it shows the call;
- shows every `request` and `retry` of a call whose text it shows (it may
  leave out the retry's `reason`, with `content: false`): they hold no
  value, and without them its reader would keep text the call voided;
- may leave out events and values; it never changes one.

A view is made from the whole log in the process that runs the tree, or
from a kept log. What a view can show of a module's own answer while it
is written is not settled here: a module's code writes no pieces, so a
view that shows only the module shows its `started`, then its `done`,
until stage 3 lets a module say that a child's field is its output.

**Retention is not disclosure.** The kept form is about what may be
*kept*. Sending events live to a reader the host allows (a page, a
worker's parent process) keeps nothing, and follows the view the host
chooses for that reader, not `log_content`. The whole form leaves the
process only that way: through the stream its caller watches, which
decides where it goes.

## Keeping a log while it is written

FunctAI gives a log's events to two kinds of receiver, each set like
`log_calls` (a program's settings, a block, `configure`), whether or not
anyone watches the call. A call made with either is watched, by it; its
behaviour does not change (laws above; the call log is the same).

**Observers** receive the kept form of every event of every call in their
scope, in order, as it happens (their `after` skips what is outside their
scope): a log for watching (a socket, a telemetry exporter, a page). An
observer never gets in the way: if it fails, the implementation warns
once and stops giving it events. A slow observer does not slow the call:
an implementation may hand events to it from another thread, in order,
and may drop them (the observer then sees a loss: *Following a log*).

**Journals** keep logs: a store, a folder (stage 2's stores are
journals). A journal keeps the kept form of the whole log of every tree
whose outermost call starts in its scope; the journal a tree has is
decided when its outermost call starts. A call inside a tree does not
start another log: a journal set only around calls inside a tree (a block
in a module's body) keeps nothing of it, and the implementation warns
once (a required journal refuses `journal-scope` when that call starts,
before it runs).

- **Appending.** The writer (the process running the tree) appends each
  event in order, alone or in batches, with the `after` the journal
  checks (*The rules a store keeps*): an append says where it goes. A
  journal **acknowledges** an append once it has kept it as surely as it
  says it keeps things (a store declares that: stage 2). The writer keeps
  every event not yet acknowledged, and sends it again after a failure
  (a resend of a kept event is a `duplicate`).
- **Best effort** (the default): a journal that fails does not stop the
  call. The implementation warns once, and keeps sending; if it gives up,
  it stops appending to that log, which stays kept up to its last
  acknowledged event, unfinished.
- **Required** (the host asks for it): the call waits for the journal's
  acknowledgement before its first request or its code runs (its
  `started` is kept), before each tool runs (what the model asked for is
  kept before anything acts on it: stage 4 says which tools need this),
  and before the call returns or raises (its last event is kept). A
  journal that cannot acknowledge (after the writer's own retries) fails
  the call with `JournalError`; the caller gets that error, and the call
  log records it.
- A writer whose append is refused `event-conflict` has lost the log to
  another writer: it stops appending to it, and never appends to it again.

A log whose outermost call's `done` or `failed` is kept is **finished**.
Until then it is **unfinished**: its writer may still be running, or may
have stopped. A later writer may continue it (a process that resumes a
waiting turn after a restart: stage 4): it reads the kept log, and
appends after the log's last kept event, numbering on from there; its
first append is refused if the log went on meanwhile. Who may continue a log,
how a stopped writer is kept out (a lease, a token), and how a log is
ended by someone other than its writer are stage 2's and stage 4's.

**The rules a store keeps.** Anything that keeps logs for others to read
appends the events of one log in order, checking each (a batch is kept
whole or not at all), in this order:

| appending an event | result |
|---|---|
| `seq` not greater than `after` | refused `event-malformed` |
| a `seq` already kept, and the same event (equal canonical JSON) | nothing changes (`duplicate`): a writer may send an event again after a failure |
| a `seq` already kept, and another event | refused `event-conflict`: two writers are writing one log |
| any event after the log's last (its outermost call's `done` or `failed`) | refused `event-after-end` |
| `after` greater than the last kept `seq` (`0` for a new log) | refused `event-gap`: events are missing before it |
| `after` less than the last kept `seq`, or `seq` less than it | refused `event-conflict`: the log went on another way |
| a first event that is not the log's outermost call's `started` (`call` not the `tree`, or another kind) | refused `event-start` |
| otherwise | kept |

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

Format 2 (2026-09-28) added `functai_event`, `tree`, `seq`, `after`, `at`,
the `request` kind, `started`'s `root`, `program`, `content`, `omitted`
and `saw`, the kept form, views, and the rules a store keeps; a
`tool_call`'s `id` is never null. Format 1 events (no `functai_event` key)
were only ever shown inside the process that made them, never kept; the
schema still accepts them, so that a reader meeting one knows it. A reader
skips an event whose `functai_event` it does not know, and an event `kind`
it does not know: a later kind never changes what the kinds above mean (a
change that would is a new format).
