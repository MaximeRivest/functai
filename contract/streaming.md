# Streaming (format 2)

A stream is **the same call, watched while it is made**. It asks the
model for the same thing, retries the same way, runs the same tools,
writes the same line to the call log, and ends with the same value (or
the same error) as calling the program. Streaming adds a view, never a
second behaviour.

This document is the contract for that view: the events a stream shows,
in what order, their JSON form, and how a stream is kept while it is
written and read again later, by the same process or another one. The
Python implementation is `python/functai/streaming.py`;
`schema/event.schema.json` checks an event's JSON form; `cases/events/`
pins replay, the stored form and the rules a store keeps.

## Words

- **Stream**: one call of a program (an AI function or a module), watched
  from its start. It shows the events of that call and of every call made
  inside it (a module's steps, a tool that calls an AI function, an
  escalation). The call a stream watches is the stream's **watched call**;
  its id names the stream.
- **Request**: one request to a model. A call makes one, or several:
  retries, tool steps, escalation.
- **Answer**: the output that is the call's answer (`program.answer` in the
  call log). Other outputs (a `reasoning` field) are shown too, as their
  own fields.
- **Stored form**: an event as it is written anywhere outside the process
  that made it (a store, a file, another process), with the call's
  `log_content` applied (below). Inside the process, events are whole.

## Events

Every event says which stream it is in and where (`stream`, `seq`), when
it happened (`at`), the call it is about (`call`, the call log's id) and
that call's program (`function`, the program's name).

| key | meaning |
|---|---|
| `functai_event` | the format, `2`. Its presence says the object is an event of this format. |
| `kind` | what happened (the table below). |
| `stream` | the id of the stream's watched call. Every event of one stream has the same `stream`. |
| `seq` | the event's position in its stream: `1` for the first event, then each event one more than the one before, with no gaps. `(stream, seq)` names one event. |
| `at` | when it happened: RFC 3339 UTC with exactly six fraction digits, the call log's time format. Not decreasing within a stream. |
| `call` | the id of the call it is about (a UUIDv7, as in the call log). |
| `function` | that call's program's name. |

| kind | when | fields |
|---|---|---|
| `started` | a call begins | `parent` (the call it runs in, or null), `root` (the outermost call of its tree, as in the call log), `program` (the call log's `program` object), `inputs` (JSON values), `content` (`true`; `false` only in a stored form that does not keep every value) and `omitted` (see *The stored form*), `saw` (the call log's `saw`: [calls.md](calls.md)) |
| `text` | a piece of an output's text is written | `field`, `answer` (true when the field is the answer), `request`, `text` |
| `thinking` | a piece of the model's own thinking that no output reads | `request`, `text` |
| `tool_call` | the model asked for a tool, and the request is complete | `id`, `name`, `input` |
| `tool_result` | the tool ran | `id`, `name`, `output` (text, as the model sees it) |
| `retry` | the model is asked again for this call's answer | `reason` (a sentence), `wait` (seconds before asking, or null) |
| `done` | the call ended with a value | `value` (JSON; a value with no JSON form is described as in the call log) |
| `failed` | the call ended with an error | `error` `{"type", "message", "code"?}` as in the call log |

`request` is the number of the call's model request that wrote the piece:
`1` for its first request, and one more for each further request of the
same call (a retry, the request after tool results, an escalation to
another model answered in this call). A request that writes no text has
no piece, so the numbers a call's pieces carry may skip.

Laws:

1. **Order.** A call's events come in the order they happened: `started`,
   then its `text`, `thinking`, `tool_call`, `tool_result` and `retry`
   events and the events of the calls inside it, then exactly one `done`
   or `failed`. A stream's first event (`seq` 1) is its watched call's
   `started`, and its last is that call's `done` or `failed`; nothing
   follows it (a reader that closed the stream stops reading before it;
   a sink still receives it: *Closing*). `seq` is that order: when calls
   inside run at the same
   time, their events are interleaved in the order the stream received
   them, and `seq` numbers that one order.
2. **Text is exact.** Within one request, the concatenation of a field's
   `text` pieces is that field's raw text in the reply (lmcc kernel §8: a
   piece is shown only once no later byte can change it). Pieces are never
   revised. What they mean as a typed value is known only at `done`.
3. **A field's text is its latest request's.** The text of a call's field
   so far is the concatenation, in `seq` order, of that field's pieces
   from the call's latest request. A `retry` voids the text before it: at
   once, every field of the call is empty until the next request writes.
   A request after tool results starts the fields afresh too, without a
   `retry`: the call's `tool_result` empties its fields the same way.
   Every new request of a call follows one of the two, and its pieces
   carry the next `request` number, so a piece whose `request` is higher
   than any before it for its call also starts every field of the call
   afresh: a view that hides retries and tool results (stage 3's outside
   view) still starts afresh at the right place.
4. **Escalation is a retry.** When a first model is unsure and another
   answers (`escalate_to`), the call shows `retry`; when the other is an AI
   function, its call is a child of this one, and its answer is this
   call's answer.
5. **Tool calls are whole.** A `tool_call` is shown once the model has
   finished asking (its input complete), never piece by piece.
6. **Nothing is invented.** A reply that arrives whole (from the reply
   cache, or from a model that cannot stream) is shown as one `text` piece
   per field. A layout whose reader cannot read a reply in pieces (lmcc
   `plan.describe()["streaming"]["mode"] == "buffered"`, such as
   `adapter="json"`) shows its fields at the end of the request.
7. **One happening, one event per stream.** When a call inside a watched
   call is itself watched (a stream opened inside a streamed module), its
   events are in both streams, each time with that stream's `stream` and
   `seq`.

## Replaying a stream

Reading a stream's events in `seq` order and applying laws 1 to 4 gives,
after event *k*, exactly what a person watching live saw after event *k*:
which calls have started and ended, each field's text so far, which text
was voided. So a stream read back from where it was kept, from its start
or after any event, shows the same thing as the stream watched live.

Concretely, a reader keeps for each call: whether it ended (`done` or
`failed`), its latest request number (0 before any piece), and each
field's text so far. `started` adds the call with no fields; a `text` or
`thinking` piece whose `request` is higher than the call's latest sets
the latest to it and empties every field of the call; a `text` piece then
appends its text to its field (a stored piece's `size` adds to the
field's length instead); a
`retry` or a `tool_result` empties every field of the call; `done` and
`failed` mark the call ended. Thinking and tool calls are shown as they
come and are not voided. `cases/events/replay-*.json` pin this.

**Resuming.** A reader that has events up to `seq` N asks for the events
after N and gets those with `seq` greater than N, in order. A reader that
reads kept events and then follows live ones drops every event whose
`seq` it already has, and treats a gap (an event whose `seq` is more than
one past the last it has) as a loss: it reads the kept events again from
its last `seq`.

## Keeping a stream while it is written

A stream can be given to a **sink** as it happens: whatever keeps or
sends events (a store, a file, a socket to another process). The sink
receives each event's stored form, in `seq` order, when it happens, not
when the call ends, so a stream is kept while it is still being written.

- An implementation lets a program run with a sink whether or not the
  caller watches the stream (a setting, like `log_calls`): a call made
  with a sink is watched, by the sink. Its behaviour does not change
  (laws above; the call log is the same).
- A sink never gets in the way of the call, as the call log never does:
  if it fails, the implementation warns once and the call goes on.
- A sink may keep events in batches, and a transport may drop them;
  neither changes an event. What is kept of a stream is always a prefix
  of it: events `1` to `n`, none missing, none reordered.

**The rules a store keeps.** Anything that keeps streams for others to
read (stage 2's conversation stores are the first) appends events of one
stream in `seq` order, and:

| appending an event | result |
|---|---|
| the next `seq` of its stream (`1` for a new stream) | kept |
| a `seq` already kept, and the same event (equal canonical JSON) | nothing changes (`duplicate`): a writer may send an event again after a failure |
| a `seq` already kept, and another event | refused `event-conflict`: two writers are writing one stream |
| a `seq` beyond the next | refused `event-gap` |
| any event after the stream's last (its watched call's `done` or `failed`) | refused `event-after-end` |
| an event whose first event would not be its watched call's `started` (`seq` 1 of another kind, or `call` not the `stream`) | refused `event-start` |

A stream is **finished** when its watched call's `done` or `failed` is
kept. A kept stream whose writer stopped before that (the process ended)
is unfinished for ever; saying so is a store's job (stage 2), and nothing
here assumes a stream finishes. Closing a stream is its end too: the
watched call's `failed` with `Cancelled` is its last event, and a sink
receives it. `cases/events/store-*.json` pin these rules.

## The stored form

An event written outside its process follows its call's `log_content`, as
the call's line in the call log does ([calls.md](calls.md), *Content*):
what the log would not keep, a stored event does not keep either. When
the call's content is whole, the stored form is the event itself. When it
is not (`log_content` false, or some inputs or outputs kept as their size
only), the stored form keeps only the values of the fields that are kept,
and every event whose content was left out says so with `"content":
false`:

| kind | stored when the call's content is not whole |
|---|---|
| `started` | `content: false`; `inputs` holds only the inputs kept (absent when none is); `omitted` lists the fields kept as their size only, as the call record does (absent when no field is kept) |
| `text` | a field that is kept: as is. Any other: `size` (the piece's length in Unicode code points) instead of `text`, and `content: false` |
| `thinking` | `size` instead of `text`, and `content: false` |
| `tool_call` | no `input`; `content: false` |
| `tool_result` | no `output`, its `size` instead; `content: false` |
| `retry` | no `reason`; `content: false` |
| `done` | `value` when the answer is kept; else no `value`, and `content: false` |
| `failed` | no `error.message`; `content: false` |

So a watcher in another process sees the call's shape (who called what,
when, how long each field grew) without the values the log would not
keep. Replaying a stored stream follows the same laws, counting a field's
`size` instead of its text. Which values a *viewer* may see (the owner,
or a caller who sees only the boundary) is a view, chosen by the host
(stage 3), and is applied on top of the stored form; a view may skip
events, so its `seq` numbers may skip too.

## Closing

Closing a stream before its call ended cancels the call: the request in
progress is stopped at its next piece, no new request or call starts,
and the call ends with the error `Cancelled` (in the call log too).
Stopping a request does not promise the provider stops generating or
billing at once. A stream nobody closes runs to its end, like a call.

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

Format 2 (2026-09-28) added `functai_event`, `stream`, `seq`, `at`,
`request`, `started`'s `root`, `program`, `content`, `omitted` and
`saw`, and the stored form. Format 1 events (no `functai_event` key) were
only ever shown inside the process that made them, never kept; the
schema still accepts them so that a reader meeting one knows it. A reader
skips an event whose `functai_event` it does not know, and an event
`kind` it does not know: a later kind never changes what the kinds above
mean (a change that would is a new format).
