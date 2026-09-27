# Streaming (format 1)

A stream is **the same call, watched while it is made**. It asks the
model for the same thing, retries the same way, runs the same tools,
writes the same line to the call log, and ends with the same value (or
the same error) as calling the program. Streaming adds a view, never a
second behaviour.

This document is the contract for that view: the events a stream shows,
in what order, and their JSON form. The Python implementation is
`python/functai/streaming.py`; `schema/event.schema.json` checks an event's
JSON form.

## Words

- **Stream**: one call of a program (an AI function or a module), started
  now and watched. It shows the events of that call and of every call made
  inside it (a module's steps, a tool that calls an AI function, an
  escalation).
- **Request**: one request to a model. A call makes one, or several:
  retries, tool steps, escalation.
- **Answer**: the output that is the call's answer (`program.answer` in the
  call log). Other outputs (a `reasoning` field) are shown too, as their
  own fields.

## Events

Every event names the call it is about (`call`, the call log's id) and
its program (`function`, the program's name).

| kind | when | fields |
|---|---|---|
| `started` | a call begins | `parent` (the call it runs in, or null), `inputs` (JSON values) |
| `text` | a piece of an output's text is written | `field`, `answer` (true when the field is the answer), `text` |
| `thinking` | a piece of the model's own thinking that no output reads | `text` |
| `tool_call` | the model asked for a tool, and the request is complete | `id`, `name`, `input` |
| `tool_result` | the tool ran | `id`, `name`, `output` (text, as the model sees it) |
| `retry` | the model is asked again for this call's answer | `reason` (a sentence), `wait` (seconds before asking, or null) |
| `done` | the call ended with a value | `value` (JSON; a value with no JSON form is described as in the call log) |
| `failed` | the call ended with an error | `error` `{"type", "message", "code"?}` as in the call log |

Laws:

1. **Order.** A call's events come in the order they happened: `started`,
   then its `text`, `thinking`, `tool_call`, `tool_result` and `retry`
   events and the events of the calls inside it, then exactly one `done`
   or `failed`. A stream's last event is its outermost call's `done` or
   `failed`, except when the stream is closed first.
2. **Text is exact.** Within one request, the concatenation of a field's
   `text` pieces is that field's raw text in the reply (lmcc kernel §8: a
   piece is shown only once no later byte can change it). Pieces are never
   revised. What they mean as a typed value is known only at `done`.
3. **A retry voids the text before it.** After `retry`, the text of that
   call's fields shown since its previous `retry` (or `started`) no longer
   counts: the model is writing the answer again. A display that shows the
   answer so far resets it. A request after tool results also starts the
   fields afresh, without a `retry` (the `tool_result` events mark it).
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

## Closing

Closing a stream before its call ended cancels the call: the request in
progress is stopped at its next piece, no new request or call starts,
and the call ends with the error `Cancelled` (in the call log too).
Stopping a request does not promise the provider stops generating or
billing at once. A stream nobody closes runs to its end, like a call.

## The answer so far

Implementations offer the answer so far, read from its text: for a text
answer, the text; for a record or a list, the JSON read so far (complete
values, and the text of a string still being written; a number, `true`,
`false` or `null` only once complete); for any other answer, nothing
until `done`. It is provisional: `done` has the typed, checked value.

## In the call log

A streamed request's exchange has `"streamed": true` and `first_delta`,
the seconds from sending the request to its first piece of content (the
wait a person feels). A cancelled call's `error` is `{"type":
"Cancelled"}`.
