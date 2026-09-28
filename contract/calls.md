# The call log (format 2)

Every call of an AI function or a module can be written down as one line
of JSON in a folder. The folder is the whole interface: FunctAI in any
language writes it, and anything else (Chattering, a notebook, `jq`) reads
it. People's judgements of those calls (right, wrong, and what the right
answer was) are lines in the same folder, so the log is also a dataset.

This document is the contract. The Python implementation is
`python/functai/calllog.py`. Every implementation must pass
`cases/rated/`, `cases/content/` and `cases/saw/`, and write records that
`schema/` accepts. Records are format 2 since 2026-09-28; a reader reads
format 1 too (*Formats*).

## Words

- **Program**: an AI function (`kind: "ai"`) or a module (`kind: "module"`,
  plain code that calls AI functions).
- **Call**: one call of a program, from its inputs to its answer or
  error. A module's call has the calls it made as children.
- **Exchange**: one request sent to a model and its reply (or error). A
  call makes zero or more: retries, tool steps, escalation.
- **Rating**: one person's judgement of one call.
- **Version**: a fingerprint of everything that decides what a program
  sends besides its inputs. Two calls with the same version were made by
  the same program.
- **Interface**: a program's inputs and outputs, typed
  ([programs.md](programs.md)). Every program has one; a module declares
  it.
- **Fields** of a call: its interface's inputs and outputs, and for an AI
  function the outputs FunctAI adds to its signature (`reasoning` with
  `module: "cot"`; `calls`, the tool calls, with tools). The record's
  `inputs` and `outputs` are named by them. The `tools` input FunctAI
  adds to a signature is the function's state (its tools, part of its
  version), not a field: no record holds it.
- **Saw**: the earlier calls a call was given as context: shown to the
  model as earlier turns, or handed to a module's code as the conversation
  so far.

## The folder

```
<folder>/
  2026-09-26/
    lambda-48213-3fa9c1.jsonl      one file per writing process and day
    chattering-lambda-1204.jsonl   another writer (ratings from an app)
  2026-09-27/
    ...
```

- The day is the UTC date when the line was written.
- Each process writes its own file and only appends to it; it never
  shares a file with another process. Python's file name is
  `<host>-<pid>-<6 random hex>.jsonl`. Other writers choose any name that
  cannot collide with another writer's, ending in `.jsonl`.
- One line is one record: UTF-8 JSON, no line breaks inside, ended by
  `\n`. A reader skips a line that does not parse (a process may be in
  the middle of writing it) and a record whose format it does not know.
- New files are created readable by their owner only (`0600`, folders
  `0700`): the log holds what people typed.
- Nothing deletes lines. Keeping the log small is deleting whole day
  folders.

**Where.** Off unless asked for. The setting `log_calls` (in
`configure`, `@ai`, or `using`) is a folder, `True` (on, in the folder
the environment names, else the default folder), or `False` (never,
whatever the environment says). Unset, the environment variable
`FUNCTAI_LOG_CALLS` decides: empty, `0`, `false`, `no` or `off` is off;
`1`, `true`, `yes` or `on` is the default folder; anything else is a
folder. `log_content` says which values are written (see *Content*): set
to false, the log records sizes, times and tokens but no values or
messages, for programs that see passwords (a clipboard helper); set to
`{"transcript": false}`, everything but that input. For `log_calls`,
settings on a function beat `configure`, which beats the environment, as
for every setting; `log_content` only ever removes (*Content*). A saved
program keeps its `log_content` and never its folder or caller. The
default folder is `$XDG_DATA_HOME/functai/calls`
(`~/.local/share/functai/calls` on Linux,
`~/Library/Application Support/functai/calls` on macOS,
`%LOCALAPPDATA%\functai\calls` on Windows).

**Never in the way.** Writing happens when a call ends, in the caller's
thread, as one unbuffered append (`O_APPEND`; no fsync, so a power cut
can lose the last lines, and a process killed mid-line leaves a partial
last line in its own file, which readers skip). If the folder cannot be written, the implementation warns
once and the call returns as it would have. A record longer than 8 MiB
loses its exchanges' messages, then its values (`truncated: true`).

## A call record

```json
{"functai_call": 2,
 "id": "01926a8e-6c1a-7b3e-9d2f-0a8c5e4b1f21",
 "parent": null,
 "root": "01926a8e-6c1a-7b3e-9d2f-0a8c5e4b1f21",
 "program": {"name": "team", "kind": "ai", "module": "support",
             "version": "sha256:9c1e…", "signature": "sha256:41ab…",
             "interface": "sha256:41ab…", "answer": "result",
             "file": "/home/maxime/support.py", "line": 12},
 "started": "2026-09-26T23:12:03.123456Z",
 "seconds": 0.412,
 "content": true,
 "inputs": {"message": "I was charged twice for one order."},
 "outputs": {"result": "billing"},
 "sizes": {"inputs": {"message": 36}, "outputs": {"result": 9}},
 "error": null,
 "model": "gpt-4.1-mini",
 "usage": {"input_tokens": 212, "output_tokens": 6, "total_tokens": 218},
 "confidence": null,
 "exchanges": [{"model": "gpt-4.1-mini", "provider": "openai",
                "started": "2026-09-26T23:12:03.124001Z", "seconds": 0.405,
                "cached": false, "finish": "stop",
                "usage": {"input_tokens": 212, "output_tokens": 6, "total_tokens": 218},
                "request": {"…": "lm15 canonical request"}, "request_hash": "sha256:07d2…",
                "response": {"…": "lm15 canonical response"}}],
 "saw": [],
 "caller": {"kind": "notebook", "notebook": "/home/maxime/triage.md"},
 "process": {"host": "lambda", "pid": 48213, "user": "maxime",
             "language": "python", "runtime": "3.13.1", "functai": "1.1.0"}}
```

| field | meaning |
|---|---|
| `functai_call` | the format, `2`. Its presence says the line is a call. |
| `id` | a UUIDv7 (time-ordered, RFC 9562), made when the call starts. |
| `parent` | the call this one ran inside (a module, a tool, an escalation), or `null`. The parent's line comes *after* its children's: lines are written when calls end. A parent that was not logged (its program had `log_calls=False`) leaves a dangling id; read it as a root. |
| `root` | the outermost call of the tree (its own `id` when `parent` is null). |
| `program.name`, `.module` | what was called: the function's name and the code module it was defined in (`__main__` for a notebook or script; for a program loaded from a saved folder, the module it was saved from). |
| `program.kind` | `"ai"` or `"module"`. |
| `program.version` | see *Versions*. |
| `program.signature` | AI functions only: lmcc's signature fingerprint (kernel §3a: its fields' directions, names, purposes, shapes and type names, in order; not the prose) computed with every type name empty (`"type": ""`). The JSON shapes decide, not how one language spells a type (`str`, `string`, `character`), so the same function in two languages has one signature. It includes the fields FunctAI adds (`reasoning`, the tool fields). A module has none. |
| `program.interface` | every program: its interface's signature ([programs.md](programs.md)): lmcc's fingerprint of the interface's inputs and outputs only, each shape without its `default`. Calls with the same `program.interface` record the same data, even when the instruction or a default changed or reasoning was turned on; `rated` pools them. It does not say which calls a program accepts (compare interfaces for that). For an AI function with neither reasoning nor tools it equals `program.signature`. |
| `program.answer` | the name of the output that is the answer: the last output of the program's interface (`result` unless the program names it). A rating of right or wrong is about this output. |
| `program.saved` | present when the program was loaded from a saved folder: `sha256:` of that folder's `functai.json`. |
| `program.file`, `.line` | where the code is, when known. |
| `started`, `seconds` | UTC start and wall-clock duration. Every time in the log is RFC 3339 UTC with exactly six fraction digits (`2026-09-26T23:12:03.123456Z`), so times sort as text. |
| `content` | `true` when every value was recorded; `false` when some or all were not (the `log_content` setting: see *Content*). |
| `omitted` | present exactly when `content` is false: `{"inputs": [...], "outputs": [...]}`, the names of the fields whose values were not recorded, in the record's order (an output FunctAI added included). When it names every field, the record keeps no value. |
| `inputs` | each input the program received, named as its interface names it, as JSON (see *Values*); a module's too (never positional names such as `arg0`). An optional input left out takes its shape's `default`, and the record holds that value (an AI function's optional inputs always have one: a model is sent every input). A module's optional input left out with no default in its shape is absent: the program's own default applied, and asking again leaves it out again ([programs.md](programs.md)). When `content` is false: only the inputs recorded, and absent when there are none. |
| `outputs` | each output the model gave (the fields FunctAI added included), as JSON; `null` when the call failed before an answer. For a module, its outputs by name (`{"result": <what it returned>}` for a module with one output named `result`). When `content` is false: only the outputs recorded (absent when there are none, `null` when the call failed before an answer). |
| `returned` | an AI function whose code changed the answer before returning it (`return round(answer, 2)`): what it returned. Absent otherwise, and when the answer is not recorded. |
| `sizes` | always: for each input and output, the length in Unicode code points of its canonical JSON (see *Canonical JSON*). |
| `error` | `null`, or `{"type", "message", "code"?}`: the error the call's program ended with (its outcome). `type` is the exception's class name (`Refusal`, `LoginRequired`, `StepLimit`, `RateLimitError`, `InterfaceError`, `JournalError`, …), `code` lmcc's refusal code (or FunctAI's) when there is one. `message` is absent when `content` is false (it can quote the reply, or an input). A call cancelled by closing its stream has `type` `Cancelled`; a call stopped at a required journal's barrier, `JournalError` with code `journal-barrier` ([streaming.md](streaming.md)). A reply kept with no values (`on_unreadable="record"`) is `outputs: {}` with the refusal as `error`. |
| `model` | the model asked for the answer (the last exchange that got a reply), or `null` when no model was called. |
| `usage` | token counts summed over this call's own exchanges (integers only). A module's usage is in its children; sum a tree for its cost. |
| `confidence` | the probability the model gave its own answer (the lowest over the outputs it measured), or `null`. Only baked models and providers that return probabilities measure it. |
| `probabilities` | `{output: {answer: probability}}` when measured, for the outputs recorded. Absent when none is. |
| `escalated` | `true` when a first model was unsure and another answered; absent otherwise. |
| `exchanges` | each model request of this call, in order, including failed attempts (`error`, whose `message` follows the call's `error.message`) and replies from the cache (`cached: true`, `seconds: 0`). `model` (as asked), `provider`, `started`, `seconds`, `cached`, `finish`, `usage` are always there. Only when `content` is true: `request` and `response` (lm15's canonical JSON: a request holds every input, a response every output) and `request_hash`, lmcc's hash of the rendered request (kernel §3a, a model step's `request`), which shows a later reader rebuilt the same request (a hash of a short value can be guessed back, so it goes with the values). A request streamed by a watched call ([streaming.md](streaming.md)) has `streamed: true` and `first_delta`, the seconds until its first piece of content (null when none came). |
| `saw` | the calls this call was given as context, beyond its inputs and its program's state (see *Saw*). `[]` when none. Always there in format 2. A format-1 record has none: what it saw was not recorded, never "none". |
| `described` | present when some value the record holds had no JSON form and is written as a description (*Values*): `{"inputs": [...], "outputs": [...]}`, their names. Such a value is not data: it cannot be asked again or shown again. |
| `journal` | present when the call had a required journal that did not confirm its last event ([streaming.md](streaming.md), *Keeping a log while it is written*): `"refused"` (the journal did not keep it) or `"unknown"` (no answer: it may be kept; the journal's log settles it). The rest of the record is the call's outcome as it was: a call whose value was not confirmed kept is not a failed call. The caller got `JournalError` (code `journal-end`) holding that outcome. |
| `caller` | who called, as the environment and the program said (see *Caller*). `{}` when nothing did. |
| `process` | the writing process: `host`, `pid`, `user` (the operating system's), `language`, `runtime` (the language's version), `functai` (the library's version). |
| `truncated` | `true` when the record was cut to fit 8 MiB. |

**Values.** A value is written as the JSON its type describes (lmcc's
`to_json`): a dataclass or pydantic model is an object, an enum its
value, a tuple a list. A value with no JSON form is
`{"$type": "<type name>", "$repr": "<text, at most 2,000 characters>"}`,
and the record's `described` names it. How a value is written depends on
the value, not on its field: an opaque field ([programs.md](programs.md))
given `[1, 2]` records `[1, 2]`, and a field of any kind given a value with
no JSON form records a description.

## Content

`log_content` decides, for each field of a call, whether its value is
written or only its size. It is:

- `true`: no objection (the default);
- `false`: no value (sizes, times and tokens only);
- a map from field names to `true` or `false` (Python
  `log_content={"transcript": False}`, TypeScript `logContent: {
  transcript: false }`, R `log_content = list(transcript = FALSE)`, Julia
  `log_content = (transcript = false,)`). The key `"*"` stands for every
  field the map does not name: `{"*": false, "question": true}` writes
  nothing but the question.

**A value is written only when no layer drops it.** The layers are the
program's own setting (kept when it is saved), each enclosing block,
`configure`, and the environment (`FUNCTAI_LOG_CONTENT`: `0`, `false`,
`no` or `off`, in any case and with white space around, drops every
field; anything else, empty or unset, drops none). A layer drops a field
when it is `false`, or a map that says `false` for it, or a map whose
`"*"` is `false` and that does not name it. `true` never keeps what
another layer drops: a host's `false` wins over a program's own `true`,
and the environment's `0` wins over everything. So a host can hold every
program it runs to "never write the transcript", and a program can ask
for less, never for more.

**The fields FunctAI adds** to an AI function (`reasoning`, `calls`) can
quote any input, anticipate any output, and quote each other (a tool
call can repeat the reasoning). One is written only when no layer drops
it and no field of the call is dropped: dropping any field, one FunctAI
added included, drops them all. A map may name them (`{"reasoning":
false}`); a program's own map that names `tools` refuses, since `tools`
is not a field.

A map in a program's own settings that names a field the program does
not have refuses `log-content-field` when the program is defined (or
loaded), naming the field: a misspelt name would otherwise write the very
value it was meant to keep out. A map in a block or in `configure` applies
to every call inside it that has the field, and to no other; it cannot be
checked for spelling, so a host that must be sure lists what may be kept
(`"*": false`). A key that is neither a field name (an ASCII identifier)
nor `"*"` refuses `log-content-field` wherever it is set (other forms of
key are kept for later: kinds of data, not names).

**What the record keeps.** When every field is written, `content` is
true and the record is whole. Otherwise `content` is false, `omitted`
names the fields not written, and the record keeps only the values of
the fields written: `inputs` and `outputs` restricted to them,
`returned` only when the answer is written, `probabilities` only for
written outputs. It keeps no exchange `request`, `response` or
`request_hash`, and no error `message` (the call's, or any exchange's),
because each can hold any value of the call (a request holds every input;
a reply, a model's thinking or an error can quote them). `sizes` always
holds every field.

**What it does not do.** `log_content` is per field, not per value: a
value copied into a written field is written with it. An output quoting
the transcript is that output's value; a module that passes the
transcript to a helper whose input is called `log` has it written under
`log`, unless that helper's field is dropped too. `sizes` tells how long a
dropped value was (a password's length, for one).

**What it is about.** `log_content` decides what is kept for watching:
the call log, and the kept form of stream events ([streaming.md](streaming.md)).
It does not decide what a reader allowed to see a value is sent live
(that is a view), nor what a conversation keeps to go on (stage 2 gives
conversation stores their own setting, and must refuse, not forget, when
the two cannot both hold).

`cases/content/*.json` pin the layers and the record.

## Saw

`saw` lists, in the order they were given, the earlier calls whose
content this call was given as context, beyond its own inputs and its
program's state:

- for an AI function, the calls whose turns were placed in its requests
  as earlier turns (lmcc's turn slots): a stateful function's history, a
  conversation's earlier turns, a helper's own earlier calls;
- for a module, the calls its code was given as the conversation so far.

Worked examples are not in it (they are the program's state, part of its
version), nor are values passed as inputs (they are recorded as inputs),
nor the call's own children. It is one list for the whole call: every
request of the call was given the same earlier calls. A call whose
context changed between its requests (an agent that compacted its
history) writes its entries as of its first request, then, last,
`{"context": "changed"}`: an entry no reader knows today, so what it saw
is not known (`unknown-key`) rather than wrongly known. A later format of
entry will say how it changed.

Each entry is an object:

| key | meaning |
|---|---|
| `call` | the id of a call that was shown. |
| `steps` | `true` when the call was shown with its own steps (its model replies, tool calls and results), not only its inputs and outputs. Never the calls made inside its tools (lmcc never writes those): showing deeper calls would be another key. Absent: inputs and outputs only. |
| `without` | the names of that call's fields that were left out when it was shown (a bulky input, a photo after its first answer, a reasoning the next model is not shown). Absent: none. With `steps`, never the field that holds a step's tool calls (`calls`): its tool steps would answer no call. |
| `slot` | the lmcc turn slot the call was placed in (kernel §3a, *Slots*). Absent: `turns`, lmcc's default slot. |

or, as the **first** entry only, `{"saw_of": <id>}`: everything the call
with that id saw, entry for entry, in its order. A conversation's turn
that sees every earlier turn writes `[{"saw_of": <previous turn>},
{"call": <previous turn>}]`, so its record does not grow with the
conversation (lmcc F7: copying what each turn saw grows with the square
of its length). A writer uses `saw_of` only when the entries it stands
for are exactly those this call was given.

A later writer may add keys to an entry, or entries of another kind (a
summary of old turns, a turn merged from another program): the schema
accepts any object. A reader that does not know a key, or an entry with
no `call`, cannot know what was shown (`unknown-key`).

**The turn an entry stands for.** Showing the call again is placing, in
the entry's slot, the lmcc turn (kernel §3a) the call's record stands
for, with every field in `without` taken out of its inputs, its outputs
and each of its model steps' outputs.

- The turn's `signature` is the record's `program.signature`; its
  `inputs` are the record's `inputs`; its `outputs` the record's
  `outputs`: what the model gave (the fields FunctAI added included), not
  what the code returned (`returned`).
- Without `steps` the turn has no steps (its inputs, then its outputs).
- With `steps`, its model and tool steps come too; a model step whose
  outputs held a field taken out loses its recorded message, so it is
  written from its values (the recorded message would show the field).
  An entry whose `without` names the field that holds a step's tool calls
  is refused `turn-invalid` (lmcc's word): the turn would be invalid.
- How a record's exchanges become those steps is **not fixed yet**, and
  no reader may guess it: which exchanges become model steps (failed
  attempts, a reply that could not be read and the retry after it,
  escalation to another model, replies from the cache); how each step's
  outputs are read from its reply (with the plan of the call's version);
  how tool steps are read from the next request; which lmcc `replay`
  mode writes them. Stage 5 (`rated` with earlier turns) fixes it, and a
  replay that rebuilds a request checks it against the exchange's
  `request_hash`. Until then a reader shows calls without steps only.

`cases/saw/12`, `13` and `16` pin the turn, starting from the turn a
record stands for.

**Reading it.** The calls a call saw are its entries with `saw_of`
replaced, recursively, by the entries of the call it names. They are not
known, and a reader that needs them must say so rather than guess, when:
the call's own record has no `saw` (`not-recorded`); a `saw_of` names a
call whose record the reader does not have, or that has no `saw`
(`missing-call`); an entry is one the reader does not know (`unknown-key`);
or following `saw_of` comes back to a call already followed (`saw-cycle`).
`saw` holds ids only, so it is kept when `content` is false.

**Knowing is not replaying.** Knowing which calls a call saw is not
enough to show them again. That also needs, for each entry in order:

1. an entry a call can have been shown (else `turn-invalid`);
2. the call's record (else `missing-call`, naming it);
3. the values it was shown with, as data: the record not truncated;
   every input and output not in `without` recorded, and none of them
   written as a description (`described`); with `steps`, a whole record
   (`content` true) whose every exchange keeps its `request_hash` and,
   when a reply came (a `finish`), its `response` (else `not-kept`,
   naming it).

This says the log **keeps** what showing again needs; it does not show
anything (with `steps`, stage 5's). A reader that replays says which
entry fails, and never replays in part.

`cases/saw/*.json` pin these rules.

## Caller

A JSON object saying who called. The environment variable
`FUNCTAI_CALLER` holds one (set by whatever starts the process: rat for
a notebook kernel, Chattering for an agent or a shortcut), and the
setting `caller` (a dict, usually in `with functai.configure(caller=...)`)
adds keys to it for a block of code. Conventional keys:

| key | value |
|---|---|
| `kind` | `notebook`, `script`, `agent`, `conversation`, `shortcut`, `schedule`, `api`, `test` |
| `user` | the person, when the process knows it (a shared machine's account is not a person) |
| `notebook`, `cell` | a notebook's path, the cell that ran |
| `conversation` | a Chattering conversation key |
| `evaluation` | set by `evaluate`: the evaluation's `run` id. These calls answer known questions; they are not use. |
| `optimization` | set by `.opt`: an id per optimization. Not use either. |

## A rating record

```json
{"functai_rating": 1,
 "id": "01926a90-01b2-7c44-8a10-5d1e2f3a4b5c",
 "call": "01926a8e-6c1a-7b3e-9d2f-0a8c5e4b1f21",
 "at": "2026-09-26T23:15:40.002000Z",
 "by": "maxime",
 "verdict": "wrong",
 "answer": "shipping",
 "reasons": ["wrong category"],
 "note": "A parcel that never came is shipping, even if they want money back.",
 "origin": "review"}
```

| field | meaning |
|---|---|
| `functai_rating` | the format, `1`. |
| `id` | a UUIDv7. |
| `call` | the call judged. |
| `at` | when (the log's time format). |
| `by` | who: a person, not an account (Python's default is the caller's `user`, else the operating system's). One person's later rating replaces their earlier one for that call. |
| `verdict` | `"right"`: the answer is correct for this input (not "nice": correct). `"wrong"`: it is not. `null`: this person withdraws their rating. |
| `answer` | with `"wrong"`: what the answer should have been, as JSON in the answer's type. Optional: someone may know an answer is wrong without knowing the right one. |
| `outputs` | optional: right values for other named outputs, `{name: value}`. |
| `reasons`, `note` | optional: short tags and a sentence. |
| `origin` | how the judgement was made: `"review"` (someone judged it; the default) or `"edit"` (someone changed the output while using it: renaming a generated title is a correction of the title program). Edits are data too, but noisier: people change things for other reasons. |
| `sample` | optional: the id of a random draw of calls this rating belongs to. Ratings people chose to make are not a fair sample of the calls; only a random draw measures how often the program is right. |

## Rows with known answers (`rated`)

Ratings become data that `evaluate` and `.opt` read: one row per call,
with the inputs under their names and the right answer under the output's
name. Every implementation computes the same rows:

The reader is given the program's `name`, and when it knows them its
`module`, its current `signature` and `interface` (the values its calls'
`program.signature` and `program.interface` would have now), and a person
`by`. It reads records of the formats it knows (1 and 2), and skips the
others: a record of another format makes no row and is not counted.

1. Take the calls of the program: `program.name` equal, and
   `program.module` equal when a module is given.
2. For each call, each person's latest rating counts (latest `at`; equal
   times: the larger `id`; lines may be in any order). A `null` verdict
   removes that person's rating. With `by`, only that person's ratings
   are read. A call with no counting rating is not rated: no row, not
   counted.
3. A rated call is left out, and counted under the first of these that
   applies. A signature or an interface is given, and the call matches
   none of them (`other_signature`: its inputs or outputs have changed
   since): a call matches the given `interface` when its
   `program.interface` equals it, or, with no `program.interface`
   (format 1), when its `program.signature` does (the two are equal for
   an AI function with neither reasoning nor tools); it matches the given
   `signature` when its `program.signature` equals it. Not all its inputs
   were recorded as data (`no_content`: `content` false, with no
   `omitted` or with `omitted.inputs` not empty; or `described.inputs`
   not empty). Or no counting rating gives a value (`no_answer`).
4. What a rating gives: `"right"` gives the call's own
   `outputs[program.answer]` (nothing when the call has no answer: it
   failed; nothing when the answer was not recorded, or was written as a
   description); `"wrong"` gives its `answer` under `program.answer` and
   its `outputs`, and nothing when it has neither: it says what the
   answer is not, not what it is.
5. One row per remaining call: the inputs; then the values of the latest
   rating that gives any (the answer under `program.answer`, then the
   other outputs in the rating's order); then `call`, `version` (the
   call's), `rating` (that rating's verdict), `rated_by`, `origin`
   (`"review"` when absent), `sample` (`null` when absent) and
   `disputed`. Only outputs a rating gives are keys. Data keeps its
   names: when an input or output already has one of these names, the
   added key gets underscores in front until it is free (`_version`).
6. `disputed` is true when the counting ratings do not all have the same
   verdict, or two of the ratings that give values give different ones
   (compared as canonical JSON).
7. Rows are in the order of the calls' `started`, then `id`.

`cases/rated/*.json` pin these rules.

## Versions

A version is `"sha256:"` and the hex SHA-256 of the canonical JSON of a
small document. It needs no model and no network: a program knows its
version before its first call (Python: `fn.version`).

**An AI function:** `{"request": R}` when the model writes the whole
body, `{"code": C, "request": R}` when code of the function's own runs
beside the model. The same AI function written in two languages (the same
instruction, fields, layout, worked examples and tools) therefore has one
version, and its ratings pool.

- `R` is what the function sends for a fixed sample input: the request
  it renders (instruction, layout, worked examples, tools; no conversation
  memory) under fixed model capabilities, as `"sha256:"` + SHA-256 of its
  canonical JSON. It is the first entry of `fingerprints.requests` in a
  saved program's `functai.json`, so a saved folder names the version it
  holds. The sample input has, for each input field, by its JSON Schema:
  the first `enum` value; the first non-null option of an `anyOf`;
  `"example text"` for a string, `3` for an integer, `2.5` for a number,
  `true` for a boolean, `[]` for an array, `{}` for an object. The fixed
  capabilities are `instruct`, `native_structured_output`,
  `native_function_calling` and `stop_sequences` true;
  `native_reasoning` and `assistant_prefill` false; provider `"probe"`.
  A function on a baked model renders with the layout and facts the
  model was trained with. When the sample cannot be rendered, `R` is
  `"refused:<lmcc refusal code>"`.
- The model writes the whole body when the body holds nothing but a
  description, output declarations and "the answer is the value". Python:
  after the docstring, only `x = _ai`, `x: T = _ai`, `x: T = _ai["…"]`,
  `...`, `pass`, and a last `return`, `return _ai` or `return ...`. A
  language whose AI functions have no body (a TypeScript `ai(...)` given
  no function) always has `{"request": R}`. Default values of inputs are
  not code: the values a call used are its logged inputs.
- `C` is present only when the function runs code of its own, such as
  `return round(_ai, 2)`. It is `"sha256:"` + SHA-256 of the function's source text (UTF-8),
  dedented, without its decorators (the text `functai.save` writes, so a
  loaded program has the version of the one that was saved), read once
  per function. A change in the code around the prompt
  (`return round(answer, 2)`) is a new version too. Where the source
  cannot be read, an implementation hashes what it has (Python: the
  compiled code) and the version is stable only within that runtime.

The model and its sampling settings are not part of a version: they are
where a version runs, and the record says which (`model`, exchanges).

**A module:** `{"code": {key: C}, "ai": {key: version}, "interface": I}`
over every piece of code the module reaches (the same graph
`functai.check` and `functai.save` follow): each plain function's,
class's and the module's own source hash under its `module:name` key,
each AI function's version under its key, and the module's interface `I`
as data ([programs.md](programs.md)). Optimizing an AI function inside a
module is a new version of the module; so is changing what it declares it
takes or gives, even where the declaration is not in the code the version
hashes (a TypeScript module's `input` and `output`). Versions computed
before 2026-09-28 have no `"interface"`: every module's version changed
once then.

## Canonical JSON

lmcc kernel §3a: object keys sorted by code point, `,` and `:` with no
whitespace, non-ASCII written as UTF-8, no NaN or infinities; strings
escape `"`, `\` and U+0000 to U+001F (`\b` `\f` `\n` `\r` `\t`, the
others `\u00xx` in lowercase hex); numbers by lmcc §7a: integers in
decimal, every other number as ECMAScript's `Number::toString` writes it
(`1.0` is `1`, `1e-07` is `1e-7`, `0.00001` is `0.00001`). These are the
bytes every implementation hashes and measures (`sizes`), whatever its
host's JSON writer spells.

## Formats

Format 2 (2026-09-28) changed what `content: false` can mean, so it is a
new format, not an edit of format 1:

- `content: false` with `omitted` keeps the values of the fields
  `omitted` does not name. In format 1, `content: false` means no value
  at all, and a format-1 reader or schema that meets a record of some
  values must not take it for one.
- `program.interface` on every record; `program.signature` for AI
  functions only.
- `saw` on every record.
- `request_hash` on exchanges; no error message in any exchange when
  `content` is false.
- `described`, naming values written as descriptions; `journal`, when a
  required journal did not confirm the call's end.

A reader reads both formats. A format-1 record is read as it always was:
`content: false` means no value, no `saw` means not recorded, and it has
no `program.interface`. A writer writes format 2. A reader that knows
only format 1 skips format-2 records, as it skips every format it does
not know: its sums over a tree then miss those calls until it learns
format 2.

## For implementers

- Write the record when the call ends, in one `write` of the whole line.
  Never let logging raise into the call.
- Make the id when the call starts, and make it available on what the
  call returns (Python: `prediction.call_id`) so a caller can rate it.
- Know what a call saw before its first request, and write it (`saw`,
  and the `started` event's) even when it is `[]`.
- Write format 2; read formats 1 and 2.
- Propagate the current call to anything that runs inside it
  (Python: a `ContextVar`, which `evaluate`'s threads copy). A call made
  from a thread that did not copy the context has no parent.
- A reader is lenient: unknown fields are ignored, unknown formats are
  skipped, a partial last line is skipped.
