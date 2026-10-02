# Reading the cases

Each case is one JSON file, `NN-what-it-shows.json` (in `events/`,
`kind-NN-…`). It holds a `description` (the rule it pins, in words) and
the data below. `make.py` writes every one from the rules (the script
named beside each folder); never edit one by hand. A harness passes a
case when doing what the case says gives exactly its `expect`: compared
as JSON (canonical: keys in any order, `1` and `1.0` equal), lists in
order, unless a folder says otherwise.

What every folder shares:

- **Ids** are UUIDv7-shaped strings (`01926c00-0001-7000-8000-000000000000`);
  only equality matters.
- **A value with no JSON form** is written `{"$type", "$repr"}` (exactly
  those two keys). The harness gives the program a native value of its
  own for it (a data frame, an object), and reads such a value back as
  that description (calls.md, *Values*).
- **A refusal** is `{"refuses": code, …}` with the code from
  `../README.md`, *Refusal codes*, and, where a case names one, the
  `field` or `event` at fault.

## `functions/` (functions.py)

`{"definition": {name, description, inputs: [{name, shape, desc?,
optional?}], outputs: [{name, shape, desc?}], settings, state, tools?},
"expect": {"signature", "sample", "request", "request_hash", "version",
"signature_id"}}`. The harness defines the AI function from the
definition, then compares: `signature` (lmcc's plain-data form, compared
with `type` left out), `sample` (the sample input), `request` (the
request for the sample under the probe facts, `../models.json`),
`request_hash`, `version`, `signature_id` (the call log's
`program.signature`).

## `rated/` (rated.py)

`{"records": [call and rating records], "rated": {name, module,
signature, by, interface?}, "expect": {"rows": [...], "left_out":
{reason: count}}}`: `rated(records, **rated)` gives exactly `expect`.
Records are of formats 1 and 2 (and, in one case, of a format no reader
knows, which is skipped).

## `scores/` (scores.py)

- `"kind": "interval"`: `{"values", "expect": {"mean", "low", "high"}}`,
  null where there is none; compared within 1e-12.
- `"kind": "match"`: `{"answers", "prediction", "expect": {metric: 0 or
  1}}`: `exact_match`, and `<output>_match` per output when there are
  several.

## `saved/` (saved.py)

`{"manifest", "node": key or null, "expect": {"refuses": code, …} or
{"loads": {"name", "module", "version", "signature_id", "requests"},
"sends"?: [{"inputs", "request_hash"}], …}}`, where `expect` also holds
`"describe": {"interface"} or {"refuses": code}`. `loads`: loading the node (or the manifest's entry
when `node` is null) in the harness's language. `sends`: the loaded
function called with `inputs` (an optional one left out) sends the
request whose hash is `request_hash`, under the probe facts. `describe`:
describing the node without loading it.

## `programs/` (programs.py)

Told apart by `"program"`:

- `"ai"`: `{"definition" (as in functions/), "expect": {"interface",
  "signature", "signature_id"} or {"refuses": "interface-malformed",
  "field"}, "binds": [{"inputs", "expect": {"inputs"}}]}`. Defining the
  AI function gives that interface and signature (the call log's
  `program.interface`), or is refused. Each bind calls it with `inputs`
  and expects the values it is called with, bound (programs.md,
  *Binding a call's inputs*: an optional input left out, or given null
  that null does not fit, takes its default), or `{"refuses":
  "interface-input", "field"}` with no request sent.
- `"module"`: `{"interface", "expect": {"signature"}, "checks": [...]}`.
  The harness defines a module with the interface. Each check gives
  `inputs` (JSON) and expects `{"inputs": what its code gets}` (bound) or
  `{"refuses": "interface-input", "field"}`; or gives what the code
  `returned` and expects `{"outputs"}` or `{"refuses":
  "interface-output", "field"}`. An object that is exactly `{"$type",
  "$repr"}`, or `{"$type", "$repr", "$text"}`, stands for a value with no
  JSON form: the harness gives the program a native value of its own for
  it, whose type has no text of its own, or whose own text is `$text` (a
  data frame, a date).
- `"message"`: `{"interface", "checks": [{"inputs", "log_content"?,
  "expect": {"refuses", "field", "quotes"}}]}`. The harness defines a
  module with the interface (and `log_content` as its own setting) and
  calls it with `inputs`: it is refused, and the error's message contains
  `quotes`, or, when `quotes` is null, holds no part of the value at
  fault (the harness looks for the value's canonical JSON, and for its
  `$repr`). For a stand-in, `quotes` is its description's `$repr`: a
  harness whose native value the language describes otherwise looks for
  that description instead.
- `"definitions"`: `{"interfaces": [{"interface", "ai"?: true, "expect":
  {"signature"} or {"refuses": "interface-malformed", "field": name or
  null}}]}`. Defining a module with each interface, or reading it from a
  saved folder; with `ai`, an AI function's interface (when the function
  is defined or its node read), whose shapes may carry keywords the
  vocabulary does not list.
- `"same-data"`: `{"interfaces", "expect": {"signatures"}, "checks":
  [{"inputs", "expect": [one result per interface, as a module check's
  inputs]}]}`.

## `content/` (content.py)

`{"fields": {"inputs", "outputs", "added"}, "layers": [{"where": "own" |
"block" | "configure", "log_content"}] (closest first), "environment":
FUNCTAI_LOG_CONTENT or null, "record": the call's whole record,
"expect": {"record": as written} or {"refuses": "log-content-field",
"field"}}`. `added` are the outputs FunctAI adds (`reasoning`, `calls`),
also listed among `outputs`. A program's own refusal comes when it is
defined; a block's or `configure`'s when it is set.

## `saw/` (saw.py)

- `"kind": "read"`: `{"records", "queries": [{"call", "expect": {"saw":
  [entries, saw_of expanded]} or {"unknown": code, "call"}, "keeps":
  {"ok": true} or {"refuses": code, "call"}}]}`.
- `"kind": "shown"`: `{"turn": an lmcc turn, "entry", "expect": {"slot",
  "turn"} or {"refuses": "turn-invalid"}}`.

## `events/` (events.py)

Events are streaming.md's, format 2. A **position** is `{"writer",
"seq"}`, or null for "before the first event"; positions are compared as
a whole, both numbers. The file name's first word is the `"kind"`:

- `replay`: `{"events", "expect": {"views": [state after each event],
  "finished"}, "resume": [{"after": position, "expect": {"events":
  [positions]} or {"refuses": "event-unknown"}}]}`. A state is, for
  each call started so far, `{"ended": null | "done" | "failed",
  "fields": {name: text so far}}`. `resume`: what a source holding these
  events gives a reader that has events up to `after`.
- `follow`: `{"received": [events, in the order the reader got them],
  "recover"?: {...}, "expect": {"results": [one per event received, up
  to an unknown format], "state": {tree: {"calls", "finished"}},
  "recover"?: {"reads", "state"}}}`. The reader follows one form,
  keeping a last event per tree; each result is `"kept"` (the next
  event), `"duplicate"`, `"stale"`, `"rewind"`, `"loss"` or
  `"unknown-format"` (it stops there: nothing after it is read). `state`
  is the replay of what it holds at the end. After a rewind it holds
  what it had up to the event named, values the kept form lacks
  included, then the rest. `recover`, when present:
  - `"reader"`: the form the reader follows: `"kept"` (the kept form, or
    a view made from it) or `"live"` (the whole log, or a view made from
    it: it may show values the kept form lacks). A live reader got each
    event it received from the process of that event's writer.
  - `"from"`: where it reads again: `"store"`, or a writer number (the
    process of that writer).
  - `"source"`: the events that source gives, in the reader's form.

  The reader resumes in place when the source can give every event it
  holds as it holds it: always for `"kept"`, and for `"live"` only when
  every event it holds is `from`'s writer's. Then it reads after its
  last event, and from the beginning if the source answers
  `event-unknown`. Otherwise it reads from the beginning at once.
  `expect.recover.reads` are its reads in order, `[{"after": position or
  null, "expect": {"events": [positions]} or {"refuses":
  "event-unknown"}}]`, and `state` the replay of what it holds after
  them.
- `kept`: `{"events": [a whole log], "kept": {call id: {"inputs": {name:
  bool}, "outputs": {name: bool}}}, "expect": {"events": [the kept
  log]}}`. `kept` says which fields each call's `log_content` keeps (the
  job of content/).
- `store`: `{"steps", "reads", "expect": {"logs": {tree: {"events":
  [positions], "writer": the last number given, "finished"}}}}`, the
  steps run in order against one empty store, then the reads. A step is
  one of:
  - `{"append": event, "expect": "kept" | "duplicate" | code}`;
  - `{"batch": [events], "expect": "kept" | "duplicate" | {"refuses":
    code, "event": position}}`: one step, kept whole or not at all;
  - `{"claim": tree, "expect": {"writer", "after": position} or
    {"refuses": code}}`: a later writer claiming the log.

  A read is `{"tree", "after": position or null, "expect": {"events":
  [positions]} or {"refuses": "event-unknown"}}`.
- `journal`: a writer keeping one AI function's log (the tree's only
  call) in a journal that fails. `{"mode": "required" | "best-effort",
  "retries": n, "events": [the call's whole log when nothing fails],
  "script": [...], "expect": {...}}`. The writer sends each event, in
  order, at most `1 + retries` times in a row before giving up on it for
  now. Each send meets the script's next word (`"ok"` once the script
  ends):
  - `"ok"`: the append reaches the journal and its answer comes back;
  - `"lost"`: it reaches the journal, which applies it, and the answer
    is lost;
  - `"down"`: it does not reach the journal;
  - `"conflict"`: the journal answers `event-conflict` and keeps
    nothing.

  Between sends, the script may say what another writer does, which
  takes no send:
  - `"claimed"`: another writer claims the log (the writer is fenced);
  - `"ended"`: another writer claims the log and ends it, appending its
    outermost call's `failed` with `{"type": "Cancelled"}` after the last
    kept event.

  `expect`:
  - `"log"`: the events the writer made (a stopped call's `failed`
    `JournalError` `journal-barrier` in place of what it did not do);
  - `"trace"`: each send, `{"seq", "transport": the script's word,
    "answer": the journal's answer, or null when none came back}`, and
    each other writer's step, `{"other": "claimed" | "ended", "writer":
    the number its claim gave}`;
  - `"shown"`: the seqs given to readers, in order;
  - `"kept"`: `{"events": [positions], "finished"}`, what the journal
    holds at the end;
  - `"caller"`: `{"returns": value}` or `{"raises": error}`, where a
    `JournalError` `journal-end` is `{"type", "code", "journal":
    "refused" | "unknown", "event": position, "outcome": {"done": value}
    or {"failed": error}}`;
  - `"record"`: `{"error", "journal"?}`, the call record's;
  - `"settled"` (when the caller was told `"unknown"`): what reading the
    journal for `caller.raises.event` says: `"kept"`, `"not-kept"` (the
    log is unfinished) or `"another-end"` (another writer ended it).

  The errors in `caller` and `record` leave out `message` (the events
  in `log` have it).
- `receivers`: `{"scenarios": [{"layers": [{"where": "own" | "block" |
  "configure", "observers"?: [names], "journal"?: {"name", "mode":
  "required" | "best-effort"} or null}] (closest first), "expect":
  {"observers": [names], "journal": {...} or null} or {"refuses":
  "journal-policy", "observers", "journal"}}]}`. `"own"` is the
  program's own setting; `"block"` and `"configure"` are the host's. A
  layer without `"journal"` sets none; `"journal": null` sets "no
  journal". Journals are the same when name and mode are. Observers are
  listed outermost first. A refused scenario's `observers` and `journal`
  are where the refused tree's `started` and `failed` go.
