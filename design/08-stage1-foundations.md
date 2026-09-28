# 08 — Stage 1, foundations: the contract

*Status, 2026-09-28: contract written on the branch `stage1-contract`,
not implemented. Builds on `06-surfaces-research.md` and
`07-vignettes.md` (and the section "Maxime's answers" there, which
binds). The contract text is the authority; this note says why it is
what it is, what each implementer must build, and what is still open.*

Stage 1 lays four foundations every later stage stands on:

1. stream events a store can keep while they are written, and a client
   can resume after an event number;
2. a whole program (a module) declares its interface, so it can be
   described, served, conversed with and checked without running it;
3. a call records what it was shown (`saw`), so a turn can be asked
   again later with each helper's memory as it was;
4. per-input (and per-output) control of what the log keeps.

All four are true foundations, and I found no fifth that must come
first (see *What is not here*). Two of them turned out to be one piece of
work: once events are kept outside the process (1), they are a log, and
must obey the log's content rules (4). The contract says so (the *stored
form*), which the brief did not ask for but the design needs: without
it, `log_content=False` would leak through every store.

## What changed

| file | change |
|---|---|
| `contract/streaming.md` | format 2: `functai_event`, `stream`, `seq`, `at`, `request` on text and thinking, `root`/`program`/`content`/`omitted`/`saw` on `started`; law 3 restated with request numbers; *Replaying a stream*, *Keeping a stream while it is written* (sinks, the rules a store keeps), *The stored form*, *Formats* |
| `contract/programs.md` | new: the interface, its id, how each kind of program has one, checking values against it, where it is written |
| `contract/calls.md` | *Content* (the `log_content` map, which layer decides, what a record keeps, `omitted`); *Saw*; `program.signature` of a module; inputs named by the interface; `rated` rules 3 and 4 for records that keep some values; a module's version includes its interface |
| `contract/saved.md` | a module node's `interface`; *Describing without loading* |
| `contract/functions.md`, `contract/README.md` | pointers; which cases each language passes |
| `contract/schema/event.schema.json` | format 2, with format 1 kept (`if functai_event … else`) |
| `contract/schema/call.schema.json` | `omitted`, `saw`; `program` moved to `$defs` (events reference it) |
| `contract/schema/saved.schema.json` | `$defs/interface`, `field`, `input`; `interface` on nodes |
| `contract/cases/` | new generators `programs.py`, `content.py`, `saw.py`, `events.py`, `schemas.py`; `rated.py` and `saved.py` extended; `make.py` writes 8 folders and checks every case against the schemas as it writes it |

105 cases (43 new: 8 `programs/`, 13 `content/`, 9 `saw/`, 22
`events/`, `rated/12`, `rated/13`, `saved/12`; every `saved/` case gained
an `expect.describe`).

---

## 1. Events you can resume and keep

### What the contract now says

Every event carries:

```json
{"functai_event": 2, "kind": "text", "stream": "0192…a1", "seq": 57,
 "at": "2026-09-28T10:00:03.120000Z", "call": "0192…b7", "function": "answer",
 "field": "result", "answer": true, "request": 2, "text": "in Leeds"}
```

- `stream` is the id of the **watched call** (the call the stream was
  opened on, or the call a sink was given); `seq` counts that stream's
  events from 1 without gaps; `(stream, seq)` names an event.
- `at` is the call log's time format.
- `started` also carries `root`, the call log's `program` object, the
  `content` flag (and `omitted`), and `saw`, so a kept stream alone says
  what was running, for whom it was part of a tree, and what it was
  shown, even when the process died before the call log's line was
  written (the call log writes when a call ends).
- `text` and `thinking` carry `request`: which of the call's model
  requests wrote the piece.

A **sink** receives every event's stored form in `seq` order as it
happens, whether or not anyone watches the stream (a setting, like
`log_calls`); this is the hook stage 2's stores plug into. **The rules a
store keeps** are fixed now (kept, `duplicate`, `event-conflict`,
`event-gap`, `event-after-end`, `event-start`), because they are what
"persist a stream while it is written" means; stores themselves are
stage 2.

### Decisions and the alternatives an expert would weigh

**Name the sequence after the watched call, not the root or the turn.**
The brief suggested "the root call?". An event stream is a view of one
call and everything inside it; that call is usually a root, but not
always: a stream opened inside a module (Python `fn.stream()` in a
module's body) watches a child. If the numbering were per root, two
streams on one tree (the module's and the child's) would both number
from 1 under the same key, and a store could not keep both. Per watched
call, each stream is its own sequence, and an event watched by both is
two events (law 7). Stage 2's turn is one call of the conversation's
program, so a turn's stream key is its call id: vignette 3's
`store.watch(turnId, { after })` works unchanged. `turn` is not a field
here because turns do not exist yet; stage 2 can map turn to call.

**Dense integers, assigned by the one writer.** Considered: opaque
cursors (Kafka offsets, Redis stream ids, ElectricSQL's durable streams),
timestamps, or UUIDv7 per event. Dense integers let a reader detect loss
(a gap), let a store make appends idempotent (same `seq`, same bytes:
no-op) and detect a second writer (same `seq`, other bytes), and map
straight onto Server-Sent Events' `id:` / `Last-Event-ID`. The single
writer is the process running the call, which already serialises the
stream (law 1). An expert would reject opaque cursors here because
FunctAI owns the writer; opacity buys nothing and loses gap detection.

**A format marker on every event** (`functai_event: 2`), as CloudEvents
puts `specversion` on every event: kept events outlive the code that
wrote them, and a reader must be able to skip an unknown format. The
current event (no marker) is format 1; adding required fields is a new
format under the contract's own rule. Format 1 events were never kept, so
nothing on disk changes meaning; the schema still accepts them, which is
also why Python's existing schema test still passes.

**`at` on every event.** Cost: about 35 bytes per event, so roughly
35 KB on a reply of a thousand pieces. Benefit: replay at the pace it
happened, stalls visible, stores that expire by time, and every event
self-describing. Rejected: a relative offset from `started` (smaller, but
an event read alone means nothing). *Trade-off: events are larger.*

**How a voided text is represented.** Two mechanisms, both in the data:

- positional, as before: `retry` and (newly explicit) `tool_result`
  empty the call's fields at once;
- self-describing: every piece says which `request` wrote it, and a
  higher number starts the fields afresh.

In a whole stream they agree (every new request follows a `retry` or a
`tool_result`). The second exists for views that hide the first: the
outside view of vignette 11 hides retries (their reasons can quote a
helper) and tool events, and without request numbers an outside watcher
could not know the answer so far was voided. With both, a replay of the
kept events gives, after each event, exactly what a live watcher had
(`replay-*` cases), and a filtered view gives what its own watcher had
(`replay-08`, `replay-09`). Rejected: a new `request` event marking each
request (Vercel's `start-step`): it would work too, but adds a kind every
consumer must handle, and still leaves filtered views (which may hide it)
guessing.

Python today empties a call's fields when the new request is *sent*, a
moment no event shows; the contract moves that to the `tool_result`
before it (no piece can arrive in between, so only the instant differs).
*Implementers: the in-process "answer so far" must follow law 3 as
written.*

**The stored form.** An event written outside its process follows the
call's `log_content` exactly as the call's line does: values of fields
not kept become sizes (`text` → `size`) or disappear (`tool_call.input`,
`retry.reason`, `done.value` when the answer is not kept,
`error.message`), and every such event says `content: false`. Inside the
process, events stay whole (the caller has the values anyway). Who may
*see* what (owner or outside caller) is a separate view, stage 3's.
*Trade-off: a host that wants to broadcast a password helper's reply
live to the person who typed it, without keeping it, must do so from the
in-process stream, not through a store.*

**A sink never breaks the call**, like the call log. *Open for stage 2:*
a conversation store may be important enough that a host wants the call
to fail when the store cannot be written (see questions).

### What Python and TypeScript must build

- Events gain `stream`, `seq`, `at`, `request`, and `started` gains
  `root`, `program`, `content`, `saw` (and `omitted` in the stored form).
  Python: fields on the `Event` dataclasses, stamped by the `Stream` when
  it emits (per stream, so an event watched by two streams is two
  objects); `to_dict()` writes format 2. TypeScript: the `StreamEvent`
  union gains the fields; `stream.ts` stamps them in `push`.
- A request counter per call (the engine's `new_request`), carried by
  text and thinking pieces.
- The answer so far follows law 3 (reset at `retry` and `tool_result`).
- A function giving an event's stored form for its call's content
  (Python `event.stored()` or `to_dict(stored=True)`; TypeScript
  `stored(event)`), from the same resolution the call log uses.
- A sink setting (proposed: Python `events=` / `configure(events=...)`,
  TypeScript `events`, Julia `events`), called with each stored event in
  `seq` order, in any call it covers, streamed or not; its failure warns
  once.
- A replay function over event dicts (Python `functai.replay(events)`,
  TypeScript `replay(events)`) giving the view the cases describe; the
  reconnecting client and stage 2 need it anyway.
- Pass `cases/events/replay-*` and `stored-*`; `store-*` when stores
  exist (stage 2), or earlier with a small in-memory event log if the
  implementer prefers.

R has no streaming (`r/README.md`); when it has, events are lists with
the same names, and a replay is a tibble per call. Julia streams
(`julia/src`); its events are NamedTuples or structs with the same field
names (`seq`, `stream`, `at`), and the sink is a keyword (`events =
sink`).

---

## 2. A whole program declares its interface

### What the contract now says (`programs.md`)

```json
{"description": "Answer a customer's message.",
 "inputs": [{"name": "message", "shape": {"type": "string"}},
            {"name": "tone", "shape": {"type": "string", "default": "kind"}, "optional": true}],
 "outputs": [{"name": "result", "shape": {"type": "string"}}]}
```

- Every program has one. An AI function's is its definition's inputs and
  outputs (without the fields FunctAI adds: `reasoning`, the tool
  fields). A module declares its own.
- Its **id** is lmcc's signature fingerprint of the fields (all
  `purpose: plain`, `type: ""`). A module's call record's
  `program.signature` is that id; an AI function's `program.signature` is
  unchanged (its lmcc signature), and equals the interface id when it has
  neither reasoning nor tools (`programs/01`, `02`).
- A module's call **checks** its inputs before its code runs
  (`interface-input`) and its outputs when it returns
  (`interface-output`), by JSON Schema 2020-12 on the values' JSON form.
  A shape `{}` (unannotated, or a type with no JSON form, lmcc's own
  default for runtime-only types) accepts anything.
- A module's call record names inputs by the interface (never `arg0`),
  outputs by name; its version includes the interface.
- A saved module node has `interface`; every program node can be
  **described** without loading (`saved.md`), in any language: a module
  by its `interface`, an AI function from its `signature`.

### Decisions and alternatives

**Form: FunctAI's own definition form, not lmcc's field list, not
ProgramIR's.** `functions.md` already describes an AI function as
`inputs`/`outputs` of `{name, shape, desc?}`; the interface is that,
plus `optional` on inputs. ProgramIR's D-036 gave module nodes "a
signature in the same field-record form as leaves" and made "prediction
fields = declared outputs, exactly" checkable. Borrowed: the declared
signature on the module node, the checkability, and its structural
(never nominal) admission. Not borrowed: lmcc's `direction`/`purpose`
list form for the interface itself, because purposes (`reasoning`,
`tools.calls`) are how a model answers, not what a caller passes, and a
server or an R user should not have to filter them out.

**Checking both ends, always.** Pydantic makes return validation opt-in
(`validate_call(validate_return=True)`), and an expert in "light by
default" might ask for the same. I chose to check both ends on every
call, because an interface that is not enforced is documentation: a
served module or a conversation turn relies on it, and the check costs a
JSON conversion and a schema test per call, nothing beside a model call.
Lightness comes from `{}`: a module that declares no types is never
refused. *Trade-off: a Python module annotated `-> list[float]` that
returns a NumPy array now fails `interface-output` where it used to
work, unless the log's JSON conversion turns the array into a list.*

**Optional inputs** are a key (`optional: true`) and not only a shape
`default`, because TypeScript already has optional inputs whose shape
has no default (left out is `null`), and nullable is not optional there.
A default with a JSON form goes in the shape (what TypeScript and Python
tools already do). `optional` is not part of the id (data compatibility
does not change when an input becomes optional); a `default` in the
shape is (it is part of the shape, as lmcc fingerprints it today).

**A module's version now includes its interface.** Needed for
TypeScript, whose `input`/`output` are data beside `run` (its
`run.toString()` would not change when a type does). Applied to every
language for one rule. *Trade-off: every existing module's version
changes once; ratings do not pool across versions anyway (`rated`
filters by signature, not version), but comparisons by version across
that date will see two versions.*

**Where the interface is written.** In the saved node (module) and in
each language's program object (`support.interface`). Not in every call
record: records carry ids, as AI functions' do. I considered a
`functai_program` record in the call log, written once per version per
log file, so a reader of the log (Chattering's Programs pages, which
today guess allowed answers from the prompt, `design/74`) sees each
version's interface. It is additive and cheap, but it is a new record
kind with its own dedup and truncation questions and no stage-1 consumer;
I left it for stage 2, where the conversation store needs program
descriptions too. See questions.

**The AI node's description is its instruction.** Describing a saved AI
function reads its `signature`, whose `instructions` include
`Function: <name>` and guidance. *Trade-off: the description a server
shows for a saved AI function is the instruction text, not the
docstring; output `desc`s are not in saved signatures (they go into the
instruction). Adding `interface` to AI nodes too would fix both, at the
cost of a second copy that must agree with the signature. Left out for
now.*

### What Python and TypeScript must build

- Python: `FunctAIModule.interface` from `inspect.signature` and type
  hints (shape via the same `shape_of` as AI functions; defaults with a
  JSON form into the shape and `optional`; `*args`/`**kwargs` as one
  optional list/object each); `FunctAIFunc.interface` from its
  definition; checks at call and return; `program.signature` on module
  records; the version document gains `"interface"`; `save` writes the
  node's `interface`; `describe(path)` (or `functai.check`) reads it.
- TypeScript: a breaking `module(name, { description?, input, output |
  outputs, uses, settings? }, run)`, `run` given one object of inputs
  (and a second argument later for stage 3's `earlier`); `interface` on
  modules and AI functions; checks; records named by interface; version
  with interface; `describe` of a saved node.
- Pass `cases/programs/*` (AI cases need only the definition →
  interface step; module cases need a module built from an interface
  given as data, which the harness can do with a trivial `run`),
  `saved/*` `expect.describe`.

R has no modules. Spelled in R's own idiom, a module would be a formula
and a function:

```r
support <- ai_program(reply ~ message, function(message) {
  answer(message, topic(message))
}, message = "text")
ai_interface(support)      # a list, the contract's JSON form
```

Julia derives it from `@program function support(message::String)::String`
(`FunctAI.interface(support)`), shapes by the same type mapping as `@ai`.

---

## 3. A call records what it was shown (`saw`)

### What the contract now says (`calls.md`, *Saw*)

```json
"saw": [{"saw_of": "<turn 3's call>"}, {"call": "<turn 3's call>"}]
"saw": [{"call": "<t1>", "without": ["photo"]}, {"call": "<t2>", "steps": true}]
```

- The earlier calls a call was given as context, beyond its inputs and
  its program's state, in the order given: earlier turns placed in an AI
  function's requests; the conversation handed to a module's code.
  Not worked examples (state, in the version), not inputs, not children.
- Each entry says **how** the call was shown: `steps` (with its model
  replies and tool traffic) and `without` (fields left out). A later
  writer may add keys; a reader that does not know one must not replay.
- `{"saw_of": id}` as the first entry stands for everything that call
  saw.
- `[]` is "nothing"; absent is "not recorded", never "nothing". Every
  writer from now on writes it.
- The `started` event carries it too.

### Decisions and alternatives

**Ids, not copies.** dspy-session stores a snapshot of each helper's
memory in every turn; lmcc measured that copies grow with the square of
the conversation (F7). The call log already keeps every call, so ids
suffice.

**But ids alone are quadratic too.** Maxime's default is that the model
sees the whole conversation. With plain id lists, turn *n* lists *n−1*
ids: a thousand-turn conversation writes half a million ids (about 25
MB). `saw_of` makes the common case two entries per turn and the total
linear; windows (`last_turns(10)`) stay short lists. Rejected: ranges
("turns 1 to 40 of the branch"), because branches are stage 2's tree and
the call log has no turn order to range over; and a per-call flag
("everything before me"), because "everything" depends on the branch,
which a later reader must not have to reconstruct. *Trade-off: reading
a turn's `saw` follows a chain of records (linear in the conversation's
length); a lost record in the chain makes every later turn unreplayable
(`missing-call`), not just one.*

**How each call was shown, per entry.** Replaying needs to know not only
which calls but how they were written: with steps or not (vignette 11's
`remember(..., steps=True)`), without a bulky input or a photo
(vignette 10: "saw records whether the photo was sent"). A single
setting per call would not describe vignette 10's mixed turns. Unknown
entry keys refuse replay (`unknown-key`), like X.509's critical
extensions: a later stage can add a way of showing a call (a summary, a
merged turn from another program, vignette 9) without old readers
replaying it wrongly.

**Absent means unknown.** Python's `stateful=True` has shown earlier
turns since 1.0 without recording it. Reading an absent `saw` as `[]`
would replay those calls without their context, silently. The saved
manifest already uses this pattern (`language`, `body`, `version`:
"absent in folders written before…").

**Not recorded: the rendered request.** An exchange's `request` holds
the exact bytes but only with content on, and grows quadratically. lmcc's
turns keep a hash of the request (`request` on a model step); adding
that hash to exchanges would let a replay *prove* it rebuilt the same
request. Useful, not needed for stage 1; noted for stage 3.

### What Python and TypeScript must build

- Know the context before the first request, and write `saw` on every
  record and `started` event, `[]` when none. Python's `stateful=True`
  history: the ids of the calls in the window (the history must keep
  them). TypeScript has no memory yet: always `[]` until stage 2.
- A reader: `saw_of` expansion and the four reasons (`not-recorded`,
  `missing-call`, `unknown-key`, `saw-cycle`) over records
  (`cases/saw/*`). Stage 3 and stage 5 (`rated` with `earlier`) use it.

R and Julia: records gain `saw` (always `[]` today: neither has memory);
the reader is a function over records (R `ai_saw(records, call)`
returning a tibble or a reason; Julia `saw(records, call)`), needed by
their `rated` in stage 5.

---

## 4. What the log keeps, per input and output

### What the contract now says (`calls.md`, *Content*)

- `log_content` is `true`, `false`, or a map `{"transcript": false}`.
  Per field, the closest layer that answers decides (the function's own,
  blocks from the innermost, `configure`, the environment, `true`).
- A misspelt name in a program's own map refuses `log-content-field` at
  definition; in a block or `configure` it applies only where the field
  exists.
- A record that does not keep every value has `content: false`; when it
  keeps some, `omitted` lists the fields it did not keep and `inputs`,
  `outputs`, `returned`, `probabilities` hold the kept ones. It keeps no
  exchange request or reply and no error message.
- `rated`: a call whose inputs were not all kept is left out
  (`no_content`); a call that kept every input but not the answer makes
  a row from a correction, not from "right" (`no_answer`).

### Decisions and alternatives

**One setting, widened, rather than a second one.** Considered:
`log_omit=["transcript"]`, a size limit (`log_content_limit=128 KiB`,
Chattering's current rule), and per-field marks in the signature
(`Annotated[str, functai.not_logged]`). The map keeps one concept with
one name and spells naturally in all four languages (dict, object,
named list, NamedTuple); a block-level map covers Chattering's "every
program with a transcript input". A size limit is a policy the host can
build on top (it knows sizes before calling). Per-field marks in the
signature would put a logging choice into the program's identity.

**Per-field layering, not whole-map replacement.** With whole-map
replacement a program's `{"notes": false}` would silently undo a host's
`{"transcript": false}`. Per field, each layer answers only for what it
names; a bare `true` or `false` still answers for everything, so a
function's own `true` keeps working as today. *Trade-off: a host cannot
force a value out of the log for a program whose own setting is `true`;
that is today's rule for the boolean, and "the host owns policy" argues
for a host-level override that beats programs. Open question.*

**`content: false` widened, instead of a new call-record format.** The
honest alternative was `functai_call: 2` for records that keep some
values, as the contract's versioning rule demands when a meaning
changes. I weighed it and chose widening because every existing reader
stays *correct*: a format-1 reader sees `content: false`, treats the
call as having no values, and leaves it out of `rated` (`no_content`),
which is right when inputs are missing; and it never produces a row with
a missing input. A new format number would instead make old readers skip
these records entirely, which also hides them from tree sums (a
module's cost would silently lose a child). The README's rule exists to
protect readers; here they are protected without it. *Trade-off: the
word `content: false` now means "not every value", which is less
precise than its name; and the old schema (which forbade `inputs` when
`content` is false) would reject new records.*

**Drop every exchange message, and the error message, when any value is
left out.** A request holds every input; a reply, the model's thinking
or an error message can quote any of them. Searching them for the
omitted value is unreliable (a partial quote slips through). *Trade-off:
a call that keeps everything but a transcript loses its request and
reply in the log, so debugging it needs a live run. Chattering today
loses everything for such calls, so this is still a gain.*

**`sizes` stay complete**, as today, so a reader knows how big what was
left out was.

### What Python and TypeScript must build

- The setting accepts a map; per-field resolution through the layers;
  the refusal for a program's own misspelt map.
- Records written as `content/*` expect (the pure step: "the whole
  record, with these decisions, becomes this").
- `rated` reads `omitted` (rules 3 and 4; `rated/12`, `rated/13`).
- The stored form of events from the same decisions.
- A saved program keeps its map (`log_content` is already kept).

R: `log_content = list(transcript = FALSE)` (or `c(transcript =
FALSE)`); Julia: `log_content = (transcript = false,)`. Both must read
the new records in `rated` (`rated/13` fails in every language today).

---

## Cases, and which fail today

Every case below is written by a script from the rules, and checked
against the schemas as it is written (`make.py` → `schemas.py`).

| folder | cases | who must pass them | fails today |
|---|---|---|---|
| `rated/12-some-inputs-not-written` | 1 | every language | passes everywhere (old readers already leave it out) |
| `rated/13-every-input-written-an-output-not` | 1 | every language | Python, TypeScript (seen); R and Julia by the same logic (see the check results in the report) |
| `saved/12-a-module-and-its-interface` | 1 | every language | passes (the loaders refuse `saved-not-ai`, as expected) |
| `saved/*` `expect.describe` | 12 | every language | not read by any harness yet |
| `programs/` | 8 | AI cases: every language; module cases: languages with modules | no harness yet |
| `content/` | 13 | every language | no harness yet |
| `saw/` | 9 | every language | no harness yet |
| `events/replay-*`, `stored-*` | 16 | languages that stream (Python, TypeScript, Julia) | no harness yet |
| `events/store-*` | 6 | anything that keeps streams (stage 2) | no harness yet |

The new folders are not read by any harness today, so they cannot fail
yet: each implementer adds the harness with the code. Only `rated/13`
turns an existing harness red, in all four languages, until each reads
`omitted`.

## What is not here, on purpose

- **Stores, conversations, turns, `request_id`, stopping a call from
  another process**: stage 2. Nothing here blocks them: a turn's stream
  is keyed by its call id; `saw` holds call ids that turns can use; a
  store's append rules are fixed.
- **Knowing a stream is dead** (a writer that stopped without its last
  event): stage 2 (leases or heartbeats). Today a kept stream without its
  end is simply unfinished (`replay-07`).
- **Views** (owner, outside caller): stage 3. This note fixes only that a
  view applies on top of the stored form and may skip `seq` numbers.
- **Approval events, tool effects**: stage 4, as new kinds (a reader
  skips kinds it does not know; a new kind never changes old ones).
- **A program description record in the call log**: see questions.

## Open questions for Maxime

1. **A host override that beats programs for content.** Today a
   program's own `log_content=True` beats a host's block. Should a host
   be able to say "never write this field, whatever the program says"
   (policy is the host's), for example with a separate `log_never`
   setting that only removes?
2. **A sink that fails.** "Never in the way" (warn once, go on) is the
   call log's rule and the one written here. For a conversation store,
   should the host be able to make a failed append fail the call?
3. **Program records in the call log** (`functai_program`, once per
   version per file): add in stage 2 with the store, or now?
4. **`content: false` widened vs. a new record format.** I chose
   widening because old readers stay correct; the contract's letter
   (a meaning change is a new format) argued the other way. Confirm.
5. **Checking a module's outputs on every call**, or only where the
   interface crosses a boundary (serving, a store)? I chose every call.
6. **An AI node's `interface` in saved folders** (so descriptions carry
   the docstring and output descriptions), or derive it from the
   signature as now?
7. **`at` on every event** (35 bytes each), or only on the events that
   are not pieces?

## Trade-offs, in one list

- Events are bigger: `functai_event`, `stream`, `seq`, `at`, `request`
  add about 150 bytes to every piece.
- Old event consumers that switch on the fields they know are unaffected;
  consumers that validated against the old schema's closed `oneOf` still
  pass (format 1 is kept in the schema).
- The in-process "answer so far" empties at `tool_result` instead of when
  the next request is sent (a moment no event shows).
- A password helper's reply cannot be broadcast live through a store; only
  from the in-process stream.
- A module's outputs are checked on every call; a Python module returning
  a value whose JSON form does not fit its annotation now fails.
- Every module's version changes once (the interface joins its version).
- A saved AI function is described by its instruction, not its
  docstring, and without output descriptions.
- `saw_of` chains make one lost record hide every later turn's context.
- `saw` absent means unknown, so every record written before today,
  including every stateful Python call, can never be replayed faithfully.
- A record that keeps some values keeps no request, reply or error
  message.
- `content: false` now means "not every value", which its name says
  less precisely.
- A host cannot force a field out of the log against a program's own
  `true` (question 1).
- `rated/13` turns every language's existing harness red until each
  reads `omitted`: the stage cannot merge until all four do (the
  contract's rule: every implementation in the same commit).
