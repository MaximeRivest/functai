# 08 — Stage 1, foundations: the contract

*Status, 2026-09-28: contract written on the branch `stage1-contract`
(8f3f1d7), reviewed twice and corrected on `stage1-contract-v2`
(52e701d), re-reviewed twice and corrected again on `stage1-contract-v3`;
not implemented. Builds on `06-surfaces-research.md` and
`07-vignettes.md` (and the section "Maxime's answers" there, which
binds). The contract text is the authority; this note says why it is
what it is, what each implementer must build, and what is still open.
Sections 1 to 4 describe the contract as it is now. The two sections at
the end, **After review** and **After the second review**, are the
record of how it got here: every reviewer finding, and what was done
about it. Where the older of them says otherwise, sections 1 to 4 win.*

Stage 1 lays four foundations every later stage stands on:

1. a call tree's events, as one log that a store can keep while it is
   written, that a reader can follow and resume, and that a later writer
   can continue after the first one stopped;
2. every program declares its interface, so it can be described, served,
   conversed with and checked without running it;
3. a call records what it was shown (`saw`), so a turn can be asked
   again later with each helper's memory as it was, and a reader knows
   when it cannot be;
4. per-input (and per-output) control of what the log keeps.

All four are true foundations, and none of the reviews found a fifth
that must come first (see *What is not here*). Two are one piece of
work: once events are kept outside the process (1), they are a log, and
obey the log's content rules (4): the *kept form*.

## What changed

| file | change |
|---|---|
| `contract/streaming.md` | format 2: `functai_event`, `tree`, `writer`, `seq`, `after`, `at`; the `request` kind; law 8 (a request is an exchange); replaying, resuming (`event-unknown`), following (stale, duplicate, rewind, loss); the kept form; views; *Where each form can be read*; observers and journals, the required journal's barriers and what happens when it does not confirm (`JournalError` `journal-barrier` / `journal-end`); *Continuing a log* (claims, fencing); the rules a store keeps (claim, append, read); what a reader does with what it does not know |
| `contract/programs.md` | new: the interface (closed), its signature (shapes without defaults), which interfaces are refused (form and meaning field by field), the vocabulary "fits" uses, code-point order for names, how each kind of program has one (AI functions' optional inputs with defaults, Python's `Any` opaque, `*args`/`**kwargs`) |
| `contract/calls.md` | format 2: `omitted`, `program.interface`, `saw`, `request_hash`, `described`, `journal`; *Content* (only removes; added fields go together; `tools` is not a field); *Saw* (a context that changed; the turn a record stands for, steps deferred to stage 5; what the log must keep to show a call again); `rated` rules 3 and 4 for kept and described values; *Versions* with the interface |
| `contract/functions.md` | a definition's inputs may be `optional`, with a `default` in the shape; the signature's field leaves the default out |
| `contract/saved.md` | nodes' `interface`; loading takes optional inputs from it; describing without loading; the schema before the interface's own rules |
| `contract/README.md` | what a reader does with what it does not know (one table for everything); the refusal codes FunctAI defines; which cases each language passes |
| `contract/schema/` | `event` (format 2 with `writer`, kinds open), `call` (format 2 with `described`, `journal`), `interface` (new, its own file), `saved` (refers to `interface`) |
| `contract/cases/` | generators `programs.py`, `content.py`, `saw.py`, `events.py`, `schemas.py` new; `rated.py`, `saved.py`, `functions.py` extended; `make.py` writes 8 folders and checks every case against the schemas as it writes it |
| `python/tests/test_contract.py` | the saved-manifest fixture loads the interface schema beside the manifest's (a registry): the only change outside `contract/` and `design/` |

162 cases, 112 more than the 50 before stage 1 (see *Cases*).

---

## 1. A call tree's events: one log, kept, followed, continued

### What the contract says (`streaming.md`)

```json
{"functai_event": 2, "kind": "text", "tree": "0192…a1", "writer": 1, "seq": 57, "after": 56,
 "at": "2026-09-28T10:00:03.120000Z", "call": "0192…b7", "function": "answer",
 "field": "result", "answer": true, "text": "in Leeds"}
```

- **One log per call tree.** `tree` is the outermost call's id. `seq`
  numbers the tree's events densely in the whole log. `after` is the
  event before this one *in the form being read*. A stream opened on an
  inner call shows the same events with the same numbers (law 7).
- **Writers.** `writer` is the number of the process that numbered the
  event: 1, or the number a later writer got when it claimed the log.
  `(tree, writer, seq)` names an event.
- **`request` events.** A call's `request` and `retry` empty its fields.
  Every view keeps both, so no view keeps text the call voided. A
  call's *n*th `request` is its record's *n*th exchange.
- **Forms.** The *whole* log exists only in the running process. The
  *kept* form follows each call's `log_content`: no piece of a dropped
  field, not even its size, and no thinking. A *view* (stage 3 names
  them) leaves out events and values, never alters one, and keeps
  requests and retries. What makes a form leaves out kinds and keys it
  does not know.
- **Replaying, resuming, following.** A reader replays a form to get what
  a live watcher of it saw. It resumes after an event named by writer and
  seq; a source that lacks that event says `event-unknown`, and the
  reader starts again. A follower keeps a last event per tree. For each
  event it receives, it drops stale and duplicate ones and takes the
  next. When a later writer continues from an earlier point, it rewinds;
  otherwise it has lost events and reads again. It stops at a format it
  does not know.
- **Where each form can be read.** The process gives any form, while it
  runs. A store gives the kept form and views made from it, nothing
  else.
- **Observers and journals.** Observers get the kept form, best effort.
  Journals keep whole trees' kept logs with conditional, answered
  appends, and come in two modes:
  - *Best effort* (the default): the call never waits.
  - *Required*: the call waits at three barriers: its start, before each
    tool, and its end. The outcome is decided before the journal is
    asked, and never changed by its answer. If the barrier at the start
    or before a tool is not confirmed, the call stops with `JournalError`
    `journal-barrier`. If the end is not confirmed, the caller gets
    `JournalError` `journal-end`, which holds the outcome and says
    `"refused"` or `"unknown"`, and the record says the same. The last
    event is shown to readers only once kept.
- **Continuing a log.** A later writer claims the log from the store,
  which gives it the next writer number and fences every earlier writer.
  It numbers on from the last kept event, reusing numbers of events
  never kept (the writer number tells them apart). The next request
  number comes from the kept `request` events.
- **The rules a store keeps.** Claim, append (malformed, duplicate,
  conflict, fenced, after-end, gap, start: in that order), read.

### Decisions, and the alternatives an expert would weigh

**Dense integers under one writer, not opaque cursors.** Kafka offsets,
Redis stream ids and UUIDv7 per event were weighed. Dense integers give
three things that opaque cursors lose, since FunctAI owns the writer.
They let a reader detect loss (with `after`) and let a store make appends
idempotent (same seq, same bytes). They also detect a second writer
(same seq, other bytes). And they map onto Server-Sent Events' `id:`, as
`writer` and `seq` together.

**One log per tree, source positions in every form, `after` for each
form's chain** (the first repair's choice, endorsed by both
re-reviews). This is event sourcing's shape: one stream per aggregate,
with projections over it. A view's positions do not change when the
view's rules do (EventStoreDB's filtered subscriptions). `after` is the
conditional-append position and the follower's loss check in any form.

**Writer numbers, issued by the store (the second review's blocker).**
The first repair let a later writer "number on from the last kept
event". Both re-reviewers showed that this reuses `(tree, seq)`
identities a live reader already holds. The reader drops the new events
as duplicates and never sees the end. Three answers were weighed:

- *Durable position allocation before publication* (Astra's first
  option): reserve seq ranges in the journal before showing any event
  live, so a number is never reused. Rejected: it puts a durable write
  on the path of every live event. A best-effort journal that is down
  would then stall live streaming, which "best effort never gets in the
  way" forbids. And a reader holding events that were never kept still
  has to learn that the log went on from an earlier point, so a reset
  rule is needed anyway.
- *A writer term chosen by the writer* (Opus's proposal, read
  literally: "the last term + 1"). Written into the generator first,
  this failed its own case (`store-10`). Two writers continuing at once
  both pick 2, and the store cannot fence the loser. Worse, if their
  first events are byte-identical (the clamped `at` makes that possible),
  the loser's append is answered `duplicate` and both believe they own
  the log.
- *A writer number issued by the store* (chosen): a claim is an epoch
  bump at the store, as Kafka's controller issues leader epochs and
  Raft's vote issues terms. Numbers are unique per log. A claim fences
  every earlier writer at once, even one still running whose next event
  would fit the chain. Readers rewind when a later writer's `after` is
  below their last event (Kafka KIP-320's divergence check). Stage 2's
  lease is a policy on who may claim; the mechanism is here.

**Which forms can be resumed where.** Opus's second failure was a reader
of a form the store cannot make (a live owner view showing a field that
is never kept). Such a reader looped on "loss" for ever. The answer has
two parts. First, name what a store can give: the kept form and views of
it. Second, give a reader whose last event the store lacks a definite
answer (`event-unknown`), so it starts again in a form the store gives.
The reader loses what was shown live and never kept, which is exactly
what retention promised.

**The terminal outcome when an acknowledgement is lost (the second
review's other blocker).** Uncertainty about a commit is not failure of
the computation. The outcome (value or error) is decided before the
journal is asked, and is recorded in the terminal event and in the call
record. A required journal adds *confirmation*, which is confirmed,
refused, or unknown:

- *Refused*: the journal answered, and did not keep the event.
- *Unknown*: no answer came; the event may be kept. The writer cannot
  tell a lost answer from an append that never arrived; only the store
  can.

The caller gets `JournalError` (`journal-end`) holding the outcome. The
record keeps the outcome and says `journal`. The terminal event is shown
to readers only once kept (commit, then publish: the transactional
outbox's rule), so no reader sees an end a store does not hold. The
alternative, turning a lost acknowledgement into a failed call, is what
Astra showed to be false: the store would say success while the caller
and the record said failure. The writer cannot repair that, since the
store rightly refuses a `failed` after a kept `done`.

**Barriers: start, each tool, end.** Only the outermost call has start
and end barriers. Appends are in order, so the tree's end covers every
child. The tool barrier keeps the *fact* that a tool was asked for
before it runs, so a watcher can later say "may have run". It is not a
checkpoint: with content not whole, the kept `tool_call` has no input
(Opus S2). The checkpoint a waiting turn resumes from is a separate
record, stage 4's, with its own barrier before effects.

**`at` on every event.** It costs about 35 bytes per event. It gives
replay at the pace things happened, visible stalls, expiry by time, and
self-describing events. Order is `seq`'s. `at` never goes back (a clock
set back repeats the last time).

**Forward compatibility, once for everything.** Opus S1 found the schema
refusing event kinds the prose told readers to skip. The rule is now one
table in `contract/README.md`. A reader that only reads skips what it
does not know. Anything that decides what is kept, shown, accepted or
replayed fails closed:

- form makers leave unknown kinds and keys out;
- `saw` readers refuse an entry they do not know;
- interface readers refuse a key they do not know.

This is X.509's critical extensions, and JSON Schema 2019's required
vocabularies, applied per role instead of per key. Event kinds are open
in the schema.

### What Python and TypeScript must build

- Events: `tree`, `writer`, `seq`, `after`, `at` (clamped), the `request`
  event, pieces without `request`; one numbering per tree shared by every
  stream opened in it; law 3 as written (the answer so far empties at
  `request` and `retry`); `tool_call.id` always set.
- The kept form (`kept-*`), leaving out unknown kinds and keys; replay
  and follow over event dicts (`replay-*`, `follow-*`: stale, duplicate,
  rewind, loss, unknown format; one state per tree).
- Observers and journals as settings (Python `observers=`, `journal=`;
  TypeScript `observers`, `journal`), best effort and required. The
  writer's confirm/resend loop, the three barriers, `JournalError`
  `journal-barrier` / `journal-end` with `outcome` and `journal`, the
  terminal event withheld from readers until confirmed (`journal-*`).
- An in-memory store with claim, append and read (`store-*`), so the
  rules are tested before stage 2's stores exist.

R has no streaming yet; its events will be lists with these names. Julia
streams: NamedTuples or structs with these field names, and
`journal = j` as a keyword.

---

## 2. Every program declares its interface

### What the contract says (`programs.md`)

```json
{"description": "Answer a customer's message.",
 "inputs": [{"name": "message", "shape": {"type": "string"}},
            {"name": "tone", "shape": {"type": "string", "default": "kind"}, "optional": true}],
 "outputs": [{"name": "result", "shape": {"type": "string"}}]}
```

- **What an interface is.** Every program has one. An AI function's is
  its definition's inputs and outputs, without the fields FunctAI adds.
  An AI function's optional input always has a default in its shape: a
  model is sent every input, and TypeScript's default is `null`. A
  module declares its interface. In Python it is derived from the
  function: `Any`, `object` and unannotated arguments are opaque, and
  `functai.JSON` means any JSON.
- **Keys.** The interface is closed: a key it does not name is refused,
  not ignored. `opaque` is for values that may have no JSON form.
  `optional` means an input may be left out; left out, it takes its
  shape's default, or else stays out (never null).
- **Its signature.** The signature is lmcc's fingerprint of the fields.
  Each shape is taken *without its own `default`*, all plain and
  untyped. It says what recorded data looks like, not what is accepted:
  `optional`, `opaque`, defaults and words are not in it. It is the
  record's `program.interface`.
- **Checks.** A module checks its inputs before its code runs and its
  outputs when it returns. "Fits" uses a listed vocabulary:
  - `type`, `enum`, `const`, `anyOf`;
  - the array, object, string-length and number-bound keywords;
  - `$ref` into the shape's own `$defs`;
  - annotation words, which are never checked.

  Integers are numbers with no fraction. Equality is canonical JSON.
  Lengths count code points. Any other keyword refuses the interface.
- **Refusals.** Form and meaning are checked together, field by field.
  The first field at fault is named. Among several unknown names, the
  first in code-point order is named. For outputs, unknown keys are
  checked before missing ones.
- **Where it is written.** The interface is written in a saved node
  (loading an AI function takes its optional inputs from it) and in
  every record, as the signature. A module's version includes it.

### Decisions, and the alternatives an expert would weigh

**FunctAI's own definition form, not lmcc's field list, not ProgramIR's.**
Borrowed from ProgramIR (D-036): the declared signature on a module
node, checkability, and structural admission. Not borrowed: purposes in
the interface, because a caller does not pass a `tools.calls` field.

**Check both ends on every call.** An interface that is not enforced is
documentation. The check costs a JSON conversion and a schema test.
Lightness comes from `{}` and `opaque`: an undeclared module is refused
only for its names.

**Defaults are behaviour, not data** (Opus S4, the second review). The
first repair kept `default` in the fingerprint, for lmcc parity. That
split a program's records whenever a default changed, including a
default computed when the program loads (a date), so ratings split per
process. In JSON Schema a default is an annotation. Behaviour belongs in
the version:

- a module's default is in its version, through its interface;
- an AI function's default is bound before the request, and the record
  holds the value used;
- lmcc never sees a default: the signature's fields drop it
  (`functions/12`).

*Cost:* two versions of a module that differ only by a default pool
their ratings. That is right: the records say which value each call
used.

**A portable vocabulary for "fits"** (Opus S6). "Draft 2020-12" alone
is not portable. TypeScript has zod; R and Julia have uneven validators;
`format`, `pattern`, `multipleOf` and remote `$ref` differ between them.
The expert choice is to name the keywords, say how each reads, and
refuse the rest. The generator checks values by those rules and asserts
that a 2020-12 validator agrees on every case. The keywords are the ones
Python's `shape_of` and pydantic write, less the non-portable ones.

*Cost:* a module annotated with a pydantic `constr(pattern=...)` refuses
at definition. Declare it opaque, or use a plainer type. A later
contract may add keywords, and older readers refuse them (fail closed).

**Opaque stays out of the signature** (Astra S3). A field `{}` and an
opaque field `{}` have one signature, and records from both pool. The
danger Astra found was descriptions (`$type`, `$repr`) passing as data.
It is closed where it matters, not by splitting identity:

- records now name their descriptions (`described`);
- `rated` leaves such inputs out (`no_content`, `rated/16`);
- replay refuses them (`not-kept`, `saw/15`).

An opaque field given `[1, 2]` recorded real data, and that pools
correctly. A descriptor record in stage 2 must be keyed by the whole
interface's hash (or the version), never by this data signature.

**Optional inputs.** Omission is preserved for modules: the program's
own default applies. AI functions must send something, so theirs always
have a JSON default in the interface. That is what makes save → describe
→ load → call with a left-out input send the same bytes in another
language (`saved/16`'s `sends`).

**The interface is closed** (Opus S1, on interfaces). A later key (a
label for private data, question 1) may narrow what is accepted or say
how a field is kept. An old checker that ignored it would fail open.

*Cost:* an older reader cannot describe a newer interface until it
learns the key.

**Its own schema file** (Astra M3). It is a shared type: manifests,
program objects, stage 2's descriptors and stage 3's served
descriptions all use it. Python's saved-manifest test loads it through a
registry now (one fixture changed).

### What Python and TypeScript must build

- `interface` on every program object; the module checks with the
  vocabulary, refusals and orders above; records named by interface;
  versions with it; `save` writes each node's interface; `describe(path)`.
- Python: `Any`/`object` opaque for modules, `functai.JSON`; defaults
  never in the lmcc signature; an AI function's optional input with a
  non-JSON default refuses at definition.
- TypeScript: the breaking `module(name, { description?, input, output |
  outputs, uses }, run)`; an optional AI input without `.default()` gets
  `default: null` in its interface shape (not in its lmcc shape).
- Pass `programs/*` (`binds` for AI cases, `checks` for modules),
  `functions/12`, `saved/*` including `describe` and `saved/16`'s `sends`.

R: `ai_program(reply ~ message, function(message) ...)`, unannotated
arguments opaque until `types = list(message = "text")`. Julia:
`@program`, `Any` opaque.

---

## 3. A call records what it was shown (`saw`)

### What the contract says (`calls.md`, *Saw*)

```json
"saw": [{"saw_of": "<turn 3's call>"}, {"call": "<turn 3's call>"}]
"saw": [{"call": "<t1>", "without": ["photo"]}, {"call": "<t2>", "steps": true, "slot": "helper"}]
```

- The earlier calls a call was given as context, in order, and how
  each was shown:
  - `steps`: with its own model and tool steps, never deeper calls;
  - `without`: fields left out; with `steps`, never the tool calls;
  - `slot`: the lmcc slot it was placed in.
- `saw_of` as the first entry stands for everything that call saw, so a
  whole conversation's records grow linearly. `[]` is "nothing"; absent
  is "not recorded".
- A call whose context changed between its requests writes
  `{"context": "changed"}` last. Readers today say `unknown-key`, not a
  list that is true only of the first request.
- **Knowing** (the entries, `saw_of` expanded) is separate from the log
  **keeping** what showing them again needs:
  - the record exists, is not truncated, and holds the values shown, as
    data (not descriptions);
  - with `steps`: every exchange's request hash, and its reply when one
    came;
  - an entry no call can have been shown refuses `turn-invalid`.
- **The turn a record stands for.** The signature is
  `program.signature`, the inputs are the record's inputs, and the
  outputs are the record's `outputs`, not `returned`. How exchanges
  become steps is deferred to stage 5, with a list of what it must
  settle. Until then, calls are shown without steps.

### Decisions, and the alternatives an expert would weigh

**Ids, not copies; `saw_of`, not id lists.** Copies grow with the
square of a conversation's length (lmcc F7), and so do id lists under
"the model sees the whole conversation". `saw_of` keeps it linear.

*Cost:* one lost record hides every later turn's context
(`missing-call`).

**Absent means unknown.** Python's `stateful=True` shown history was
never recorded. Reading absent as `[]` would replay without context,
silently.

**"Replay OK" renamed to what it proves** (Opus S7, Astra S2). The
first repair reported `replay: ok` on retention alone. Astra showed
three failures:

- a record with its replies stripped passed;
- a description of an opaque value passed as the value;
- `without` on the tool-calls field with `steps` made a turn that lmcc
  refuses (`turn-invalid`, reproduced with the real kernel).

Retention, reconstruction and rendering are different operations. The
cases now say `keeps`, check the replies, the request hashes and
`described`, and refuse the invalid entry with lmcc's own word.

The record-to-turn operation for steps is honestly deferred, with the
exact list Opus gave:

- which exchanges become model steps;
- how each step's outputs are read;
- where tool steps come from;
- which lmcc `replay` mode writes them.

Specifying it now would mean guessing FunctAI's retry transcript, which
lmcc turns cannot hold (a retry's correction message is no step kind).
Stage 5 must also check each rebuilt request against `request_hash`.

**A context that changes mid-call** (Opus minor, compaction in agents).
A writer that cannot describe the change must not write a list that is
only true of the first request. An entry no reader knows makes every
reader say "not known" today, without a format change.

### What Python and TypeScript must build

- `saw` on every record and `started` event (`[]` when none); Python's
  `stateful=True` history as ids.
- The reader (`saw/* read`): expansion, the four reasons, `keeps`.
- The `shown` rule when conversations come (stage 2/3).

R and Julia: records gain `saw` (`[]`) and `described`; the reader is a
function over records, needed by their `rated` in stage 5.

---

## 4. What the log keeps, per input and output

### What the contract says (`calls.md`, *Content*)

- `log_content` can be:
  - `true`;
  - `false`;
  - a map from field names to booleans, where `"*"` stands for the
    fields the map does not name.
- A value is written only when no layer drops it. The layers are the
  program's own setting, blocks, `configure`, and the environment. The
  environment's `0` drops everything; `true` never keeps what another
  layer drops.
- The fields FunctAI adds (`reasoning`, `calls`) go together, and go
  whenever any field goes. `tools` is not a field: naming it in a
  program's own map refuses.
- A misspelt name in a program's own map refuses `log-content-field`.
  In a host layer it cannot be checked; `"*": false` is the host's safe
  list. A key that is not a name refuses everywhere, kept for labels.
- **The record.** A record that does not keep every value:
  - is format 2, with `content: false` and `omitted` naming the fields
    not kept;
  - keeps no exchange request, reply or request hash, and no error
    message;
  - keeps `sizes` whole.
- **The event log.** The kept form of events follows the same decisions.
- **Scope.** `log_content` is retention for watching. It is not
  disclosure (live views) and not memory (conversation stores, stage 2).
- **`rated`.** A call whose inputs were not all kept as data is left out
  (`no_content`). A call that kept every input but not the answer makes
  a row only from a correction.

### Decisions, and the alternatives an expert would weigh

**One setting, widened; per-field layering that only removes.** A size
limit is the host's own policy. Per-field marks in the signature would
put logging into the program's identity. Only-removes is what makes
"the host owns policy" true: a program can ask for less, never for more.

**Added fields go together** (Opus S5, Astra S1). Two rules were
possible: an added field goes when a plain field goes (the old
generator), or when any field goes (the old prose). An added field can
quote another (a tool call can repeat the reasoning), so the
conservative closure is the privacy rule, and it is the simpler one to
state. *Cost:* a host that hides only the tool calls also loses the
reasoning.

**Format 2, not a widened `content: false`** (the first repair, both
re-reviews agreeing). A format number is the only promise a reader in
another language, years later, can check.

**Drop every message when any value is dropped.** A request holds every
input, and a reply or an error can quote any of them. Searching them for
the dropped value is unreliable.

*Cost:* such calls cannot be debugged from the log.

### What Python and TypeScript must build

- Maps with `"*"`, only-removes layering, added fields together, the
  refusals; records as `content/*` expect; `described` and `journal` on
  records; `rated` reads `omitted` and `described` (`rated/12` to `16`).

R: `log_content = list("*" = FALSE, question = TRUE)`. Julia:
`Dict("*" => false, "question" => true)`. Both read format 2 in `rated`.

---

## Cases, and which fail today

Every case is written by a script from the rules and checked against the
schemas as it is written. `make.py` run twice gives no difference. It
also checks that the schemas refuse 16 leaking or contradictory objects
and accept 3 a later writer may write.

| folder | cases | who must pass them | fails today |
|---|---|---|---|
| `functions/` | 12 (`12` new) | every language | `12` wherever a language puts a default into its signature (Python passes; TS, R, Julia to be seen) |
| `rated/` | 16 (`12`–`16` new) | every language | `13`–`16` in every language (format 2 not read yet) |
| `saved/` | 16 (`12`–`16` new) | every language | `14` in TS, R, Julia; `16` where the loader ignores the interface; `describe` and `sends` have no harness |
| `programs/` | 15 | AI cases every language; others with modules | no harness yet |
| `content/` | 19 | every language | no harness yet |
| `saw/` | 16 | `read` every language; `shown` with replay | no harness yet |
| `events/` | 51: `replay` 14, `follow` 8, `kept` 9, `store` 10, `journal` 10 | streaming languages; `store-*` with stores; `journal-*` with journals | no harness yet |
| `scores/` | 17 | every language | unchanged |

## What is not here, on purpose

- **Stores, conversations, turns, `request_id`, leases, stopping a call
  from another process**: stage 2. Everything here is shaped for them:
  - a turn's log is keyed by its call id;
  - claims give leases their mechanism;
  - a store's rules are fixed.
- **Knowing a writer is dead** (heartbeats, lease expiry): stage 2. Until
  then an unfinished log is simply unfinished.
- **Views and who chooses them; forwarding a child's field as a module's
  answer; re-attributing a nested approval to the boundary**: stage 3.
- **Approval events, tool effects, the resume checkpoint, a tool
  invocation id unique within a call**: stage 4, as new kinds and keys
  under the extension rule.
- **Showing a call again with its steps**: stage 5.

## Open questions for Maxime

See the end of *After the second review*.

## Trade-offs, in one list

Everything the contract now costs, whichever review it came from:

**Formats and compatibility**
- Format 2: a format-1-only reader skips new records (its sums over a
  tree miss them) until it learns format 2; every implementation reads
  both.
- `saw` is required in format 2; format-1 records stay unreplayable.

**Events and logs**
- Events are bigger: `functai_event`, `tree`, `writer`, `seq`, `after`,
  `at` add about 170 bytes to every piece, and `request` events add one
  event per model request.
- A reader following a live form that shows values never kept loses them
  if it must resume from a store: it starts again in the kept form.
- A rewinding reader either keeps past states or reads again from the
  start.
- A later writer reuses the seq numbers of events never kept. Only
  `writer` tells them apart, so any tool that indexes events by
  `(tree, seq)` alone is wrong after a hand-over.
- A claim fences the earlier writer even when it is still running.
  Stage 2's lease must decide who may claim, or a careless claim stops a
  healthy call.
- Readers and stores skip or drop unknown kinds and keys: an older store
  making views strips a newer writer's keys.

**Journals**
- A required journal adds waits at a call's start, before each tool, and
  at its end, and can make a call raise `JournalError`. A caller must
  handle `journal-end` holding a good outcome.
- The last event of a call with a required journal reaches readers only
  once kept: one journal round trip of extra latency at the end.

**What the log keeps**
- `true` in `log_content` never keeps what another layer drops. The
  environment's `0` drops everything.
- Any dropped field takes the reasoning, the tool calls, every request,
  reply, request hash and error message with it: such calls cannot be
  debugged from the log.
- Per-field retention does not follow values: a value copied into a kept
  field is kept (labels: question 1).
- `sizes` still tell a dropped value's length.
- A kept log shows no progress for a dropped field.
- Observers get only the kept form.
- Journals keep whole trees only.

**Views**
- Views must show requests and retries: an outside caller learns that a
  request was retried.
- A module's boundary view shows no reply while it is written (stage 3).

**Interfaces**
- Defaults are out of the signature: records of versions that differ
  only by a default pool (the record says which value each call used).
- A value with no JSON form never fits a non-opaque field: a Python
  module annotated `Any` is opaque, and cannot be served until it
  declares data.
- Returned keys that are not declared outputs are refused.
- Interfaces are closed: an older reader refuses a newer interface's key
  instead of describing it.
- Module shapes are limited to the listed keywords: `pattern`, `oneOf`,
  `multipleOf` refuse.
- AI functions' optional inputs must have a JSON default.
- A saved folder carries each AI node's interface beside its signature,
  checked to agree.

**Replay**
- `saw` cannot yet show calls with their steps (stage 5). A call whose
  context changed mid-call is "not known".
- Opaque and any-JSON fields share a data signature; `described` keeps
  descriptions out of rows and replays, but a descriptor must not be
  looked up by that signature.

**Merging**
- This branch cannot merge alone. `rated/13`–`16`, `saved/14` and
  `saved/16`, and `functions/12` where a language differs, fail until
  each implementation follows (AGENTS.md: every implementation in the
  same commit).

## After review (2026-09-28)

*History: the first repair's record (52e701d), kept as it was written.
Sections 1 to 4 above describe the contract as it is now, and the next
section says what changed after the second review; where this section
differs (its "What Python and TypeScript must build now", its case
counts, per-writer numbering, "replay"), they win.*

Two reviews of 8f3f1d7: Codex Astra ("redo the affected foundations",
delegation 42f9104f, with runnable counterexamples) and Opus ("accept
with fixes", delegation fbd0cc3f). Both found real faults; where they
agreed, they were right every time I checked. This section says what the
corrected contract (branch `stage1-contract-v2`) does about each finding,
where they disagreed and what I chose, what it costs, and what is left
for Maxime.

### What changed, in short

- **The call log is format 2.** A record may keep some values and not
  others (`omitted`, always present when `content` is false); every
  record has `program.interface` and `saw`; `program.signature` is for
  AI functions only; exchanges gain `request_hash`, and lose every error
  message when content is not whole. Format 1 stays exactly as it was at
  8cd4597 (the schema checks both).
- **`log_content` only removes.** A value is written only when no layer
  drops it; `FUNCTAI_LOG_CONTENT=0` drops everything; `"*": false` lets a
  host list what may be kept. The fields FunctAI adds (reasoning, tool
  fields) are fields of the call, and go whenever any other field goes.
  `log_content` is about what is kept for watching, not what a reader
  allowed to see a value is sent, nor what a conversation keeps.
- **One log per call tree.** Events carry `tree` (the outermost call's
  id), `seq` (dense in the whole log) and `after` (the event before it in
  the form being read). Streams, the kept form and views are forms of one
  log: an event has one name, `(tree, seq)`, everywhere.
- **A `request` event**, which with `retry` is what empties a call's
  fields. Every view keeps both (a retry without its reason), so a view
  never keeps text the call voided. Pieces no longer carry `request`.
- **The kept form** (was "the stored form") keeps no piece of a field it
  does not keep, not even its size, and no thinking; `done.value` only
  when every output it holds is kept.
- **Observers and journals** instead of one kind of sink: observers are
  best effort; journals keep whole logs, with acknowledged, conditional
  appends (`after`), best effort or required (the call waits before its
  first request, before each tool runs, and before it returns). A log is
  unfinished until its end is kept, and a later writer may continue it.
- **Interfaces**: the id is called the interface's *signature* and says
  only what data looks like; an optional input left out stays left out
  (never null); `opaque` fields (no JSON form) are told apart from `{}`
  (any JSON); a return type is one output; returned keys that are not
  outputs are refused; malformed interfaces are refused
  (`interface-malformed`); saved AI nodes carry `interface`, checked
  against their signature.
- **`saw`**: knowing what a call saw is separated from being able to show
  it again (`missing-call`, `not-kept`); entries gain `slot`; the schema
  accepts entries a later writer adds; `steps` is never deeper calls; the
  turn a `without`/`steps` entry stands for is defined and pinned.
- **A capability matrix** in `contract/README.md` says which cases each
  language must pass.

### Every finding, and what was done

Astra's findings:

| id | finding | done |
|---|---|---|
| B1.1 | `done.value` of several outputs keeps an omitted output | `done.value` defined (an AI function's answer; a module's outputs) and kept only when every output it holds is kept (`kept-07`). The schema refuses a `done` with a value marked not whole. |
| B1.2 | nested exchange error messages survive | Every exchange's `error.message` goes when content is not whole (`content/02`, `03`); the schema refuses it. |
| B1.3 | implicit fields (reasoning, tools) have no rule | They are fields of the call: nameable in a map, and dropped whenever any other field is (`content/03`, `08`, `15`; `kept-08`). |
| B2 | a mandatory best-effort sink blocks durable approval and restart | Observers (best effort) and journals (acknowledged, conditional appends; best effort or required, with barriers before the first request, before each tool runs and before the call ends; the writer keeps what is not acknowledged). "Unfinished for ever" is gone: a later writer may continue a log (`store-08`); fencing and ending are stage 2/4's. |
| B3 | filtered streams: no coherent gap rule; stale text after a hidden retry | `after` on every event gives each form its own chain, so following detects loss in any form (`follow-*`); views must keep `request` and `retry` (`replay-08` to `11`; Astra's counterexample is `replay-10`); resuming is in the same form, from the reader's own state. |
| B4 | the call format changes meaning without a bump | Format 2 (see *Disagreements*). |
| B5 | the interface id does not identify behaviour; impossible optional inputs; saved AI descriptions lose `optional` | The id is the interface's *signature*, stated to be about data only (`programs/12`); omission is not null (an optional input with no default stays left out: `programs/06`); a default that does not fit is refused; AI nodes carry `interface`, checked (`saved/13`, `14`). |
| S1 | hosts cannot enforce retention; retention conflated with transport | Only-removes layering; `"*"`; retention (kept form, call log) separated from disclosure (views, live delivery) and from conversation memory (stage 2). Non-transitivity stated in `calls.md`. |
| S2 | `saw` is lineage, not a replay specification | Separate reading and replaying (`saw/10`, `11`); `without` with `steps` defined (`saw/12`, `13`); `slot`; one context per call stated; `steps` never deeper calls; `request_hash` on exchanges. |
| S3 | `{}` means both any JSON and no JSON form | `opaque: true` (shape `{}`) for no JSON form; `{}` alone is any JSON; a value with no JSON form fits only an opaque field (`programs/08`, `09`). The stage-1 refusal `interface-untyped` is gone: stages 2 and 3 say how their boundaries refuse opaque fields. |
| S4 | definition validity, refusal precedence, conformance coverage | `interface-malformed` with its order (`programs/11`); input and output check order in `programs.md`; the README's matrix. Harnesses per language are the implementations' work, not written here. |
| minor | `at` never decreasing on a wall clock | `at` is the writer's clock when it numbers the event, clamped to never go back; order is `seq`'s. |
| minor | "never refused" overclaims; resume vs reconstruct | "Refused only for its names"; resuming continues the reader's own state (said in *Replaying*). |
| minor | case count | 55 new at 8f3f1d7 (fixed above); 81 now. |
| vignettes | `tool_call` id nullable; child calls not linked to their tool invocation; disconnect vs cancel; "saved already" | `tool_call.id` is required (lmcc always has one). The tool-invocation link on a child's `started`, and a tool invocation id unique across a call's requests, are stage 4's (lmcc's `call_1` repeats across replies). A reader that stops reading changes nothing (*Closing*). "Saved already" needs a required journal's acknowledgement of `started`, plus stage 2's turn record. |

Opus's findings:

| id | finding | done |
|---|---|---|
| B1 | content fails open | Only-removes layering; environment `0` absolute (`content/05`, `06`, `07`). Misspelt names in host layers: `"*": false` (see *Disagreements*). |
| B2 | the stored form covers conversation stores | `log_content` is about what is kept for watching; conversation stores get their own setting in stage 2, and must refuse, not forget, when the two cannot both hold. |
| B3 | "unfinished for ever" forbids durable turns | Removed; a later writer continues after the last kept event; the next request number comes from the log's `request` events. |
| B4 | gap detection contradicts views | `after` (see *Disagreements*). |
| B5 | per-piece sizes leak | No piece of a dropped field is kept, not even its size; thinking is not kept; `tool_result` keeps no size either. |
| S1 | name-based redaction does not follow the value | Stated in `calls.md`, *What it does not do*. Labels are a question for Maxime; map keys that are not names are refused now, kept for them. |
| S2 | `program.signature` means two things | `program.interface` on every record; `signature` for AI functions only; `rated` compares interfaces (`rated/14`). |
| S3 | sink scope undefined | Observers: every call in scope. Journals: the whole log of every tree whose outermost call starts in scope; a journal set only inside a tree warns (or, required, refuses `journal-scope`). A stream and a journal on one tree share one numbering. After a failed append: resend (idempotent); after `event-conflict`: stop for good. Observers never slow the call; required journals wait only at barriers. |
| S4 | an outside view cannot show a module's reply as written; per-root numbering | The view rules no longer say "views only skip"; a module boundary shows `started` and `done` until stage 3 adds forwarding (`replay-12`). Per-tree numbering adopted. |
| S5 | stores need a conditional append; order of checks | `after` is the expected position; the rules table has an order. |
| S6 | `saw` slots; schema blocks new entry kinds | `slot`; the schema accepts any entry object (`saw/07`). |
| S7 | knowing is not replaying; request hash | `not-kept`, `missing-call` on replay; `request_hash` only when content is whole. |
| S8 | optional inputs contradict the check | Optional with no default stays left out; a default that does not fit is refused. |
| S9 | `{}` two meanings; `interface-untyped` premature | `opaque`; refusal left to stages 2 and 3. |
| S10 | several outputs undefined | A return type is one output; several are declared (`programs/10`). |
| S11 | `done.value` leaks | As Astra B1.1. |
| S12 | describing a saved AI function exposes its prompt | AI nodes carry `interface` (description, words, `optional`); old folders fall back, with a warning in `saved.md` not to show that to outside callers. |
| S13 | amend the README's rule for widening | Not done: format 2 instead. |
| minor | `rated/13` description | Fixed. |
| minor | `omitted` presence | Present exactly when `content` is false (format 2); the schema says so. |
| minor | when `at` is stamped | When numbered, clamped. |
| minor | `tool_call` gets no size | Neither kind keeps a size now. |
| minor | the interface schema's file | Not moved (see *Disagreements*). |
| minor | `default` is in the id | Stated in `programs.md` and pinned (`programs/12`). |
| minor | extra returned keys dropped silently | Now refused (`programs/07`). |
| minor | prose wrapping, law 3, duplicate README row | Rewritten. |
| minor | the R sketch says the input twice | See *For R and Julia* below. |
| vignette 3 | turn id before the call | Stage 2 must mint the turn's call id when the turn is created. |

The first worker's seven questions, as settled here: (1) host override:
yes, as the rule itself (only removes); (2) a failing sink: best effort
or required, chosen by the host; (3) program records in the log:
`program.interface` now, the descriptor record in stage 2; (4) widening
`content: false`: no, format 2; (5) checking outputs on every call: yes;
(6) `interface` on AI nodes: yes; (7) `at` on every event: yes.

### Disagreements, both positions, and what I chose

**The call format.** Astra: bump to format 2; a format number is a
promise, and the base schema rejects 9 of the 81 new records. Opus: widen
`content: false` and amend the README's rule; old readers take the
conservative path, the classic must-ignore evolution. *Chosen: format 2.*
Opus's argument holds for one consumer (`rated`), not for every one: a
reader that validates against the published format-1 schema rejects the
records, and a reader that uses `content: false` to mean "safe to share,
no values" would share values. A format number is the only promise a
reader in another language, years later, can check. Maxime's standing
rule (existing data keeps its meaning) and "development cost is not a
constraint" both point the same way. *Cost: a format-1-only reader skips
format-2 records, so its sums over a tree miss them until it learns
format 2; every implementation must read both formats.* Format 2 was then
used to tidy what a bump makes cheap: `omitted` always present when
`content` is false, `program.interface`, `saw` required, `request_hash`.

**Numbering.** Opus: one sequence per call tree, every stream a
projection. Astra: either dense numbering per view, or source positions
with an explicit cursor. *Chosen: one log per tree, dense in the whole
log, source positions (`seq`) kept in every form, and `after` on every
event.* This is Opus's structure and Astra's second option, made
checkable. Per tree gives each happening one name everywhere (a stream
opened inside a module and the tree's journal no longer hold two copies
of one event under two numbers; stage 4 can refer to an event), one
conflict domain per tree, and the event-sourcing shape (one stream per
aggregate, projections over it). Source positions are what EventStoreDB's
filtered subscriptions and Kafka offsets use: a view's positions do not
change when a view's rules change. Dense per-view numbering was rejected:
it needs a map from view position to log position in every store, and
every view's numbers shift when its policy changes. Plain source
positions without `after` lose loss detection in every form but the
whole one; `after` restores it (a reader checks that each event comes
after the last it has) and is exactly the expected position a
conditional append needs. It also let the kept form drop events (pieces of
a dropped field) without inventing placeholders, which fixed the size
leak. *Costs: one integer more per event; a form of a log is no longer a
per-event function (its `after` depends on what it left out before);
a journal cannot keep only part of a tree (a journal set inside a tree
warns or refuses); a stream opened on an inner call starts at a `seq`
above 1.*

**Retention, disclosure, memory.** Astra: separate retention (what is
kept) from disclosure (who may receive a live value); a process boundary
is not a retention boundary. Opus: scope the stored form to observability;
a conversation store keeps what the conversation needs. *Chosen: three
dimensions, each with its place.* Retention for watching is
`log_content` (the call log, the kept form, observers, journals).
Disclosure is a view (stage 3), applied to whatever a reader is sent,
live or kept; the whole form leaves the process only through the stream
its caller watches. Memory is a conversation store's own setting (stage
2); when it and `log_content` cannot both hold (a transcript the host
never keeps, in a conversation that must remember it), stage 2 refuses
rather than silently forgetting (Maxime's answer 8). *Cost: observers get
the kept form; a host that wants whole events live in another process
forwards them from the stream it watches.*

**Sinks and journals.** Both reviewers: "every failure warns" with
"unfinished for ever" blocks vignette 4. Opus proposed a knob
(`on_error="warn" | "fail"`); Astra a separate required, acknowledged
path. *Chosen: two receivers.* A knob on one kind of sink would leave
open what "fail" means (fail on which event? before or after the tool?);
a required journal says when the call waits (barriers) and what the
writer keeps. Stage 2's stores are journals.

**Optional inputs with no default.** Opus: refuse at definition, or skip
the fit check for the filled-in default. Astra: either preserve omission
until binding, or require a valid portable default; keep host-native
defaults. *Chosen: preserve omission.* Refusing would forbid ordinary
Python (`def f(since: date = TODAY)`, a sentinel default) and TypeScript
optional parameters; filling in null contradicts the shape. Left out
means the program's own default applies and the record has no value, so
asking again from the record leaves it out again: faithful. *Cost: the
record of such a call does not say what value the program used.*

**Per-piece sizes.** Opus: drop, or one size per field per request. Astra:
at most one per field per request. *Chosen: none.* The call record's
`sizes` gives the total; a size per request would need an event with no
other purpose. *Cost: a watcher of a kept log cannot show a dropped
field's progress.*

**Misspelt names in host layers.** Opus: warn once per name that matches
no field in a block. *Not done:* a block around many programs names
fields most of them lack, so the warning is noise, or needs end-of-block
bookkeeping that still says nothing when the block's one call is the
wrong one. *Chosen instead:* `"*": false`, a list of what may be kept,
which a misspelling can only narrow.

**The interface schema's own file.** Opus: move it to
`interface.schema.json`. *Not done:* Python's saved-manifest test loads
`saved.schema.json` without a registry; a `$ref` to another file would
turn it red for a reason that is not the contract's. It stays in
`saved.schema.json` `$defs/interface` (make.py checks against it there).
Worth moving with the Python implementation of stage 1.

**Returned keys that are not outputs.** The first draft dropped them
silently; Opus asked to say why. *Changed: refused*, like an input the
interface lacks. A key that is not an output is a mistake (a misspelt
output name), and several outputs are always declared, so this never
costs a light prototype anything.

### Trade-offs, in one list

- Format 2: a format-1-only reader skips new records (their sums miss
  them) until it learns format 2; every implementation reads both.
- `true` in `log_content` never keeps what another layer drops; a
  program cannot insist on being logged. The environment's `0` drops
  everything, for every program.
- Any dropped field takes the reasoning, every request, reply, request
  hash and error message with it: such calls cannot be debugged from the
  log.
- Per-field retention does not follow values: a value copied into a
  kept field is kept (stated, not solved; labels are a question below).
- `sizes` still tell a dropped value's length.
- Observers receive the kept form only.
- A kept log of a dropped field shows no progress for it; a watcher sees
  requests and the end.
- A journal keeps whole trees only; one set inside a tree warns or, when
  required, refuses.
- A required journal can stop a call (`JournalError`), and adds a wait
  before the first request, before each tool and at the end.
- Views must show a call's requests and retries (without reasons): an
  outside caller learns that a request was retried.
- A module's outside view shows no reply while it is written (until stage
  3).
- One integer (`after`) more on every event; `request` events add one
  event per model request.
- The interface's signature includes defaults (lmcc parity): changing a
  default changes it.
- An optional input left out with no JSON default has no value in the
  record.
- A value with no JSON form never fits a field that is not opaque: a
  Python module annotated `Any` must say opaque (or be unannotated) to
  take a data frame.
- Returned keys that are not declared outputs are refused.
- A saved folder carries each AI node's interface twice (once in its
  signature's fields), checked to agree.
- `saw` in format 2 is required; format-1 records stay unreplayable.
- `request_hash` on exchanges only when content is whole.
- The interface schema stays inside `saved.schema.json`.
- This branch cannot merge alone: `rated/13`, `14`, `15` fail in every
  language and `saved/14` in TypeScript, R and Julia, until each
  implements format 2 and the interface check (AGENTS.md: every
  implementation in the same commit).

### What later stages must provide (and this contract does not block)

- **Stage 2 (stores, conversations).** An `append(tree, events)` that is
  atomic per batch and conditional on the first event's `after` (the
  rules table), acknowledged with a declared durability (memory, a
  process, fsync); `read(tree, after)` and a way to be told of new
  events (`watch`, or `subscribe`); a turn's call id minted when the turn
  is created, before the call; a turn-to-predecessor record
  (`started.parent` is the call tree's parent, not the conversation's
  previous turn); `request_id` deduplication; writer leases or tokens,
  and who may end a log another writer left unfinished; stopping a call
  from another process; the conversation store's own retention setting,
  and its refusal when `log_content` forbids what a conversation must
  remember; a program descriptor record (the interface a
  `program.interface` names), so a reader of the log can build a form.
- **Stage 3 (serving, views).** Named views and who chooses them; how a
  module says that a child's field is its output as it is written (or
  that its boundary is buffered); refusals for opaque fields at a
  boundary; a remote call's place in the caller's tree.
- **Stage 4 (tools, approval).** A tool invocation id unique within a
  call (lmcc's ids repeat across replies), carried by the child calls a
  tool makes; approval events; which tools need a required journal's
  barrier; "may have run" after a crash; the writer that resumes a
  waiting turn continues its log.
- **Stage 5 (`rated` with `earlier`).** Replaying from `saw`, using the
  `shown` rules and refusing with `missing-call` / `not-kept`.

### What Python and TypeScript must build now (replacing sections 1–4's lists)

- Write call records in format 2 (`omitted`, `program.interface`,
  `program.signature` for AI functions only, `saw`, `request_hash`);
  read formats 1 and 2 (`rated`: `interface`, rule 3's matching, skip
  unknown formats: `rated/13` to `15`).
- `log_content` maps with `"*"`, only-removes layering over every layer,
  fields FunctAI adds, and the record of `content/*`.
- Events: `tree`, `seq`, `after`, `at` (clamped), the `request` event;
  retry and request empty fields; `tool_call.id` always set; one
  numbering per tree shared by every stream opened in it; the kept form
  (`kept-*`); replay and follow over dicts (`replay-*`, `follow-*`).
- Observers and journals (settings; best effort and required; barriers;
  resend; stop on conflict); `store-*` with an in-memory journal.
- Interfaces: `opaque`, omission, several outputs declared, refusal of
  undeclared returned keys, `interface-malformed`; `interface` on every
  program object and on every saved node, checked at load
  (`programs/*`, `saved/13`, `14`, `15`, `expect.describe`).
- `saw`: the reader (`read` cases) and, when conversations come, the
  `shown` rules.

**For R and Julia** (they shape the design now, implement later): R
reads format 2 in `rated` and writes it; `log_content = list("*" =
FALSE, question = TRUE)` spells a host's list (the name `*` needs quoting
in R, and in Julia a `Dict("*" => false, "question" => true)` or
`var"*"`; each may offer a plainer alias for that list). R's module sketch in section 2
said its input twice; a better one lets the formula carry names and the
function carry code, with shapes from a type map only when given:
`support <- ai_program(reply ~ message, function(message) ...)`, an
unannotated R argument being opaque until a type is given
(`ai_program(..., types = list(message = "text"))`). Julia's `@program`
derives opaque from an untyped argument (`Any`).

### Cases: what changed, and what fails today

131 cases (81 more than the 50 before stage 1: `content/` 16, `events/`
32, `programs/` 12, `saw/` 13, `rated/12` to `15`, `saved/12` to `15`).
Every case is written by the generators from the rules; `make.py` run
twice leaves no difference; every case passes its schema; `make.py` also
checks that the schemas refuse 14 leaking or contradictory objects.

Cases whose meaning changed on purpose: every `content/`, `events/`,
`programs/` and `saw/` case (none had a harness); `rated/12` and `13`
(their records are format 2; 13's description corrected); `saved/01` to
`11` (their AI nodes now carry `interface`, and `05`'s describe reads the
signature as an old folder's must). `rated/01` to `11`, `functions/` and
`scores/` are byte for byte what they were at 8cd4597.

Observed (2026-09-28, this branch):

| language | result | failing, all new cases |
|---|---|---|
| Python | 320 passed, 3 failed | `rated/13`, `14`, `15` |
| TypeScript (`npm test`; `npm run check` passes) | 100 passed, 4 failed | `rated/13`, `14`, `15`, `saved/14` |
| R (`r/check`, with `LD_LIBRARY_PATH` pointing at nixpkgs' curl for `libcurl.so.4`) | 400 expectations, 6 failed, 1 error | `rated/13`, `14`, `15`, `saved/14` |
| Julia (`julia/check`) | 392 passed, 6 failed | `rated/13`, `14`, `15`, `saved/14` |

`tools/crosslang.py` and `./check` as a whole were not run (the latter
stops at the first red step, which is expected here).

### Questions for Maxime

1. **Labels instead of names** (Opus S1). Should a program be able to
   mark a field's kind of data (`Annotated[str, functai.private]`,
   `t.string().private()`), outside the signature, so that one host rule
   ("never keep private fields") covers every program, and stage 3's
   views read the same mark? Map keys that are not names are refused now,
   so such keys can be added later without being silently ignored.
2. **Conversation memory against `log_content`.** When a host never keeps
   the transcript and a conversation must remember it, should stage 2
   refuse to open a persistent conversation (proposed), keep it in memory
   only, or let the conversation store's own setting win?
3. **Required journals' barriers.** Before every tool (proposed for now),
   or only before tools that declare `effects="changes"` once stage 4
   declares effects?
4. **Coalescing pieces.** May a writer merge adjacent pieces of one field
   before numbering them (fewer events, a little more latency)? Law 2
   allows it; nothing asks for it yet.
5. **Python's `Any`**: opaque (it means any object, proposed) or any JSON?

---

## After the second review (2026-09-28)

Two re-reviews of 52e701d, both "accept with fixes": Opus (delegation
1c295283, `probes-critic.py`) and Codex Astra (delegation 8f89531a,
`probes-independent.py`). The parent reproduced every failure they
reported. Both endorsed the architecture:

- format 2;
- one log per tree;
- `request` events;
- observers versus journals;
- `log_content` that only removes;
- `opaque`;
- lineage versus replay.

That architecture is kept. This section maps every finding to what was
done, says where I disagreed and why, and lists the costs and the
questions. The branch is `stage1-contract-v3`. Sections 1 to 4 above
describe the result.

### The blockers

**Opus B1 / Astra B1: event identity across a writer hand-over.** Both
showed a later writer "numbering on from the last kept event" reusing
`(tree, seq)` identities a live reader holds, with the reader dropping
the new events as duplicates and never seeing `done`. Opus also showed a
reader of a form the store cannot rebuild looping on "loss" for ever.
Done:

- `writer` on every event, so identity is `(tree, writer, seq)`.
- Writer numbers are *issued by the store* through a claim, which fences
  every earlier writer at once. See section 1 for why this beats both
  proposals as literally stated: the writer-chosen term failed its own
  race case in the generator.
- A follower **rewinds** when a later writer's `after` is below its
  last event, and drops **stale** events of earlier writers.
- *Where each form can be read*: a store gives the kept form and views
  of it; a read after an event it lacks refuses `event-unknown`, and the
  reader starts again.
- Where the next request number comes from is now in `streaming.md`.
- Cases:
  - `follow-03`: Opus's hand-over;
  - `follow-04`: Astra's unacknowledged tail, and a stale event;
  - `store-07`: the livelock, now `event-unknown`;
  - `store-08`: claim, fencing, and a late event that fits the chain,
    refused;
  - `store-10`: two claims.

**Astra B2: the terminal outcome when an acknowledgement is lost.** Done:
the required-journal state machine in `streaming.md`.

- The outcome is decided first, and never changed by the journal.
- The confirmation is confirmed, refused, or unknown.
- A barrier not passed makes the outcome `JournalError`
  `journal-barrier`.
- An end not confirmed raises `JournalError` `journal-end`, holding the
  outcome.
- The record keeps the outcome and says `journal`: `"refused"` or
  `"unknown"`.
- The terminal event is shown live only once kept.

A deterministic fault script pins it: `journal-01` to `10`, each giving
the caller's result, the record, the store's log, what readers were
shown, and every append attempt. Among them are the three cases Astra
asked for:

- fail before commit, refused (`02`) and with no answer (`04`);
- commit, then lose the acknowledgement (`03`, and `05` where a resend
  answers duplicate);
- a lost final acknowledgement on an already failed call (`06`).

### Should-fix, Opus

| id | finding | done |
|---|---|---|
| S1 | event `kind` closed in the schema while prose says readers skip; decide the extension policy once | Kinds open in the schema (any identifier); one table in `contract/README.md`. Readers skip; form makers leave out unknown kinds and keys (`kept-09`); `saw` readers refuse; interfaces are closed and refuse (`programs/11`); unknown formats stop a reader (`follow-07`); unknown kinds take their place (`replay-14`, `follow-06`). |
| S2 | the tool barrier promises inputs the kept form drops; the journal is not the checkpoint; stage 2's store called a journal | `streaming.md` now says a journal is the watching log, and that conversation memory (stage 2) and the resume checkpoint (stage 4) are separate records under their own retention. The tool barrier is kept, but described as what it is: the fact of the request, so "may have run" can be said. The stage-4 checkpoint (lmcc steps, provider items, tool outputs as parts, invocation ids) is in *What later stages must provide*. "Stage 2's stores are journals" is gone. |
| S3 | opaque logging prose against cases | `programs.md`: `opaque` governs checking and boundaries; how a value is written depends on the value (`calls.md` *Values*, `programs/08`'s description). |
| S4 | defaults in the data signature split ratings | Defaults are left out of the interface's signature and out of lmcc's signature (`programs/12` with a date default, `functions/12`). A module's defaults stay in its version through its interface. |
| S5 | added-field rule: prose against generator | The prose's rule (any field dropped drops every added field), in the generator too. Tools cases: `content/17`, `18`. `tools` is not a field (`content/19`). |
| S6 | "fits" not portable | A listed vocabulary with each keyword's reading. Anything else refuses. The generator checks by those rules, and asserts that a 2020-12 validator agrees (`programs/14`, `11`). |
| S7 | record → turn unspecified | The turn's signature, inputs and outputs are fixed. The steps operation is deferred to stage 5 with Opus's list, and readers show no steps until then. The result is renamed `keeps`, since it proves retention only. |
| S8 | "first unknown" depends on map order | Code-point order; unknown names before anything else; outputs: unknown keys, then each output in order (`programs/15`). |
| S9 | design/08 sections 1–4 still describe the old design | Rewritten (above). |
| minor | "never changes one" too strong | "Never alters a value it shows, never invents one". Re-attribution is left to stage 3. |
| minor | follower state per tree | Stated; `follow-08`. |
| minor | an inner stream's first `after` is 0 | Law 7 says so; `follow-05`. |
| minor | store table: malformed rows; unreachable "seq less than it" | Schema-invalid and wrong-tree appends are `event-malformed`; the unreachable clause is removed. **Not done**: refusing a first `started` with a non-null `parent`. A tree's outermost call may have a parent in another process (stage 3's remote call; the call log already allows dangling parents). |
| minor | "take and give the same data" overclaims | "Record the same data". |
| minor | `request` events and exchanges | Law 8: the *n*th request is the *n*th exchange. |
| minor | context that changes between requests | `{"context": "changed"}` as a last entry (`saw/14`). |
| minor | refusal codes registry | `contract/README.md`, *Refusal codes FunctAI defines*. |
| minor | `**kwargs` mapping | `programs.md`. |

### Should-fix and minor, Astra

| id | finding | done |
|---|---|---|
| S1 | added-field retention: prose against generator | As Opus S5. |
| S2 | "replay OK" stronger than the evidence; `without` + `steps` can make an invalid turn | `keeps` checks: with `steps`, the replies and request hashes; values that are descriptions refuse. An entry leaving out the tool calls with steps refuses `turn-invalid`, lmcc's word, and I reproduced lmcc's refusal with the real kernel (`saw/15`, `16`, probe A7). The record → turn step is deferred as in Opus S7. **Not done**: a full "record → normalized turn → rendered request" case. That operation is the one deferred: writing a case for it now would fix FunctAI's retry transcript in a form lmcc turns cannot hold. |
| S3 | opaque and JSON share a data signature, so descriptions pool as data | **Partly, and differently.** `described` on records; `rated` leaves described inputs out and gives nothing for a described answer (`rated/16`); replay refuses them (`saw/15`). The signature stays blind to `opaque`, by decision (section 2): an opaque field that received JSON recorded real data. Descriptors in stage 2 must be keyed by the full interface or the version. |
| S4 | optional metadata preserved in isolation, not end to end | A definition's inputs carry `optional`, with the default in the shape. An AI function's optional input must have a default. Loading takes both from the interface. `saved/16` pins save → describe → load → call with the input left out, and with it given (`sends`). `programs/13` pins the binding. The no-default case is a module's only (`programs/06`); explicit null is checked like any value. |
| M1 | structural against semantic refusal precedence | Form and meaning are checked together, field by field, naming the first field at fault, in both the rules and the generator (`programs/11`, including a form fault after a meaning fault). For a saved folder, the manifest's schema comes first (`saved-malformed`), in both loading and describing. **Not added**: a saved case whose manifest fails the schema. Python's harness asserts that every saved manifest passes the schema, so it would turn red for a harness assumption rather than a contract fault; the order is in `saved.md` and the generator's `describe`. |
| M2 | cursor advancement over unknown kinds; unknown formats | A reader takes the place of an unknown kind; an unknown format stops it (`follow-06`, `07`). |
| M3 | keep the interface schema independent | Done: `schema/interface.schema.json`. Python's saved-manifest fixture now uses a registry: the only change to a language folder. |

### Where I disagreed, or chose between the reviewers

- **Writer terms: issued by the store, not chosen by the writer.** The
  literal form of Opus's fix (the last term + 1) cannot fence two
  concurrent continuers. My first cut failed `store-10` exactly so. The
  claim keeps Opus's structure (terms, fencing, rewind) with Kafka's
  and Raft's issuance.
- **Durable allocation (Astra's first option): rejected** as the
  primary mechanism. It makes live delivery wait on a journal, and still
  needs the rewind rule.
- **The tool barrier: kept** (Astra Q3), against Opus's "move it to
  stage 4". Its meaning is narrowed to the fact of the request (audit,
  "may have run"). The effect-safety barrier belongs to stage 4's
  checkpoint, as Opus said.
- **Added fields: the conservative closure** (Astra), not Opus's
  "plain fields only".
- **`opaque` stays out of the signature** (against the letter of Astra
  S3). Eligibility, not identity, keeps descriptions out, and the stage-2
  descriptor must not be addressed by the data signature.
- **A non-null `parent` on a log's first event is allowed** (against
  Opus's minor): remote calls in stage 3.
- **No full record → turn → render case yet** (against Astra S2's last
  request): the operation is deferred, not half-specified.

### Cases that changed meaning on purpose

- **Every `events/` case**: `writer` on events; follow cases now give
  `results` and `state`; store cases give `steps` and `reads`.
- **`programs/01` to `04`**: the new `binds` key.
- **`programs/06`, `07`, `08`, `11`, `12`**: new orders, the vocabulary,
  defaults out of the signature.
- **`saw/01` to `11`**: `replay` renamed `keeps`, with stricter checks.

`rated/01` to `15`, `saved/01` to `15`, `functions/01` to `11`, `scores/`
and `content/01` to `16` are byte for byte what they were at 52e701d.

New cases (31):

- `content/17`–`19`;
- `events/follow-03`–`08`, `journal-01`–`10`, `kept-09`, `replay-14`,
  `store-10`;
- `functions/12`;
- `programs/13`–`15`;
- `rated/16`;
- `saved/16`;
- `saw/14`–`16`.

### Trade-offs taken in this repair

All are in the list after section 4. These are new:

- Identity is `(tree, writer, seq)`: a tool keyed by `(tree, seq)` alone
  is wrong after a hand-over.
- A claim fences a running writer: the lease (stage 2) must decide who
  may claim.
- Resuming a live-only form from a store starts again in the kept form.
- A required journal's end is shown one round trip later. A caller must
  handle `JournalError` `journal-end` holding a good outcome, and the
  record says `journal` beside a successful outcome.
- Two module versions that differ only by a default pool their ratings.
- Pydantic `pattern`, `oneOf` and `multipleOf` refuse a module
  interface.
- Interfaces are closed: an older reader refuses a newer key.
- An AI function's optional input needs a JSON default (TypeScript's
  becomes `null` in its interface).
- Dropping `calls` drops the reasoning, and the other way round.
- Showing calls with steps waits for stage 5.

### What later stages must provide (additions)

- **Stage 2.**
  - Leases on claims: who may claim, and when a writer counts as stopped.
  - Durability declared per store.
  - A descriptor record keyed by the program's version, or by a hash of
    the whole interface, never by `program.interface` alone.
  - Conversation memory as its own record, under its own retention.
- **Stage 3.**
  - Views that forward a child's field as a module's answer, or
    re-attribute a nested event to the boundary.
  - `media` in the vocabulary.
- **Stage 4.**
  - The resume checkpoint: lmcc's current-turn steps with provider
    thinking and signature items, tool outputs as parts, tool invocation
    ids unique within a call and carried by child calls, what each tool
    was given, and "may have run".
  - Its own barrier before effectful tools.
- **Stage 5.** The record → turn operation with steps, checked against
  `request_hash`.

### Questions for Maxime

1. **Who may claim a log** (stage 2's lease): the first claimant until
   its lease expires (proposed), or the latest, as the mechanism now
   allows?
2. **Required journals and live delivery.** The terminal event waits for
   the journal before any reader sees it (proposed, so no reader sees an
   end a store lacks). Should a host be able to show it at once and
   accept that a reader may see an end that is later not kept?
3. **The added-fields closure.** Should a host that drops only the tool
   calls keep the reasoning (Opus's reading), or lose it (chosen, the
   conservative one)?
4. **`opaque` in the data signature**: out (chosen; eligibility keeps
   descriptions out), or in (Astra's reading; records of `{}` and opaque
   fields stop pooling)?
5. **Closed interfaces.** Refuse unknown keys (chosen), or allow a
   marked class of ignorable keys (`x_…`) for display-only extensions?
6. The first repair's five questions (labels, memory against
   `log_content`, barriers, coalescing, Python's `Any`). Both
   re-reviewers answered 4 and 5 alike: coalescing before numbering;
   `Any` opaque. `Any` is now opaque in `programs.md`. Say if you
   disagree.
