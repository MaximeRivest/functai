# 09 — Stage 1.1: decisions, and the rating fixes

*Status, 2026-09-30: decided by Maxime in conversation (answers below),
written into the contract on the branch `stage1.1`, with cases that fail
before the implementations change. Builds on
`08-stage1-foundations.md`, whose open questions this closes (those that
belong to stage 1), and on an outside review of the call log's ratings
(fixes F1, F2, F5 below).*

Stage 1 left 37 questions open (11 found while the four languages were
built, 15 in `08`, 11 in `07`). Seven were settled by releases or by
Maxime's earlier answers, one was asked twice, twelve belong to later
stages (listed at the end, with a leaning), and seventeen are decided
here. One review of the rating code found three faults worth fixing now.
They go together because two of them change what a record says about a
program (its version, where it lives): records change once, not twice.

## What changes, in one table

| # | Decision | Contract |
|---|---|---|
| A6 | A call's inputs are **bound**: converted to their field's type when the meaning is clear, refused otherwise. An input with no declared type is text. | `programs.md` *Binding a call's inputs* (new); `functions.md` |
| A7 | Error messages quote the value at fault, cut short, except a value the log is told to drop. | `programs.md` |
| B14 | Defaults count in a program's **version** by their logic (what is written), not by the value they give today. Every `default` is out of the **signatures** that pool records. | `calls.md` *Versions*; `programs.md` *signature*; `saved.md` |
| A3 | Records are **closed**: a record shape with no `additionalProperties` holds only the members it names. | `programs.md`, `functions.md` |
| A4 | Each exchange's `request_hash` is the hash of the request that exchange sent (a re-ask's included). | `calls.md` |
| A5 | With tools, `outputs.calls` holds every tool call the model asked for in the call, in order, across steps. | `calls.md`, `functions.md` |
| A8 | A host may refuse a program's own observers; a watcher never triggers itself, whatever kind of watcher; a scheduler that cannot preempt keeps "a slow watcher never slows a call" only for watchers that yield, and says so. | `streaming.md` |
| A9 | A program's code cannot write a dropped value through `returned`: it is kept only when nothing of the call is dropped. | `calls.md` *Content* |
| A10 | The sample value of a shape whose `type` is a list is the first non-null type's. | `calls.md` *Versions* |
| A11 | Julia: a call waits for the calls it started; a default is bound as any input is (A6); a `missing` default is sent as `null`. | Julia only (A6 covers the second) |
| B4 | A writer may join adjacent pieces of one field before numbering them. | `streaming.md` |
| B8 | Dropping one added field (`reasoning`, `calls`) drops only it; dropping an input or output still drops both. | `calls.md` *Content* |
| F1 | Code defined at a language's top level (a notebook, a script) is known by its file too: `rated` does not pool two notebooks' `summarize`. | `calls.md` *rated* |
| F2 | A rating made under a computer account and no named person is kept on its own: it neither replaces nor is replaced. | `calls.md` *A rating record*, *rated*; `schema/rating.schema.json` |
| F5 | A record names the lmcc and lm15 that made it. | `calls.md`; `schema/call.schema.json` |

Taken as they stand (no contract change): B5 (Python's `Any` is
opaque), B7 (a required journal's end is shown once kept), B9 (`opaque`
stays out of the data signature), B11 (one journal per tree), B12 (a
fenced writer's resend is refused), C10 (calls inside calls, for stages
2 and 3: helpers remember inputs and outputs only; outsiders see the
boundary; a tool's AI function keeps its own model unless the caller's
block sets one; "only the first layer in context" stays a toggle).

No format number changes. Every new key is optional, and every reader
already refuses what it does not know inside a closed object, so the
schemas widen: `rating.by` becomes optional and `rating.account`,
`process.lmcc`, `process.lm15`, a saved node's `defaults` are new. What
changes meaning is computed values: versions (B14) and signatures (B14,
only for shapes with a `default` inside).

---

## A6. Binding a call's inputs

**Maxime:** check declared types; an input with no type is text;
convert whatever has a clear text form; lenient, not verbose, for data
scientists.

**The rule** (`programs.md`, *Binding a call's inputs*). Every
program's call binds each given input to its field before anything
else, then checks the bound value (*Checking values*). Binding converts
toward the shape's `type` when the meaning is clear:

| the field wants | converted | refused |
|---|---|---|
| text | a number (its canonical JSON: `42`, `2.5`), a boolean (`true`, `false`), a list or record (JSON, indented by two spaces, as worked examples write it); a value with no JSON form whose type defines its own text (a data frame, a date) | a value whose only text is the language's default (`<object at 0x…>`) |
| an integer | a number with no fraction (`5.0`), text that reads as one (`"5"`, `" 5 "`, `"5.0"`) | `2.5`, `"abc"`, `true` |
| a number | text that reads as one (`"2.5"`) | `"abc"`, `true`, not-a-number, infinities |
| a boolean | nothing: `true` and `false` only | `"yes"`, `1` |
| a list, a record | each item and member, by its own shape; a record keeps only the members it names (A3) | an item or member that is refused |
| anything (`{}`, opaque) | nothing | nothing |

- **Missing values.** The language's missing value (Python `None` and
  `NaN`, R `NA` and `NULL`, Julia `missing` and `nothing`, TypeScript
  `null` and `undefined`) is `null`. `null` given to an optional input
  that `null` does not fit is that input left out: it takes its default.
  This is what a data scientist mapping a table with gaps expects.
  `null` given to a required input that `null` does not fit is refused.
- **Choices** (`enum`, `const`) bind by their `type` first when the shape
  has one (`Literal[1, 2]` given `"2"` is `2`), then must match exactly.
- **`anyOf`**: `null` stays `null`; otherwise the first option the value
  binds to and fits.
- **Where.** AI functions and modules alike, before the request or the
  code. The call's record holds the bound values, so a dataset made from
  ratings holds `5`, never `5` in one row and `"5"` in the next.
- **Defaults** bind the same way when the program is defined; a default
  that does not bind refuses `interface-malformed` (Julia's `1.0` for an
  integer becomes `1`: A11).
- **Outputs are not converted.** What a module's code returns and what a
  model replies are checked as they are. A program's outputs are its
  promise; converting them would hide a bug in its own code, and a
  model's reply that does not fit is already asked again (`parse-value`).
- **Refusal:** `InterfaceError`, code `interface-input`, naming the
  input, before any request or any of the module's code. AI functions'
  refusals are recorded as modules' are.

**Alternatives an expert would weigh.** *Strict everywhere* (no
conversion: R's and TypeScript's behaviour today) is simpler to state
and catches more mistakes, but makes every table with a numeric column
fail on a text input, and every CSV-read integer column fail on an
integer input: the most common data-science paths. *Python's old
behaviour* (anything to text, with `str()`) sends `<object at 0x…>` and
`True` to models and records mixed types. The table above is pydantic's
"lax" mode, narrowed: no `"yes"` for a boolean (its meaning depends on
the language), no truncating `2.5` to `2`.

*Costs:* a whole table passed where one sentence was meant is sent as
text, quietly (it has a text form). A record input given a row with
more columns drops the extra columns, quietly (A3).

## A7. Error messages quote the value

**Maxime:** yes.

An `interface-input` or `interface-output` message names the program,
the field and what it expected, and quotes the value: its canonical JSON
(or its description, for a value with no JSON form) cut to 80 code
points, then `…`. When any layer's `log_content` drops that field, the
message names the field and what it expected, never the value.

*Why the exception:* error messages travel further than the log
(tracebacks, error trackers, a served program's reply). The log already
keeps no message when a field is dropped, but the exception itself is
raised to the caller. The exception costs nothing to a user who never
drops a field.

*Cost:* a value of a kept field reaches whatever records the program's
exceptions.

## B14. Defaults, by their logic

**Maxime:** defaults count in the fingerprint by their logic, not their
value: `today()` is the same version every day; changing `"kind"` to
`"formal"`, or `today()` to `yesterday()`, is a new version. Approved:
in the version; not in the signatures that pool ratings.

**Versions** (`calls.md`). An AI function's and a module's version
document gain `"defaults": {input: D}`, present when an input has a
default, where `D` is:

- `{"code": "<text>"}` when the default is written as an expression that
  is neither a constant nor a name (`today()`, `3 * 60`, `DEFAULT.lower()`),
  `<text>` being the expression as the language writes it, normalized by
  it (Python: `ast.unparse`; R: `deparse`; Julia: `string` of the
  expression; TypeScript: a default given as a function, its source
  text);
- `{"value": <JSON>}` otherwise (a constant, or a name such as
  `DEFAULT_TONE`: a name counts by the value it holds, so editing
  `DEFAULT_TONE` elsewhere is a new version).

When the source cannot be read (a function typed at a bare Python
prompt), the default counts by its value, and the language warns once.

**Signatures** (`programs.md`, `calls.md`). `program.interface` and
`program.signature` leave out every `default` keyword at any depth of a
shape (a member's default inside a record, pydantic's), not only the
field's own. A record type whose field defaults to today's date no
longer splits a function's ratings by day. Only keywords are removed: a
member *named* `default` stays.

**Saved folders** (`saved.md`). A node gains `defaults`, `{input:
{"code": text}}` for its inputs whose default counts by code, so a
loaded program has the version of the one that was saved (its values are
in the interface).

**Limits, stated in the contract:** a change inside what a default calls
(rewriting `today`) is not seen; the same computed default written in
two languages (`today()`, `Sys.Date()`) gives two versions, as code of
the function's own already does; a default inside a record type counts
by the value the request carries (the record type is defined outside the
function).

*Cost:* every function and module with a default gets a new version
once. Functions without defaults keep theirs; signatures change only for
shapes with a `default` inside.

## A3. Records are closed

A shape with `properties` and no `additionalProperties` is a record: it
holds the members it names and no other (`additionalProperties: false`
is implied). A shape with `additionalProperties` (a map, or an explicit
`true`) is open as it says. lmcc already sends records closed to
providers (D-57); the contract now reads them the same way.

- A model's reply with a member its record does not name does not fit:
  it is unreadable (`parse-value`) and asked again.
- A module's returned record with such a member is refused
  (`interface-output`).
- An input: binding keeps the named members only (A6).

*Why:* R returns a record as a tibble only when it is closed; Python
dataclasses and pydantic models drop extras; a record that could hold
anything is a map. *Cost:* a program that relied on extra members
passing through a record must declare a map.

## A4. The request hash of a re-ask

Each exchange's `request_hash` is `"sha256:"` + SHA-256 of the canonical
JSON of the lmcc request that exchange sent, whatever made it: a render,
or a re-ask (the request it follows, then the reply's message and the
correction). A resend with a larger budget sends the same lmcc request
(the budget is lm15's), so it has the first one's hash. It exists to show a later reader rebuilt
the request it names; a hash of another request would show nothing.

## A5. `outputs.calls`

With tools, the output `calls` is every tool call the model asked for in
the call, in the order asked, across every step (each as lmcc's
`ToolCall`), not the last step's list (which is empty when the model
answered). The contract's own example already showed this; three
languages recorded `[]`.

## A8. Watchers

- **A host may refuse a program's own observers.** A layer's
  `program_observers: false` (Python `configure(program_observers=False)`,
  TypeScript `programObservers: false`, R `program_observers = FALSE`,
  Julia `program_observers = false`) means observers set in a program's
  own settings are given no event of calls under it; the host's own
  observers still are. As `log_content`, it only removes: a program
  cannot turn it back on. Where events go is policy, and policy is the
  host's (Maxime's answer 10, `07`).
- **A watcher never triggers itself**: calls made while delivering an
  event to a watcher, or by code a watcher runs on the events it is
  given, are not given to that watcher. For a watcher that is a queue
  (Julia's `Channel`), FunctAI cannot see who reads it; the rule then
  covers calls made while an event is being put, and the language says
  so.
- **"A slow watcher never slows a call"** holds in every language for
  watchers that return promptly or yield. Julia's tasks switch only when
  they yield: a watcher that computes for a long time without yielding
  holds a thread. Julia documents it; the contract names it.

## A9. `returned` and dropped fields

`returned` (what an AI function's code returned, when it changed the
answer) is kept only when no field of the call is dropped. Code can put
any value into what it returns (`return critique, _ai`); keeping
`returned` whenever the answer is kept wrote a dropped `critique`
through it.

## A10. The sample of a type list

`{"type": ["string", "null"]}`: the sample value is the first non-null
type's (`"example text"`). R, TypeScript and Julia already did this;
Python crashed.

## B4. Joining pieces

A writer may join adjacent pieces of one field (the same call, field
and kind, nothing between them) into one event before numbering it.
Readers see fewer, longer pieces with the same text. Both second-round
reviewers of `08` agreed.

## B8. Added fields, one at a time

**Maxime:** no, dropping the tool calls should not drop the reasoning.

Dropping `reasoning` drops only it; dropping `calls` drops only it.
Dropping any input or output of the program drops both: both routinely
repeat inputs word for word.

*Cost:* a reasoning can mention what it is about to do with a tool ("I
will look up order 1042"): a host that drops `calls` to keep order
numbers out must drop `reasoning` too, and now has to say so.

## F1. Notebooks are known by their file

**The fault** (reproduced by the review): `rated` found a program's
calls by `program.name`, `program.module` and its signature. Every
notebook and script is `__main__` in Python, and the default log folder
is shared by every project, so `summarize(text) -> str` in two
notebooks pooled their ratings.

**The rule.** When a program is defined at its language's top level
(Python `__main__`, R's global environment, Julia `Main`, TypeScript a
file run directly), its `program.file` is the notebook's or script's
path: the notebook itself, never a temporary file a kernel runs cells
from. `rated` is given the program's `file` when it is defined there,
and takes only calls whose `program.file` is equal; a call with no
`program.file` does not match. The reader may be told to pool across
files (Python `functai.rated(fn, any_file=True)`), for a notebook that
was moved or renamed.

*Rejected:* warning when a dataset mixes prompts (the review's other
half). Improving a function changes its prompt on purpose, and its
ratings must still count.

*Cost:* moving a notebook starts its ratings afresh unless the reader
pools across files. A notebook whose path the kernel does not give (a
plain Jupyter kernel with no file name) records no file, and its
ratings pool with other file-less code of the same name.

## F2. Ratings under an account

**The fault** (reproduced): a rating's `by` defaulted to the operating
system's account; on a shared account (the family on `lambda`) one
person's `wrong` silently replaced another's `right`, and `disputed`
stayed false.

**The rule.** `by` names a person, and is present only when one was
named (the rating's `by`, or the caller's `user`). A rating made with no
person named has no `by` and has `account`, the operating system's
account. `rated`: each person's latest rating counts, as before; a
rating with no `by` counts on its own: it replaces none and none
replaces it, and its `null` verdict withdraws nothing. Two such ratings
that disagree make the row `disputed`. Chattering always names the
person.

*Cost:* one person who rates the same call twice without naming
themselves shows as a disagreement with themselves. A flag is cheaper
than a lost rating.

## F5. The lmcc and lm15 that made a record

`process.lmcc` and `process.lm15`: the versions of the libraries that
built the request and sent it, when the language can tell. A version is
a fingerprint of what lmcc renders; when an lmcc update changes every
version, the records say why.

---

## Not done here, on purpose

- **F3, ratings outliving a cleanup.** Keeping a log small is deleting
  whole day folders, and a rated call's inputs live in its day. The fix
  (keep rated calls when a day goes) changes the folder's layout, which
  Chattering reads too: its own note, before stage 5.
- **Reading speed** (every read scans the folder) and **calls lost to a
  killed process** (journals partly cover them): later.

## For later stages (the leaning, to confirm when the stage starts)

| # | Question | Stage | Leaning |
|---|---|---|---|
| B2 | a host that never keeps transcripts vs a conversation that must remember | 2 | refuse a stored conversation |
| B6 | who may take over an unfinished log | 2 | the first claimant until its lease expires |
| B15 | what a live page keeps when a writer restarts | 2 | as now |
| C6 | R conversations change in place | 2 | yes, as ellmer's |
| C7 | adding reasoning or a tool mid-conversation | 2 | a rule in lmcc |
| C8 | after a merge, what the next turn sees | 2 | the merge recorded as the program's own turn |
| C11 | a remembering program called inside another | 2 | refused unless declared |
| C5 | summarizing old turns | 2 | a vignette to write |
| B1 | marking a field "private" | 3 | yes, views need the same mark |
| B10b | a format number on an interface served alone | 3 | yes |
| B3 | journal barriers before every tool, or only tools that change things | 4 | only those |
| C4 | rated conversation turns as worked examples | 5 | skip for now |
