# 19 — Notes a program declares, and the answer so far

*Status, 2026-11-04: built in Python and TypeScript on the branch
`setup-and-events`, with the contract (programs.md *Notes*, streaming.md
*Notes*, *Views*, *Replaying*, *The answer so far*; tools.md *Notes a tool
makes*; the interface schema) and its cases (`programs/27` to `30`,
`views/outside-04`, `events/replay-15`, `partial/`); in R and Julia on
2026-11-04 (branch `rj-catch-up`).*

## Why

A company's backend (Bigspin's Quiver, 2026-11-04, Chattering 01a1a681)
wants its pages to show a pipeline while it runs: the answer as it is
written, the data each step gives, and what plain code says between them
(the stage, each page captured). FunctAI's events already carry most of it
(design/17, streaming.md). The agent building Quiver found four gaps; each
was checked against the code:

| gap | true? | decision |
|---|---|---|
| a note has no type: a renamed note leaves a page blank, silently | yes | **notes are declared** in the interface, typed, checked where made |
| a module's caller sees only its start and end (the `outside` view) | yes, on purpose: helpers' prompts, tools' inputs and pages' text stay inside | **notes declared `outside`** are shown; nothing else changes |
| TypeScript has no answer so far as a value | yes, and the contract asked for it | **built**, typed (`SoFar<A>`), with cases every language passes |
| Quiver keeps no event logs | yes | Quiver's own: it plugs a store into FunctAI's |

It also missed two things FunctAI had: `Follower` (a reader that follows a
log live, resumes after a dropped connection, and empties a field's text
when a request starts again) and `answer_from` (a module's answer, as one
helper writes it, shown to its caller). Quiver uses both instead of its own.

## What was built

**Declared notes** (programs.md, *Notes*). An interface may have `notes`:
`{name, shape, desc?, latest?, outside?}`. A module declares them; an AI
function's are its tools' (two tools that declare one name differently
refuse the function). When code makes a note inside a call:

1. the innermost program refuses a name it does not declare, when it
   declares notes (a program that declares none refuses none: as before);
2. every program of the calls around it that declares the name refuses
   data that does not fit, so whatever a reader is promised has its shape.

The refusal is `InterfaceError` with code `interface-note`, raised where
the note is made, whether or not anything watches.

**Notes are not in the interface's signature.** A note is an event, never
a value of a call: declaring one must not split a program's records or
ratings (the reason Quiver's agent had not seen). They are in the module's
version (its interface is), and in the served `/interface`.

**The outside view** shows each note the outermost program declares
`outside`, made by any call of its tree, re-addressed to it, with its data
when the form has it. Whoever makes the view gives it those names (a
server, a host), as it gives `answer_from`: events do not carry an
interface.

**A reader's state** holds each call's notes, in order (present once a call
made one, so no replay case changed).

**The answer so far is read by its shape** (streaming.md). Before, a
string was shown as written, so a choice showed "med" on its way to
"medium": a value it can never take. Now a string whose shape is a choice
waits for its closing quote; a record starts with `{`, a list with `[`; a
text that cannot be the start of JSON gives nothing (Python's
`partial_json` raised). `cases/partial/` pin the reader: escapes, fences,
`$ref`, nullable records, tuples, maps. Python and TypeScript export it
(`answer_so_far` / `answerSoFar`) for a page that knows the program.

## Trade-offs, stated

- **R and Julia** declare notes as their own code does: Julia
  `@program notes = ["stage" => Note(OneOf(...); latest = true)]` (pairs,
  in order: a `Dict` has none, and is refused), R `ai_program(..., .notes =
  list(stage = note_type(choice(...), latest = TRUE)))`. R's streams are
  callbacks, so its answer so far is `answer_so_far(text, fn)`, read from
  the text a handler collected; Julia's is `answer_so_far(stream)`.
- **Undeclared notes still work** in a program that declares none: making
  every note declared would break every program that uses `note` today.
- **`latest` is a hint to readers**, not a rule FunctAI enforces: every
  note is still an event; a reader may keep the latest alone.
- **The answer so far can still show a half-written string** where the shape
  is not a choice ("Checkout hi" for "Checkout hides fees"): it is what was
  written, and every reader knows it is provisional.
- **An `anyOf` of several shapes** (other than one shape and null) is read
  as `{}`: a choice inside it shows as written. Following every option
  would guess which one the model is writing.
