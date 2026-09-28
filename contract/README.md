# The FunctAI contract

What every FunctAI implementation (Python in `../python`, TypeScript in
`../ts`, R in `../r`, Julia in `../julia`; see `../design/01-many-languages.md`) must
agree on, so that one implementation's output is another's input: the
same function sends the same bytes, has the same version, logs the same
records, gets the same score, and loads from the same saved folder in
every language.

This folder is the authority: when an implementation and the contract
disagree, the implementation is wrong. No language's folder is imported
by another; they meet only here.

| document | what it fixes | cases |
|---|---|---|
| [`functions.md`](functions.md) | an AI function: the signature a definition becomes, its layout ([`layouts/`](layouts/)), what a model can do ([`models.json`](models.json)), worked examples, the request, what happens when a reply cannot be read | `cases/functions/` |
| [`calls.md`](calls.md) | the call log: every call as one line of JSON, ratings, the rows with known answers they make, versions | `cases/rated/` |
| [`scores.md`](scores.md) | whether an answer is right, the score and its 95% range | `cases/scores/` |
| [`programs.md`](programs.md) | a program's interface: the inputs and outputs every AI function has and every module declares; its id; checking values against it | `cases/programs/` |
| [`calls.md`](calls.md), *Content* and *Saw* | which values a record keeps, per input and output; the calls a call was shown as context | `cases/content/`, `cases/saw/` |
| [`saved.md`](saved.md) | a saved program's manifest, and loading its AI functions in another language | `cases/saved/` |
| [`streaming.md`](streaming.md) | a stream: the same call, watched; its events, their JSON form, replaying them, keeping them while they are written | `cases/events/` |

`schema/` holds JSON Schemas (draft 2020-12) for a call record, a rating
record, a stream event and a saved manifest. Every record, event and
manifest an implementation writes must pass them, and every one in the
cases does (`make.py` checks each against its schema as it writes it).

**Cases are written from the rules, never from an implementation's
output.** `cases/make.py` writes all of them (`python/.venv/bin/python
contract/cases/make.py`); `../check` fails when the committed cases
differ from what it writes. One step is borrowed: turning a signature, a
layout and values into a request is lmcc's rule, pinned byte for byte by
lmcc's own corpus, so `cases/functions.py` asks lmcc for that step and
derives everything FunctAI decides itself. The layouts and the model
table are data both implementations must reproduce exactly.

`../check` runs every implementation against every case.

**Formats.** A change to a format is a new format number
(`functai_call: 2`), never an edit of format 1: logs and saved folders
outlive the code that wrote them. The one exception is a new *optional*
field that readers may ignore (streaming added `streamed` and
`first_delta` to exchanges this way; saved manifests gained `language`,
`body` and `version`, 2026-09-27): readers already skip fields they do
not know, so every older log and reader stays valid. Changing what a
field means, or requiring a new one, is a new format.

The streaming contract's cases (`cases/events/`) pin what can be
checked on events as data: replaying a stream, its stored form, the
rules a store keeps. How a call *makes* its events (the laws of order,
exact text, retries) is checked by each language's own tests with a fake
model (`python/tests/test_streaming.py`). Shared cases for that need a
fake model every language can run (a script of replies and how they are
cut into pieces).

Which cases each language must pass: every language that reads the call
log passes `rated/`, `content/` and `saw/`; every language that has
modules passes `programs/`; every language that streams passes
`events/` (`store-*` cases once it keeps streams: stage 2). R has
neither modules nor streaming yet.

Each rule of "Rows with known answers" was broken on purpose in the
Python implementation (thirteen breaks: the earliest rating instead of
the latest, ties by time only, ignoring withdrawals, the module, the
signature, the order, …) and at least one case failed every time
(2026-09-26).
