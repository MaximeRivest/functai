# The FunctAI contract

What every FunctAI implementation (Python in `../python`, TypeScript in
`../ts`; R and Julia next, see `../design/01-many-languages.md`) must
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
| [`saved.md`](saved.md) | a saved program's manifest, and loading its AI functions in another language | `cases/saved/` |
| [`streaming.md`](streaming.md) | a stream: the same call, watched; its events and their JSON form | (none yet) |

`schema/` holds JSON Schemas (draft 2020-12) for a call record, a rating
record, a stream event and a saved manifest. Every record, event and
manifest an implementation writes must pass them.

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

The streaming contract has a schema but no cases yet: its laws are
checked by the Python tests (`python/tests/test_streaming.py`) with a
fake model. Shared cases need a fake model every language can run (a
script of replies and how they are cut into pieces).

Each rule of "Rows with known answers" was broken on purpose in the
Python implementation (thirteen breaks: the earliest rating instead of
the latest, ties by time only, ignoring withdrawals, the module, the
signature, the order, …) and at least one case failed every time
(2026-09-26).
