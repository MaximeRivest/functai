# The FunctAI contract

What every FunctAI implementation (Python in `../python`; TypeScript,
R and Julia next, see `../design/01-many-languages.md`) must agree on, so
that one implementation's output is another's input and a dashboard can
read both. This folder is the authority: when an implementation and the
contract disagree, the implementation is wrong. No language's folder is
imported by another; they meet only here.

- [`calls.md`](calls.md): the call log. Every call of an AI function or
  module as one line of JSON in a folder; people's ratings of those calls;
  the rows with known answers the ratings make; what a program's
  version is.
- [`streaming.md`](streaming.md): streaming. A stream is the same call,
  watched while it is made; the events it shows, their order, and their
  JSON form.
- `schema/`: JSON Schemas (draft 2020-12) for a call record, a rating
  record and a stream event. Every record and event an implementation
  writes must pass them.
- `cases/`: logs and the rows `rated` must make from them, written from
  the rules in `calls.md` (`cases/make.py` writes them). An
  implementation passes a case when its `rated` gives exactly the
  expected rows and left-out counts.

The Python implementation checks both in `python/tests/test_call_log.py`
(`cd python && uv run pytest tests/test_call_log.py`); `../check` runs
every implementation against the contract.

A change to the format is a new format number (`functai_call: 2`), never
an edit of format 1: logs outlive the code that wrote them. The one
exception is a new *optional* field that readers may ignore (streaming
added `streamed` and `first_delta` to exchanges this way, 2026-09-27):
readers already skip fields they do not know, so every older log and
reader stays valid. Changing what a field means, or requiring a new one,
is a new format.

The streaming contract has a schema but no cases yet: its laws are
checked by the Python tests (`python/tests/test_streaming.py`) with a fake
model. Shared cases need a fake model both languages can run (a script
of replies and how they are cut into pieces), which is the next thing to
write when the TypeScript implementation starts streaming.

Each rule of "Rows with known answers" was broken on purpose in the
Python implementation (thirteen breaks: the earliest rating instead of
the latest, ties by time only, ignoring withdrawals, the module, the
signature, the order, …) and at least one case failed every time
(2026-09-26).
