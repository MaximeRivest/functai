# The FunctAI contract

What every FunctAI implementation (Python here, TypeScript next) must
agree on, so that one implementation's output is another's input and a
dashboard can read both.

- [`calls.md`](calls.md): the call log. Every call of an AI function or
  module as one line of JSON in a folder; people's ratings of those calls;
  the rows with known answers the ratings make; what a program's
  version is.
- `schema/`: JSON Schemas (draft 2020-12) for a call record and a rating
  record. Every record an implementation writes must pass them.
- `cases/`: logs and the rows `rated` must make from them, written from
  the rules in `calls.md` (`cases/make.py` writes them). An
  implementation passes a case when its `rated` gives exactly the
  expected rows and left-out counts.

The Python implementation checks both in `tests/test_call_log.py`
(`uv run pytest tests/test_call_log.py`).

A change to the format is a new format number (`functai_call: 2`), never
an edit of format 1: logs outlive the code that wrote them.

Each rule of "Rows with known answers" was broken on purpose in the
Python implementation (thirteen breaks: the earliest rating instead of
the latest, ties by time only, ignoring withdrawals, the module, the
signature, the order, …) and at least one case failed every time
(2026-09-26).
