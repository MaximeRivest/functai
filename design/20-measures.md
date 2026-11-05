# 20 — Measures: cases, results, ratings, and honest numbers

*Status, 2026-11-05: proposed and built the same day in the contract,
Python and TypeScript (branch `evals`). R and Julia read every older
case as before; the new cases (`measures/`) wait for them. Evidence for
each statistical choice is in `design/20-evidence/`: scripts that
compute exact or simulated coverage with the contract's own rule code
(`contract/cases/measure_rules.py`), and their output (`*.txt`); run
them with any Python 3.*

## Why

FunctAI measures a program with `evaluate` (rows with known answers, a
score with its 95% range) and compares two with `compare` (paired, row
by row). Quiver, built on FunctAI, has grown four separate kinds of
evidence around that: tests on examples, deploy checks, people's yes/no
verdicts on items of answers (Poser's insights), and side-by-side picks
of two answers. They never meet: each lives in its own table and page,
and the deploy rule can use only the first.

Looking at what Quiver actually needs (Chattering 01a1abb7, 2026-11-05)
showed five problems that belong to FunctAI, not to Quiver:

1. **`compare`'s range is wrong where it matters most.** It is a paired
   t interval. For right-or-wrong scores in the regime real evaluations
   live in (most cases right for both versions, a few that differ), its
   "95%" range holds the truth 68% of the time at 10 cases and 84% at
   30, and as little as 1 to 3% for some true rates. When no case
   changed it says the difference is exactly 0, with no uncertainty at
   all: every `review_mood` version scores 30/30, so Quiver's "not
   worse" deploy rule passes whatever changed. (`20-evidence/coverage.txt`,
   exact enumeration over every possible table.)
2. **Nothing says what a case is.** `compare` needs the same rows in the
   same order; two evaluations on overlapping data cannot be compared at
   all, and two runs of the same version on the same input cannot be
   recognised as such. Results cannot be reused, repeats cannot be told
   apart from new cases, and a deploy check runs everything again.
3. **A rating is only "right or wrong about one whole call, by a
   person".** Quiver needs a person's answer to a question about one item
   of an answer ("Is this a real problem?" of insight 3), a score, a pick
   between two answers, and the same judgements made by an AI judge or by
   code, said apart from people's. So it keeps two tables of its own,
   and FunctAI's `rated` and `.opt` never see them.
4. **AI judges are numbers no one checked.** A judge program can be a
   `metric`, but nothing records what it said, nothing compares it with
   people, and nothing combines a few people's judgements with many of
   a judge's.
5. **Fair evidence is not told from convenient evidence.** Format 1 has
   `sample` (the rating belongs to a random draw), but nothing draws, and
   nothing records whether the person knew which version made what they
   judged.

## The model: three things

Everything is statistics over three kinds of data, all of which are
already, or now become, records of the call log:

| thing | what it is | where it lives |
|---|---|---|
| **case** | inputs a program can be asked again (with the earlier turns, for a conversation's turn), and what is known of a good answer | a row (any table), named by its **case key** |
| **result** | what one setup answered on one case, once | a call record, carrying the case key (`case`) and which repeat it is (`repeat`) |
| **rating** | who said what about a result: a person, an account, a judge program, or code; about the whole answer or one item of it, or about two results side by side | a rating record: format 1 when it says only "right or wrong", format 2 otherwise |

Tests, deploy checks, item verdicts and side-by-side picks are all
ratings of results on cases. A scoreboard, a comparison and a deploy
rule are statistics over them, the same for a mood classifier, a
usability study, a conversation or an agent with tools.

### Cases and their keys

A case key is `"sha256:"` and the SHA-256 of the canonical JSON of
`{"inputs": I}` (with `"earlier"`, `"sections"` and `"helpers"` when the
row keeps a conversation's context: calls.md, *Rows that keep their
context*), `I` the row's values of the program's inputs, as a call
record writes them. It depends on nothing else in the row: correcting
the expected answer, adding a tag or moving the row keeps its key.

A call's case is the `case` its record carries (written by `evaluate`),
or else the key of its own recorded inputs, when every input was
recorded as data and it saw nothing (a live call at an address). So a
version's answers to real traffic are results on cases, and a new
version run on rows made from those calls (a shadow run) is compared on
the same keys.

**Held out, the same everywhere.** A row's `split` column says
`"train"` or `"held_out"`. A row without one is held out when the first
13 hex digits of SHA-256(`"functai-split:" + salt + ":" + K`), read as
an integer and divided by 2^52, is below the held-out share (`K` the
row's conversation id when it has one, so a conversation stays on one
side; else its case key). Adding cases never moves an old one, every
language assigns the same side, and no seed is needed. The optimizers
refuse rows marked `held_out`: a held-out case shown to an optimizer
measures nothing.

**Random draws, the same everywhere.** `draw(ids, n, seed)` keeps the
`n` ids with the smallest SHA-256(`"functai-draw:" + seed + ":" + id`):
uniform, reproducible in every language, and stable (a draw over more
calls keeps the earlier picks that still rank). Ratings of drawn calls
carry `sample: seed`; only those measure a program without the bias of
people choosing what to judge.

### Results

`evaluate` writes two optional fields on the calls it makes: `case`
(the key) and `repeat` (1, 2, … when a case is asked several times).
Readers that do not know them ignore them, so the call log stays format
2 (calls.md, *Formats*: a new optional field).

- **Repeats** (`repeats=k`) ask each case `k` times, each repeat after
  the first as another replicate (replies.md: a cached reply would repeat
  the first answer, which measures nothing; a replicate has its own). A case's value is the mean of its repeats, so the
  range is still over cases, the unit that is independent; how much a
  setup disagrees with itself is reported beside it (*Self-agreement*).
- **Reuse** (`reuse=` a folder or store) takes the results already in
  the log: calls of the same program with the same `program.setup`, on
  the same case, that ended without an error and recorded their
  outputs, oldest first. Only what is missing is asked. A deploy check
  then costs only the new version's calls. Failed calls are asked again
  (most failures are a provider's, not the program's). What it hides: a
  reused answer was drawn once; `repeats` is how to see past it.

### Ratings, format 2

A rating says who judged, what, about which question, and how fair the
judgement was:

```json
{"functai_rating": 2, "id": "…", "at": "…",
 "call": "…", "item": "/insights/3",
 "criterion": "real_problem", "question": "Is this a real problem?",
 "kind": "label", "answers": {"yes": 1, "no": 0, "unsure": null},
 "value": "yes",
 "by": "megan", "blind": true, "sample": "draw-2026-11-05"}
```

- **What**: `call` (one result), with `item`, a JSON Pointer (RFC 6901)
  into its outputs for one item of the answer; or `calls`, two results
  on the same case, in the order they were shown.
- **Which question**: `criterion`, a name; `question`, its words for
  people. Format 1's meaning is the criterion `correct`.
- **The answer**: `value`, read by `kind`:
  - `verdict`: `"right"` or `"wrong"` (with `answer` and `outputs`, the
    correction, as in format 1);
  - `label`: one of the record's `answers`, each the number it counts
    for (from 0 to 1) or `null` (it counts for nothing: "unsure");
  - `score`: a number within the record's `scale` `[lowest, highest]`;
  - `choice`: one of the two `calls`, or `"tie"`.
  - `null` withdraws the rater's earlier rating, as in format 1.
- **Who**: `by` (a person), `account` (an operating system's account,
  which names no one), or `judge` (`{"name", "setup"}` for a program,
  `{"name", "code"}` for code: a judge is known by what it runs on, so a
  new judge prompt is a new judge). A program judge's own call is
  `judge_call`.
- **How fair**: `blind` (the rater was not told which version made what
  they judged), `sample` (it belongs to a random draw).

A rating format 1 can say is written as format 1 (a person's or an
account's right or wrong about a whole call), so R, Julia and older
readers keep reading every such rating. Format 1 gains `blind`, an
optional field (a reader that ignores it reads the verdict as before).

**What counts** generalises format 1's rule 2: for each target and
criterion, a person's latest rating counts (a later one replaces it, a
`null` one withdraws it), an account's counts on its own, and a judge's
latest counts for each order it was shown a pair in. A judge is asked
both orders on purpose, because judges favour a position [2]; a person
judging the same pair again changes their mind, whichever side each
answer was on.

### Criteria

A criterion is data: a name, a kind, its answers or scale, and for
items, the list it judges (`items: "/insights"`) and how a result's items
add up (`"mean"`: the share of good items; `"sum"`: how many good items,
counted from the share times the list's length). It may name a judge:
code (`(case, answer) -> value`) or a program, given by name what it
asks for: the case's inputs, `answer`, `outputs`, `expected`, `item`,
and for a pair `first` and `second`.

The value of a result under a criterion and one source of ratings:

- a whole-answer criterion: the mean of that source's counting ratings;
- an item criterion: each item's mean, then the items' mean (or the
  count);
- **a failed call** counts as the worst answer: 0 for a whole-answer
  criterion, no items (a count of 0), and it loses a pair to any answer
  (a tie when both failed). A version that fails often must not look
  good because its failures go unjudged.

The value of a setup on a case is the mean over its results. Every
range below is over cases.

## The statistics

Every number is computed from data by formulas the contract writes
down (`contract/measures.md`), the same to within 1e-9 in every
language: no resampling, no random numbers, no optimiser with a
tolerance a language may read differently.

| question | method | why this one |
|---|---|---|
| how good is one setup | the mean over cases; Wilson's interval when every case is 0 or 1, else Student's t (scores.md, unchanged) | Wilson keeps its coverage near 0% and 100%, where LLM scores sit [4] |
| is B better than A | the mean of B − A over shared cases. When every value is 0 or 1: the MOVER-Wilson interval with the continuity-corrected φ [6][7]; else the paired t interval | exact coverage over every possible table (`20-evidence/coverage.txt`, 120 realistic settings): never below 0.919 at 20 cases and 0.938 at 30, where the paired t falls to 0.013 and 0.019. Newcombe's uncorrected φ (method 10) falls to 0.756. The t interval stays for fractions (repeats, items), where it holds 0.95 to 0.96 from 10 cases, and 0.86 to 0.95 at 5 (`paired_frac.txt`) |
| does B win side by side | the mean of B's share of each case's picks (ties half); Wilson's interval on that share | never a zero-width range, even when every pick is a tie |
| a judge on everything, people on a random few | prediction-powered inference with power tuning (PPI++ [1][8]): λ fitted to the data, within [0, 1]; the people's part's variance counted with two pseudo-observations, half the values' range either side; Student's t with n − 2 degrees of freedom | published PPI++ held the truth 72 to 93% of the time with 6 to 60 people's judgements (`ppi.txt`); with the pseudo-observations, 94 to 98%. It is narrower than people alone once people judged 15 or more cases and the judge is good (0.24 against 0.38 with 15 of 200 cases); with fewer, people alone is as narrow or narrower, so the scoreboard shows both |
| how do many versions rank | Bradley–Terry, fitted by Hunter's MM algorithm [9] on per-case outcomes, with one virtual tie between every two versions that met; Wald ranges from the observed information, the first version fixed at 0 | the method of Chatbot Arena; the virtual tie keeps a version that never lost finite |
| would a few cases change the answer | the fewest cases whose removal turns the difference's sign (exact: drop the largest first); for a ranking, the fewest cases whose removal changes which version leads (found greedily, so "at most") | a top ranking can rest on a handful of judgements [5]; saying how many is cheap and honest |
| can this judge stand in for people | on targets both rated: agreement and Cohen's κ (right or wrong, labels), mean absolute difference and correlation (scores), the same winner (picks); a judge asked both orders: how often it picked the same answer both times, and how often the first shown won | the calibration a judge needs before it may decide anything |
| does a setup agree with itself | over cases with two or more results: the share of pairs of results with the same value (1 − their mean difference for fractions), and the spread within cases | tells "B beat A" from luck when answers vary run to run |

Comparisons say what they found, in words a deploy rule can act on:
`better` (the whole range above 0), `worse` (below 0), and with a margin
`m` (how much worse is still acceptable, the program's to set): `same`
(within ±m), `not_worse` (above −m), `not_better`, else `unclear`. With
no margin, "not worse" can only mean "not shown worse", which is what
Quiver's rule says today: with this design it at least says how little
the cases could show.

## What every program gets

```python
ev = functai.evaluate(prog, rows, repeats=3, reuse=log, criteria=[correct, real_problem])
functai.compare(ev_before, ev_after, margin=0.05)           # shared cases, paired, with outcome and fragility
functai.rate(pred, "yes", criterion=real_problem, item="/insights/3", blind=True, sample=draw_id)
functai.prefer(pred_a, pred_b, winner=pred_b, blind=True)
functai.judge(professor, calls, real_problem)               # an AI judge's ratings, kept in the log
functai.judge_pairs(picker, pairs, preference)              # both orders, kept
functai.scoreboard(prog, criteria=[...], records=log)       # per setup: each criterion by each source, PPI included
functai.agreement(log, real_problem, a="people", b=professor)
functai.rank(log, preference)                               # Bradley–Terry over every version
functai.draw(call_ids, 20, seed="draw-1")                   # a fair sample to judge
functai.split_of(row)                                       # held out or not, the same in every language
functai.disagreements(log, prog, criterion)                 # cases to look at, or to add
```

TypeScript has the same, camel-cased. Quiver uses these and keeps
nothing of its own beyond pages: its rating and preference tables give
way to rating records in the call log store it already keeps.

## Acceptance

The same calls, unchanged, on four shapes (tests in both languages):

1. `review_mood`, known answers: `evaluate` with `correct` (exact
   match), repeats, reuse, and a compare whose outcome is `unclear`
   when 30 cases are all right for both (the old rule said "no
   difference").
2. A Poser-like study: no answers, a list of insights judged item by
   item by people (a random draw, blind) and by a judge program;
   scoreboard with people, judge and PPI rows; agreement of the judge
   with people.
3. A conversation's turns: cases with earlier turns, their own keys, a
   conversation on one side of the split.
4. An agent with tools: results with steps, failures counted as the
   worst answer, pairs judged both orders, a ranking of three versions.

## Trade-offs

- **R and Julia lag.** They read every rating they read today (format 1
  is still written whenever it can say the rating), but not format 2,
  and they have no measures until they catch up. The cases wait for
  them (`measures/`).
- **More conservative ranges.** The new paired range is wider than the
  old one when few cases differ (that is the point). Deploy checks that
  passed with "no difference" now say `unclear`, and pass only if the
  host accepts "not shown worse".
- **The case is the unit.** Items of one answer are not independent,
  so they are averaged within the case before any range. A study of
  five sites has five units, however many insights it found: its ranges
  are wide, and that is true.
- **PPI's guarantee needs a random draw.** People's ratings they chose
  to make are counted, but only as "people (chosen)", never in the
  combined estimate.
- **Wald ranges for rankings** are approximate with few comparisons;
  the fragility count says how much to trust the order.
- **Agreement ignores clustering** (several items of one answer); it is
  a description of the judge, not a test.
- **Reuse hides randomness** (said above); **failures count as the worst
  answer**, even a provider's outage.
- **Criteria are given to `evaluate` and `scoreboard`, not declared on a
  program.** How a program is judged is the measurer's (a suite's, a
  group's), as in Inspect's tasks; programs stay what they send.

## References

- [1] https://doi.org/10.48550/arxiv.2301.09633 — Angelopoulos et al. 2023, Prediction-Powered Inference
- [2] https://doi.org/10.48550/arxiv.2406.07791 — Judging the Judges, 2024
- [3] https://doi.org/10.48550/arxiv.2411.00640 — Miller 2024, Adding Error Bars to Evals
- [4] https://doi.org/10.48550/arxiv.2503.01747 — Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints, 2025
- [5] https://doi.org/10.48550/arxiv.2508.11847 — Dropping Just a Handful of Preferences Can Change Top LLM Rankings, 2025
- [6] https://doi.org/10.1002/(sici)1097-0258(19981130)17:22<2635::aid-sim954>3.3.co;2-3 — Newcombe 1998, paired differences of proportions
- [7] https://doi.org/10.1002/sim.3738 — Tang, Li, Tang 2009, MOVER intervals for paired proportions
- [8] https://doi.org/10.48550/arxiv.2311.01453 — Angelopoulos, Duchi, Zrnic 2023, PPI++
- [9] https://doi.org/10.1214/aos/1079120141 — Hunter 2004, MM algorithms for Bradley–Terry models

## What was built

Built as proposed, in the contract, Python and TypeScript, the same day.
Where it differs, or what the proposal left open:

- **The contract's rules are code** (`contract/cases/measure_rules.py`);
  the 91 cases in `measures/` are written from it, and each language's
  harness runs them through its own implementation
  (`python/tests/test_contract_measures.py`, `ts/tests/measures.test.ts`).
  The schemas refuse malformed ratings of format 2 (13 refusals in
  `cases/schemas.py`).
- **Repeats are replicates**, not calls around the reply cache: with a
  cache on, asking again gives the same repeats, cheaply.
- **`compare` pairs by case** and no longer refuses rows that differ:
  rows in another order, a subset, or an extra column that is not an input
  compare on the cases both answered (Python warns how many are left out).
  `test_evaluation.py` and `calls.test.ts` changed accordingly.
- **A judge's ratings are reused** by `evaluate(reuse=)`: the same call
  and the same judge (setup, or code hash) is not asked again.
- **Code judges** are called by name in Python (their parameters) and with
  one object in TypeScript (`({ item, url }) => …`): JavaScript cannot read
  parameter names. A code judge's id is the hash of its text in each
  language, so the same judge written in two languages is two judges, as a
  tool's code makes two setups.
- **Reused answers** are rebuilt as the program's types in Python where
  they build (dataclasses, pydantic models, enums: `signature.from_json`),
  and left as data otherwise.
- **A module's setup counts its code, not what a closure captured**: two
  modules made by one factory with different numbers have one setup (found
  by a test; design/18 says so for tools). Measure such variants as
  separate functions.
- **The four shapes** are tests in both languages (`test_measures.py`,
  `measures.test.ts`): a classifier with known answers, a study judged
  item by item by people and a judge, a conversation's turns, agents with
  tools ranked side by side.
- **R and Julia** pass every case they passed before (`rated/15`'s unknown
  rating format moved from 2 to 9), and read every rating Python and
  TypeScript write for a whole call's right or wrong.
