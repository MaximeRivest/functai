# Measures (format 1)

How good a program is, how it compares with another, and how far to
trust the answer, computed the same way in every language from the
same records: the same cases, results and ratings give the same numbers
(within 1e-9) in Python and TypeScript, and in every language that
measures. Why each rule is this one, with the evidence:
[`../design/20-measures.md`](../design/20-measures.md).

`cases/measures/*.json` pin these rules; `cases/measure_rules.py` is
them as code (the cases are written from it).

## Words

- **Case**: inputs a program can be asked, and what is known of a good
  answer: a row of any table (its columns named like the program's
  inputs are its inputs; others are kept). Named by its **case key**.
- **Result**: one answer of one setup to one case: a call record
  ([calls.md](calls.md)) and its case key.
- **Rating**: a judgement of a result (or of one item of it, or of two
  results side by side), by a person, an account, a judge program or
  code ([calls.md](calls.md), *A rating record*, formats 1 and 2).
- **Criterion**: the question a rating answers, as data (*Criteria*).
- **Source**: whose ratings are read: `people` (persons and accounts),
  or one judge, by the setup of its program or the hash of its code.
- **Group**: what results are measured by: their `program.setup` (the
  default: what ran), or their `program.version`.

## Case keys

A case key is `"sha256:"` and the hex SHA-256 of the canonical JSON
([calls.md](calls.md), *Canonical JSON*) of an object with:

- `inputs`: the row's value of each of the program's inputs it has,
  written as a call record writes a value (*Values*): an input the row
  leaves out is not there (an optional input takes the program's
  default, which is the version's, not the case's);
- `earlier`, `sections`, `helpers`: the row's, when it has them as
  context (calls.md, *Rows that keep their context*: a key that is not
  one of the program's inputs) and they are not empty.

Nothing else of the row counts: correcting its answer, adding a tag or
moving it keeps its key.

**A call's case** is its record's `case` when it has one (written by
`evaluate`, *Results*). Otherwise, for a format-2 record whose `saw` is
empty and that kept every input as data (calls.md, rule 3: not
`no_content`), the key of `{"inputs": its inputs}`; otherwise it has
none, and is not a result. A live call is therefore a result on the
case of its own inputs (with the defaults it was given, which its
record holds), and rows made from such records (a shadow run) have the
same keys.

## Held out

A row's side (`split_of`) is its `split` column when it has one that is
not one of the program's inputs: `"train"` or `"held_out"`; any other
value is refused (`case-split`). Otherwise it is `"held_out"` when

    int(first 13 hex digits of SHA-256(UTF-8 of "functai-split:" + salt + ":" + G), 16) / 2^52 < share

and `"train"` otherwise, where `share` (from 0 to 1; default 0.2; out of
range: `case-split`) is the share to hold out, `salt` a text (default
empty: another salt draws another split of the same rows), and `G` the
row's `conversation` when it is a non-empty text (and not an input):
every turn of a conversation is on one side; else its case key.

An optimizer is never given a row whose side is `held_out` (it
refuses, naming how many): a held-out case shown to an optimizer then
measures what it was shown.

## Random draws

`draw(ids, n, seed)`: the ids, each once, ranked by the hex SHA-256 of
`"functai-draw:" + seed + ":" + id` (then by id), and the first `n`. A
draw is uniform, the same in every language, and stable: drawing from
more ids keeps each earlier pick that still ranks. A rating of a drawn
call (or item) carries `sample: seed`: only such ratings measure a
program without the bias of people choosing what to judge.

## Results

`evaluate` writes, on each call it makes, `case` (the row's key) and
`repeat` (`1`, `2`, … when a case is asked several times; absent means
1). Both are optional fields of call format 2.

A program's results are its calls (`program.name` equal, and
`program.module` and `program.file` when given, as `rated` reads them),
the latest writer of each id, in order of `started` then `id`, that have
a case. Each result's group is its `program.setup` (or
`program.version`; null when the record has none), and it **failed**
when its `error` is not null.

**Reusing results.** Asked to reuse a log, `evaluate` takes, for each
case, the calls of the same program with the same `program.setup` and
case that ended without an error and recorded their outputs (`content`
true), oldest first, as its first repeats, and asks only for the rest.
A repeat after the first is asked as another replicate
([replies.md](replies.md): `replicate` one more for each repeat), so a
reply cache gives it its own answer, and asking again gives the same
repeats.

## Criteria

A criterion is `{"name", "kind", "items"?, "aggregate"?, "answers"?,
"scale"?, "question"?}`:

| kind | a rating's `value` | the number it counts for |
|---|---|---|
| `verdict` | `"right"` or `"wrong"` | 1 or 0 |
| `label` | a key of the rating's `answers` | that answer's number (from 0 to 1), or nothing when it is `null` ("unsure") or not an answer |
| `score` | a number | `(value − lowest) / (highest − lowest)` with the rating's `scale` `[lowest, highest]`; the value as it is without one |
| `choice` | one of the pair's `calls`, or `"tie"` | the later of the two groups' share: 1 if it won, 0 if it lost, ½ a tie |

`items`, a JSON Pointer into a result's outputs naming a list
(`"/insights"`), makes a criterion judge each item of that list (the
rating's `item` is the pointer and the item's index: `"/insights/3"`).
`aggregate` says how a result's items add up: `"mean"` (the default:
the share of good items) or `"sum"` (how many: that share times the
list's length).

**Bounds.** A criterion's values lie between 0 and 1, except with
`aggregate` `"sum"` and for a score without a scale. A bounded value's
range is kept within its bounds (0 and 1; −1 and 1 for a difference).

## Reading ratings

A rating of format 1 is read as criterion `correct`, kind `verdict`,
value its `verdict`, about its whole call. A rating of format 2 is read
as it is. Records of any other format are skipped.

**What counts.** For each criterion (name and kind equal) and target
(the calls and the item; the whole answer has no item):

1. a person's (`by`) latest rating counts (latest `at`; equal times: the
   larger `id`), whichever order a pair was shown in;
2. a judge's latest counts for each order it was shown a pair in (a
   judge is asked both orders on purpose);
3. each rating under an account counts on its own;
4. a `null` value withdraws (1, 2) or counts for nothing (3).

People's ratings that do not carry `sample` were chosen by people:
they are read under `people`, never under `people` *drawn*.

## Values

**A result's value** under a criterion and a source:

- whole-answer criterion: the mean of the numbers of the source's
  counting ratings of the whole answer; none when there are none;
- item criterion: for each item of the list, the mean of the source's
  numbers; then, over the items that have one, their mean (`"mean"`), or
  that mean times the list's length (`"sum"`); none when no item has
  one;
- **a failed result** is the worst answer: 0 for a whole-answer
  criterion; for an item criterion, no items (`"mean"`: none; `"sum"`:
  0). An empty list's `"sum"` is 0; a missing list is no items.

**A group's value on a case**: the mean of its results' values that
exist. Every range below is over cases, never over results or items:
items of one answer, and repeats of one case, are not independent.

**Side by side.** For two groups A and B, each case's value is the mean,
over every pair of one A result and one B result on that case, of: ½
when both failed, 1 when only the A result failed, 0 when only the B
result failed, else the mean of the source's counting picks of that
pair (B's result 1, A's 0, a tie ½), when there are any.

## The statistics

Means are summed in the order given. `t975`, Wilson's interval and
`interval` are [scores.md](scores.md)'s. Variances divide by `n − 1`.

### Is B better than A (`paired`)

`a`, `b`: the two groups' values on their shared cases (key order); `d =
b − a`; `diff` the mean of `d`. With fewer than two cases there is no
range. When every value of `a` and `b` is 0 or 1 (`method`
`"mover-wilson"`):

- `n11`, `n10`, `n01`, `n00` count the cases right for both, A only, B
  only, neither; `pA = (n11 + n10)/n`, `pB = (n11 + n01)/n`; `[lA, uA]`
  and `[lB, uB]` their Wilson intervals (exact 0 and 1 at the ends);
- `x = n11·n00 − n10·n01`; when `x > 0`, `x = max(x − n/2, 0)`;
  `φ = x / √((n11+n10)(n01+n00)(n11+n01)(n10+n00))`, 0 when that
  product is 0;
- `low = diff − √((pB−lB)² − 2φ(pB−lB)(uA−pA) + (uA−pA)²)`,
  `high = diff + √((uB−pB)² − 2φ(uB−pB)(pA−lA) + (pA−lA)²)` (each
  square root of the larger of its argument and 0), kept within −1
  and 1.

Otherwise (`"paired-t"`): `diff ± t975(n−1)·sd(d)/√n` (kept within −1
and 1 for a bounded criterion). Also: `better`, `worse`, `same` (how
many `d` are above, below, equal to 0), `n`, `fragility` and `outcome`.

### Does B win side by side (`win_rate`)

`v`: B's share on each case. `rate` the mean; the range Wilson's
interval of `Σv` successes in `n` (fractional successes allowed; none
with fewer than two cases); the outcome of `low − ½` and `high − ½`;
fragility of `v − ½`.

### Outcomes

From a range `[low, high]` of a difference and a margin `m` (absent or
0: none), the first that holds: no range: `unclear`; `low > 0`:
`better`; `high < 0`: `worse`; with a margin: `low > −m` and `high < m`:
`same`; `low > −m`: `not_worse`; `high < m`: `not_better`; else
`unclear`.

### Fragility

The fewest values whose removal takes the mean of `d` to 0 or across it:
0 when the mean is 0; otherwise remove the values most in the mean's
direction first (ties: the earlier), and count until the remaining mean
is no longer of that sign, or none remain.

### A judge on everything, people on a random few (`ppi`)

`f` the judge's values and `y` people's (drawn) values, by case. `L`:
the cases both have (`n` of them, key order); `U`: the cases only the
judge has (`N`). With `n < 3`: nothing.

- `vf`: the variance of the judge's values over `L` then `U` (0 with
  fewer than two);
- `λ = 0` when `N = 0` or `vf = 0`; else `cov_L(y, f) / ((1 + n/N)·vf)`,
  kept within 0 and 1;
- `r = y − λf` over `L`; the estimate `λ·mean_U(f) + mean(r)` (only
  `mean(r)` when `N = 0`);
- `h`: half the bounds' width for a bounded criterion (½ for
  values, 1 for differences); else half the range of every value seen
  (`y`, then `f` over `L` and `U`);
- `v_r = (Σ(r − mean r)² + 2h²) / (n + 1)` (two pseudo-observations, `h`
  either side of the mean);
- `se = √(λ²·vf/N + v_r/n)` (only `v_r/n` when `N = 0`); the range is the
  estimate `± t975(n − 2)·se`, kept within the bounds when bounded.

Also `n`, `unlabeled` (`N`), `lambda`. **Paired** (`ppi_paired`): the
same over the differences B − A of the cases both groups have, for the
judge and for the people, as `diff`, with an `outcome`.

### How do many versions rank (`rank`)

Outcomes `(case, first, second, first's share)`. Each two versions
that met get one more outcome, a tie (½). The **reference** is the first
version given that met any; only versions connected to it through
outcomes are ranked (the others have `null` strengths).

- `W_i`: the sum of version i's shares; `n_ij`: the number of outcomes
  between i and j;
- start every `p` at 1; repeat: for each ranked i, `p'_i = W_i / Σ_j
  n_ij/(p_i + p_j)` (j over ranked versions in order, every `p` from the
  previous round); divide every `p'` by the reference's; stop when the
  largest `|ln p'_i − ln p_i|` is below 1e-12, or after 10,000 rounds;
- `strength_i = ln p_i`, the reference's 0; `vs_reference = p_i / (p_i +
  1)`;
- the range: `strength ± 1.959964·se`, `se²` the diagonal of the inverse
  of the information over the ranked versions but the reference
  (`I_ii = Σ_j n_ij π_ij π_ji`, `I_ij = −n_ij π_ij π_ji`, `π_ij = p_i/(p_i
  + p_j)`, the sum over every ranked j, the reference included),
  inverted by Gauss–Jordan elimination with partial pivoting (the
  largest absolute value, the first on ties); the reference's range is
  `[0, 0]`;
- `order`: ranked versions by strength, highest first (ties: the
  earlier given);
- `fragility`: at most how many cases change which version leads, found
  greedily: at each step, of the remaining cases (key order), remove
  the one after whose removal the leader's lead over the next is
  smallest (a removal after which the leader is no longer ranked, or
  fewer than two are, is not counted; a lead ≤ 0, or another leader,
  ends the count there; leads within 1e-9 of each other: the earlier
  case); `null` when 20 removals do not change it.

### Can this judge stand in for people (`agreement`)

Two sources' values on the targets both have (or, side by side, both
sources' B shares on the pairs both judged). When every value is 0 or
1, or for side by side (each share read as `first`, `tie` or `second`
by whether it is below, at or above ½): `agree`, the share of equal
values, with Wilson's interval, and Cohen's κ `(p_o − p_e)/(1 − p_e)`
(`p_e` over the categories both use; `null` when `p_e` is 1).
Otherwise `mad`, the mean absolute difference, and `r`, Pearson's
correlation (`null` when either side does not vary).

**Positions** (`positions`): of a judge's counting picks, the pairs it
was shown both ways: `consistent`, the share where it picked the same
answer both times (a tie both times is the same); and of its picks that
are not ties, `first_wins`, the share where the first shown won; each
with Wilson's interval.

### Does a setup agree with itself (`self_agreement`)

Over the cases where it has two or more values: each case's mean, over
its pairs of values, of `1 − |difference|`; their mean with `interval`
(kept within 0 and 1); `spread`, the square root of the mean, over those
cases, of the variance of the case's values (dividing by their number).

## Scoreboard

For a program, its criteria (single answers) and a grouping: for each
group, in order of its first result:

- `results`, `cases`, `failed`; `seconds`, the mean of its results'
  `seconds`;
- `input_tokens`, `output_tokens`: summed over the trees of its results
  (each result and every call inside it, by `parent`), over the calls
  that asked a model (`model` not null), `null` when one of them did not
  report the count; `spent_usd`, the sum of their `spent.usd`;
- `measures`: for each criterion, in order, for each source: `people`,
  `people drawn`, then each judge that rated any of its results' targets
  (in order of its first counting rating), then each such judge
  `combined` with people drawn (`ppi`): `{"criterion", "aggregate"
  (item criteria, else null), "source", "judge" (the judge's name, else
  null), "mean", "low", "high", "cases"}`, a source with no value left
  out. A source's estimate is `interval` over its values by case (key
  order), kept within the bounds.

**Comparing two groups** (`compare`): `paired` over the cases both have
values on (key order), with `only_a` and `only_b`, the cases only one
has.

**Disagreements**: the cases on which the groups' values differ
(`"groups"`), then the cases on which one group's results differ
(`"repeats"`); each `{"case", "kind", "spread", "values": [{"group",
"value"}]}`, by kind, then spread (largest first), then case, then
group (a null group as empty text).
