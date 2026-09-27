# Scores (format 1)

How right a function is, measured the same way in every language: the
same rows and answers give the same score and the same range, so a score
measured in R compares with one measured in Python.

`cases/scores/*.json` pin these rules.

## Is an answer right? (`exact_match`)

The default metric compares each output the data has a right answer for
with the prediction's. A row scores `1` when every one of them is equal,
`0` otherwise.

Two values are equal when, after normalizing:

- **text** is normalized by splitting it at white space, joining the
  pieces with one space (so white space at the ends goes, and runs of it
  become one space), then applying Unicode full case folding
  (`CaseFolding.txt`, statuses C and F; Python's `str.casefold`), code
  point by code point, as [`unicode/casefold.json`](unicode/casefold.json)
  lists it (Unicode 16.0). So `"  New   York "` equals `"new york"`, and
  `"Straße"` equals `"STRASSE"`. A host whose own Unicode data is older
  may fold a character added since differently; nothing else may differ.
- anything else is compared as it is: numbers by value (`1` equals
  `1.0`), lists and records item by item, without normalizing the text
  inside them.

**White space** is these characters (what Python's `str.split()` and
`str.strip()` treat as white space): U+0009 to U+000D, U+001C to U+001F,
U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029,
U+202F, U+205F, U+3000. (JavaScript's `\s` differs: it adds U+FEFF and
leaves out U+001C to U+001F and U+0085.)

When the answer is a record and the data has columns named like its
fields, those fields are compared instead of the whole record. When more
than one output or field is compared, each is also scored on its own
(`<name>_match`); `exact_match` is all of them together.

**A failed row** (the call raised: an unreadable reply, a provider error)
scores 0 for every metric. It is kept in the results with its error.

## The score and its range

The score is the mean of the rows' values, summed in row order. With
fewer than two rows there is no range.

The range is a 95% interval:

- **Rows that are all 0 or 1** (right or wrong): Wilson's score
  interval, with `z = 1.959964`:
  - `center = (p + z²/(2n)) / (1 + z²/n)`
  - `half = z · √(p(1−p)/n + z²/(4n²)) / (1 + z²/n)`
  - `low = center − half` and `high = center + half`, kept within
    [0, 1]; exactly `0` when `p` is 0, exactly `1` when `p` is 1.
- **Other scores** (a judge's 0 to 1, a count): Student's t.
  - `sd = √(Σ(v − mean)² / (n − 1))`
  - `half = t · sd / √n`, where `t` is the 97.5% quantile of Student's t
    with `n − 1` degrees of freedom: for 1 to 30 degrees, these values:
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
    2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
    2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
    2.045, 2.042. Beyond 30, the Cornish-Fisher expansion
    `z + (z³ + z)/(4d) + (5z⁵ + 16z³ + 3z)/(96d²)`, with the same `z`
    and `d` the degrees of freedom.
  - `low = mean − half`, `high = mean + half`.

Implementations agree to within `1e-12`: the formulas are exact, but
the order of floating-point operations may differ.
