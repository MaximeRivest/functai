# functai 0.1.0 (unreleased)

The first R implementation of FunctAI, held to the same contract as the
Python and TypeScript packages (`../contract`): every function, score,
rating and saved-folder case, and a check against Python and TypeScript
themselves (`../tools/crosslang.py`).

* `ai()`: a vectorised function whose body a model writes. Types are
  prototypes (`character()`, `factor(levels = ...)`, `record()`, a zero-row
  tibble, `vctrs::list_of()`, `optional()`, `described()`); answers come
  back as those types, several outputs as tibble columns that `mutate()`
  splices in.
* Rows run at once over curl (`concurrency`, default 8). A row with a
  missing input is `NA` without a call; failed rows are `NA` with one
  warning, listed by `ai_problems()`.
* The layouts `xml` (default), `chat` and `json`, templates, `module = "cot"`,
  tools (`ai_tool()`); unreadable replies asked again in the contract's words;
  values checked against their types; transient provider errors re-sent.
* `evaluate()` with broom's `tidy()`, `glance()`, `augment()`; the same scores
  and intervals as Python.
* `predict()` and `augment()` for AI functions, with tidymodels' `.pred` columns.
* The call log (`calls()`, `rate()`, `rated()`): the same folder and records
  as Python and TypeScript.
* `labeled_few_shot()`, `bootstrap_few_shot()`, `with_demos()`,
  `with_instructions()`, `update()`.
* `read_ai()` runs AI functions saved in Python or TypeScript; `write_ai()`.
* The `tickets` and `field_notes` datasets.
