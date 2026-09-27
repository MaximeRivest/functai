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
* `ai_model()`: an AI function as a parsnip model (engine `"functai"`), for
  workflows, rsample, tune (`examples = tune()`, `worked_examples()`) and
  yardstick. Fitting reads the outcome's levels and picks worked examples;
  it calls nothing. `extract_fit_engine()` gives the AI function back.
* `predict()` and `augment()` for AI functions, with tidymodels' columns:
  `.pred_class` for a choice, `.pred` otherwise. `samples = k` answers each
  row `k` times: `type = "prob"` gives each class's share, the class is the
  majority. Without it, probabilities are refused (asked directly) or `NA`
  (through parsnip): FunctAI never makes a probability up.
* `evaluate()` scores any model with a `predict()` method (a parsnip fit, a
  workflow) with the same interval as an AI function.
* `vignette("getting-started")`: from dplyr and ggplot2 to a first tested
  model, for people who have not used tidymodels.
* `vignette("tidymodels")`.
* `?tickets` states the shop's house rules the labels follow.
* The call log (`calls()`, `rate()`, `rated()`): the same folder and records
  as Python and TypeScript.
* `labeled_few_shot()`, `bootstrap_few_shot()`, `with_demos()`,
  `with_instructions()`, `update()`.
* `read_ai()` runs AI functions saved in Python or TypeScript; `write_ai()`.
* The `tickets` and `field_notes` datasets.
