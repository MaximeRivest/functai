# functai 0.1.0 (unreleased)

## Stage 1 foundations (the contract at `c1e5063`)

* The call log is format 2 (`contract/calls.md`): every record has
  `program.interface`, `saw` (`[]`: R's functions are shown no earlier
  call) and, when its content is whole, each exchange's `request_hash`.
  `rated()`, `calls()` and `read_log` read formats 1 and 2 and skip any
  other.
* `log_content` per field, as a layer that only removes: the function's
  own, each `with_ai_config()` block, `ai_config()` and
  `FUNCTAI_LOG_CONTENT=0` (which now wins over every setting; before, a
  function's own `log_content` beat it). `c(transcript = FALSE)`, a named
  list, or `c("question")` (the fields that may be kept). A record not
  whole says what it `omitted`, keeps no request, reply, request hash or
  error message, and drops the reasoning and tool calls with any field.
  A name that is not one of the function's fields refuses when the
  function is defined (`log-content-field`). A one-output function's
  answer is `result` in the log; a map may call it by the formula's name
  too. GEPA's own calls keep no value when the function it improves drops
  a field.
* `defaults_to()`: an input the caller may leave out; the R function's
  argument defaults to it, and the call sends and records it. Defaults
  are out of the signature and the version (`functions/12`).
* `ai_interface()`: what an AI function, or any program in a saved
  folder, takes and gives, as every language describes it. `ai()` checks
  the interface when the function is defined (`interface-malformed`);
  `write_ai()` writes it; `read_ai()` checks a node's interface against
  its signature (`saved-differs`) and takes its optional inputs from it;
  describing a folder needs no loading (`saved-no-interface` for an old
  module node).
* `rated()` pools calls by interface: calls whose data has the same shape
  pool, reasoning or not; a call whose inputs were not kept as data, or
  a right verdict on an answer not kept, is left out and counted.
* Saved folders are checked against the contract's schema before
  anything else (`saved-malformed`); `calls()` gains `content` and `saw`.
* Refusals the contract names are errors of class `functai_refusal`
  (and `functai_<code>`), with `code` and `field`.
* A field name that is not an ASCII identifier (`my.message`) is refused
  when the function is defined (lmcc's `signature-malformed`).

### After review

* An input left out is sent with its default exactly as the interface
  holds it (its JSON), not through an R copy: a record's default that
  leaves out a member no longer gains a `null` for it after `read_ai()`.
  A saved object shape is a tibble column only when every member is
  required; otherwise its values stay JSON.
* A default of `null` is sent; `predict()` makes one call per row of
  `new_data` even when every input is left out, and none for an empty
  table.
* Values are checked on every call by the contract's vocabulary: a given
  input that does not fit fails its row before any request
  (`interface-input`, recorded as `InterfaceError`); a reply that does not
  fit is re-asked (`parse-value`), bounds, `const`, references, array and
  object rules included. A number is no longer truncated to fit a whole
  number, nor a value given to a choice turned into text.
* `defaults_to()` casts as vctrs does (`defaults_to(2.5, integer())` is
  refused), reads a sentence as words about the value's own type, and
  takes a date as text.
* A `log_content` name is the field of that name: a function named like
  one of its inputs, or an answer named like a field FunctAI adds, no
  longer moves that field's rule to the answer.
* A tool-using call's record holds `outputs.calls` (the value lmcc's
  finished turn holds: its last model step's) and its size.
* `read_saw()` reads only call records of formats 1 and 2; an entry whose
  known keys hold values of another kind, or two different records with
  one id, is not guessed at.
* A saved node without an interface is checked like any other; each probe
  needs its own fingerprint (`saved-differs`); load and describe refusals
  carry `field`; describing an optional input with no default no longer
  prints `default null`.
* GEPA's own calls keep no value when any layer in force (a block, `ai_config()`,
  the environment) drops a field of the function it improves.
* `calls()` gains `omitted`. Interface refusals name the first field at
  fault, inputs first, and the call they came from.

## Before stage 1

The first R implementation of FunctAI, held to the same contract as the
Python and TypeScript packages (`../contract`): every function, score,
rating and saved-folder case, and a check against Python and TypeScript
themselves (`../tools/crosslang.py`).

* A temperature or top_p a model does not take (GPT-6, the o-series and
  GPT-5, Claude 5) is left out of its requests with one warning, instead of
  failing every row (`contract/models.json`, `fixed_sampling`).
* `ai()`: a vectorised function whose body a model writes, written like a
  model formula: `ai(team ~ message, "Which team should answer?", team =
  choice("shipping", "billing"))`. Each field is a line of a codebook: a
  sentence describes it, a type types it (`integer()`, `choice()`,
  `record()`, `vctrs::list_of()`, `optional()`, `described()`), and one left
  out is text. `.data` gives the fields their columns' types, as `lm()`
  reads them, and `~ .` is every other column. Interactions, transformed
  columns and intercepts are refused, with the reason. Answers come back as
  their types, several outputs as tibble columns that `mutate()` splices in.
* `choice()`: one of a set of answers, a factor; `choice(approve = "the
  rules allow it", ...)` tells the model what each answer means.
* The column named in the formula is where `evaluate()`, the worked
  examples and `rated()` read and write the answer. In the definition a
  single output is still `result`, so the same function has the same
  version in every language.
* `ai_tool(lookup_order, "...")`: a tool's inputs are its function's
  arguments, its name the function's.
* `gepa()`: the instruction rewritten from the function's mistakes, as
  Python's `GEPA` does (`design/04-gepa.md`); `ai_trials()` gives the
  search. `set_engine("functai", method = "gepa", teacher = ...)` makes a
  tidymodels `fit()` learn its instruction, so resampling measures the
  search. Tutorials 4 and 7 use it on `gpt-5.4-nano`.
* A setting given twice (`.lm = "a", .lm = "b"`) is an error; the first
  used to win silently.
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
* The `tickets`, `field_notes` and `refunds` datasets (`refunds`: 120 refund
  requests with the decision the shop's rules give, for decision models).
* Models that measure their own probabilities (TypeSafe's Jev,
  `lm = "jev-latest"`): `predict(type = "prob")` and `augment()` give the
  measured `.pred_<level>` columns from one call a row, no votes needed;
  a fitted `ai_model()` shares those calls between its class and
  probability predictions; the call log's `confidence` is the probability
  the model gave its answer. Jev's calls run in parallel like any other.
* `calls()` has `reasoning_tokens` and `total_tokens`: Gemini's
  `output_tokens` leave its hidden reasoning out, so a cost is
  `total_tokens - input_tokens` at the output price.
