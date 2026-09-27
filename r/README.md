# functai for R

**Write a function's inputs, the type of its answer and a sentence saying
what it does. A language model writes the body. You measure how well.**

```r
library(functai)
library(dplyr)

ai_config(lm = "gpt-4.1-mini", temperature = 0)

team <- ai("team", "Which team should answer this customer message?",
  message = character(),
  .returns = factor(levels = c("shipping", "billing", "product", "account")))

team("I was charged twice for order B-2210.")
#> [1] billing
#> Levels: shipping billing product account

tickets |> mutate(team = team(message)) |> count(team)
#> # A tibble: 4 × 2
#>   team         n
#>   <fct>    <int>
#> 1 shipping    22
#> 2 billing     16
#> 3 product     24
#> 4 account     18
```

An AI function is an ordinary vectorised R function, so it is a column in
any dplyr verb. The answer comes back as the type you declared (here a
factor with your levels), checked against it: a reply that does not fit is
asked again once, then refused. Each row is one model call, and up to 8 run
at once (`concurrency`): the 80 tickets above took 9 seconds with
`gpt-4.1-mini`.

The same function written in Python or TypeScript has the same version,
writes the same call log and loads from the same saved folder: this
package follows the [FunctAI contract](../contract/).

**Status: 0.1.0, not on CRAN.** It needs lmcc and lm15 for R, which are not
on CRAN either (see *Install*).

**New to this?** `vignette("getting-started", package = "functai")` starts
from dplyr and ggplot2 and ends with your first tested model: no
tidymodels knowledge assumed. `vignette("tidymodels", package = "functai")`
goes further, for tidymodels users.

## Install

```r
# install.packages("remotes")
remotes::install_github("MaximeRivest/lmcc", subdir = "r")
remotes::install_github("lm15-dev/lm15-r")
remotes::install_github("MaximeRivest/functai", subdir = "r")
```

Set the key of the provider you call (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...), in `~/.Renviron` for
example. Any model [lm15](https://lm15.dev) reaches works: `"claude-haiku-4-5"`,
`"gemini:gemini-2.5-flash"`, `"groq:openai/gpt-oss-120b"`, a local
`"ollama:qwen3.5:0.8b"`.

## Types

Inputs and answers are prototypes, the way vctrs writes types:

| you write | the model gives | you get |
|---|---|---|
| `character()` | text | a character vector |
| `integer()`, `double()`, `logical()` | a number, yes or no | that vector |
| `factor(levels = c("a", "b"))` | one of the levels | a factor with those levels |
| `record(name = character(), age = integer())`, or a zero-row `tibble()` | a record | a tibble column |
| `vctrs::list_of(.ptype = character())` | a list | a `list_of` column |
| `optional(x)` | `x`, or nothing | `NA` where there is nothing |
| `described(x, "...")` | | words the model reads about the field |

Several outputs splice in as columns; a record comes back as one tibble
column, which `tidyr::unpack()` spreads:

```r
triage <- ai("triage", "Read the support ticket.",
  message = described(character(), "the customer's own words"),
  .outputs = list(
    summary = described(character(), "one sentence, no names"),
    urgent  = logical()))

tickets |> mutate(triage(message)) |> select(id, summary, urgent)
#>      id summary                                                      urgent
#>   <int> <chr>                                                        <lgl>
#> 1     1 Customer's order A-1042 has not arrived after three weeks.   TRUE
#> ...

order_ref <- ai("order_ref", "The order the customer is talking about.",
  message = character(),
  .returns = record(order = optional(character()), days_waiting = optional(integer())))

tickets |> mutate(ref = order_ref(message)) |> tidyr::unpack(ref) |> select(id, order, days_waiting)
#>      id order  days_waiting
#> 1     1 A-1042           21
#> 2     2 <NA>             NA
#> 3     3 B-2210           NA
```

A row with a missing input is `NA` without a call. When some calls fail
(after the re-asks and the provider retries), their rows are `NA` and one
warning says how many; `ai_problems()` lists them. A single call that fails
is an error, as is any failure with `.on_error = "stop"`.

## How often is it right?

```r
ev <- evaluate(team, tickets, expected = category)
ev
#> <evaluation of team> 80 rows
#>   exact_match: 0.93  (95% interval 0.85 to 0.97)

tidy(ev)      # broom's columns: metric, estimate, conf.low, conf.high, n, failed
glance(ev)    # the first metric, in one row
augment(ev)   # every row: the data, .pred_class, .call, .error and the score
```

The interval is Wilson's for right-or-wrong scores (Student's t for others),
computed exactly as Python and TypeScript compute it. `metric =
function(row, prediction) ...` scores with your own rule.

## A tidymodels model

An AI function is also a parsnip model, so it fits, predicts, resamples and
tunes like any other, in the same workflows:

```r
library(tidymodels)

spec <- ai_model("classification", "Which team should answer this customer message?") |>
  set_engine("functai", lm = "gpt-4.1-mini")

fitted <- fit(spec, category ~ message, data = train)   # free: no call, no weights
augment(fitted, test) |> accuracy(category, .pred_class)
extract_fit_engine(fitted)                              # the AI function, callable on columns
```

`examples` (how many training rows it sees as worked examples) is tunable
with `tune()`. `set_engine("functai", samples = 5)` answers each row five
times, so `type = "prob"` gives each class's share of the answers and the
class is the majority vote. `evaluate()` scores any model, whether an AI
function, a parsnip fit or a workflow, with the same 95% interval.

**`vignette("tidymodels", package = "functai")` teaches all of it**, with
real results: zero-shot against a tf-idf model, tuning, votes as
probabilities, and a classical model trained on the language model's labels.

`predict(team, new_data)` and `augment(team, new_data)` also work on the AI
function itself: `.pred_class` for a choice, `.pred` for other answers,
`.pred_<output>` for several, then `.call` (the call's id, which is what
you rate) and `.error`.

## Making it better

```r
taught <- team |> labeled_few_shot(train, k = 8)                  # rows become worked examples
better <- team |> bootstrap_few_shot(train, teacher = "gpt-4.1")  # runs that were right become examples
evaluate(better, test, expected = category)
```

Each returns an improved copy with a new `ai_version()`; the function you
pass is unchanged. `with_demos()`, `with_instructions()` set them by hand;
`update(team, lm = "claude-haiku-4-5")` is a copy with other settings.

## Tools

```r
lookup <- ai_tool(function(order) orders[[order]], "lookup_order",
  "Look up where an order is.", order = character())

support <- ai("support", "Answer the customer, looking up their order.",
  message = character(), .tools = list(lookup))
```

## The call log and ratings

```r
ai_config(log_calls = TRUE)           # or FUNCTAI_LOG_CALLS=1

scored <- augment(team, tickets)
rate(scored$.call[3], "wrong", answer = "product")   # a person's correction
rated(team)                                           # rows with known answers, typed like team's
calls(team)                                           # every call: inputs, outputs, tokens, time
```

Every call is one line of JSON in `~/.local/share/functai/calls`, the folder
Python and TypeScript write too: each reads the others' calls and ratings.
Nothing is written unless you turn the log on.

## Across languages

```r
team <- read_ai("team/")     # a folder Python's functai.save (or TypeScript) wrote
write_ai(team, "team/")      # for Python's and TypeScript's loaders
```

`read_ai()` checks the function sends exactly what it sent where it was
saved, and has its version, before any call. It refuses, with the reason,
what only the saving language can run: code of its own around the model,
tools, a baked model.

## Settings

`ai_config(...)` for the session, `with_ai_config(code, ...)` and
`local_ai_config(...)` for a block (as withr does), dotted arguments to
`ai()` for one function (`.lm`, `.temperature`, `.max_tokens`, `.adapter`
(`"xml"`, `"chat"`, `"json"`), `.module = "cot"` (reasoning first),
`.retries`, `.concurrency`, `.router` (an `lm15::new_router()`), ...), and
`update()` for a copy.

## Not here yet

Compared with Python: streaming, modules (your code around several AI
functions, logged as one call), stateful memory, escalation to a bigger
model, the reply cache, `InstructionSearch`, baking (training your own
weights), and templates written as R functions. Python cannot yet load a
function saved from R or TypeScript (R and TypeScript load Python's).

## Developing

```bash
r/check          # installs lmcc and lm15 for R from ../lmcc and ../lm15-dev, then the tests
../check         # every language, and each against the others
```

`r/tools/env.nix` is the R environment the checks use on NixOS
(`nix shell --impure -f r/tools/env.nix`); `r/tools/live.R` calls real
models (costs cents).
