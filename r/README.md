# functai for R

**Write it like a model formula: what comes out, `~`, what goes in, and a
sentence saying what it does. A language model writes the body. You
measure how well.**

```r
library(functai)
library(dplyr)

ai_config(lm = "gpt-4.1-mini", temperature = 0)

team <- ai(team ~ message, "Which team should answer this customer message?",
  team = choice("shipping", "billing", "product", "account"))

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

`team ~ message` reads as it does in `lm()`: *team, from the message*.
The result is an ordinary vectorised R function, `team(message)`, so it is
a column in any dplyr verb. The answer comes back as the type you declared
(here a factor with your levels), checked against it: a reply that does not
fit is asked again once, then refused. Each row is one model call, and up to 8 run
at once (`concurrency`): the 80 tickets above took 9 seconds with
`gpt-4.1-mini`.

The same function written in Python, TypeScript or Julia has the same
version, writes the same call log and loads from the same saved folder:
this package follows the [FunctAI contract](https://github.com/MaximeRivest/functai/tree/master/contract).

**Status: 0.1.0, not on CRAN.** Neither are lmcc and lm15 for R, which it
needs; installing from GitHub brings them (see *Install*).

**Learning it?** [Eight tutorials](https://maximerivest.github.io/functai/r/index.html), from a first function
to decision models, model choice by cost, tidymodels and life in
production, each run end to end on current models for under a dollar.
[The manual](https://maximerivest.github.io/functai/r/manual/index.html) has every function's help page, the
vignettes and the news.

**New to this?** [Getting started](https://maximerivest.github.io/functai/r/manual/articles/getting-started.html)
starts from dplyr and ggplot2 and ends with your first tested model: no
tidymodels knowledge assumed. [tidymodels](https://maximerivest.github.io/functai/r/manual/articles/tidymodels.html)
goes further, for tidymodels users. Both are also vignettes, installed when
you install with `build_vignettes = TRUE`.

## Install

```r
# install.packages("remotes")
remotes::install_github("MaximeRivest/functai", subdir = "r")   # and lmcc and lm15, from GitHub
```

Set the key of the provider you call (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...), in `~/.Renviron` for
example. Any model [lm15](https://lm15.dev) reaches works: `"claude-haiku-4-5"`,
`"gemini:gemini-2.5-flash"`, `"groq:openai/gpt-oss-120b"`, a local
`"ollama:qwen3.5:0.8b"`.

## Writing one

The formula names the outputs on the left and the inputs on the right,
joined by `+`. Then each field is a line of a codebook: **a sentence
describes it, a type types it**, and a field you leave out is text.

```r
triage <- ai(summary + urgent ~ message, "Read the support ticket.",
  message = "the customer's own words",
  summary = "one sentence, no names",
  urgent  = logical(),
  .name = "triage")

tickets |> mutate(triage(message)) |> select(id, summary, urgent)
#>      id summary                                                      urgent
#>   <int> <chr>                                                        <lgl>
#> 1     1 Customer's order A-1042 has not arrived after three weeks.   TRUE
#> ...
```

One output names the function after itself (`team`); several come back as
tibble columns that `mutate()` splices in, and the function needs a
`.name`.

**Types from a table**, as `lm(y ~ x, data = ...)` reads them: with
`.data`, each field has its column's type (a factor is a choice of its
levels, a number a number, a logical yes or no), and `decision ~ .` means
every other column. Only the column types are read, never the rows.

```r
refund <- ai(decision ~ message + price + days_since_delivery + final_sale,
  "Should the shop refund this request?",
  .data = refunds,
  price = "in dollars")
```

| you write | the model gives | you get |
|---|---|---|
| nothing, or a sentence | text | a character vector |
| `integer()`, `double()`, `logical()` | a number, yes or no | that vector |
| `choice("a", "b")` | one of them | a factor with those levels |
| `record(name = "...", age = integer())` | a record | a tibble column |
| `vctrs::list_of(.ptype = character())` | a list | a `list_of` column |
| `optional(x)` | `x`, or nothing | `NA` where there is nothing |
| `described(x, "...")` | | a type with words the model reads |
| `defaults_to("kind", x)` | | an input the caller may leave out, sent as `"kind"` then |

An input with a default is an argument with a default, as in any R
function: `reply <- ai(reply ~ message + tone, "Answer the customer.", tone
= defaults_to("kind", choice("kind", "brief")))` is called `reply(message)`
or `reply(message, tone = "brief")`. `ai_interface(reply)` is what it takes
and gives, as every language describes it; a function whose interface no
language could read back (a default that does not fit its type, a field
name that is not an identifier) is refused when it is defined.

A choice can say what each answer means, and the model reads it:

```r
item_state <- ai(state ~ message, "What state is the item in?",
  state = choice(
    unopened = "still sealed, never opened",
    used     = "used for a while, works fine, no longer wanted",
    damaged  = "broken or damaged when it arrived",
    faulty   = "worked at first, then failed in normal use"))
```

Printing a function shows it as you wrote it: the formula, the sentence,
and each field with its type and words.

The formula means what it means for any model (*this column, from those*),
so the same formula fits a logistic regression, `ai()`, or `ai_model()` in
tidymodels. What only a regression has, interactions (`a * b`),
transformed columns (`log(x)`) and intercepts, `ai()` refuses and says why:
a language model reads all its inputs together, as they are.

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

With no `expected`, the right answers are the column named like the
formula's output (`evaluate(refund, refunds)` scores against `decision`).
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

**[The tidymodels vignette](https://maximerivest.github.io/functai/r/manual/articles/tidymodels.html) teaches all of it**, with
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
learned <- team |> gepa(train, teacher = "gpt-6-sol")             # the instruction, rewritten from mistakes
evaluate(learned, test, expected = category)
```

`gepa()` shows a stronger model the function's answers with feedback in
words ("wrong: the right answer is billing") and keeps the best
instruction it writes, chosen on rows it never shows the teacher
(`ai_trials()` is the search; `design/04-gepa.md` says how it differs from
the paper's GEPA). On refund decisions it took `gpt-5.4-nano` from 68% to
93% right on rows it never saw (tutorial 4). In tidymodels,
`set_engine("functai", method = "gepa")` makes `fit()` learn it, and
resampling measure it.

Each returns an improved copy with a new `ai_version()`; the function you
pass is unchanged. `with_demos()`, `with_instructions()` set them by hand;
`update(team, lm = "claude-haiku-4-5")` is a copy with other settings.

## Tools

```r
lookup_order <- function(order) orders[[order]]

support <- ai(reply ~ message, "Answer the customer, looking up their order.",
  .tools = list(ai_tool(lookup_order, "Look up where an order is.", order = "like A-1042")))
```

A tool's inputs are the function's arguments, and its name the function's.

## The call log and ratings

```r
ai_config(log_calls = TRUE)           # or FUNCTAI_LOG_CALLS=1

scored <- augment(team, tickets)
rate(scored$.call[3], "wrong", answer = "product")   # a person's correction
rated(team)                                           # rows with known answers, typed like team's
calls(team)                                           # every call: inputs, outputs, tokens, time
```

Every call is one line of JSON in `~/.local/share/functai/calls`, the folder
Python, TypeScript and Julia write too: each reads the others' calls and
ratings (call log format 2, and format 1). Nothing is written unless you
turn the log on.

`log_content` says which values a call's line keeps, per field, and only
ever removes: a value is written when no layer (the function's own
setting, each `with_ai_config()` block, `ai_config()`,
`FUNCTAI_LOG_CONTENT=0`) drops it. `c(transcript = FALSE)` keeps
everything but the transcript; `c("question")` keeps only the question,
the host's safe list. The line then says what it left out, keeps the
sizes, and keeps no request or reply. `rated()` leaves out calls whose
inputs were not kept, and pools calls that record the same data (turning
reasoning on, or changing a default, starts no new pool).

## Across languages

```r
team <- read_ai("team/")     # a folder Python's functai.save wrote, or TypeScript's, or Julia's
write_ai(team, "team/")      # for TypeScript's and Julia's loaders
```

`read_ai()` checks the function sends exactly what it sent where it was
saved, and has its version, before any call. It refuses, with the reason,
what only the saving language can run: code of its own around the model,
tools, a baked model. `ai_interface("team/")` describes a saved program
without loading or running it, a Python module included.

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
function saved in R (TypeScript and Julia can). The
[home page](https://maximerivest.github.io/functai/#what-each-language-has) compares the four languages.

## Developing

From the repository, with lmcc and lm15-dev checked out beside it:

```bash
r/check          # installs lmcc and lm15 for R from ../lmcc and ../lm15-dev, then the tests
./check          # every language, and each against the others
r/tutorials      # runs the eight tutorials on real models, writes their outputs (about 40 cents)
```

`r/tools/env.nix` is the R environment the checks use on NixOS
(`nix shell --impure -f r/tools/env.nix`); `r/tools/live.R` calls real
models (costs cents).
