# FunctAI for R, in eight tutorials

*Functions whose body is a language model, used like any other R function: in `mutate()`, measured with intervals, chosen by cost, trusted with decisions, fitted in tidymodels, and kept honest in use.*

Each tutorial starts from a question about real data, builds the answer one small step at a time, and ends with what it cost. Each runs top to bottom in a fresh R session, on models current in September 2026 (`gpt-6-luna` for everyday work; `gpt-6-sol`, `claude-sonnet-5`, `claude-haiku-4-5`, `gemini-3.8-flash`, `gemini-3.1-flash-lite` and `gpt-5.4-nano` where a comparison needs them; and TypeSafe's
`jev-latest`, a model built for decisions, in tutorials 5 and 6). Every output on these pages is from a real run.

| | Tutorial | You will | Cost of a run |
|---|---|---|---|
| 1 | [Your first AI function](01-first-function.md) | sort 80 customer messages into teams, check them, improve them | under 1¢ |
| 2 | [Answers you can compute with](02-types.md) | turn bird-survey notes into typed columns: factors, counts that may be missing, records, lists | about 1¢ |
| 3 | [Is it right?](03-is-it-right.md) | measure with intervals, baselines, a confusion matrix, and run-to-run variation | under 1¢ |
| 4 | [Making it better without fooling yourself](04-making-it-better.md) | improve a refund decision with rules, examples and a teacher, on three piles of rows; and when nobody wrote the rules, let a stronger model write them from the mistakes (GEPA) | about 10¢ |
| 5 | [Choosing a model](05-choosing-a-model.md) | compare eight models (TypeSafe's Jev among them) on accuracy, cost and speed, with paired tests and a rule | about 30¢ |
| 6 | [Decision models](06-decisions.md) | approve, deny or ask a person: costs of mistakes, rules in R, probabilities from votes and from Jev, a decision tree | about 7¢ |
| 7 | [AI functions in tidymodels](07-tidymodels.md) | fit, resample, tune and score a language model like any parsnip model; a fit that learns its instruction from its mistakes | about 5¢ |
| 8 | [Living with it](08-living-with-it.md) | tools, the call log, people's corrections, versions, saving | under 1¢ |

The same series exists [for Python](../tutorials/index.md), on the same
datasets, where baking a model you own takes tutorial 7's place.

## Before you start

You need R 4.1 or later, and a key for at least one model provider (OpenAI's, for most of the series) in your `~/.Renviron`:

```{.r .no-run}
install.packages(c("remotes", "dplyr", "ggplot2", "tidymodels", "textrecipes", "glmnet", "rpart.plot"))
remotes::install_github("MaximeRivest/lmcc", subdir = "r")
remotes::install_github("lm15-dev/lm15-r")
remotes::install_github("MaximeRivest/functai", subdir = "r")
```

Tutorial 1 assumes you know dplyr and a little ggplot2. Tutorial 7 assumes tidymodels. Nothing else is assumed; each tutorial says at the top what it covers, with a three-question check so you can skip what you already know.

## How these were made

The series was designed after reading the R tutorials people recommend most (R for Data Science, tidymodels' Get Started, Tidy Modeling with R, Supervised Machine Learning for Text Analysis, Advanced R, and the LLM packages' own guides, among others) for how they open, teach, and what they leave out. The notes are in [`design/02-r-tutorials.md`](https://github.com/MaximeRivest/functai/blob/master/design/02-r-tutorials.md). To run them yourself from a checkout: `r/tutorials` (all eight, or name some), which writes every output back into these pages.
