# FunctAI for Julia, in eight tutorials

*Functions whose body is a language model, used like any other Julia function: broadcast over a column, measured with intervals, chosen by cost, trusted with decisions, fitted in MLJ and formulas, and kept honest in use.*

Each tutorial starts from a question about real data, shows where it is going, builds the answer one small step at a time, and ends with what it cost. Each runs top to bottom in a fresh Julia session, on models current in September 2026 (`gpt-6-luna` for everyday work; `gpt-6-sol`, `claude-sonnet-5`, `claude-haiku-4-5`, `gemini-3.8-flash`, `gemini-3.1-flash-lite` and `gpt-5.4-nano` where a comparison needs them; and TypeSafe's `jev-latest`, a model built for decisions, in tutorials 5 and 6). Every output on these pages is from a real run.

| | Tutorial | You will | Cost of a run |
|---|---|---|---|
| 1 | [Your first AI function](01-first-function.md) | sort 80 customer messages into teams with `@ai` and a dot, check them, improve them | under 1¢ |
| 2 | [Answers you can compute with](02-types.md) | turn bird-survey notes into typed columns: `@enum`s, `Union{Int,Missing}`, structs, vectors | about 1¢ |
| 3 | [Is it right?](03-is-it-right.md) | measure with intervals, baselines as plain Julia functions, a confusion table, run-to-run variation | under 1¢ |
| 4 | [Making it better without fooling yourself](04-making-it-better.md) | improve a refund decision with rules, examples and a teacher, on three piles of rows; and when nobody wrote the rules, let a stronger model write them from the mistakes (GEPA) | about 9¢ |
| 5 | [Choosing a model](05-choosing-a-model.md) | compare eight models (TypeSafe's Jev among them) on accuracy, cost and speed, with paired tests and a rule | about 30¢ |
| 6 | [Decision models](06-decisions.md) | approve, deny or ask a person: costs of mistakes, rules as a tested Julia function, probabilities from Jev, a second opinion as a `@program`, a decision tree | about 3¢ |
| 7 | [AI functions in MLJ and formulas](07-mlj-and-formulas.md) | fit, cross-validate and tune a language model as an MLJ model; a fit that learns its instruction from its mistakes; an AI function as a feature in a GLM formula | about 10¢ |
| 8 | [Living with it](08-living-with-it.md) | tools, streaming, the call log, people's corrections, versions, saving | under 1¢ |

The same series exists [for Python](../tutorials/index.md) and [for R](../r/index.md), on the same datasets. A function's version, its call log and its saved folder are the same in all of them: ratings made in R pool with calls made in Julia, and a function saved in Python loads here.

## Before you start

You need Julia 1.10 or later, and a key for at least one model provider (OpenAI's, for most of the series) in the environment Julia starts with (`export OPENAI_API_KEY=sk-...`, or `ENV["OPENAI_API_KEY"] = "sk-..."` in `~/.julia/config/startup.jl`). FunctAI and the two packages it uses are not in the General registry yet; add them in this order:

```{.julia .no-run}
using Pkg
Pkg.add(url = "https://github.com/MaximeRivest/lmcc", subdir = "julia")
Pkg.add(url = "https://github.com/lm15-dev/LM15.jl")
Pkg.add(url = "https://github.com/MaximeRivest/functai", subdir = "julia")
Pkg.add(["DataFrames", "CairoMakie", "HypothesisTests", "MLJ", "NaiveBayes", "MLJNaiveBayesInterface",
         "GLM", "StatsModels", "CategoricalArrays", "DecisionTree"])
```

Tutorial 1 assumes you know a little DataFrames.jl. Tutorial 7 assumes MLJ's basics. Nothing else is assumed; from tutorial 2 on, each says at the top what it covers, with a three-question check so you can skip what you already know.

For the reference (every function, with its documentation) and short how-to guides, see the [FunctAI.jl manual](https://github.com/MaximeRivest/functai/tree/master/julia/docs), built with Documenter from the package's docstrings; `?name` in the REPL shows the same text.

## How these were made

The series was designed after reading the documentation Julia users trust (DataFrames' "First steps", MLJ's and Makie's "Getting started", JuMP's tutorials, Flux's one-minute quickstart, GLM's manual, the Julia manual's rules for docstrings, and PromptingTools.jl, the most used LLM package in Julia) for how they open, teach, and what they leave out. The notes are in [`design/05-julia-tutorials.md`](https://github.com/MaximeRivest/functai/blob/master/design/05-julia-tutorials.md). To run them yourself from a checkout: `julia/tutorials` (all eight, or name some), which writes every output back into these pages.
