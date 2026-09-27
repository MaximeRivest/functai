# 05: Teaching FunctAI for Julia: what the best Julia documentation does, and the series it gave

Written 2026-09-27, before the eight tutorials in `docs/julia/` and the
Documenter site in `julia/docs/`. Like `02-r-tutorials.md` for R, it reads
the documentation Julia users meet and trust, at its opening and its
structure: DataFrames.jl ("First steps", its index), MLJ ("Getting
started", "Learning MLJ"), Makie ("Getting started"), JuMP ("Getting
started with JuMP", its index and `make.jl`), Flux ("A neural network in
one minute"), Turing ("Getting started"), GLM's manual, DataFramesMeta and
TidierData (their index pages), PromptingTools.jl (the most used LLM
package in Julia), and the Julia manual's own rules for docstrings
("Writing Documentation"). Sources read from their repositories on
2026-09-27.

## What each one does

| Resource | Voice | How it opens | How it teaches | What it covers well | What it glosses over |
|---|---|---|---|---|---|
| **DataFrames, First steps** | Plain, exhaustive | Installing, running the tests, checking the version | One constructor, then one operation after another, each a REPL exchange with its printed table | Every way to do a thing; the `source => function => target` mini-language | Why; a task to finish |
| **MLJ, Getting started** | Precise, a little formal | "This page introduces some MLJ basics, assuming familiarity with machine learning" | `@repl` blocks on the iris data: load, choose a model, `evaluate`, then fit and predict | Evaluation as the first act, not the last; the same verbs for every model | Anything but iris; what a result means for a decision |
| **Makie, Getting started** | Warm, second person | **The figure you will make, shown first**, then "you only need an internet connection" | Builds that figure one call at a time, `!!! info` boxes for asides | The mental model (Figure, Axis, plot) before the catalogue | Data |
| **JuMP, Getting started** | Teacherly, exact | **Learning intentions** in three bullets, then "what is JuMP / a solver" | **The complete code first**, then "step by step", each line again with prose; Literate files run on every build | Vocabulary before syntax; the reader always sees the whole | Real problems (they come in later tutorials) |
| **Flux, A neural network in one minute** | Brisk | "If you have used neural networks before… If you haven't, then [a gentler page]" | One commented block that runs as is, then "features to note" | Showing the whole loop at once; routing readers by what they know | Explanation, deliberately |
| **Turing, Getting started** | Neutral | Installation and supported versions | One small model, sampled and plotted | Setting expectations about versions and platforms | Why the model is right |
| **GLM manual** | Reference | Installation, then the two ways to fit | Signatures, argument lists, one worked example per family | Exactness; formulas vs matrices | Choosing a model |
| **DataFramesMeta / TidierData** | Friendly, opinionated | What the package is *for*, and whom (Tidier: "an R user's love letter") | One macro per section, with small tables | Meeting readers where they come from (dplyr) | Measurement |
| **PromptingTools.jl** | Chatty | Getting a key, then `ai"What is the capital of France?"` | Features one after another, with printed costs ("Be in control of your spending!") | Cost from the first call; ease | **Whether the answers are right**: nothing measures them |
| **Julia manual, Writing Documentation** | Normative | The rules | Nine numbered rules for a docstring | The signature first, one imperative line, `# Examples` as **doctests**, "See also" | — |

Documenter is the common ground: every package above builds its site with
it, runs `jldoctest` blocks as tests, generates the reference from
docstrings (`@docs`), and organizes pages as tutorials, manual or guides,
and API (JuMP's `tutorials/`, `manual/`, `api/`; DataFrames' "Manual" and
"API"; MLJ's many pages under "Getting started" and "Learning MLJ").

## What to take from them

1. **Show the whole thing first** (Makie's figure, JuMP's complete code,
   Flux's one block), then build it step by step. Each tutorial opens with
   its situation and the few lines it ends with.
2. **Learning intentions, then vocabulary** (JuMP): "you will…" bullets at
   the top; a term (a token, a version, a layout) is explained where it
   first matters.
3. **Code, then what Julia prints.** Julia readers read REPL output: a
   `DataFrame`'s header with column types, a `Vector{Team}`, an `@enum`
   value. Every cell's output is the real `show` of a real run.
4. **Julia's own words for Julia's own ideas**: broadcasting (`team.(xs)`),
   `missing`, multiple dispatch (an `AIFunction` *is* a `Function`), `do`
   blocks, `NamedTuple`s, `@enum`, structs, `transform`/`ByRow`, formulas,
   MLJ machines. No R or Python idiom translated word for word.
5. **Evaluation is not the last chapter** (MLJ's first act). And what the
   Julia LLM documentation leaves out (PromptingTools counts cost but
   never correctness) is what this series is for: every tutorial measures.
6. **Route readers** (Flux): the index says who each tutorial is for, and
   what it assumes (DataFrames for all; MLJ for 7).
7. **Docstrings by the manual's rules**, doctests where no model is
   needed, and a reference generated from them, so the docs cannot drift
   from the code.

From the R study (`02-r-tutorials.md`), unchanged because they are about
the subject, not the language: a question about real data first; the trap
before the fix; always a baseline; a score is a proportion with an
interval; the cost in dollars from the call log; "Your turn" tasks; one
mental model per tutorial; the whole game last.

## The series

The same eight stories as Python and R, on the same datasets, translated
by purpose:

| # | Tutorial | Julia's way |
|---|---|---|
| 1 | Your first AI function | `@ai` with an `@enum`, `team.(df.message)`, `transform`, a Makie bar chart |
| 2 | Answers you can compute with | Julia types as the answer's promise: `Int`, `Union{Int,Nothing}`, `@enum`, a struct, `Vector`; `missing` |
| 3 | Is it right? | `evaluate` with intervals; baselines as plain Julia functions; a confusion table with `combine`/`unstack`; run-to-run variation |
| 4 | Making it better without fooling yourself | three piles; rules, `labeled_few_shot`, a teacher; `compare`; `gepa`, the instruction rewritten from the mistakes |
| 5 | Choosing a model | eight models (Jev among them) by accuracy, cost and speed; `compare` in pairs; a rule in code |
| 6 | Decision models | the model reads, Julia decides (a plain function); expected cost from Jev's probabilities |
| 7 | AI functions in MLJ and formulas | `machine(AIModel(…), X, y)`, `evaluate!` with cross-validation, `examples` tuned; an AI column inside a GLM formula |
| 8 | Living with it | tools as Julia functions, streaming, the call log, ratings, versions, saving and loading across languages, `@program` |

What differs from R, stated: no votes (`samples = k`) in Julia 0.1.0, so
tutorial 6 takes Python's route (why a model's own confidence is not a
probability, then Jev's measured ones); tutorial 7 cross-validates `AIModel(method = :gepa)` as R's
tutorial 7 resamples its GEPA fit; plots are Makie (CairoMakie), the
package the *Julia Data Science* book and Makie's docs teach.

## The site

`julia/docs/` is a Documenter site, as every Julia package has: Home, the
eight tutorials (the pages of `docs/julia/`, copied at build with their
recorded outputs), how-to guides, "Where Julia differs", and the reference
generated from the docstrings (`checkdocs = :exports`: every exported name
documented). Docstring examples that need no model are `jldoctest`s, run
by `Pkg.test()` (`Documenter.doctest`). The tutorials run once, on real
models, with `julia/tutorials` (each page in a fresh Julia process, its
outputs written back), like `r/tutorials`; the site build runs nothing that
costs money.
