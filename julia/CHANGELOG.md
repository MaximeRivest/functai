# FunctAI.jl changelog

## 0.1.0 (unreleased)

The first Julia implementation of FunctAI, held to the same contract as the
Python, TypeScript and R packages (`../contract`): every function, score,
rating and saved-folder case, and a check against the other three languages
themselves (`../tools/crosslang.py`). Answered live through OpenAI,
Anthropic and Gemini (`tools/live.jl`, 2026-09-27: 21 of 21).

- `@ai function name(inputs...)::Type … end`: the first string says what it
  does (a docstring's `# Arguments` list describes the inputs),
  `name::T = ai"words"` declares outputs, and code after them runs on the
  answers. Types are Julia's (`@enum`, `OneOf(...)`, structs, `NamedTuple`s,
  `Vector`, `Dict`, `Union{T,Nothing}`), and answers come back as them; an
  answer that does not fit is asked again. Several outputs return a
  `NamedTuple`. Keyword inputs and defaults work as in any function.
  `?name` documents it; `FunctAI.prompt(f, …)` shows the exact request.
- `AIFunction(name, description; inputs, output(s), settings...)`: the same
  function, as data.
- Columns: broadcasting, `map` and DataFrames' `ByRow` run the calls
  `concurrency` (8) at a time, in order. `missing` in, `missing` out, with
  no call; a failed row is `missing` with one warning (`problems()`).
- Settings: `configure!` for the session, `with_settings` for a block (and
  the tasks it starts: ScopedValues), `configure(f; …)` for a copy, a
  function's own over all. `reasoning = true` is Python's `module = "cot"`.
- Tools are Julia functions (arguments from the method, words from the
  docstring); `@program` makes code that calls AI functions one call in the
  log, with theirs as its children.
- `stream(f, …)`: iterate for the answer's text, `eachevent(s)` for everything,
  `fetch(s)` for the typed value, `close(s)` to cancel; the do form prints as
  it comes.
- The call log, ratings and `rated` rows (contract/calls.md): written and
  read with Python, TypeScript and R in one folder. `rate(p, :right)`,
  `rate(p; answer = …)`, `calls(f)`.
- `evaluate` (Wilson's and Student's ranges, as every language computes
  them; an `Evaluation` is a Tables.jl table), `compare`, `exact_match`.
- Improving returns a copy: `labeled_few_shot`, `bootstrap_few_shot`,
  `random_search`, `instruction_search`, `with_demos`, `with_instructions`.
- `AIModel`: `fit(AIModel("…"), @formula(team ~ message), data)` and
  `predict`, fitting with no call (a categorical outcome's levels are the
  choice; its worked examples are the rows); an MLJ model too
  (`machine(AIModel("…"), X, y)`). StatsModels and CategoricalArrays are
  package extensions; MLJModelInterface is a dependency, as MLJ asks of
  packages that provide models.
- An AI function inside a formula (`lm(@formula(price ~ sqft + stars(description)), homes)`)
  runs its column concurrently: StatsModels broadcasts it.
- `FunctAI.save` / `FunctAI.load` (contract/saved.md): folders from any
  language load and are checked to send what was saved; `types` gives the
  fields their Julia types back.
- `FunctAI.login`, `logins`, `logout`: lm15's sign-ins, shared by every language.
- Eight tutorials (`docs/julia/`, on the website and in the manual), run on
  real models by `julia/tutorials`, and a Documenter manual (`julia/docs`):
  guides whose examples run on every build, the reference from the
  docstrings, doctests run by `Pkg.test()`. Designed from a reading of the
  documentation Julia users trust (`design/05-julia-tutorials.md`).
- Datasets: `FunctAI.tickets()`, `field_notes()`, `refunds()`, the same as
  Python's and R's, as Tables.jl tables.
- `Union{T,Missing}` answers: the model may leave them empty, and they come
  back `missing` (so a column of them is a column with holes, as Julia data
  has). `predict.(f, column)` is concurrent. `p.probabilities` is keyed by
  the answer's own type (`p.probabilities.state[damaged]`).
- `eachevent(s)` (not `events`, which Makie exports); `ai"…"` not exported
  (PromptingTools.jl exports its own); MLJ's `predict` works on AI functions.
- `model_capabilities` returns a `NamedTuple`; a mistyped setting suggests
  the one you meant; keyword inputs are read after the positional ones, as
  written; `functai.json` is indented, as Python and TypeScript write it;
  `instruction_search` shows the proposing model only the function's
  inputs and outputs, never a table's other columns.
- `gepa(f, rows; selection, teacher, budget)`: the instruction rewritten from
  the function's mistakes, as in Python, R and TypeScript (the same algorithm
  and prompts, design/04-gepa.md); returns the copy and every instruction
  tried. `AIModel(method = :gepa)` (and `:bootstrap`) learns while fitting,
  so MLJ's cross-validation measures the search. Live on the tutorials'
  jobs: refund decisions on gpt-5.4-nano, 64% to 95% and 67% to 79% on
  unseen rows in two runs; tickets, 85% to 97.5% cross-validated.
- A precompile workload: the first call of a session compiles in about 10
  seconds instead of about 60 (the rest is lm15's network code).
