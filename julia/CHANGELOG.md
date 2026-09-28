# FunctAI.jl changelog

## 0.1.0 (unreleased)

### Stage 1 foundations (contract at c1e5063, design/08)

- **The call log is format 2**, and both formats are read. Records carry
  `program.interface`, `saw` (`[]`: Julia shows no earlier call yet),
  `request_hash` on exchanges, `described` for values with no JSON form, and
  `journal` when a required journal did not confirm the end. `rated` reads
  `omitted` and `described`, and pools calls by interface (`rated(f)` passes
  `f`'s), so turning reasoning on no longer splits a function's ratings.
- **`log_content` per field**: `(transcript = false,)`, `Dict("*" => false,
  "question" => true)`. A value is written only when no layer drops it (a
  function's own setting, each `with_settings` block, `configure!`,
  `FUNCTAI_LOG_CONTENT=0`); dropping any field drops the reasoning and tool
  calls FunctAI adds and every request, reply and error message. A
  function's own map naming a field it lacks, or a key that is not a name
  anywhere, is refused (`LogContentError`).
- **Interfaces** (contract/programs.md): `FunctAI.interface(f)` for AI
  functions and programs, `FunctAI.interface_signature(f)` (the log's
  `program.interface`). An AI function's input with a default is optional:
  its default, evaluated when the function is defined, is in its interface
  and sent when left out, and left out of the lmcc signature
  (`functions/12`). The interface is checked at definition, after lmcc's
  check (`InterfaceError` `interface-malformed`). `AIFunction(…; defaults)`.
- **`@program` declares its interface** from its typed arguments and return
  type (untyped or `Any`: opaque; `outputs = (a = T, …)` for several; a
  literal default in the interface), checked on every call, inputs before
  the code runs and outputs when it returns (`InterfaceError`
  `interface-input` / `interface-output`, naming the field). A program's
  version includes its interface. `AIProgram(name, code; interface)` builds
  one from an interface as data. **Breaking**: a `@program` call given a
  value that does not fit its declared type now throws `InterfaceError`
  instead of Julia's `MethodError`, and a typed argument is converted to its
  type (`5.0` to an `Int` argument is `5`).
- **Saved folders**: every node written has its `interface`; loading an AI
  function takes its optional inputs and their defaults from it, and checks
  it against the signature (`saved-differs`, `interface-malformed`);
  `FunctAI.describe(path)` describes any node without loading it
  (`saved-no-interface` for a module saved before interfaces). The
  manifest's schema is checked, as code (`saved-malformed`).
- **Events are format 2**: one log per call tree, numbered once (`tree`,
  `writer`, `seq`, `after` a `FunctAI.Position`, `at`), with a `request`
  event per model request; a stream opened on a call inside a program shows
  the tree's numbers from that call (law 7). Events as data:
  `FunctAI.Event(json)`, `replay`, `resume`, `Follower` with `receive!` and
  `recover!` (stale, duplicate, next, rewind, loss, unknown format; resuming
  in place only from a source that holds what the reader was shown), and the
  kept form (`kept_event`, `kept_log`).
- **Observers and journals** as settings (`observers = [f]`, `journal =
  store`, `FunctAI.Journal(store; required = true)`, `journal = false`):
  observers add up over the layers and get the kept form; one journal per
  tree, with the host's holding (`JournalError` `journal-policy`, and
  `journal-scope` for a required one set inside a tree). A writer appends the
  kept form in order on its own task, sends again what is not confirmed, and
  for a required journal waits at the start, before each tool and at the end
  (`journal-barrier`: the tool does not run; `journal-end` holding the
  outcome, its event and tree; `FunctAI.settle`). `FunctAI.MemoryStore`
  keeps logs by the store rules (`claim!`, `keep!` of an event or a batch,
  `events_after`).
- **What a call saw**: `FunctAI.saw(records, id)` and `keeps_saw` read any
  language's records (`not-recorded`, `missing-call`, `unknown-key`,
  `saw-cycle`, `not-kept`, `turn-invalid`).
- Harnesses for every stage-1 case the contract assigns to Julia:
  `functions/`, `rated/`, `saved/` (with `describe` and `sends`),
  `programs/`, `content/`, `saw/` (`read`), `events/` (`replay`, `follow`,
  `kept`, `store`, `journal`, `receivers`).

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
