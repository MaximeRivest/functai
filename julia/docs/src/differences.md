# Where Julia differs

FunctAI is one contract and four native implementations. What must agree does: the same function has the same version and signature, sends the same bytes, writes the same call log and ratings, scores the same way and saves to the same folder in Python, TypeScript, R and Julia. How the code looks is each language's own. Here is where Julia's differs, and why.

## By design

- **Types are Julia's.** `@enum`s, structs, `NamedTuple`s, `Union{T,Missing}`, `Vector`s. They are written as the JSON Schema Python writes for the same type, so `mood` with an `@enum` answer has the version of Python's `mood` with a `Literal[...]` answer.
- **Several outputs return all of them**, as a `NamedTuple` that destructures and becomes columns (`ByRow(f) => AsTable`). Python returns the last one. The call log still names the last as the answer.
- **Input descriptions come from the docstring's `# Arguments` list**, the way Julia documents functions; output descriptions from `ai"words"`.
- **`reasoning = true`** is Python's `module = "cot"`: `module` is a keyword Julia can't parse as a setting's name.
- **Settings follow Julia's conventions**: `configure!` changes the session (the `!`), `with_settings` covers a block and the tasks it starts (ScopedValues), `configure(f; …)` returns a copy.
- **A column keeps the answers it paid for.** A failed row is `missing` with one warning (`FunctAI.problems()`), where plain Julia would stop at the first error; when every row fails, it throws.
- **Improving returns a copy** (`labeled_few_shot(f, rows)`), never changes `f`: Julia's convention for a function without `!`.
- **Code of your own is versioned by its parsed form**, not its text: reformatting or editing comments doesn't make a new version. A Julia function with code of its own never shares a version with another language's (its code is Julia).
- **Streams are iterators**: `for piece in stream(f, x)`, `eachevent(s)` (like `eachline`), `fetch(s)`, `close(s)`.

## Names shared with other packages

Julia refuses to guess when two packages loaded with `using` export different functions under one name. FunctAI's exports avoid the common ones, except where the name is the right one:

- `predict` is StatsAPI's (the same function GLM and StatsModels extend). MLJ has its own `predict`; FunctAI adds its methods to MLJ's too, but with `using FunctAI, MLJ` the bare name is ambiguous. Load one with `import`, as [tutorial 7](tutorials/07-mlj-and-formulas.md) does.
- `evaluate` is FunctAI's; MLJ also exports one. Same remedy.
- `FunctAI.save`, `FunctAI.load`, `FunctAI.login` are not exported: `save` and `load` are FileIO's names.
- `ai"…"` is not exported either: PromptingTools.jl exports its own `ai"…"`. Inside `@ai` it's read from your code before anything runs, so FunctAI's needs no import.

## Not in Julia yet

- **Saving code of your own or tools** from Julia (a saved folder carries no Julia code yet).
- **The reply cache, stateful memory, escalation** (`escalate_to`; a [`@program`](@ref) does it by hand), **votes** (R's `samples`), and **baking** (training your own model).
- **Probabilities per class from `AIModel`**: predictions are deterministic. Calibrated probabilities come from a model that measures them (TypeSafe's Jev): `predict(f, x).probabilities`.
- **A program's version** follows the AI functions and programs it names, not plain Julia functions it calls.
