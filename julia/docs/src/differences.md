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
- **Training** (a head or a generative student). Julia writes the training examples every trainer reads, byte for byte Python's (`FunctAI.bake_examples`, `export_examples`), and calls a student trained anywhere through the server that serves it (`FunctAI.baked`); Python's `bake` (here, on Tinker, on Prime) or any trainer does the training. A baked *head* (a classifier) runs only in Python.
- **Votes** (R's `samples`).
- **Probabilities per class from `AIModel`**: predictions are deterministic. Calibrated probabilities come from a model that measures them (TypeSafe's Jev): `predict(f, x).probabilities`.
- **A program's own settings**: a `@program` takes no settings of its own (`plugins`, `log_content`, …); set them around its calls (`with_settings`) or on the AI functions it calls.
- **A program's version** follows the AI functions and programs it names (globals of its module, and the ones it captured where it was written), not plain Julia functions it calls.

## By design, for the stages Python built first

- **`train_test`** is Python's `functai.split` (`split` is Base's).
- **`merge!(chat, branches, judge)`** is Python's `chat.merge(...)`; `turns(chat; all = true)` its `chat.all_turns()`; `approve!`, `deny!`, `resume!`, `abandon!`, `stop!` its turn methods; `on!(f, plugin, hook)` its decorators. A conversation's turn is `FunctAI.turn(chat, id)` (`turn` is LM15's export).
- **A baked student is called through a server** (vLLM's OpenAI-compatible chat endpoint, with the student's chat template and thinking off), not in this process: Julia has no Hugging Face tokenizer with chat templates. The messages are the ones it was trained on, so the tokens are too, when the server serves the baked folder's tokenizer.
- **Refusals of a bake carry codes** (`baked-fixed`, `baked-derived`, `baked-changed`, `baked-format`, `bake-rows`): Python's `BakeError` says the same in its message.
- **A served program runs on HTTP.jl** (`serve`); the same routes, keys, views and errors as Python's.
