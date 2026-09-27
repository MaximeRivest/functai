# FunctAI.jl

Typed Julia functions whose body a language model writes. You write the
signature; a model writes the body. Then you run it over a column, measure
how often it is right, have people rate its calls, improve it, and save it.

```julia
using FunctAI

@enum Mood happy unhappy mixed

@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end

mood("It broke after one day.")        # unhappy::Mood
df.mood = mood.(df.review)             # the whole column, 8 calls at a time
evaluate(mood, labelled)               # how often it is right, with a 95% range
```

**Learn it** with [eight tutorials](../docs/julia/index.md) (from a first
function to decisions, MLJ and living with a function in use; every output
from a real run) and the [manual](docs/) (guides and the reference, built
with Documenter: `julia --project=julia/docs julia/docs/make.jl`).

FunctAI exists in Python, TypeScript, R and Julia, held together by one
[contract](../contract): the same function has the same version, writes the
same call log and saves to the same folder in every language. A function
improved in Python loads here and sends the same bytes.

## Install

Julia 1.10 or newer. FunctAI is not in the General registry yet, and
neither are the two packages under it (lmcc, lm15), so for now it runs
from a checkout of this repository next to `lmcc` and `lm15-dev`:

```julia
using Pkg
Pkg.develop(path = "functai/julia")    # Project.toml's [sources] finds lmcc and lm15
```

Set the key of the provider you use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `GROQ_API_KEY`, …), or sign in once with
`FunctAI.login("claude")` (subscriptions and keys are saved in lm15's file,
shared by every language). With no model named, FunctAI uses the first
provider it finds a key for.

## A function

The body of an `@ai` function says what it does, then (optionally) which
outputs the model writes, then (optionally) your own code.

```julia
@ai function triage(ticket::String; product::String = "the shop")
    """
    Read a support ticket.

    # Arguments
    - `ticket`: the customer's own words
    """
    summary::String = ai"one sentence, no names"
    minutes::Int    = ai"minutes to fix"
end

(; summary, minutes) = triage("I was charged twice for one order.")
```

- **The description** is the first string. A docstring's `# Arguments` list
  describes the inputs, the way Julia documents functions anyway.
- **Outputs** are `name::Type = ai"words"`. With several, calling returns
  them all as a `NamedTuple`, which destructures and becomes columns
  (`ByRow(triage) => AsTable`). With none, the one output is the return
  type (`String` when none is written).
- **Types** are Julia's: `String`, numbers, `Bool`, an `@enum`,
  `OneOf(:yes, :no)` (a choice without an `@enum`), `Vector{T}`,
  `Dict{String,T}`, `NamedTuple`s, your own structs, and `Union{T,Nothing}`.
  Answers come back as those types; an answer that does not fit (a choice
  outside the list, a missing field) is asked again.
- **Code of your own** after the outputs runs on them, and its value is
  what the call returns:

  ```julia
  @ai function price(item::String)::Float64
      "Estimate the price in US dollars."
      usd::Float64 = ai"the price"
      round(usd; digits = 2)
  end
  ```
- **`missing` in, `missing` out**, without a call (and without paying for one).
- `?mood` shows what it does; `FunctAI.prompt(mood, "…")` shows the exact
  request, as a conversation, without sending it.

Without the macro, the same function is data:

```julia
mood = AIFunction("mood", "How does the customer feel about what they bought?";
                  inputs = (review = String,), output = Mood)
```

## Settings

```julia
@ai lm = "claude-haiku-4-5" temperature = 0 function mood(review::String)::Mood … end

FunctAI.configure!(lm = "gpt-4.1-mini")          # for the session
with_settings(lm = "gemini:gemini-2.5-flash") do  # for a block (and the tasks it starts)
    mood.(reviews)
end
fast = configure(mood; lm = "groq:openai/gpt-oss-120b")   # a copy with other settings
```

A function's own settings win over `with_settings`, which wins over
`configure!`. Among them: `lm`, `temperature`, `max_tokens`, `adapter`
(`:xml`, `:chat`, `:json`), `template`, `reasoning = true` (the model writes
its reasoning first), `tools`, `retries`, `concurrency`, `log_calls`.
`?configure!` lists them all.

## Tables

An AI function is a function, so every table tool in Julia takes it:

```julia
df.mood = mood.(df.review)                                  # broadcasting
transform(df, :review => ByRow(mood) => :mood)              # DataFrames
transform(df, :ticket => ByRow(triage) => AsTable)          # several outputs, several columns
lm(@formula(price ~ sqft + stars(description)), homes)      # inside a GLM formula
```

All of these run the calls `concurrency` (8) at a time. A row whose call
fails is `missing`, with one warning (`FunctAI.problems()` lists them), so
the answers you paid for are kept; when every row fails, the error is thrown.

## Measuring

```julia
e = evaluate(mood, labelled)        # columns named like the inputs and outputs
e.score, e.low, e.high              # 0.83, 0.64, 0.93
DataFrame(e)                        # every row: its answer, its score, its error
compare(evaluate(mood, labelled), evaluate(mood_v2, labelled))   # a paired difference, with its range
```

Text is compared ignoring case and repeated white space; `metric = (row, out) -> …`
scores any other way. The score and its range are computed exactly as in
every other FunctAI language (Wilson's interval for right-or-wrong).

## As a model: fit, formulas, MLJ

```julia
m = fit(AIModel("Which team handles this ticket?"), @formula(team ~ message), train)
predict(m, test)                    # categorical in, categorical out, with the same levels
evaluate(m, test)
```

Fitting calls nothing: it reads the outcome's type (a categorical outcome's
levels become the choice) and picks worked examples from the rows. The same
`AIModel` is an MLJ model: `machine(AIModel("…"), X, y)`, with `examples` a
hyperparameter to tune. Predictions are deterministic: providers give an
answer, not a probability for each class.

## Rating and improving

```julia
FunctAI.configure!(log_calls = true)        # every call is a line of JSON in a folder
p = predict(mood, "Arrived broken, but support was great.")
rate(p, :right)
rate(p; answer = mixed, note = "broken item, good help")   # a correction

rows = rated(mood)                          # rows with known answers, from people's ratings
better = bootstrap_few_shot(mood, rows)     # an improved copy; mood is unchanged
better, trials = gepa(mood, rows; teacher = "gpt-6-sol")   # the instruction rewritten from its mistakes
```

`labeled_few_shot`, `bootstrap_few_shot`, `random_search`,
`instruction_search` and `gepa` each return an improved copy: only the instruction and
the worked examples change, and so does the version. `with_demos` and
`with_instructions` do it by hand. The log folder is shared with Python,
TypeScript and R: calls and ratings made in any of them pool here.

## Streaming

```julia
for piece in stream(haiku, "the first snow")
    print(piece)
end

s = stream(support, "Where is order A-1042?")
foreach(println, eachevent(s))  # started, text, tool_call, tool_result, retry, done
fetch(s)                        # the typed answer, the same as calling
close(s)                        # cancels the call
```

## Tools and programs

```julia
"Look up where an order is."
lookup_order(order::String) = …

@ai tools = [lookup_order] function support(message::String)::String
    "Answer the customer, looking up their order."
end

@program function reply(ticket::String)             # code that calls AI functions:
    triage(ticket).minutes > 60 ? escalate(ticket) : support(ticket)
end
```

A tool is a Julia function: its arguments' types and its docstring tell the
model how to call it. A program is one call in the log, with the AI calls
it made as its children.

## Saving

```julia
FunctAI.save("saved/mood", better)                     # functai.json: any language loads it
mood = FunctAI.load("saved/mood"; types = (result = Mood,))
```

A saved folder holds shapes, not Julia types; `types` gives them back. A
function saved in Python, TypeScript or R loads here when it has no code of
its own; what it cannot run is refused with a reason (`LoadRefused`).

## Where Julia differs, stated

- **Several outputs return all of them** (a `NamedTuple`); Python returns
  the last. The call log still names the last as the answer.
- **`reasoning = true`** is Python's `module = "cot"` (`module` is a Julia keyword).
- **A column keeps the answers it paid for**: a failed row is `missing`
  with a warning, where plain Julia would stop at the first error.
- **Code of your own is versioned by its parsed form**, not its text:
  reformatting or editing comments does not make a new version. A Julia
  function with code of its own never shares a version with another
  language (its code is Julia).
- **Not saved from Julia yet**: code of your own and tools (a folder
  carries no Julia code). A program's version follows the AI functions and
  programs it names, not plain Julia functions it calls.
- **Names shared with MLJ**: both export `predict` and `evaluate`. With
  both loaded, bring one in with `import` (tutorial 7 does). `FunctAI.save`,
  `load` and `login` are not exported (FileIO's names), nor is `ai"…"`
  (PromptingTools.jl's): `@ai` reads it from your code.
- **Not yet**: the reply cache, stateful memory, escalation to another
  model (a `@program` does it by hand), and baking (training your own weights).

## Developing

```bash
julia/check                                     # the contract's data, then Pkg.test() (every contract case, the doctests)
julia/tutorials [docs/julia/0N-*.md ...]        # run the tutorials on real models, write their outputs (about 60 cents)
julia --project=julia/docs julia/docs/make.jl   # the manual (Documenter), into julia/docs/build
set -a; source ~/Projects/lm15-dev/.env; set +a
julia --project=julia julia/tools/live.jl       # against real models (costs cents)
```
