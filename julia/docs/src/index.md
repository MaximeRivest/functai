# FunctAI.jl

*Typed Julia functions whose body a language model writes.*

You write the signature; a model writes the body. Then you run it over a column, measure how often it is right, have people rate its calls, improve it, and save it.

```julia
using FunctAI

@enum Mood happy unhappy mixed

@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end

mood("It broke after one day.")        # unhappy::Mood
df.mood = mood.(df.review)             # the whole column, 8 calls at a time
evaluate(mood, labelled)               # how often it is right, with a 95% interval
```

An AI function is a Julia `Function`: call it, broadcast it, pass it to `ByRow`, `map` it, put it in a `@formula`. Its answers are the types you declare (an `@enum`, a struct, `Union{Int,Missing}`, a `Vector`), and a reply that doesn't fit its type is asked again, never made up.

## What it covers

- **Write** an AI function with [`@ai`](@ref), or as data with [`AIFunction`](@ref); see exactly what the model reads with [`FunctAI.prompt`](@ref).
- **Run it on tables**: broadcasting, `map` and DataFrames' `ByRow` run the calls concurrently; `missing` in, `missing` out.
- **Measure** with [`evaluate`](@ref): a score with a 95% interval, every row as a table; [`compare`](@ref) two versions row by row.
- **Improve** with [`labeled_few_shot`](@ref), [`bootstrap_few_shot`](@ref), [`random_search`](@ref), [`instruction_search`](@ref) and [`gepa`](@ref) (the instruction rewritten from the function's mistakes); each returns an improved copy.
- **Model** with [`AIModel`](@ref): `fit`/`predict` with formulas, and an MLJ model.
- **Watch** a call with [`stream`](@ref); give it **tools** (Julia functions); group calls into a [`@program`](@ref).
- **Log** every call, [`rate`](@ref) them, and turn ratings into rows with known answers with [`rated`](@ref).
- **Save** with [`FunctAI.save`](@ref) and **load** with [`FunctAI.load`](@ref), in any FunctAI language.
- **Converse**: a [`conversation`](@ref) is a program's calls that remember each other, kept in a store any process (and Python) opens: branches, helpers that remember only when told, turns stopped or resumed from anywhere.
- **Ask first**: tools say what they do (`effects`), and `approve` asks a function at once, or a person later, from any process; a turn that waited goes on paying for nothing twice.
- **Extend** with [`Plugin`](@ref)s: hooks over turns, context, calls, requests and tools, whose changes are recorded; [`compaction`](@ref) and [`delegate`](@ref) are built with them.
- **Serve** a program over HTTP with [`serve`](@ref), and use one served in any language with [`remote`](@ref).
- **Run long**: the reply cache (in memory, or on disk, shared with Python), a progress line, [`prune_calls`](@ref), [`quotes_found`](@ref), escalation to a model that is surer.
- **Bake**: [`FunctAI.bake_examples`](@ref) writes the training examples every trainer reads; [`FunctAI.baked`](@ref) calls a student trained anywhere.

## One contract, four languages

FunctAI exists in Python, TypeScript, R and Julia, held together by one [contract](https://github.com/MaximeRivest/functai/tree/master/contract): the same function has the same [`version`](@ref), writes the same call log and saves to the same folder in every language. A function improved in Python loads here and sends the same bytes; ratings made in R pool with calls made in Julia.

It stands on two libraries: [lmcc](https://github.com/MaximeRivest/lmcc) (how values are written into a prompt and read back) and [lm15](https://github.com/lm15-dev/LM15.jl) (every provider, one wire, and sign-ins).

## Install

Julia 1.10 or later. FunctAI and the two packages under it are not in the General registry yet; add them in this order:

```julia
using Pkg
Pkg.add(url = "https://github.com/MaximeRivest/lmcc", subdir = "julia")
Pkg.add(url = "https://github.com/lm15-dev/LM15.jl")
Pkg.add(url = "https://github.com/MaximeRivest/functai", subdir = "julia")
```

Set the key of the provider you use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …) in the environment, or sign in once with [`FunctAI.login`](@ref). With no model named, FunctAI uses the first provider it finds a key for.

## Where to go

- New here: [the tutorials](tutorials/index.md), from a first function to decisions, MLJ and living with a function in use.
- A task in mind: [the guides](guides/functions.md).
- Coming from Python or R: [where Julia differs](differences.md).
- A name: [the reference](reference.md), or `?name` in the REPL.
