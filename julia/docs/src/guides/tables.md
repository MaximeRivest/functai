# Columns and tables

An AI function is a Julia `Function`, so every way Julia applies a function to a column applies it: a dot, `map`, DataFrames' `ByRow`, a `@formula`. All of them run the calls concurrently.

## A dot, or `map`

```julia
df.mood = mood.(df.review)                   # one call per row, 8 at a time, in order
answers = map(mood, reviews)                 # the same
priced = price.(items, currencies)           # a dot over several columns: row by row
predictions = predict.(mood, df.review)      # every row's Prediction, concurrently too
```

The number in flight is the `concurrency` setting (default 8): `with_settings(concurrency = 32) do … end`, or `configure(mood; concurrency = 2)` for a provider with a tight rate limit. The calls are tasks on one thread: they wait on the network, not the CPU, so threads would add nothing.

## DataFrames

```julia
transform(df, :review => ByRow(mood) => :mood)
transform(df, :ticket => ByRow(triage) => AsTable)     # several outputs: several columns
transform(df, [:message, :price] => ByRow(refund) => :decision)
```

`ByRow` hands the whole column to `map`, so it is concurrent. DataFramesMeta (`@rtransform`) and TidierData (`@mutate`) come down to the same. (A row given whole, `AsTable(:) => ByRow(f)`, is applied one row after another by DataFrames: pass the columns instead.)

## When a row fails

A column is many calls, and some may fail: a provider has a bad minute, a reply can't be read even after asking again. FunctAI keeps the answers you paid for:

- a failed row is `missing`, with **one** warning for the whole column;
- `FunctAI.problems()` lists the failed rows and their errors, so you can fix the cause and run just those rows again;
- when **every** row fails, nothing was answered: the first error is thrown instead (usually a setting to fix: a key, a model name).

One call on its own (`mood(text)`) throws its error, as any function does.

## `missing`

`missing` in, `missing` out, with no call:

```julia
mood.(["Love it", missing])        # [happy, missing]
```

## Formulas

StatsModels applies a function in a formula to its whole column with a dot, so an AI function is a feature like `log(x)`:

```julia
using GLM
lm(@formula(price ~ sqft + stars(description)), homes)   # stars: an AI function returning a number or a Bool
```

Its answer must be a number or a `Bool`: formulas don't turn a function's categorical answer into dummy columns. To predict a column with an AI function instead, see [`AIModel`](@ref) (formulas and MLJ).

## Tables in, tables out

Everything that takes rows (`evaluate`, the optimizers, `with_demos`, `AIModel`) takes any Tables.jl table (a `DataFrame`, a `NamedTuple` of vectors, a CSV file's table) or a vector of `NamedTuple`s or `Dict`s. Everything that returns rows returns a Tables.jl table or a vector of `NamedTuple`s: `DataFrame(evaluate(…))`, `DataFrame(calls(f))`, `DataFrame(rated(f))`.
