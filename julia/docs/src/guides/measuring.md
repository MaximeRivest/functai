# Measuring and improving

```@setup measuring
using FunctAI, DataFrames
```

## `evaluate`

```julia
e = evaluate(mood, rows)                         # rows: any table; inputs and answers by column name
e = evaluate(mood, rows; expected = :label)      # the answers are in another column
e.score, e.low, e.high                           # the mean and its 95% interval
DataFrame(e)                                     # every row: its columns, pred_<output>, its score, error, call
```

Each row's inputs are its columns named like the function's inputs. A row whose call fails scores 0 and keeps its error. The calls carry `caller.evaluation = e.run` in the call log, so they are never mistaken for real use.

The default metric, [`exact_match`](@ref), compares text ignoring case and runs of white space (Unicode case folding, as every FunctAI language does it), numbers by value, and an `@enum` or `Symbol` answer with text by its name:

```@example measuring
exact_match(Dict("result" => "Billing "), Dict("result" => "billing"))
```

Any other metric is a function of the row and the outputs, each a `NamedTuple`:

```julia
evaluate(solve, rows; metric = (row, out) -> abs(out.result - row.answer) < 0.01)
evaluate(triage, rows; metric = Dict("summary_ok" => (row, out) -> length(out.summary) < 120,
                                     "minutes_ok" => (row, out) -> out.minutes == row.minutes))
```

`evaluate` takes any function: a plain Julia one is called with each row, so baselines are measured the same way:

```@example measuring
always_billing(row) = "billing"
rows = [(message = "Charged twice", result = "billing"), (message = "Late parcel", result = "shipping")]
evaluate(always_billing, rows)
```

## The interval

Right-or-wrong scores get Wilson's interval; any other score Student's t. Both are computed exactly as in every FunctAI language, so a score measured in Julia compares with one measured in R:

```@example measuring
score_interval(vcat(ones(72), zeros(8)))
```

## Two versions, row by row

On the same rows, only the rows where two versions disagree say which is better. [`compare`](@ref) counts them and gives the mean difference with a paired interval:

```julia
compare(evaluate(team, rows), evaluate(team_rules, rows))
```

## Improving

Each optimizer returns an improved **copy**; the function you pass is unchanged. Only the instruction and the worked examples change, and so does the [`version`](@ref).

| | What it does | Calls |
|:--|:--|:--|
| [`with_demos`](@ref), [`with_instructions`](@ref) | set them by hand | none |
| [`labeled_few_shot`](@ref) | `k` rows with known answers become worked examples | none |
| [`bootstrap_few_shot`](@ref) | runs the function (or a `teacher` model) on rows; the runs the metric accepts become worked examples, reasoning and tool calls included | one per row tried |
| [`random_search`](@ref) | several sets of examples, each scored on `valset`; the best wins | many |
| [`gepa`](@ref) | a stronger model (`teacher`) reads the function's answers on a few rows, with feedback in words, and rewrites the instruction; candidates are kept in a Pareto pool scored on `selection`, combined, and the best (of equals, the shortest) wins | up to `budget` |
| [`instruction_search`](@ref) | a stronger model (`prompt_lm`) proposes instructions; each is tried on minibatches of `valset`; the best wins | many |

```julia
better = bootstrap_few_shot(refund, train; teacher = "gpt-6-sol", max_bootstrapped = 4)
rewritten, trials = gepa(refund, train; selection = dev, teacher = "gpt-6-sol", budget = 300)
DataFrame(trials)                                # every instruction tried, and why it was kept or dropped
```

Keep three piles of rows: one to learn from, one to choose on, one to test **once**. Choosing the best of several versions on the same rows flatters the winner, and a search is itself random: measure its result on rows it never saw. `AIModel(method = :gepa)` runs the search inside `fit!`, so MLJ's cross-validation measures the search itself.

## Datasets

Three small labelled tables to learn with, the same as Python's and R's: [`FunctAI.tickets`](@ref), [`FunctAI.field_notes`](@ref), [`FunctAI.refunds`](@ref). Each docstring has the rules its labels follow.

```@example measuring
first(DataFrame(FunctAI.tickets()), 3)
```
