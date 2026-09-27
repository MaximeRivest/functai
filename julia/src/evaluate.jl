# How often a function is right (contract/scores.md): run it on rows with
# known answers, score each row, and give the score with its 95% range.

const Z = 1.959964
const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
              2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]

t975(df) = df <= length(T975) ? T975[df] : Z + (Z^3 + Z) / (4df) + (5Z^5 + 16Z^3 + 3Z) / (96df^2)

"""
    score_interval(values) -> (; mean, low, high)

The mean and its 95% range: Wilson's interval when every value is 0 or 1
(right or wrong), Student's t otherwise. With fewer than two values there is
no range (`nothing`).
"""
function score_interval(values::AbstractVector{<:Real})
    n = length(values)
    n == 0 && return (mean=nothing, low=nothing, high=nothing)
    m = 0.0
    for v in values
        m += v
    end
    m /= n
    n < 2 && return (mean=m, low=nothing, high=nothing)
    if all(v -> v == 0 || v == 1, values)
        z2 = Z * Z
        center = (m + z2 / (2n)) / (1 + z2 / n)
        half = Z * sqrt(m * (1 - m) / n + z2 / (4n^2)) / (1 + z2 / n)
        return (mean=m, low=m == 0 ? 0.0 : max(0.0, center - half), high=m == 1 ? 1.0 : min(1.0, center + half))
    end
    ss = sum(v -> (v - m)^2, values)
    half = t975(n - 1) * sqrt(ss / (n - 1)) / sqrt(n)
    (mean=m, low=m - half, high=m + half)
end

"A value as scores compare it: its JSON (an @enum or Symbol is its text), text normalized."
function normed(v)
    j = try
        jsonvalue(v)
    catch
        return v
    end
    j isa AbstractString ? normalize_text(j) : j
end
function same_answer(a, b)
    a, b = normed(a), normed(b)
    (a isa AbstractString || b isa AbstractString) && return a == b
    try
        LMCC.json_equal(a, b)
    catch
        isequal(a, b)
    end
end

"""
    exact_match(answers, prediction) -> Dict

The default metric: `1` when every output the answers have a value for equals
the prediction's (text compared ignoring case and repeated white space;
numbers by value), else `0`. With several, each also gets `<name>_match`.
"""
function exact_match(answers::AbstractDict, prediction::AbstractDict)
    keys_ = [k for k in keys(prediction) if haskey(answers, k)]
    isempty(keys_) && throw(ArgumentError("exact_match: the data has no column for any output ($(join(keys(prediction), ", ")))"))
    each = OrderedDict{String,Float64}(String(k) => same_answer(answers[k], prediction[k]) ? 1.0 : 0.0 for k in keys_)
    out = OrderedDict{String,Float64}("exact_match" => all(==(1.0), values(each)) ? 1.0 : 0.0)
    length(each) > 1 && for (k, v) in each
        out["$(k)_match"] = v
    end
    out
end

"One row of an evaluation: the row, the outputs, each metric's score, the error, the call."
struct RowResult
    row::OrderedDict{String,Any}
    outputs::Union{Nothing,NamedTuple}
    scores::OrderedDict{String,Float64}
    error::Union{Nothing,String}
    call::Union{Nothing,String}
end

"""
    Evaluation

What [`evaluate`](@ref) measured: `score`, `low` and `high` (the first
metric's mean and its 95% range), `summary` (every metric), `run` (its id in
the call log), and every row, as a table (`DataFrame(e)` works: the row's
columns, `pred_<output>`, each metric, `error`, `call`).
"""
struct Evaluation
    fn::String
    run::String
    results::Vector{RowResult}
    metrics::Vector{String}
    outputs::Vector{String}
end

"Each row's value for a metric (the first by default); a failed row counts 0."
scores(e::Evaluation, metric::AbstractString=first(e.metrics)) = Float64[get(r.scores, metric, 0.0) for r in e.results]

function Base.getproperty(e::Evaluation, name::Symbol)
    name === :score && return score_interval(scores(e)).mean
    name === :low && return score_interval(scores(e)).low
    name === :high && return score_interval(scores(e)).high
    name === :summary && return [(metric=m, score_interval(scores(e, m))..., n=length(e.results),
                                  failed=count(r -> r.error !== nothing, e.results)) for m in e.metrics]
    name === :failed && return count(r -> r.error !== nothing, e.results)
    getfield(e, name)
end
Base.propertynames(::Evaluation) = (:score, :low, :high, :summary, :failed, :run, :results, :metrics)
Base.length(e::Evaluation) = length(e.results)

function rows_table(e::Evaluation)
    columns = String[]
    for r in e.results, k in keys(r.row)
        k in columns || push!(columns, k)
    end
    extra = vcat(["pred_$o" for o in e.outputs], e.metrics, ["error", "call"])
    names = Tuple(Symbol.(vcat(columns, extra)))
    [NamedTuple{names}(Tuple(vcat(Any[get(r.row, c, missing) for c in columns],
                                  Any[r.outputs === nothing ? missing : get(r.outputs, Symbol(o), missing) for o in e.outputs],
                                  Any[get(r.scores, m, 0.0) for m in e.metrics],
                                  Any[something(r.error, missing), something(r.call, missing)]))) for r in e.results]
end

Tables.istable(::Type{Evaluation}) = true
Tables.rowaccess(::Type{Evaluation}) = true
Tables.rows(e::Evaluation) = rows_table(e)
Tables.schema(e::Evaluation) = Tables.schema(rows_table(e))

rows_text(n) = n == 1 ? "1 row" : "$n rows"
fmt(x) = x === nothing ? "—" : @sprintf("%.2f", x)
function Base.show(io::IO, e::Evaluation)
    print(io, "Evaluation of ", e.fn, " on ", rows_text(length(e.results)), ": ", e.metrics[1], " ", fmt(e.score),
          " (95% range ", fmt(e.low), " to ", fmt(e.high), ")")
end
function Base.show(io::IO, ::MIME"text/plain", e::Evaluation)
    println(io, "Evaluation of ", e.fn, " on ", rows_text(length(e.results)))
    width = maximum(length, e.metrics)
    for s in e.summary
        println(io, "  ", rpad(s.metric, width), "  ", fmt(s.mean), "  (95% range ", fmt(s.low), " to ", fmt(s.high), ")")
    end
    failed = e.failed
    failed > 0 && println(io, "  ", failed, " row", failed == 1 ? "" : "s", " failed (they score 0): ",
                          first(something(e.results[findfirst(r -> r.error !== nothing, e.results)].error, ""), 120))
    print(io, "  rows: DataFrame(e) or Tables.rows(e); the run's calls carry caller.evaluation = \"", e.run, "\"")
end

"A metric: `(row, outputs) -> number`, called with the row and the outputs as NamedTuples."
function metric_scores(metric, row, outputs)
    nt_row = NamedTuple(Symbol(k) => v for (k, v) in row)
    if metric isa AbstractDict || metric isa NamedTuple
        return OrderedDict{String,Float64}(String(k) => Float64(m(nt_row, outputs)) for (k, m) in pairs(metric))
    end
    v = metric(nt_row, outputs)
    v isa AbstractDict ? OrderedDict{String,Float64}(String(k) => Float64(x) for (k, x) in v) :
                         OrderedDict{String,Float64}(string(nameof(metric)) => Float64(v))
end

"""
    evaluate(f, data; expected, metric, concurrency) -> Evaluation

Run `f` on every row of `data` (a DataFrame, any Tables.jl table, or a vector
of NamedTuples or Dicts) and score it. A row's inputs are its columns named
like `f`'s inputs; the right answers are the columns named like its outputs,
or `expected` (a column name for the answer, or `output => column` pairs).
The default metric is [`exact_match`](@ref); `metric` is a function
`(row, outputs) -> score` (0 to 1, or any number), or several by name. A
failed row scores 0 and keeps its error. Calls run `concurrency` at a time
and carry `caller.evaluation` in the call log.

```julia
e = evaluate(mood, reviews)      # reviews has columns review and result
e.score, e.low, e.high
DataFrame(e)                     # every row, its answer, its score
```
"""
function evaluate(f::Union{AIFunction,Function}, data; expected=nothing, metric=nothing, concurrency=nothing)
    rows = [row_dict(r) for r in rows_of(data)]
    outputs = f isa AIFunction ? output_names(f) : ["result"]
    answer = f isa AIFunction ? answer_name(f) : "result"
    mapping = expected === nothing ? OrderedDict{String,String}(o => o for o in outputs if any(r -> haskey(r, o), rows)) :
              expected isa Union{AbstractString,Symbol} ? OrderedDict{String,String}(answer => String(expected)) :
              OrderedDict{String,String}(String(k) => String(v) for (k, v) in pairs(expected))
    metric === nothing && isempty(mapping) &&
        throw(ArgumentError("evaluate: the rows have no column for any output ($(join(outputs, ", "))); pass expected = \"column\" or a metric"))
    run = new_id()
    n = something(concurrency, f isa AIFunction ? effective(f.own)[:concurrency] : 8)
    results, _ = run_concurrently(rows, n) do row
        try
            p = with_settings(; caller=Dict("evaluation" => run)) do
                f isa AIFunction ? predict_inputs(f, row_inputs(f, row)) : f(NamedTuple(Symbol(k) => v for (k, v) in row))
            end
            p === missing && throw(ArgumentError("an input is missing"))
            outs = p isa Prediction ? p.outputs : (result=p,)
            s = if metric === nothing
                answers = JObj(o => row[c] for (o, c) in mapping if haskey(row, c))
                exact_match(answers, JObj(o => outs[Symbol(o)] for o in keys(mapping) if haskey(outs, Symbol(o))))
            else
                metric_scores(metric, row, outs)
            end
            RowResult(row, outs, s, nothing, p isa Prediction ? p.call : nothing)
        catch err
            err = unwrap(err)
            RowResult(row, nothing, OrderedDict{String,Float64}(), "$(error_type(err)): $(error_message(err))", nothing)
        end
    end
    metrics = if metric === nothing
        vcat(["exact_match"], length(mapping) > 1 ? ["$(k)_match" for k in keys(mapping)] : String[])
    else
        found = String[]
        for r in results, k in keys(r.scores)
            k in found || push!(found, k)
        end
        isempty(found) ? (metric isa Union{AbstractDict,NamedTuple} ? String.(collect(keys(metric))) : [string(nameof(metric))]) : found
    end
    Evaluation(f isa AIFunction ? f.definition.name : string(nameof(f)), run, RowResult[results...], metrics, outputs)
end

"""
    compare(before, after)

Two evaluations of the same rows, row by row: for each metric both have,
`before` and `after` (the means), `diff` with its 95% range `low` to `high`
(a paired t interval), and how many rows got `better`, `worse` or stayed
the `same`. When the range includes 0, the change could be luck.
"""
function compare(before::Evaluation, after::Evaluation)
    length(before) == length(after) || throw(ArgumentError("compare needs the same rows: $(length(before)) vs $(length(after))"))
    for (i, (a, b)) in enumerate(zip(before.results, after.results))
        isequal(a.row, b.row) || throw(ArgumentError("compare needs the same rows in the same order; row $i differs"))
    end
    shared = [m for m in before.metrics if m in after.metrics]
    isempty(shared) && throw(ArgumentError("no metric in common: $(before.metrics) vs $(after.metrics)"))
    map(shared) do m
        a, b = scores(before, m), scores(after, m)
        d = b .- a
        n = length(d)
        mean_d = sum(d) / n
        low, high = if n >= 2
            sd = sqrt(sum((d .- mean_d) .^ 2) / (n - 1))
            half = t975(n - 1) * sd / sqrt(n)
            (mean_d - half, mean_d + half)
        else
            (nothing, nothing)
        end
        (metric=m, before=sum(a) / n, after=sum(b) / n, diff=mean_d, low=low, high=high,
         better=count(>(0), d), worse=count(<(0), d), same=count(==(0), d), n=n)
    end
end
