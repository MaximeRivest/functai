# An AI function as a model: `fit` it on a table, `predict` new rows. The
# language model is the engine; fitting reads the outcome's type (and, for a
# categorical outcome, its levels) and picks worked examples from the rows.
# Fitting calls nothing and costs nothing; predicting calls the model.
#
#     m = fit(AIModel("Which team handles this ticket?"), @formula(team ~ message), train)
#     predict(m, test)
#
# The same type is an MLJ model (`machine(AIModel(...), X, y)`), and the
# formula face comes with StatsModels (ext/FunctAIStatsModelsExt.jl).

"""
    AIModel(description = ""; examples = 16, seed = 0, name = "", levels = nothing,
            lm = nothing, reasoning = false, settings = (;), method = :labeled,
            teacher = nothing, budget = 300)

A model whose predictions a language model makes, for rows of a table.
`fit(AIModel(…), X, y)` (or with a formula: `fit(AIModel(…), @formula(y ~ a + b), data)`)
builds an AI function whose inputs are the columns of `X`, whose answer has
`y`'s type (a categorical `y`: a choice of its levels; `levels`: a choice of
these), and whose worked examples are `examples` rows of the training data
(a seeded sample). Nothing is called until `predict`.

It is also an MLJ model: `machine(AIModel("…"), X, y)`, with `examples` a
hyperparameter to tune. Predictions are deterministic: providers give an
answer, not a probability for each class.

`name` is the function's name (the instruction starts "Function: <name>");
empty, it is the outcome's column name. Other settings (`temperature`,
`adapter`, `concurrency`, …) go in `settings`.

**Fitting that learns.** `method` says what fitting does with the rows:

- `:labeled` (the default): `examples` rows become worked examples; nothing
  is called.
- `:bootstrap`: the function (or a `teacher` model) runs on the rows, and up
  to `examples` runs it got right become worked examples, reasoning included
  ([`bootstrap_few_shot`](@ref)).
- `:gepa`: a `teacher` model rewrites the instruction from the function's
  mistakes on the rows, within `budget` calls ([`gepa`](@ref)); then
  `examples` rows become worked examples (`examples = 0` for none).

With `:bootstrap` and `:gepa`, fitting calls the model and costs money. Inside
MLJ's `evaluate!`, each fold is fitted on its own rows, so cross-validation
measures the whole procedure, search included, on rows it never saw: the
honest number for a searched instruction.
"""
mutable struct AIModel <: MMI.Deterministic
    description::String
    examples::Int
    seed::Int
    name::String
    levels::Union{Nothing,Vector{String}}
    lm::Union{Nothing,String}
    reasoning::Bool
    settings::NamedTuple
    method::Symbol
    teacher::Union{Nothing,String}
    budget::Int
end
function AIModel(description::AbstractString=""; examples::Integer=16, seed::Integer=0, name::AbstractString="", levels=nothing,
                 lm=nothing, reasoning::Bool=false, settings=(;), method::Symbol=:labeled, teacher=nothing, budget::Integer=300)
    m = AIModel(String(description), examples, seed, String(name), levels === nothing ? nothing : string.(collect(levels)),
                lm === nothing ? nothing : String(lm), reasoning, (; pairs(settings)...), method,
                teacher === nothing ? nothing : String(teacher), budget)
    msg = MMI.clean!(m)
    isempty(msg) || @warn msg
    m
end

function MMI.clean!(m::AIModel)
    msg = ""
    if m.examples < 0
        msg *= "examples must be at least 0; set to 0. "
        m.examples = 0
    end
    settings_dict(m.settings)            # an unknown setting throws here, not at the first call
    m.method in (:labeled, :bootstrap, :gepa) ||
        throw(ArgumentError("method is :labeled, :bootstrap or :gepa, not $(repr(m.method))"))
    if m.budget < 1
        msg *= "budget must be at least 1; set to 300. "
        m.budget = 300
    end
    msg
end

"""
    AIModelFit

A fitted [`AIModel`](@ref): `predict(m, newdata)` answers each row;
`m.fn` is the AI function it built (its instruction, examples, version);
`evaluate(m, data)` measures it on rows with known answers.
"""
struct AIModelFit
    model::AIModel
    fn::AIFunction
    outcome::String
    rebuild::Any            # the predictions as the outcome's kind of vector
    formula::Any
end

Base.show(io::IO, m::AIModelFit) = print(io, "AIModelFit(", m.outcome, " ~ ", join(input_names(m.fn), " + "),
                                         "; ", length(m.fn.demos), " worked examples)")
function Base.show(io::IO, ::MIME"text/plain", m::AIModelFit)
    println(io, "AI model of ", m.outcome, " from ", join(input_names(m.fn), ", "))
    show(io, MIME"text/plain"(), m.fn)
end

"The Julia type a column's values are read as by an AI function's input."
column_spec(::Type{T}) where {T} = (S = nonmissingtype(T);
    S <: AbstractString ? String : S === Bool ? Bool : S <: Integer ? Int : S <: Real ? Float64 : S === Symbol ? String :
    S <: Union{AbstractVector,AbstractDict,NamedTuple} ? S : Any)

"""
    outcome_spec(y, levels) -> (spec, rebuild)

What an outcome column's answers are: a choice of `levels` when given, else
by the column's type (text, a number, true/false; `Symbol`s: a choice of the
values seen). `rebuild` turns the predictions into the column's kind of
vector. A package extension adds categorical columns (their levels).
"""
function outcome_spec(y::AbstractVector, levels)
    levels === nothing || return (OneOf(levels), identity)
    S = nonmissingtype(eltype(y))
    S === Symbol && return (OneOf(sort!(unique(skipmissing(y)))), identity)
    spec = S <: AbstractString ? String : S === Bool ? Bool : S <: Integer ? Int : S <: Real ? Float64 :
           S === Any ? throw(ArgumentError("the outcome column has values of type Any; give levels = [...] or a typed column")) : S
    (spec, identity)
end

"""
    fit(m::AIModel, X, y) -> AIModelFit

`X` is a table (its columns are the inputs), `y` the outcome. Nothing is
called: the rows become the function's worked examples.
"""
function StatsAPI.fit(m::AIModel, X, y::AbstractVector; name=nothing, formula=nothing)
    Tables.istable(X) || throw(ArgumentError("X is a table (a DataFrame, a NamedTuple of columns, …), not a $(typeof(X))"))
    cols = Tables.columntable(X)
    length(y) == Tables.rowcount(cols) || throw(ArgumentError("X has $(Tables.rowcount(cols)) rows but y has $(length(y))"))
    spec, rebuild = outcome_spec(y, m.levels)
    outcome = something(name, isempty(m.name) ? "outcome" : m.name)
    fname = isempty(m.name) ? replace(String(outcome), r"[^A-Za-z0-9_]" => "_") : m.name
    fname = occursin(r"^[A-Za-z_]", fname) ? fname : "_" * fname
    inputs = [n => column_spec(eltype(c)) for (n, c) in pairs(cols)]
    String(outcome) in string.(first.(inputs)) && throw(ArgumentError("the outcome $outcome is also an input"))
    settings = merge(m.settings, m.lm === nothing ? (;) : (lm=m.lm,), m.reasoning ? (reasoning=true,) : (;))
    fn = AIFunction(fname, m.description; inputs, outputs=[Symbol(outcome) => spec], settings...)
    rows = [merge(NamedTuple(r), NamedTuple{(Symbol(outcome),)}((v,))) for (r, v) in zip(Tables.rowtable(cols), y)]
    rows = [r for r in rows if r[Symbol(outcome)] !== missing]
    fn = if m.method === :bootstrap
        bootstrap_few_shot(fn, rows; teacher=m.teacher, max_bootstrapped=min(4, m.examples), max_labeled=m.examples, seed=m.seed)
    elseif m.method === :gepa
        searched, _ = gepa(fn, rows; teacher=m.teacher, budget=m.budget, seed=m.seed)
        labeled_few_shot(searched, rows; k=m.examples, seed=m.seed)    # after gepa: its instruction, and examples
    else
        labeled_few_shot(fn, rows; k=m.examples, seed=m.seed)
    end
    AIModelFit(m, fn, String(outcome), rebuild, formula)
end

"""
    predict(m::AIModelFit, newdata)

The model's answer for each row of `newdata` (its columns named like the
inputs), `concurrency` calls at a time: as for a column, a row whose call
fails (or that has a `missing` input) is `missing`, with one warning.
"""
function StatsAPI.predict(m::AIModelFit, newdata)
    names = Symbol.(input_names(m.fn))
    cols = Tables.columntable(newdata)
    absent = [n for n in names if !haskey(cols, n)]
    isempty(absent) || throw(ArgumentError("newdata has no column $(join(absent, ", "))"))
    args = [Tuple(cols[n][i] for n in names) for i in 1:Tables.rowcount(cols)]
    m.rebuild(call_each(m.fn, args))
end

"""
    evaluate(m::AIModelFit, data)

How often the fitted model is right on rows with known answers (the outcome's column).
"""
evaluate(m::AIModelFit, data; kw...) = evaluate(m.fn, data; expected=m.outcome, kw...)

# ------------------------------------------------------------------ MLJ

function MMI.fit(m::AIModel, verbosity::Int, X, y)
    fitted = StatsAPI.fit(m, X, y)
    rebuild = fitted.rebuild
    if m.levels === nothing && is_categorical(y)
        classes = MMI.classes(first(skipmissing(y)))       # the training pool: predictions compare with its values
        byname = Dict(string(c) => c for c in classes)
        rebuild = function (v)            # a CategoricalVector with the training pool, as MLJ classifiers return
            any(ismissing, v) && return [x === missing ? missing : byname[string(x)] for x in v]
            out = similar(y, length(v))
            for (i, x) in enumerate(v)
                out[i] = byname[string(x)]
            end
            out
        end
    end
    fitted = AIModelFit(m, fitted.fn, fitted.outcome, rebuild, nothing)
    verbosity > 0 && @info "AIModel: $(length(fitted.fn.demos)) worked examples" *
                           (m.method === :labeled ? "; nothing was called" : ", after $(m.method)")
    (fitted, nothing, (examples=length(fitted.fn.demos), version=version(fitted.fn)))
end
MMI.predict(::AIModel, fitted::AIModelFit, Xnew) = StatsAPI.predict(fitted, Xnew)
MMI.fitted_params(::AIModel, fitted::AIModelFit) = (fn=fitted.fn, instructions=instructions(fitted.fn), demos=demos(fitted.fn))

"Whether a column is categorical (a package extension says so for CategoricalArrays)."
is_categorical(y) = false

MMI.metadata_pkg(AIModel; name="FunctAI", uuid="feb8c6c9-9090-4354-9aa7-5403c44e4a77",
                 url="https://github.com/MaximeRivest/functai", julia=true, license="MIT", is_wrapper=false)
MMI.metadata_model(AIModel;
    input_scitype=MMI.Table,
    target_scitype=AbstractVector,
    load_path="FunctAI.AIModel",
    human_name="AI function model")

# MLJ's `predict` (MLJModelInterface's) is another function than StatsAPI's,
# which FunctAI exports: with `using MLJ`, `predict(f, x)` on an AI function
# must still work, so it has the same methods.
MMI.predict(f::AIFunction, args...; kw...) = StatsAPI.predict(f, args...; kw...)
function Base.Broadcast.broadcasted(::typeof(MMI.predict), f::AIFunction, args...)
    Base.Broadcast.broadcasted(StatsAPI.predict, f, args...)
end
