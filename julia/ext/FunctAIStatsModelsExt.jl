# Formulas (StatsModels): `fit(AIModel("…"), @formula(team ~ message + channel), data)`.
# An AI model reads its inputs as they are, so the right-hand side names
# columns; transformations (log(x), interactions) belong to linear models.
#
# The other direction needs nothing here: an AI function inside a formula
# (`@formula(price ~ sqft + stars(description))`) is broadcast over its
# column by StatsModels, which runs its calls `concurrency` at a time.
module FunctAIStatsModelsExt

using FunctAI
import StatsAPI
import StatsModels
using StatsModels: FormulaTerm, Term, ConstantTerm, AbstractTerm
import Tables

plain_terms(t::Term) = [t.sym]
plain_terms(::ConstantTerm) = Symbol[]
plain_terms(t::Tuple) = reduce(vcat, (plain_terms(x) for x in t); init=Symbol[])
plain_terms(t::AbstractTerm) = throw(ArgumentError(
    "an AI model reads its inputs as they are: name columns (team ~ message + channel), not $(t)"))

function StatsAPI.fit(m::FunctAI.AIModel, f::FormulaTerm, data; kw...)
    f.lhs isa Term || throw(ArgumentError("an AI model predicts one column: write outcome ~ inputs, not $(f.lhs) ~ …"))
    outcome = f.lhs.sym
    inputs = plain_terms(f.rhs)
    isempty(inputs) && throw(ArgumentError("the formula names no input column"))
    cols = Tables.columntable(data)
    for n in (outcome, inputs...)
        haskey(cols, n) || throw(ArgumentError("the data has no column $n"))
    end
    X = NamedTuple{Tuple(inputs)}(Tuple(cols[n] for n in inputs))
    StatsAPI.fit(m, X, cols[outcome]; name=String(outcome), formula=f, kw...)
end

StatsModels.formula(m::FunctAI.AIModelFit) = m.formula === nothing ?
    throw(ArgumentError("this model was fitted without a formula")) : m.formula

end
