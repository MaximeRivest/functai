# Categorical columns (CategoricalArrays): a categorical outcome is a choice
# of its levels, and predictions come back categorical with the same levels;
# a categorical value given to an input is its text.
module FunctAICategoricalArraysExt

using FunctAI
using CategoricalArrays: CategoricalArray, CategoricalValue, categorical, levels, isordered, unwrap

FunctAI.is_categorical(::CategoricalArray) = true
FunctAI.jsonvalue(x::CategoricalValue) = FunctAI.jsonvalue(unwrap(x))
FunctAI.column_spec(::Type{<:Union{Missing,CategoricalValue{T}}}) where {T} = FunctAI.column_spec(T)

function FunctAI.outcome_spec(y::CategoricalArray, given)
    given === nothing || return (OneOf(given), identity)
    lv = string.(levels(y))
    isempty(lv) && throw(ArgumentError("the categorical outcome has no levels"))
    (OneOf(lv), v -> categorical(v; levels=lv, ordered=isordered(y)))
end

end
