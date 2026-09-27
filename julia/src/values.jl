# Types and values. A field's Julia type becomes its JSON Schema shape
# (contract/functions.md, "A definition"), written the way Python writes the
# same type, so the same function has the same version in both languages.
# Values cross as JSON: Julia values are written as JSON for the prompt and
# the call log, and the model's JSON is read back as the declared type.

const JObj = LMCC.JObj

"""
    OneOf(values...)

An answer from a fixed list, without declaring an `@enum`:
`::OneOf(:happy, :unhappy, :mixed)` answers a `Symbol`,
`::OneOf("yes", "no")` a `String`. Its shape is the choice
`{"enum": [...], "type": "string"}`, as Python's `Literal[...]`.

# Examples
```jldoctest
julia> species = ["mallard", "blue jay", "other"];

julia> print(FunctAI.LMCC.json_text(FunctAI.shape_of(OneOf(species))))
{"enum":["mallard","blue jay","other"],"type":"string"}
```
"""
struct OneOf{T}
    values::Vector{T}
    function OneOf{T}(values) where {T}
        isempty(values) && throw(ArgumentError("OneOf needs at least one value"))
        allunique(values) || throw(ArgumentError("OneOf has a value twice: $(values)"))
        new{T}(collect(T, values))
    end
end
OneOf(values::Symbol...) = OneOf{Symbol}(collect(values))
OneOf(values::AbstractString...) = OneOf{String}(String[values...])
OneOf(values::AbstractVector{Symbol}) = OneOf{Symbol}(values)
OneOf(values::AbstractVector{<:AbstractString}) = OneOf{String}(String.(values))
Base.show(io::IO, o::OneOf) = print(io, "OneOf(", join(repr.(o.values), ", "), ")")
Base.:(==)(a::OneOf, b::OneOf) = a.values == b.values
Base.hash(o::OneOf, h::UInt) = hash(o.values, hash(:OneOf, h))

# ------------------------------------------------------------------ shapes

is_user_struct(T) = T isa DataType && isstructtype(T) && isconcretetype(T) && fieldcount(T) > 0 &&
                    !(T <: Number) && !(T <: AbstractString) && parentmodule(T) ∉ (Core, Base) &&
                    !(T <: Function) && nameof(parentmodule(T)) !== :Dates

strip_missing(T) = T isa Union ? Base.typesplit(T, Missing) : T

"""
    shape_of(T)

The JSON Schema shape of a Julia type, as FunctAI writes it in every language:
`String` → `{"type": "string"}`, `Int` → `integer`, `Float64` → `number`,
`Bool` → `boolean`, an `@enum` or [`OneOf`](@ref) → a choice,
`Union{T,Missing}` or `Union{T,Nothing}` → optional (the model may leave it
empty: `missing` or `nothing` back),
`Vector{T}` → a list, `Dict{String,T}` → a map, a `NamedTuple` or a struct →
a record (every field required, in declaration order), `Union{T,Nothing}` →
optional, `Any` → any JSON. A JSON Schema `Dict` is used as it is.

# Examples
```jldoctest
julia> print(FunctAI.LMCC.json_text(FunctAI.shape_of(Union{Int,Missing})))
{"anyOf":[{"type":"integer"},{"type":"null"}]}

julia> print(FunctAI.LMCC.json_text(FunctAI.shape_of(Vector{@NamedTuple{name::String, age::Int}})))
{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}
```
"""
shape_of(o::OneOf) = LMCC.jobj("enum" => Any[string(v) for v in o.values], "type" => "string")
shape_of(d::AbstractDict) = LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in d))
function shape_of(T::Type; where::AbstractString="value")
    T === Any && return JObj()
    T === Nothing && return LMCC.jobj("type" => "null")
    T === Missing && return LMCC.jobj("type" => "null")
    T === Union{} && throw(ArgumentError("$where: a field cannot have type Union{}"))
    T === Bool && return LMCC.jobj("type" => "boolean")
    (T <: AbstractString || T === Symbol || T <: AbstractChar) && return LMCC.jobj("type" => "string")
    T <: Integer && return LMCC.jobj("type" => "integer")
    T <: Real && return LMCC.jobj("type" => "number")
    if T isa Union
        # missing and nothing are both JSON's null: the answer may be left empty
        parts = Any[p === Missing ? Nothing : p for p in Base.uniontypes(T)]
        unique!(parts)
        sort!(parts; by=p -> p === Nothing)           # the value first, null last: Python's Optional[T]
        return LMCC.jobj("anyOf" => Any[shape_of(p; where) for p in parts])
    end
    T <: Enum && return LMCC.jobj("enum" => Any[string(x) for x in instances(T)], "type" => "string")
    if T <: AbstractVector || T <: AbstractSet
        E = eltype(T)
        return E === Any ? LMCC.jobj("type" => "array") : LMCC.jobj("type" => "array", "items" => shape_of(E; where="$where[]"))
    end
    if T <: AbstractDict
        V = valtype(T)
        return V === Any ? LMCC.jobj("type" => "object") : LMCC.jobj("type" => "object", "additionalProperties" => shape_of(V; where="$where[]"))
    end
    if (T <: NamedTuple && isconcretetype(T)) || is_user_struct(T)
        names = fieldnames(T)
        props = JObj(String(n) => shape_of(t; where="$where.$n") for (n, t) in zip(names, fieldtypes(T)))
        return LMCC.jobj("type" => "object", "properties" => props, "required" => Any[String(n) for n in names])
    end
    throw(ArgumentError("$where: FunctAI cannot write $T as JSON; use String, a number, Bool, an @enum, OneOf(...), " *
                        "a Vector, a Dict{String,T}, a NamedTuple or a struct of those"))
end
shape_of(x; where::AbstractString="value") =
    throw(ArgumentError("$where: expected a type, OneOf(...) or a JSON Schema Dict, not $(repr(x))"))

"""
    julia_type(shape)

The Julia type values of a shape are read as, for a function loaded from a
saved folder (its Julia types are gone; its shapes remain): a record is a
`NamedTuple`, a choice a `String`, a list a `Vector`, a map a `Dict`.
"""
function julia_type(shape::AbstractDict)
    if haskey(shape, "anyOf") || haskey(shape, "oneOf")
        options = [s for s in get(shape, "anyOf", get(shape, "oneOf", Any[])) if get(s, "type", nothing) != "null"]
        nullable = any(s -> get(s, "type", nothing) == "null", get(shape, "anyOf", get(shape, "oneOf", Any[])))
        T = length(options) == 1 ? julia_type(options[1]) : Any
        return nullable ? Union{T,Nothing} : T
    end
    haskey(shape, "enum") && return all(v -> v isa AbstractString, shape["enum"]) ? String : Any
    t = get(shape, "type", nothing)
    t == "string" && return String
    t == "integer" && return Int
    t == "number" && return Float64
    t == "boolean" && return Bool
    t == "null" && return Nothing
    t == "array" && return haskey(shape, "items") ? Vector{julia_type(shape["items"])} : Vector{Any}
    if t == "object"
        props = get(shape, "properties", nothing)
        if props isa AbstractDict && !isempty(props)
            names = Tuple(Symbol(k) for k in keys(props))
            return NamedTuple{names,Tuple{(julia_type(v) for v in values(props))...}}
        end
        extra = get(shape, "additionalProperties", nothing)
        return Dict{String,extra isa AbstractDict ? julia_type(extra) : Any}
    end
    Any
end

"The type of a field's values: its declared type, the values of a `OneOf`, or a shape's."
valuetype(T::Type) = T
valuetype(::OneOf{T}) where {T} = T
valuetype(shape::AbstractDict) = julia_type(shape)

# ------------------------------------------------------------------ values → JSON

struct NoJSON <: Exception
    msg::String
end
Base.showerror(io::IO, e::NoJSON) = print(io, e.msg)

"""
    jsonvalue(x)

A Julia value as the JSON its type describes: a struct or `NamedTuple` is
an object, an `@enum` its name, a `Symbol` its text, `missing` and
`nothing` are null. A value with no JSON form throws `NoJSON`.
"""
jsonvalue(x::Nothing) = nothing
jsonvalue(::Missing) = nothing
jsonvalue(x::Bool) = x
jsonvalue(x::AbstractString) = String(x)
jsonvalue(x::AbstractChar) = string(x)
jsonvalue(x::Symbol) = String(x)
jsonvalue(x::Enum) = string(x)
jsonvalue(x::Integer) = typemin(Int64) <= x <= typemax(Int64) ? Int64(x) : BigInt(x)
function jsonvalue(x::Real)
    f = Float64(x)
    isfinite(f) || throw(NoJSON("$x has no JSON form"))
    f
end
jsonvalue(x::Union{AbstractVector,Tuple,AbstractSet}) = Any[jsonvalue(v) for v in x]
jsonvalue(x::AbstractDict) = JObj(string(k) => jsonvalue(v) for (k, v) in x)
jsonvalue(x::NamedTuple) = JObj(String(k) => jsonvalue(v) for (k, v) in pairs(x))
function jsonvalue(x::T) where {T}
    is_user_struct(T) || throw(NoJSON("a $(T) has no JSON form"))
    JObj(String(n) => jsonvalue(getfield(x, n)) for n in fieldnames(T))
end

"A value as the call log writes it: its JSON, or `{\$type, \$repr}` when it has none."
function logvalue(x)
    try
        return jsonvalue(x)
    catch err
        err isa NoJSON || rethrow()
        text = sprint(show, x; context=:limit => true)
        return LMCC.jobj("\$type" => string(typeof(x)), "\$repr" => first(text, 2000))
    end
end

"Code points of a JSON value's canonical form (the call log's `sizes`)."
jsonsize(v) = length(LMCC.canonical_json(v))

# ------------------------------------------------------------------ JSON → values

"The reply's value did not fit its type: an unreadable reply (`parse-value`)."
misfit_refusal(where, msg) = LMCC.Refusal("parse-value", "$where: $msg")

"""
    fromjson(T, value)

The model's JSON value as a Julia `T`, checked: a choice outside its list, a
missing record field or a fraction for an integer throws `parse-value`.
"""
fromjson(spec, v) = fromjson(spec, v, "value")
fromjson(::Type{Any}, v, where) = v
fromjson(shape::AbstractDict, v, where) = (p = misfit(shape, v, where); p === nothing ? convert_loaded(julia_type(shape), v) : throw(misfit_refusal(where, p)))
function fromjson(o::OneOf{T}, v, where) where {T}
    v isa AbstractString || throw(misfit_refusal(where, "expected one of $(o.values), got $(repr(v))"))
    i = findfirst(x -> string(x) == v, o.values)
    i === nothing && throw(misfit_refusal(where, "$(repr(v)) is not one of $(join(string.(o.values), ", "))"))
    o.values[i]
end
function fromjson(::Type{T}, v, where) where {T}
    T === Nothing && (v === nothing ? (return nothing) : throw(misfit_refusal(where, "expected null, got $(repr(v))")))
    if T isa Union
        v === nothing && (Nothing <: T ? (return nothing) : Missing <: T ? (return missing) :
                          throw(misfit_refusal(where, "expected a value, got null")))
        S = Base.typesplit(Base.typesplit(T, Nothing), Missing)
        S isa Union || return fromjson(S, v, where)
        for part in Base.uniontypes(S)
            try
                return fromjson(part, v, where)
            catch err
                err isa LMCC.Refusal || rethrow()
            end
        end
        throw(misfit_refusal(where, "$(repr(v)) fits none of $(T)"))
    end
    v === nothing && throw(misfit_refusal(where, "expected $(T), got null"))
    if T === Bool
        v isa Bool || throw(misfit_refusal(where, "expected true or false, got $(repr(v))"))
        return v
    end
    if T <: AbstractString || T === Symbol
        v isa AbstractString || throw(misfit_refusal(where, "expected text, got $(repr(v))"))
        return T === Symbol ? Symbol(v) : T(v)
    end
    if T <: AbstractChar
        (v isa AbstractString && length(v) == 1) || throw(misfit_refusal(where, "expected one character, got $(repr(v))"))
        return T(first(v))
    end
    if T <: Integer
        (v isa Real && !(v isa Bool) && isinteger(v)) || throw(misfit_refusal(where, "expected an integer, got $(repr(v))"))
        return T(v)
    end
    if T <: Real
        (v isa Real && !(v isa Bool)) || throw(misfit_refusal(where, "expected a number, got $(repr(v))"))
        return T(v)
    end
    if T <: Enum
        v isa AbstractString || throw(misfit_refusal(where, "expected one of $(join(string.(instances(T)), ", ")), got $(repr(v))"))
        for x in instances(T)
            string(x) == v && return x
        end
        throw(misfit_refusal(where, "$(repr(v)) is not one of $(join(string.(instances(T)), ", "))"))
    end
    if T <: AbstractVector || T <: AbstractSet
        v isa AbstractVector || throw(misfit_refusal(where, "expected a list, got $(repr(v))"))
        E = eltype(T)
        items = [fromjson(E, x, "$where[$(i)]") for (i, x) in enumerate(v)]
        C = isconcretetype(T) ? T : T <: AbstractSet ? Set{E} : Vector{E}
        return C(items)
    end
    if T <: AbstractDict
        v isa AbstractDict || throw(misfit_refusal(where, "expected an object, got $(repr(v))"))
        K, V = keytype(T), valtype(T)
        C = isconcretetype(T) ? T : Dict{K,V}
        return C(convert_key(K, k) => fromjson(V, x, "$where.$k") for (k, x) in v)
    end
    if (T <: NamedTuple && isconcretetype(T)) || is_user_struct(T)
        v isa AbstractDict || throw(misfit_refusal(where, "expected an object with $(join(fieldnames(T), ", ")), got $(repr(v))"))
        args = map(fieldnames(T), fieldtypes(T)) do n, FT
            haskey(v, String(n)) || (FT >: Nothing ? (return nothing) : throw(misfit_refusal(where, "missing $(n)")))
            fromjson(FT, v[String(n)], "$where.$n")
        end
        return T <: NamedTuple ? T(Tuple(args)) : T(args...)
    end
    throw(misfit_refusal(where, "cannot read $(repr(v)) as $(T)"))
end

convert_key(::Type{Symbol}, k) = Symbol(k)
convert_key(::Type{K}, k) where {K} = K <: AbstractString ? K(k) : k

"A JSON value checked against a shape by `misfit`, then read as `julia_type(shape)`."
convert_loaded(T, v) = try
    fromjson(T, v, "value")
catch err
    err isa LMCC.Refusal ? v : rethrow()
end

"""
    misfit(shape, value, where)

The first place a JSON value does not fit a shape, as a sentence, or
`nothing`: the part of JSON Schema shapes use (type, enum, anyOf/oneOf,
properties, required, additionalProperties, items).
"""
function misfit(shape::AbstractDict, value, where::AbstractString)
    options = get(shape, "anyOf", get(shape, "oneOf", nothing))
    if options isa AbstractVector
        return any(o -> misfit(o, value, where) === nothing, options) ? nothing : "$where: $(repr(value)) fits none of its options"
    end
    if haskey(shape, "enum")
        any(e -> LMCC.json_equal(e, value), shape["enum"]) || return "$where: $(repr(value)) is not one of $(join(string.(shape["enum"]), ", "))"
    end
    t = get(shape, "type", nothing)
    if t isa AbstractString
        ok = t == "string" ? value isa AbstractString :
             t == "integer" ? (value isa Real && !(value isa Bool) && isinteger(value)) :
             t == "number" ? (value isa Real && !(value isa Bool)) :
             t == "boolean" ? value isa Bool :
             t == "null" ? value === nothing :
             t == "array" ? value isa AbstractVector :
             t == "object" ? value isa AbstractDict : true
        ok || return "$where: expected $t, got $(repr(value))"
    end
    if value isa AbstractVector && get(shape, "items", nothing) isa AbstractDict
        for (i, x) in enumerate(value)
            p = misfit(shape["items"], x, "$where[$i]")
            p === nothing || return p
        end
    end
    if value isa AbstractDict
        props = get(shape, "properties", JObj())
        for name in get(shape, "required", Any[])
            haskey(value, name) || return "$where: missing $name"
        end
        extra = get(shape, "additionalProperties", nothing)
        for (k, x) in value
            if haskey(props, k)
                p = misfit(props[k], x, "$where.$k")
                p === nothing || return p
            elseif extra === false
                return "$where: unexpected $k"
            elseif extra isa AbstractDict
                p = misfit(extra, x, "$where.$k")
                p === nothing || return p
            end
        end
    end
    nothing
end
