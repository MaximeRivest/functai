# Program interfaces (contract/programs.md): the inputs a program takes and
# the outputs it gives, as data every language reads. Its signature (what
# recorded data looks like), which interfaces are refused, and whether a
# value fits a field, by the vocabulary the contract lists, so every
# language checks alike without a full JSON Schema validator.

"""
    InterfaceError

A program given, or returning, what its interface does not take or give
(`code` `"interface-input"` / `"interface-output"`), or an interface that
is refused when its program is defined or read (`"interface-malformed"`).
`field` names the field at fault (`nothing` when the fault is not a
field's). contract/programs.md.
"""
struct InterfaceError <: Exception
    code::String
    field::Union{Nothing,String}
    msg::String
end
Base.showerror(io::IO, e::InterfaceError) =
    print(io, "InterfaceError [", e.code, "]", e.field === nothing ? "" : " ($(e.field))", ": ", e.msg)

# matched whole: \A and \z (a regular expression's `$` also matches before a final newline)
const FIELD_NAME = r"\A[A-Za-z_][A-Za-z0-9_]*\z"
const DEFS_REF = r"\A#/\$defs/([A-Za-z0-9_.-]+)\z"
is_name(x) = x isa AbstractString && occursin(FIELD_NAME, x)

const JSON_TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")
const ANNOTATION_KINDS = Dict{String,Any}("title" => AbstractString, "description" => AbstractString, "format" => AbstractString,
                                          "\$comment" => AbstractString, "deprecated" => Bool, "readOnly" => Bool,
                                          "writeOnly" => Bool, "examples" => AbstractVector, "default" => Any)
const ASSERTIONS = Set(["type", "enum", "const", "anyOf", "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
                        "properties", "required", "additionalProperties", "minLength", "maxLength", "minimum", "maximum",
                        "exclusiveMinimum", "exclusiveMaximum", "\$ref", "\$defs"])
const INPUT_KEYS = Set(["name", "shape", "desc", "type", "opaque", "optional"])
const OUTPUT_KEYS = Set(["name", "shape", "desc", "type", "opaque"])

"Equal as canonical JSON (calls.md): `1` and `1.0` are one value; `true` is not `1`."
same_json(a, b) = LMCC.canonical_json(a) == LMCC.canonical_json(b)

"The JSON kind of a JSON value, integers being numbers with no fraction (`5.0` is one)."
function json_kind(v)
    v === nothing && return "null"
    v isa Bool && return "boolean"
    v isa Real && return isinteger(v) ? "integer" : "number"
    v isa AbstractString && return "string"
    v isa AbstractVector && return "array"
    v isa AbstractDict && return "object"
    "other"
end
is_number(v) = v isa Real && !(v isa Bool)

"A shape without its own `default`: what its data looks like."
data_shape(shape::AbstractDict) = JObj(String(k) => v for (k, v) in shape if k != "default")

"""
Whether a shape uses the listed vocabulary, each keyword (annotations
included) with a value of its kind. `carry`: an AI function's shape, whose
other keywords are lmcc's, carried and never read here.
"""
function well_formed(shape, root; carry::Bool=false)
    shape isa AbstractDict || return false
    for (k, v) in shape
        if haskey(ANNOTATION_KINDS, k)
            v isa ANNOTATION_KINDS[k] || return false
            continue
        end
        if !(k in ASSERTIONS)
            carry && continue
            return false
        end
        ok = if k == "type"
            names = v isa AbstractVector ? v : Any[v]
            !isempty(names) && all(n -> n isa AbstractString && n in JSON_TYPES, names) && allunique(names)
        elseif k == "enum"
            v isa AbstractVector && !isempty(v)
        elseif k in ("anyOf", "prefixItems")
            v isa AbstractVector && !isempty(v) && all(x -> well_formed(x, root; carry), v)
        elseif k == "items"
            well_formed(v, root; carry)
        elseif k in ("properties", "\$defs")
            v isa AbstractDict && all(x -> well_formed(x, root; carry), values(v))
        elseif k == "additionalProperties"
            v isa Bool || well_formed(v, root; carry)
        elseif k == "required"
            v isa AbstractVector && all(x -> x isa AbstractString, v) && allunique(v)
        elseif k in ("minItems", "maxItems", "minLength", "maxLength")
            json_kind(v) == "integer" && v >= 0
        elseif k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
            is_number(v)
        elseif k == "uniqueItems"
            v isa Bool
        elseif k == "\$ref"
            m = v isa AbstractString ? match(DEFS_REF, v) : nothing
            defs = get(root, "\$defs", nothing)
            m !== nothing && defs isa AbstractDict && haskey(defs, m.captures[1])
        else
            true                                    # const: any value
        end
        ok || return false
    end
    true
end

"The `\$defs` a shape checks the same value against: its `\$ref`, and its `anyOf`'s, without passing into the value."
function same_value_refs(shape)
    out = Set{String}()
    shape isa AbstractDict || return out
    haskey(shape, "\$ref") && push!(out, match(DEFS_REF, shape["\$ref"]).captures[1])
    for x in get(shape, "anyOf", Any[])
        union!(out, same_value_refs(x))
    end
    out
end

"Whether a `\$defs` entry reaches itself through `\$ref` and `anyOf` alone: checking against it would never end."
function loops(root)
    defs = get(root, "\$defs", nothing)
    defs isa AbstractDict || return false
    graph = Dict(String(n) => same_value_refs(d) for (n, d) in defs)
    function reaches(start, seen)
        for n in get(graph, start, ())
            (n == first(seen) || (!(n in seen) && reaches(n, (seen..., n)))) && return true
        end
        false
    end
    any(n -> reaches(n, (n,)), keys(graph))
end

"Whether a JSON value fits a shape of the vocabulary (programs.md, \"Checking values\")."
function fits_shape(v, shape::AbstractDict, root)
    t = json_kind(v)
    if haskey(shape, "\$ref")
        name = match(DEFS_REF, shape["\$ref"]).captures[1]
        fits_shape(v, root["\$defs"][name], root) || return false
    end
    if haskey(shape, "type")
        names = shape["type"] isa AbstractVector ? shape["type"] : Any[shape["type"]]
        (t in names || (t == "integer" && "number" in names)) || return false
    end
    haskey(shape, "enum") && !any(x -> same_json(x, v), shape["enum"]) && return false
    haskey(shape, "const") && !same_json(shape["const"], v) && return false
    haskey(shape, "anyOf") && !any(s -> fits_shape(v, s, root), shape["anyOf"]) && return false
    if t == "string"
        n = length(v)                                # code points
        n < get(shape, "minLength", 0) && return false
        haskey(shape, "maxLength") && n > shape["maxLength"] && return false
    end
    if t in ("integer", "number")
        haskey(shape, "minimum") && v < shape["minimum"] && return false
        haskey(shape, "maximum") && v > shape["maximum"] && return false
        haskey(shape, "exclusiveMinimum") && v <= shape["exclusiveMinimum"] && return false
        haskey(shape, "exclusiveMaximum") && v >= shape["exclusiveMaximum"] && return false
    end
    if t == "array"
        prefix = get(shape, "prefixItems", Any[])
        for (x, s) in zip(v, prefix)
            fits_shape(x, s, root) || return false
        end
        if haskey(shape, "items")
            for x in v[length(prefix)+1:end]
                fits_shape(x, shape["items"], root) || return false
            end
        end
        length(v) < get(shape, "minItems", 0) && return false
        haskey(shape, "maxItems") && length(v) > shape["maxItems"] && return false
        get(shape, "uniqueItems", false) === true && !allunique(LMCC.canonical_json.(v)) && return false
    end
    if t == "object"
        props = get(shape, "properties", JObj())
        any(k -> !haskey(v, k), get(shape, "required", Any[])) && return false
        for (k, x) in v
            if haskey(props, k)
                fits_shape(x, props[k], root) || return false
            elseif haskey(shape, "additionalProperties")
                extra = shape["additionalProperties"]
                (extra === false || (extra isa AbstractDict && !fits_shape(x, extra, root))) && return false
            end
        end
    end
    true
end

"A value's JSON form, or `NOJSON` when it has none (it fits only an opaque field)."
function json_form(v)
    try
        jsonvalue(v)
    catch err
        err isa NoJSON || rethrow()
        NOJSON
    end
end
struct NoJSONForm end
const NOJSON = NoJSONForm()

"Whether a value (a Julia value, or JSON) fits a field of an interface."
function fits(value, field::AbstractDict)
    get(field, "opaque", false) === true && return true
    data = json_form(value)
    data === NOJSON && return false
    shape = data_shape(field["shape"])
    fits_shape(data, shape, shape)
end

malformed(field, msg) = InterfaceError("interface-malformed", field, msg)

"""
    interface_fault(interface; ai = false) -> InterfaceError or nothing

The first reason an interface is refused when its program is defined or read
(programs.md, "Interfaces that are refused"), form and meaning together,
field by field; `nothing` when it is not. `ai`: an AI function's, whose
shapes may carry lmcc's other keywords.
"""
function interface_fault(iface; ai::Bool=false)
    (iface isa AbstractDict && issubset(keys(iface), ("description", "inputs", "outputs")) &&
     get(iface, "description", nothing) isa AbstractString && get(iface, "inputs", nothing) isa AbstractVector &&
     get(iface, "outputs", nothing) isa AbstractVector && !isempty(iface["outputs"])) ||
        return malformed(nothing, "an interface is an object of description, inputs and outputs (at least one), and nothing else")
    seen = Set{String}()
    for direction in ("inputs", "outputs")
        allowed = direction == "inputs" ? INPUT_KEYS : OUTPUT_KEYS
        for f in iface[direction]
            name = f isa AbstractDict ? get(f, "name", nothing) : nothing
            at = name isa AbstractString ? String(name) : nothing
            f isa AbstractDict || return malformed(at, "a field is an object")
            extra = setdiff(keys(f), allowed)
            isempty(extra) || return malformed(at, "a field of $direction has no key $(first(sort!(collect(extra))))")
            (is_name(name) && !(name in seen)) ||
                return malformed(at, "a field's name is an ASCII identifier, used once among the inputs and outputs: $(repr(name))")
            push!(seen, name)
            any(k -> haskey(f, k) && !(f[k] isa AbstractString), ("desc", "type")) && return malformed(at, "desc and type are text")
            any(k -> haskey(f, k) && f[k] !== true, ("opaque", "optional")) && return malformed(at, "opaque and optional are true when present")
            shape = get(f, "shape", nothing)
            shape isa AbstractDict || return malformed(at, "a shape is an object")
            (well_formed(shape, shape; carry=ai) && !loops(shape)) ||
                return malformed(at, ai ? "its shape uses a listed keyword with a value of the wrong kind, or a reference that does not end" :
                                          "its shape uses a keyword the vocabulary does not list, one with a value of the wrong kind, or a reference that does not end")
            ai && get(f, "optional", false) === true && !haskey(shape, "default") &&
                return malformed(at, "an AI function's optional input has a default (a model is sent every input)")
            get(f, "opaque", false) === true && !isempty(shape) && return malformed(at, "an opaque field's shape is {}")
            if haskey(shape, "default")
                ds = data_shape(shape)
                fits_shape(shape["default"], ds, ds) || return malformed(at, "its default $(LMCC.json_text(shape["default"])) does not fit its shape")
            end
        end
    end
    nothing
end

"Refuse an interface that `interface_fault` refuses."
function check_interface(iface; ai::Bool=false, what::AbstractString="")
    fault = interface_fault(iface; ai)
    fault === nothing && return iface
    throw(InterfaceError(fault.code, fault.field, isempty(what) ? fault.msg : "$what: $(fault.msg)"))
end

"""
    interface_signature(interface)

The interface's signature (programs.md): lmcc's fingerprint of its fields,
each plain and untyped, each shape without its own `default`. The call log's
`program.interface`: calls with one signature record the same data.
"""
interface_signature(iface::AbstractDict) = LMCC.sha256_of(Any[
    LMCC.jobj("direction" => d == "inputs" ? "input" : "output", "name" => f["name"], "purpose" => "plain",
              "shape" => data_shape(f["shape"]), "type" => "") for d in ("inputs", "outputs") for f in iface[d]])

"""
    check_inputs(interface, given) -> OrderedDict

The inputs a program's call has (programs.md, "Checking values", 1): every
input given is one the interface has, fits, and a required one is there; an
optional input left out takes its shape's `default` (a copy of it), and
otherwise stays left out. Throws `InterfaceError` `interface-input`, naming
the input; its message says what the value is (its kind, its size), never
the value: an error's message goes where the input's value may not.
"""
function check_inputs(iface::AbstractDict, given::AbstractDict)
    names = [f["name"] for f in iface["inputs"]]
    unknown = sort!([String(k) for k in keys(given) if !(String(k) in names)])
    isempty(unknown) || throw(InterfaceError("interface-input", unknown[1], "it takes no input $(unknown[1]); its inputs: $(join(names, ", "))"))
    out = OrderedDict{String,Any}()
    for f in iface["inputs"]
        name = f["name"]
        if haskey(given, name)
            v = given[name]
            fits(v, f) || throw(InterfaceError("interface-input", name, "$name: $(describe_misfit(v, f))"))
            out[name] = v
        elseif get(f, "optional", false) !== true
            throw(InterfaceError("interface-input", name, "no value for $name"))
        elseif haskey(f["shape"], "default")
            out[name] = LMCC.deepcopy_json(f["shape"]["default"])
        end                                           # else left out: the program's own default applies
    end
    out
end

"""
    check_returned(interface, value) -> OrderedDict

A program's outputs by name, from what its code returned (programs.md,
"Checking values", 2): the value when it has one output; a record (a
`NamedTuple` or a `Dict`) holding each output by name, and nothing else,
when it has several. The record needs no JSON form of its own: each output
is checked against its field (an opaque one is never checked). Throws
`InterfaceError` `interface-output`, naming the field.
"""
function check_returned(iface::AbstractDict, value)
    outputs = iface["outputs"]
    first_name = outputs[1]["name"]
    values = if length(outputs) == 1
        OrderedDict{String,Any}(first_name => value)
    else
        value isa Union{NamedTuple,AbstractDict} ||
            throw(InterfaceError("interface-output", first_name, "it returned a $(typeof(value)), not a record of its outputs ($(join((f["name"] for f in outputs), ", ")))"))
        OrderedDict{String,Any}(string(k) => v for (k, v) in pairs(value))
    end
    names = [f["name"] for f in outputs]
    unknown = sort!([k for k in keys(values) if !(k in names)])
    isempty(unknown) || throw(InterfaceError("interface-output", unknown[1], "it returned $(unknown[1]), which is not an output ($(join(names, ", ")))"))
    out = OrderedDict{String,Any}()
    for f in outputs
        name = f["name"]
        haskey(values, name) || throw(InterfaceError("interface-output", name, "it returned no $name"))
        fits(values[name], f) || throw(InterfaceError("interface-output", name, "$name: $(describe_misfit(values[name], f))"))
        out[name] = values[name]
    end
    out
end

"Why a value does not fit a field, without the value: its kind and size, and the field's shape."
function describe_misfit(v, f)
    data = json_form(v)
    data === NOJSON && return "a $(typeof(v)) has no JSON form, and the field is not opaque"
    "$(json_kind_text(data)) does not fit $(first(LMCC.json_text(data_shape(f["shape"])), 300))"
end
json_kind_text(d) = d === nothing ? "null" : d isa Bool ? "a boolean" : d isa Real ? (isinteger(d) ? "an integer" : "a number") :
                    d isa AbstractString ? "a text of $(length(d)) characters" : d isa AbstractVector ? "a list of $(length(d)) items" :
                    d isa AbstractDict ? "an object of $(length(d)) members" : "a value"
