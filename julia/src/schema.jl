# The contract's JSON Schemas (contract/schema, draft 2020-12), read as data
# and applied as JSON Schema applies them: a call record, a stream event, a
# program's interface, a saved manifest and a rating are checked against the
# schema itself, never against a transcription of it. The keywords below are
# the ones the contract's schemas use; a schema that uses another is refused
# when the package loads, so a keyword the contract adds cannot be skipped
# unnoticed. Patterns are read as ECMA-262 reads them (JSON Schema's
# dialect): `$` is the end of the text, never "before a final newline".

const SCHEMA_NAMES = ("call", "event", "interface", "saved", "rating")
const SCHEMA_KEYWORDS = Set(["\$schema", "\$id", "\$defs", "\$ref", "title", "description",
                             "type", "const", "enum", "pattern", "minLength", "minimum", "maximum",
                             "required", "properties", "additionalProperties", "propertyNames", "maxProperties",
                             "items", "minItems", "uniqueItems", "allOf", "anyOf", "oneOf", "not", "if", "then", "else"])

"""
A schema's pattern as ECMA-262 reads it: its one `\$`, the last anchor, is
the end of the text. A pattern that uses a class escape (`\\d`, `\\w`, `\\s`,
`\\b` and their negations) is refused: Julia's PCRE reads those with Unicode
properties, ECMA-262 does not (`\\d` is ASCII digits there, `\\s` its own
list of spaces), so a pattern using one could pass here and fail elsewhere.
"""
function ecma_pattern(p::AbstractString)
    occursin(r"\A[^$]*\$\)*\z", p) || occursin(r"\A[^$]*\z", p) ||
        error("a schema pattern with \$ other than as its last anchor: $p")
    occursin(r"(?<!\\)(?:\\\\)*\\[dDwWsSbB]", p) &&
        error("a schema pattern with a class escape (\\d \\w \\s \\b), which PCRE and ECMA-262 read differently: $p")
    Regex(replace(p, r"\$(?=\)*\z)" => s"\\z"))
end

"Every schema keyword the validator reads; a schema using another is not read."
function check_keywords(schema, where)
    schema isa Bool && return
    schema isa AbstractDict || error("$where: a schema is an object or a boolean")
    for (k, v) in schema
        k in SCHEMA_KEYWORDS || error("$where: the schema keyword $k is not one this package reads")
        if k in ("properties", "\$defs")
            for (n, s) in v
                check_keywords(s, "$where/$k/$n")
            end
        elseif k in ("allOf", "anyOf", "oneOf")
            foreach(((i, s),) -> check_keywords(s, "$where/$k/$i"), enumerate(v))
        elseif k in ("additionalProperties", "propertyNames", "items", "not", "if", "then", "else")
            check_keywords(v, "$where/$k")
        end
    end
end

struct Schemas
    docs::Dict{String,Any}              # by file name (call.schema.json)
    patterns::Dict{String,Regex}
end

function load_schemas()
    docs = Dict{String,Any}()
    patterns = Dict{String,Regex}()
    for name in SCHEMA_NAMES
        file = "$name.schema.json"
        doc = contract_json("schema/$file")
        check_keywords(doc, file)
        docs[file] = doc
    end
    function collect_patterns(x)
        if x isa AbstractDict
            p = get(x, "pattern", nothing)
            p isa AbstractString && (patterns[p] = ecma_pattern(p))
            foreach(collect_patterns, values(x))
        elseif x isa AbstractVector
            foreach(collect_patterns, x)
        end
    end
    foreach(collect_patterns, values(docs))
    Schemas(docs, patterns)
end
const SCHEMAS = load_schemas()

"The schema a `\$ref` names, and the document it is in (`#/\$defs/x`, `other.schema.json#/\$defs/x`, `other.schema.json`)."
function resolve_ref(ref::AbstractString, doc::AbstractString)
    file, pointer = occursin('#', ref) ? split(ref, '#'; limit=2) : (ref, "")
    file = isempty(file) ? doc : String(last(split(file, '/')))
    target = SCHEMAS.docs[file]
    for part in split(pointer, '/'; keepempty=false)
        target = target[replace(part, "~1" => "/", "~0" => "~")]
    end
    (target, file)
end

json_type_is(v, t) = t == "null" ? v === nothing :
                     t == "boolean" ? v isa Bool :
                     t == "integer" ? (v isa Real && !(v isa Bool) && isfinite(v) && isinteger(v)) :
                     t == "number" ? (v isa Real && !(v isa Bool)) :
                     t == "string" ? v isa AbstractString :
                     t == "array" ? v isa AbstractVector :
                     t == "object" ? v isa AbstractDict : false

"""
    schema_fault(schema, value, doc; at = "") -> String or nothing

Why `value` does not pass `schema` (a node of the document `doc`), as JSON
Schema draft 2020-12 reads the contract's schemas, or `nothing` when it
passes. The first fault found, with where it is in the value.
"""
function schema_fault(schema, v, doc::AbstractString, at::AbstractString="")
    schema === true && return nothing
    schema === false && return "$(at_text(at)): nothing is allowed here"
    if haskey(schema, "\$ref")
        target, tdoc = resolve_ref(schema["\$ref"], doc)
        f = schema_fault(target, v, tdoc, at)
        f === nothing || return f
    end
    if haskey(schema, "type")
        ts = schema["type"] isa AbstractVector ? schema["type"] : Any[schema["type"]]
        any(t -> json_type_is(v, t), ts) || return "$(at_text(at)): not of type $(join(ts, " or "))"
    end
    haskey(schema, "const") && !same_json(schema["const"], v) && return "$(at_text(at)): not $(LMCC.json_text(schema["const"]))"
    haskey(schema, "enum") && !any(x -> same_json(x, v), schema["enum"]) && return "$(at_text(at)): not one of the values it may be"
    if v isa AbstractString
        haskey(schema, "minLength") && length(v) < schema["minLength"] && return "$(at_text(at)): shorter than $(schema["minLength"])"
        haskey(schema, "pattern") && !occursin(SCHEMAS.patterns[schema["pattern"]], v) &&
            return "$(at_text(at)): does not match $(schema["pattern"])"
    end
    if v isa Real && !(v isa Bool)
        haskey(schema, "minimum") && v < schema["minimum"] && return "$(at_text(at)): less than $(schema["minimum"])"
        haskey(schema, "maximum") && v > schema["maximum"] && return "$(at_text(at)): more than $(schema["maximum"])"
    end
    if v isa AbstractDict
        for k in get(schema, "required", ())
            haskey(v, k) || return "$(at_text(at)): no $k"
        end
        haskey(schema, "maxProperties") && length(v) > schema["maxProperties"] && return "$(at_text(at)): more than $(schema["maxProperties"]) members"
        props = get(schema, "properties", nothing)
        extra = get(schema, "additionalProperties", nothing)
        names = get(schema, "propertyNames", nothing)
        for (k, x) in v
            key = String(k)
            if names !== nothing
                f = schema_fault(names, key, doc, "$at/$key (its name)")
                f === nothing || return f
            end
            if props !== nothing && haskey(props, key)
                f = schema_fault(props[key], x, doc, "$at/$key")
                f === nothing || return f
            elseif extra !== nothing
                f = schema_fault(extra, x, doc, "$at/$key")
                f === nothing || return f
            end
        end
    end
    if v isa AbstractVector
        haskey(schema, "minItems") && length(v) < schema["minItems"] && return "$(at_text(at)): fewer than $(schema["minItems"]) items"
        get(schema, "uniqueItems", false) === true && !allunique(LMCC.canonical_json.(v)) && return "$(at_text(at)): items not unique"
        if haskey(schema, "items")
            for (i, x) in enumerate(v)
                f = schema_fault(schema["items"], x, doc, "$at/$(i - 1)")
                f === nothing || return f
            end
        end
    end
    for s in get(schema, "allOf", ())
        f = schema_fault(s, v, doc, at)
        f === nothing || return f
    end
    if haskey(schema, "anyOf")
        any(s -> schema_fault(s, v, doc, at) === nothing, schema["anyOf"]) || return "$(at_text(at)): fits none of the shapes it may have"
    end
    if haskey(schema, "oneOf")
        n = count(s -> schema_fault(s, v, doc, at) === nothing, schema["oneOf"])
        n == 1 || return "$(at_text(at)): fits $(n == 0 ? "none" : "more than one") of the shapes it has one of"
    end
    haskey(schema, "not") && schema_fault(schema["not"], v, doc, at) === nothing && return "$(at_text(at)): has what it must not"
    if haskey(schema, "if")
        branch = schema_fault(schema["if"], v, doc, at) === nothing ? get(schema, "then", nothing) : get(schema, "else", nothing)
        if branch !== nothing
            f = schema_fault(branch, v, doc, at)
            f === nothing || return f
        end
    end
    nothing
end
at_text(at) = isempty(at) ? "the value" : at

"""
    schema_fault(name, value) -> String or nothing

Why `value` does not pass the contract's schema `name` (`"call"`,
`"event"`, `"interface"`, `"saved"`, `"rating"`), or `nothing`.
"""
schema_fault(name::AbstractString, v) = (file = "$name.schema.json"; schema_fault(SCHEMAS.docs[file], v, file, ""))
