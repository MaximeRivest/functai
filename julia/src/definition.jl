# A definition becomes an lmcc signature (contract/functions.md, "The
# signature"): the fields in their order, the instructions from the name, the
# description and the guidance. And the sample input a version is rendered
# with (contract/calls.md, "Versions").

"One input or output: its name, its Julia type (or `OneOf`, or a shape when loaded), its shape, words about it."
struct FieldDef
    name::String
    spec::Any
    shape::JObj
    desc::Union{Nothing,String}
end
FieldDef(name, spec; desc=nothing) = FieldDef(String(name), spec, shape_of(spec), desc === nothing || isempty(desc) ? nothing : String(desc))

"What every language's AI function comes down to (contract/functions.md, \"A definition\")."
struct Definition
    name::String
    description::String
    inputs::Vector{FieldDef}
    outputs::Vector{FieldDef}           # in order; the last is the answer
    written::Union{Nothing,String}      # the instructions as another language wrote them (a loaded function)
end

answer_name(d::Definition) = last(d.outputs).name

const TOOL_LIST = LMCC.jobj("type" => "array", "items" => LMCC.jobj("type" => "object", "properties" => LMCC.jobj(
        "name" => LMCC.jobj("type" => "string"),
        "description" => LMCC.jobj("anyOf" => Any[LMCC.jobj("type" => "string"), LMCC.jobj("type" => "null")]),
        "parameters" => LMCC.jobj("anyOf" => Any[LMCC.jobj("type" => "object"), LMCC.jobj("type" => "null")])),
    "required" => Any["name", "description", "parameters"]))
const CALL_LIST = LMCC.jobj("type" => "array", "items" => LMCC.jobj("type" => "object", "properties" => LMCC.jobj(
        "id" => LMCC.jobj("type" => "string"), "name" => LMCC.jobj("type" => "string"), "input" => LMCC.jobj("type" => "object")),
    "required" => Any["id", "name", "input"]))

"The instructions: an improved one as it is (trimmed), else the name, the description and the guidance."
function instructions_of(d::Definition, improved, include_name::Bool)
    improved === nothing || return trim_white(improved)
    d.written === nothing || return d.written
    head = String[]
    include_name && !isempty(d.name) && push!(head, "Function: $(d.name)")
    description = trim_white(d.description)
    isempty(description) || push!(head, description)
    top = trim_white(join(head, "\n\n"))
    lines = String[]
    described_inputs = [f for f in d.inputs if f.desc !== nothing]
    if !isempty(described_inputs)
        push!(lines, "Parameter guidance:")
        append!(lines, ["- $(f.name): $(f.desc)" for f in described_inputs])
        push!(lines, "")
    end
    described_outputs = [f for f in d.outputs if f.desc !== nothing]
    if !isempty(described_outputs)
        push!(lines, "Output guidance:")
        append!(lines, ["- $(f.name): $(f.desc)" for f in described_outputs])
        push!(lines, "")
    end
    guidance = trim_white(join(lines, "\n"))
    isempty(guidance) && return top
    top * (isempty(top) ? "" : "\n\n") * guidance
end

"The lmcc signature: the inputs, the tools, reasoning, the tool calls, the outputs (outputs carry no desc)."
function signature_of(d::Definition, improved, include_name::Bool, reasoning::Bool, tools::Bool)
    fields = Any[]
    for f in d.inputs
        field = LMCC.jobj("name" => f.name, "direction" => "input", "shape" => f.shape, "purpose" => "plain")
        f.desc === nothing || (field["desc"] = f.desc)
        push!(fields, field)
    end
    tools && push!(fields, LMCC.jobj("name" => "tools", "direction" => "input", "shape" => TOOL_LIST, "purpose" => "tools", "type" => "list[Tool]"))
    names = Set(f.name for f in vcat(d.inputs, d.outputs))
    reasoning && !("reasoning" in names) &&
        push!(fields, LMCC.jobj("name" => "reasoning", "direction" => "output", "shape" => LMCC.jobj("type" => "string"), "purpose" => "reasoning"))
    tools && push!(fields, LMCC.jobj("name" => "calls", "direction" => "output", "shape" => CALL_LIST, "purpose" => "tools.calls", "type" => "list[ToolCall]"))
    for f in d.outputs
        push!(fields, LMCC.jobj("name" => f.name, "direction" => "output", "shape" => f.shape, "purpose" => "plain"))
    end
    LMCC.signature_from_dict(LMCC.jobj("instructions" => instructions_of(d, improved, include_name), "fields" => fields))
end

"A value for a shape, for the sample input (contract/calls.md, \"Versions\")."
function sample_value(shape::AbstractDict)
    haskey(shape, "enum") && return first(shape["enum"])
    if haskey(shape, "anyOf")
        options = [s for s in shape["anyOf"] if get(s, "type", nothing) != "null"]
        return isempty(options) ? nothing : sample_value(first(options))
    end
    t = get(shape, "type", nothing)
    t == "string" ? "example text" : t == "integer" ? 3 : t == "number" ? 2.5 : t == "boolean" ? true :
    t == "array" ? Any[] : t == "object" ? JObj() : t == "null" ? nothing : "example text"
end

"The sample input: a value for each plain input."
sample_inputs(sig::LMCC.Signature) =
    JObj(f.name => sample_value(f.shape) for f in sig.fields if f.direction == "input" && f.purpose == "plain")

"""
A call's `program.signature` (contract/calls.md): lmcc's fingerprint with every
type name empty, so the shapes decide, not how a language spells types.
"""
signature_id(sig::LMCC.Signature) = LMCC.sha256_of(Any[LMCC.jobj("direction" => f.direction, "name" => f.name,
    "purpose" => isempty(f.purpose) ? "plain" : f.purpose, "shape" => f.shape, "type" => "") for f in sig.fields])

"Values as their fields expect them: JSON, and a non-text value given to a text input written as text."
function prepare_inputs(sig::LMCC.Signature, values::AbstractDict)
    out = JObj()
    for f in sig.fields
        (f.direction == "input" && haskey(values, f.name)) || continue
        v = values[f.name]
        v = v isa AbstractDict && !(v isa JObj) ? LMCC.deepcopy_json(jsonvalue(v)) : jsonvalue(v)
        if get(f.shape, "type", nothing) == "string" && !haskey(f.shape, "enum") && v !== nothing && !(v isa AbstractString)
            v = v isa Union{AbstractDict,AbstractVector} ? json_indented(v) : string(v)
        end
        out[f.name] = v
    end
    out
end

"JSON indented by two spaces, as Python's `json.dumps(indent=2, ensure_ascii=False)` writes it."
function json_indented(v, depth::Int=0)
    pad, close = "  "^(depth + 1), "  "^depth
    if v isa AbstractDict
        isempty(v) && return "{}"
        return "{\n" * join(("$pad$(LMCC.json_text(String(k))): $(json_indented(x, depth + 1))" for (k, x) in v), ",\n") * "\n$close}"
    elseif v isa AbstractVector
        isempty(v) && return "[]"
        return "[\n" * join(("$pad$(json_indented(x, depth + 1))" for x in v), ",\n") * "\n$close]"
    end
    LMCC.json_text(v)
end
