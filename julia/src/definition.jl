# A definition becomes an lmcc signature (contract/functions.md, "The
# signature"): the fields in their order, the instructions from the name, the
# description and the guidance. And the sample input a version is rendered
# with (contract/calls.md, "Versions").

"""
One input or output: its name, its Julia type (or `OneOf`, or a shape when
loaded), its shape (without a default), words about it, and, for an input a
caller may leave out, its default: `default` the value sent (as JSON, in the
interface's shape) and `native` the value the function's code gets (a copy
of the value it was defined with; a loaded function runs no code of its own,
and its is the JSON).
"""
struct FieldDef
    name::String
    spec::Any
    shape::JObj
    desc::Union{Nothing,String}
    optional::Bool
    default::Any
    native::Any
end
FieldDef(name, spec, shape, desc) = FieldDef(String(name), spec, shape, desc, false, nothing, nothing)

"""
An optional input a call left out: its default. Its JSON (`data`, the
interface's `default`, which a saved folder keeps) is what the call sends,
logs and shows, never made again from the value: so a function sends the
same request before and after saving and loading, whatever the value's type
does when it is built or iterated. The function's own code gets `native`, a
copy of the value the function was defined with, anew for each call.
"""
struct LeftOut
    data::Any
    native::Any
end
LeftOut(x::FieldDef) = LeftOut(x.default, x.native)
jsonvalue(x::LeftOut) = LMCC.deepcopy_json(x.data)

"The inputs as the function's own code gets them: a default left out is a copy of its value, its own."
code_inputs(inputs::AbstractDict) = OrderedDict{String,Any}(k => v isa LeftOut ? deepcopy(v.native) : v for (k, v) in inputs)
FieldDef(name, spec; desc=nothing) = FieldDef(String(name), spec, shape_of(spec), desc === nothing || isempty(desc) ? nothing : String(desc))

"The field's lmcc shape: its shape without its own `default` (functions.md, \"The signature\")."
lmcc_shape(f::FieldDef) = haskey(f.shape, "default") ? data_shape(f.shape) : f.shape

"The field as the interface writes it (programs.md): an optional input's default is in its shape."
function interface_field(f::FieldDef; input::Bool)
    shape = LMCC.deepcopy_json(f.shape)
    f.optional && (shape["default"] = LMCC.deepcopy_json(f.default))
    out = LMCC.jobj("name" => f.name, "shape" => shape)
    f.desc === nothing || (out["desc"] = f.desc)
    f.spec isa Type && (out["type"] = string(f.spec))
    input && f.optional && (out["optional"] = true)
    out
end

"What every language's AI function comes down to (contract/functions.md, \"A definition\")."
struct Definition
    name::String
    description::String
    inputs::Vector{FieldDef}
    outputs::Vector{FieldDef}           # in order; the last is the answer
    written::Union{Nothing,String}      # the instructions as another language wrote them (a loaded function)
end

answer_name(d::Definition) = last(d.outputs).name

"""
The interface of an AI function's definition (programs.md, "How each program
has one"): its description, inputs (optional ones with their defaults) and
outputs, without the fields FunctAI adds.
"""
interface_of(d::Definition) = LMCC.jobj("description" => d.description,
    "inputs" => Any[interface_field(f; input=true) for f in d.inputs],
    "outputs" => Any[interface_field(f; input=false) for f in d.outputs])

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
        field = LMCC.jobj("name" => f.name, "direction" => "input", "shape" => lmcc_shape(f), "purpose" => "plain")
        f.desc === nothing || (field["desc"] = f.desc)
        push!(fields, field)
    end
    tools && push!(fields, LMCC.jobj("name" => "tools", "direction" => "input", "shape" => TOOL_LIST, "purpose" => "tools", "type" => "list[Tool]"))
    names = Set(f.name for f in vcat(d.inputs, d.outputs))
    reasoning && !("reasoning" in names) &&
        push!(fields, LMCC.jobj("name" => "reasoning", "direction" => "output", "shape" => LMCC.jobj("type" => "string"), "purpose" => "reasoning"))
    tools && push!(fields, LMCC.jobj("name" => "calls", "direction" => "output", "shape" => CALL_LIST, "purpose" => "tools.calls", "type" => "list[ToolCall]"))
    for f in d.outputs
        push!(fields, LMCC.jobj("name" => f.name, "direction" => "output", "shape" => lmcc_shape(f), "purpose" => "plain"))
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

"JSON indented by `width` spaces, as Python's `json.dumps(indent=width, ensure_ascii=False)` writes it."
function json_indented(v, depth::Int=0; width::Int=2)
    pad, close = " "^(width * (depth + 1)), " "^(width * depth)
    if v isa AbstractDict
        isempty(v) && return "{}"
        return "{\n" * join(("$pad$(LMCC.json_text(String(k))): $(json_indented(x, depth + 1; width))" for (k, x) in v), ",\n") * "\n$close}"
    elseif v isa AbstractVector
        isempty(v) && return "[]"
        return "[\n" * join(("$pad$(json_indented(x, depth + 1; width))" for x in v), ",\n") * "\n$close]"
    end
    LMCC.json_text(v)
end
