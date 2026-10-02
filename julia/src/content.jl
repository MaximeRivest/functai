# What the log keeps of a call (contract/calls.md, "Content"): `log_content`
# per field, in layers that only remove (a program's own setting, each
# enclosing `with_settings` block, `configure!`, the environment), and the
# record a call writes when some value is not kept. The kept form of stream
# events follows the same decisions (events.jl).

"""
    LogContentError

A `log_content` map refused (code `"log-content-field"`, contract/calls.md,
"Content"): in a program's own settings, a name that is not one of its
fields (a misspelling would write the value it was meant to keep out);
anywhere, a key that is neither a field name nor `"*"`.
"""
struct LogContentError <: FunctAIError
    code::String
    field::String
    msg::String
end
LogContentError(field, msg) = LogContentError("log-content-field", String(field), msg)
Base.showerror(io::IO, e::LogContentError) = print(io, "LogContentError [", e.code, "] (", e.field, "): ", e.msg)

"""
A `log_content` setting as data: `true`, `false`, or a map of field names
(and `"*"`, every field the map does not name) to booleans. A key that is
not a name refuses wherever it is set.
"""
function content_setting(v)
    v isa Bool && return v
    v isa Union{NamedTuple,AbstractDict} ||
        throw(ArgumentError("log_content is true, false, or a map of field names to true or false (`(transcript = false,)`, `Dict(\"*\" => false, \"question\" => true)`), not $(repr(v))"))
    out = OrderedDict{String,Bool}()
    for (k, x) in pairs(v)
        key = string(k)
        (key == "*" || is_name(key)) ||
            throw(LogContentError(key, "$(repr(key)) is neither a field name nor \"*\" (other keys are kept for kinds of data)"))
        x isa Bool || throw(ArgumentError("log_content: $key is true or false, not $(repr(x))"))
        out[key] = x
    end
    out
end

"A program's own map may name only its fields: checked when it is defined (or loaded, or configured)."
function check_own_content(own::AbstractDict{Symbol}, fields, program::AbstractString)
    v = get(own, :log_content, nothing)
    v isa AbstractDict || return nothing
    names = vcat(fields.inputs, fields.outputs)
    for key in keys(v)
        key == "*" || key in names ||
            throw(LogContentError(key, "$program's own log_content names $key, which is not one of its fields ($(join(names, ", ")))" *
                                       (key == "tools" ? ": its tools are its state, not a field of a call" : "")))
    end
    nothing
end

"Whether one layer's setting drops a field: false, or a map that says false for it, or whose \"*\" is false and does not name it."
says_drop(v::Bool, name) = !v
says_drop(v::AbstractDict, name) = haskey(v, name) ? !v[name] : get(v, "*", nothing) === false
says_drop(::Nothing, name) = false

const CONTENT_OFF = ("0", "false", "no", "off")

"""
For each field, whether its value is written (calls.md, "Content"): only
when no layer drops it (`layers`, closest first; `environment` the value of
`FUNCTAI_LOG_CONTENT`, or `nothing` when unset), and a field FunctAI added
(reasoning, calls) only when no input or output of the program is dropped
(it can quote any of them); dropping one added field drops only it.
"""
function content_keep(fields, layers, environment)
    off = environment !== nothing && lowercase(strip(environment)) in CONTENT_OFF
    keep = OrderedDict{String,Bool}()
    for name in vcat(fields.inputs, fields.outputs)
        keep[name] = !off && !any(v -> says_drop(v, name), layers)
    end
    if !all(keep[n] for n in vcat(fields.inputs, fields.outputs) if !(n in fields.added))
        for n in fields.added
            keep[n] = false
        end
    end
    keep
end

"The fields a `log_content` layer in effect drops for a call: an error message never quotes their values (programs.md)."
function dropped_fields(own::AbstractDict{Symbol}, fields)
    try
        keep = content_keep(fields, content_layers(own), environment_content())
        String[n for (n, x) in keep if !x]
    catch err
        err isa InterruptException && rethrow()
        ["*"]
    end
end

"The `log_content` layers around a program's call, closest first: its own, each enclosing block, `configure!`'s."
content_layers(own::AbstractDict{Symbol}) = Any[v for v in (get(own, :log_content, nothing),
    (get(b, :log_content, nothing) for b in Iterators.reverse(SCOPED_LAYERS[]))...,
    lock(() -> get(GLOBAL_SETTINGS, :log_content, nothing), SETTINGS_LOCK)) if v !== nothing]

environment_content() = (v = get(ENV, "FUNCTAI_LOG_CONTENT", nothing); v === nothing ? nothing : String(v))

"""
Which of a call's fields are kept, as the kept form of its events reads it:
by input and by output; and `holds`, the outputs its `done` event's value
holds when the call says so (`nothing`: as its program's kind says, an AI
function's answer or a module's outputs).
"""
struct Keep
    inputs::OrderedDict{String,Bool}
    outputs::OrderedDict{String,Bool}
    holds::Union{Nothing,Vector{String}}
end
Keep(inputs::AbstractDict, outputs::AbstractDict) =
    Keep(OrderedDict{String,Bool}(inputs), OrderedDict{String,Bool}(outputs), nothing)
"The `Keep` of a call's fields `(inputs, outputs, added[, holds])` from `content_keep`'s decisions."
Keep(fields::NamedTuple, keep::AbstractDict) =
    Keep(OrderedDict{String,Bool}(n => keep[n] for n in fields.inputs), OrderedDict{String,Bool}(n => keep[n] for n in fields.outputs),
         haskey(fields, :holds) ? String[fields.holds...] : nothing)
whole(k::Keep) = all(values(k.inputs)) && all(values(k.outputs))
kept(k::Keep, name) = get(k.inputs, name, get(k.outputs, name, false))     # a name that is no field: not kept

const ALWAYS_KEPT = ("functai_call", "id", "parent", "root", "program", "started", "seconds", "sizes", "model", "usage",
                     "confidence", "caller", "process", "saw", "escalated", "truncated", "journal", "invocation",
                     "conversation", "writer", "replayable")
const EXCHANGE_DROPPED = ("request", "response", "request_hash")
const ERROR_KEPT = ("type", "code")

function exchange_without_content(x)
    ex = JObj(String(a) => LMCC.deepcopy_json(b) for (a, b) in x if !(a in EXCHANGE_DROPPED))
    haskey(ex, "error") && (ex["error"] = error_without_content(ex["error"]))
    ex
end

error_without_content(e) = e isa AbstractDict ? JObj(String(k) => v for (k, v) in e if k in ERROR_KEPT) : e

"""
    written_record(record, keep::Keep) -> record

The record a call writes (calls.md, "Content", "What the record keeps"): the
whole record when every field is kept; otherwise `content: false`,
`omitted` naming the fields not kept, only the kept values, no exchange
request, reply or request hash, and of each error only its type and code.
"""
function written_record(rec::AbstractDict, keep::Keep)
    whole(keep) && return LMCC.deepcopy_json(rec)
    k(name) = kept(keep, name)
    out = JObj()
    for (key, v) in rec
        if key in ALWAYS_KEPT
            out[key] = LMCC.deepcopy_json(v)
        elseif key == "content"
            out["content"] = false
            out["omitted"] = LMCC.jobj("inputs" => Any[n for (n, x) in keep.inputs if !x],
                                       "outputs" => Any[n for (n, x) in keep.outputs if !x])
        elseif key == "inputs"
            inputs = JObj(n => LMCC.deepcopy_json(x) for (n, x) in v if k(n))
            isempty(inputs) || (out["inputs"] = inputs)
        elseif key == "outputs"
            if v === nothing
                out["outputs"] = nothing
            else
                outputs = JObj(n => LMCC.deepcopy_json(x) for (n, x) in v if k(n))
                isempty(outputs) || (out["outputs"] = outputs)
            end
        elseif key == "described"
            described = JObj(d => Any[n for n in names if k(n)] for (d, names) in v)
            any(!isempty, values(described)) && (out["described"] = described)
        elseif key in ("returned", "steps", "sections")
            nothing                         # each can hold any value of the call (a summary quotes the turns): only whole
        elseif key == "changes"
            out["changes"] = Any[JObj(a => LMCC.deepcopy_json(b) for (a, b) in c if a != "change") for c in v]   # who, not what
        elseif key == "probabilities"
            probabilities = JObj(n => LMCC.deepcopy_json(x) for (n, x) in v if k(n))
            isempty(probabilities) || (out["probabilities"] = probabilities)
        elseif key == "error"
            out["error"] = error_without_content(v)
        elseif key == "exchanges"
            out["exchanges"] = Any[exchange_without_content(x) for x in v]
        end
    end
    out
end
