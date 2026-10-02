# Tools: Julia functions the model may call. A tool is described to the model
# by its name, its docstring and the JSON Schema of its arguments, all read
# from the function itself:
#
#     "Current weather in a city."
#     weather(city::String) = …
#     @ai tools = [weather] function assistant(question::String)::String
#         "Answer, using tools when needed."
#     end

"""
    AITool

A tool the model may call: `name`, `description`, `parameters` (JSON Schema),
the Julia function that runs it, and its `effects`: `"reads"` (it only
looks), `"changes"` (it writes, sends or pays), or `nothing` (it says
nothing, which every approval rule treats as `"changes"`). Made by
[`tool`](@ref), or from a function given in `tools = [...]`.
"""
struct AITool
    name::String
    description::Union{Nothing,String}
    parameters::JObj
    run::Any
    argnames::Vector{Symbol}
    argspecs::Vector{Any}
    effects::Union{Nothing,String}
end
AITool(name, description, parameters, run, argnames, argspecs) = AITool(name, description, parameters, run, argnames, argspecs, nothing)

const EFFECTS = ("reads", "changes")
function effects_value(effects)
    effects === nothing && return nothing
    e = String(effects)
    e in EFFECTS || throw(ArgumentError("effects is :reads or :changes (or left out: unknown, which counts as :changes); not $(repr(effects))"))
    e
end

Base.show(io::IO, t::AITool) = print(io, "tool ", t.name, "(", join(("$n::$(s isa Type ? s : typeof(s))" for (n, s) in zip(t.argnames, t.argspecs)), ", "), ")")

"The docstring of a function, as text, or `nothing` when it has none."
function doc_text(f)
    mod, sym = parentmodule(f), nameof(f)
    b = Base.Docs.Binding(mod, sym)
    meta = Base.Docs.meta(mod)
    haskey(meta, b) || return nothing
    texts = String[]
    for d in values(meta[b].docs)
        push!(texts, join(string.(collect(d.text))))
    end
    text = trim_white(join(texts, "\n\n"))
    isempty(text) ? nothing : text
end

"""
    tool(f; name, description, parameters, effects)

A tool from a Julia function. Its arguments (names and types) come from the
function's one method, its description from its docstring; give `parameters`
(a JSON Schema) and `description` to say them yourself. `effects` says what
it does to the world: `:reads` (it only looks: a search, reading a file) or
`:changes` (it writes, sends or pays). Left out, it is unknown, which every
approval rule treats as `:changes`: forgetting to declare is safe. An AI
function or a program is a tool too: its inputs are the tool's.

```julia
"Look up an order by its number."
lookup(order::String) = orders[order]
"Refund an order."
refund(order::String) = payments.refund(order)
@ai tools = [tool(lookup; effects = :reads), tool(refund; effects = :changes)] function helper(question::String)::String
    "Help with the order."
end
```
"""
function tool(f; name=nothing, description=nothing, parameters=nothing, effects=nothing)
    if f isa AITool
        (name === nothing && description === nothing && parameters === nothing && effects === nothing) && return f
        return AITool(something(name, f.name), description === nothing ? f.description : String(description),
                      parameters === nothing ? f.parameters : LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in parameters)),
                      f.run, f.argnames, f.argspecs, effects === nothing ? f.effects : effects_value(effects))
    end
    eff = effects_value(effects)
    tool_name = String(something(name, string(nameof(f))))
    occursin(r"^[A-Za-z_][A-Za-z0-9_-]*$", tool_name) || throw(ArgumentError("a tool's name is letters, digits, _ and -: $(repr(tool_name))"))
    desc = description === nothing ? doc_text(f) : String(description)
    if parameters !== nothing
        return AITool(tool_name, desc, LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in parameters)), f, Symbol[], Any[], eff)
    end
    ms = [m for m in methods(f) if m.nargs >= 1]
    length(ms) == 1 || throw(ArgumentError("tool $tool_name: $(length(ms)) methods, so its arguments are not clear; " *
                                           "define one method, or give parameters = Dict(\"type\" => \"object\", …)"))
    m = only(ms)
    m.isva && throw(ArgumentError("tool $tool_name: a tool takes named arguments, not varargs"))
    names = Base.method_argnames(m)[2:end]
    specs = Any[m.sig.parameters[2:end]...]
    any(n -> n === Symbol("#unused#") || n === Symbol(""), names) &&
        throw(ArgumentError("tool $tool_name: every argument needs a name"))
    props = JObj(String(n) => shape_of(T; where="$tool_name.$n") for (n, T) in zip(names, specs))
    # no additionalProperties: Gemini refuses the keyword in function declarations
    params = LMCC.jobj("type" => "object", "properties" => props, "required" => Any[String(n) for n in names])
    AITool(tool_name, desc, params, f, collect(names), specs, eff)
end

tool_data(t::AITool) = LMCC.jobj("name" => t.name, "description" => t.description, "parameters" => t.parameters)

"Run a tool on the model's input (a JSON object): each argument read as its type."
function invoke_tool(t::AITool, input::AbstractDict)
    isempty(t.argnames) && return t.run(input)
    args = map(t.argnames, t.argspecs) do n, T
        haskey(input, String(n)) || (T >: Nothing ? (return nothing) : throw(ArgumentError("the model gave no $(n)")))
        v = input[String(n)]
        try
            fromjson(T, v, String(n))
        catch err
            err isa LMCC.Refusal ? throw(ArgumentError(err.hint)) : rethrow()
        end
    end
    t.run(args...)
end

"""
What a tool says it does (contract/tools.md, "Effects"): its `effects`; an
AI function used as a tool reads, unless one of its own tools changes things
or says nothing; a program's code may do anything, so it reads only when it
says so.
"""
effects_of(t::AITool) = t.effects

# ------------------------------------------------------------------ asking a person (contract/tools.md)

"""
    Approval

One tool call a person is asked about: `call` (the id of the AI function's
call that asked), `invocation` (the tool call's number in that call), `id`
(the id the model gave it), `name` and `input` (the tool and what it would
be given), `effects`, `path` (where it is, by names: `support/answer/refund`,
for rules written before any call exists), `site` (the asking call's place
in its tree), `plugin` (which plugin asks) and `question` (why, when it says).
"""
struct Approval
    call::String
    invocation::Int
    id::String
    name::String
    input::Any
    effects::Union{Nothing,String}
    path::String
    site::String
    plugin::String
    question::Union{Nothing,String}
end
Approval(call, invocation, id, name, input, effects, path, site) =
    Approval(String(call), Int(invocation), String(id), String(name), input, effects, String(path), String(site), "approval", nothing)
with_input(a::Approval, input) = Approval(a.call, a.invocation, a.id, a.name, input, a.effects, a.path, a.site, a.plugin, a.question)
asked_by(a::Approval, plugin, question) = Approval(a.call, a.invocation, a.id, a.name, a.input, a.effects, a.path, a.site,
                                                   String(plugin), question === nothing ? nothing : String(question))
Base.show(io::IO, a::Approval) = print(io, "Approval(", a.path, " #", a.invocation, ", ", something(a.effects, "effects unknown"), ")")

function approval_json(a::Approval)
    out = LMCC.jobj("call" => a.call, "invocation" => a.invocation, "id" => a.id, "name" => a.name, "input" => logvalue(a.input),
                    "effects" => a.effects, "path" => a.path, "site" => a.site, "plugin" => a.plugin)
    a.question === nothing || (out["question"] = a.question)
    out
end
function approval_from(d::AbstractDict)
    Approval(String(d["call"]), Int(d["invocation"]), String(d["id"]), String(d["name"]), get(d, "input", nothing),
             get(d, "effects", nothing), String(something(get(d, "path", nothing), d["name"])), String(something(get(d, "site", nothing), "")),
             String(something(get(d, "plugin", nothing), "approval")), get(d, "question", nothing))
end

"A tool call's approval path: the names of the calls from the outermost one to the call that asked, then the tool's."
approval_path(site::AbstractString, name) = join(vcat([String(first(split(p, '#'))) for p in split(site, '/') if !isempty(p)], String(name)), "/")

const RULES = ("changes", "all")
const DENIED = "The person did not allow this call."

"An `approve` setting: a function (asked at once), `:changes`, `:all`, or tool names and approval paths."
function approve_setting(v)
    v === nothing && return nothing
    v isa Function && return v
    v isa Union{Symbol,AbstractString} && String(v) in RULES && return String(v)
    v isa Union{AbstractVector,Tuple,AbstractSet} && all(x -> x isa Union{AbstractString,Symbol} && !isempty(String(x)), v) &&
        return String[String(x) for x in v]
    throw(ArgumentError("approve is a function, :changes, :all, or a list of tool names and approval paths; not $(repr(v))"))
end

"""
Whether a rule asks a person about this tool call (contract/tools.md):
`"changes"` (and a function, asked what `"changes"` asks) for a tool that
changes things or says nothing; `"all"` for every one; a list for a tool
named in it, or whose path is in it or ends with `/<entry>`.
"""
function asks(rule, a::Approval)
    rule === nothing && return false
    (rule isa Function || rule == "changes") && return a.effects != "reads"
    rule == "all" && return true
    any(e -> e == a.name || e == a.path || endswith(a.path, "/" * e), rule)
end

"`(allowed, reason)` from what an approve function answered: `true`, `false`, or a reason to refuse (a text)."
function verdict_of(answer)
    answer === true && return (true, nothing)
    (answer === false || answer === nothing) && return (false, nothing)
    answer isa AbstractString && return (false, String(answer))
    throw(ArgumentError("an approve function answers true, false, or a reason to refuse (a text); not $(repr(answer))"))
end

"What the model is shown for a refused tool call."
denial(reason) = reason === nothing || isempty(reason) ? DENIED : "$DENIED Reason: $reason"

# approvals a call of this process waits for on a stream: (call, invocation, plugin) => (allowed, reason, by)
const PENDING_LOCK = Threads.Condition()
const PENDING_ANSWERS = Dict{Tuple{String,Int,String},Tuple{Bool,Union{Nothing,String},Union{Nothing,String}}}()
const PENDING_ASKED = Dict{Tuple{String,Int,String},Approval}()

"Answer an approval a call of this process waits for (on a stream)."
function answer_here(call::AbstractString, invocation::Integer, allowed::Bool; reason=nothing, by=nothing, plugin="approval")
    lock(PENDING_LOCK) do
        PENDING_ANSWERS[(String(call), Int(invocation), String(plugin))] = (allowed, reason === nothing ? nothing : String(reason),
                                                                           by === nothing ? nothing : String(by))
        notify(PENDING_LOCK)
    end
    nothing
end

waiting_here() = lock(() -> [a for (k, a) in PENDING_ASKED if !haskey(PENDING_ANSWERS, k)], PENDING_LOCK)
