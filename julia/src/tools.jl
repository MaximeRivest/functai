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
the Julia function that runs it, and what it says of itself: `effects`
(`"reads"`: it only looks; `"changes"`: it writes, sends or pays; `nothing`:
it says nothing, which every approval rule treats as `"changes"`), `needs`
(the host names its code reaches and the secrets it uses, by name), `notes`
(the notes its code makes, as an interface declares them) and `code_id`
(what identifies its code beyond the function's own text). Made by
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
    needs::Union{Nothing,JObj}            # {"domains", "secrets"} (contract/tools.md, "What a tool needs")
    notes::Union{Nothing,Vector{Any}}     # its notes, as an interface's `notes` (contract/tools.md, "Notes a tool makes")
    code_id::Union{Nothing,String}        # what identifies its code beyond its text (contract/calls.md, "Setups")
    program::Any                          # the AI function or program it is (a program as a tool), or nothing
end
AITool(name, description, parameters, run, argnames, argspecs, effects=nothing) =
    AITool(name, description, parameters, run, argnames, argspecs, effects, nothing, nothing, nothing, nothing)
AITool(name, description, parameters, run, argnames, argspecs, effects, needs, notes, code_id) =
    AITool(name, description, parameters, run, argnames, argspecs, effects, needs, notes, code_id, nothing)

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

# a domain code needs (contract/calls.md, "Reach"): a host name, or "*." and one (every name under it), in lower case
const DOMAIN = r"\A(\*\.)?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\z"

"""
What code needs to reach, as data (contract/tools.md, "What a tool needs"):
`{"domains", "secrets"}`, the domains in lower case, each list sorted and
once each; `nothing` when nothing was said.
"""
function needs_value(v; where::AbstractString="needs")
    v === nothing && return nothing
    v isa Union{AbstractDict,NamedTuple} ||
        throw(ArgumentError("$where: needs is (domains = [...], secrets = [...]): the host names its code reaches and the names of the secrets it uses"))
    given = Dict(String(k) => x for (k, x) in pairs(v))
    extra = setdiff(keys(given), ("domains", "secrets"))
    isempty(extra) || throw(ArgumentError("$where: needs has domains and secrets, not $(join(sort!(collect(extra)), ", "))"))
    out = JObj()
    for key in ("domains", "secrets")
        items = get(given, key, String[])
        (items isa Union{AbstractVector,Tuple} && all(x -> x isa Union{AbstractString,Symbol} && !isempty(strip(String(x))), items)) ||
            throw(ArgumentError("$where: needs.$key is a list of names"))
        names = String[strip(String(x)) for x in items]
        if key == "domains"
            names = lowercase.(names)
            bad = findfirst(x -> !occursin(DOMAIN, x), names)
            bad === nothing || throw(ArgumentError("$where: $(repr(names[bad])) is not a host name (api.example.com) or *. and one (*.example.com)"))
        end
        out[key] = Any[sort!(unique(names))...]
    end
    out
end

"Several programs' needs as one (each list sorted, once each); `nothing` when none says any."
function merged_needs(needs...)
    said = [n for n in needs if n !== nothing]
    isempty(said) && return nothing
    LMCC.jobj("domains" => Any[sort!(unique(String[x for n in said for x in get(n, "domains", Any[])]))...],
              "secrets" => Any[sort!(unique(String[x for n in said for x in get(n, "secrets", Any[])]))...])
end

"""
    tool(f; name, description, parameters, effects, needs, notes, code_id)

A tool from a Julia function. Its arguments (names and types) come from the
function's one method, its description from its docstring; give `parameters`
(a JSON Schema) and `description` to say them yourself. `effects` says what
it does to the world: `:reads` (it only looks: a search, reading a file) or
`:changes` (it writes, sends or pays). Left out, it is unknown, which every
approval rule treats as `:changes`: forgetting to declare is safe. An AI
function or a program is a tool too: its inputs are the tool's.

- `needs`: what its code reaches, `(domains = ["api.example.com",
  "*.example.com"], secrets = ["EXAMPLE_API_KEY"])`: host names and the
  names of the secrets it uses, never their values. Behind a wall that lets
  through less (the `wall` setting), the function that has it is refused
  before its first call ([`ReachError`](@ref)), and what it may reach counts
  in its [`setup`](@ref).
- `notes`: the notes its code makes ([`note`](@ref)), by name: a type, or
  [`Note`](@ref)`(type; latest, outside)`, as pairs in order
  (`["page captured" => Page, "stage" => Note(OneOf(:walking, :done); latest = true)]`).
  Every AI function that has the tool declares them, and each note is checked
  where it is made.
- `code_id`: what identifies its code beyond the function's own text (a
  revision, a bundle's hash): two tools one factory made have one text.

```julia
"Look up an order by its number."
lookup(order::String) = orders[order]
"Refund an order."
refund(order::String) = payments.refund(order)
@ai tools = [tool(lookup; effects = :reads), tool(refund; effects = :changes, needs = (domains = ["api.stripe.com"],))] function helper(question::String)::String
    "Help with the order."
end
```

A tool returns text, a value (its JSON is shown), a picture, a sound, a video
or a document (an lm15 part: `LM15.image(path = "page.png")`), a list holding
one (a picture with its caption), or [`later`](@ref)`(key)` when its result
comes later.
"""
function tool(f; name=nothing, description=nothing, parameters=nothing, effects=nothing, needs=nothing, notes=nothing, code_id=nothing)
    if f isa AITool
        all(x -> x === nothing, (name, description, parameters, effects, needs, notes, code_id)) && return f
        tool_name = String(something(name, f.name))
        return AITool(tool_name, description === nothing ? f.description : String(description),
                      parameters === nothing ? f.parameters : LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in parameters)),
                      f.run, f.argnames, f.argspecs, effects === nothing ? f.effects : effects_value(effects),
                      needs === nothing ? f.needs : needs_value(needs; where="tool $tool_name"),
                      notes === nothing ? f.notes : notes_of(notes; where="tool $tool_name"),
                      code_id === nothing ? f.code_id : String(code_id), f.program)
    end
    eff = effects_value(effects)
    tool_name = String(something(name, string(nameof(f))))
    occursin(r"^[A-Za-z_][A-Za-z0-9_-]*$", tool_name) || throw(ArgumentError("a tool's name is letters, digits, _ and -: $(repr(tool_name))"))
    said = (needs_value(needs; where="tool $tool_name"), notes_of(notes; where="tool $tool_name"), code_id === nothing ? nothing : String(code_id))
    desc = description === nothing ? doc_text(f) : String(description)
    if parameters !== nothing
        return AITool(tool_name, desc, LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in parameters)), f, Symbol[], Any[], eff, said...)
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
    AITool(tool_name, desc, params, f, collect(names), specs, eff, said...)
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

# ------------------------------------------------------------------ what a tool returns (contract/tools.md)

"The kinds of lm15 part a tool's result may hold."
const RESULT_PARTS = ("text", "image", "audio", "video", "document")

"""
    Later

What a tool returns when its result comes later ([`later`](@ref)): the `key`
it is delivered under.
"""
struct Later
    key::String
end

"""
    later(key) -> Later

What a tool returns when its result comes later: it started something that
ends elsewhere (a job, a cloud browser, a person's reply) and whoever learns
the result delivers it under `key` (put an id in it: `"harvest:8f2c"`).

Only in a conversation's turn: the turn runs the step's other tool calls,
then stops, saved, and waits ([`Waiting`](@ref)); [`deliver!`](@ref)`(chat,
key, result)`, from any process that opens the conversation, gives the tool
its result and the turn goes on. The tool does not run again. Anywhere else
nothing could wait for it, and the call refuses `later-unavailable`
([`ConversationError`](@ref)). A result delivered before the turn waits for
it is used at once.

```julia
"Visit a site in a cloud browser; returns what it saw."
function start_harvest(url::String)
    job = harvests.start(url; webhook = "https://example.com/done")
    later("harvest:\$(job.id)")
end

# in the webhook's handler, any process:
deliver!(conversation(agent, "visits"; store = "chats/"), "harvest:\$job_id", report)
```
"""
function later(key::AbstractString)
    isempty(strip(key)) && throw(ArgumentError("a later result's key is a non-empty text (put an id in it: \"harvest:8f2c\")"))
    Later(String(key))
end

is_part(v) = v isa LM15.Part

"An lm15 part as its JSON form; a part read from a file holds the file's bytes, so the result is whole wherever it is read again."
function part_json(p)
    p.type in RESULT_PARTS || throw(ArgumentError("a tool's result holds text, pictures, sounds, videos and documents; not a $(p.type) part"))
    if p isa LM15.MediaPart && p.path !== nothing && p.data === nothing
        p = typeof(p)(; (f => getfield(p, f) for f in fieldnames(typeof(p)) if !(f in (:path, :data)))...,
                      data=Base64.base64encode(read(p.path)))
    end
    JObj(LM15.to_dict(p))
end

as_result_text(v) = v isa AbstractString ? String(v) : LMCC.json_text(logvalue(v))

"""
What a tool returned, as the model is shown it (contract/tools.md, "What a
tool returns"): text (a text; any other value, as JSON), or, when it
returned a part (a picture, a sound, a document) or a list holding one, a
list of lm15 parts in their JSON form: each text a text part, each part
itself, each other value a text part of its JSON.
"""
function result_of(out)
    is_part(out) && return Any[part_json(out)]
    if out isa Union{AbstractVector,Tuple} && any(is_part, out)
        return Any[is_part(x) ? part_json(x) : LMCC.textpart(as_result_text(x)) for x in out]
    end
    as_result_text(out)
end

"A tool result's text: itself, or its text parts joined by a new line."
result_text(output::AbstractString) = String(output)
result_text(output) = join((String(p["text"]) for p in output if p isa AbstractDict && get(p, "type", nothing) == "text"), "\n")

"""
A tool result given as data (a plugin's change, a result given for a tool
that may have run, a result delivered later): text, or a list of lm15 parts
(their JSON form, or lm15's parts).
"""
function check_result(output, where::AbstractString)
    output isa AbstractString && return String(output)
    is_part(output) && return Any[part_json(output)]
    if output isa Union{AbstractVector,Tuple} && !isempty(output)
        return Any[if is_part(p)
                       part_json(p)
                   elseif p isa AbstractDict && get(p, "type", nothing) in RESULT_PARTS
                       LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in p))
                   elseif p isa AbstractString
                       LMCC.textpart(String(p))
                   else
                       throw(ArgumentError("$where: a part of a tool's result is text, a picture, a sound, a video or a document (an lm15 part), not $(repr(p))"))
                   end for p in output]
    end
    throw(ArgumentError("$where is text, or a list of lm15 parts (text, pictures, sounds, videos, documents)"))
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
