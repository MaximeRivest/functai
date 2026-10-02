# Plugins (contract/plugins.md): code that changes what programs do, through
# seven hooks, with every change returned as data and recorded, so a rated
# call is asked again as it was.
#
#     modes = Plugin("modes"; version = "1.0.0")
#     on!(modes, :before_call) do call
#         Change(sections = ["Answer carefully. Cite the file you read."], tools = ["read_note"])
#     end
#     configure!(plugins = [modes])          # every call; or a conversation's, or a function's own
#
# FunctAI's own features are plugins written with these hooks and nothing
# private: `approve =` is the `approval` plugin, `compaction()` and
# `delegate()` are in builtins.jl.

const PLUGIN_API = 1
const PLUGIN_APIS = (1,)

# hook => (what it does, the Change fields it may return)
const HOOKS = OrderedDict{Symbol,Tuple{Symbol,Tuple}}(
    :turn_start => (:change, (:inputs,)),
    :context => (:change, (:keep, :without, :sections)),
    :before_call => (:change, (:instruction, :sections, :lm, :settings, :tools)),
    :request => (:change, ()),                     # returns an LM15.Request: the escape hatch
    :tool_call => (:change, (:inputs, :block)),
    :tool_result => (:change, (:output,)),
    :turn_end => (:hear, ()))

const PLUGIN_NAME = r"^[a-z][a-z0-9_-]{0,63}$"

"""
    Plugin(name; version = "0.0.0", api = 1, description = "", hooks...)

A named, versioned set of hooks: functions FunctAI calls at fixed points of a
call or a conversation's turn (contract/plugins.md). Each is given an event
and returns a [`Change`](@ref) (or `nothing`: no opinion). The hooks:

| hook | when | a `Change` of |
|---|---|---|
| `turn_start` | a conversation's turn is sent, before it is recorded | `inputs` |
| `context` | its earlier turns are chosen | `keep`, `without`, `sections` |
| `before_call` | an AI function is about to be asked (helpers too) | `instruction`, `sections`, `lm`, `settings`, `tools` |
| `request` | a provider request is about to be sent | returns another `LM15.Request` |
| `tool_call` | a tool is about to run | `inputs`, `block`; or `ask(event)` a person |
| `tool_result` | a tool ran | `output` |
| `turn_end` | a turn ended (hears only; may [`remember!`](@ref) entries) | |

Register them as keywords, or with [`on!`](@ref):

```julia
guard = Plugin("no-deletes"; version = "1.0.0",
               tool_call = t -> t.name == "delete_file" ? Change(block = "deleting files is not allowed here") : nothing)
configure!(plugins = [guard])
```

`name` is lower-case letters, digits, `_` and `-` (its changes and entries
are named by it); `api` is the plugin API it was written for (`1`; another
refuses `plugin-api`).
"""
struct Plugin
    name::String
    version::String
    api::Int
    description::String
    handlers::OrderedDict{Symbol,Vector{Any}}
end
function Plugin(name::AbstractString; version="0.0.0", api::Integer=PLUGIN_API, description::AbstractString="", hooks...)
    occursin(PLUGIN_NAME, name) || throw(PluginError("plugin-name", "a plugin's name is lower-case letters, digits, '_' or '-' " *
                                                     "(at most 64), starting with a letter; not $(repr(name))"))
    api in PLUGIN_APIS || throw(PluginError("plugin-api", "plugin $name is written for plugin API $api; this FunctAI implements " *
                                            join(PLUGIN_APIS, ", "); plugin=name))
    p = Plugin(String(name), string(version), Int(api), String(description), OrderedDict{Symbol,Vector{Any}}())
    for (hook, f) in hooks
        on!(f, p, hook)
    end
    p
end
Base.show(io::IO, p::Plugin) = print(io, "Plugin(", repr(p.name), ", ", p.version, ": ",
                                     isempty(p.handlers) ? "no hooks" : join(keys(p.handlers), ", "), ")")

"""
    on!(f, plugin, hook)

Register `f` (a function of one event) for `hook` (`:turn_start`, `:context`,
`:before_call`, `:request`, `:tool_call`, `:tool_result`, `:turn_end`). Within
a plugin, handlers run in the order they were registered. A hook that does
not exist refuses `plugin-hook`: a misspelt hook would otherwise never run.

```julia
on!(modes, :before_call) do call
    call.function == "answer" ? Change(sections = ["Cite your sources."]) : nothing
end
```
"""
function on!(f, p::Plugin, hook)
    h = Symbol(hook)
    haskey(HOOKS, h) || throw(PluginError("plugin-hook", "$(p.name): there is no hook $(repr(hook)); hooks: " *
                                          join(keys(HOOKS), ", "); plugin=p.name, hook=string(hook)))
    f isa Union{Function,Base.Callable} || applicable(f, nothing) ||
        throw(ArgumentError("$(p.name).$h: a handler is a function of one event"))
    push!(get!(() -> Any[], p.handlers, h), f)
    p
end

"A plugin's manifest, as data: name, version, api, description, the hooks it uses."
describe(p::Plugin) = LMCC.jobj("name" => p.name, "version" => p.version, "api" => p.api, "description" => p.description,
                                "hooks" => Any[String(h) for h in sort!(collect(keys(p.handlers)))])

"""
    Change(; instruction, sections, lm, settings, tools, keep, without, inputs, block, output)

What a hook changes, as data (contract/plugins.md, "The fields"). Each hook
accepts some fields; another refuses `plugin-change`.

- `instruction`: the instruction itself, for this call; `sections`: text
  added after it, in order (`before_call`, `context`);
- `lm`: the model for this call; `settings`: lm15 settings for it
  (`temperature`, `max_tokens`, `reasoning`, …) (`before_call`);
- `tools`: the names of the tools offered, from the function's own (`before_call`);
- `keep`: the ids of the earlier turns shown; `without`: fields left out of
  earlier turns, a list for every turn or `Dict(turn => names)` (`context`);
- `inputs`: inputs replaced, by name (`turn_start`: the turn's; `tool_call`: the tool's);
- `block`: the tool may not run, and why (`tool_call`);
- `output`: the tool's result as the model is shown it (`tool_result`).
"""
struct Change
    instruction::Any
    sections::Any
    lm::Any
    settings::Any
    tools::Any
    keep::Any
    without::Any
    inputs::Any
    block::Any
    output::Any
end
const CHANGE_FIELDS = fieldnames(Change)
Change(; instruction=nothing, sections=nothing, lm=nothing, settings=nothing, tools=nothing, keep=nothing, without=nothing,
       inputs=nothing, block=nothing, output=nothing) = Change(instruction, sections, lm, settings, tools, keep, without, inputs, block, output)
"The fields a change sets, in order."
change_fields(c::Change) = [k for k in CHANGE_FIELDS if getfield(c, k) !== nothing]
function Base.show(io::IO, c::Change)
    print(io, "Change(", join(("$k = $(repr(getfield(c, k)))" for k in change_fields(c)), ", "), ")")
end

"A change as the record keeps it."
function change_record(c::Change)
    out = JObj()
    for k in change_fields(c)
        v = getfield(c, k)
        out[String(k)] = k in (:sections, :tools, :keep) ? Any[String(x) for x in v] :
                         k in (:settings, :inputs) ? JObj(String(a) => logvalue(b) for (a, b) in pairs(v)) : logvalue(v)
    end
    out
end

# ------------------------------------------------------------------ loading

const LOADED_PLUGINS = Dict{Tuple{String,Float64},Plugin}()
const LOADED_LOCK = ReentrantLock()

"""
    load_plugin(path) -> Plugin

A plugin from a Julia file that defines `plugin` (a [`Plugin`](@ref)), read
in a module of its own. Loading runs the file: load only code you trust. A
file is read again when it changed. A `plugins` setting may name the file
instead: `configure!(plugins = ["plugins/modes.jl"])`.
"""
function load_plugin(path::AbstractString)
    p = abspath(expanduser(path))
    isfile(p) || throw(PluginError("plugin-load", "$p: no such file"))
    key = (p, mtime(p))
    hit = lock(() -> get(LOADED_PLUGINS, key, nothing), LOADED_LOCK)
    hit === nothing || return hit
    m = Module(Symbol("FunctAIPlugin_", string(hash(key); base=16)))
    Core.eval(m, :(using FunctAI))
    try
        Base.include(m, p)
    catch err
        throw(PluginError("plugin-load", "$p: $(sprint(showerror, err))"))
    end
    # in the newest world: the file's bindings are newer than this code (Julia 1.12 partitions them by world)
    Base.invokelatest(isdefined, m, :plugin) || throw(PluginError("plugin-load", "$p defines no `plugin = Plugin(...)`"))
    ext = Base.invokelatest(getfield, m, :plugin)
    ext isa Plugin || throw(PluginError("plugin-load", "$p: `plugin` is a $(typeof(ext)), not a Plugin"))
    lock(() -> (LOADED_PLUGINS[key] = ext), LOADED_LOCK)
    ext
end

"A `plugins` setting: a list of Plugins (or files that define one)."
function plugins_setting(v)
    (v isa Union{AbstractVector,Tuple} && !(v isa AbstractString)) ||
        throw(ArgumentError("plugins is a list: plugins = [my_plugin, \"plugins/modes.jl\"]"))
    for e in v
        e isa Union{Plugin,AbstractString} || throw(ArgumentError("a plugin is a Plugin or the path of a file that defines one; not $(repr(e))"))
    end
    Any[v...]
end

resolve_plugin(e::Plugin) = e
resolve_plugin(e::AbstractString) = load_plugin(e)

"""
The plugins around a call, in the order their handlers run (contract/plugins.md,
"Order"): `layers` closest first (`(where, settings)`, `where` `:own`,
`:block` or `:configure`); the program's own first, then each layer from the
innermost out, so the host's handlers see (and have the last word on) what
the program's did. A plugin set in several layers runs once, in its
outermost. A host layer's `program_plugins = false` drops the program's own.
"""
function plugins_in_order(layers)
    vetoed = any(l -> first(l) !== :own && get(last(l), :program_plugins, nothing) === false, layers)
    seen = Set{UInt}()
    kept = Vector{Vector{Plugin}}()
    for (where, s) in Iterators.reverse(layers)            # outermost first: a plugin runs in its outermost layer
        mine = Plugin[]
        if !(vetoed && where === :own)
            for e in something(get(s, :plugins, nothing), Any[])
                p = resolve_plugin(e)
                objectid(p) in seen && continue
                push!(seen, objectid(p))
                push!(mine, p)
            end
        end
        push!(kept, mine)
    end
    Plugin[p for layer in Iterators.reverse(kept) for p in layer]
end

"The setting layers around a call with these own settings, closest first (`extra`: just outside its own: a conversation's)."
function setting_layers(own::AbstractDict{Symbol}, extra=())
    global_now = lock(() -> copy(GLOBAL_SETTINGS), SETTINGS_LOCK)
    Any[(:own, own), extra..., ((:block, b) for b in Iterators.reverse(SCOPED_LAYERS[]))..., (:configure, global_now)]
end

plugins_around(own::AbstractDict{Symbol}, extra=()) = plugins_in_order(setting_layers(own, extra))
has_hook(plugins, hook::Symbol) = any(p -> !isempty(get(p.handlers, hook, ())), plugins)

# ------------------------------------------------------------------ running hooks

"The changes made in one place, for the record: `{plugin, version, hook, change}` each."
struct Applied
    items::Vector{Any}
end
Applied() = Applied(Any[])
add!(a::Applied, p::Plugin, hook::Symbol, change::JObj) =
    push!(a.items, LMCC.jobj("plugin" => p.name, "version" => p.version, "hook" => String(hook), "change" => change))

"Errors a handler throws that are never a plugin's failure: the call stops, or waits, as it would anyway."
passes_through(err) = err isa Union{FunctAIError,Cancelled,TurnWaiting,InterruptException}

"""
Every handler of `hook`, in order: each is given the event as the changes
before it left it, and its change is checked, recorded and applied. A
handler that throws stops the call (`plugin-failed`).
"""
function run_hook(hook::Symbol, plugins, event, applied::Applied, apply)
    allowed = HOOKS[hook][2]
    for p in plugins
        for f in get(p.handlers, hook, ())
            event.plugin = p
            got = try
                Base.invokelatest(f, event)
            catch err
                err = unwrap(err)
                passes_through(err) && rethrow()
                throw(PluginError("plugin-failed", "plugin $(p.name) failed in $hook: $(error_type(err)): $(error_message(err))";
                                  plugin=p.name, hook=String(hook)))
            finally
                event.plugin = nothing
            end
            got === nothing && continue
            got isa Change || throw(PluginError("plugin-change", "plugin $(p.name): $hook returns a Change or nothing, not a $(typeof(got))";
                                                plugin=p.name, hook=String(hook)))
            fields = change_fields(got)
            wrong = [f for f in fields if !(f in allowed)]
            isempty(wrong) || throw(PluginError("plugin-change", "plugin $(p.name): $hook cannot change $(join(wrong, ", ")) " *
                                                "(it may change $(isempty(allowed) ? "nothing" : join(allowed, ", ")))";
                                                plugin=p.name, hook=String(hook)))
            isempty(fields) && continue
            try
                apply(got)
            catch err
                err isa PluginError && rethrow()
                err isa Union{ArgumentError,MethodError,KeyError,InterfaceError} || rethrow()
                throw(PluginError("plugin-change", "plugin $(p.name): $hook: $(error_message(err))"; plugin=p.name, hook=String(hook)))
            end
            add!(applied, p, hook, change_record(got))
        end
    end
    nothing
end

texts_of(sections) = (sections isa AbstractString || !all(s -> s isa AbstractString, sections)) ?
    throw(ArgumentError("sections is a list of texts")) : String[s for s in sections if !isempty(strip(s))]

# ------------------------------------------------------------------ events

"""
The events hooks are given. Each has the fields its hook says
(contract/plugins.md) and, inside a conversation's turn,
`FunctAI.entries(event, kind)` (this plugin's entries on the turn's branch)
and `FunctAI.remember!(event, kind, data)` (keep one at the turn).
"""
abstract type HookEvent end

"A conversation's turn, before it is recorded: `inputs` (by name), `conversation` (its id), `parent`, `program`."
mutable struct TurnStartEvent <: HookEvent
    inputs::OrderedDict{String,Any}
    conversation::String
    parent::Union{Nothing,String}
    program::Any
    plugin::Union{Nothing,Plugin}
    turn_run::Any
end

"An earlier turn a turn would be shown: `id`, `inputs`, `outputs`, the fields already left out (`without`)."
struct ShownTurn
    id::String
    inputs::JObj
    outputs::JObj
    without::Vector{String}
end
ShownTurn(id, inputs, outputs) = ShownTurn(String(id), JObj(inputs), JObj(outputs), String[])

"What a turn is shown: `turns` (the earlier turns the conversation's rule picked), `sections` so far, `conversation`, `parent`, `program`."
mutable struct ContextEvent <: HookEvent
    turns::Vector{ShownTurn}
    sections::Vector{String}
    conversation::Any
    parent::Union{Nothing,String}
    program::Any
    plugin::Union{Nothing,Plugin}
    turn_run::Any
end

"""
An AI function about to be asked: `function` (its name), `program`, `inputs`
(bound), `instruction` (as it stands), `lm` and `settings` (as they stand),
`tools` (the names offered), `all_tools` (its own), `sections` so far,
`path` (its place in the call tree by names: `support/answer`),
`conversation` and `turn` (when it runs in one).
"""
mutable struct BeforeCallEvent <: HookEvent
    instruction::String
    var"function"::String
    program::Any
    inputs::OrderedDict{String,Any}
    lm::Any
    settings::Dict{String,Any}
    tools::Vector{String}
    all_tools::Vector{String}
    sections::Vector{String}
    path::String
    conversation::Union{Nothing,String}
    turn::Union{Nothing,String}
    plugin::Union{Nothing,Plugin}
    turn_run::Any
end

"The provider request about to be sent (`request`, an `LM15.Request`): a handler may return another. `function`, `path`."
mutable struct RequestEvent <: HookEvent
    request::Any
    var"function"::String
    path::String
    plugin::Union{Nothing,Plugin}
    turn_run::Any
end

"""
A tool about to run: `name`, `input` (as it stands), `effects`, `path`
(`support/answer/refund`), `invocation`, `id`, `function` (the AI function
that asked), `approval` (the [`Approval`](@ref) a person would be shown),
`settings` (the asking call's). `ask(event; reason, decide)` asks a person
whether it may run.
"""
mutable struct ToolCallEvent <: HookEvent
    name::String
    input::Any
    effects::Union{Nothing,String}
    path::String
    invocation::Int
    id::String
    var"function"::String
    approval::Approval
    settings::Dict{Symbol,Any}
    plugin::Union{Nothing,Plugin}
    turn_run::Any
    call::Any
    refused::Union{Nothing,String}
end

"A tool ran: `name`, `input`, `output` (text, as it stands), `path`, `invocation`, `function`."
mutable struct ToolResultEvent <: HookEvent
    name::String
    input::Any
    output::String
    path::String
    invocation::Int
    var"function"::String
    plugin::Union{Nothing,Plugin}
    turn_run::Any
end

"""
A conversation's turn ended, before its end is recorded (the next turn sees
what it keeps): `turn`, `state` (`"done"`, `"failed"`, `"stopped"`),
`inputs`, `outputs`, `conversation`, `parent`. `FunctAI.turns(event)` gives the
done turns of its branch, this one last when it is done.
"""
mutable struct TurnEndEvent <: HookEvent
    turn::String
    state::String
    inputs::JObj
    outputs::JObj
    conversation::Any
    parent::Union{Nothing,String}
    plugin::Union{Nothing,Plugin}
    turn_run::Any
end

plugin_name(e::HookEvent) = e.plugin === nothing ? "" : e.plugin.name

"""
    FunctAI.entries(event, kind) -> Vector

This plugin's entries of a kind on the branch of the turn the event is in
(`[]` outside a conversation), oldest first: `(turn, data, at)` each.
"""
function entries(e::HookEvent, kind::AbstractString)
    run = e.turn_run
    run === nothing && return Any[]
    entries(run.conv, plugin_name(e), kind; branch=run.turn)
end
entries(e::ContextEvent, kind::AbstractString) = entries(e.conversation, plugin_name(e), kind; branch=e.parent)
function entries(e::TurnEndEvent, kind::AbstractString)
    found = entries(e.conversation, plugin_name(e), kind; branch=e.parent)
    log = read_log!(e.conversation)
    mine = [(turn=get(r, "turn", nothing), data=LMCC.deepcopy_json(get(r, "data", nothing)), at=get(r, "at", nothing))
            for r in log.entries if get(r, "turn", nothing) == e.turn && get(r, "plugin", nothing) == plugin_name(e) && get(r, "entry", nothing) == kind]
    vcat(found, mine)
end

"""
    FunctAI.remember!(event, kind, data)

Keep an entry of this plugin at the turn the event is in (it then belongs to
the branches through that turn). Outside a conversation it is kept nowhere:
an `ArgumentError`.
"""
function remember!(e::HookEvent, kind::AbstractString, data)
    run = e.turn_run
    run === nothing && throw(ArgumentError("remember! keeps an entry in a conversation; this call runs in none"))
    remember!(run.conv, plugin_name(e), kind, data; turn=run.turn)
end
remember!(e::TurnEndEvent, kind::AbstractString, data) = remember!(e.conversation, plugin_name(e), kind, data; turn=e.turn)

"The done turns of the ended turn's branch, this one last when it is done."
function turns(e::TurnEndEvent)
    log = read_log!(e.conversation)
    out = ShownTurn[ShownTurn(turn_id(st), something(get(st.record, "inputs", nothing), JObj()), ended_outputs(st))
                    for st in branch(log, e.parent) if turn_state(st) == "done"]
    e.state == "done" && push!(out, ShownTurn(e.turn, e.inputs, e.outputs))
    out
end

"""
    ask(event::ToolCallEvent; reason = nothing, decide = nothing) -> Bool

Ask a person whether the tool may run: in a conversation the turn waits,
saved, and goes on when someone answers; on a stream the call waits for
[`approve!`](@ref); a plain call refuses (`approval-required`). `decide` (a
function of the [`Approval`](@ref) returning `true`, `false` or a reason)
answers in place of a person. Returns `true`, or `false` (the model is then
shown the refusal and the person's reason).
"""
function ask(e::ToolCallEvent; reason=nothing, decide=nothing)
    a = asked_by(with_input(e.approval, e.input), e.plugin === nothing ? "approval" : e.plugin.name, reason)
    allowed, why, _by = ask_person(e.call, a; decide)
    allowed || (e.refused = denial(why))
    allowed
end

# ------------------------------------------------------------------ what the engine asks

"A call as its `before_call` hooks left it: settings, the instruction (`nothing`: the program's), sections, the tools offered, the changes."
struct Shaped
    settings::Dict{Symbol,Any}
    instruction::Union{Nothing,String}
    sections::Vector{String}
    context_sections::Vector{String}
    tools::Union{Nothing,Vector{String}}
    applied::Applied
end

const LM15_SETTINGS = Set(String.(fieldnames(LM15.Config)))

"Apply lm15 settings to FunctAI settings: the five FunctAI names its own, the rest inside `config`."
function with_lm15_settings(s::Dict{Symbol,Any}, given)
    out = copy(s)
    extra = Dict{Symbol,Any}()
    old = get(out, :config, nothing)
    if old isa LM15.Config
        for n in fieldnames(LM15.Config)
            v = getfield(old, n)
            (v === nothing || (v isa Tuple && isempty(v))) || (extra[n] = v)
        end
    elseif old !== nothing
        for (k, v) in pairs(old)
            extra[Symbol(k)] = v
        end
    end
    for (k, v) in pairs(given)
        key = Symbol(k)
        key in CONFIG_KEYS ? (out[key] = v) : (extra[key] = v)
    end
    isempty(extra) || (out[:config] = (; extra...))
    out
end

"""
`before_call` for an AI function's call: the sections its context gives
first (its conversation's turn, a row asked again), then each handler's.
"""
function shape_call(f, inputs, s::Dict{Symbol,Any}, call, base_instruction::AbstractString)
    applied = Applied()
    given = context_sections(call)
    sections = copy(given)
    all_tools = String[t.name for t in f.tools]
    plugins = plugins_around(f.own)
    has_hook(plugins, :before_call) || return Shaped(s, nothing, sections, given, nothing, applied)
    conv = call === nothing ? nothing : call.conversation
    run = call === nothing ? nothing : call.turn_run
    shown = Dict{String,Any}(String(k) => s[k] for k in CONFIG_KEYS if get(s, k, nothing) !== nothing)
    event = BeforeCallEvent(String(base_instruction), f.definition.name, f, OrderedDict{String,Any}(String(k) => v for (k, v) in inputs),
                            get(s, :lm, nothing), shown, copy(all_tools), copy(all_tools), copy(sections), names_path(call),
                            conv !== nothing ? conv["id"] : run !== nothing ? run.conv.id : nothing,
                            conv !== nothing ? conv["turn"] : run !== nothing ? run.turn : nothing, nothing, run)
    out = copy(s)
    offered = Ref{Any}(nothing)
    instruction = Ref{Any}(nothing)
    run_hook(:before_call, plugins, event, applied, function (c)
        if c.instruction !== nothing
            (c.instruction isa AbstractString && !isempty(strip(c.instruction))) || throw(ArgumentError("instruction is the text of an instruction"))
            instruction[] = event.instruction = String(c.instruction)
        end
        c.sections === nothing || append!(event.sections, texts_of(c.sections))
        if c.lm !== nothing
            (c.lm isa AbstractString && !isempty(strip(c.lm))) || throw(ArgumentError("lm is a model's name"))
            out[:lm] = event.lm = String(c.lm)
        end
        if c.settings !== nothing
            bad = sort!([String(k) for k in keys(c.settings) if !(String(k) in LM15_SETTINGS)])
            isempty(bad) || throw(ArgumentError("settings are lm15 settings ($(join(sort!(collect(LM15_SETTINGS)), ", "))), not $(join(bad, ", "))"))
            out = with_lm15_settings(out, c.settings)
            for (k, v) in pairs(c.settings)
                event.settings[String(k)] = v
            end
        end
        if c.tools !== nothing
            names = String[String(x) for x in c.tools]
            unknown = sort!(setdiff(names, all_tools))
            isempty(unknown) || throw(ArgumentError("$(f.definition.name) has no tool $(repr(first(unknown))) (its tools: " *
                                                    "$(isempty(all_tools) ? "none" : join(all_tools, ", ")))"))
            offered[] = event.tools = [n for n in all_tools if n in names]
        end
    end)
    Shaped(out, instruction[], event.sections, given, offered[], applied)
end

"A call's place in its tree by names (`support/answer`): what a host's rules read."
names_path(call) = call === nothing ? "" : join([String(first(split(p, '#'))) for p in split(call.site, '/') if !isempty(p)], "/")

"""
The `request` hook (the escape hatch): the request to send, and whether a
handler replaced it (the call is then not replayable: no one can rebuild it).
"""
function plugin_request(request, call, function_name)
    call === nothing && return (request, false)
    plugins = plugins_around(call.fn isa AIFunction ? call.fn.own : Dict{Symbol,Any}())
    has_hook(plugins, :request) || return (request, false)
    event = RequestEvent(request, String(function_name), names_path(call), nothing, call.turn_run)
    changed = false
    for p in plugins, f in get(p.handlers, :request, ())
        event.plugin = p
        got = try
            Base.invokelatest(f, event)
        catch err
            err = unwrap(err)
            passes_through(err) && rethrow()
            throw(PluginError("plugin-failed", "plugin $(p.name) failed in request: $(error_type(err)): $(error_message(err))";
                              plugin=p.name, hook="request"))
        finally
            event.plugin = nothing
        end
        (got === nothing || got === event.request) && continue
        got isa LM15.Request || throw(PluginError("plugin-change", "plugin $(p.name): request returns an LM15.Request or nothing";
                                                  plugin=p.name, hook="request"))
        event.request = got
        changed = true
        push!(call.changes, LMCC.jobj("plugin" => p.name, "version" => p.version, "hook" => "request",
                                      "change" => LMCC.jobj("request" => "replaced")))
    end
    changed && (call.replayable = false)
    (event.request, changed)
end

"""
`tool_call` for one tool call: `(input it runs with, nothing)`, or `(nothing,
what the model is shown instead)`. Handlers run in order, then the
`approval` plugin (`approve =`) last, on the input they left: a host's rule
judges what will run. A handler that throws blocks the tool.
"""
function plugin_tool_call(call, a::Approval, s::Dict{Symbol,Any})
    plugins = Plugin[plugins_around(call.fn isa AIFunction ? call.fn.own : Dict{Symbol,Any}())..., APPROVAL_PLUGIN]
    event = ToolCallEvent(a.name, LMCC.deepcopy_json(a.input), a.effects, a.path, a.invocation, a.id, call.name, a, s, nothing,
                          call.turn_run, call, nothing)
    applied = Applied()
    blocked = Ref{Any}(nothing)
    apply = function (c)
        if c.inputs !== nothing
            c.inputs isa Union{AbstractDict,NamedTuple} || throw(ArgumentError("inputs is a Dict of the tool's inputs"))
            base = event.input isa AbstractDict ? JObj(event.input) : JObj()
            for (k, v) in pairs(c.inputs)
                base[String(k)] = logvalue(v)
            end
            event.input = base
        end
        if c.block !== nothing
            c.block isa AbstractString || throw(ArgumentError("block is the reason, a text"))
            blocked[] = String(c.block)
        end
    end
    try
        for p in plugins
            (blocked[] !== nothing || event.refused !== nothing) && break
            run_hook(:tool_call, [p], event, applied, apply)
        end
    catch err
        (err isa PluginError && err.code == "plugin-failed") || rethrow()
        warn_once("tool-call:$(err.plugin)", "$(sprint(showerror, err)): the tool does not run")
        blocked[] = "a check on this tool call failed ($(err.plugin))"
        push!(applied.items, LMCC.jobj("plugin" => err.plugin, "version" => "", "hook" => "tool_call",
                                       "change" => LMCC.jobj("block" => blocked[])))
    end
    append!(call.changes, applied.items)
    if blocked[] !== nothing
        who = isempty(applied.items) ? "?" : last(applied.items)["plugin"]
        return (nothing, "This call was blocked ($who): $(blocked[])")
    end
    event.refused === nothing || return (nothing, event.refused)
    (event.input, nothing)
end

"`tool_result`: what the model is shown of a tool's result."
function plugin_tool_result(call, a::Approval, input, output::AbstractString)
    plugins = plugins_around(call.fn isa AIFunction ? call.fn.own : Dict{Symbol,Any}())
    has_hook(plugins, :tool_result) || return String(output)
    event = ToolResultEvent(a.name, input, String(output), a.path, a.invocation, call.name, nothing, call.turn_run)
    applied = Applied()
    run_hook(:tool_result, plugins, event, applied, function (c)
        c.output isa AbstractString || throw(ArgumentError("output is the text the model is shown"))
        event.output = String(c.output)
    end)
    append!(call.changes, applied.items)
    event.output
end

# ------------------------------------------------------------------ asking a person (contract/tools.md, "Who answers a rule")

"""
Ask whether a tool call may run: `(allowed, reason refused, by whom)`.
`decide` answers in place of a person. A call being resumed takes the
answer recorded for it. Emits `approval` then `approved`. Otherwise: in a
conversation the turn waits (`TurnWaiting`: the turn is saved as waiting);
on a stream the call waits for an answer in this process; a plain call
refuses `approval-required`.
"""
function ask_person(call, a::Approval; decide=nothing)
    run = call.turn_run
    if run !== nothing
        known = recorded_approval(run, call, a)
        if known !== nothing
            allowed, reason, by, fresh = known
            if fresh                                     # answered while the turn waited: the log says so now
                frontier!(call.tree)
                emit_approved(call, a, allowed, by, reason)
            end
            return (allowed, reason, by)
        end
    end
    frontier!(call.tree)
    data = LMCC.jobj("id" => a.id, "invocation" => a.invocation, "name" => a.name, "input" => logvalue(a.input),
                     "effects" => a.effects, "path" => a.path, "to" => APPROVALS_TO[], "plugin" => a.plugin)
    a.question === nothing || (data["question"] = a.question)
    emit!(call, :approval, data)
    allowed, reason, by = if decide !== nothing
        (verdict_of(Base.invokelatest(decide, a))..., nothing)
    elseif run !== nothing
        throw(TurnWaiting(Approval(a.call, a.invocation, a.id, a.name, a.input, a.effects, a.path, call.site, a.plugin, a.question)))
    elseif !isempty(call.watchers)
        wait_here(call, a)
    else
        throw(ApprovalError("$(a.path): this tool call needs a person's answer ($(a.plugin)), and a plain call has nobody to ask. " *
                            "Give approve = a function, stream the call and answer with approve!(stream), or use a conversation, " *
                            "where the turn waits."; approval=a))
    end
    emit_approved(call, a, allowed, by, reason)
    run === nothing || note_approval!(run, call, a, allowed, reason, by)
    (allowed, reason, by)
end

function emit_approved(call, a::Approval, allowed::Bool, by, reason)
    data = LMCC.jobj("id" => a.id, "invocation" => a.invocation, "verdict" => allowed ? "yes" : "no", "by" => by, "plugin" => a.plugin)
    reason === nothing || (data["reason"] = reason)
    emit!(call, :approved, data)
end

"Wait in this process for a stream's answer (`approve!(s)`, `deny!(s)`); a closed stream stops the wait."
function wait_here(call, a::Approval)
    key = (a.call, a.invocation, a.plugin)
    lock(PENDING_LOCK) do
        PENDING_ASKED[key] = a
    end
    try
        while true
            got = lock(PENDING_LOCK) do
                haskey(PENDING_ANSWERS, key) ? pop!(PENDING_ANSWERS, key) : nothing
            end
            got === nothing || return got
            check_cancelled(call)
            sleep(0.05)
        end
    finally
        lock(() -> delete!(PENDING_ASKED, key), PENDING_LOCK)
    end
end

"Whom an approval is addressed to: `\"owner\"`, or `\"caller\"` when a served program lets its caller answer (contract/serving.md)."
const APPROVALS_TO = ScopedValue("owner")

# ------------------------------------------------------------------ the approval plugin (approve =)

"""
`approve =` as a plugin (contract/tools.md): a rule names which tool calls a
person is asked about; a function answers in place of a person.
"""
const APPROVAL_PLUGIN = Plugin("approval"; version="1.0.0", description="approve =: ask before tools run, as a rule says",
    tool_call=function (t)
        rule = get(t.settings, :approve, nothing)
        asks(rule, t.approval) || return nothing
        ask(t; decide=rule isa Function ? rule : nothing)
        nothing
    end)
