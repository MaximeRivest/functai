# Settings: where a call goes and how it behaves. A function's own settings
# beat `with_settings` blocks, which beat `configure!`, which beats the
# defaults: the order every FunctAI language uses.

const SETTING_DOCS = (
    lm = "the model: \"gpt-4.1-mini\", \"claude-haiku-4-5\", \"groq:openai/gpt-oss-120b\", …",
    router = "the lm15 router calls go through (keys, base URLs; a fake in tests)",
    temperature = "sampling temperature", max_tokens = "the token budget of a reply",
    top_p = "nucleus sampling", stop = "stop sequences", seed = "the provider's seed",
    config = "other lm15 Config fields, as keywords: (reasoning = LM15.Reasoning(effort=\"low\"),)",
    adapter = "the layout: :xml (default), :chat, :json, or an lmcc adapter or artifact",
    template = "a chat template: [:system => \"…\", :turns, :user => \"{review}\"]",
    reasoning = "true: the model writes its reasoning before the answer (chain of thought)",
    include_name = "false: leave the function's name out of the instruction",
    capabilities = "facts about the model that replace the table's: Dict(\"native_reasoning\" => false)",
    retries = "re-asks after an unreadable reply (default 1)",
    api_retries = "re-sends after a transient provider error (default 3)",
    max_steps = "model requests per tool loop (default 8)",
    tool_errors = ":report (the model sees a tool's error, default) or :raise",
    log_calls = "the call log: a folder, true (the default folder) or false; unset: the environment variable `FUNCTAI_LOG_CALLS` decides",
    log_content = "what the call log keeps: false (sizes, times and tokens, never values or messages), or a map of fields: (transcript = false,), Dict(\"*\" => false, \"question\" => true); layers only remove",
    observers = "functions (or Channels) given the kept form of every event of the calls in scope: [e -> println(e)]; layers add up",
    journal = "where each call tree's kept log is kept while it is written: a store (best effort), Journal(store; required = true), or false (none)",
    caller = "who is calling, added to the environment variable `FUNCTAI_CALLER`: Dict(\"kind\" => \"notebook\")",
    concurrency = "calls in flight at once over a column (broadcasting, map, evaluate; default 8)",
)
const SETTING_NAMES = keys(SETTING_DOCS)

const DEFAULTS = Dict{Symbol,Any}(:retries => 1, :api_retries => 3, :max_steps => 8, :tool_errors => :report,
                                  :include_name => true, :reasoning => false, :concurrency => 8)

const GLOBAL_SETTINGS = Dict{Symbol,Any}()
const SETTINGS_LOCK = ReentrantLock()
const SCOPED_SETTINGS = ScopedValue(Dict{Symbol,Any}())
# each enclosing with_settings block's own settings, outermost first: the
# settings that are layered rather than overridden (log_content, observers,
# journal) read every layer
const SCOPED_LAYERS = ScopedValue(Dict{Symbol,Any}[])

"The Levenshtein distance between two short words (for suggesting a setting's name)."
function edit_distance(a::AbstractString, b::AbstractString)
    a, b = collect(a), collect(b)
    d = collect(0:length(b))
    for i in 1:length(a)
        prev, d[1] = d[1], i
        for j in 1:length(b)
            prev, d[j+1] = d[j+1], min(d[j+1] + 1, d[j] + 1, prev + (a[i] != b[j]))
        end
    end
    d[end]
end

function check_setting(name::Symbol, value)
    if !(name in SETTING_NAMES)
        given = lowercase(String(name))
        near = [String(n) for n in SETTING_NAMES if edit_distance(given, String(n)) <= 2 ||
                (length(given) >= 3 && (occursin(given, String(n)) || occursin(String(n), given)))]
        hint = name === :module ? "; for reasoning before the answer, use reasoning = true" :
               name === :include_fn_name_in_instructions ? "; use include_name" :
               isempty(near) ? "" : "; did you mean $(join(near, " or "))?"
        throw(ArgumentError("unknown setting $(name)$hint. Settings: $(join(SETTING_NAMES, ", "))"))
    end
    value === nothing && return nothing
    name === :tool_errors && !(value in (:report, :raise, "report", "raise")) &&
        throw(ArgumentError("tool_errors is :report or :raise, not $(repr(value))"))
    name in (:retries, :api_retries, :max_steps, :concurrency) && !(value isa Integer && value >= (name === :concurrency || name === :max_steps ? 1 : 0)) &&
        throw(ArgumentError("$name is a whole number$(name in (:concurrency, :max_steps) ? " of at least 1" : " of at least 0"), not $(repr(value))"))
    name in (:reasoning, :include_name) && !(value isa Bool) && throw(ArgumentError("$name is true or false, not $(repr(value))"))
    name === :log_calls && !(value isa Union{Bool,AbstractString}) && throw(ArgumentError("log_calls is a folder, true or false, not $(repr(value))"))
    name === :lm && !(value isa AbstractString) && throw(ArgumentError("lm is a model name like \"gpt-4.1-mini\", not $(repr(value))"))
    name === :caller && !(value isa AbstractDict || value isa NamedTuple) && throw(ArgumentError("caller is a Dict, not $(repr(value))"))
    name === :log_content && content_setting(value)
    name === :observers && !(value isa Union{AbstractVector,Tuple}) && throw(ArgumentError("observers is a list of functions (or Channels), not $(repr(value))"))
    name === :journal && journal_setting(value)
    nothing
end

function settings_dict(kw)
    out = Dict{Symbol,Any}()
    for (k, v) in pairs(kw)
        name = Symbol(k)
        check_setting(name, v)
        out[name] = v === nothing ? nothing :
                    name === :caller ? Dict{String,Any}(String(a) => b for (a, b) in pairs(v)) :
                    name === :tool_errors ? Symbol(v) :
                    name === :log_content ? content_setting(v) :
                    name === :observers ? Any[v...] :
                    name === :journal ? journal_setting(v) : v
    end
    out
end

"""
    configure!(; settings...)

Settings for every AI function in this process (their own settings still
win). A setting given as `nothing` goes back to its default. Returns the
settings now in force.

```julia
FunctAI.configure!(lm = "gpt-4.1-mini", temperature = 0)
```

Settings: $(join(("`$k`: $v" for (k, v) in pairs(SETTING_DOCS)), "; ")).
"""
function configure!(; kw...)
    given = settings_dict(kw)
    lock(SETTINGS_LOCK) do
        for (k, v) in given
            v === nothing ? delete!(GLOBAL_SETTINGS, k) : (GLOBAL_SETTINGS[k] = v)
        end
        (; (k => v for (k, v) in sort!(collect(GLOBAL_SETTINGS); by=first))...)
    end
end

"""
    with_settings(f; settings...)
    with_settings(settings...) do ... end

Run `f()` with these settings over `configure!`'s, in this task and every
task it starts (a function's own settings still win).

```julia
with_settings(lm = "claude-haiku-4-5", log_calls = true) do
    mood.(reviews)
end
```
"""
function with_settings(f; kw...)
    given = settings_dict(kw)
    merged = merge(SCOPED_SETTINGS[], given)
    if haskey(given, :caller) && haskey(SCOPED_SETTINGS[], :caller) && given[:caller] !== nothing
        merged[:caller] = merge(SCOPED_SETTINGS[][:caller], given[:caller])
    end
    with(f, SCOPED_SETTINGS => merged, SCOPED_LAYERS => push!(copy(SCOPED_LAYERS[]), given))
end

"The settings a function with `own` settings runs with."
function effective(own::AbstractDict{Symbol})
    out = copy(DEFAULTS)
    global_now = lock(() -> copy(GLOBAL_SETTINGS), SETTINGS_LOCK)
    scoped = SCOPED_SETTINGS[]
    caller = Dict{String,Any}()
    for layer in (global_now, scoped, own)
        for (k, v) in layer
            v === nothing && continue
            out[k] = v
            k === :caller && merge!(caller, v)
        end
    end
    out[:caller] = caller
    out
end

setting(s::AbstractDict{Symbol}, k::Symbol) = get(s, k, nothing)

const CONFIG_KEYS = (:temperature, :max_tokens, :top_p, :stop, :seed)

"The lm15 `Config` of the settings (and `overrides`), or `nothing` when they set none."
function config_of(s::AbstractDict{Symbol}; overrides...)
    c = Dict{Symbol,Any}()
    extra = setting(s, :config)
    if extra isa LM15.Config
        for n in fieldnames(LM15.Config)
            v = getfield(extra, n)
            (v === nothing || (v isa Tuple && isempty(v))) || (c[n] = v)
        end
    elseif extra !== nothing
        for (k, v) in pairs(extra)
            c[Symbol(k)] = v
        end
    end
    for k in CONFIG_KEYS
        v = setting(s, k)
        v === nothing && continue
        k === :stop && isempty(v) && continue
        c[k] = k === :stop ? Tuple(String.(v isa AbstractString ? (v,) : v)) : v
    end
    for (k, v) in overrides
        c[k] = v
    end
    isempty(c) ? nothing : LM15.Config(; c...)
end
