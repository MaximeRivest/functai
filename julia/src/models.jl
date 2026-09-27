# Which model, how to reach it, and what it can do (contract/functions.md,
# "Capabilities"). lm15 routes a model name to a provider; the facts about
# the model come from the contract's table, never from guessing.

const PROBE = Dict{String,Bool}(k => v for (k, v) in MODELS["probe"] if k != "about")
const NATIVE = Set{String}(MODELS["native"]["providers"])
const NO_STOP = Set{String}(MODELS["native"]["no_stop_sequences"])
const PREFILL = Set{String}(MODELS["native"]["assistant_prefill"])
const REASONING_PREFIXES = Dict{String,Vector{String}}(k => String.(v) for (k, v) in MODELS["native"]["reasoning_prefixes"])
const CHAT_COMPLETIONS = Set{String}(MODELS["chat_completions"]["providers"])
const TOOL_HOSTS = Set{String}(MODELS["native_tool_hosts"]["providers"])
const JUDGMENT_ONLY = Set{String}(MODELS["judgment_only"]["providers"])
const SPEAKS_AS = Dict{String,String}(k => v for (k, v) in MODELS["speaks_as"] if k != "about")
const FIXED_SAMPLING = Dict{String,Vector{String}}(k => String.(v) for (k, v) in MODELS["fixed_sampling"] if k != "about")

"""
    model_capabilities(provider, model) -> NamedTuple

What FunctAI declares `model` served by `provider` can do, from the
contract's table (never guessed): native tool calls, reasoning, stop
sequences, enforced JSON, a prefilled reply. These facts decide how a
function's layout is written for the model; a function's `capabilities`
setting replaces any of them.

# Examples
```jldoctest
julia> model_capabilities("anthropic", "claude-haiku-4-5")
(assistant_prefill = false, instruct = true, native_function_calling = true, native_reasoning = true, native_structured_output = true, stop_sequences = true)
```
"""
model_capabilities(provider::AbstractString, model::AbstractString) =
    (; (Symbol(k) => v for (k, v) in sort!(collect(capabilities_of(provider, model)); by=first))...)

"The facts as a `Dict` (what binding a layout reads)."
function capabilities_of(provider::AbstractString, model::AbstractString)
    provider in JUDGMENT_ONLY && return Dict{String,Bool}("native_structured_output" => true)
    haskey(SPEAKS_AS, provider) && return capabilities_of(SPEAKS_AS[provider], model)
    caps = Dict{String,Bool}("instruct" => true)
    if provider in NATIVE
        caps["native_function_calling"] = true
        caps["native_structured_output"] = true
        caps["stop_sequences"] = !(provider in NO_STOP)
        caps["native_reasoning"] = any(p -> startswith(model, p), get(REASONING_PREFIXES, provider, String[]))
        caps["assistant_prefill"] = provider in PREFILL && !caps["native_reasoning"]
    elseif provider in CHAT_COMPLETIONS
        merge!(caps, Dict("native_function_calling" => true, "native_structured_output" => true,
                          "stop_sequences" => false, "native_reasoning" => false))
    else
        caps["native_function_calling"] = provider in TOOL_HOSTS
        caps["stop_sequences"] = false
    end
    caps
end

"The facts for a call: the table, Anthropic's temperature rule, then the function's own `capabilities`."
function call_capabilities(provider, model, s::AbstractDict{Symbol})
    caps = capabilities_of(provider, model)
    anthropic = provider == "anthropic" || get(SPEAKS_AS, provider, "") == "anthropic"
    t = setting(s, :temperature)
    if anthropic && get(caps, "native_reasoning", false) && t !== nothing && t != 1
        caps["native_reasoning"] = false
        caps["assistant_prefill"] = true
    end
    own = setting(s, :capabilities)
    own === nothing || for (k, v) in pairs(own)
        caps[String(k)] = v
    end
    caps
end

"Settings a model refuses: sampling for fixed-sampling models; every knob for the ChatGPT subscription."
function refused_settings(provider, model)
    provider == "openai-codex" && return [:temperature, :top_p, :max_tokens]    # no knobs, no output cap
    prefixes = get(FIXED_SAMPLING, get(SPEAKS_AS, provider, provider), String[])
    any(p -> startswith(model, p), prefixes) ? [:temperature, :top_p] : Symbol[]
end

const REFUSED_WARNED = Set{String}()
const WARN_LOCK = ReentrantLock()

"Settings a model does not take are left out of its requests, with one warning per provider."
function adjust_settings(s::AbstractDict{Symbol}, provider, model)
    drop = [k for k in refused_settings(provider, model)
            if setting(s, k) !== nothing && (k === :max_tokens || provider == "openai-codex" || setting(s, k) != 1)]
    isempty(drop) && return s
    key = "$provider:$(join(drop, ","))"
    first_time = lock(WARN_LOCK) do
        key in REFUSED_WARNED ? false : (push!(REFUSED_WARNED, key); true)
    end
    first_time && @warn "$provider:$model does not take $(join(drop, ", ")); left out of its requests"
    out = copy(s)
    for k in drop
        out[k] = nothing
    end
    out
end

"Friendly account prefixes, as in Python: `claude:` is the Claude subscription (`claude-code:`), …"
const PREFIX_ALIASES = Dict("claude" => "claude-code", "chatgpt" => "openai-codex", "copilot" => "github-copilot", "kimi" => "kimi-code")

function model_string(lm::AbstractString)
    i = findfirst(':', lm)
    i === nothing && return String(lm)
    head = lowercase(lm[1:prevind(lm, i)])
    haskey(PREFIX_ALIASES, head) ? PREFIX_ALIASES[head] * lm[i:end] : String(lm)
end

const DEFAULT_PICKS = [
    "OPENAI_API_KEY" => "gpt-4.1-mini", "ANTHROPIC_API_KEY" => "claude-haiku-4-5",
    "GEMINI_API_KEY" => "gemini:gemini-2.5-flash", "GOOGLE_API_KEY" => "gemini:gemini-2.5-flash",
    "GROQ_API_KEY" => "groq:openai/gpt-oss-120b", "OPENROUTER_API_KEY" => "openrouter:openai/gpt-4.1-mini",
]

"A model this machine can use when none is configured: the first provider with a key in the environment."
function default_model()
    for (key, model) in DEFAULT_PICKS
        isempty(get(ENV, key, "")) || return model
    end
    nothing
end

# ------------------------------------------------------------------ routers

"""
    route(router, model) -> (provider, model)

Which provider `router` sends `model` to, and the model's name there. A
router of your own (a fake in tests, a gateway) adds a method of this and
of `LM15.complete` (and optionally `LM15.stream`) for its type.
"""
function route(router, model::AbstractString)
    r = LM15.resolve(router, model)
    (String(r.provider), String(r.model))
end

const ROUTERS = Dict{Any,Any}()
const ROUTERS_LOCK = ReentrantLock()
shared_router(key, make) = lock(() -> get!(make, ROUTERS, key), ROUTERS_LOCK)
"Forget which router each model goes through (after signing in or out)."
forget_routes!() = lock(() -> filter!(p -> !(p.first isa Tuple && p.first[1] === :for), ROUTERS), ROUTERS_LOCK)

"The providers with a saved sign-in (lm15's credentials file, shared by every language)."
function signed_in_providers()
    try
        Set{String}(c.provider for c in LM15.connections(LM15.local_auth()))
    catch
        Set{String}()
    end
end

"""
The router a call goes through when the settings name none: the one that
holds this machine's saved sign-in for the provider, else one that reads
API keys from the environment (as Python chooses).
"""
default_router(model::AbstractString) = shared_router((:for, String(model)), () -> pick_router(model))

function pick_router(model)
    env = shared_router(:env, LM15.LMRouter)
    provider = try
        route(env, model)[1]
    catch err
        err isa LM15.LM15Error || rethrow()
        nothing
    end
    if provider === nothing || provider in signed_in_providers()
        return shared_router(:auth, () -> LM15.LMRouter(LM15.RouterConfig(; auth=LM15.local_auth())))
    end
    env
end
