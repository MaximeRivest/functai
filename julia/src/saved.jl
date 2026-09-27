# Saved programs (contract/saved.md). `FunctAI.load` runs an AI function
# saved in any language, from its folder's `functai.json`, after checking it
# sends exactly what was saved; it refuses, with a reason, what it cannot run
# (code of its own, tools, a baked model). `FunctAI.save` writes an AI
# function defined here as a folder every language's loader reads.

const SAVED_FORMAT = 1

"""
    LoadRefused

A saved program this loader will not run; `code` says why
(`saved-malformed`, `saved-format`, `saved-not-ai`, `saved-code`,
`saved-tools`, `saved-model`, `saved-differs`: contract/saved.md).
"""
struct LoadRefused <: Exception
    code::String
    msg::String
end
Base.showerror(io::IO, e::LoadRefused) = print(io, "LoadRefused [", e.code, "]: ", e.msg)
refuse_load(code, msg) = throw(LoadRefused(code, msg))

const SNAKE_SETTINGS = (:retries, :api_retries, :max_steps, :tool_errors, :capabilities, :log_content)

function check_form(m)
    m isa AbstractDict || refuse_load("saved-malformed", "functai.json is not a JSON object")
    get(m, "functai_saved", nothing) == SAVED_FORMAT ||
        refuse_load("saved-format", "functai.json is format $(repr(get(m, "functai_saved", nothing))); this loader reads format $SAVED_FORMAT")
    (get(m, "entry", nothing) isa AbstractString && get(m, "nodes", nothing) isa AbstractDict) ||
        refuse_load("saved-malformed", "functai.json needs entry and nodes")
    m
end

"""
    from_manifest(manifest; node, types, saved_id) -> AIFunction

An AI function from a saved manifest (the parsed `functai.json`); `node`
names one by key (`module:name`), the entry by default. See [`load`](@ref).
"""
function from_manifest(manifest; node=nothing, types=(;), saved_id=nothing)
    m = check_form(manifest)
    language = something(get(m, "language", nothing), "python")
    key = something(node, m["entry"])
    n = get(m["nodes"], key, nothing)
    n isa AbstractDict || refuse_load("saved-malformed", "functai.json has no node $(repr(key))")
    kind = get(n, "kind", nothing)
    kind == "ai" || refuse_load("saved-not-ai", "$key is $(kind == "module" ? "a module" : "a $kind"): code in $language, " *
                                               "which this loader cannot run. Its AI functions load by key.")
    data = get(n, "ai", nothing)
    data isa AbstractDict || refuse_load("saved-malformed", "$key has no \"ai\" entry")
    (haskey(data, "body") && data["body"] === nothing) ||
        refuse_load("saved-code", "$key runs code of its own beside the model (written in $language); only $language can run it")
    tools = get(data, "tools", nothing)
    tools isa AbstractVector && !isempty(tools) &&
        refuse_load("saved-tools", "$key has tools ($(join(LMCC.json_text.(tools), ", "))): a tool is code")
    settings_in = something(get(data, "settings", nothing), JObj())
    for (k, v) in settings_in
        v isa AbstractDict && (haskey(v, "baked") || haskey(v, "node")) &&
            refuse_load("saved-model", "$key: setting $k is $(LMCC.json_text(v)), not something this loader can reach")
    end
    sig = try
        LMCC.signature_from_dict(data["signature"])
    catch err
        err isa LMCC.Refusal ? refuse_load("saved-malformed", "$key: $(err.hint)") : rethrow()
    end
    given = Dict{String,Any}(String(k) => v for (k, v) in pairs(types))
    field(f) = begin
        spec = if haskey(given, f.name)
            T = pop!(given, f.name)
            LMCC.json_equal(shape_of(T), f.shape) ||
                throw(ArgumentError("types: $(f.name) was saved as $(LMCC.json_text(f.shape)), but $(spec_text(T)) is $(LMCC.json_text(shape_of(T)))"))
            T
        else
            JObj(f.shape)
        end
        FieldDef(f.name, spec, JObj(f.shape), f.direction == "input" ? f.desc : nothing)
    end
    inputs, outputs = FieldDef[], FieldDef[]
    reasoning = false
    for f in sig.fields
        if f.direction == "input" && f.purpose == "plain"
            push!(inputs, field(f))
        elseif f.direction == "output" && f.purpose == "plain"
            push!(outputs, field(f))
        elseif f.purpose == "reasoning"
            reasoning = true
        else
            refuse_load("saved-tools", "$key: field $(f.name) ($(f.purpose)) needs tools")
        end
    end
    isempty(given) || throw(ArgumentError("types: $(join(keys(given), ", ")) is not a field of $key"))
    own = Dict{Symbol,Any}()
    lm = get(settings_in, "lm", nothing)
    lm isa AbstractString && (own[:lm] = lm)
    (get(settings_in, "module", nothing) == "cot" || reasoning) && (own[:reasoning] = true)
    get(settings_in, "include_fn_name_in_instructions", true) === false && (own[:include_name] = false)
    adapter = get(settings_in, "adapter", nothing)
    adapter === nothing || (own[:adapter] = adapter)
    for k in SNAKE_SETTINGS
        v = get(settings_in, String(k), nothing)
        v === nothing || (own[k] = k === :tool_errors ? Symbol(v) : v)
    end
    template = get(data, "template", nothing)
    template isa AbstractVector && (own[:template] = Any[template...])
    config = something(get(data, "config", nothing), JObj())
    if !isempty(config)
        rest = JObj()
        for (k, v) in config
            k in ("temperature", "max_tokens", "top_p", "seed") ? (own[Symbol(k)] = v) :
            k == "stop" ? (own[:stop] = String.(v)) : (rest[k] = v)
        end
        isempty(rest) || (own[:config] = LM15.from_dict(LM15.Config, rest))
    end
    state = something(get(data, "state", nothing), LMCC.jobj("instructions" => nothing, "demos" => Any[]))
    definition = Definition(String(n["name"]), "", inputs, outputs, sig.instructions)
    f = AIFunction(definition, own, AITool[], String(n["module"]), nothing, nothing, saved_id,
                   get(state, "instructions", nothing), Any[], nothing, nothing, nothing, nothing, nothing, nothing,
                   Dict{Any,Any}(), ReentrantLock())
    f = with_demos(f, something(get(state, "demos", nothing), Any[]))
    # it must send what was saved (contract/saved.md, "Loading", step 6)
    want = something(get(something(get(data, "fingerprints", nothing), JObj()), "requests", nothing), Any[])
    for (i, probe) in enumerate(something(get(data, "probes", nothing), Any[]))
        i > length(want) && break
        got = request_hash(f, probe)
        got == want[i] || refuse_load("saved-differs", "$key: for probe $(i - 1) ($(first(LMCC.json_text(probe), 200))) " *
                                                       "it would send $got, but $(want[i]) was saved")
    end
    v = get(data, "version", nothing)
    v isa AbstractString && v != version(f) &&
        refuse_load("saved-differs", "$key: its version here is $(version(f)), but $v was saved")
    f
end

"""
    FunctAI.load(path; node, types) -> AIFunction

An AI function saved in any language (Python, TypeScript, R, Julia), from
its folder (or its `functai.json`). It is checked to send exactly the bytes
it sent where it was saved; what it cannot run (code of its own, tools, a
baked model) is refused with a [`LoadRefused`](@ref) that says why.

A saved folder keeps shapes, not Julia types: answers come back as `String`,
numbers, `Vector`s, `Dict`s and `NamedTuple`s. `types` gives fields their
Julia types back when the shapes agree:

```julia
mood = FunctAI.load("saved/mood"; types = (result = Mood,))
mood("Arrived broken.")          # unhappy::Mood
```
"""
function load(path::AbstractString; node=nothing, types=(;))
    file = isdir(path) ? joinpath(path, "functai.json") : path
    text = read(file, String)
    manifest = try
        LMCC.parse_json(text)
    catch err
        refuse_load("saved-malformed", "$file: $(sprint(showerror, err))")
    end
    from_manifest(manifest; node, types, saved_id="sha256:" * LMCC.sha256_hex(text))
end

"""
    to_manifest(f) -> Dict

The manifest of an AI function defined here: the part every language reads
(contract/saved.md). See [`save`](@ref).
"""
function to_manifest(f::AIFunction)
    f.body === nothing || refuse_load("saved-code",
        "$(f.definition.name) runs Julia code of its own after the model answers; a saved folder carries no Julia code yet")
    isempty(f.tools) || refuse_load("saved-tools", "$(f.definition.name) has tools: a tool is code, and a saved folder carries none from Julia yet")
    s = f.own
    key = "$(f.module_name):$(f.definition.name)"
    settings = LMCC.jobj("module" => get(s, :reasoning, false) ? "cot" : "predict",
                         "include_fn_name_in_instructions" => get(s, :include_name, true))
    get(s, :lm, nothing) isa AbstractString && (settings["lm"] = s[:lm])
    haskey(s, :adapter) && (settings["adapter"] = adapter_data(s[:adapter]))
    for k in SNAKE_SETTINGS
        v = get(s, k, nothing)
        v === nothing || (settings[String(k)] = v isa Symbol ? String(v) : jsonvalue(v))
    end
    config = config_of(s)
    requests = String[]
    sig = signature(f)
    probes = Any[sample_inputs(sig)]
    for d in f.demos[1:min(3, length(f.demos))]
        ins = get(d, "inputs", nothing)
        ins isa AbstractDict && !any(p -> LMCC.json_equal(p, ins), probes) && push!(probes, ins)
    end
    for p in probes
        push!(requests, request_hash(f, p))
    end
    ai = LMCC.jobj("settings" => settings, "config" => config === nothing ? JObj() : LM15.to_dict(config),
                   "template" => haskey(s, :template) ? Any[template_messages(s[:template])...] : nothing,
                   "tools" => Any[], "teacher" => nothing,
                   "state" => LMCC.jobj("instructions" => f.instructions, "demos" => Any[LMCC.deepcopy_json(d) for d in f.demos]),
                   "requires" => Any[], "signature" => LMCC.signature_to_dict(sig), "probes" => probes,
                   "fingerprints" => LMCC.jobj("signature" => LMCC.signature_fingerprint(sig), "requests" => Any[requests...]),
                   "body" => nothing, "version" => version(f))
    LMCC.jobj("functai_saved" => SAVED_FORMAT, "language" => "julia", "entry" => key,
              "created" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "+00:00",
              "functai" => string(FUNCTAI_VERSION),
              "nodes" => LMCC.jobj(key => LMCC.jobj("kind" => "ai", "module" => f.module_name, "name" => f.definition.name, "ai" => ai)))
end

"""
    FunctAI.save(path, f) -> the manifest's path

Write an AI function to a folder (its `functai.json`), for any language's
loader: the signature, the instruction, the worked examples, the layout and
the settings, with the fingerprints a loader checks. A function with code of
its own or tools is refused: they are Julia code, which the folder cannot
carry yet.
"""
function save(path::AbstractString, f::AIFunction)
    manifest = to_manifest(f)
    mkpath(path)
    file = joinpath(path, "functai.json")
    write(file, LMCC.json_text(manifest; spaced=true) * "\n")
    file
end
