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
`saved-tools`, `saved-model`, `saved-differs`, `saved-no-interface`,
`interface-malformed`: contract/saved.md).
"""
struct LoadRefused <: Exception
    code::String
    msg::String
end
Base.showerror(io::IO, e::LoadRefused) = print(io, "LoadRefused [", e.code, "]: ", e.msg)
refuse_load(code, msg) = throw(LoadRefused(code, msg))

const SNAKE_SETTINGS = (:retries, :api_retries, :max_steps, :tool_errors, :capabilities, :log_content)


"""
Why a manifest does not pass `schema/saved.schema.json` (with
`schema/interface.schema.json` for every node's interface), read as the
schemas themselves, or `nothing`.
"""
manifest_fault(m) = schema_fault("saved", m)

"The manifest's first checks (saved.md, step 1): its format, then its schema."
function check_form(m)
    m isa AbstractDict || refuse_load("saved-malformed", "functai.json is not a JSON object")
    get(m, "functai_saved", nothing) == SAVED_FORMAT ||
        refuse_load("saved-format", "functai.json is format $(repr(get(m, "functai_saved", nothing))); this loader reads format $SAVED_FORMAT")
    why = manifest_fault(m)
    why === nothing || refuse_load("saved-malformed", why)
    m
end

"The plain fields of an AI node's signature, as an interface's fields (by direction)."
plain_fields(n, direction) = [f for f in n["ai"]["signature"]["fields"]
                              if something(get(f, "purpose", nothing), "plain") == "plain" && f["direction"] == direction]

"An AI node's interface must describe the data its signature takes and gives (saved.md, step 6)."
function check_node_interface(key, n)
    iface = n["interface"]
    fault = interface_fault(iface; ai=n["kind"] == "ai")
    fault === nothing || refuse_load("interface-malformed", "$key: its interface is refused: $(fault.msg)")
    n["kind"] == "ai" || return iface
    theirs = interface_signature(LMCC.jobj("inputs" => plain_fields(n, "input"), "outputs" => plain_fields(n, "output")))
    interface_signature(iface) == theirs ||
        refuse_load("saved-differs", "$key: its interface promises other data than its signature takes and gives")
    iface
end

"""
    FunctAI.describe(path_or_manifest; node) -> Dict

What a saved program takes and gives, without loading or running anything
(contract/saved.md, "Describing without loading"): the node's interface
(the entry's by default), checked. A module saved in any language is
described too. An AI node written before nodes had an interface is
described by its signature (its instruction as the description: do not
show it to outside callers as it is). Throws [`LoadRefused`](@ref).
"""
function describe(m::AbstractDict; node=nothing)
    check_form(m)
    key = something(node, m["entry"])
    n = get(m["nodes"], key, nothing)
    n isa AbstractDict || refuse_load("saved-malformed", "functai.json has no node $(repr(key))")
    n["kind"] in ("ai", "module") || refuse_load("saved-not-ai", "$key is a $(n["kind"]): plain code, with no interface")
    haskey(n, "interface") && return LMCC.deepcopy_json(check_node_interface(key, n))
    n["kind"] == "module" && refuse_load("saved-no-interface", "$key was saved before programs had interfaces: what it takes is not known")
    LMCC.deepcopy_json(signature_interface(key, n))
end

"""
The interface an AI node written before nodes had one is read from
(saved.md, "Describing without loading"): its signature's plain fields, its
instruction as the description, no input optional. Checked as every
interface read from a folder is (`interface-malformed`).
"""
function signature_interface(key, n)
    field(f) = begin
        out = LMCC.jobj("name" => f["name"], "shape" => LMCC.deepcopy_json(f["shape"]))
        d = get(f, "desc", nothing)
        d isa AbstractString && !isempty(d) && (out["desc"] = d)
        t = get(f, "type", nothing)
        t isa AbstractString && (out["type"] = t)
        out
    end
    iface = LMCC.jobj("description" => n["ai"]["signature"]["instructions"],
                      "inputs" => Any[field(f) for f in plain_fields(n, "input")],
                      "outputs" => Any[field(f) for f in plain_fields(n, "output")])
    fault = interface_fault(iface; ai=true)
    fault === nothing || refuse_load("interface-malformed", "$key: the interface its signature gives is refused: $(fault.msg)")
    iface
end
describe(path::AbstractString; node=nothing) = describe(first(manifest_at(path)); node)

function manifest_at(path::AbstractString)
    file = isdir(path) ? joinpath(path, "functai.json") : path
    text = read(file, String)
    manifest = try
        LMCC.parse_json(text)
    catch err
        refuse_load("saved-malformed", "$file: $(sprint(showerror, err))")
    end
    (manifest, text)
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
    declared = haskey(n, "interface") ? JObj(check_node_interface(key, n)) : nothing
    declared === nothing && signature_interface(key, n)            # an old folder's: checked, as describing checks it
    optional = Dict{String,Any}()          # an optional input's default, from the node's interface (saved.md, step 5)
    if declared !== nothing
        for f in declared["inputs"]
            get(f, "optional", false) === true && (optional[f["name"]] = f["shape"]["default"])
        end
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
        desc = f.direction == "input" ? f.desc : nothing
        f.direction == "input" && haskey(optional, f.name) || return FieldDef(f.name, spec, JObj(f.shape), desc)
        # the default is the JSON the folder keeps, sent as it is: never read as a type (a loaded function runs no code)
        default = optional[f.name]
        # a default that counts by its code keeps it (saved.md, a node's defaults): the loaded version counts it so
        code = get(get(get(n, "defaults", JObj()), f.name, JObj()), "code", nothing)
        FieldDef(f.name, spec, data_shape(JObj(f.shape)), desc, true, LMCC.deepcopy_json(default), LMCC.deepcopy_json(default),
                 code === nothing ? nothing : String(code), nothing)
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
        v === nothing || (own[k] = k === :tool_errors ? Symbol(v) : k === :log_content ? content_setting(v) : v)
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
    # described as describe() says: the node's interface, else its instruction (all an old folder holds)
    definition = Definition(String(n["name"]), declared === nothing ? sig.instructions : declared["description"], inputs, outputs, sig.instructions)
    f = AIFunction(definition, own, AITool[], String(n["module"]), nothing, nothing, saved_id,
                   get(state, "instructions", nothing), Any[], nothing, nothing, nothing, nothing, nothing, nothing,
                   declared, Dict{Any,Any}(), ReentrantLock())
    check_own_content(f.own, program_fields(f; reasoning=get(f.own, :reasoning, false) === true), key)
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
    manifest, text = manifest_at(path)
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
              "nodes" => LMCC.jobj(key => saved_node(f, ai)))
end

"An AI node (saved.md): its interface, the code of each default that counts by it, and its `ai` entry."
function saved_node(f::AIFunction, ai)
    node = LMCC.jobj("kind" => "ai", "module" => f.module_name, "name" => f.definition.name,
                     "interface" => LMCC.deepcopy_json(interface(f)))
    code = JObj(x.name => LMCC.jobj("code" => x.code) for x in f.definition.inputs if x.code !== nothing)
    isempty(code) || (node["defaults"] = code)
    node["ai"] = ai
    node
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
    write(file, json_indented(manifest; width=1) * "\n")          # indented as Python and TypeScript write it: a folder to read and diff
    file
end
