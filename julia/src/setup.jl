# What a program runs on (contract/calls.md, "Setups"; design/18).
#
# A version says what a program *is*: its instruction, examples, layout,
# code. Its setup says what it runs on: which model answers each step, with
# which settings and plugins, with which tool code, able to reach what. Two
# runs of one version on two models have two setups, so a score, a
# comparison or a release belongs to exactly one thing that ran.
#
#     setup(mood)                       # "sha256:…", where it runs now
#     setup_document(mood)              # what it counts, as data
#     setup_diff(old_folder, new_folder)  # what differs, step by step

# FunctAI's settings that change what the model is asked or what comes back, with their defaults
const SETUP_SETTINGS = (("max_steps", 8), ("retries", 1), ("tool_errors", "report"), ("on_unreadable", "raise"))

# setups being worked out in this task: a program that comes back to itself names no setup
const SETUP_WORKING = ScopedValue{Tuple}(())

# ------------------------------------------------------------------ the model

"A baked model by its files (weights included): `sha256:` of its `baked.json`'s hashes."
baked_id(b::BakedModel) = LMCC.sha256_of(something(get(b.meta, "hashes", nothing), JObj()))

"""
A model as a setup names it: `{"provider", "model"}` as lm15 routes it (a
bare name and the name with its provider are one model), `{"baked": id}`,
`{"provider": null, "model": name}` for a name lm15 cannot route, or
`nothing` when none can be named (none set, none found).
"""
function setup_model(lm, s)
    lm isa BakedModel && return LMCC.jobj("baked" => baked_id(lm))
    lm === nothing && setting(s, :router) === nothing && (lm = default_model())
    lm === nothing && return nothing
    given = Dict{Symbol,Any}(s)
    given[:lm] = lm
    try
        r = resolve_route(given)
        LMCC.jobj("provider" => r.provider, "model" => r.wire)
    catch err
        err isa InterruptException && rethrow()
        LMCC.jobj("provider" => nothing, "model" => lm isa AbstractString ? model_string(lm) : string(lm))
    end
end

"The lm15 settings set in any layer, as lm15's canonical JSON; `nothing` when none is."
function setup_config(s)
    c = config_of(s)
    c === nothing && return nothing
    d = LM15.to_dict(c)
    isempty(d) ? nothing : JObj(d)
end

# ------------------------------------------------------------------ tools' code

const CODE_LOCK = ReentrantLock()
const METHOD_CODE = IdDict{Any,Any}()

"""
A method's code as data: its definition as written in its file (read there
and parsed, as `@program` hashes its own expression), or, when its file
cannot be read (a REPL), its lowered code. A value a closure captured, or
code it calls, is not in it: a tool's `code_id` says what identifies it
beyond that.
"""
function method_code(m::Method)
    lock(CODE_LOCK) do
        get!(METHOD_CODE, m) do
            file = string(m.file)
            expr = nothing
            if isfile(file)
                try
                    text = read(file, String)
                    starts = [0; findall(==(UInt8('\n')), codeunits(text))]
                    if 1 <= m.line <= length(starts)
                        ex, _ = Meta.parse(text, starts[m.line] + 1; raise=false)
                        ex isa Expr && ex.head in (:error, :incomplete) || (expr = ex)
                    end
                catch err
                    err isa InterruptException && rethrow()
                end
            end
            expr !== nothing && return LMCC.jobj("source" => code_data(expr isa Expr ? Base.remove_linenums!(expr) : expr))
            ci = try
                Base.uncompressed_ast(m)
            catch
                nothing
            end
            ci === nothing ? LMCC.jobj("method" => string(m.sig)) : LMCC.jobj("lowered" => Any[repr(x) for x in ci.code])
        end
    end
end

"`sha256:` of a function's code: each of its methods' (contract/calls.md, *Setups*: as the language reads its own code)."
function function_code(f)
    ms = sort!(collect(methods(f)); by=m -> (string(m.file), m.line, string(m.sig)))
    LMCC.sha256_of(Any[method_code(m) for m in ms])
end

"A tool as a setup counts it: its code, or the setup of the program it is."
function tool_entry(t::AITool)
    p = t.program
    (p isa AIFunction || (p isa AIProgram && p.remote === nothing)) && return LMCC.jobj("setup" => setup(p))
    p isa AIProgram && return LMCC.jobj("described" => true)               # a program served elsewhere: its server's
    code = LMCC.jobj("language" => "julia", "code" => function_code(t.run))
    t.code_id === nothing || (code["id"] = t.code_id)
    LMCC.jobj("code" => code)
end

# ------------------------------------------------------------------ reach (contract/calls.md, "Reach")

"""
What a program's code needs to reach (contract/calls.md, *Reach*): its own,
its tools', and (a program) every AI function's and program's it reaches.
"""
function needs_of(x, seen=Set{UInt}())
    objectid(x) in seen && return nothing
    push!(seen, objectid(x))
    if x isa AIFunction
        return merged_needs((t.program isa Union{AIFunction,AIProgram} ? needs_of(t.program, seen) : t.needs for t in x.tools)...)
    elseif x isa AIProgram
        return merged_needs(x.needs, (needs_of(r, seen) for r in reaches(x))...)
    end
    nothing
end

wall_of(s) = (w = setting(s, :wall); w === nothing ? nothing : w isa AbstractString ? LMCC.jobj("name" => w) : w)

"`\"open\"` with no wall; else what the wall lets through (or the program's needs), and its limits."
function reach_of(program, s)
    wall = wall_of(s)
    wall === nothing && return "open"
    needs = something(needs_of(program), JObj())
    out = JObj(k => Any[sort!(unique(String[x for x in get(wall, k, get(needs, k, Any[]))]))...] for k in ("domains", "secrets"))
    haskey(wall, "limits") && (out["limits"] = LMCC.deepcopy_json(wall["limits"]))
    out
end

lets_through(granted, domain) = any(g -> g == domain || (startswith(g, "*.") && endswith(domain, g[2:end])), granted)

"Refuse a program that needs what its wall does not let through (`ReachError`, `needs-unmet`), before its first call."
function check_reach(program, s)
    wall = wall_of(s)
    (wall === nothing || !(haskey(wall, "domains") || haskey(wall, "secrets"))) && return nothing
    needs = something(needs_of(program), JObj())
    missing_ = Dict{String,Any}("domains" => Any[], "secrets" => Any[])
    haskey(wall, "domains") && (missing_["domains"] = Any[d for d in get(needs, "domains", Any[]) if !lets_through(wall["domains"], d)])
    haskey(wall, "secrets") && (missing_["secrets"] = Any[x for x in get(needs, "secrets", Any[]) if !(x in wall["secrets"])])
    isempty(missing_["domains"]) && isempty(missing_["secrets"]) && return nothing
    what = join(vcat(String.(missing_["domains"]), ["the secret $x" for x in missing_["secrets"]]), ", ")
    throw(ReachError("$(program_name(program)) needs $what, which the wall $(repr(wall["name"])) does not let through", missing_))
end

"Where a call runs (its record's `placement`): the wall's name, and an AI function's server address."
function placement_of(s; ai::Bool)
    out = JObj()
    try
        wall = wall_of(s)
        wall === nothing || (out["wall"] = wall["name"])
        url = setting(s, :base_url)
        ai && url isa AbstractString && (out["base_url"] = url)
    catch err
        err isa InterruptException && rethrow()
    end
    isempty(out) ? nothing : out
end

# ------------------------------------------------------------------ the document

function plugin_entry(p::Plugin)
    out = LMCC.jobj("name" => p.name, "version" => p.version)
    p.config === nothing || (out["config"] = LMCC.deepcopy_json(p.config))
    out
end

"An AI function's setup document, with the settings in force now."
function ai_document(f::AIFunction)
    s = effective(f.own)
    doc = LMCC.jobj("version" => version(f), "model" => setup_model(get(s, :lm, nothing), s))
    config = setup_config(s)
    config === nothing || (doc["config"] = config)
    settings_ = JObj()
    for (k, default) in SETUP_SETTINGS
        v = get(s, Symbol(k), nothing)
        settings_[k] = v === nothing ? default : v isa Symbol ? String(v) : v
    end
    caps = get(s, :capabilities, nothing)
    caps === nothing || (settings_["capabilities"] = jsonvalue(caps))
    doc["settings"] = settings_
    target = get(s, :escalate_to, nothing)
    if target !== nothing
        to = target isa AIFunction ? LMCC.jobj("setup" => setup(target)) : setup_model(target, s)
        doc["escalate"] = LMCC.jobj("to" => to, "below" => something(get(s, :escalate_below, nothing), 0.9))
    end
    running = plugins_around(f.own)
    isempty(running) || (doc["plugins"] = Any[plugin_entry(p) for p in running])
    isempty(f.tools) || (doc["tools"] = JObj(t.name => tool_entry(t) for t in f.tools))
    doc["reach"] = reach_of(f, s)
    code_id = get(s, :code_id, nothing)
    code_id === nothing || (doc["code_id"] = String(code_id))
    doc
end

"The AI functions a program reaches, by the keys its version uses (`module:name`)."
function reached_ai(p::AIProgram)
    code, ai = JObj(), Dict{String,Any}()
    seen = Set{UInt}()
    walk(x) = begin
        objectid(x) in seen && return
        push!(seen, objectid(x))
        for r in reaches(x)
            r isa AIFunction ? (ai["$(r.module_name):$(r.definition.name)"] = r) : walk(r)
        end
    end
    walk(p)
    ai
end

"A program's setup document: its version, the setup of each AI function it reaches, its reach."
function module_document(p::AIProgram)
    s = effective(Dict{Symbol,Any}())
    doc = LMCC.jobj("version" => version(p),
                    "ai" => JObj(k => setup(x) for (k, x) in sort!(collect(reached_ai(p)); by=first)),
                    "reach" => reach_of(p, s))
    code_id = get(s, :code_id, nothing)
    code_id === nothing || (doc["code_id"] = String(code_id))
    doc
end

"""
    setup_document(program) -> Dict

What a program runs on, as data (contract/calls.md, *Setups*), with the
settings in force where it is asked: for an AI function its `version`, the
`model` it would ask, the lm15 settings set (`config`), FunctAI's settings
that change what the model is asked or what comes back (`settings`), the
backup model (`escalate`), its `plugins` in the order they run with their
config, its `tools`' code, and what it may `reach`; for a program its
`version`, the setup of every AI function it reaches (`ai`), and its `reach`.
"""
function setup_document(program::Union{AIFunction,AIProgram})
    program isa AIProgram && program.remote !== nothing &&
        throw(ArgumentError("$(program.name) is served elsewhere: what it runs on is its server's"))
    working = SETUP_WORKING[]
    any(x -> x === program, working) && return LMCC.jobj("cycle" => program_name(program))
    with(SETUP_WORKING => (working..., program)) do
        program isa AIFunction ? ai_document(program) : module_document(program)
    end
end

"""
    setup(program) -> String

What a program runs on, beside what it is (its [`version`](@ref)): `sha256:`
of the canonical JSON of its [`setup_document`](@ref), worked out where it
runs, from the settings in force there, before its first call and without
the network. The same function, on the same model, with the same settings,
plugins and walls, has one setup in every FunctAI language; on another
model, another setup. Every call records it (`program.setup`), so a score,
a comparison or a release belongs to exactly one thing that ran.

```julia
setup(mood)                                       # "sha256:…"
with_settings(lm = "claude-haiku-4-5") do
    setup(mood)                                   # another
end
```
"""
setup(program::Union{AIFunction,AIProgram}) = LMCC.sha256_of(setup_document(program))

"A call record's `program.setup`: what it runs on, when it can be worked out (a setup never stops a call)."
function known_setup!(info, program)
    try
        info["setup"] = setup(program)
    catch err
        err isa InterruptException && rethrow()
        warn_once("setup:$(program_name(program)):$(typeof(err))", "the setup of $(program_name(program)) could not be worked out " *
                  "($(sprint(showerror, err))); its records name none")
    end
    info
end

# ------------------------------------------------------------------ what differs

function setup_differences!(out, a, b, path)
    if a isa AbstractDict && b isa AbstractDict
        for k in sort!(collect(union(keys(a), keys(b))); by=string)
            setup_differences!(out, get(a, k, nothing), get(b, k, nothing), isempty(path) ? String(k) : "$path.$k")
        end
    elseif !same_json(a, b)
        push!(out, (path, a, b))
    end
    out
end

"Each program's setup document by key: a program here (and the AI functions it reaches), or a saved folder's nodes."
function setup_documents(source)
    if source isa Union{AIFunction,AIProgram}
        out = Dict{String,Any}(program_name(source) => setup_document(source))
        source isa AIProgram && for (k, x) in reached_ai(source)
            out[k] = setup_document(x)
        end
        return out
    end
    source isa AbstractString || throw(ArgumentError("setup_diff compares programs, or saved folders (their functai.json): not $(repr(source))"))
    path = isdir(source) ? joinpath(source, "functai.json") : source
    manifest = LMCC.parse_json(read(path, String))
    entry = get(manifest, "entry", nothing)
    out = Dict{String,Any}()
    for (key, node) in something(get(manifest, "nodes", nothing), JObj())
        haskey(node, "setup") || continue
        out[key == entry ? String(node["name"]) : String(key)] = node["setup"]
    end
    out
end

"""
    setup_diff(a, b) -> Vector{NamedTuple}

What differs between two setups, step by step: each `(program, field,
before, after)`, `field` a dotted path into the setup document
(`model.model`, `config.temperature`, `plugins`). `a` and `b`: programs
here, or saved folders. A table: `DataFrame(setup_diff(a, b))`.

```julia
setup_diff("releases/v1", "releases/v2")
# (program = "write_insights", field = "model.model", before = "claude-haiku-5-5", after = "gpt-6-luna")
```
"""
function setup_diff(a, b)
    da, db = setup_documents(a), setup_documents(b)
    out = NamedTuple{(:program, :field, :before, :after),Tuple{String,String,Any,Any}}[]
    for program in sort!(collect(union(keys(da), keys(db))))
        # a program's "ai" names its steps' setups by hash: each step is compared on its own instead
        strip_ai(d) = d isa AbstractDict ? JObj(k => v for (k, v) in d if k != "ai") : JObj()
        for (field, x, y) in setup_differences!(Any[], strip_ai(get(da, program, nothing)), strip_ai(get(db, program, nothing)), "")
            push!(out, (program=program, field=field, before=x, after=y))
        end
    end
    out
end
