# The contract every language shares (../../contract), held by the Julia
# implementation: the same cases Python, TypeScript and R pass.

const CONTRACT = normpath(joinpath(@__DIR__, "..", "..", "contract"))
read_json(path) = LMCC.parse_json(read(path, String))
cases(folder) = [(replace(f, ".json" => ""), read_json(joinpath(CONTRACT, "cases", folder, f)))
                 for f in sort(readdir(joinpath(CONTRACT, "cases", folder))) if endswith(f, ".json")]

refusal_of(f) = try
    f()
    nothing
catch e
    e
end
same(a, b) = LMCC.canonical_json(a) == LMCC.canonical_json(b)

"A value with no JSON form, standing for the case's {\"\$type\", \"\$repr\"} (the harness's own native value)."
struct Stand
    desc::Dict{String,Any}
end
FunctAI.jsonvalue(::Stand) = throw(FunctAI.NoJSON("a stand-in has no JSON form"))
is_stand(v) = v isa AbstractDict && Set(keys(v)) == Set(["\$type", "\$repr"])
native(v) = is_stand(v) ? Stand(Dict{String,Any}(v)) : v
back(v::Stand) = v.desc
back(v::AbstractDict) = Dict{String,Any}(String(k) => back(x) for (k, x) in v)
back(v::NamedTuple) = Dict{String,Any}(String(k) => back(x) for (k, x) in pairs(v))
back(v::AbstractVector) = Any[back(x) for x in v]
back(v) = v

"A module from an interface as data; its code keeps what it was given, and returns `returned[]`."
function capturing(iface)
    got = Ref{Any}(nothing)
    returned = Ref{Any}(nothing)
    p = AIProgram("m", (; kw...) -> (got[] = Dict{String,Any}(String(k) => v for (k, v) in kw); returned[]); interface=iface)
    (p, got, returned)
end

"What calling a module with `inputs` gives: the inputs its code got, or its refusal."
function module_inputs(p, got, inputs)
    got[] = nothing
    err = refusal_of(() -> p(; (Symbol(k) => native(v) for (k, v) in inputs)...))
    # the code ran (what it returned is another check's)
    (err === nothing || err isa InterfaceError && err.code == "interface-output") && return Dict("inputs" => back(got[]))
    @test got[] === nothing                            # the code did not run
    err isa InterfaceError ? Dict("refuses" => err.code, "field" => err.field) : err
end

"Valid inputs for a module, to call it for what it returns."
valid_inputs(iface) = Dict{String,Any}(f["name"] => get(f, "opaque", false) === true ? Stand(Dict{String,Any}("\$type" => "X", "\$repr" => "x")) :
                                       FunctAI.sample_value(f["shape"]) for f in iface["inputs"] if get(f, "optional", false) !== true)

"A function with the case's fields: its inputs, its outputs, reasoning and tools when FunctAI adds those fields."
function content_function(fields; own...)
    added = fields["added"]
    tools = "calls" in added ? [tool(x -> ""; name="ci_log", description="The CI log.", parameters=Dict("type" => "object"))] : []
    AIFunction("triage_build", "Is the build broken?"; inputs=[Symbol(n) => String for n in fields["inputs"]],
               outputs=[Symbol(n) => String for n in fields["outputs"] if !(n in added)],
               reasoning="reasoning" in added, tools, own...)
end


@testset "the contract" begin

@testset "the contract's data in data/contract is the contract's" begin
    for f in ["layouts/xml.json", "layouts/chat.json", "layouts/json.json", "models.json", "unicode/casefold.json",
              ("schema/$n.schema.json" for n in FunctAI.SCHEMA_NAMES)...]
        @test read(joinpath(FunctAI.CONTRACT_DATA, f), String) == read(joinpath(CONTRACT, f), String)
    end
end

@testset "the $name layout is the contract's" for name in ("xml", "chat", "json")
    mine = LMCC.dump(FunctAI.resolve_adapter(name); registry=FunctAI.registry())
    theirs = read_json(joinpath(CONTRACT, "layouts", "$name.json"))
    delete!(mine["versions"], "kernel")
    delete!(theirs["versions"], "kernel")
    @test LMCC.canonical_json(mine) == LMCC.canonical_json(theirs)
end

@testset "capabilities follow the contract's table" begin
    table = read_json(joinpath(CONTRACT, "models.json"))
    for p in table["native"]["providers"]
        caps = model_capabilities(p, "some-model")
        @test caps.stop_sequences == !(p in table["native"]["no_stop_sequences"])
        @test caps.native_function_calling
    end
    @test model_capabilities("anthropic", "claude-sonnet-4-5").native_reasoning
    @test model_capabilities("anthropic", "claude-3-5-haiku").assistant_prefill
    @test model_capabilities("claude-code", "claude-sonnet-4-5") == model_capabilities("anthropic", "claude-sonnet-4-5")
    @test model_capabilities("groq", "x").native_function_calling
    @test !model_capabilities("ollama", "x").native_function_calling
    @test model_capabilities("typesafe", "x") == (native_structured_output = true,)
end

"""
Call an AI function as a user does (keywords by name), where the probe facts
hold (contract/calls.md, "Versions"): on a fake model named `probe` with the
probe's capabilities (contract/models.json), and without the function's own
model and sampling settings (where a version runs is not part of what it
sends). The request it sent, and the record it wrote. Its reply is not read
(the call fails): what it sent and what it was called with are observed.
"""
function call_under_probe(f, inputs)
    router = FakeRouter(Any["(no answer)"]; provider="probe")
    dir = mktempdir()
    g = configure(f; lm="probe", temperature=nothing, top_p=nothing, max_tokens=nothing, seed=nothing, stop=nothing, config=nothing)
    try
        with_settings(router=router, capabilities=Dict{String,Any}(FunctAI.PROBE), retries=0, log_calls=dir) do
            g(; (Symbol(k) => v for (k, v) in inputs)...)
        end
    catch err
        err isa LMCC.Refusal || rethrow()
    end
    (request=only(router.requests), record=only(first(FunctAI.read_log(dir))))
end

"A contract definition, written the way a Julia user writes it without the macro (an optional input: `defaults`)."
function julia_function(d)
    field(f) = get(f, "desc", nothing) === nothing ? f["shape"] : f["shape"] => f["desc"]
    settings = Dict{Symbol,Any}()
    haskey(d["settings"], "adapter") && (settings[:adapter] = d["settings"]["adapter"])
    get(d["settings"], "module", nothing) == "cot" && (settings[:reasoning] = true)
    get(d["settings"], "include_fn_name_in_instructions", true) === false && (settings[:include_name] = false)
    tools = [tool(x -> ""; name=t["name"], description=t["description"], parameters=t["parameters"]) for t in get(d, "tools", Any[])]
    f = AIFunction(d["name"], d["description"];
                   inputs=[Symbol(x["name"]) => field(x) for x in d["inputs"]],
                   outputs=[Symbol(x["name"]) => field(x) for x in d["outputs"]],
                   defaults=[Symbol(x["name"]) => x["shape"]["default"] for x in d["inputs"] if get(x, "optional", false) === true],
                   tools, instructions=d["state"]["instructions"], settings...)
    isempty(d["state"]["demos"]) ? f : with_demos(f, d["state"]["demos"])
end

function without_type(sig)
    data = LMCC.signature_to_dict(sig)
    Dict("instructions" => data["instructions"],
         "fields" => [Dict(k => v for (k, v) in merge(Dict{String,Any}("purpose" => "plain"), f) if k != "type" && v !== nothing) for f in data["fields"]])
end

@testset "function case $name" for (name, c) in cases("functions")
    f = julia_function(c["definition"])
    want = c["expect"]["signature"]
    @test LMCC.canonical_json(without_type(FunctAI.signature(f))) ==
          LMCC.canonical_json(Dict("instructions" => want["instructions"],
                                   "fields" => [merge(Dict{String,Any}("purpose" => "plain"), x) for x in want["fields"]]))
    # the sample input is the function's own (calls.md, "Versions"), and so is the request rendered for it
    @test same(FunctAI.sample_inputs(FunctAI.signature(f)), c["expect"]["sample"])
    request = FunctAI.probe_request(f)
    @test LMCC.canonical_json(request) == LMCC.canonical_json(c["expect"]["request"])
    @test LMCC.sha256_of(request) == c["expect"]["request_hash"]
    @test version(f) == c["expect"]["version"]
    @test signature_id(f) == c["expect"]["signature_id"]
end
@test length(cases("functions")) >= 10

@testset "score case $name" for (name, c) in cases("scores")
    if c["kind"] == "interval"
        got = score_interval(Float64.(c["values"]))
        for k in (:mean, :low, :high)
            want = c["expect"][String(k)]
            want === nothing ? (@test got[k] === nothing) : (@test abs(got[k] - want) < 1e-12)
        end
    else
        ks = [k for k in keys(c["prediction"]) if haskey(c["answers"], k)]
        got = exact_match(c["answers"], Dict(k => c["prediction"][k] for k in ks))
        @test Dict(got) == Dict(k => Float64(v) for (k, v) in c["expect"])
    end
end

@testset "rated case $name" for (name, c) in cases("rated")
    recs = [r for r in c["records"] if haskey(r, "functai_call")]
    ratings = [r for r in c["records"] if haskey(r, "functai_rating")]
    q = c["rated"]
    rows, left = FunctAI.rated_rows(recs, ratings; name=q["name"], module_name=get(q, "module", nothing),
                                    signature=get(q, "signature", nothing), interface=get(q, "interface", nothing),
                                    by=get(q, "by", nothing))
    @test LMCC.json_text(rows) == LMCC.json_text(c["expect"]["rows"])
    @test LMCC.canonical_json(left) == LMCC.canonical_json(c["expect"]["left_out"])
end

@testset "saved case $name" for (name, c) in cases("saved")
    node = get(c, "node", nothing)
    if haskey(c["expect"], "refuses")
        err = refusal_of(() -> FunctAI.from_manifest(c["manifest"]; node))
        @test err isa LoadRefused && err.code == c["expect"]["refuses"]
    else
        f = FunctAI.from_manifest(c["manifest"]; node)
        want = c["expect"]["loads"]
        @test f.definition.name == want["name"]
        @test f.module_name == want["module"]
        @test version(f) == want["version"]
        @test signature_id(f) == want["signature_id"]
        probes = c["manifest"]["nodes"][something(node, c["manifest"]["entry"])]["ai"]["probes"]
        @test [FunctAI.request_hash(f, p) for p in probes] == want["requests"]
        # called with an optional input left out, it sends its default (saved.md, step 5): the request it sent
        for x in get(c["expect"], "sends", Any[])
            sent = call_under_probe(f, x["inputs"]).request
            @test LMCC.sha256_of(LM15.to_dict(sent)) == x["request_hash"]
        end
    end
    # described without loading
    want = c["expect"]["describe"]
    if haskey(want, "refuses")
        err = refusal_of(() -> FunctAI.describe(c["manifest"]; node))
        @test err isa LoadRefused && err.code == want["refuses"]
    else
        @test same(FunctAI.describe(c["manifest"]; node), want["interface"])
    end
end

# ------------------------------------------------------------------ programs/: interfaces

@testset "programs case $name" for (name, c) in cases("programs")
    kind = c["program"]
    if kind == "ai"
        if haskey(c["expect"], "refuses")
            err = refusal_of(() -> julia_function(c["definition"]))
            @test err isa InterfaceError && err.code == c["expect"]["refuses"] && err.field == c["expect"]["field"]
        else
            f = julia_function(c["definition"])
            @test same(FunctAI.interface(f), c["expect"]["interface"])
            @test FunctAI.interface_signature(f) == c["expect"]["signature"]
            @test signature_id(f) == c["expect"]["signature_id"]
            for b in c["binds"]
                # called with these inputs, the values it is called with: the ones its record holds
                @test same(Dict("inputs" => call_under_probe(f, b["inputs"]).record["inputs"]), b["expect"])
            end
        end
    elseif kind == "module"
        p, got, returned = capturing(c["interface"])
        @test FunctAI.interface_signature(p) == c["expect"]["signature"]
        for check in c["checks"]
            if haskey(check, "inputs")
                @test same(module_inputs(p, got, check["inputs"]), check["expect"])
            else
                returned[] = native(check["returned"]) isa Stand ? native(check["returned"]) :
                             check["returned"] isa AbstractDict ? Dict{String,Any}(k => native(v) for (k, v) in check["returned"]) : check["returned"]
                dir = mktempdir()
                err = refusal_of(() -> with_settings(() -> p(; (Symbol(k) => v for (k, v) in valid_inputs(c["interface"]))...); log_calls=dir))
                # the outputs the program gave: its record's (a value with no JSON form is written as a description:
                # the harness's own stand-in, read back as the case's)
                rec = only(first(FunctAI.read_log(dir)))
                result = if err === nothing
                    stands = Dict(k => v for (k, v) in (returned[] isa AbstractDict ? returned[] : Dict(only(c["interface"]["outputs"])["name"] => returned[])))
                    described = get(get(rec, "described", Dict()), "outputs", Any[])
                    Dict("outputs" => Dict(k => (k in described ? back(stands[k]) : v) for (k, v) in rec["outputs"]))
                else
                    @test rec["outputs"] === nothing && rec["error"]["code"] == err.code
                    err isa InterfaceError ? Dict("refuses" => err.code, "field" => err.field) : err
                end
                @test same(result, check["expect"])
            end
        end
    elseif kind == "definitions"
        for x in c["interfaces"]
            ai = get(x, "ai", false) === true
            fault = FunctAI.interface_fault(x["interface"]; ai)
            got = fault === nothing ? Dict("signature" => FunctAI.interface_signature(x["interface"])) :
                  Dict("refuses" => fault.code, "field" => fault.field)
            @test same(got, x["expect"])
            if !ai              # a module defined with it is refused, or defined, the same way
                err = refusal_of(() -> AIProgram("m", (; kw...) -> nothing; interface=x["interface"]))
                @test haskey(x["expect"], "refuses") ?
                      err isa InterfaceError && err.code == x["expect"]["refuses"] && err.field == x["expect"]["field"] : err === nothing
            end
        end
    elseif kind == "same-data"
        @test [FunctAI.interface_signature(x) for x in c["interfaces"]] == c["expect"]["signatures"]
        modules = [capturing(x) for x in c["interfaces"]]
        for check in c["checks"]
            @test same([module_inputs(p, got, check["inputs"]) for (p, got, _) in modules], check["expect"])
        end
    end
end

# ------------------------------------------------------------------ content/: what the log keeps

@testset "content case $name" for (name, c) in cases("content")
    layers = Dict(l["where"] => l["log_content"] for l in c["layers"])
    fields = c["fields"]
    refused = nothing
    f = nothing
    try
        f = content_function(fields; (haskey(layers, "own") ? (log_content=layers["own"],) : (;))...)   # a program's own: when it is defined
        block = haskey(layers, "block") ? (log_content=layers["block"],) : (;)
        configured = haskey(layers, "configure") ? layers["configure"] : nothing
        configured === nothing || FunctAI.configure!(log_content=configured)                 # configure's: when it is set
        try
            env = c["environment"] === nothing ? ("FUNCTAI_LOG_CONTENT" => nothing) : ("FUNCTAI_LOG_CONTENT" => c["environment"])
            withenv(env) do
                with_settings(; block...) do                                                  # a block's: when it is set
                    s = FunctAI.effective(f.own)
                    fs = FunctAI.program_fields(f, s)
                    @test fs.inputs == fields["inputs"] && fs.outputs == fields["outputs"] && fs.added == fields["added"]
                    keep = FunctAI.content_keep(fs, FunctAI.content_layers(f.own), FunctAI.environment_content())
                    written = FunctAI.written_record(c["record"], FunctAI.Keep(fs, keep))
                    @test same(Dict("record" => written), c["expect"])
                end
            end
        finally
            FunctAI.configure!(log_content=nothing)
        end
    catch e
        e isa LogContentError || rethrow()
        refused = e
    end
    if haskey(c["expect"], "refuses")
        @test refused isa LogContentError && refused.code == c["expect"]["refuses"] && refused.field == c["expect"]["field"]
    else
        @test refused === nothing
    end
end

# ------------------------------------------------------------------ saw/: what a call saw

@testset "saw case $name" for (name, c) in cases("saw")
    c["kind"] == "read" || continue            # "shown" is for languages that show context again (stage 3, 5)
    for q in c["queries"]
        known = try
            Dict("saw" => FunctAI.saw(c["records"], q["call"]))
        catch e
            e isa FunctAI.SawUnknown || rethrow()
            Dict("unknown" => e.code, "call" => e.call)
        end
        @test same(known, q["expect"])
        keeps = try
            FunctAI.keeps_saw(c["records"], q["call"])
            Dict("ok" => true)
        catch e
            e isa FunctAI.SawUnknown || rethrow()
            Dict("refuses" => e.code, "call" => e.call)
        end
        @test same(keeps, q["keeps"])
    end
end

@testset "a sampling the model does not take is left out, with one warning" begin
    s(; kw...) = Dict{Symbol,Any}(kw)
    @test FunctAI.adjust_settings(s(temperature=1), "openai", "gpt-6-luna")[:temperature] == 1
    got = @test_logs (:warn, r"does not take temperature, top_p") FunctAI.adjust_settings(s(temperature=0, top_p=0.5), "openai", "gpt-6-luna")
    @test got[:temperature] === nothing && got[:top_p] === nothing
    @test_logs FunctAI.adjust_settings(s(temperature=0, top_p=0.5), "openai", "gpt-6-luna")      # once per provider
    @test FunctAI.adjust_settings(s(temperature=0), "claude-code", "claude-sonnet-5")[:temperature] === nothing
    @test FunctAI.adjust_settings(s(temperature=0), "anthropic", "claude-haiku-4-5")[:temperature] == 0
    @test isempty(FunctAI.refused_settings("gemini", "gemini-3.8-flash"))
end

end
