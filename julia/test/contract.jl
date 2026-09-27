# The contract every language shares (../../contract), held by the Julia
# implementation: the same cases Python, TypeScript and R pass.

const CONTRACT = normpath(joinpath(@__DIR__, "..", "..", "contract"))
read_json(path) = LMCC.parse_json(read(path, String))
cases(folder) = [(replace(f, ".json" => ""), read_json(joinpath(CONTRACT, "cases", folder, f)))
                 for f in sort(readdir(joinpath(CONTRACT, "cases", folder))) if endswith(f, ".json")]

@testset "the contract" begin

@testset "the contract's data in data/contract is the contract's" begin
    for f in ["layouts/xml.json", "layouts/chat.json", "layouts/json.json", "models.json", "unicode/casefold.json"]
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
        @test caps["stop_sequences"] == !(p in table["native"]["no_stop_sequences"])
        @test caps["native_function_calling"]
    end
    @test model_capabilities("anthropic", "claude-sonnet-4-5")["native_reasoning"]
    @test model_capabilities("anthropic", "claude-3-5-haiku")["assistant_prefill"]
    @test model_capabilities("claude-code", "claude-sonnet-4-5") == model_capabilities("anthropic", "claude-sonnet-4-5")
    @test model_capabilities("groq", "x")["native_function_calling"]
    @test !model_capabilities("ollama", "x")["native_function_calling"]
    @test model_capabilities("typesafe", "x") == Dict("native_structured_output" => true)
end

"A contract definition, written the way a Julia user writes it without the macro."
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
    request = FunctAI.probe_request(f, c["expect"]["sample"])
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
                                    signature=get(q, "signature", nothing), by=get(q, "by", nothing))
    @test LMCC.json_text(rows) == LMCC.json_text(c["expect"]["rows"])
    @test LMCC.canonical_json(left) == LMCC.canonical_json(c["expect"]["left_out"])
end

@testset "saved case $name" for (name, c) in cases("saved")
    node = get(c, "node", nothing)
    if haskey(c["expect"], "refuses")
        err = try
            FunctAI.from_manifest(c["manifest"]; node)
            nothing
        catch e
            e
        end
        @test err isa LoadRefused && err.code == c["expect"]["refuses"]
    else
        f = FunctAI.from_manifest(c["manifest"]; node)
        want = c["expect"]["loads"]
        @test f.definition.name == want["name"]
        @test f.module_name == want["module"]
        @test version(f) == want["version"]
        @test signature_id(f) == want["signature_id"]
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
