# The contract's cases for long agents and pipelines (contract/cases/agents),
# and the answer so far (contract/cases/partial),
# run through the Julia implementation: the same cases Python and TypeScript pass.

"What a tool written in Julia returns for the case's data: a text, lm15's part, or a list of them."
native_result(x::AbstractDict) = LM15.from_dict(LM15.Part, Dict{String,Any}(x))
native_result(x::AbstractVector) = Any[native_result(v) for v in x]
native_result(x) = x

@testset "agents case $name" for (name, c) in cases("agents")
    kind = c["kind"]
    if kind == "result"
        result = FunctAI.result_of(native_result(c["returned"]))
        @test same(result, c["expect"]["result"])
        event = Dict{String,Any}("output" => FunctAI.result_text(result))
        result isa AbstractString || (event["parts"] = result)
        @test same(event, c["expect"]["event"])
    elseif kind == "state"
        now = at_time(c["now"])
        old_clock = FunctAI.CONVERSATION_CLOCK[]
        FunctAI.CONVERSATION_CLOCK[] = () -> now
        try
            log = FunctAI.apply!(FunctAI.ConvLog(), c["records"])
            e = c["expect"]
            @test Dict(t => FunctAI.turn_state(st) for (t, st) in log.turns) == e["states"]
            @test Dict(t => Any[x["key"] for x in FunctAI.undelivered(st, log.delivered)] for (t, st) in log.turns
                       if !isempty(FunctAI.undelivered(st, log.delivered))) == e["later"]
            @test Dict(t => Any[x["invocation"] for x in FunctAI.unfinished(st)] for (t, st) in log.turns
                       if !isempty(FunctAI.unfinished(st))) == e["unfinished"]
            @test same(Dict(k => r["output"] for (k, r) in log.delivered), e["delivered"])
            resumes = Dict(t => (isempty(FunctAI.unanswered(st)) && isempty(FunctAI.undelivered(st, log.delivered)) ? true :
                                 Dict("refuses" => "turn-state")) for (t, st) in log.turns if FunctAI.turn_state(st) == "waiting")
            @test same(resumes, e["resumes"])
        finally
            FunctAI.CONVERSATION_CLOCK[] = old_clock
        end
    elseif kind == "kept"
        keeps = Dict(call => FunctAI.Keep(FunctAI.OrderedDict{String,Bool}(k["inputs"]), FunctAI.OrderedDict{String,Bool}(k["outputs"]))
                     for (call, k) in c["kept"])
        got = FunctAI.kept_log(c["events"], keeps)
        @test same(Any[FunctAI.event_json(e) for e in got], c["expect"]["events"])
    elseif kind == "limits"
        plugin = limits(; (Symbol(k) => v for (k, v) in c["limits"])...)
        sp = c["spent"]
        spent = Spent(Any[(x[1], x[2]) for x in sp["exchanges"]], sp["seconds"], get(sp, "reported", 0.0))
        at = c["at"]
        handler = only(plugin.handlers[Symbol(at["hook"])])
        got = at["hook"] == "before_call" ? handler((spent=spent, lm=at["lm"])) :
              handler((spent=spent, step=at["step"], max_steps=at["max_steps"], tools=String[at["tools"]...],
                       last_step=at["step"] >= at["max_steps"]))
        want = c["expect"]["change"]
        if want === nothing
            @test got === nothing
        else
            @test got isa Change && same(Dict(String(k) => getfield(got, k) for k in FunctAI.change_fields(got)), want)
        end
    elseif kind == "spent"
        s = Spent(Any[(x[1], x[2]) for x in c["exchanges"]], 0.0, get(c, "reported", 0.0))
        @test same(s.usage, c["expect"]["usage"])
        @test s.tokens == c["expect"]["tokens"]
        got, want = FunctAI.cost(s, c["prices"]), c["expect"]["cost"]
        @test (got === nothing) == (want === nothing)
        want === nothing || @test abs(got - want) < 1e-12
    elseif kind == "pictures"
        results = [FunctAI.StepResult(r["invocation"], "look", Dict{String,Any}(), r["output"]) for r in c["results"]]
        got = FunctAI.pictures_to_drop(results; last=c["last"], at_once=c["at_once"])
        @test same(Dict(string(k) => v for (k, v) in got), c["expect"]["replace"])
    else
        error("a kind of agents case this harness does not know: $kind")
    end
end
@test length(cases("agents")) >= 37

@testset "partial case $name" for (name, c) in cases("partial")
    for check in c["checks"]
        got = answer_so_far(check["text"], c["shape"])
        @test same(got === nothing ? Dict("nothing" => true) : Dict("value" => got), check["expect"])
    end
end
