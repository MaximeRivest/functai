# The Julia face: the macro, calling, columns, streaming, tools, retries,
# settings, evaluation, improving, saving. A fake router; no network.

@enum Mood happy unhappy mixed

struct Person
    name::String
    age::Int
end

xml(pairs...) = join(("<$k>\n$v\n</$k>" for (k, v) in pairs), "\n")
fake(replies...; kw...) = FakeRouter(Any[replies...]; kw...)
"The error a macro raises while expanding (Julia wraps it in a LoadError)."
expansion_error(ex) = try
    macroexpand(@__MODULE__, ex)
    nothing
catch e
    e isa LoadError ? e.error : e
end
using_fake(f, router; kw...) = with_settings(f; lm="gpt-4.1-mini", router, kw...)

@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end

@testset "the Julia face" begin

@testset "@ai: a typed answer, the contract's version" begin
    r = fake(xml(:result => "unhappy"))
    @test using_fake(() -> mood("Broke after a day."), r) === unhappy
    @test length(r.requests) == 1
    # the same function as the contract's case 01 (Python's Literal[...] choice)
    case01 = read_json(joinpath(CONTRACT, "cases", "functions", "01-an-answer-from-a-list.json"))["expect"]
    @test version(mood) == case01["version"]
    @test signature_id(mood) == case01["signature_id"]
    @test mood isa Function
    @test sprint(show, mood) == "mood(review::String) -> Mood"
    @test occursin("How does the customer feel", string(@doc mood))
    @test FunctAI.instructions(mood) == "Function: mood\n\nHow does the customer feel about what they bought?"
end

@testset "several outputs come back together; # Arguments describes the inputs" begin
    @ai function triage(ticket::String; team::String = "support")
        """
        Read the ticket.

        # Arguments
        - `ticket`: the customer's own words,
          as they wrote them
        """
        summary::String = ai"one sentence, no names"
        minutes::Int = ai"minutes to fix"
    end
    @test occursin("Parameter guidance:\n- ticket: the customer's own words, as they wrote them", FunctAI.instructions(triage))
    @test occursin("Output guidance:\n- summary: one sentence, no names\n- minutes: minutes to fix", FunctAI.instructions(triage))
    @test FunctAI.input_names(triage) == ["ticket", "team"]          # as written: positional, then keywords
    r = fake(xml(:summary => "Charged twice.", :minutes => "15"))
    (; summary, minutes) = using_fake(() -> triage("I was charged twice"), r)
    @test summary == "Charged twice." && minutes === 15
    @test occursin("<team>\nsupport\n</team>", r.requests[1].messages[end].parts[1].text)
    @test expansion_error(:(@ai function nope(x::String)
        """
        Do it.

        # Arguments
        - `y`: not an input
        """
    end)) isa ArgumentError
end

@testset "code of its own runs on the outputs; it is part of the version" begin
    @ai function price(item::String)::Float64
        "Estimate the price in US dollars."
        usd::Float64 = ai"the price"
        round(usd; digits = 1)
    end
    @ai function price_raw(item::String)::Float64
        "Estimate the price in US dollars."
        usd::Float64 = ai"the price"
    end
    r = fake(xml(:usd => "3.14159"))
    @test using_fake(() -> price("a mug"), r) == 3.1
    @test price.code !== nothing && price_raw.code === nothing
    @test version(price) != version(price_raw)
    @test expansion_error(:(@ai function wrong(x::String)
        "Two outputs."
        a::String = ai"first"
        b::String = ai"second"
        return a
    end)) isa ArgumentError
end

@testset "types: OneOf, structs, NamedTuples, lists, maps, optional" begin
    @ai function vibe(text::String)::OneOf(:calm, :tense)
        "The vibe."
    end
    @test using_fake(() -> vibe("hi"), fake(xml(:result => "tense"))) === :tense
    @ai adapter = :json function who(text::String)::Person
        "Who is described?"
    end
    @test using_fake(() -> who("Ana, 31"), fake("{\"result\": {\"name\": \"Ana\", \"age\": 31}}")) == Person("Ana", 31)
    @ai function tags(text::String)::Vector{String}
        "Tags."
    end
    @test using_fake(() -> tags("x"), fake(xml(:result => "[\"a\", \"b\"]"))) == ["a", "b"]
    @ai function counts(text::String)::Dict{String,Int}
        "Counts."
    end
    @test using_fake(() -> counts("x"), fake(xml(:result => "{\"a\": 2}"))) == Dict("a" => 2)
    @ai function maybe(text::String)::Union{Int,Nothing}
        "A number, if there is one."
    end
    @test using_fake(() -> maybe("x"), fake(xml(:result => "null"))) === nothing
    @ai function maybe2(text::String)::Union{Int,Missing}
        "A number, if there is one."
    end
    @test FunctAI.shape_of(Union{Int,Missing}) == FunctAI.shape_of(Union{Int,Nothing})
    @test using_fake(() -> maybe2("x"), fake(xml(:result => "null"))) === missing
    r = FakeRouter(; responder=(req, i) -> xml(:result => occursin("<text>\nx", req.messages[end].parts[1].text) ? "null" : "4"))
    got = using_fake(() -> maybe2.(["x", "y"]), r)
    @test isequal(got, [missing, 4]) && eltype(got) == Union{Missing,Int}
    @ai function pair(text::String)::@NamedTuple{a::Int, b::String}
        "Two things."
    end
    @test using_fake(() -> pair("x"), fake(xml(:result => "{\"a\": 1, \"b\": \"z\"}"))) == (a=1, b="z")
    @test FunctAI.shape_of(Dict{String,Int}) == Dict("type" => "object", "additionalProperties" => Dict("type" => "integer"))
    @test_throws ArgumentError FunctAI.shape_of(Tuple{Int,String})
end

@testset "predict. over a column; typed probabilities" begin
    r = FakeRouter(; responder=(req, i) -> (sleep(0.01); xml(:result => "happy")))
    ps = using_fake(() -> predict.(mood, ["a", "b", missing]), r)
    @test ps[1] isa Prediction && ps[1].value === happy && ps[3] === missing
    @test isempty(ps[1].probabilities)
    p = Prediction(happy, (result=happy,), "result", "id", nothing, Any[], Any[],
                   FunctAI.JObj("result" => FunctAI.JObj("happy" => 0.9, "unhappy" => 0.1, "mixed" => 0.0)))
    @test p.probabilities.result[happy] == 0.9 && keytype(p.probabilities.result) == Mood
end

@testset "the datasets" begin
    t = FunctAI.tickets()
    @test length(t.id) == 80 && eltype(t.id) == Int && count(ismissing, t.order_id) > 0
    @test startswith(t.message[1], "Hi, my order A-1042")
    r = FunctAI.refunds()
    @test length(r.id) == 120 && eltype(r.final_sale) == Bool && eltype(r.price) == Float64
    @test Set(r.decision) == Set(["approve", "deny"])
    n = FunctAI.field_notes()
    @test length(n.id) == 60 && eltype(n.count) == Union{Missing,Int}
end

@testset "missing in, missing out, without a call" begin
    r = fake()
    @test using_fake(() -> mood(missing), r) === missing
    @test isempty(r.requests)
end

@testset "a column runs concurrently and keeps its order" begin
    answers = Dict("a" => "happy", "b" => "unhappy", "c" => "mixed")
    inflight, most = Ref(0), Ref(0)
    r = FakeRouter(; responder=(req, i) -> begin
        inflight[] += 1
        most[] = max(most[], inflight[])
        sleep(0.02)
        inflight[] -= 1
        xml(:result => answers[match(r"<review>\n(.)\n", req.messages[end].parts[1].text)[1]])
    end)
    reviews = repeat(["a", "b", "c"], 8)
    got = using_fake(() -> mood.(reviews), r)
    @test got == repeat([happy, unhappy, mixed], 8)
    @test eltype(got) == Mood
    @test most[] == 8                            # 8 at a time (the default concurrency)
    most[] = 0
    using_fake(() -> mood.(reviews), r; concurrency=3)
    @test most[] == 3
    @test using_fake(() -> map(mood, ["a", missing]), r) |> x -> x[1] === happy && x[2] === missing
    # DataFrames' ByRow calls map through invoke: it gets the concurrent map too
    @test using_fake(() -> invoke(map, Tuple{typeof(mood),AbstractVector}, mood, ["b"]), r) == [unhappy]
end

@testset "failed rows are missing, with one warning; all failed throws" begin
    r = FakeRouter(; responder=(req, i) -> occursin("bad", req.messages[end].parts[1].text) ? LM15.AuthError("no key") : xml(:result => "happy"))
    got = @test_logs (:warn, r"1 of 3 calls of mood failed") match_mode = :any using_fake(() -> mood.(["ok", "bad", "ok"]), r)
    @test isequal(got, [happy, missing, happy])
    @test only(problems()).row == 2
    @test_throws LM15.AuthError using_fake(() -> mood.(["bad", "bad"]), r)
end

@testset "an unreadable reply is asked again with lmcc's hint" begin
    r = fake("I think they are sad", xml(:result => "unhappy"))
    @test using_fake(() -> mood("x"), r) === unhappy
    @test length(r.requests) == 2
    @test startswith(r.requests[2].messages[end].parts[1].text, "Your reply could not be read: ")
    # a choice outside the list is unreadable too
    r = fake(xml(:result => "angry"), xml(:result => "mixed"))
    @test using_fake(() -> mood("x"), r) === mixed
    @test occursin("angry", r.requests[2].messages[end].parts[1].text)
    # retries = 0: the refusal is the error
    r = fake("nope")
    err = try using_fake(() -> mood("x"), r; retries=0) catch e e end
    @test err isa LMCC.Refusal && startswith(err.code, "parse-")
end

@testset "a cut-off reply is sent again with twice the budget" begin
    r = fake((text="<result>\nhap", finish="length"), xml(:result => "happy"))
    @test using_fake(() -> mood("x"), r; max_tokens=100) === happy
    @test r.requests[2].config.max_tokens == 200
end

# functions.md, "When the reply cannot be read": a cut reply, all thinking and no answer.
cut_off_reply(req; adaptations=()) = LM15.Response(; model=req.model, finish_reason="length",
    message=LM15.Message(; role="assistant", parts=(LM15.ThinkingPart(; text="still thinking"),)),
    usage=LM15.Usage(; input_tokens=3, output_tokens=900, reasoning_tokens=900), adaptations=Tuple(adaptations))

@testset "a cut-off reply: twice a set budget, never a guessed one; the refusal says what happened" begin
    r = FakeRouter(; responder=(req, i) -> cut_off_reply(req))
    err = try using_fake(() -> mood("x"), r; max_tokens=500, retries=2) catch e e end
    @test [q.config.max_tokens for q in r.requests] == [500, 1000, 2000]
    @test err isa LMCC.Refusal && endswith(err.hint,
        "; the model spent 900 of its 900 output tokens thinking; raise max_tokens (it was 2000) or ask for less")
    # no budget: the reply had the most the call allows; the old rule re-sent it with 2048 after 128000
    notes = (LM15.Adaptation(; field="config.max_tokens", action="defaulted",
                             reason="the Messages API requires max_tokens and none was set; the model's output ceiling was used", applied=128000),
             LM15.Adaptation(; field="config.reasoning.thinking_budget", action="dropped",
                             reason="budget_tokens is rejected by the API", asked=32000))
    r = FakeRouter(; responder=(req, i) -> cut_off_reply(req; adaptations=notes))
    err = try using_fake(() -> mood("x"), r; retries=2) catch e e end
    @test length(r.requests) == 1
    @test endswith(err.hint, "; the model spent 900 of its 900 output tokens thinking; no max_tokens was set, and lm15 sent 128000, " *
        "the most it knows this model to allow: lower the reasoning effort or ask for less " *
        "(lm15 adapted the request: config.reasoning.thinking_budget dropped: budget_tokens is rejected by the API)")
    r = FakeRouter(; responder=(req, i) -> cut_off_reply(req))
    err = try using_fake(() -> mood("x"), r; retries=2) catch e e end
    @test length(r.requests) == 1
    @test endswith(err.hint, "; no max_tokens was set, so the provider used its own maximum: lower the reasoning effort or ask for less")
end

@testset "a transient provider error is sent again" begin
    r = fake(LM15.RateLimitError("slow down"; retry_after=0.01), xml(:result => "happy"))
    @test using_fake(() -> mood("x"), r) === happy
    @test length(r.requests) == 2
end

@testset "a reply the provider stopped is never an answer, and is not asked again" begin
    r = fake((text = "", finish = "content_filter"), xml(:result => "happy"))
    err = try using_fake(() -> mood("x"), r; retries = 2) catch e e end
    @test err isa FunctAI.LMCC.Refusal && err.code == "parse-filtered" && length(r.requests) == 1
    @test occursin("safety filter, or the model declining", err.hint)
    r = fake((text = xml(:result => "happy"), finish = "content_filter"))
    err = try using_fake(() -> mood("x"), r) catch e e end
    @test err isa FunctAI.LMCC.Refusal && err.code == "parse-filtered"
end

@testset "a provider asking for a longer wait than api_max_wait is not waited: raised at once, saying until when" begin
    r = fake(LM15.RateLimitError("window spent"; retry_after=11270), xml(:result => "happy"))
    started = time()
    err = try using_fake(() -> mood("x"), r) catch e e end
    @test err isa LM15.RateLimitError && length(r.requests) == 1 && time() - started < 5
    msg = sprint(showerror, err)
    @test occursin(r"wait 3 h 7 min \(until \d{4}-\d\d-\d\d \d\d:\d\d UTC\)", msg)
    @test occursin("api_max_wait=60 s", msg) && occursin("api_max_wait = 11270", msg)
    @test err.retry_after == 11270
    @test_throws ArgumentError FunctAI.check_setting(:api_max_wait, -1)
    @test FunctAI.wait_duration.([45, 125, 720, 3600, 11270]) == ["45 s", "2 min 5 s", "12 min", "1 h", "3 h 7 min"]
end

@testset "tools: the model calls Julia functions" begin
    orders = Dict("A1" => "shipped")
    "Look up an order by its number."
    lookup(order::String) = orders[order]
    @ai tools = [lookup] function helper(question::String)::String
        "Help with the order."
    end
    @test helper.tools[1].description == "Look up an order by its number."
    @test helper.tools[1].parameters["properties"]["order"] == Dict("type" => "string")
    r = fake((calls=[("c1", "lookup", (order="A1",))],), xml(:result => "It shipped."))
    @test using_fake(() -> helper("Where is A1?"), r) == "It shipped."
    tool_msg = r.requests[2].messages[end]
    @test tool_msg.role == "tool" && tool_msg.parts[1].content[1].text == "shipped"
    # a tool's error is reported to the model
    r = fake((calls=[("c1", "lookup", (order="ZZ",))],), xml(:result => "No such order."))
    @test using_fake(() -> helper("Where is ZZ?"), r) == "No such order."
    @test startswith(r.requests[2].messages[end].parts[1].content[1].text, "error: KeyError")
    # the loop stops at max_steps
    r = FakeRouter(; responder=(req, i) -> (calls=[("c$i", "lookup", (order="A1",))],))
    @test_throws StepLimit using_fake(() -> helper("loop"), r; max_steps=2)
end

@testset "streaming: the same call, watched" begin
    r = fake(xml(:result => "unhappy"); piece=4)
    s = using_fake(() -> stream(mood, "Broke."), r)
    pieces = collect(s)
    @test join(pieces) == "unhappy"
    @test length(pieces) > 1
    @test fetch(s) === unhappy
    kinds = [e.kind for e in eachevent(s)]
    @test first(kinds) === :started && last(kinds) === :done
    @test all(e -> FunctAI.event_json(e) isa AbstractDict, eachevent(s))
    # a router with no stream: one piece per field (nothing invented)
    s = using_fake(() -> stream(mood, "Broke."), Whole(fake(xml(:result => "mixed"))))
    @test collect(s) == ["mixed"] && fetch(s) === mixed
    # the do form
    seen = String[]
    @test using_fake(() -> stream(p -> push!(seen, p), mood, "x"), fake(xml(:result => "happy"))) === happy
    @test join(seen) == "happy"
    # a retry shows as an event
    s = using_fake(() -> stream(mood, "x"), fake("??", xml(:result => "happy")))
    @test fetch(s) === happy && any(e -> e.kind === :retry, eachevent(s))
    # closing cancels
    slow = FakeRouter(; responder=(req, i) -> (sleep(0.3); xml(:result => "happy")))
    s = using_fake(() -> stream(mood, "x"), slow)
    close(s)
    @test_throws Cancelled fetch(s)
end

@testset "render: the request, without sending it" begin
    r = fake()
    req = using_fake(() -> render(mood, "Broke."), r; temperature=0)
    @test req isa LM15.Request && req.model == "gpt-4.1-mini" && req.config.temperature == 0
    @test isempty(r.requests)
end

@testset "settings: own beat with_settings beat configure!" begin
    r = fake(xml(:result => "happy"), xml(:result => "happy"), xml(:result => "happy"))
    FunctAI.configure!(lm="configured-model", router=r, temperature=0.5)
    try
        mood("x")
        @test r.requests[end].model == "configured-model" && r.requests[end].config.temperature == 0.5
        with_settings(temperature=0.2) do
            mood("x")
        end
        @test r.requests[end].config.temperature == 0.2
        configure(mood; temperature=0.1)("x")
        @test r.requests[end].config.temperature == 0.1
    finally
        FunctAI.configure!(lm=nothing, router=nothing, temperature=nothing)
    end
    @test_throws ArgumentError FunctAI.configure!(modle="x")
    @test occursin("did you mean lm", try FunctAI.configure!(lmm="x") catch e sprint(showerror, e) end)
    @test occursin("did you mean temperature", try FunctAI.configure!(temprature=0) catch e sprint(showerror, e) end)
    @test_throws ArgumentError with_settings(() -> nothing; (Symbol("module") => :cot,)...)
    @test settings(configure(mood; temperature=0)) == (temperature=0,)
end

@testset "reasoning = true: the model reasons first" begin
    @ai reasoning = true function solve(problem::String)::Float64
        "Solve the word problem."
    end
    @test any(f -> f.name == "reasoning", FunctAI.signature(solve).fields)
    r = fake(xml(:reasoning => "2 + 2", :result => "4"))
    p = using_fake(() -> predict(solve, "Two plus two?"), r)
    @test p.value == 4.0 && p.outputs.reasoning == "2 + 2"
end

@testset "a template of pairs" begin
    @ai template = [:system => "You judge moods.", :user => "Review: {review}"] function mood_t(review::String)::String
        "How does the customer feel?"
    end
    r = fake("  unhappy  ")
    @test using_fake(() -> mood_t("Broke."), r) == "unhappy"
    @test r.requests[1].messages[end].parts[1].text == "Review: Broke."
end

@testset "worked examples and instructions are copies" begin
    taught = with_demos(mood, ["Broke in a day." => unhappy, (review="Love it!", result=happy)])
    @test length(taught.demos) == 2 && isempty(mood.demos)
    @test version(taught) != version(mood)
    r = fake(xml(:result => "happy"))
    using_fake(() -> taught("x"), r)
    @test length(r.requests[1].messages) == 5
    better = with_instructions(mood, "Say how the customer feels.")
    @test FunctAI.instructions(better) == "Say how the customer feels."
    @test with_instructions(better, nothing) |> version == version(mood)
end

@testset "AIFunction without the macro" begin
    f = AIFunction("mood", "How does the customer feel about what they bought?";
                   inputs=(review=String,), output=Mood)
    @test version(f) == version(mood) && signature_id(f) == signature_id(mood)
    @test using_fake(() -> f("x"), fake(xml(:result => "mixed"))) === mixed
end

@testset "evaluate, compare" begin
    rows = [(review="Broke.", result="unhappy"), (review="Love it", result="happy"),
            (review="Fine, late", result="mixed"), (review="Meh", result="unhappy")]
    r = FakeRouter(; responder=(req, i) -> xml(:result => occursin("Love", req.messages[end].parts[1].text) ? "happy" : "unhappy"))
    e = using_fake(() -> evaluate(mood, rows), r)
    @test e.score == 0.75 && e.low !== nothing && e.high !== nothing
    @test length(collect(FunctAI.Tables.rows(e))) == 4
    @test occursin("exact_match", sprint(show, MIME"text/plain"(), e))
    r2 = FakeRouter(; responder=(req, i) -> xml(:result => "happy"))
    e2 = using_fake(() -> evaluate(mood, rows), r2)
    c = only(compare(e, e2))
    @test c.before == 0.75 && c.after == 0.25 && c.worse == 2
    with_rules = [(; r..., rules="refunds are billing") for r in rows]
    e3 = using_fake(() -> evaluate(mood, with_rules), FakeRouter(; responder=(req, i) -> xml(:result => "happy")))
    err = try compare(e, e3) catch x x end
    @test err isa ArgumentError && occursin("row 1 differs: column \"rules\" only in the second. Evaluate both versions", err.msg)
    # a conventional model is evaluated the same way
    rule(row) = occursin("Love", row.review) ? "happy" : "unhappy"
    @test evaluate(rule, rows; expected=:result).score == 0.75
    # a custom metric
    e3 = using_fake(() -> evaluate(mood, rows; metric=(row, out) -> out.result === happy ? 1 : 0), r2)
    @test e3.score == 1.0
end

@testset "labeled_few_shot, bootstrap_few_shot" begin
    rows = [(review="r$i", result=isodd(i) ? "happy" : "unhappy") for i in 1:10]
    l = labeled_few_shot(mood, rows; k=3)
    @test length(l.demos) == 3 && isempty(mood.demos)
    r = FakeRouter(; responder=(req, i) -> xml(:result => "happy"))
    b = using_fake(() -> bootstrap_few_shot(mood, rows; max_bootstrapped=2, max_labeled=4), r)
    @test length(b.demos) == 4
    @test count(d -> haskey(d, "steps"), b.demos) == 2      # two recorded runs the metric accepted
end

@testset "random_search, instruction_search" begin
    rows = [(review="r$i", result=isodd(i) ? "happy" : "unhappy") for i in 1:6]
    r = FakeRouter(; responder=(req, i) -> occursin("Propose a new instruction", something(req.system, "")) ?
        xml(:result => "Say the mood in one word. ($i)") : xml(:result => occursin(r"r[135]", req.messages[end].parts[1].text) ? "happy" : "unhappy"))
    best, trials = using_fake(() -> random_search(mood, rows; candidates=2, max_bootstrapped=2, max_labeled=2), r)
    @test best isa AIFunction && length(trials) == 5 && all(t -> 0 <= t.score <= 1, trials)
    best, trials = using_fake(() -> instruction_search(mood, rows; candidates=3, trials=4, max_bootstrapped=1, max_labeled=1), r)
    @test length(trials) == 4 && best isa AIFunction
    @test any(t -> t.instruction_text !== nothing && startswith(t.instruction_text, "Say the mood"), trials)
    # the proposer sees the function's inputs and outputs, never other columns
    text = FunctAI.examples_text(mood, [FunctAI.row_dict((review="r", result="happy", secret="the state"))])
    @test occursin("happy", text) && !occursin("the state", text)
end

@testset "prompt: the request as a conversation" begin
    p = using_fake(() -> FunctAI.prompt(with_demos(mood, ["Broke." => unhappy]), "It broke."), fake())
    shown = sprint(show, MIME"text/plain"(), p)
    @test occursin("system\nFunction: mood", shown) && occursin("assistant\n<result>\nunhappy", shown)
    @test p.request isa LM15.Request
end

@testset "save and load: the same version, types given back" begin
    dir = mktempdir()
    taught = with_demos(configure(mood; temperature=0), ["Broke in a day." => unhappy])
    FunctAI.save(dir, taught)
    loaded = FunctAI.load(dir)
    @test version(loaded) == version(taught) && signature_id(loaded) == signature_id(taught)
    @test using_fake(() -> loaded("x"), fake(xml(:result => "happy"))) == "happy"
    typed = FunctAI.load(dir; types=(result=Mood,))
    @test using_fake(() -> typed("x"), fake(xml(:result => "happy"))) === happy
    @test_throws ArgumentError FunctAI.load(dir; types=(result=Int,))
    @ai function coded(x::String)::Int
        "A number."
        n::Int = ai"it"
        n + 1
    end
    @test_throws LoadRefused FunctAI.save(dir, coded)
end

@testset "a program: its AI calls are its children" begin
    @program function support(ticket::String)
        m = mood(ticket)
        m === unhappy ? "sorry" : "thanks"
    end
    dir = mktempdir()
    r = fake(xml(:result => "unhappy"))
    @test using_fake(() -> support("Broke."), r; log_calls=dir) == "sorry"
    recs, _ = FunctAI.read_log(dir)
    child = only(filter(c -> c["program"]["kind"] == "ai", recs))
    parent = only(filter(c -> c["program"]["kind"] == "module", recs))
    @test child["parent"] == parent["id"] && child["root"] == parent["id"]
    @test parent["outputs"] == Dict("result" => "sorry") && parent["inputs"] == Dict("ticket" => "Broke.")
    v = version(support)
    @test v != version(configure(mood; lm="x")) && startswith(v, "sha256:")
    # a stream of a program shows its children
    s = using_fake(() -> stream(support, "Broke."), fake(xml(:result => "unhappy")))
    @test fetch(s) == "sorry"
    @test count(e -> e.kind === :started, eachevent(s)) == 2
end

end

@testset "OneOf: what each answer means reaches the model as every language writes it (contract case 18)" begin
    want = LMCC.parse_json(read(joinpath(@__DIR__, "..", "..", "contract", "cases", "functions", "18-choices-described-one-by-one.json"), String))
    route = AIFunction("route", "Which team answers?";
        inputs=(message=String => "the customer's words",
                channel=OneOf("email" => "written at leisure, often long", "chat" => "short,\n   often several messages")),
        outputs=(urgency=OneOf("low", "high" => "the customer is losing money or time now") => "how soon to answer",
                 result=OneOf("shipping" => "where the parcel is, damage in transit", "billing" => " charges, refunds ")),
        lm="gpt-4.1-mini")
    @test version(route) == want["expect"]["version"]
    @test occursin("- channel: one of:\n  - email: written at leisure, often long\n  - chat: short, often several messages\n",
                   instructions(route))
    # the shape is the plain choice's; an empty meaning describes nothing; Symbols too
    @test FunctAI.shape_of(OneOf("a", "b" => "  ")) == FunctAI.shape_of(OneOf("a", "b"))
    @test OneOf("a", "b" => "  ").meanings == Pair{String,String}[]
    @test OneOf(:low, :high => "now").meanings == ["high" => "now"]
    @test_throws ArgumentError OneOf("a", :b)
    @test_throws ArgumentError OneOf("a", "a" => "twice")
    @test sprint(show, OneOf("low", "high" => "now")) == "OneOf(\"low\", \"high\" => \"now\")"
end

@testset "on_unreadable = :record keeps an unreadable reply: missing, the refusal said and recorded, a conversation goes on" begin
    dir = mktempdir()
    r = fake("no tags at all", (text = "", finish = "content_filter"))
    p = using_fake(() -> predict(mood, "Where is it?"), r; retries=0, on_unreadable=:record, log_calls=dir)
    @test p.answer === missing && p.value === missing && isempty(p.outputs)
    @test p.refusal isa LMCC.Refusal && p.refusal.code == "parse-missing-fields"
    @test p.turn.meta["refusal"]["code"] == "parse-missing-fields"
    @test using_fake(() -> mood("x"), r; retries=0, on_unreadable=:record, log_calls=dir) === missing   # a stopped reply too
    recs = first(FunctAI.read_log(dir))
    @test length(recs) == 2
    @test all(rec -> isempty(rec["outputs"]) && rec["error"]["type"] == "Refusal" && !haskey(rec, "returned"), recs)
    @test Set(rec["error"]["code"] for rec in recs) == Set(["parse-missing-fields", "parse-filtered"])
    # a conversation's turn is done with no outputs, and the next one goes on
    @ai function tutor(message::String)::String
        "Tutor."
    end
    r = fake("nope", xml(:result => "fine"))
    chat = conversation(tutor, "kept")
    @test using_fake(() -> chat("first"), r; retries=0, on_unreadable=:record) === missing
    @test using_fake(() -> chat("second"), r; retries=0, on_unreadable=:record) == "fine"
    @test [t.state for t in turns(chat)] == ["done", "done"]
    # the default still raises; a setting that is neither is refused
    @test_throws LMCC.Refusal using_fake(() -> mood("x"), fake("nope"); retries=0)
    @test_throws ArgumentError configure(mood; on_unreadable=:keep)
end

@testset "api_key and base_url: the function's own key and address; the address is saved, the key never" begin
    seen = Tuple{String,String}[]
    server = FunctAI.HTTP.serve!("127.0.0.1", 0; listenany=true) do req
        push!(seen, (req.target, FunctAI.HTTP.header(req, "Authorization")))
        FunctAI.HTTP.Response(400, ["Content-Type" => "application/json"], "{\"error\":{\"message\":\"seen\"}}")
    end
    try
        url = "http://127.0.0.1:$(FunctAI.HTTP.port(server))/v1"
        f = configure(mood; lm="openai:qwen3", api_key="sk-local", base_url=url, api_retries=0)
        @test_throws Exception f("x")
        @test length(seen) == 1
        @test startswith(seen[1][1], "/v1/")                  # the request went to the function's own address
        @test seen[1][2] == "Bearer sk-local"                 # with its own key
        folder = mktempdir()
        FunctAI.save(folder, f)
        saved = read(joinpath(folder, "functai.json"), String)
        @test occursin(url, saved) && !occursin("sk-local", saved)
        @test FunctAI.load(folder).own[:base_url] == url
    finally
        close(server)
    end
    @test_throws ArgumentError configure(mood; base_url=3)
end
