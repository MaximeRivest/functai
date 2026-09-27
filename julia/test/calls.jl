# The call log (contract/calls.md), written and read by Julia.

@testset "the call log" begin

@testset "a call is one line: its program, values, exchanges, process" begin
    dir = mktempdir()
    r = fake(xml(:result => "unhappy"))
    p = using_fake(() -> predict(mood, "Broke."), r; log_calls=dir, temperature=0)
    recs, ratings = FunctAI.read_log(dir)
    rec = only(recs)
    @test isempty(ratings)
    @test rec["functai_call"] == 1 && rec["id"] == p.call && rec["parent"] === nothing && rec["root"] == p.call
    @test occursin(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", p.call)
    prog = rec["program"]
    @test prog["name"] == "mood" && prog["kind"] == "ai" && prog["module"] == "__main__"
    @test prog["version"] == version(mood) && prog["signature"] == signature_id(mood) && prog["answer"] == "result"
    @test occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$", rec["started"])
    @test rec["inputs"] == Dict("review" => "Broke.") && rec["outputs"] == Dict("result" => "unhappy")
    @test !haskey(rec, "returned")
    @test rec["sizes"]["inputs"]["review"] == length("\"Broke.\"")
    @test rec["model"] == "gpt-4.1-mini" && rec["usage"]["input_tokens"] == 10
    ex = only(rec["exchanges"])
    @test ex["provider"] == "openai" && ex["cached"] == false && ex["finish"] == "stop"
    @test ex["request"]["config"]["temperature"] == 0 && haskey(ex, "response")
    @test rec["process"]["language"] == "julia" && rec["process"]["pid"] == getpid()
    day = only(readdir(dir))
    file = only(readdir(joinpath(dir, day)))
    @test endswith(file, ".jsonl")
    Sys.isunix() && (@test filemode(joinpath(dir, day, file)) & 0o777 == 0o600)
    Sys.isunix() && (@test filemode(joinpath(dir, day)) & 0o777 == 0o700)
end

@testset "log_content = false: sizes, times and tokens, no values" begin
    dir = mktempdir()
    using_fake(() -> mood("secret"), fake(xml(:result => "happy")); log_calls=dir, log_content=false)
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["content"] == false
    @test !haskey(rec, "inputs") && !haskey(rec, "outputs")
    @test rec["sizes"]["inputs"]["review"] == 8
    @test !haskey(only(rec["exchanges"]), "request")
end

@testset "a failed call has its error; a re-ask is a second exchange" begin
    dir = mktempdir()
    @test_throws LMCC.Refusal using_fake(() -> mood("x"), fake("??", "still ??"); log_calls=dir)
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["error"]["type"] == "Refusal" && startswith(rec["error"]["code"], "parse-")
    @test length(rec["exchanges"]) == 2 && rec["outputs"] === nothing
    dir = mktempdir()
    @test_throws LM15.AuthError using_fake(() -> mood("x"), fake(LM15.AuthError("no key")); log_calls=dir)
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["error"]["type"] == "AuthError" && only(rec["exchanges"])["error"]["type"] == "AuthError"
end

@testset "code of its own: what it returned" begin
    @ai function doubled(x::Int)::Int
        "Double it."
        y::Int = ai"twice x"
        y + 1
    end
    dir = mktempdir()
    @test using_fake(() -> doubled(2), fake(xml(:y => "4")); log_calls=dir) == 5
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["outputs"] == Dict("y" => 4) && rec["returned"] == 5
end

@testset "a streamed request says so" begin
    dir = mktempdir()
    s = using_fake(() -> stream(mood, "x"), fake(xml(:result => "happy")); log_calls=dir)
    fetch(s)
    ex = only(only(first(FunctAI.read_log(dir)))["exchanges"])
    @test ex["streamed"] == true && ex["first_delta"] isa Real
end

@testset "who called: \$FUNCTAI_CALLER, the caller setting, evaluate's run" begin
    dir = mktempdir()
    withenv("FUNCTAI_CALLER" => "{\"kind\": \"notebook\", \"notebook\": \"/x.md\"}") do
        using_fake(() -> mood("x"), fake(xml(:result => "happy")); log_calls=dir, caller=Dict("cell" => 3))
    end
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["caller"] == Dict("kind" => "notebook", "notebook" => "/x.md", "cell" => 3)
    dir = mktempdir()
    e = using_fake(() -> evaluate(mood, [(review="x", result="happy")]), fake(xml(:result => "happy")); log_calls=dir)
    @test only(first(FunctAI.read_log(dir)))["caller"]["evaluation"] == e.run
end

@testset "FUNCTAI_LOG_CALLS turns it on; log_calls = false wins" begin
    dir = mktempdir()
    withenv("FUNCTAI_LOG_CALLS" => dir) do
        using_fake(() -> mood("x"), fake(xml(:result => "happy")))
        using_fake(() -> mood("x"), fake(xml(:result => "happy")); log_calls=false)
    end
    @test length(first(FunctAI.read_log(dir))) == 1
    withenv("FUNCTAI_LOG_CALLS" => "off") do
        @test FunctAI.folder_of(nothing) === nothing
        @test FunctAI.folder_of(true) == FunctAI.default_log_folder()
    end
end

@testset "rate, calls, rated: ratings become rows with known answers" begin
    dir = mktempdir()
    r = FakeRouter(; responder=(req, i) -> xml(:result => "unhappy"))
    p1, p2, p3 = using_fake(() -> [predict(mood, t) for t in ("Broke.", "Late, but fine.", "Love it")], r; log_calls=dir)
    rate(p1, :right; by="ana", folder=dir)
    rate(p2; answer=mixed, by="ana", note="late but fine", folder=dir)
    rate(p3, :wrong; by="ben", folder=dir)                     # wrong, no correction: left out
    @test_throws ArgumentError rate(p1, :right; answer=happy, folder=dir)
    @test_throws ArgumentError rate(p1, :maybe; folder=dir)
    rows = @test_logs (:info, r"1 rated call") match_mode = :any using_fake(() -> rated(mood), r; log_calls=dir)
    @test length(rows) == 2
    @test rows[1].review == "Broke." && rows[1].result === unhappy && rows[1].rating == "right" && rows[1].rated_by == "ana"
    @test rows[2].result === mixed && rows[2].rating == "wrong" && rows[2].disputed == false
    # rated rows are evaluation data
    e = using_fake(() -> evaluate(mood, rows), r)
    @test e.score == 0.5
    # ana withdraws her rating
    rate(p1, nothing; by="ana", folder=dir)
    @test length(using_fake(() -> rated(mood), r; log_calls=dir)) == 1
    cs = using_fake(() -> calls(mood), r; log_calls=dir)
    @test length(cs) == 3 && cs[1].name == "mood" && cs[1].language == "julia" && cs[1].started isa FunctAI.Dates.DateTime
    @test length(calls(mood; folder=dir)) == 3
end

end
