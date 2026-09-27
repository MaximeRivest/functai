# GEPA, offline: a fake model answers by rules, and a fake teacher writes instructions.

@testset "gepa" begin

GOOD = "Use the labels exactly: booking, cancelation, information."
intent_rows = [(query=q, intent=i) for (q, i) in [
    ("I need to reserve a room.", "booking"), ("How do I get there?", "information"), ("Cancel my reservation.", "cancelation"),
    ("Book me a suite.", "booking"), ("Please book a table for two.", "booking"), ("What time is breakfast?", "information"),
    ("I want to cancel tonight.", "cancelation"), ("Is there parking?", "information")]]

last_text(req) = join((p.text for p in req.messages[end].parts if p isa LM15.TextPart), "")
query_of(req) = (m = match(r"<query>\n(.*?)\n</query>"s, last_text(req)); m === nothing ? "" : m[1])
reflecting(req) = occursin("You improve the instruction", something(req.system, ""))
combining(req) = occursin("Two instructions for the same function", something(req.system, ""))
label_of(q) = occursin(r"reserve|book"i, q) ? "booking" : occursin(r"cancel"i, q) ? "cancelation" : "information"
classifier() = AIFunction("intent", "Classify the user's intent."; inputs=(query=String,),
                          output=OneOf("booking", "cancelation", "information"))

@testset "it rewrites the instruction from its mistakes, never showing the choosing rows" begin
    r = FakeRouter(; responder=(req, i) -> reflecting(req) || combining(req) ? xml(:result => GOOD) :
        xml(:result => occursin("Use the labels exactly", something(req.system, "")) ? label_of(query_of(req)) : "information"))
    dir = mktempdir()
    f = classifier()
    better, trials = using_fake(() -> gepa(f, intent_rows; expected=:intent, budget=60, seed=1), r; log_calls=dir)
    @test FunctAI.instructions(better) == GOOD
    @test version(better) != version(f)
    shown = join((last_text(q) for q in r.requests if reflecting(q)), "\n")
    @test occursin("wrong: the right answer is", shown)
    @test occursin("- result: one of booking, cancelation, information", shown)
    # the choosing rows: the first half after the seeded shuffle, never shown to the teacher
    for row in shuffle(Xoshiro(1), [FunctAI.row_dict(x) for x in intent_rows])[1:4]
        @test !occursin(row["query"], shown)
    end
    chosen = filter(t -> t.chosen, trials)
    @test length(chosen) == 1 && only(chosen).kind === :reflect && only(chosen).score == 1.0
    @test last(trials).calls <= 60
    @test first(trials).candidate == 1 && first(trials).kind === :written
    recs, _ = FunctAI.read_log(dir)
    @test all(c -> haskey(c["caller"], "optimization"), recs)               # every call is part of the optimization
    @test any(c -> c["program"]["name"] == "_reflect" && c["program"]["module"] == "functai.meta", recs)
    @test any(c -> haskey(c["caller"], "evaluation"), recs)                  # and evaluations inside it are both
end

@testset "a row runs once per instruction; the written one is kept when nothing beats it" begin
    seen = String[]
    r = FakeRouter(; responder=(req, i) -> reflecting(req) ? xml(:result => "Still vague.") :
        (push!(seen, "$(req.system)|$(query_of(req))"); xml(:result => "information")))
    f = classifier()
    kept, trials = using_fake(() -> gepa(f, intent_rows; expected=:intent, budget=40), r)
    @test allunique(seen)
    @test length(seen) == last(trials).calls
    @test kept === f
end

@testset "a proposal that copies an input is dropped, and the next reflection is told" begin
    long = vcat([(query="Hello there, I would like to know about option number $i please.", intent="information") for i in 1:6],
                [(query="Please book the room number $i for the whole of next week.", intent="booking") for i in 1:6])
    said = String[]
    r = FakeRouter(; responder=(req, i) -> begin
        reflecting(req) || return xml(:result => "information")
        push!(said, last_text(req))
        q = match(r"\n  query: ([^\n]*)", last_text(req))[1]
        xml(:result => "If the message says '$q', answer booking.")
    end)
    f = classifier()
    kept, trials = using_fake(() -> gepa(f, long; expected=:intent, budget=40), r)
    @test any(t -> t.note == "copied an input: dropped", trials)
    @test any(s -> occursin("dropped: it copied an input", s), said[2:end])
    @test FunctAI.instructions(kept) == FunctAI.instructions(f)
end

@testset "the frontier keeps candidates best somewhere, drops the dominated; pairs win different rows" begin
    scores = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    @test collect(FunctAI.frontier(scores)) == [3 => 2]
    @test FunctAI.best_pair(scores[1:2], FunctAI.frontier(scores[1:2])) == (1, 2)
    @test FunctAI.best_pair(scores, FunctAI.frontier(scores)) === nothing
end

@testset "fields are described for the teacher in words (the same words as Python, R and TypeScript)" begin
    f = AIFunction("team", "Which team?"; inputs=(message=String => "the customer's words", n=Union{Int,Nothing}, tags=Vector{String}),
                   output=OneOf("a", "b"))
    @test FunctAI.fields_text(f) ==
          "Inputs:\n- message: text. the customer's words\n- n: a whole number, or nothing\n- tags: a list of text\nOutputs:\n- result: one of a, b"
end

@testset "a custom metric and feedback" begin
    words = String[]
    r = FakeRouter(; responder=(req, i) -> reflecting(req) ? (push!(words, last_text(req)); xml(:result => GOOD)) :
        xml(:result => occursin("Use the labels exactly", something(req.system, "")) ? label_of(query_of(req)) : "information"))
    better, _ = using_fake(() -> gepa(classifier(), intent_rows; budget=40, seed=2,
                                      metric=(row, out) -> out.result == row.intent,
                                      feedback=(row, out, err) -> out === nothing ? "failed" : "the label is $(row.intent)"), r)
    @test FunctAI.instructions(better) == GOOD
    @test occursin("feedback: the label is", first(words))
end

end
