# The model face: fit/predict, formulas (StatsModels, GLM), DataFrames'
# ByRow, and MLJ. A fake router; no network.

using DataFrames, CategoricalArrays, StatsModels, GLM
import MLJBase

@testset "AI functions as models" begin

tickets = DataFrame(message=["I was charged twice.", "My parcel never came.", "The kettle broke.", "Refund please.",
                             "Box arrived crushed.", "Screen flickers."],
                    team=categorical(["billing", "shipping", "product", "billing", "shipping", "product"]))
by_word(req) = (t = req.messages[end].parts[1].text;
                xml(:team => occursin(r"charg|Refund", t) ? "billing" : occursin(r"parcel|Box", t) ? "shipping" : "product"))

@testset "fit with a formula calls nothing; predict answers each row" begin
    r = FakeRouter(; responder=(req, i) -> by_word(req))
    m = using_fake(() -> fit(AIModel("Which team handles this ticket?"; examples=3), @formula(team ~ message), tickets), r)
    @test isempty(r.requests)
    @test m isa AIModelFit && length(m.fn.demos) == 3
    @test FunctAI.signature(m.fn).fields[end].shape == Dict("enum" => ["billing", "product", "shipping"], "type" => "string")
    @test startswith(FunctAI.instructions(m.fn), "Function: team\n\nWhich team handles this ticket?")
    @test StatsModels.formula(m) == @formula(team ~ message)
    got = using_fake(() -> predict(m, tickets), r)
    @test got isa CategoricalArray && levels(got) == levels(tickets.team)
    @test got == tickets.team
    @test using_fake(() -> evaluate(m, tickets), r).score == 1.0
    @test_throws ArgumentError fit(AIModel(), @formula(team ~ log(message)), tickets)
end

@testset "a text outcome is text; Symbols are a choice; levels choose" begin
    df = DataFrame(q=["a", "b"], answer=["x", "y"])
    m = fit(AIModel(), @formula(answer ~ q), df)
    @test FunctAI.signature(m.fn).fields[end].shape == Dict("type" => "string")
    m = fit(AIModel(; levels=["x", "y", "z"]), @formula(answer ~ q), df)
    @test FunctAI.signature(m.fn).fields[end].shape["enum"] == ["x", "y", "z"]
    m = fit(AIModel(), (q=["a", "b"],), [:up, :down])
    @test FunctAI.signature(m.fn).fields[end].shape["enum"] == ["down", "up"]
end

@testset "an AI function inside a GLM formula runs concurrently" begin
    @ai function stars(description::String)::Float64
        "How many stars would a guest give this listing, from 1 to 5?"
    end
    homes = DataFrame(sqft=[50.0, 80, 120, 60, 95, 150, 70, 110],
                      description=["cramped", "nice", "lovely", "dark", "bright", "stunning", "ok", "great"])
    homes.price = 2 .* homes.sqft .+ 10 .* [1, 3, 5, 1, 4, 5, 2, 4]
    score = Dict("cramped" => 1, "nice" => 3, "lovely" => 5, "dark" => 1, "bright" => 4, "stunning" => 5, "ok" => 2, "great" => 4)
    inflight, most = Ref(0), Ref(0)
    r = FakeRouter(; responder=(req, i) -> begin
        inflight[] += 1; most[] = max(most[], inflight[]); sleep(0.02); inflight[] -= 1
        xml(:result => string(score[match(r"<description>\n(\w+)\n", req.messages[end].parts[1].text)[1]]))
    end)
    model = using_fake(() -> GLM.lm(@formula(price ~ sqft + stars(description)), homes), r)
    @test coef(model) ≈ [0.0, 2.0, 10.0] atol = 1e-6
    @test most[] > 1
end

@testset "DataFrames: ByRow and AsTable" begin
    @ai function triage2(ticket::String)
        "Read the ticket."
        summary::String = ai"one sentence"
        minutes::Int = ai"minutes to fix"
    end
    r = FakeRouter(; responder=(req, i) -> xml(:summary => "ok", :minutes => "5"))
    df = DataFrame(ticket=["a", "b", "c"])
    out = using_fake(() -> transform(df, :ticket => ByRow(triage2) => AsTable), r)
    @test out.summary == ["ok", "ok", "ok"] && out.minutes == [5, 5, 5]
    out = using_fake(() -> transform(df, :ticket => ByRow(mood) => :mood), FakeRouter(; responder=(req, i) -> xml(:result => "happy")))
    @test out.mood == [happy, happy, happy]
    @test using_fake(() -> evaluate(mood, DataFrame(review=["x"], result=["happy"])), FakeRouter(; responder=(req, i) -> xml(:result => "happy"))).score == 1.0
end

@testset "MLJ: a machine, predict, and examples to tune" begin
    X = select(tickets, :message)
    y = tickets.team
    r = FakeRouter(; responder=(req, i) -> by_word(req))
    mach = MLJBase.machine(AIModel("Which team handles this ticket?"; examples=2, name="team"), X, y)
    using_fake(() -> MLJBase.fit!(mach; verbosity=0), r)
    @test isempty(r.requests)
    ŷ = using_fake(() -> MLJBase.predict(mach, X), r)
    @test all(ŷ .== y) && levels(ŷ[1]) == levels(y)
    @test ŷ isa CategoricalVector && levels(ŷ) == levels(y)
    @test MLJBase.report(mach).examples == 2
    @test MLJBase.fitted_params(mach).fn isa AIFunction
    @test MLJBase.input_scitype(AIModel) == MLJBase.Table
    # MLJ's predict (another function than StatsAPI's) works on AI functions too
    r = FakeRouter(; responder=(req, i) -> xml(:result => "happy"))
    @test using_fake(() -> MLJBase.predict(mood, "x"), r).value === happy
    @test all(p -> p.value === happy, using_fake(() -> MLJBase.predict.(mood, ["x", "y"]), r))
end

end
