# A reasoning output on a model that thinks natively (contract/functions.md, *Reasoning*): the request asks for
# the thinking and its summary, and the function's own reasoning setting (config) chooses the effort.

@ai reasoning = true function solve_native(puzzle::String)::Int
    "Solve the puzzle."
end

@testset "native reasoning: the thinking and its summary; the function's own effort" begin
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-fable-5-1", "claude-mythos-5", "claude-opus-4-8")
        caps = model_capabilities("anthropic", model)
        @test caps.native_reasoning && !caps.assistant_prefill
    end
    thinker() = FakeRouter(; provider="anthropic", responder=(req, i) -> LM15.Response(; model=req.model,
        message=LM15.Message(; role="assistant", parts=(LM15.ThinkingPart(; text="Twelve pens leave 14 for two notebooks."),
                                                         LM15.TextPart(; text="<result>\n12\n</result>"))),
        finish_reason="stop", usage=LM15.Usage(; input_tokens=10, output_tokens=5)))
    sent(r) = LM15.to_dict(r.requests[end].config.reasoning)
    r = thinker()
    p = with_settings(() -> predict(solve_native, "x"); lm="anthropic:claude-opus-5-5", router=r)
    @test p.outputs.reasoning == "Twelve pens leave 14 for two notebooks."
    @test sent(r) == Dict("effort" => "medium", "summary" => "auto")
    for (mine, wanted) in ((LM15.Reasoning(; effort="high"), Dict("effort" => "high", "summary" => "auto")),
                           (LM15.Reasoning(; effort="low", summary="detailed"), Dict("effort" => "low", "summary" => "detailed")),
                           (LM15.Reasoning(; effort="off"), Dict("effort" => "off")))
        with_settings(() -> solve_native("x"); lm="anthropic:claude-opus-5-5", router=r, config=(reasoning=mine,))
        @test sent(r) == wanted
    end
end
