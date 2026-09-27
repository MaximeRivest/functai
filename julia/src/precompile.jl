# Compiled when the package is installed, so the first call in a session is
# not a minute of compiling: a typical function is defined, versioned,
# rendered, called (on a column too), streamed and evaluated against a
# stand-in model. Nothing is sent and nothing is written.

"A stand-in model for the precompile workload: answers every request from the plan's own layout."
struct PrecompileRouter end
route(::PrecompileRouter, model::AbstractString) = ("openai", String(model))
function LM15.complete(::PrecompileRouter, request::LM15.Request)
    text = occursin("{", something(request.system, "")) && request.config.response_format !== nothing ?
           "{\"result\": \"happy\"}" : "<result>\nhappy\n</result>"
    LM15.Response(; model=request.model, finish_reason="stop", usage=LM15.Usage(; input_tokens=1, output_tokens=1),
                  message=LM15.Message(; role="assistant", parts=(LM15.TextPart(; text),)))
end
LM15.stream(r::PrecompileRouter, request::LM15.Request) = collect(LM15.response_to_events(LM15.complete(r, request)))

@setup_workload begin
    @compile_workload begin
        f = AIFunction("mood", "How does the customer feel?"; inputs=(review=String,), output=OneOf("happy", "unhappy", "mixed"))
        version(f)
        signature_id(f)
        with_settings(; lm="gpt-4.1-mini", router=PrecompileRouter(), log_calls=false) do
            f("It broke.")
            f.(["It broke.", "Love it."])
            predict(f, "x")
            render(f, "x")
            fetch(LM15.stream(f, "x"))
            evaluate(f, [(review="x", result="happy")])
            with_demos(f, ["It broke." => "unhappy"])("x")
        end
        define_ai(@__MODULE__, LineNumberNode(1, :precompile), (),
                  :(function mood(review::String)::String
                        "How does the customer feel?"
                    end), nothing)
        for model in ("openai:gpt-4.1-mini", "anthropic:claude-haiku-4-5", "gemini:gemini-2.5-flash")
            # the provider's wire request, built offline (lm15's plan sends nothing)
            try
                LM15.plan(LM15.LMRouter(), LM15.Request(model, LM15.user("x"); config=LM15.Config(; temperature=0.0)))
            catch
            end
        end
    end
end
