# The Julia half of ../../tools/crosslang.py (run by it, with its working
# folder): load what Python saved, write the same function here, log and
# rate into the same folder, and read everyone's calls and ratings back.
using FunctAI
import LMCC, LM15

work = ARGS[1]
python = LMCC.parse_json(read(joinpath(work, "python.json"), String))
say(msg) = println("  ok    ", msg)
check(cond, what) = cond || error("julia: $what")

# 1. what Python saved loads here and sends the same bytes
for (name, want) in python
    if name == "rounded"
        err = try
            FunctAI.load(joinpath(work, "saved", name))
            nothing
        catch e
            e
        end
        check(err isa LoadRefused && err.code == "saved-code", "$name: expected saved-code, got $(repr(err))")
        say("$name: refused in Julia (saved-code: it runs Python code of its own)")
        continue
    end
    f = FunctAI.load(joinpath(work, "saved", name))
    check(version(f) == want["version"], "$name version: $(version(f)) != $(want["version"])")
    check(signature_id(f) == want["signature"], "$name signature")
    inputs = (; (Symbol(k) => v for (k, v) in want["inputs"])...)
    mine = LMCC.canonical_json(LM15.to_dict(render(f; inputs...)))
    check(mine == LMCC.canonical_json(want["request"]), "$name request:\n$mine\n$(LMCC.canonical_json(want["request"]))")
    say("$name: saved in Python, loaded in Julia, same version and same request")
end

# 2. the same function, written here with Julia types
@enum Mood happy unhappy mixed
@ai module_name = "shop" lm = "gpt-4.1-mini" temperature = 0 function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end
check(version(mood) == python["mood"]["version"], "mood version")
check(signature_id(mood) == python["mood"]["signature"], "mood signature")
say("mood written in Julia (an @enum answer) has Python's version and signature")

# 3. one log: log and rate here, then read everyone's calls and ratings
struct Fixed end
FunctAI.route(::Fixed, model::AbstractString) = ("openai", String(model))
LM15.complete(::Fixed, request::LM15.Request) = LM15.Response(; model=request.model, finish_reason="stop",
    message=LM15.Message(; role="assistant", parts=(LM15.TextPart(; text="<result>\nhappy\n</result>"),)),
    usage=LM15.Usage(; input_tokens=10, output_tokens=5))
log = joinpath(work, "log")
p = with_settings(router=Fixed(), log_calls=log) do
    predict(mood, "Exactly what I hoped for.")
end
rate(p, :right; by="dana", folder=log)
calls_, ratings = FunctAI.read_log(log)
rows, _ = FunctAI.rated_rows(calls_, ratings; name="mood", module_name="shop", signature=signature_id(mood))
write(joinpath(work, "julia-rated.json"), LMCC.json_text(rows))

# 4. what Julia saves is a manifest every language reads (Python checks it against the schema)
FunctAI.save(joinpath(work, "julia-saved", "mood"), with_demos(mood, ["Broke in a day." => unhappy]))
