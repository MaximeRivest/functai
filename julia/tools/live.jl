# FunctAI for Julia against real models (costs cents). Not run by Pkg.test().
#
#     set -a; source ~/Projects/lm15-dev/.env; set +a
#     julia --project=julia julia/tools/live.jl [model ...]
using FunctAI

models = isempty(ARGS) ? ["gpt-4.1-mini", "claude-haiku-4-5", "gemini:gemini-2.5-flash"] : ARGS
failed = 0
function check(f, what)
    t0 = time()
    try
        out = f()
        println("  ok    $what ($(round(time() - t0; digits=1)) s): ", first(repr(out), 140))
    catch err
        global failed += 1
        println("  FAIL  $what: ", first(sprint(showerror, err), 500))
    end
end

@enum Mood happy unhappy mixed
struct Person
    name::String
    age::Int
end
"Look up where an order is."
lookup_order(order::String) = order == "A-1042" ? "stuck at the carrier since Monday" : "unknown order"

for model in models
    println(model)
    with_settings(lm=model) do
        @ai temperature = 0 function mood(review::String)::Mood
            "How does the customer feel about what they bought?"
        end
        check("a choice (@enum)") do
            mood("It broke after one day and support never answered.")
        end
        check("a struct, json layout") do
            @ai adapter = :json function person(text::String)::Person
                "Who is described?"
            end
            person("Ana turned 31 last week.")
        end
        check("several outputs, reasoning first") do
            @ai reasoning = true max_tokens = 4000 function solve(problem::String)
                "Solve the word problem."
                unit_price::Float64 = ai"the price of one item"
                total::Float64 = ai"the answer"
            end
            solve("A pen costs 3 dollars. How much do 7 pens cost?")
        end
        check("a tool (a Julia function)") do
            @ai tools = [lookup_order] function support(message::String)::String
                "Answer the customer, looking up their order."
            end
            support("Where is my order A-1042?")
        end
        check("streaming") do
            @ai function haiku(topic::String)::String
                "A haiku about the topic."
            end
            s = stream(haiku, "the first snow")
            text = join(collect(s))
            value = fetch(s)
            strip(text) == strip(value) || error("pieces $(repr(text)) != value $(repr(value))")
            (pieces=count(e -> e.kind === :text, events(s)), value=value)
        end
        check("a column, concurrently") do
            mood.(["Love it, works perfectly.", "Broke in a day.", missing, "Good product but arrived late and damaged."])
        end
        check("evaluate") do
            e = evaluate(mood, [(review="Love it, works perfectly.", result="happy"), (review="Broke in a day.", result="unhappy"),
                                (review="Good product but the box arrived damaged and late.", result="mixed")])
            sprint(show, e)
        end
    end
end
exit(failed == 0 ? 0 : 1)
