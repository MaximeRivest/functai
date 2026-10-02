# Stage 1's foundations, as a Julia user meets them (design/08): interfaces
# (optional inputs of an AI function, @program's declared interface, checked
# on every call), the call tree's log (format 2 events, one numbering for a
# tree, a stream opened inside a program), observers and journals end to end,
# saving and describing an interface, and reading what a call saw. A fake
# router; no network.

"A store whose appends of some kinds get no answer, or are refused (a journal that fails)."
struct FailingStore
    store::FunctAI.MemoryStore
    unanswered::Vector{Symbol}
    refused::Vector{Symbol}
end
FailingStore(; unanswered=Symbol[], refused=Symbol[]) = FailingStore(FunctAI.MemoryStore(), unanswered, refused)
struct NoAnswer <: Exception end
function FunctAI.keep!(s::FailingStore, e::FunctAI.Event)
    e.kind in s.unanswered && throw(NoAnswer())
    e.kind in s.refused && throw(FunctAI.StoreRefusal("event-conflict", FunctAI.Position(e), "refused"))
    FunctAI.keep!(s.store, e)
end
FunctAI.events_after(s::FailingStore, tree, after) = FunctAI.events_after(s.store, tree, after)

@testset "the foundations" begin

@testset "an AI function's input with a default may be left out: the interface says so, the call sends it" begin
    @ai function reply(message::String; tone::String = "kind")::String
        "Answer the customer."
    end
    iface = FunctAI.interface(reply)
    @test iface["description"] == "Answer the customer."
    @test iface["inputs"][2] == Dict("name" => "tone", "shape" => Dict("type" => "string", "default" => "kind"),
                                     "type" => "String", "optional" => true)
    @test FunctAI.interface_signature(reply) == signature_id(reply)          # the default is not in the signature
    left_out = fake(xml(:result => "Hello."))
    given = fake(xml(:result => "Hello."))
    using_fake(() -> reply("Hi"), left_out)
    using_fake(() -> reply("Hi"; tone="kind"), given)
    @test LM15.to_dict(left_out.requests[1]) == LM15.to_dict(given.requests[1])
    # a default is data, known when the function is defined; one that does not fit its type is refused
    err = try
        @eval @ai function count_items(items::Vector{String}; at_least::Int = "ten")::Int
            "Count them."
        end
    catch e
        e
    end
    @test err isa InterfaceError && err.code == "interface-malformed" && err.field == "at_least"
    @test_throws InterfaceError AIFunction("f"; inputs=(x=Dict("type" => "string", "minLength" => "3"),), output=String)
    # the data constructor: defaults
    g = AIFunction("reply", "Answer."; inputs=(message=String, tone=String), defaults=(tone="kind",), output=String)
    @test FunctAI.interface(g)["inputs"][2]["optional"] == true
    bound = FunctAI.bind_inputs(g, ("Hi",), ())          # a default left out is sent as its JSON
    @test Dict(k => FunctAI.jsonvalue(v) for (k, v) in bound) == Dict("message" => "Hi", "tone" => "kind")
end

@testset "@program declares its interface, and every call is checked against it" begin
    @program function answer(message::String; tone::String = "kind", since = time())::String
        "Answer a customer's message."
        string(message, "/", tone)
    end
    iface = FunctAI.interface(answer)
    @test iface["description"] == "Answer a customer's message."
    @test [f["name"] for f in iface["inputs"]] == ["message", "tone", "since"]
    @test iface["inputs"][2]["shape"] == Dict("type" => "string", "default" => "kind")
    @test iface["inputs"][3] == Dict("name" => "since", "shape" => Dict(), "opaque" => true, "optional" => true)   # untyped: opaque; its default is Julia code
    @test iface["outputs"] == [Dict("name" => "result", "shape" => Dict("type" => "string"), "type" => "String")]
    @test answer("Hi") == "Hi/kind"
    @test answer(3) == "3/kind"                          # bound: a number to text is its text (programs.md)
    err = try answer(nothing) catch e e end
    @test err isa InterfaceError && err.code == "interface-input" && err.field == "message"
    @program function wrong(x::String)::Int
        "Not a number."
        x
    end
    err = try wrong("a") catch e e end
    @test err isa InterfaceError && err.code == "interface-output" && err.field == "result"
    # several outputs, declared; a key it does not declare is refused
    @program outputs = (team = String, minutes = Int) function sort_ticket(ticket::String)
        (team = "billing", minutes = 5.0)
    end
    @test sort_ticket("x") === (team = "billing", minutes = 5)           # 5.0 is an integer (programs.md), returned as the Int declared
    @program outputs = (team = String, minutes = Int) function chatty(ticket::String)
        (team = "billing", minutes = 5, note = "extra")
    end
    err = try chatty("x") catch e e end
    @test err isa InterfaceError && err.code == "interface-output" && err.field == "note"
    # the interface is in the version
    @test version(answer) != version(wrong)
    # a program from an interface as data
    p = AIProgram("echo", (; text) -> uppercase(text);
                  interface=Dict("description" => "", "inputs" => [Dict("name" => "text", "shape" => Dict("type" => "string"))],
                                 "outputs" => [Dict("name" => "result", "shape" => Dict("type" => "string"))]))
    @test p("hi") == "HI" && p(; text="yo") == "YO"
    @test_throws InterfaceError p(; txt="yo")
    # refused, and logged: only what the interface names, and its kept form never fails on the name it lacks
    dir = mktempdir()
    seen = FunctAI.Event[]
    with_settings(log_calls=dir, log_content=Dict("*" => false), observers=[e -> push!(seen, e)]) do
        @test_throws InterfaceError p(; text="a", secret="b")
    end
    @test FunctAI.drain()
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["error"]["code"] == "interface-input" && !haskey(rec, "inputs") && rec["sizes"]["inputs"] == Dict("text" => 3)
    @test [e.kind for e in seen] == [:started, :failed]
end

@testset "a stream's events are the tree's log: format 2, one numbering, requests" begin
    r = fake(xml(:result => "unhappy"); piece=4)
    s = using_fake(() -> stream(mood, "Broke."), r)
    @test fetch(s) === unhappy
    evs = collect(eachevent(s))
    @test [e.seq for e in evs] == 1:length(evs)
    @test all(e -> e.writer == 1 && e.tree == evs[1].call, evs)
    @test evs[1].after === nothing && all(i -> evs[i].after == FunctAI.Position(evs[i-1]), 2:length(evs))
    @test [e.kind for e in evs][1:2] == [:started, :request] && evs[2].request == 1 && evs[end].kind === :done
    @test all(e -> FunctAI.event_fault(FunctAI.event_json(e)) === nothing, evs)
    @test evs[1].saw == [] && evs[1].program["interface"] == FunctAI.interface_signature(mood)
    @test issorted([e.at for e in evs])
    # a retry empties the answer so far, and the request after it too
    s = using_fake(() -> stream(mood, "x"), fake("??", xml(:result => "happy")))
    fetch(s)
    kinds = [e.kind for e in eachevent(s)]
    @test count(==(:request), kinds) == 2 && :retry in kinds
    @test FunctAI.replay(collect(eachevent(s))).finished
    # a follower of the whole log takes each event as the next
    follower = FunctAI.Follower(:live)
    @test all(==(:kept), [FunctAI.receive!(follower, e) for e in eachevent(s)])
end

@testset "a stream opened on a call inside a program shows the tree's numbers (law 7)" begin
    inner = Ref{Any}(nothing)
    @program function watcher(t::String)
        inner_stream = stream(mood, t)
        v = fetch(inner_stream)
        inner[] = collect(eachevent(inner_stream))
        string(v)
    end
    s = using_fake(() -> stream(watcher, "Broke."), fake(xml(:result => "unhappy")))
    @test fetch(s) == "unhappy"
    outer = collect(eachevent(s))
    @test length(outer) > length(inner[])
    mine = inner[]
    @test mine[1].after === nothing
    @test mine[1].seq == 2
    @test mine[1].parent == outer[1].call
    @test [FunctAI.Position(e) for e in mine] == [FunctAI.Position(e) for e in outer if e.call == mine[1].call]
    @test all(i -> mine[i].after == FunctAI.Position(mine[i-1]), 2:length(mine))
end

@testset "observers get the kept form, and add up over the layers" begin
    seen, host = FunctAI.Event[], FunctAI.Event[]
    f = configure(mood; observers=[e -> push!(seen, e)], log_content=false)
    with_settings(observers=[e -> push!(host, e)]) do
        using_fake(() -> f("secret review"), fake(xml(:result => "happy")))
    end
    @test FunctAI.drain()                                                      # observers get events on tasks of their own
    @test [e.kind for e in seen] == [e.kind for e in host] == [:started, :request, :done]
    @test seen[1].content == false && !haskey(seen[1], "inputs") && seen[1].omitted["inputs"] == ["review"]
    @test seen[end].content == false && !haskey(seen[end], "value")
    @test seen[2].after == FunctAI.Position(seen[1])                           # each observer's own chain
    # a failing observer is given no more events, with one warning; the call goes on
    boom = e -> error("down")
    @test (@test_logs (:warn, r"observer .* failed") match_mode = :any begin
        v = using_fake(() -> mood("x"), fake(xml(:result => "happy")); observers=[boom])
        FunctAI.drain()
        v
    end) === happy
end

@testset "a required journal: barriers, and an end it does not confirm" begin
    store = FunctAI.MemoryStore()
    p = using_fake(() -> predict(mood, "Broke."), fake(xml(:result => "unhappy")); journal=FunctAI.Journal(store; required=true))
    @test FunctAI.finished(store, p.call)
    log = FunctAI.events_after(store, p.call, nothing)
    @test [e.kind for e in log] == [:started, :request, :text, :text, :text, :done]      # "unhappy" came in three pieces
    # the end refused: the outcome stands, the caller is told, the record says so, no reader sees the end
    dir = mktempdir()
    refusing = FailingStore(refused=[:done])
    s = using_fake(() -> stream(mood, "Broke."), fake(xml(:result => "unhappy"));
                   journal=FunctAI.Journal(refusing; required=true), log_calls=dir)
    err = try fetch(s) catch e e end
    @test err isa JournalError && err.code == "journal-end" && err.journal == "refused" && err.outcome.done === unhappy
    @test !any(e -> e.kind === :done, eachevent(s))
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["journal"] == "refused" && rec["outputs"] == Dict("result" => "unhappy") && rec["error"] === nothing
    # no answer to the end: settled by reading the journal for its event
    silent = FailingStore(unanswered=[:done])
    err = try using_fake(() -> mood("x"), fake(xml(:result => "happy")); journal=FunctAI.Journal(silent; required=true)) catch e e end
    @test err isa JournalError && err.journal == "unknown" && FunctAI.settle(silent, err.tree, err.event) === :not_kept
    # a tool call the journal does not keep: the tool does not run
    ran = Ref(false)
    "Look up an order."
    look(order::String) = (ran[] = true; "shipped")
    @ai tools = [look] function helper(q::String)::String
        "Help."
    end
    stuck = FailingStore(unanswered=[:tool_call])
    err = try
        using_fake(() -> helper("Where is A1?"), fake((calls=[("c1", "look", (order="A1",))],), xml(:result => "It shipped."));
                   journal=FunctAI.Journal(stuck; required=true, retries=0))
    catch e
        e
    end
    @test !ran[]
    @test err isa JournalError && err.code == "journal-end" && err.outcome.failed isa JournalError && err.outcome.failed.code == "journal-barrier"
    # a required journal set only inside a tree refuses the call inside
    @program function inside(t::String)
        with_settings(() -> mood(t); journal=FunctAI.Journal(FunctAI.MemoryStore(); required=true))
    end
    err = try using_fake(() -> inside("x"), fake(xml(:result => "happy"))) catch e e end
    @test err isa JournalError && err.code == "journal-scope"
end

@testset "saving writes the interface; describing reads it without loading; loading keeps optional inputs" begin
    @ai function greet(name::String; tone::String = "warm")::String
        "Greet someone."
    end
    dir = mktempdir()
    FunctAI.save(dir, greet)
    @test FunctAI.describe(dir) == FunctAI.interface(greet)
    loaded = FunctAI.load(dir)
    @test FunctAI.interface(loaded) == FunctAI.interface(greet) && version(loaded) == version(greet)
    a, b = fake(xml(:result => "Hi")), fake(xml(:result => "Hi"))
    using_fake(() -> loaded("Ana"), a)
    using_fake(() -> greet("Ana"), b)
    @test LM15.to_dict(a.requests[1]) == LM15.to_dict(b.requests[1])
end

@testset "what a call saw: Julia's calls see no earlier call" begin
    dir = mktempdir()
    p = using_fake(() -> predict(mood, "x"), fake(xml(:result => "happy")); log_calls=dir)
    records = first(FunctAI.read_log(dir))
    @test FunctAI.saw(records, p.call) == []
    @test FunctAI.keeps_saw(records, p.call) === nothing
end

end
