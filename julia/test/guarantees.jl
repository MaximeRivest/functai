# What stage 1 promises that the contract's cases, which check events and
# records as data, cannot see: what a receiver is given (never a value the
# log does not keep, in any field of the object), observers that never slow
# a call, a store that checks and copies what it keeps, a tree whose end is
# last, a closed stream, a program checked however it is called, defaults
# made anew, a re-ask's request and its hash kept together, old saved
# folders checked, tool calls recorded, and every event and record passing
# the contract's schemas. Each was a counterexample in a review of the first
# implementation (2026-09-28); each drives the library's own path. A fake
# router; no network.

"An object with no JSON form."
struct Opaque end
FunctAI.jsonvalue(::Opaque) = throw(FunctAI.NoJSON("an Opaque has no JSON form"))

datum_parent(e) = e.parent

"Everything an object holds, field by field (not only its JSON form)."
holds(x, text) = occursin(text, sprint(dump, x; context=:limit => false)) || occursin(text, repr(x))

"A store whose appends of one seq get no answer; the others are kept."
struct SilentAt <: FunctAI.EventStore
    store::FunctAI.MemoryStore
    seqs::Vector{Int}
end
FunctAI.keep!(s::SilentAt, e::FunctAI.Event) = e.seq in s.seqs ? error("the network is down") : FunctAI.keep!(s.store, e)
FunctAI.events_after(s::SilentAt, tree, after) = FunctAI.events_after(s.store, tree, after)

"A store whose own code is wrong: a fault, not an answer."
struct Broken <: FunctAI.EventStore end
FunctAI.keep!(::Broken, e::FunctAI.Event) = e.no_such_key

"A store that never answers."
struct Hanging <: FunctAI.EventStore
    gate::Channel{Nothing}
end
FunctAI.keep!(s::Hanging, e::FunctAI.Event) = (take!(s.gate); :kept)

"A best-effort store whose appends hold the thread they run on (a C library's blocking write)."
struct Blocking <: FunctAI.EventStore
    store::FunctAI.MemoryStore
end
FunctAI.keep!(s::Blocking, e::FunctAI.Event) = (Libc.systemsleep(0.3); FunctAI.keep!(s.store, e))

@testset "the guarantees" begin

@testset "a receiver is given the kept form, and nothing else of a call" begin
    # a module's value, not kept: neither in its JSON nor anywhere in the object an observer or a store gets
    @program function echo_secret(x::String)::String
        x
    end
    seen, store = FunctAI.Event[], FunctAI.MemoryStore()
    with_settings(log_content=false, observers=[e -> push!(seen, e)], journal=FunctAI.Journal(store; required=true)) do
        echo_secret("SECRET-1")
    end
    @test FunctAI.drain()
    @test [e.kind for e in seen] == [:started, :done]
    @test !any(e -> holds(e, "SECRET-1"), seen)
    @test !any(e -> holds(e, "SECRET-1"), FunctAI.events_after(store, seen[1].tree, nothing))
    @test !hasfield(FunctAI.Event, :native)                         # an event holds its JSON form and nothing else

    # an AI function that returns several outputs: its done event holds them all, so it keeps none when one is dropped
    f = AIFunction("triage", "Triage."; inputs=(message=String,), outputs=(private_note=String, team=String))
    seen, store = FunctAI.Event[], FunctAI.MemoryStore()
    s = using_fake(() -> stream(f, "hello"), fake(xml(:private_note => "SSN 123-45-6789", :team => "billing"));
                   log_content=(private_note=false,), observers=[e -> push!(seen, e)], journal=FunctAI.Journal(store; required=true))
    @test fetch(s) == (private_note="SSN 123-45-6789", team="billing")
    @test FunctAI.drain()
    kept_done = only(e for e in FunctAI.events_after(store, s.log[1].tree, nothing) if e.kind === :done)
    observed_done = only(e for e in seen if e.kind === :done)
    for d in (kept_done, observed_done)
        @test d.content == false && !haskey(d, "value") && !holds(d, "123-45-6789")
    end
    @test !any(e -> holds(e, "123-45-6789"), seen)                  # its text pieces are not kept either
    whole_done = only(e for e in eachevent(s) if e.kind === :done)  # the whole form (the stream its caller watches) has it
    @test whole_done.value == Dict("private_note" => "SSN 123-45-6789", "team" => "billing")
    # its answer alone is kept when the other output is: one output's function keeps its done value
    seen = FunctAI.Event[]
    using_fake(() -> f("hello"), fake(xml(:private_note => "n", :team => "billing")); log_content=(team=true,), observers=[e -> push!(seen, e)])
    @test FunctAI.drain()
    @test only(e for e in seen if e.kind === :done).value == Dict("private_note" => "n", "team" => "billing")
end

@testset "an error's message says what a value is, never the value" begin
    @program function check_pin(pin::Int)::Bool
        pin == 1234
    end
    vault(user) = "hunter2-secret"            # a mistake: text, not a number
    @program function login(user::String)::Bool
        check_pin(vault(user))
    end
    seen = Any[]
    dir = mktempdir()
    err = try
        with_settings(log_content=Dict("pin" => false), observers=[e -> push!(seen, FunctAI.event_json(e))], log_calls=dir) do
            login("maxime")
        end
    catch e
        e
    end
    @test FunctAI.drain()
    @test err isa InterfaceError && err.field == "pin" && !occursin("hunter2", sprint(showerror, err))
    @test occursin("a text of 14 characters", err.msg)
    @test !any(e -> occursin("hunter2", FunctAI.LMCC.json_text(e)), seen)
    @test !any(r -> occursin("hunter2", FunctAI.LMCC.json_text(r)), first(FunctAI.read_log(dir)))
end

@testset "a slow, blocked or failing observer never slows the call" begin
    f = AIFunction("team", "Which team?"; inputs=(message=String,), output=String)
    go(; kw...) = using_fake(() -> f("I was charged twice."), fake(xml(:result => "billing")); kw...)
    go(observers=[e -> nothing])
    @test FunctAI.drain()
    # slow: the call does not wait for it; it still gets every event, in order
    got = Any[]
    slow = e -> (sleep(0.2); push!(got, e))
    slow(nothing); empty!(got)                                      # compiled before it is timed
    t = @elapsed go(observers=[slow])
    @test t < 0.5                                                  # given in the call, its 5 events would take 1 s
    @test FunctAI.drain(10)
    kinds = [e.kind for e in got]
    @test kinds[1:2] == [:started, :request] && kinds[end] === :done && all(==(:text), kinds[3:end-1])
    # blocked on a gate: the call's code runs and returns meanwhile
    gate, entered = Channel{Nothing}(1), Channel{Nothing}(1)
    blocked = e -> e.kind === :started && (put!(entered, nothing); take!(gate))
    job = @async go(observers=[blocked])
    take!(entered)
    @test timedwait(() -> istaskdone(job), 5.0) === :ok && fetch(job) == "billing"
    put!(gate, nothing)
    # a Channel nobody reads: the call ends
    ch = Channel{Any}(1)
    job = @async go(observers=[ch])
    @test timedwait(() -> istaskdone(job), 5.0) === :ok
    @test !FunctAI.drain(0.2)                                     # it has not taken them: drain says so
    kinds = Symbol[]
    while isempty(kinds) || last(kinds) !== :done
        push!(kinds, take!(ch).kind)                               # in order, as they happened
    end
    @test kinds[1:2] == [:started, :request]
    @test FunctAI.drain()
    # an observer that calls an AI function itself: no deadlock, and that call is a tree of its own
    inner = Ref{Any}(nothing)
    reentrant = e -> e.kind === :started && (inner[] = predict(f, "from an observer"))
    outer = FunctAI.Event[]
    configure!(lm="gpt-4.1-mini", router=FakeRouter(Any[]; responder=(r, i) -> xml(:result => "billing")))
    try
        p = with_settings(observers=[reentrant, e -> push!(outer, e)]) do
            predict(f, "x")
        end
        @test FunctAI.drain(10)
        @test inner[] isa Prediction && inner[].call != p.call
        @test all(e -> e.tree == p.call, outer)                    # nothing of the observer's call in this tree
    finally
        configure!(lm=nothing, router=nothing)
    end
    # too slow: beyond its buffer an observer loses events, and sees a loss in the positions it is given
    @test FunctAI.OBSERVER_BUFFER >= 1000
end

@testset "a best-effort journal never makes the call wait, even one that holds its thread" begin
    f = AIFunction("team", "Which team?"; inputs=(message=String,), output=String)
    using_fake(() -> f("x"), fake(xml(:result => "billing")))
    store = Blocking(FunctAI.MemoryStore())
    t = @elapsed using_fake(() -> f("x"), fake(xml(:result => "billing")); journal=store)
    Threads.nthreads() > 1 && @test t < 0.3                      # with one thread, a store that never yields holds everything
    @test FunctAI.drain(10) && FunctAI.finished(store.store, only(FunctAI.trees(store.store)))
end

@testset "a journal's faults, its silence and its barriers" begin
    f = AIFunction("team", "Which team?"; inputs=(message=String,), output=String)
    # a store that is not one is refused where it is set
    @test_throws ArgumentError FunctAI.Journal(42)
    @test_throws ArgumentError with_settings(() -> nothing; journal="a folder")
    # a store whose own code fails: the call stops at its start, and the error says why
    router = fake(xml(:result => "billing"))
    err = try using_fake(() -> f("x"), router; journal=FunctAI.Journal(Broken(); required=true)) catch e e end
    @test err isa JournalError && err.code == "journal-end" && err.outcome.failed.code == "journal-barrier"
    @test err.journal == "unknown"                                 # a fault is no answer: the journal did not say it refused
    @test err.cause isa ArgumentError && isempty(router.requests)
    @test occursin("the journal met", sprint(showerror, err))
    # a store that never answers: the send times out, the barrier fails, the call does not hang
    hanging = Hanging(Channel{Nothing}(Inf))
    t = @elapsed err = try
        using_fake(() -> f("x"), fake(xml(:result => "billing")); journal=FunctAI.Journal(hanging; required=true, retries=0, timeout=0.2))
    catch e
        e
    end
    @test err isa JournalError && err.outcome.failed.code == "journal-barrier" && err.cause isa FunctAI.NoAnswer
    @test t < 5
    foreach(_ -> put!(hanging.gate, nothing), 1:8)
    # a barrier waits for its own event, not for an event made after it (a call beside it in the tree)
    store = SilentAt(FunctAI.MemoryStore(), [3])
    tree = "01926c00-0001-7000-8000-00000000000a"
    w = FunctAI.LogWriter(FunctAI.Journal(store; required=true, retries=0), tree)
    prog = FunctAI.LMCC.jobj("name" => "m", "kind" => "module", "module" => "t", "version" => "sha256:" * "1"^64,
                             "interface" => "sha256:" * "2"^64, "answer" => "result")
    e1 = FunctAI.Event(:started, tree, 1, 1, nothing, "2026-09-28T10:00:00.010000Z", tree, "m",
                       FunctAI.LMCC.jobj("parent" => nothing, "root" => tree, "program" => prog, "inputs" => FunctAI.LMCC.jobj(),
                                         "content" => true, "saw" => Any[]))
    e2 = FunctAI.Event(:request, tree, 1, 2, FunctAI.Position(1, 1), "2026-09-28T10:00:00.020000Z", tree, "m",
                       FunctAI.LMCC.jobj("request" => 1, "model" => nothing))
    e3 = FunctAI.Event(:request, tree, 1, 3, FunctAI.Position(1, 2), "2026-09-28T10:00:00.030000Z", tree, "m",
                       FunctAI.LMCC.jobj("request" => 2, "model" => nothing))
    n1, n2, n3 = push!(w, e1), push!(w, e2), push!(w, e3)
    @test FunctAI.confirmation(w, n3) === :unanswered
    @test FunctAI.confirmation(w, n2) === :confirmed && FunctAI.confirmation(w, n1) === :confirmed
    # settling from the error alone: it carries its store
    err = try
        using_fake(() -> f("x"), fake(xml(:result => "billing")); journal=FunctAI.Journal(FailingStore(unanswered=[:done]); required=true))
    catch e
        e
    end
    @test err isa JournalError && err.journal == "unknown" && FunctAI.settle(err) === :not_kept
end

"A store that another writer claims from when a tool call arrives (the writer is fenced), and whose first answer is lost."
mutable struct Fencing <: FunctAI.EventStore
    store::FunctAI.MemoryStore
    lost_first::Bool
end
function FunctAI.keep!(s::Fencing, e::FunctAI.Event)
    e.kind === :tool_call && e.writer == 1 && FunctAI.claim!(s.store, e.tree)
    answer = FunctAI.keep!(s.store, e)
    s.lost_first && (s.lost_first = false; error("the answer was lost"))
    answer
end
FunctAI.events_after(s::Fencing, tree, after) = FunctAI.events_after(s.store, tree, after)

@testset "a required journal through a real call: a lost answer resent, a fenced writer's tool never runs" begin
    ran = Ref(false)
    look = tool(order -> (ran[] = true; "shipped"); name="look", description="Look an order up.",
                parameters=Dict("type" => "object", "properties" => Dict("order" => Dict("type" => "string"))))
    helper = AIFunction("helper", "Help."; inputs=(q=String,), output=String, tools=[look])
    store = Fencing(FunctAI.MemoryStore(), true)
    router = fake((calls=[("c1", "look", (order="A1",))],), xml(:result => "It shipped."))
    dir = mktempdir()
    err = try using_fake(() -> helper("Where is A1?"), router; journal=FunctAI.Journal(store; required=true), log_calls=dir) catch e e end
    @test !ran[] && length(router.requests) == 1                     # the start's lost answer was resent (duplicate); the tool never ran
    @test err isa JournalError && err.code == "journal-end" && err.journal == "refused"
    @test err.outcome.failed isa JournalError && err.outcome.failed.code == "journal-barrier"
    @test err.cause isa FunctAI.StoreRefusal && err.cause.code == "event-conflict"
    kept = FunctAI.events_after(store.store, err.tree, nothing)
    @test kept[1].kind === :started && !FunctAI.finished(store.store, err.tree)   # the log waits for the writer that claimed it
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["journal"] == "refused" && rec["error"]["code"] == "journal-barrier"
end

@testset "a store checks every event against the schema, and keeps copies" begin
    @program function echo(x::String)::String
        x
    end
    seen = FunctAI.Event[]
    with_settings(observers=[e -> push!(seen, e)]) do
        echo("original")
    end
    @test FunctAI.drain()
    start = FunctAI.event_json(seen[1])
    # a started whose program is not the call log's program object: malformed, as JSON and as an Event
    bad = deepcopy(start)
    bad["program"] = Dict{String,Any}()
    by_hand = FunctAI.Event(:started, bad["tree"], 1, 1, nothing, bad["at"], bad["call"], bad["function"],
                            Dict{String,Any}(k => v for (k, v) in bad if k in ("parent", "root", "program", "inputs", "content", "saw")))
    for x in (bad, by_hand)
        err = try FunctAI.keep!(FunctAI.MemoryStore(), x) catch e e end
        @test err isa FunctAI.StoreRefusal && err.code == "event-malformed"
    end
    # an Event made by hand is no certificate
    invalid = FunctAI.Event(:started, "not-an-id", 1, 1, nothing, "not-a-time", "not-an-id", "f", FunctAI.LMCC.jobj())
    err = try FunctAI.keep!(FunctAI.MemoryStore(), invalid) catch e e end
    @test err isa FunctAI.StoreRefusal && err.code == "event-malformed"
    err = try FunctAI.keep!(FunctAI.MemoryStore(), [invalid]) catch e e end
    @test err isa FunctAI.StoreRefusal && err.code == "event-malformed" && err.event == FunctAI.Position(1, 1)
    # what it keeps changes with nothing its callers do: what they gave, what they read, JSON made from it
    store = FunctAI.MemoryStore()
    given = FunctAI.event_json(seen[1])
    @test FunctAI.keep!(store, given) === :kept
    given["inputs"]["x"] = "CHANGED BY THE WRITER"
    read1 = FunctAI.events_after(store, seen[1].tree, nothing)
    read1[1].inputs["x"] = "CHANGED BY A READER"
    FunctAI.event_json(read1[1])["inputs"]["x"] = "CHANGED IN JSON"
    @test FunctAI.events_after(store, seen[1].tree, nothing)[1].inputs == Dict("x" => "original")
    @test FunctAI.keep!(store, [FunctAI.event_json(seen[1]), FunctAI.event_json(seen[2])]) === :kept     # a batch too
    @test FunctAI.keep!(store, seen[1]) === :duplicate                              # the same event, still
    # an event itself cannot be changed through what it gives
    e = seen[1]
    e.program["name"] = "changed"
    @test e.program["name"] == "echo"
end

@testset "a follower drops what is not an event of its format, and goes on" begin
    f = FunctAI.Follower(:kept)
    @test FunctAI.receive!(f, Dict("functai_event" => 2, "kind" => "text")) === :malformed
    @test !f.stopped
    @test_throws ArgumentError FunctAI.Event(Dict("functai_event" => 2, "kind" => "text"))
end

@testset "a tree's end is its last event: its outermost call waits for the calls inside it" begin
    gate, child_started = Channel{Nothing}(1), Channel{Nothing}(1)
    child_task = Ref{Task}()
    @program function late_child(x::Int)::String
        put!(child_started, nothing)
        take!(gate)
        "child"
    end
    @program function parent_ends(x::Int)::String
        child_task[] = @async late_child(1)
        take!(child_started)
        "parent"
    end
    seen, store = FunctAI.Event[], FunctAI.MemoryStore()
    @async (sleep(0.3); put!(gate, nothing))                   # released later, by someone else
    v = @test_logs (:warn, r"still running") match_mode = :any with_settings(() -> parent_ends(1);
                                                                           observers=[e -> push!(seen, e)], journal=FunctAI.Journal(store; required=true))
    @test v == "parent" && fetch(child_task[]) == "child"
    @test FunctAI.drain()
    order = [(e.function, e.kind) for e in seen]
    @test order == [("parent_ends", :started), ("late_child", :started), ("late_child", :done), ("parent_ends", :done)]
    kept = FunctAI.events_after(store, seen[1].tree, nothing)
    @test [(e.function, e.kind) for e in kept] == order && FunctAI.finished(store, seen[1].tree)
    # many calls ending on other threads while the outermost call ends: every one ends before it, numbering dense
    begun = Channel{Int}(Inf)
    @program function quick(i::Int)::Int
        put!(begun, i)
        sleep(0.002 * (i % 7))
        i
    end
    @program function scatter(n::Int)::Int
        for i in 1:n
            Threads.@spawn quick(i)                        # never waited for
        end
        foreach(_ -> take!(begun), 1:n)                    # every one has begun its call: they are steps of this one
        n
    end
    store = FunctAI.MemoryStore()
    s = nothing
    @test (@test_logs (:warn, r"still running") match_mode = :any begin
        s = with_settings(() -> stream(scatter, 30); journal=FunctAI.Journal(store; required=true), log_content=false)
        fetch(s)
    end) == 30
    evs = collect(eachevent(s))
    @test length(evs) == 62 && [e.seq for e in evs] == 1:62 && evs[end].call == evs[1].call && evs[end].kind === :done
    @test FunctAI.replay(evs).finished && all(c -> c.ended === :done, values(FunctAI.replay(evs).calls))
    @test [FunctAI.Position(e) for e in FunctAI.events_after(store, evs[1].tree, nothing)] == [FunctAI.Position(e) for e in evs]
    # a call that begins once the tree has ended (a task the program did not wait for, slower to start) is a tree of its own
    late = Ref{Task}()
    @program function starts_late(x::Int)::Int
        late[] = @async (sleep(0.2); quick(x))
        x
    end
    seen = FunctAI.Event[]
    @test (@test_logs (:warn, r"has ended") match_mode = :any begin
        v = with_settings(() -> starts_late(3); observers=[e -> push!(seen, e)])
        fetch(late[])
        v
    end) == 3
    @test FunctAI.drain()
    trees = Dict(e.function => (e.tree, datum_parent(e)) for e in seen if e.kind === :started)
    @test trees["quick"][1] != trees["starts_late"][1] && trees["quick"][2] === nothing
    # work meant to outlive the call is started detached: a tree of its own, which the call does not wait for
    gate2 = Channel{Nothing}(1)
    @program function warm(x::Int)::Int
        take!(gate2)
        x
    end
    detached_task = Ref{Task}()
    @program function answer_now(x::Int)::Int
        FunctAI.detached() do
            detached_task[] = @async warm(x)
        end
        x
    end
    seen = FunctAI.Event[]
    @test with_settings(() -> answer_now(7); observers=[e -> push!(seen, e)]) == 7
    put!(gate2, nothing)
    @test fetch(detached_task[]) == 7
    @test FunctAI.drain()
    trees = Dict(e.function => e.tree for e in seen if e.kind === :started)
    @test trees["warm"] != trees["answer_now"]
end

@testset "closing a program's stream: no call starts after, and it ends Cancelled" begin
    gate, entered = Channel{Nothing}(1), Channel{Nothing}(1)
    ran = Ref(false)
    @program function cancellation_child(x::String)::String
        ran[] = true
        x
    end
    @program function cancellable(x::String)::String
        put!(entered, nothing)
        take!(gate)
        cancellation_child(x)
    end
    s = stream(cancellable, "a")
    take!(entered)
    close(s)
    put!(gate, nothing)
    err = try fetch(s) catch e e end
    @test err isa Cancelled && !ran[]
    @test [e.kind for e in eachevent(s)] == [:started, :failed]
    # a body that returns after the stream was closed does not end with its value
    @program function returns_anyway(x::String)::String
        put!(entered, nothing)
        take!(gate)
        x
    end
    s = stream(returns_anyway, "a")
    take!(entered)
    close(s)
    put!(gate, nothing)
    @test (try fetch(s) catch e e end) isa Cancelled
end

@testset "a program is checked however it is called, and a refused call is a call" begin
    @program function takes(x::String; y::Int)::Int
        y
    end
    for (thunk, field) in ((() -> takes(), "x"), (() -> takes("a"), "y"), (() -> takes("a"; y=1, z=2), "z"),
                           (() -> takes("a", "b"; y=1), nothing), (() -> takes("a"; x="b", y=1), "x"))
        seen = FunctAI.Event[]
        dir = mktempdir()
        err = try with_settings(thunk; observers=[e -> push!(seen, e)], log_calls=dir) catch e e end
        @test FunctAI.drain()
        @test err isa InterfaceError && err.code == "interface-input" && err.field == field
        @test [e.kind for e in seen] == [:started, :failed]
        @test only(first(FunctAI.read_log(dir)))["error"]["code"] == "interface-input"
    end
    @test takes("a"; y=2) == 2 && takes(; x="a", y=3) == 3          # every input may be given by name
    # a program with no inputs, from the macro and from data
    @program function no_inputs()::String
        "ok"
    end
    @test no_inputs() == "ok"
    p = AIProgram("constant", (; kw...) -> 1; interface=Dict("description" => "", "inputs" => Any[],
                                                              "outputs" => [Dict("name" => "result", "shape" => Dict("type" => "integer"))]))
    @test p() == 1
    # several outputs, one opaque: the record is not asked for a JSON form, each output is checked by its field
    @program outputs = (obj = Any, n = Int) function opaque_outputs(x::Int)
        (obj=Opaque(), n=x)
    end
    dir = mktempdir()
    r = with_settings(() -> opaque_outputs(1); log_calls=dir)
    @test r.n == 1 && r.obj isa Opaque
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["described"] == Dict("inputs" => [], "outputs" => ["obj"]) && rec["outputs"]["n"] == 1
end

@testset "defaults: data in the interface, made anew for every call" begin
    @program function default_vector(x::Vector{Int} = [1])::Int
        push!(x, 2)
        length(x)
    end
    @test default_vector() == 2 && default_vector() == 2
    @test FunctAI.interface(default_vector)["inputs"][1]["shape"]["default"] == [1]
    # a constant is data too, and so is its value in the interface
    @eval const DEFAULT_TONE = "kind"
    @eval @program function toned(m::String; tone::String = DEFAULT_TONE)::String
        tone
    end
    @test FunctAI.interface(toned)["inputs"][2]["shape"]["default"] == "kind" && toned("x") == "kind"
    # an AI function's default is data sent to the model: a literal or a constant; each call gets a copy
    @ai function tagger(text::String; tags::Vector{String} = String[])::String
        "Tag."
        result::String = ai"the tags"
        push!(tags, "x")
        join(tags, ",")
    end
    @test using_fake(() -> tagger("a"), fake(xml(:result => "t"))) == "x"
    @test using_fake(() -> tagger("a"), fake(xml(:result => "t"))) == "x"
    # a default that uses another input, or is computed, is refused when the function is defined
    @eval message = "A GLOBAL OF THE SAME NAME"
    err = try @eval(@ai function echo_default(message::String; echo::String = message)::String
        "Echo."
    end) catch e e end
    @test err isa ArgumentError && occursin("uses message, another input", err.msg)
    err = try @eval(@ai function stamped(x::String; at::Float64 = time())::String
        "Stamp."
    end) catch e e end
    @test err isa ArgumentError && occursin("is computed", err.msg)
    err = try @eval(@ai function not_const(x::String; tone::String = message)::String
        "Tone."
    end) catch e e end
    @test err isa ArgumentError && occursin("not a constant", err.msg)
    @eval @ai function const_tone(x::String; tone::String = DEFAULT_TONE)::String
        "Tone."
    end
    @test FunctAI.interface(const_tone)["inputs"][2]["shape"]["default"] == "kind"
end

@testset "a value with no JSON form is said to be one, however it is held" begin
    @program function opaque_echo(x)
        x
    end
    dir = mktempdir()
    with_settings(() -> opaque_echo(Dict("obj" => Opaque())); log_calls=dir)
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["described"] == Dict("inputs" => ["x"], "outputs" => ["result"])
    # a dictionary that holds those keys itself is data
    dir = mktempdir()
    with_settings(() -> opaque_echo(Dict("\$type" => "a", "\$repr" => "b")); log_calls=dir)
    @test !haskey(only(first(FunctAI.read_log(dir))), "described")
end

@testset "each exchange's hash is of the request it sent, through mixed retries" begin
    f = AIFunction("one", "Answer."; inputs=(x=String,), output=String)
    dir = mktempdir()
    r = fake("unreadable", (text="<result>cut", finish="length"), xml(:result => "ok"))
    @test using_fake(() -> f("hi"), r; retries=2, log_calls=dir) == "ok"
    rec = only(first(FunctAI.read_log(dir)))
    hashes = [x["request_hash"] for x in rec["exchanges"]]
    @test [length(x.messages) for x in r.requests] == [1, 3, 1]
    @test hashes[1] != hashes[2] && hashes[3] == hashes[1]           # the third sent the first's messages again
end

@testset "a saved folder written before interfaces is checked as every interface is" begin
    f = AIFunction("old"; inputs=(x=String,), output=String)
    m = FunctAI.to_manifest(f)
    n = m["nodes"][m["entry"]]
    delete!(n, "interface")
    n["ai"]["signature"]["fields"][1]["shape"]["minLength"] = "three"
    for g in (() -> FunctAI.describe(m), () -> FunctAI.from_manifest(m))
        err = try g() catch e e end
        @test err isa LoadRefused && err.code == "interface-malformed"
    end
end

@testset "a tool-using call records its tool calls, and their size" begin
    t = tool(x -> "ok"; name="lookup", description="Look it up.", parameters=Dict("type" => "object"))
    f = AIFunction("tooluser", "Help."; inputs=(x=String,), output=String, tools=[t])
    dir = mktempdir()
    using_fake(() -> f("hi"), fake((calls=[("c1", "lookup", (a=1,))],), xml(:result => "ok")); log_calls=dir)
    rec = only(first(FunctAI.read_log(dir)))
    @test collect(keys(rec["outputs"])) == ["calls", "result"] && rec["outputs"]["result"] == "ok"
    @test haskey(rec["sizes"]["outputs"], "calls")
    dir = mktempdir()
    using_fake(() -> f("hi"), fake((calls=[("c1", "lookup", (a=1,))],), xml(:result => "ok")); log_calls=dir, log_content=(x=false,))
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["omitted"] == Dict("inputs" => ["x"], "outputs" => ["calls"]) && haskey(rec["sizes"]["outputs"], "calls")
    # what stops a call is never a tool's answer
    stopper = tool(x -> throw(JournalError("journal-scope", "set around the tree")); name="lookup", description="Look it up.",
                   parameters=Dict("type" => "object"))
    g = AIFunction("tooluser", "Help."; inputs=(x=String,), output=String, tools=[stopper])
    err = try using_fake(() -> g("hi"), fake((calls=[("c1", "lookup", (a=1,))],), xml(:result => "ok"))) catch e e end
    @test err isa JournalError && err.code == "journal-scope"
end

@testset "every event and record a call makes passes the contract's schemas" begin
    lookup = tool(order -> "It left the depot."; name="lookup", description="Look an order up.",
                  parameters=Dict("type" => "object", "properties" => Dict("order" => Dict("type" => "string")), "required" => ["order"]))
    f = AIFunction("support", "Answer."; inputs=(message=String, secret=String), output=String, tools=[lookup], reasoning=true)
    g = AIFunction("team", "Which team?"; inputs=(message=String,), output=String)
    @program function prog(message::String)::String
        g(message)
    end
    replies = [Any[(text="", calls=[("c1", "lookup", (order="B-2210",))]), xml(:reasoning => "hm", :result => "On its way.")],
               Any[(text="", calls=[("c1", "lookup", (order="B-2210",))]), xml(:reasoning => "hm", :result => "On its way.")],
               Any[xml(:result => "billing")], Any[ErrorException("boom")], Any["??", xml(:result => "billing")]]
    runs = [(true, f, ("Where is B-2210?", "pw")), (Dict("secret" => false), f, ("Where is B-2210?", "pw")),
            (false, prog, ("hi",)), (Dict("message" => false), g, ("hi",)), (true, g, ("hi",))]
    events, records = Any[], Any[]
    for ((content, fn, args), rs) in zip(runs, replies)
        dir = mktempdir()
        store = FunctAI.MemoryStore()
        try
            using_fake(fake(rs...); log_content=content, observers=[e -> push!(events, FunctAI.event_json(e))], log_calls=dir,
                       journal=FunctAI.Journal(store; required=true), api_retries=0) do
                s = stream(fn, args...)
                foreach(e -> push!(events, FunctAI.event_json(e)), eachevent(s))
                try fetch(s) catch end
            end
        catch
        end
        FunctAI.drain()
        append!(events, (FunctAI.event_json(e) for t in FunctAI.trees(store) for e in FunctAI.events_after(store, t, nothing)))
        append!(records, first(FunctAI.read_log(dir)))
    end
    @test length(events) > 50 && length(records) >= 5
    @test all(e -> FunctAI.schema_fault("event", e) === nothing, events)
    @test all(r -> FunctAI.schema_fault("call", r) === nothing, records)
    # and the schemas are read as schemas: what they refuse is refused
    e = deepcopy(first(events))
    delete!(e["program"], "interface")
    @test FunctAI.schema_fault("event", e) !== nothing
    e = deepcopy(first(events)); e["function"] = ""
    @test FunctAI.schema_fault("event", e) !== nothing
    r = deepcopy(first(records)); r["id"] = "not-an-id\n"
    @test FunctAI.schema_fault("call", r) !== nothing
end

end
