# What stage 1 promises that the contract's cases, which check events and
# records as data, cannot see: what a receiver is given (never a value the
# log does not keep, in any field of the object), observers that never slow
# a call, a store that checks and copies what it keeps, a tree whose end is
# last, a closed stream, a program checked however it is called, defaults
# made anew, a re-ask's request and its hash kept together, old saved
# folders checked, tool calls recorded, and every event and record passing
# the contract's schemas. Each was a counterexample in a review of the first
# implementation (2026-09-28); each drives the library's own path. The
# second review's (2026-09-29) follow them: defaults that are snapshots, the
# same after saving and loading; every call, not only the outermost, ending
# after the calls inside it, with nothing joining it in between; closing
# while a call waits for them; observers not fed their own calls; one
# thread; positions as JSON numbers; store faults and one send at a time;
# schema patterns. A fake router; no network.

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

"A structured default: a struct holding a list."
struct Shelf
    name::String
    books::Vector{String}
end

"A struct whose constructor changes what it is given: building it again from its JSON would change it."
struct Incremented
    n::Int
    Incremented(n) = new(n + 1)
end

"A record whose field is `missing`: its JSON is `{\"n\": null}`."
const MissingRecord = NamedTuple{(:n,),Tuple{Missing}}

"A store whose first append meets a passing condition that throws a `KeyError` (a cache miss, say)."
mutable struct MissOnce <: FunctAI.EventStore
    store::FunctAI.MemoryStore
    missed::Bool
end
function FunctAI.keep!(s::MissOnce, e::FunctAI.Event)
    s.missed || (s.missed = true; throw(KeyError("a cache entry")))
    FunctAI.keep!(s.store, e)
end
FunctAI.events_after(s::MissOnce, tree, after) = FunctAI.events_after(s.store, tree, after)

"A store that never answers, and counts the appends it was given."
struct CountingHang <: FunctAI.EventStore
    gate::Channel{Nothing}
    calls::Threads.Atomic{Int}
end
FunctAI.keep!(s::CountingHang, e::FunctAI.Event) = (Threads.atomic_add!(s.calls, 1); take!(s.gate); :kept)

"The request a function sends when called with `args` (the fake router's, as a dict)."
function sent_request(f, args...)
    r = FakeRouter(Any[]; responder=(req, i) -> xml(:result => "ok"))
    using_fake(() -> f(args...), r; retries=0)
    LM15.to_dict(only(r.requests))
end

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
    # a store that never yields holds its thread: with one default thread, calls on it wait (documented at drain;
    # the one-thread promise, for code that yields, is tested in a process of its own below)
    Threads.nthreads(:default) > 1 && @test t < 0.3
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
    @program function default_seen(x::Vector{Int} = [1])::Int
        n = length(x)
        push!(x, 2)
        n
    end
    dir = mktempdir()
    @test with_settings(() -> [default_seen(), default_seen()]; log_calls=dir) == [1, 1]
    @test [r["inputs"]["x"] for r in first(FunctAI.read_log(dir))] == [[1], [1]]     # what the code got, each call
    # a constant that can change (a Vector) is Julia code, as Julia runs it: the record never claims a value
    # the code did not get (the contract: a module's own default applied, no value recorded)
    @eval const GROWING_ITEMS = [1]
    @eval @program function default_const(x::Vector{Int} = GROWING_ITEMS)::Int
        n = length(x)
        push!(x, 2)
        n
    end
    dir = mktempdir()
    @test with_settings(() -> [default_const(), default_const()]; log_calls=dir) == [1, 2]
    @test all(r -> !haskey(get(r, "inputs", Dict()), "x"), first(FunctAI.read_log(dir)))
    @test !haskey(FunctAI.interface(default_const)["inputs"][1]["shape"], "default")
    @test FunctAI.interface(default_const)["inputs"][1]["optional"] == true
    # an indexed global is read, not written: not a typed vector literal
    @eval COUNTS = [5]
    @test FunctAI.is_literal(:(Int[1, 2]), Main) && !FunctAI.is_literal(:(COUNTS[1]), Main)
    @test FunctAI.is_literal(:(Vector{Int}[]), Main) && FunctAI.is_literal(:(Base.String["a"]), Main)
    @test !FunctAI.is_literal(:(Vector{COUNTS}[]), Main)
    @eval @program function counted(n::Int = COUNTS[1])::Int
        n
    end
    @test !haskey(FunctAI.interface(counted)["inputs"][1]["shape"], "default") && counted() == 5
    err = try @eval(@ai function counted_ai(x::String; n::Int = COUNTS[1])::String
        "Count."
    end) catch e e end
    @test err isa ArgumentError && occursin("is computed", err.msg)
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
    # a constant whose value can change is refused for an AI function: it would be sent as it was when defined
    err = try @eval(@ai function const_items(x::String; items::Vector{Int} = GROWING_ITEMS)::String
        "Items."
    end) catch e e end
    @test err isa ArgumentError && occursin("can change", err.msg)
    @test FunctAI.frozen((a=1, b="x", c=:s)) && !FunctAI.frozen((a=[1],)) && !FunctAI.frozen(Dict(1 => 2))
end

@testset "an AI function's default is its own: changing its source later changes nothing, saved or not" begin
    # mutable and structured defaults, each changed after the function is defined
    numbers, shelf, index = [1], Shelf("kept", ["Emma"]), Dict("a" => [1])
    f = AIFunction("defaults", "Use them."; inputs=(x=Vector{Int}, s=Shelf, d=Dict{String,Vector{Int}}, t=NamedTuple{(:n,),Tuple{Int}}),
                   output=String, defaults=(x=numbers, s=shelf, d=index, t=(n=1,)))
    before_change = sent_request(f)
    push!(numbers, 2); push!(shelf.books, "CHANGED"); push!(index["a"], 2); index["b"] = [3]
    after_change = sent_request(f)
    @test LMCC.json_equal(before_change, after_change)
    @test !occursin("CHANGED", LMCC.json_text(after_change))
    @test FunctAI.interface(f)["inputs"][1]["shape"]["default"] == [1]
    # saved and loaded: the same request, left out or given, and the loaded one shares nothing with the saved
    path = mktempdir()
    FunctAI.save(path, f)
    g = FunctAI.load(path; types=(s=Shelf,))
    @test LMCC.json_equal(sent_request(f), sent_request(g))
    given = ([7], Shelf("given", ["Ulysses"]), Dict("z" => [9]), (n=2,))
    @test LMCC.json_equal(sent_request(f, given...), sent_request(g, given...))
    # the @ai macro takes the same path: a literal default is data, the same after loading
    @ai function shelved(q::String; books::Vector{String} = ["Emma"], opts::NamedTuple{(:n,),Tuple{Int}} = (n = 1,))::String
        "Answer."
    end
    path = mktempdir()
    FunctAI.save(path, shelved)
    @test LMCC.json_equal(sent_request(shelved, "q"), sent_request(FunctAI.load(path), "q"))
end

@testset "a default's JSON is what is sent, never made again from its value: the same bytes before and after saving and loading" begin
    # a set whose order is its layout's (building it again would change its order), a constructor that changes what it
    # is given, and structured, nested, numeric and ordered defaults, with and without `types` when loaded
    s = Set(1:1000)
    foreach(i -> delete!(s, i), 11:1000)
    cases = Any[
        ("set", Set{Int}, s),
        ("constructor", Incremented, Incremented(1)),
        ("struct", Shelf, Shelf("kept", ["Emma", "Ulysses"])),
        ("nested", Dict{String,Vector{NamedTuple{(:a, :b),Tuple{Int,Float64}}}}, Dict("z" => [(a=1, b=2.5)], "a" => [(a=3, b=-1.0)], "m" => [])),
        ("ordered", NamedTuple{(:z, :a, :m),Tuple{Int,String,Vector{Symbol}}}, (z=1, a="b", m=[:y, :x])),
        ("numbers", Vector{Float64}, [1, 2.5, 1e300, -3]),
        ("int as float", Float64, 3),
        ("whole float as int", Int, 1.0),
        ("ints as floats", Vector{Float64}, [1, 2]),
        ("big", BigInt, big(2)^80),
        ("missing inside", MissingRecord, (n=missing,)),
        ("missing", Union{Missing,String}, missing),
    ]
    for (label, T, v) in cases
        f = AIFunction("defaulted", "Use it."; inputs=(q=String, x=T), output=String, defaults=(x=v,))
        data = FunctAI.interface(f)["inputs"][2]["shape"]["default"]
        # the value's JSON as it was, of its declared type (Julia's convert, when it is not one)
        @test LMCC.json_text(data) == LMCC.json_text(FunctAI.jsonvalue(v isa T ? v : convert(T, v)))
        sent = LMCC.json_text(sent_request(f, "q"))
        path = mktempdir()
        FunctAI.save(path, f)
        for g in (FunctAI.load(path), FunctAI.load(path; types=(x=T,)))
            @test LMCC.json_text(FunctAI.interface(g)["inputs"][2]["shape"]["default"]) == LMCC.json_text(data)
            same = LMCC.json_text(sent_request(g, "q")) == sent
            same || @info "the requests differ" label
            @test same                                                    # byte for byte
        end
        @test LMCC.json_text(sent_request(f, "q")) == sent                # and the same on every call
    end
    # what is sent is the JSON; the set's own order, when it was defined, is the one sent
    f = AIFunction("set_default"; inputs=(x=Set{Int},), output=String, defaults=(x=s,))
    @test occursin(FunctAI.json_indented(FunctAI.jsonvalue(s)), only(only(sent_request(f)["messages"])["parts"])["text"])
    # the function's own code gets a copy of the value it was defined with: no constructor runs again, each call its own
    own = AIFunction("own_code"; inputs=(x=Incremented, s=Shelf), output=String,
                     defaults=(x=Incremented(1), s=Shelf("kept", ["Emma"])),
                     body=(ins, outs) -> (push!(ins["s"].books, "CHANGED"); (ins["x"].n, length(ins["s"].books))))
    r = FakeRouter(Any[]; responder=(req, i) -> xml(:result => "ok"))
    @test using_fake(() -> [own(), own()], r; retries=0) == [(2, 2), (2, 2)]
    typed = AIFunction("typed"; inputs=(x=Int, y=Vector{Float64}), output=String, defaults=(x=1.0, y=[1, 2]),
                       body=(ins, outs) -> (ins["x"], ins["y"]))
    got = using_fake(() -> typed(), FakeRouter(Any[]; responder=(req, i) -> xml(:result => "ok")); retries=0)
    @test got[1] === 1 && got[2] isa Vector{Float64} && got[2] == [1.0, 2.0]
    @test !occursin("CHANGED", LMCC.json_text(LM15.to_dict(r.requests[2])))
    # a missing default is sent as null; a missing given is still missing out, with no call
    m = AIFunction("maybe"; inputs=(x=Union{Missing,String},), output=String, defaults=(x=missing,))
    r = FakeRouter(Any[]; responder=(req, i) -> xml(:result => "ok"))
    @test using_fake(() -> m(), r; retries=0) == "ok" && length(r.requests) == 1
    @test using_fake(() -> m(missing), r; retries=0) === missing && length(r.requests) == 1
    # a record's field typed Missing reads null, as its shape says
    @test FunctAI.fromjson(MissingRecord, Dict{String,Any}("n" => nothing)) === (n=missing,)
end

@testset "observers and stores defined after a task began run for that task's calls" begin
    # a worker started first (its world is older than what is defined after it), as a queue worker at a REPL
    f = AIFunction("late", "Answer."; inputs=(x=String,), output=String)
    router = FakeRouter(Any[]; responder=(req, i) -> xml(:result => "ok"))
    run_one(job) = try
        using_fake(() -> f(job.x), router; retries=0, job.settings...)
    catch err
        err
    end
    jobs, results = Channel{Any}(4), Channel{Any}(4)
    worker = Threads.@spawn for job in jobs
        put!(results, run_one(job))
    end
    seen = Threads.Atomic{Int}(0)
    obs = @eval late_observer(e) = Threads.atomic_add!($seen, 1)
    @eval struct LateStore <: FunctAI.EventStore
        store::FunctAI.MemoryStore
    end
    @eval FunctAI.keep!(s::LateStore, e::FunctAI.Event) = FunctAI.keep!(s.store, e)
    store = @eval LateStore(FunctAI.MemoryStore())
    for settings in ((observers=[obs],), (journal=FunctAI.Journal(store; required=true),))
        put!(jobs, (x="a", settings=settings))
        @test timedwait(() -> isready(results), 60.0) === :ok
        @test take!(results) == "ok"
    end
    @test FunctAI.drain(10)
    @test seen[] > 0                                                      # the observer was given the worker's call
    @test FunctAI.finished(store.store, only(FunctAI.trees(store.store))) # and the store kept its tree
    # and it still works from here
    before = seen[]
    @test using_fake(() -> f("b"), router; retries=0, observers=[obs]) == "ok"
    @test FunctAI.drain(10) && seen[] > before
    close(jobs)
    @test timedwait(() -> istaskdone(worker), 30.0) === :ok
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
    observed, observed_lock = Any[], ReentrantLock()           # the observer's task pushes beside the reading task
    for ((content, fn, args), rs) in zip(runs, replies)
        dir = mktempdir()
        store = FunctAI.MemoryStore()
        try
            using_fake(fake(rs...); log_content=content, log_calls=dir,
                       observers=[e -> lock(() -> push!(observed, FunctAI.event_json(e)), observed_lock)],
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
    @test FunctAI.drain()
    append!(events, lock(() -> copy(observed), observed_lock))
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


@testset "every call ends after the calls made inside it: a stream on a call inside a tree is whole once it ends" begin
    gate, inner = Channel{Nothing}(1), Channel{Any}(1)
    returned = Threads.Atomic{Bool}(false)
    leaf_begun, mid_returned = Channel{Nothing}(1), Threads.Atomic{Bool}(false)
    @program function nest_leaf()::String
        put!(leaf_begun, nothing)
        take!(gate)
        "leaf"
    end
    @program function nest_mid()::String
        Threads.@spawn nest_leaf()                 # the branch's grandchild, never waited for by mid's code
        take!(leaf_begun)                          # it has begun its call: it is a step of mid
        mid_returned[] = true
        "mid"
    end
    @program function nest_branch()::String
        Threads.@spawn nest_mid()                  # never waited for by the branch's code
        while !mid_returned[]
            sleep(0.001)
        end
        returned[] = true
        "branch"
    end
    @program function nest_root()::String
        s = stream(nest_branch)                    # a stream on a call inside the tree
        put!(inner, s)
        fetch(s)
        "root"
    end
    @test_logs (:warn, r"nest_mid returned") (:warn, r"nest_branch returned") match_mode = :any begin
        s = stream(nest_root)
        inside = take!(inner)
        @test timedwait(() -> returned[], 10) === :ok
        sleep(0.2)                                 # time enough for an early end to be numbered
        @test isopen(inside) && !any(e -> e.kind in (:done, :failed), inside.log)
        put!(gate, nothing)
        exhausted = [(e.function, e.kind) for e in eachevent(inside)]
        @test fetch(s) == "root" && fetch(inside) == "branch"
        @test exhausted == [(e.function, e.kind) for e in inside.log]       # nothing came after it was exhausted
        at(k) = findfirst(==(k), exhausted)
        @test exhausted[end] == ("nest_branch", :done)
        @test at(("nest_leaf", :done)) < at(("nest_mid", :done)) < at(("nest_branch", :done))
        whole = collect(eachevent(s))
        @test (whole[end].function, whole[end].kind) == ("nest_root", :done)
        @test FunctAI.replay(whole).finished && all(c -> c.ended === :done, values(FunctAI.replay(whole).calls))
    end
end

@testset "closing a stream while its call waits for the calls inside it ends that call Cancelled" begin
    gate, entered = Channel{Nothing}(1), Channel{Nothing}(1)
    waiting = Threads.Atomic{Bool}(false)
    @program function joined_child()::String
        put!(entered, nothing)
        take!(gate)
        "child"
    end
    @program function joined_parent()::String
        Threads.@spawn joined_child()
        take!(entered)
        waiting[] = true
        "parent"
    end
    dir = mktempdir()
    @test_logs (:warn, r"joined_parent returned") match_mode = :any begin
        s = with_settings(() -> stream(joined_parent); log_calls=dir)
        @test timedwait(() -> waiting[], 10) === :ok
        sleep(0.1)                                 # its code has returned; it waits for its child
        @test !any(e -> e.function == "joined_parent" && e.kind in (:done, :failed), s.log)
        close(s)
        put!(gate, nothing)
        @test (try fetch(s) catch e e end) isa Cancelled
        @test [(e.function, e.kind) for e in eachevent(s)] ==
              [("joined_parent", :started), ("joined_child", :started), ("joined_child", :failed), ("joined_parent", :failed)]
    end
    recs = Dict(r["program"]["name"] => r for r in first(FunctAI.read_log(dir)))
    @test recs["joined_parent"]["error"]["type"] == "Cancelled" && recs["joined_child"]["error"]["type"] == "Cancelled"
end

@testset "no call joins a call between the end of the calls inside it and its own end" begin
    # roots that each start a call they never wait for, at every offset around their end, on several threads:
    # each child is either a step of its root, ended before it, or a tree of its own; nothing is lost or late
    seen, seen_lock = FunctAI.Event[], ReentrantLock()
    kids = Task[]
    @program function race_child(i::Int)::Int
        i
    end
    delay = Ref(0)
    @program function race_root(i::Int)::Int
        d, go = delay[], Threads.Atomic{UInt64}(0)
        push!(kids, Threads.@spawn begin
            while go[] == 0
                yield()                            # waits, yielding, for the root to be about to return
            end
            race_child(i)
        end)
        sleep(0.0005)                              # the task is running
        go[] = time_ns()
        while time_ns() < go[] + d                 # the root goes on for d ns (a safepoint, never a bare spin)
            GC.safepoint()
        end
        i
    end
    store = FunctAI.MemoryStore()
    n = 1500
    Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do     # the warnings about unawaited calls are expected
        with_settings(observers=[e -> lock(() -> push!(seen, e), seen_lock)], journal=FunctAI.Journal(store; required=true)) do
            for i in 1:n
                delay[] = rand(isodd(i) ? (0:20_000) : (0:200_000))     # around the window, and past it
                race_root(i)
            end
        end
        foreach(wait, kids)
    end
    @test !any(istaskfailed, kids) && sort!(fetch.(kids)) == 1:n
    @test FunctAI.drain(60)
    trees = Dict{String,Vector{FunctAI.Event}}()
    foreach(e -> push!(get!(() -> FunctAI.Event[], trees, e.tree), e), seen)
    children = Dict{String,Tuple{String,Any}}()        # a child's call id => (its tree, its parent)
    whole = true
    for (tree, evs) in trees
        sort!(evs; by=e -> e.seq)
        whole &= [e.seq for e in evs] == 1:length(evs) && evs[end].call == tree && evs[end].kind === :done
        whole &= [FunctAI.Position(e) for e in FunctAI.events_after(store, tree, nothing)] == [FunctAI.Position(e) for e in evs]
        for e in evs
            e.kind === :started && e.function == "race_child" && (children[e.call] = (tree, e.parent))
        end
        ends = Dict(e.call => i for (i, e) in enumerate(evs) if e.kind in (:done, :failed))
        whole &= all(e -> e.kind !== :started || haskey(ends, e.call), evs)
    end
    @test whole                                          # every tree dense, ended by its root, kept as seen
    @test length(children) == n                          # every child call is in exactly one log
    @test all(((tree, parent),) -> parent === nothing ? true : parent == tree, values(children))
    # some were steps of their root (with one thread, a child runs only once its root yields: after its end)
    Threads.nthreads(:default) > 1 && @test count(((tree, parent),) -> parent !== nothing, values(children)) > 0
end

@testset "an observer is not given the calls its own code makes" begin
    f = AIFunction("summarise", "Summarise."; inputs=(message=String,), output=String)
    router = FakeRouter(Any[]; responder=(r, i) -> xml(:result => "ok"))
    by_a, by_b = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
    summariser = e -> e.kind === :done && (Threads.atomic_add!(by_a, 1); f("a summary"))
    echoer = e -> e.kind === :done && (Threads.atomic_add!(by_b, 1); f("an echo"))
    settle_down() = (FunctAI.drain(10); sleep(0.3); FunctAI.drain(10))
    configure!(lm="gpt-4.1-mini", router=router, observers=[summariser])
    try
        f("x")
        settle_down()
        @test by_a[] == 1 && length(router.requests) == 2          # its own call is not given to it: no loop
        # two such observers: each is given the other's call, and not the call that makes, so both stop
        by_a[] = 0
        configure!(observers=[summariser, echoer])
        f("y")
        settle_down()
        @test by_a[] == 2 && by_b[] == 2
        # an observer that only watches is given every call, the observers' calls included
        watched = Threads.Atomic{Int}(0)
        by_a[] = 0
        configure!(observers=[summariser, e -> e.kind === :done && Threads.atomic_add!(watched, 1)])
        f("z")
        settle_down()
        @test by_a[] == 1 && watched[] == 2
    finally
        configure!(lm=nothing, router=nothing, observers=nothing)
    end
end

@testset "with one thread, an observer or a store that yields never slows a call" begin
    script = raw"""
    using FunctAI
    struct Sleepy <: FunctAI.EventStore
        store::FunctAI.MemoryStore
    end
    FunctAI.keep!(s::Sleepy, e::FunctAI.Event) = (sleep(0.3); FunctAI.keep!(s.store, e))
    @program function plain(x::String)::String
        x
    end
    go() = with_settings(() -> plain("x"); observers=[e -> sleep(0.3)], journal=Sleepy(FunctAI.MemoryStore()))
    go(); FunctAI.drain(10)
    t = minimum(@elapsed(go()) for _ in 1:3)
    ok = FunctAI.drain(10)
    println("threads=", Threads.nthreads(:default), "+", Threads.nthreads(:interactive), " seconds=", t, " drained=", ok)
    """
    out = IOBuffer()
    cmd = addenv(`$(Base.julia_cmd()) --threads=1 --startup-file=no --project=$(Base.active_project()) -e $script`,
                 "FUNCTAI_LOG_CALLS" => "0")
    p = run(pipeline(cmd; stdout=out, stderr=out); wait=false)
    timedwait(() -> process_exited(p), 600.0) === :ok || kill(p, Base.SIGKILL)
    text = String(take!(out))
    m = match(r"threads=1\+0 seconds=(\S+) drained=true", text)
    m === nothing && @info "the one-thread process said" text
    @test m !== nothing && parse(Float64, m.captures[1]) < 0.25     # waiting for either would take 0.6 s or more
end

@testset "a position's whole numbers may be written as JSON numbers: 1.0 is 1" begin
    @program function numbered(x::String)::String
        x
    end
    s = stream(numbered, "ok")
    fetch(s)
    e1, e2 = FunctAI.event_json(s.log[1]), FunctAI.event_json(s.log[2])
    e2["writer"], e2["seq"] = 1.0, 2.0
    e2["after"] = Dict{String,Any}("writer" => 1.0, "seq" => 1.0)
    @test FunctAI.event_fault(e2) === nothing
    store = FunctAI.MemoryStore()
    @test FunctAI.keep!(store, e1) === :kept && FunctAI.keep!(store, e2) === :kept
    kept = FunctAI.events_after(store, e1["tree"], Dict("writer" => 1.0, "seq" => 1.0))
    @test length(kept) == 1 && kept[1].seq === 2 && kept[1].after == FunctAI.Position(1, 1)
    follower = FunctAI.Follower(:kept)
    @test FunctAI.receive!(follower, e1) === :kept && FunctAI.receive!(follower, e2) === :kept
    # a count no Int holds passes the schema (its integers are unbounded): refused as malformed, never a crash
    huge = deepcopy(e1)
    huge["seq"] = 2.0^70
    @test FunctAI.event_fault(huge) === nothing
    err = try FunctAI.keep!(FunctAI.MemoryStore(), huge) catch e e end
    @test err isa FunctAI.StoreRefusal && err.code == "event-malformed"
    @test FunctAI.receive!(FunctAI.Follower(:kept), huge) === :malformed
end

@testset "a store's passing error is sent again; a store that hangs holds one send, never more" begin
    f = AIFunction("team", "Which team?"; inputs=(message=String,), output=String)
    store = MissOnce(FunctAI.MemoryStore(), false)
    @test using_fake(() -> f("x"), fake(xml(:result => "billing")); journal=FunctAI.Journal(store; required=true)) == "billing"
    @test FunctAI.finished(store.store, only(FunctAI.trees(store.store)))
    hang = CountingHang(Channel{Nothing}(Inf), Threads.Atomic{Int}(0))
    err = try
        using_fake(() -> f("x"), fake(xml(:result => "billing")); journal=FunctAI.Journal(hang; required=true, retries=3, timeout=0.1))
    catch e
        e
    end
    @test err isa JournalError && err.cause isa FunctAI.NoAnswer
    @test hang.calls[] == 1                        # every resend waited for the one send it had, none started beside it
    foreach(_ -> put!(hang.gate, nothing), 1:16)
    @test FunctAI.drain(10)
end

@testset "a schema pattern PCRE and ECMA-262 would read differently is refused" begin
    @test_throws ErrorException FunctAI.ecma_pattern("^\\d+\$")
    @test_throws ErrorException FunctAI.ecma_pattern("^\\w\$")
    @test FunctAI.ecma_pattern("^[0-9]+\$") isa Regex
    @test FunctAI.ecma_pattern("^a\\\\d\$") isa Regex          # an escaped backslash, then d
end

end
