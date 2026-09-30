# The contract's events/ cases (contract/streaming.md, format 2), run by the
# Julia implementation: replaying and following logs, their kept form, the
# rules a store keeps, a writer keeping a log in a journal that fails, and
# which observers and journal a tree gets from the layers around it.

import FunctAI: Event, Position, event_json, position_json

pos_json(p) = position_json(p)

"Wait (up to `seconds`) until `cond()` holds: for what a call does not wait for."
function eventually(cond; seconds=5.0)
    deadline = time() + seconds
    while !cond() && time() < deadline
        sleep(0.005)
    end
    cond()
end
positions(evs) = Any[pos_json(Position(e)) for e in evs]
answer_json(r::Symbol) = replace(String(r), "_" => "-")

"A read's answer as the cases write it: the positions given, or the refusal."
read_answer(f) = try
    Dict("events" => positions(f()))
catch e
    e isa FunctAI.StoreRefusal || rethrow()
    Dict("refuses" => e.code)
end

# ------------------------------------------------------------------ a journal that fails, as a script says

"""
A journal whose transport follows a script (cases/README.md, `journal`):
each send meets the script's next word; between sends, another writer may
claim the log, or claim it and end it.
"""
mutable struct ScriptedStore <: FunctAI.EventStore
    store::FunctAI.MemoryStore
    script::Vector{Any}
    trace::Vector{Any}
end
struct Unreachable <: Exception end
# reading is never fenced, and takes no send: the caller settles by reading the journal itself
FunctAI.events_after(j::ScriptedStore, tree, after) = FunctAI.events_after(j.store, tree, after)
FunctAI.claim!(j::ScriptedStore, tree) = FunctAI.claim!(j.store, tree)

function next_word!(j::ScriptedStore, tree)
    while true
        word = isempty(j.script) ? "ok" : popfirst!(j.script)
        word in ("claimed", "ended") || return word
        claim = try
            FunctAI.claim!(j.store, tree)
        catch e
            e isa FunctAI.StoreRefusal || rethrow()
            nothing
        end
        push!(j.trace, Dict("other" => word, "writer" => claim === nothing ? nothing : claim.writer))
        if word == "ended" && claim !== nothing
            # the other writer numbers on from what its claim named, and ends the log: failed, Cancelled
            kept = FunctAI.events_after(j.store, tree, nothing)
            last = kept[end]
            data = FunctAI.LMCC.jobj("error" => FunctAI.LMCC.jobj("type" => "Cancelled"))
            seq = claim.after.seq + 1
            at = "2026-09-28T10:00:$(lpad(50 + seq ÷ 100, 2, '0')).$(lpad((seq % 100) * 10000, 6, '0'))Z"
            name = only(e.function for e in kept if e.kind === :started && e.call == tree)
            FunctAI.keep!(j.store, Event(:failed, tree, claim.writer, seq, claim.after, at, tree, name, data))
        end
    end
end

function FunctAI.keep!(j::ScriptedStore, e::Event)
    word = next_word!(j, e.tree)
    if word == "down"
        push!(j.trace, Dict("seq" => e.seq, "transport" => word, "answer" => nothing))
        throw(Unreachable())
    elseif word == "conflict"
        push!(j.trace, Dict("seq" => e.seq, "transport" => word, "answer" => "event-conflict"))
        throw(FunctAI.StoreRefusal("event-conflict", Position(e), "another writer has the log"))
    end
    answer = try
        String(FunctAI.keep!(j.store, e))
    catch err
        err isa FunctAI.StoreRefusal || rethrow()
        err
    end
    push!(j.trace, Dict("seq" => e.seq, "transport" => word, "answer" => word == "lost" ? nothing : answer isa FunctAI.StoreRefusal ? answer.code : answer))
    word == "lost" && throw(Unreachable())
    answer isa FunctAI.StoreRefusal && throw(answer)
    Symbol(answer)
end

"An error of the type a case names (a provider's, say), with its message."
struct CaseError <: Exception
    type::String
    msg::String
end
FunctAI.error_type(e::CaseError) = e.type

"A case's time as seconds since the epoch (the clock its call is numbered by)."
case_seconds(at) = FunctAI.Dates.datetime2unix(FunctAI.Dates.DateTime(at[1:23])) + parse(Int, at[24:26]) / 1e6

"""
A writer keeping one AI function's log (the case's events: the tree's only
call) in a journal that fails, through the engine: the call starts, runs
and ends as every call does (`run_call`: the start barrier, the outcome
decided before the journal is asked, the end withheld and confirmed, the
record, the caller's error), and its code makes the case's events between
its start and its end, a tool call through the engine's own barrier
(`tool_called!`). Only the call's id and clock are the case's (`SCRIPTED`),
so its events are the case's, byte for byte.
"""
function run_journal_case(c)
    required = c["mode"] == "required"
    events = [Event(e) for e in c["events"]]
    start, outcome = events[1], events[end]
    tree = start.tree
    j = ScriptedStore(FunctAI.MemoryStore(), Any[c["script"]...], Any[])
    journal = FunctAI.Journal(j; required, retries=c["retries"])
    times = Dict(e.seq => case_seconds(e.at) for e in events)
    made = Event[]
    dir = mktempdir()
    program = start.program
    inputs = start.inputs
    fields = (inputs=collect(keys(inputs)), outputs=[program["answer"]], added=String[])
    body = call -> begin
        for e in events[2:end-1]
            data = FunctAI.LMCC.JObj(k => getproperty(e, Symbol(k)) for k in keys(getfield(e, :data)))
            e.kind === :tool_call ? FunctAI.tool_called!(call, data) : FunctAI.emit!(call, e.kind, data)
        end
        outcome.kind === :done ? outcome.value : throw(CaseError(outcome.error["type"], outcome.error["message"]))
    end
    s = with_settings(journal=journal, log_calls=dir) do
        FunctAI.with(FunctAI.SCRIPTED => (id=tree, clock=seq -> times[seq], tap=made)) do
            FunctAI.start_stream(() -> begin
                call = FunctAI.start_call(program, start.function, FunctAI.effective(Dict{Symbol,Any}()), Dict{Symbol,Any}(), inputs, fields)
                FunctAI.run_call(body, call)
            end)
        end
    end
    raised = try
        fetch(s)
        nothing
    catch e
        e
    end
    @test FunctAI.drain(10)                     # a best-effort writer may still be sending when the call has returned
    shown = [e.seq for e in eachevent(s)]
    rec = only(first(FunctAI.read_log(dir)))
    record = Dict{String,Any}("error" => rec["error"] === nothing ? nothing : Dict(k => v for (k, v) in rec["error"] if k != "message"))
    haskey(rec, "journal") && (record["journal"] = rec["journal"])
    settled = nothing
    caller = if raised isa JournalError && raised.code == "journal-end"
        raised.journal == "unknown" && (settled = String(FunctAI.settle(raised)))
        out = haskey(raised.outcome, :done) ? Dict("done" => raised.outcome.done) : Dict("failed" => FunctAI.error_json(raised.outcome.failed, false))
        Dict("raises" => Dict("type" => "JournalError", "code" => raised.code, "journal" => raised.journal,
                              "event" => pos_json(raised.event), "outcome" => out))
    elseif raised === nothing
        Dict("returns" => outcome.value)
    else
        Dict("raises" => FunctAI.error_json(raised, false))
    end
    kept = tree in FunctAI.trees(j.store) ? FunctAI.events_after(j.store, tree, nothing) : Event[]
    got = Dict{String,Any}("log" => Any[event_json(e) for e in made], "trace" => j.trace, "shown" => shown,
                           "kept" => Dict("events" => positions(kept), "finished" => FunctAI.finished(j.store, tree)),
                           "caller" => caller, "record" => record)
    settled === nothing || (got["settled"] = replace(settled, "_" => "-"))
    for k in keys(c["expect"])
        @test same(got[k], c["expect"][k])
    end
    @test Set(keys(got)) == Set(keys(c["expect"]))
end

# ------------------------------------------------------------------ receivers across layers, through settings and a call

"""
One scenario of `receivers-01`, through the settings a user writes: the
program's own (`own`), a `with_settings` block (`block`), `configure!`
(`configure`). The layers give the tree its observers and journal, and a
call made under them is refused `journal-policy` when they break the rules,
its `started` and `failed` going to every layer's observers and to the
host's journal.
"""
function run_receivers_scenario(scenario)
    stores = Dict{String,FunctAI.MemoryStore}()
    journal(x) = x === nothing ? false :
                 FunctAI.Journal(get!(() -> FunctAI.MemoryStore(x["name"]), stores, x["name"]); required=x["mode"] == "required")
    seen = Dict{String,Vector{Event}}()
    seen_lock = ReentrantLock()              # each observer runs on a task of its own, on any thread: one Dict, one lock
    observer(name) = e -> lock(() -> push!(get!(() -> Event[], seen, name), e), seen_lock)
    names = IdDict{Any,String}()
    settings(layer) = begin
        out = Dict{Symbol,Any}()
        if haskey(layer, "observers")
            fs = [observer(n) for n in layer["observers"]]
            for (fn, n) in zip(fs, layer["observers"])
                names[fn] = n
            end
            out[:observers] = fs
        end
        haskey(layer, "journal") && (out[:journal] = journal(layer["journal"]))
        haskey(layer, "program_observers") && (out[:program_observers] = layer["program_observers"])
        out
    end
    layer(where) = (i = findfirst(l -> l["where"] == where, scenario["layers"]); i === nothing ? Dict{Symbol,Any}() : settings(scenario["layers"][i]))
    own, block, configured = layer("own"), layer("block"), layer("configure")
    f = AIFunction("team", "Which team handles this?"; inputs=(message=String,), output=String, own...)
    FunctAI.configure!(; configured...)
    try
        with_settings(; block...) do
            got = FunctAI.receivers(FunctAI.receiver_layers(f.own))
            observers = [names[o] for o in got.observers]
            j = got.journal === nothing ? nothing : Dict("name" => got.journal.store.name, "mode" => FunctAI.mode(got.journal))
            want = scenario["expect"]
            @test observers == want["observers"]
            @test same(j, want["journal"])
            @test got.refused == haskey(want, "refuses")
            # a call made under these layers
            r = FakeRouter(Any["<result>\nbilling\n</result>"])
            err = refusal_of(() -> with_settings(() -> f("I was charged twice."); lm="gpt-4.1-mini", router=r))
            if haskey(want, "refuses")
                @test err isa JournalError && err.code == want["refuses"]
                @test isempty(r.requests)                                   # the tree did not run
            else
                @test err === nothing
            end
            @test FunctAI.drain(10)                       # observers get events on their own tasks
            for n in want["observers"]
                kinds = [e.kind for e in seen[n]]
                @test kinds[1] === :started && kinds[end] === (haskey(want, "refuses") ? :failed : :done)
            end
            for (n, store) in stores
                logged = FunctAI.trees(store)          # drained: a best-effort writer has sent what it had
                if want["journal"] !== nothing && want["journal"]["name"] == n
                    log = FunctAI.events_after(store, only(logged), nothing)
                    @test haskey(want, "refuses") ? [e.kind for e in log] == [:started, :failed] &&
                                                    log[end].error["code"] == "journal-policy" : log[end].kind === :done
                else
                    @test isempty(logged)
                end
            end
        end
    finally
        FunctAI.configure!(observers=nothing, journal=nothing)
    end
end

@testset "the contract's events" begin

@testset "events case $name" for (name, c) in cases("events")
    kind = c["kind"]
    if kind == "replay"
        state = FunctAI.LogState()
        views = Any[Dict("calls" => FunctAI.state_json(FunctAI.replay!(state, Event(e)))["calls"]) for e in c["events"]]
        @test same(Dict("views" => views, "finished" => state.finished), c["expect"])
        for r in c["resume"]
            @test same(read_answer(() -> FunctAI.resume(c["events"], r["after"])), r["expect"])
        end

    elseif kind == "follow"
        recover = get(c, "recover", nothing)
        f = FunctAI.Follower(recover === nothing ? :kept : Symbol(recover["reader"]))
        results = String[]
        for e in c["received"]
            r = FunctAI.receive!(f, e)          # a live reader got each event from its writer's process
            push!(results, answer_json(r))
            r === :unknown_format && break
        end
        @test results == c["expect"]["results"]
        states = Dict(tree => FunctAI.state_json(FunctAI.LogState(f, tree)) for tree in keys(f.trees))
        @test same(states, c["expect"]["state"])
        if recover !== nothing
            tree = only(keys(f.trees))
            from = recover["from"] == "store" ? :store : Int(recover["from"])
            reads = FunctAI.recover!(f, tree, recover["source"], from)
            got = Any[Dict("after" => pos_json(r.after), "expect" => r.answer isa FunctAI.StoreRefusal ?
                           Dict("refuses" => r.answer.code) : Dict("events" => positions(r.answer))) for r in reads]
            @test same(got, c["expect"]["recover"]["reads"])
            @test same(FunctAI.state_json(FunctAI.LogState(f, tree)), c["expect"]["recover"]["state"])
        end

    elseif kind == "kept"
        keeps = Dict(call => FunctAI.Keep(FunctAI.OrderedDict{String,Bool}(k["inputs"]), FunctAI.OrderedDict{String,Bool}(k["outputs"]))
                     for (call, k) in c["kept"])
        @test same(Any[event_json(e) for e in FunctAI.kept_log(c["events"], keeps)], c["expect"]["events"])

    elseif kind == "store"
        store = FunctAI.MemoryStore()
        for step in c["steps"]
            got = if haskey(step, "append")
                try
                    String(FunctAI.keep!(store, step["append"]))
                catch e
                    e isa FunctAI.StoreRefusal || rethrow()
                    e.code
                end
            elseif haskey(step, "batch")
                try
                    String(FunctAI.keep!(store, Any[step["batch"]...]))
                catch e
                    e isa FunctAI.StoreRefusal || rethrow()
                    Dict("refuses" => e.code, "event" => pos_json(e.event))
                end
            else
                try
                    claim = FunctAI.claim!(store, step["claim"])
                    Dict("writer" => claim.writer, "after" => pos_json(claim.after))
                catch e
                    e isa FunctAI.StoreRefusal || rethrow()
                    Dict("refuses" => e.code)
                end
            end
            @test same(got, step["expect"])
        end
        for r in c["reads"]
            @test same(read_answer(() -> FunctAI.events_after(store, r["tree"], r["after"])), r["expect"])
        end
        logs = Dict(tree => Dict("events" => positions(FunctAI.events_after(store, tree, nothing)),
                                 "writer" => FunctAI.writer_of(store, tree), "finished" => FunctAI.finished(store, tree))
                    for tree in FunctAI.trees(store))
        @test same(logs, c["expect"]["logs"])

    elseif kind == "journal"
        if c["mode"] == "best-effort" && any(!=("ok"), c["script"])
            @test_logs (:warn, r"best-effort journal") match_mode = :any run_journal_case(c)   # it warns, and the call goes on
        else
            run_journal_case(c)
        end

    elseif kind == "receivers"
        for (i, scenario) in enumerate(c["scenarios"])
            @testset "scenario $i" begin
                run_receivers_scenario(scenario)
            end
        end
    end
end

end

