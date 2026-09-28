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
mutable struct ScriptedStore
    store::FunctAI.MemoryStore
    script::Vector{Any}
    trace::Vector{Any}
end
struct Unreachable <: Exception end

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

"""
A writer keeping one AI function's log (the case's events: the tree's only
call) in a journal that fails: the call's steps as a call makes them, its
barriers, its end, what the caller gets and what its record says.
"""
function run_journal_case(c)
    required = c["mode"] == "required"
    events = [Event(e) for e in c["events"]]
    tree = events[1].tree
    j = ScriptedStore(FunctAI.MemoryStore(), Any[c["script"]...], Any[])
    w = FunctAI.LogWriter(FunctAI.Journal(j; required, retries=c["retries"]), tree)
    made, shown = Event[], Int[]
    outcome = nothing
    for (i, e) in enumerate(events)
        if e.call == tree && e.kind in (:done, :failed)
            outcome = e
            break
        end
        push!(made, e)
        push!(shown, e.seq)
        push!(w, e)
        if required && (i == 1 || e.kind === :tool_call) && FunctAI.confirmation(w) !== :confirmed
            # the barrier is not passed: the call stops, its outcome JournalError journal-barrier
            err = FunctAI.barrier_error(e)
            seq = e.seq + 1
            at = "2026-09-28T10:00:$(lpad(seq ÷ 100, 2, '0')).$(lpad((seq % 100) * 10000, 6, '0'))Z"
            outcome = Event(:failed, tree, e.writer, seq, Position(e), at, tree, e.function,
                            FunctAI.LMCC.jobj("error" => FunctAI.error_json(err, true)))
            break
        end
    end
    push!(made, outcome)
    push!(w, outcome)
    status = FunctAI.confirmation(w)
    close(w)
    (!required || status === :confirmed) && push!(shown, outcome.seq)
    error = outcome.kind === :done ? nothing : Dict(k => v for (k, v) in outcome.error if k != "message")
    record = Dict{String,Any}("error" => error)
    settled = nothing
    caller = if required && status !== :confirmed
        err = FunctAI.end_error(status, outcome, outcome.kind === :done ? (done=outcome.value,) : (failed=error,))
        record["journal"] = err.journal
        err.journal == "unknown" && (settled = String(FunctAI.settle(j.store, tree, err.event)))
        Dict("raises" => Dict("type" => "JournalError", "code" => err.code, "journal" => err.journal,
                              "event" => pos_json(err.event), "outcome" => Dict(String(k) => v for (k, v) in pairs(err.outcome))))
    else
        outcome.kind === :done ? Dict("returns" => outcome.value) : Dict("raises" => error)
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
    observer(name) = e -> push!(get!(() -> Event[], seen, name), e)
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
            for n in want["observers"]
                kinds = [e.kind for e in seen[n]]
                @test kinds[1] === :started && kinds[end] === (haskey(want, "refuses") ? :failed : :done)
            end
            for (n, store) in stores
                # a best-effort journal never makes the call wait: its writer may still be sending
                eventually(() -> !(want["journal"] !== nothing && want["journal"]["name"] == n) ||
                                 (length(FunctAI.trees(store)) == 1 && FunctAI.finished(store, only(FunctAI.trees(store)))))
                logged = FunctAI.trees(store)
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

