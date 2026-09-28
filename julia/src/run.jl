# One call, from its start to its end, the same for an AI function and a
# program: its `started` event, the journal's barrier at a tree's start, the
# program's code, its outcome (a value or an error) decided before the journal
# is asked, its `done` or `failed`, its record, and a required journal's
# answer about its end (contract/streaming.md, "A required journal that does
# not confirm").

started_data(call::Call) = LMCC.jobj("parent" => call.parent, "root" => call.root, "program" => call.program,
                                     "inputs" => LMCC.deepcopy_json(call.inputs), "content" => true, "saw" => Any[])

"""
Run `body(call)` as the call's code: its value is what the call returns.
`body` sets what the record keeps (`call.outputs`, `call.returned`); `value`
(what the call returns) goes in its `done` event. A stream closed while the
code ran ends the call `Cancelled`, whatever the code returned.
"""
function run_call(body::Function, call::Call)
    t = call.tree
    try
        with(CURRENT_CALL => call, STREAM_OPENING => nothing) do
            start = emit!(call, :started, started_data(call))
            value = try
                call.refusal === nothing || throw(call.refusal)
                if call.parent === nothing && required(t)
                    confirmation(t, start) === :confirmed ||
                        throw(barrier_error(start; cause=journal_cause(t), store=t.journal.store))
                end
                check_cancelled(call)
                v = body(call)
                check_cancelled(call)            # no ordinary return after the stream was closed
                call.value = v
            catch err
                conclude(call, unwrap(err))
                rethrow()
            end
            conclude(call, nothing)
            value
        end
    finally
        release!(call)
    end
end

"A call inside a tree has ended (or never will run): the tree's end no longer waits for it."
function release!(call::Call)
    call.parent === nothing && return
    call.released && return
    call.released = true
    t = call.tree
    lock(t.lock) do
        t.open -= 1
        notify(t.cond)
    end
end

"""
The outermost call waits, before its end, for the calls made inside it that
are still running (a task it started and did not wait for): a log's last
event is its outermost call's end, and every call inside has ended before it.
"""
function wait_for_inside(t::TreeLog, call::Call)
    lock(t.lock) do
        t.open > 0 && warn_once("inside:$(call.program["module"]).$(call.name)",
            "$(call.name) returned while calls made inside it are still running; its call ends once they have " *
            "(to start work that outlives a call, use FunctAI.detached)")
        while t.open > 0
            wait(t.cond)
        end
    end
end

"""
End a call with its outcome (`err`, or its value): its `done` or `failed`,
its record, and, at a tree's end, the journal's answer. With a required
journal the end is given to readers only once confirmed; when it is not,
the caller gets `JournalError` `journal-end` holding the outcome, and the
record says `journal`.
"""
function conclude(call::Call, err)
    t = call.tree
    outermost = call.parent === nothing
    kind, data = err === nothing ? (:done, LMCC.jobj("value" => logvalue(call.value))) :
                                   (:failed, LMCC.jobj("error" => error_json(err, true)))
    withhold = outermost && required(t)
    outermost && wait_for_inside(t, call)
    terminal = emit!(call, kind, data; withhold, last=outermost)
    release!(call)
    status = :confirmed
    if outermost && t.writer_task !== nothing
        withhold && (status = confirmation(t, terminal))
        close(t.writer_task)
    end
    unconfirmed = withhold && status !== :confirmed
    finish_call(call; error=err, journal=unconfirmed ? (status === :refused ? "refused" : "unknown") : nothing)
    unconfirmed && throw(end_error(status, terminal, err === nothing ? (done=call.value,) : (failed=err,);
                                   cause=journal_cause(t), store=t.journal.store))
    withhold && deliver!(t, terminal)
    nothing
end

"""
The model asked for a tool: its `tool_call` event, then, with a required
journal, the barrier (the call waits until its events so far are confirmed,
and the tool does not run when they are not); a closed stream stops it too.
"""
function tool_called!(call::Call, data::JObj)
    t = call.tree
    asked = emit!(call, :tool_call, data)
    if required(t)
        confirmation(t, asked) === :confirmed || throw(barrier_error(asked; cause=journal_cause(t), store=t.journal.store))
    end
    check_cancelled(call)
    asked
end

"""
    FunctAI.detached(f)

Run `f()` outside the call it is in: the calls it makes start trees of their
own (their own log, journal and record root) instead of being steps of that
call. A call waits, before it ends, for every call made inside it; work
meant to outlive it (a task that warms a cache, say) is started detached:

```julia
@program function answer(q::String)::String
    FunctAI.detached() do
        @async warm_up(q)          # not a step of answer: its own tree
    end
    reply(q)
end
```
"""
detached(f) = with(f, CURRENT_CALL => nothing, STREAM_OPENING => nothing)
