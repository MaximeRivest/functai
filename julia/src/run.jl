# One call, from its start to its end, the same for an AI function and a
# program: its `started` event, the journal's barrier at a tree's start, the
# program's code, its outcome (a value or an error) decided before the journal
# is asked, its `done` or `failed`, its record, and a required journal's
# answer about its end (contract/streaming.md, "A required journal that does
# not confirm").

function started_data(call::Call)
    data = LMCC.jobj("parent" => call.parent, "root" => call.root, "program" => call.program,
                     "inputs" => LMCC.deepcopy_json(call.inputs), "content" => true, "saw" => LMCC.deepcopy_json(call.saw))
    call.invocation === nothing || (data["invocation"] = call.invocation)
    data
end

"""
Run `body(call)` as the call's code: its value is what the call returns.
`body` sets what the record keeps (`call.outputs`, `call.returned`); `value`
(what the call returns) goes in its `done` event. A stream closed while the
code ran, or while the call waited for the calls made inside it, ends the
call `Cancelled`, whatever the code returned.
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
                e = unwrap(err)
                if e isa TurnWaiting
                    # a turn stopped to wait for a person: nothing ended; its log stays unfinished, for the
                    # process that resumes it (contract/tools.md), and no record says the call failed
                    call.parent === nothing && t.writer_task !== nothing && close(t.writer_task)
                    rethrow()
                end
                ended_with = conclude(call, e)
                turn_call_ended(call, ended_with)
                rethrow()
            end
            ended_with = conclude(call, nothing)
            turn_call_ended(call, ended_with)
            ended_with === nothing || throw(ended_with)      # closed while it waited for the calls inside it
            value
        end
    finally
        release!(call)
    end
end

"A call of a conversation's turn ended: its tokens count in the turn's usage; a helper the conversation remembers keeps its call."
turn_call_ended(call::Call, err) = call.turn_run === nothing || call_ended!(call.turn_run, call, err)

"A call has ended (or never will run): the call it runs inside no longer waits for it."
function release!(call::Call)
    up = call.up
    up === nothing && return
    t = call.tree
    lock(t.lock) do
        call.released && return
        call.released = true
        up.open -= 1
        notify(t.cond)
    end
end

"""
Number a call's end (`kind`, `data`) once every call made inside it has
ended, in the same hold of the tree's lock as the last look: no call can
join it between the two (one that starts later starts a tree of its own).
Every call does this, not only the outermost, so a call's end follows the
ends of all the calls inside it, at every depth, and a stream watching it
has every event it will ever have once it has the end. A stream closed while
it waited ends an ordinary return `Cancelled`: `(terminal, err)`, `err` the
outcome the end says.
"""
function end_call!(call::Call, err, done_data; withhold::Bool)
    t = call.tree
    if lock(() -> call.open > 0, t.lock)
        warn_once("inside:$(call.program["module"]).$(call.name)",
            "$(call.name) returned while calls made inside it are still running; its call ends once they have " *
            "(to start work that outlives a call, use FunctAI.detached)")
    end
    lock(t.lock) do
        while call.open > 0
            wait(t.cond)
        end
        err === nothing && cancelled(call) && (err = Cancelled())
        kind, data = err === nothing ? (:done, done_data) : (:failed, LMCC.jobj("error" => error_json(err, true)))
        terminal = emit!(call, kind, data; withhold, last=call.parent === nothing)
        call.ended = true
        (terminal, err)
    end
end

"""
End a call with its outcome (`err`, or its value): its `done` or `failed`,
once the calls made inside it have ended, its record, and, at a tree's end,
the journal's answer. With a required journal the end is given to readers
only once confirmed; when it is not, the caller gets `JournalError`
`journal-end` holding the outcome, and the record says `journal`. Returns
the error the call ended with (`Cancelled` for a value its stream was
closed on while it waited), or `nothing`.
"""
function conclude(call::Call, err)
    t = call.tree
    outermost = call.parent === nothing
    done_data = err === nothing ? LMCC.jobj("value" => logvalue(call.value)) : nothing
    withhold = outermost && required(t)
    terminal, err = end_call!(call, err, done_data; withhold)
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
    err
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
detached(f) = with(f, CURRENT_CALL => nothing, STREAM_OPENING => nothing, TURN_STARTING => nothing)
