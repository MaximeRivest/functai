# One call, from its start to its end, the same for an AI function and a
# program: its `started` event, the journal's barrier at a tree's start, the
# program's code, its outcome (a value or an error) decided before the journal
# is asked, its `done` or `failed`, its record, and a required journal's
# answer about its end (contract/streaming.md, "A required journal that does
# not confirm").

started_data(call::Call) = LMCC.jobj("parent" => call.parent, "root" => call.root, "program" => call.program,
                                     "inputs" => copy(call.inputs), "content" => true, "saw" => Any[])

"""
Run `body(call)` as the call's code: its value is what the call returns.
`body` sets what the record keeps (`call.outputs`, `call.returned`); `value`
(what the call returns) goes in its `done` event.
"""
function run_call(body::Function, call::Call)
    t = call.tree
    with(CURRENT_CALL => call, STREAM_OPENING => nothing) do
        start = emit!(call, :started, started_data(call))
        value = try
            call.refusal === nothing || throw(call.refusal)
            if call.parent === nothing && required(t)
                confirmation(t.writer_task) === :confirmed || throw(barrier_error(start))
            end
            call.value = body(call)
        catch err
            conclude(call, unwrap(err))
            rethrow()
        end
        conclude(call, nothing)
        value
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
    terminal = emit!(call, kind, data; native=err === nothing ? call.value : nothing, withhold)
    status = :confirmed
    if outermost && t.writer_task !== nothing
        withhold && (status = confirmation(t.writer_task))
        close(t.writer_task)
    end
    unconfirmed = withhold && status !== :confirmed
    finish_call(call; error=err, journal=unconfirmed ? (status === :refused ? "refused" : "unknown") : nothing)
    unconfirmed && throw(end_error(status, terminal, err === nothing ? (done=call.value,) : (failed=err,)))
    withhold && deliver!(t, terminal)
    nothing
end

"A tool is about to run: with a required journal, the call waits until its events so far are confirmed (streaming.md, barriers)."
function tool_barrier(call::Call, e::Event)
    t = call.tree
    required(t) || return nothing
    confirmation(t.writer_task) === :confirmed || throw(barrier_error(e))
    nothing
end
