# Streaming (contract/streaming.md): the same call, watched while it is made.
# It retries, runs tools, logs and ends exactly as calling the function does;
# the stream only shows it.
#
#     for piece in stream(haiku, "the first snow")
#         print(piece)
#     end
#     s = stream(solve, "10 pencils?"); foreach(println, eachevent(s)); fetch(s)

"""
    Event

One thing that happened in a watched call (contract/streaming.md): `kind`
(`:started`, `:text`, `:thinking`, `:tool_call`, `:tool_result`, `:retry`,
`:done`, `:failed`), the `call` it is about, the `function`'s name, and the
fields of its kind (`e.text`, `e.field`, `e.answer`, `e.value`, `e.error`, …).
`FunctAI.event_json(e)` is its JSON form.
"""
struct Event
    kind::Symbol
    call::String
    fn::String
    data::JObj
end
function Base.getproperty(e::Event, name::Symbol)
    name in (:kind, :call, :data) && return getfield(e, name)
    name === :function && return getfield(e, :fn)
    haskey(getfield(e, :data), String(name)) || throw(ArgumentError("a $(getfield(e, :kind)) event has no $name; it has $(join(keys(getfield(e, :data)), ", "))"))
    getfield(e, :data)[String(name)]
end
Base.propertynames(e::Event) = (:kind, :call, :function, Symbol.(keys(getfield(e, :data)))...)
function Base.show(io::IO, e::Event)
    print(io, "Event(:", e.kind, ", ", getfield(e, :fn))
    for (k, v) in getfield(e, :data)
        k == "julia_value" && continue
        print(io, ", ", k, " = ", repr(v))
    end
    print(io, ")")
end

"An event as the contract's JSON (schema/event.schema.json)."
function event_json(e::Event)
    out = LMCC.jobj("kind" => String(e.kind), "call" => e.call, "function" => getfield(e, :fn))
    for (k, v) in getfield(e, :data)
        k == "julia_value" && continue
        out[k] = v
    end
    out
end

"""
    AIStream

A call being made and watched: iterate it for the answer's text as it is
written; [`eachevent`](@ref) for everything (the calls inside it, reasoning,
tool calls, retries); `fetch` for its value, typed (the same as calling);
`close` to cancel it; `FunctAI.text(s)` for the answer so far.
"""
mutable struct AIStream <: Watch
    log::Vector{Event}
    cond::Threads.Condition
    finished::Bool
    closed::Bool
    outer::Union{Nothing,String}
    answer_text::String
    task::Union{Nothing,Task}
end

watch_closed(s::AIStream) = s.closed
watch_check(s::AIStream) = s.closed ? throw(Cancelled()) : nothing

function push_event!(s::AIStream, e::Event)
    lock(s.cond) do
        push!(s.log, e)
        notify(s.cond)
    end
end

event(call::Call, kind::Symbol, data::JObj) = Event(kind, call.id, string(call.program()["name"]), data)

function watch_started(s::AIStream, call, inputs)
    s.outer === nothing && (s.outer = call.id)
    push_event!(s, event(call, :started, LMCC.jobj("parent" => call.parent,
                                                   "inputs" => JObj(String(k) => logvalue(v) for (k, v) in pairs(inputs)))))
end

function on_text(s::AIStream, call, field, text)
    answer = field == call.program()["answer"]
    call.id == s.outer && answer && (s.answer_text *= text)
    push_event!(s, event(call, :text, LMCC.jobj("field" => field, "answer" => answer, "text" => text)))
end
on_thinking(s::AIStream, call, text) = push_event!(s, event(call, :thinking, LMCC.jobj("text" => text)))
on_tool_call(s::AIStream, call, id, name, input) =
    push_event!(s, event(call, :tool_call, LMCC.jobj("id" => id, "name" => name, "input" => input)))
function on_tool_result(s::AIStream, call, id, name, output)
    call.id == s.outer && (s.answer_text = "")
    push_event!(s, event(call, :tool_result, LMCC.jobj("id" => id, "name" => name, "output" => output)))
end
function on_retry(s::AIStream, call, reason, wait)
    call.id == s.outer && (s.answer_text = "")
    push_event!(s, event(call, :retry, LMCC.jobj("reason" => reason, "wait" => wait)))
end
function watch_ended(s::AIStream, call, value, err=nothing)
    if err === nothing
        push_event!(s, event(call, :done, LMCC.jobj("value" => logvalue(value), "julia_value" => value)))
    else
        push_event!(s, event(call, :failed, LMCC.jobj("error" => error_json(err, true))))
    end
end

function start_stream(thunk)
    s = AIStream(Event[], Threads.Condition(), false, false, nothing, "", nothing)
    s.task = @async try
        with(thunk, WATCHING => s)
    finally
        lock(s.cond) do
            s.finished = true
            notify(s.cond)
        end
    end
    s
end
"""
    stream(f, args...; kw...) -> AIStream
    stream(on_piece, f, args...; kw...) -> the value

Call `f` and watch its answer being written. Iterate the stream for the
answer's text piece by piece; `fetch(s)` gives the typed value, the same as
calling `f`. The `do` form hands each piece to `on_piece` and returns the value:

```julia
answer = stream(haiku, "the first snow") do piece
    print(piece)
end
```
"""
LM15.stream(f::AIFunction, args...; kw...) = (inputs = bind_inputs(f, args, kw); start_stream(() -> predict_inputs(f, inputs)))
LM15.stream(p::AIProgram, args...; kw...) = start_stream(() -> p(args...; kw...))
function LM15.stream(on_piece::Function, f::Union{AIFunction,AIProgram}, args...; kw...)
    s = LM15.stream(f, args...; kw...)
    try
        for piece in s
            on_piece(piece)
        end
        fetch(s)
    finally
        close(s)
    end
end

"""
    eachevent(s::AIStream)

An iterator over every event of the watched call, in order, as they happen:
the calls inside it, text, thinking, tool calls and results, retries, and
the end (`:done` or `:failed`). A new iteration starts from the first.

```julia
s = stream(support, "Where is order A-1042?")
for e in eachevent(s)
    println(e.kind, " ", e.function)
end
```
"""
eachevent(s::AIStream) = EventIterator(s)
LM15.events(s::AIStream) = EventIterator(s)
struct EventIterator
    s::AIStream
end
Base.IteratorSize(::Type{EventIterator}) = Base.SizeUnknown()
Base.eltype(::Type{EventIterator}) = Event
function Base.iterate(it::EventIterator, i::Int=1)
    s = it.s
    lock(s.cond) do
        while i > length(s.log) && !s.finished
            wait(s.cond)
        end
        i > length(s.log) ? nothing : (s.log[i], i + 1)
    end
end

"The answer's text, piece by piece (a retry's discarded text too: `fetch` has the value)."
Base.IteratorSize(::Type{AIStream}) = Base.SizeUnknown()
Base.eltype(::Type{AIStream}) = String
function Base.iterate(s::AIStream, i::Int=1)
    it = EventIterator(s)
    while true
        next = iterate(it, i)
        next === nothing && return nothing
        e, i = next
        e.kind === :text && e.answer && e.call == s.outer && return (e.text, i)
    end
end

"The call's value (the same as calling the function); throws its error."
function Base.fetch(s::AIStream)
    try
        r = fetch(s.task)
        r isa Prediction ? r.value : r
    catch err
        throw(unwrap(err))
    end
end
Base.wait(s::AIStream) = (fetch(s); nothing)

"The whole prediction, when the call ends."
prediction(s::AIStream) = try
    fetch(s.task)
catch err
    throw(unwrap(err))
end

"Stop the call: no new request starts, and it ends with `Cancelled`."
Base.close(s::AIStream) = (s.closed = true; nothing)
Base.isopen(s::AIStream) = !s.finished && !s.closed

"The answer so far (restarts after a retry). Provisional: `fetch` has the typed value."
text(s::AIStream) = s.answer_text

function Base.show(io::IO, s::AIStream)
    state = s.finished ? (istaskfailed(s.task) ? "failed" : "done") : s.closed ? "closing" : "running"
    print(io, "AIStream(", state, ", ", length(s.log), " events, answer so far: ", repr(first(s.answer_text, 60)), ")")
end
