# Streaming (contract/streaming.md): the same call, watched while it is made.
# It retries, runs tools, logs and ends exactly as calling the function does;
# the stream only shows it.
#
#     for piece in stream(haiku, "the first snow")
#         print(piece)
#     end
#     s = stream(solve, "10 pencils?"); foreach(println, eachevent(s)); fetch(s)

"""
    AIStream

A call being made and watched: iterate it for the answer's text as it is
written; [`eachevent`](@ref) for everything (format 2 events: the calls
inside it, requests, reasoning, tool calls, retries); `fetch` for its
value, typed (the same as calling); `close` to cancel it; `FunctAI.text(s)`
for the answer so far. It shows the whole log of its call's tree from that
call down, with the tree's numbers (law 7: a stream opened on a call inside
a tree starts at that call, its first `after` is `nothing`).
"""
mutable struct AIStream
    log::Vector{Event}
    calls::Set{String}
    last::Union{Nothing,Position}
    cond::Threads.Condition
    finished::Bool
    closed::Bool
    outer::Union{Nothing,String}
    answer::Union{Nothing,String}
    answer_text::String
    task::Union{Nothing,Task}
end

"The stream starts watching a call (the one it was opened on) in a tree's log."
function attach!(s::AIStream, t::TreeLog, call_id::AbstractString)
    s.outer = String(call_id)
    push!(s.calls, s.outer)
    push!(t.streams, s)
end

"An event of the tree: the stream takes it when it is about its call or a call inside it."
function offer!(s::AIStream, e::Event)
    mine = e.call in s.calls
    if !mine && e.kind === :started && get(e.data, "parent", nothing) in s.calls
        push!(s.calls, e.call)
        mine = true
    end
    mine || return
    lock(s.cond) do
        linked = relinked(e, s.last)
        s.last = Position(linked)
        if e.call == s.outer
            e.kind === :started && (s.answer = e.data["program"]["answer"])
            e.kind in (:request, :retry) && (s.answer_text = "")          # law 3
            e.kind === :text && e.data["answer"] === true && (s.answer_text *= e.data["text"])
        end
        push!(s.log, linked)
        notify(s.cond)
    end
end

function start_stream(thunk)
    s = AIStream(Event[], Set{String}(), nothing, Threads.Condition(), false, false, nothing, nothing, "", nothing)
    s.task = @async try
        with(thunk, STREAM_OPENING => s)
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
the end (`:done` or `:failed`). A new iteration starts from the first. Each is a format 2
[`Event`](@ref): `FunctAI.event_json(e)` is what a page or another process
reads.

```julia
s = stream(support, "Where is order A-1042?")
for e in eachevent(s)
    println(e.seq, " ", e.kind, " ", e.function)
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
