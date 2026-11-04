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
    passive::Bool                   # read for its result only (a conversation's turn called plainly): replies come whole
    conv_turn::Any                  # a conversation's turn: (conversation, turn id), or nothing
    answer_shape::Any               # an AI function's answer's shape (how the answer so far is read), or nothing
end
"`s.turn`: the conversation's turn a stream makes (known at once: it is saved before the model is asked)."
Base.getproperty(s::AIStream, name::Symbol) = name === :turn ? stream_turn(s) : getfield(s, name)
function stream_turn(s::AIStream)
    ct = getfield(s, :conv_turn)
    ct === nothing && throw(ArgumentError("this stream is not a conversation's turn"))
    turn(ct[1], ct[2])
end

"The stream starts watching a call (the one it was opened on) in a tree's log."
function attach!(s::AIStream, t::TreeLog, call_id::AbstractString, program=nothing)
    s.outer = String(call_id)
    # a resumed turn's log goes on without showing its call's start again: the stream knows its answer anyway
    program isa AbstractDict && (s.answer = get(program, "answer", nothing))
    push!(s.calls, s.outer)
    push!(t.streams, s)
end

"An event of the tree: the stream takes it when it is about its call or a call inside it."
function offer!(s::AIStream, e::Event)
    mine = e.call in s.calls
    if !mine && e.kind === :started && datum(e, "parent") in s.calls
        push!(s.calls, e.call)
        mine = true
    end
    mine || return
    lock(s.cond) do
        linked = relinked(e, s.last)
        s.last = Position(linked)
        if e.call == s.outer
            e.kind === :started && (s.answer = datum(e, "program")["answer"])
            e.kind in (:request, :retry) && (s.answer_text = "")          # law 3
            e.kind === :text && datum(e, "answer") === true && (s.answer_text *= datum(e, "text"))
        end
        push!(s.log, linked)
        notify(s.cond)
    end
end

function start_stream(thunk; passive::Bool=false, turn=nothing, shape=nothing)
    s = AIStream(Event[], Set{String}(), nothing, Threads.Condition(), false, false, nothing, nothing, "", nothing, passive, turn, shape)
    # a task of its own on any thread (it inherits the caller's scoped settings and call): not
    # pinned to the caller's thread, and it does not pin the caller's task to it
    s.task = Threads.@spawn try
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
LM15.stream(f::AIFunction, args...; kw...) = (inputs = bind_inputs(f, args, kw);
                                              start_stream(() -> predict_inputs(f, inputs); shape=last(interface(f)["outputs"])["shape"]))
LM15.stream(p::AIProgram, args...; kw...) = p.remote === nothing ? start_stream(() -> p(args...; kw...)) : remote_stream(p, args, kw)
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

"The answer's text so far (restarts after a retry). Provisional: `fetch` has the typed value."
text(s::AIStream) = s.answer_text

"""
    answer_so_far(s::AIStream)

The answer so far as a value (contract/streaming.md, *The answer so far*):
the text for a text answer; the JSON read so far for a record or a list
(provisional: a choice inside it shows once complete, never `"med"` for
`"medium"`); `nothing` for other answers until the call is done; the
answer itself once it is.
"""
function answer_so_far(s::AIStream)
    if s.finished && !istaskfailed(s.task)
        r = fetch(s.task)
        return r isa Prediction ? r.value : r
    end
    s.answer_shape === nothing && throw(ArgumentError("this stream is a program's: its value is fetch(s)"))
    answer_so_far(s.answer_text, s.answer_shape)
end

# ------------------------------------------------------------------ the answer so far, read by its shape

const PARTIAL_SPACE = (' ', '\t', '\n', '\r')            # JSON's white space
const PARTIAL_HEX = Set("0123456789abcdefABCDEF")
const PARTIAL_ESCAPES = Dict('"' => '"', '\\' => '\\', '/' => '/', 'b' => '\b', 'f' => '\f', 'n' => '\n', 'r' => '\r', 't' => '\t')
const PARTIAL_NUMBER = r"\A-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\z"
struct NotJSONText <: Exception end
struct NotYet end
const NOT_YET = NotYet()

"""
A shape as the reader follows it: `\$ref` followed; an `anyOf` of one shape
and null read as that shape; any other `anyOf`, or what it cannot follow, `{}`.
"""
function partial_resolve(shape, root)
    seen = Set{String}()
    while shape isa AbstractDict && get(shape, "\$ref", nothing) isa AbstractString
        ref = shape["\$ref"]
        (ref in seen || !startswith(ref, "#/\$defs/") || !(root isa AbstractDict)) && return JObj()
        push!(seen, ref)
        defs = get(root, "\$defs", nothing)
        shape = defs isa AbstractDict ? get(defs, ref[length("#/\$defs/")+1:end], JObj()) : JObj()
    end
    shape isa AbstractDict || return JObj()
    options = get(shape, "anyOf", nothing)
    if options isa AbstractVector
        is_null(o) = o isa AbstractDict && length(o) == 1 && get(o, "type", nothing) == "null"
        rest = [o for o in options if !is_null(o)]
        return length(rest) == 1 && length(options) == 2 ? partial_resolve(rest[1], root) : JObj()
    end
    shape
end

is_choice(shape) = haskey(shape, "enum") || haskey(shape, "const")

function partial_member(shape, key, root)
    props = get(shape, "properties", nothing)
    props isa AbstractDict && haskey(props, key) && return partial_resolve(props[key], root)
    extra = get(shape, "additionalProperties", nothing)
    extra isa AbstractDict ? partial_resolve(extra, root) : JObj()
end

function partial_item(shape, i::Int, root)               # i: the item's 0-based place
    prefix = get(shape, "prefixItems", nothing)
    prefix isa AbstractVector && i < length(prefix) && return partial_resolve(prefix[i+1], root)
    items = get(shape, "items", nothing)
    items isa AbstractDict ? partial_resolve(items, root) : JObj()
end

"(text so far, index after, closed) for the string whose quote is `cs[i]`."
function partial_string(cs::Vector{Char}, i::Int)
    out = IOBuffer()
    n = length(cs)
    i += 1
    while i <= n
        c = cs[i]
        c == '"' && return (String(take!(out)), i + 1, true)
        if c != '\\'
            print(out, c)
            i += 1
            continue
        end
        i + 1 > n && break                                # an escape still being written
        e = cs[i+1]
        if haskey(PARTIAL_ESCAPES, e)
            print(out, PARTIAL_ESCAPES[e])
            i += 2
            continue
        end
        e == 'u' || throw(NotJSONText())
        digits = cs[i+2:min(i + 5, n)]
        all(in(PARTIAL_HEX), digits) || throw(NotJSONText())
        length(digits) < 4 && break
        code = parse(Int, String(digits); base=16)
        if 0xD800 <= code < 0xDC00                        # a pair: wait for its second half
            rest = cs[i+6:min(i + 11, n)]
            if length(rest) < 6
                head = String(rest[1:min(2, length(rest))])
                ((!isempty(rest) && !startswith("\\u", head)) || !all(in(PARTIAL_HEX), rest[min(3, length(rest) + 1):end])) &&
                    throw(NotJSONText())
                break
            end
            (String(rest[1:2]) == "\\u" && all(in(PARTIAL_HEX), rest[3:6])) || throw(NotJSONText())
            low = parse(Int, String(rest[3:6]); base=16)
            0xDC00 <= low < 0xE000 || throw(NotJSONText())
            print(out, Char(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
            i += 12
            continue
        end
        print(out, Char(code))
        i += 6
    end
    (String(take!(out)), n + 1, false)
end

function partial_skip(cs, i)
    while i <= length(cs) && cs[i] in PARTIAL_SPACE
        i += 1
    end
    i
end

"(value or `NOT_YET`, index after, complete)."
function partial_value(cs::Vector{Char}, i::Int, shape, root)
    n = length(cs)
    i = partial_skip(cs, i)
    i > n && return (NOT_YET, i, false)
    c = cs[i]
    if c == '"'
        text, i, closed = partial_string(cs, i)
        !closed && is_choice(shape) && return (NOT_YET, i, false)   # "med" is no value of a choice
        return (text, i, closed)
    elseif c == '{'
        obj = JObj()
        i += 1
        while true
            i = partial_skip(cs, i)
            i > n && return (obj, i, false)
            cs[i] == '}' && return (obj, i + 1, true)
            if cs[i] == ','
                i += 1
                continue
            end
            cs[i] == '"' || throw(NotJSONText())
            key, i, closed = partial_string(cs, i)
            closed || return (obj, i, false)
            i = partial_skip(cs, i)
            i > n && return (obj, i, false)
            cs[i] == ':' || throw(NotJSONText())
            value, i, done = partial_value(cs, i + 1, partial_member(shape, key, root), root)
            value === NOT_YET || (obj[key] = value)
            done || return (obj, i, false)
        end
    elseif c == '['
        items = Any[]
        i += 1
        while true
            i = partial_skip(cs, i)
            i > n && return (items, i, false)
            cs[i] == ']' && return (items, i + 1, true)
            if cs[i] == ','
                i += 1
                continue
            end
            value, i, done = partial_value(cs, i, partial_item(shape, length(items), root), root)
            value === NOT_YET || push!(items, value)
            done || return (items, i, false)
        end
    end
    j = i
    while j <= n && !(cs[j] in (',', ']', '}', PARTIAL_SPACE...))
        j += 1
    end
    j > n && return (NOT_YET, j, false)                   # a number or a word that may still grow: 1 may become 12
    word = String(cs[i:j-1])
    word == "true" && return (true, j, true)
    word == "false" && return (false, j, true)
    word == "null" && return (nothing, j, true)
    occursin(PARTIAL_NUMBER, word) || throw(NotJSONText())
    (any(x -> x in ('.', 'e', 'E'), word) ? parse(Float64, word) : something(tryparse(Int, word), parse(BigInt, word)), j, true)
end

"How an answer of this shape is read so far: `:text`, `:json` or `:none`."
function partial_kind(shape)
    s = partial_resolve(shape, shape)
    t = get(s, "type", nothing)
    types = setdiff(Set{Any}(t isa AbstractVector ? t : Any[t]), Set{Any}(["null"]))
    types == Set{Any}(["string"]) && !is_choice(s) && return :text
    !isempty(types) && issubset(types, Set{Any}(["object", "array"])) && return :json
    :none
end

"""
    answer_so_far(text, shape)

The answer so far, read from its text by its shape (contract/streaming.md,
*The answer so far*): the text for a text answer; the JSON read so far for a
record or a list (complete values; a string still being written as its text
so far, unless it is a choice, which waits for its closing quote; a number,
`true`, `false` or `null` once complete); `nothing` for any other answer, and
when the text is not the start of what the shape gives. For a page that
knows the program (its interface's last output's shape) and reads a call's
`text` events. Provisional: the call's result is the typed, checked value.
"""
function answer_so_far(text::AbstractString, shape)
    kind = partial_kind(shape)
    kind === :text && return String(text)
    kind === :none && return nothing
    partial_json(text, shape)
end

"The JSON value the start of a JSON text says so far, read by `shape` when given; `nothing` when it is not JSON so far."
function partial_json(text::AbstractString, shape=nothing)
    s = strip(text, PARTIAL_SPACE)
    if startswith(s, "```")                               # a fenced block: read after its first line break
        k = findfirst('\n', s)
        s = k === nothing ? "" : s[nextind(s, k):end]
        r = rstrip(s, PARTIAL_SPACE)
        endswith(r, "```") && (s = chop(r; tail=3))
        s = strip(s, PARTIAL_SPACE)
    end
    top = shape === nothing ? JObj() : partial_resolve(shape, shape)
    t = get(top, "type", nothing)
    opens = join(x == "object" ? "{" : x == "array" ? "[" : "" for x in (t isa AbstractVector ? t : Any[t]))
    isempty(opens) && (opens = "[{")
    (isempty(s) || !(first(s) in opens)) && return nothing
    value = try
        first(partial_value(collect(s), 1, top, shape))
    catch err
        err isa NotJSONText || rethrow()
        return nothing
    end
    value === NOT_YET ? nothing : value
end

# ------------------------------------------------------------------ notes (contract/streaming.md, "Notes")

"""
    note(name, data = nothing)

Say that something happened, from code inside a call (a program's code, a
tool): `note("screen captured", (screen = 4, key = "s4.jpg"))`.

Whatever watches the call sees a `note` event (a stream, an observer, a
journal: contract/streaming.md), in its place among the call's events.
`data` is any JSON value (a record is given as its members); the kept form
of a call whose content is not whole keeps the note's `name` and leaves
`data` out. A note is not a value of the call: the call log does not keep
it. Outside any call it does nothing. A resumed turn does not show again the
notes it made before it stopped.

A program may declare its notes (`@program notes = [...]`, `tool(f; notes)`:
[`Note`](@ref)). Then, inside its calls, a note it does not declare, or whose
data does not fit what a program around it declares, throws
[`InterfaceError`](@ref) (code `interface-note`) here, whether or not
anything watches.
"""
function note(name::AbstractString, data=nothing)
    isempty(strip(name)) && throw(ArgumentError("a note's name is a non-empty text: what happened (\"screen captured\")"))
    call = CURRENT_CALL[]
    call === nothing && return nothing
    chain = Any[]                                         # each program's declared notes, outermost first
    c = call
    while c !== nothing
        pushfirst!(chain, declared_notes(c.fn))
        c = c.up
    end
    if any(!isnothing, chain)                             # declared somewhere: checked (contract/programs.md, Notes)
        json = json_form(data)
        why = json === NOJSON ? "its data has no JSON form" : note_refusal(chain, name, json)
        why === nothing || throw(InterfaceError("interface-note", String(name), "$(call.name): the note $(repr(String(name))) is refused: $why"))
        data = json
    end
    emit!(call, :note, LMCC.jobj("name" => String(name), "data" => logvalue(data)))
    nothing
end

function Base.show(io::IO, s::AIStream)
    state = s.finished ? (istaskfailed(s.task) ? "failed" : "done") : s.closed ? "closing" : "running"
    print(io, "AIStream(", state, ", ", length(s.log), " events, answer so far: ", repr(first(s.answer_text, 60)), ")")
end
