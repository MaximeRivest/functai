# A call tree's events as data (contract/streaming.md, format 2): one log
# per tree, numbered by its writer; the kept form of a log (what may be kept
# outside the process, by each call's log_content); replaying a form,
# resuming it, following it live across writers; and the rules a store of
# logs keeps (MemoryStore, in memory). Streams, observers and journals
# (journal.jl) are made of these.

const EVENT_FORMAT = 2
const EVENT_KINDS = (:started, :request, :text, :thinking, :tool_call, :tool_result, :retry, :done, :failed)
const ENVELOPE = ("functai_event", "kind", "tree", "writer", "seq", "after", "at", "call", "function")
const KIND_KEYS = Dict{Symbol,Tuple}(
    :started => ("parent", "root", "program", "inputs", "content", "omitted", "saw"), :request => ("request", "model"),
    :text => ("field", "answer", "text"), :thinking => ("text",), :tool_call => ("id", "name", "input", "content"),
    :tool_result => ("id", "name", "output", "content"), :retry => ("reason", "wait", "content"),
    :done => ("value", "content"), :failed => ("error", "content"))

"""
    Position(writer, seq)

An event of a log, named by the writer that numbered it and its `seq`
(contract/streaming.md). Two positions are the same only when both numbers
are; `nothing` stands for "before the first event". `Position(e)` is an
event's own.
"""
struct Position
    writer::Int
    seq::Int
end
Base.show(io::IO, p::Position) = print(io, "Position(", p.writer, ", ", p.seq, ")")
position_json(p::Position) = LMCC.jobj("writer" => p.writer, "seq" => p.seq)
position_json(::Nothing) = nothing
is_count(x) = x isa Integer && !(x isa Bool) && x >= 1
function position_of(x)
    x === nothing && return nothing
    x isa Position && return x
    (x isa AbstractDict && Set(keys(x)) == Set(["writer", "seq"]) && is_count(x["writer"]) && is_count(x["seq"])) ||
        throw(ArgumentError("a position is {\"writer\", \"seq\"} (integers of at least 1) or null, not $(repr(x))"))
    Position(Int(x["writer"]), Int(x["seq"]))
end

"""
    Event

One thing that happened in a call tree (contract/streaming.md, format 2):
its `kind` (`:started`, `:request`, `:text`, `:thinking`, `:tool_call`,
`:tool_result`, `:retry`, `:done`, `:failed`, or a kind a later stage adds),
where it is in its log (`tree`, `writer`, `seq`, `after`: a
[`Position`](@ref) or `nothing`), when it was numbered (`at`), the `call` it
is about and that call's program (`e.function`), and the keys of its kind
(`e.text`, `e.field`, `e.answer`, `e.value`, `e.error`, `e.request`, …).
`FunctAI.event_json(e)` is its JSON form; `FunctAI.Event(json)` reads one.

An event is a value: nothing changes it once made. Its keys are read as
copies (`e.inputs["x"] = …` changes a copy, never the event, a store's log or
what another reader was given), and it holds nothing but its JSON form.
"""
struct Event
    kind::Symbol
    tree::String
    writer::Int
    seq::Int
    after::Union{Nothing,Position}
    at::String
    call::String
    fn::String
    data::JObj          # the keys of its kind (and any a later writer added), in order; never changed once made
    # made here from data nothing else holds (no copy); the one way events are built inside the package
    global owned_event(kind, tree, writer, seq, after, at, call, fn, data::JObj) =
        new(kind, String(tree), writer, seq, after, String(at), String(call), String(fn), data)
end
"An event from its parts; `data` is copied, so the event never changes with it."
Event(kind::Symbol, tree, writer, seq, after, at, call, fn, data::AbstractDict) =
    owned_event(kind, tree, Int(writer), Int(seq), position_of(after), at, call, fn, LMCC.deepcopy_json(JObj(data)))

const EVENT_FIELDS = (:kind, :tree, :writer, :seq, :after, :at, :call)
function Base.getproperty(e::Event, name::Symbol)
    name in EVENT_FIELDS && return getfield(e, name)
    (name === :function || name === :fn) && return getfield(e, :fn)
    data = getfield(e, :data)
    haskey(data, String(name)) || throw(ArgumentError("a $(getfield(e, :kind)) event has no $name; it has $(join(keys(data), ", "))"))
    LMCC.deepcopy_json(data[String(name)])
end
Base.propertynames(e::Event) = (EVENT_FIELDS..., :function, Symbol.(keys(getfield(e, :data)))...)
Base.haskey(e::Event, key::AbstractString) = haskey(getfield(e, :data), key)
Position(e::Event) = Position(e.writer, e.seq)
"A key of an event's kind, read without a copy (inside the package, which never changes it)."
datum(e::Event, key::AbstractString, default=nothing) = get(getfield(e, :data), key, default)

function Base.show(io::IO, e::Event)
    print(io, "Event(:", e.kind, ", ", getfield(e, :fn), ", ", e.writer, "/", e.seq)
    for (k, v) in getfield(e, :data)
        print(io, ", ", k, " = ", repr(v))
    end
    print(io, ")")
end

"A copy of an event with another predecessor (a form of a log sets its own `after`) or other keys of its kind."
relinked(e::Event, after) = owned_event(e.kind, e.tree, e.writer, e.seq, after, e.at, e.call, getfield(e, :fn), getfield(e, :data))
with_data(e::Event, data::JObj) = owned_event(e.kind, e.tree, e.writer, e.seq, e.after, e.at, e.call, getfield(e, :fn), data)

"An event as the contract's JSON (schema/event.schema.json): a new object, which the event does not share."
function event_json(e::Event)
    out = LMCC.jobj("functai_event" => EVENT_FORMAT, "kind" => String(e.kind), "tree" => e.tree, "writer" => e.writer,
                    "seq" => e.seq, "after" => position_json(e.after), "at" => e.at, "call" => e.call, "function" => getfield(e, :fn))
    for (k, v) in getfield(e, :data)
        out[k] = LMCC.deepcopy_json(v)
    end
    out
end

"An event of a format this reader does not know: what its numbers mean is not known."
struct UnknownFormat <: Exception
    format::Any
end
Base.showerror(io::IO, e::UnknownFormat) = print(io, "UnknownFormat: an event of format $(repr(e.format)); this reader knows format $EVENT_FORMAT")

"""
Why a JSON object is not an event of format 2 (it does not pass
`schema/event.schema.json`, read as the schema itself), or `nothing`.
"""
function event_fault(d)
    d isa AbstractDict || return "not an object"
    get(d, "functai_event", nothing) == EVENT_FORMAT || return "not format $EVENT_FORMAT"
    schema_fault("event", d)
end
event_fault(e::Event) = event_fault(event_json(e))

"""
    Event(json::AbstractDict) -> Event

An event read from its JSON form (copied: the event does not share it).
Throws `UnknownFormat` for an event of a format this reader does not know,
and `ArgumentError` for one that is not an event of format 2 (it does not
pass the contract's schema).
"""
function Event(d::AbstractDict)
    fmt = get(d, "functai_event", nothing)
    fmt == EVENT_FORMAT || throw(UnknownFormat(fmt))
    fault = event_fault(d)
    fault === nothing || throw(ArgumentError("not an event of format $EVENT_FORMAT: $fault"))
    data = JObj(String(k) => LMCC.deepcopy_json(v) for (k, v) in d if !(k in ENVELOPE))
    owned_event(Symbol(d["kind"]), d["tree"], Int(d["writer"]), Int(d["seq"]), position_of(d["after"]), d["at"], d["call"],
                d["function"], data)
end
Event(e::Event) = e

"The events of a form, `after` set to the event before each in the form (`last` before the first)."
function relink(events, last::Union{Nothing,Position}=nothing)
    out = Event[]
    for e in events
        push!(out, relinked(e, last))
        last = Position(e)
    end
    out
end

# ------------------------------------------------------------------ replaying

"What a watcher of a form was shown of one call: whether it ended (`:done`, `:failed`, or `nothing`) and each field's text so far."
mutable struct CallState
    ended::Union{Nothing,Symbol}
    fields::OrderedDict{String,String}
end

"""
    LogState

What replaying a form of a log gives after each event (contract/streaming.md,
"Replaying"): for each call started so far, its [`CallState`](@ref); and
whether the log is `finished` (its outermost call ended).
"""
mutable struct LogState
    tree::Union{Nothing,String}
    calls::OrderedDict{String,CallState}
    finished::Bool
end
LogState() = LogState(nothing, OrderedDict{String,CallState}(), false)
Base.copy(s::LogState) = LogState(s.tree, OrderedDict(k => CallState(v.ended, copy(v.fields)) for (k, v) in s.calls), s.finished)

"""
    replay!(state, e) -> state

Apply one event (laws 1 to 4): `started` adds its call; `request` and
`retry` empty the call's fields; `text` adds to its field; `done` and
`failed` end the call. A kind this reader does not know changes nothing.
"""
function replay!(s::LogState, e::Event)
    s.tree === nothing && (s.tree = e.tree)
    e.kind in EVENT_KINDS || return s
    if e.kind === :started
        s.calls[e.call] = CallState(nothing, OrderedDict{String,String}())
        return s
    end
    c = get(s.calls, e.call, nothing)
    c === nothing && throw(ArgumentError("event $(Position(e)): its call's started is not in this form"))
    if e.kind in (:request, :retry)
        empty!(c.fields)
    elseif e.kind === :text
        c.fields[e.field] = get(c.fields, e.field, "") * e.text
    elseif e.kind in (:done, :failed)
        c.ended = e.kind
        e.call == e.tree && (s.finished = true)
    end
    s
end

"""
    replay(events) -> LogState

What a live watcher of this form of a log saw once it had these events.
"""
replay(events) = foldl(replay!, (Event(x) for x in events); init=LogState())

state_json(s::LogState) = LMCC.jobj("calls" => JObj(k => LMCC.jobj("ended" => c.ended === nothing ? nothing : String(c.ended),
                                                                    "fields" => JObj(c.fields)) for (k, c) in s.calls),
                                    "finished" => s.finished)

"""
    StoreRefusal

A store refusing an append, a claim or a read (contract/streaming.md, "The
rules a store keeps"): `code` is `event-malformed`, `event-conflict`,
`event-gap`, `event-after-end`, `event-start` or `event-unknown`; `event`
the position of the event refused (in a batch, the first refused).
"""
struct StoreRefusal <: Exception
    code::String
    event::Union{Nothing,Position}
    msg::String
end
StoreRefusal(code, msg="") = StoreRefusal(code, nothing, msg)
Base.showerror(io::IO, e::StoreRefusal) =
    print(io, "StoreRefusal [", e.code, "]", e.event === nothing ? "" : " at $(e.event)", isempty(e.msg) ? "" : ": $(e.msg)")

"""
    resume(events, after) -> Vector{Event}

What a source holding these events (one form of one log) gives a reader that
has events up to `after` (a [`Position`](@ref), or `nothing`): the events
after it. A source that does not have that event refuses `event-unknown`
(the reader then starts again from the beginning).
"""
function resume(events::AbstractVector, after)
    evs = Event[Event(x) for x in events]
    after === nothing && return evs
    p = position_of(after)
    i = findfirst(e -> Position(e) == p, evs)
    i === nothing && throw(StoreRefusal("event-unknown", "this source has no event $p"))
    evs[i+1:end]
end

# ------------------------------------------------------------------ following

"What a follower holds of one tree: its events (to rewind), and which process gave each."
mutable struct Followed
    held::Vector{Event}
    from::Vector{Any}
end
last_position(t::Followed) = isempty(t.held) ? nothing : Position(last(t.held))

"""
    Follower(form = :kept)

A reader following logs live (contract/streaming.md, "Following a log"),
keeping for each tree the events it holds. `form` is the form it reads:
`:kept` (the kept form, or a view made from it: the same from every source)
or `:live` (the whole log, or a view made from it: it may show values the
kept form lacks, and only the process that gave them can give them again).
Feed it events with [`receive!`](@ref); `LogState(f, tree)` is what it
shows; [`recover!`](@ref) reads again from a source after a loss.
"""
mutable struct Follower
    form::Symbol
    trees::OrderedDict{String,Followed}
    stopped::Bool
end
function Follower(form::Symbol=:kept)
    form in (:kept, :live) || throw(ArgumentError("a follower reads the :kept form or a :live one, not $(repr(form))"))
    Follower(form, OrderedDict{String,Followed}(), false)
end

"""
    receive!(f::Follower, e; from = e.writer) -> Symbol

Take one event received live, and say what it was: `:kept` (the next
event), `:duplicate` or `:stale` (dropped), `:rewind` (a later writer went on
from an event the reader holds: it drops what it holds after that event and
takes this one), `:loss` (events were lost: read again, [`recover!`](@ref)),
or `:unknown_format` (it stops following: nothing after is read). Positions
are compared whole, writer and seq together. `from` is the process that
gave it (a writer number, or `:store`), which decides where it may resume.

An object that says it is of format 2 but is not an event of it (it does
not pass the contract's schema) is `:malformed`: dropped, changing nothing.
Its position cannot be trusted, so the next event shows whether any was
lost.
"""
function receive!(f::Follower, x; from=nothing)
    f.stopped && throw(ArgumentError("this follower stopped at an event of a format it does not know"))
    if !(x isa Event) && !(x isa AbstractDict && get(x, "functai_event", nothing) == EVENT_FORMAT)
        f.stopped = true
        return :unknown_format
    end
    x isa Event || event_fault(x) === nothing || return :malformed
    e = Event(x)
    t = get!(() -> Followed(Event[], Any[]), f.trees, e.tree)
    last = last_position(t)
    writer = last === nothing ? 0 : last.writer
    e.writer < writer && return :stale
    last !== nothing && e.writer == writer && e.seq <= last.seq && return :duplicate
    result = if e.after == last
        :kept
    elseif e.writer > writer && (e.after === nothing || any(h -> Position(h) == e.after, t.held))
        i = e.after === nothing ? 0 : findfirst(h -> Position(h) == e.after, t.held)
        resize!(t.held, i)
        resize!(t.from, i)
        :rewind
    else
        return :loss
    end
    push!(t.held, e)
    push!(t.from, something(from, e.writer))
    result
end

"What the follower shows of a tree: the replay of what it holds."
LogState(f::Follower, tree::AbstractString) = replay(haskey(f.trees, tree) ? f.trees[tree].held : Event[])

"""
Whether a reader may resume after its last event (contract/streaming.md,
"Resuming"): when the source (`from`: `:store` or a writer's process, by its
number) can give every event it holds as it holds it. Always for the kept
form; for a live form, only from the process that gave it every event.
"""
resumes_in_place(f::Follower, t::Followed, from) = f.form === :kept || (from !== :store && all(==(from), t.from))

"""
    recover!(f::Follower, tree, source, from) -> reads

Read a tree again after a loss (contract/streaming.md, "Resuming"): after
the last event it holds when it may resume in place, and from the beginning
when it may not, or when the source answers `event-unknown`. `source` is
what gives the events: a store (`events_after(source, tree, after)`) or a
vector of the events it holds; `from` is `:store` or the writer number of
the process. Returns its reads, each `(after, events)` or `(after,
refusal)`.
"""
function recover!(f::Follower, tree::AbstractString, source, from)
    t = get!(() -> Followed(Event[], Any[]), f.trees, tree)
    reads = Any[]
    got = nothing
    if resumes_in_place(f, t, from)
        after = last_position(t)
        got = try
            source_after(source, tree, after)
        catch err
            err isa StoreRefusal && err.code == "event-unknown" || rethrow()
            err
        end
        push!(reads, (after=after, answer=got))
    end
    if got === nothing || got isa StoreRefusal
        got = source_after(source, tree, nothing)
        push!(reads, (after=nothing, answer=got))
        empty!(t.held)
        empty!(t.from)
    end
    append!(t.held, got)
    append!(t.from, fill(from, length(got)))
    reads
end
source_after(source::AbstractVector, tree, after) = resume(source, after)
source_after(source, tree, after) = events_after(source, tree, after)

# ------------------------------------------------------------------ the kept form

const ERROR_KEYS = ("type", "message", "code")
const PROGRAM_KEYS = ("name", "kind", "module", "version", "signature", "interface", "answer", "saved", "file", "line")
const SAW_KEYS = Set(["call", "steps", "without", "slot", "saw_of"])

"What a form maker keeps of an event of a kind it knows: the keys it knows, and inside the objects it knows, their known members."
function known(e::Event)
    allowed = KIND_KEYS[e.kind]
    data = JObj(k => LMCC.deepcopy_json(v) for (k, v) in getfield(e, :data) if k in allowed)
    haskey(data, "error") && data["error"] isa AbstractDict && (data["error"] = JObj(k => v for (k, v) in data["error"] if k in ERROR_KEYS))
    haskey(data, "program") && data["program"] isa AbstractDict && (data["program"] = JObj(k => v for (k, v) in data["program"] if k in PROGRAM_KEYS))
    if haskey(data, "saw") && data["saw"] isa AbstractVector
        data["saw"] = Any[x isa AbstractDict && issubset(keys(x), SAW_KEYS) ? x : JObj() for x in data["saw"]]
    end
    with_data(e, data)
end

"""
The outputs a `done` event's value holds: the ones the call said (`keep.holds`:
the writer knows its own events), else by its program's kind: an AI
function's answer, a module's outputs.
"""
value_holds(keep::Keep, program) = keep.holds !== nothing ? keep.holds :
    get(program, "kind", "ai") == "ai" ? String[program["answer"]] : collect(keys(keep.outputs))

"""
    kept_event(e, keep::Keep, program) -> Event or nothing

The kept form of one event (contract/streaming.md, "The kept form"), given
which fields of its call are kept and its call's `program` (the `started`
event's: its kind and answer). `nothing`: not in the kept log. It is made
from the event's JSON form only: nothing else of the call goes with it.
"""
function kept_event(e::Event, keep::Keep, program)
    e.kind in EVENT_KINDS || return nothing
    out = known(e)
    whole(keep) && return out
    data = getfield(out, :data)
    kind = e.kind
    if kind === :started
        inputs = JObj(k => v for (k, v) in data["inputs"] if get(keep.inputs, k, false))     # a name it does not know: not kept
        rest = JObj()
        for (k, v) in data
            k == "inputs" && continue
            if k == "content"
                rest["content"] = false
                rest["omitted"] = LMCC.jobj("inputs" => Any[k for (k, x) in keep.inputs if !x],
                                            "outputs" => Any[k for (k, x) in keep.outputs if !x])
            else
                rest[k] = v
            end
        end
        isempty(inputs) || (rest["inputs"] = inputs)
        # the event's own order: inputs where it was
        order = collect(keys(getfield(e, :data)))
        i = findfirst(==("content"), order)
        i === nothing || insert!(order, i + 1, "omitted")
        return with_data(out, JObj(k => rest[k] for k in sort(collect(keys(rest)); by=k -> something(findfirst(==(k), order), length(order) + 1))))
    elseif kind === :request
        return out
    elseif kind === :text
        return get(keep.outputs, data["field"], false) ? out : nothing
    elseif kind === :thinking
        return nothing
    elseif kind === :tool_call
        delete!(data, "input")
    elseif kind === :tool_result
        delete!(data, "output")
    elseif kind === :retry
        delete!(data, "reason")
    elseif kind === :done
        all(k -> get(keep.outputs, k, false), value_holds(keep, program)) && return out
        delete!(data, "value")
    elseif kind === :failed
        data["error"] = error_without_content(data["error"])
    end
    data["content"] = false
    out
end

"""
    kept_log(events, keeps) -> Vector{Event}

The kept form of a whole log: each event's kept form, by its call's `Keep`
(`keeps[call]`), with `after` set to the kept event before it.
"""
function kept_log(events, keeps::AbstractDict)
    evs = Event[Event(x) for x in events]
    programs = Dict(e.call => datum(e, "program") for e in evs if e.kind === :started)
    relink(filter(!isnothing, [kept_event(e, keeps[e.call], get(programs, e.call, JObj())) for e in evs]))
end

# ------------------------------------------------------------------ a store

"""
    FunctAI.EventStore

What keeps call trees' logs for others to read, by the rules every store
keeps (contract/streaming.md, "The rules a store keeps"). A store is a
subtype with three methods:

- `FunctAI.keep!(store, event_or_batch)` appends an [`Event`](@ref) (or a
  vector of them, one step): returns `:kept` or `:duplicate`, or throws a
  [`StoreRefusal`](@ref) with the contract's code. Any other exception is
  "no answer" (the append may have been kept: the writer sends it again),
  except a fault of the store's own code (`MethodError`, `ArgumentError`, …),
  which stops the journal writer, with its cause.
- `FunctAI.claim!(store, tree)` gives a later writer a [`Claim`](@ref).
- `FunctAI.events_after(store, tree, after)` reads the kept events after a
  position (every event after `nothing`), or refuses `event-unknown`.

[`MemoryStore`](@ref) is one, in memory.
"""
abstract type EventStore end

"""
    MemoryStore(name = "memory")

A store of logs in memory, by the rules every store keeps (contract/
streaming.md, "The rules a store keeps"): [`claim!`](@ref), [`keep!`](@ref)
(an event, or a batch), [`events_after`](@ref). Each claim and each append
is one step per log. Every event is checked against the contract's event
schema before anything else, whether it comes as JSON or as an `Event`. A
journal ([`Journal`](@ref)) keeps call trees' logs in it while they are
written:

```julia
store = FunctAI.MemoryStore()
with_settings(journal = FunctAI.Journal(store; required = true)) do
    support("Where is my parcel?")
end
FunctAI.events_after(store, tree, nothing)
```
"""
mutable struct MemoryStore <: EventStore
    name::String
    logs::Dict{String,Vector{Event}}
    writers::Dict{String,Int}
    lock::ReentrantLock
end
MemoryStore(name::AbstractString="memory") = MemoryStore(String(name), Dict{String,Vector{Event}}(), Dict{String,Int}(), ReentrantLock())
Base.show(io::IO, s::MemoryStore) = print(io, "MemoryStore(", repr(s.name), ", ", length(s.logs), " logs)")

"What a claim gives a later writer: its writer number, and the position of the last kept event it numbers on from."
struct Claim
    writer::Int
    after::Position
end

is_end(log, tree) = !isempty(log) && last(log).call == tree && last(log).kind in (:done, :failed)

"Whether the store holds the end of a tree's log (its outermost call's `done` or `failed`)."
finished(s::MemoryStore, tree::AbstractString) = lock(() -> is_end(get(s.logs, tree, Event[]), tree), s.lock)
finished(s, tree::AbstractString) = is_end(events_after(s, tree, nothing), tree)

"""
    claim!(store, tree) -> Claim

A later writer claims an unfinished log: the store gives it a writer number
(one more than any it gave for the log) and the position of the last kept
event, and from then on refuses every event of an earlier writer (fencing).
Refused `event-unknown` for a log it does not have, `event-after-end` for a
finished one.
"""
function claim!(s::MemoryStore, tree::AbstractString)
    lock(s.lock) do
        log = get(s.logs, tree, Event[])
        isempty(log) && throw(StoreRefusal("event-unknown", "no log $tree"))
        is_end(log, tree) && throw(StoreRefusal("event-after-end", "the log $tree is finished"))
        s.writers[tree] = get(s.writers, tree, 1) + 1
        Claim(s.writers[tree], Position(last(log)))
    end
end

"""
An append's event, checked first against the contract's schema (as JSON, or
as an `Event`'s JSON form: a struct built by hand is no certificate), as an
`Event` that shares nothing with what was given.
"""
function checked_event(x)
    json = x isa Event ? event_json(x) : x
    fault = event_fault(json)
    fault === nothing || throw(StoreRefusal("event-malformed", stated_position(json), "not an event of format $EVENT_FORMAT: $fault"))
    Event(json)
end
stated_position(x) = x isa AbstractDict && is_count(get(x, "writer", nothing)) && is_count(get(x, "seq", nothing)) ?
                     Position(Int(x["writer"]), Int(x["seq"])) : nothing

"The answer to one append, on a log and its writer number (checked in the contract's order); the log is changed when kept."
function append_one!(log::Vector{Event}, writer::Int, x, tree=nothing)
    e = checked_event(x)
    p = Position(e)
    tree !== nothing && e.tree != tree && throw(StoreRefusal("event-malformed", p, "an event of another log"))
    a = e.after
    a !== nothing && (e.seq <= a.seq || a.writer > e.writer) &&
        throw(StoreRefusal("event-malformed", p, "its after is not an event before it"))
    !isempty(log) && e.writer != writer &&
        throw(StoreRefusal("event-conflict", p, "writer $(e.writer) is not the log's (writer $writer): fenced, or a number never given"))
    i = findfirst(k -> k.seq == e.seq, log)
    if i !== nothing
        LMCC.canonical_json(event_json(log[i])) == LMCC.canonical_json(event_json(e)) && return :duplicate
        throw(StoreRefusal("event-conflict", p, "another event holds seq $(e.seq): two writers are writing one log"))
    end
    is_end(log, e.tree) && throw(StoreRefusal("event-after-end", p, "the log is finished"))
    last_p = isempty(log) ? nothing : Position(last(log))
    if a != last_p
        (a === nothing ? 0 : a.seq) > (last_p === nothing ? 0 : last_p.seq) &&
            throw(StoreRefusal("event-gap", p, "events are missing before it"))
        throw(StoreRefusal("event-conflict", p, "the log went on another way"))
    end
    isempty(log) && (e.kind !== :started || e.call != e.tree || e.writer != 1) &&
        throw(StoreRefusal("event-start", p, "a log starts with its outermost call's started, from writer 1"))
    push!(log, e)
    :kept
end

"""
    keep!(store, event) -> :kept or :duplicate
    keep!(store, events::AbstractVector) -> :kept or :duplicate

Append an event (or a JSON object) to its tree's log, as the store rules
check it: an event sent again is `:duplicate`; otherwise it throws a
[`StoreRefusal`](@ref) (`event-malformed`, `event-conflict`, `event-gap`,
`event-after-end`, `event-start`). A batch is checked event by event as if
each were appended alone, and kept whole or not at all: `:duplicate` when
every event is one; a refusal names the first event refused. What is kept
is a copy: nothing the caller does to what it gave changes the log.
"""
function keep!(s::MemoryStore, x)
    tree = x isa Event ? x.tree : x isa AbstractDict ? get(x, "tree", nothing) : nothing
    tree isa AbstractString || throw(StoreRefusal("event-malformed", "not an event"))
    lock(s.lock) do
        log = get!(() -> Event[], s.logs, tree)
        try
            append_one!(log, get(s.writers, tree, 1), x)
        finally
            isempty(log) && delete!(s.logs, tree)
        end
    end
end
function keep!(s::MemoryStore, xs::AbstractVector)
    isempty(xs) && return :duplicate
    first_tree = xs[1] isa Event ? xs[1].tree : xs[1] isa AbstractDict ? get(xs[1], "tree", nothing) : nothing
    first_tree isa AbstractString || throw(StoreRefusal("event-malformed", "not an event"))
    lock(s.lock) do
        trial = copy(get(s.logs, first_tree, Event[]))
        writer = get(s.writers, first_tree, 1)
        answers = Symbol[]
        for x in xs
            try
                push!(answers, append_one!(trial, writer, x, first_tree))
            catch err
                err isa StoreRefusal || rethrow()
                p = err.event !== nothing ? err.event : stated_position(x isa Event ? event_json(x) : x)
                throw(StoreRefusal(err.code, p, err.msg))
            end
        end
        isempty(trial) || (s.logs[first_tree] = trial)
        all(==(:duplicate), answers) ? :duplicate : :kept
    end
end

"""
    events_after(store, tree, after) -> Vector{Event}

The kept events of a tree's log after the one `after` names (all of them
after `nothing`). Refused `event-unknown` when the store does not have that
event. Reading is never fenced.
"""
events_after(s::MemoryStore, tree::AbstractString, after) = lock(() -> resume(copy(get(s.logs, tree, Event[])), after), s.lock)

"Every tree the store has a log of."
trees(s::MemoryStore) = lock(() -> collect(keys(s.logs)), s.lock)

"The last writer number the store gave for a tree's log (1 before any claim)."
writer_of(s::MemoryStore, tree::AbstractString) = lock(() -> get(s.writers, tree, 1), s.lock)

"""
    settle(store, tree, event::Position) -> :kept, :not_kept or :another_end

What reading the journal says of an end its writer could not confirm
(`JournalError` `journal-end` with `journal == "unknown"`; contract/
streaming.md): the log holds that event (`:kept`); it does not and the log
is finished (`:another_end`: another writer ended it, and the outcome was
not kept); or it does not and the log is unfinished (`:not_kept`, final
only once the caller has claimed the log).
"""
function settle(s, tree::AbstractString, event::Position)
    log = events_after(s, tree, nothing)
    any(e -> Position(e) == event, log) && return :kept
    is_end(log, tree) ? :another_end : :not_kept
end
