# Keeping a log while it is written (contract/streaming.md): observers and
# journals, set where log_calls is (a function's own settings, with_settings
# blocks, configure!); how the layers combine (observers add up; one journal
# per tree, and the host's holds: `journal-policy`); a writer that appends a
# tree's kept events to its journal in order, sends again what is not
# confirmed, and, for a required journal, waits at the three barriers; and
# the per-tree log every stream, observer and journal is fed from.

"""
    Journal(store; required = false, retries = 2)

Where each call tree's kept log is kept while it is written (contract/
streaming.md, "Keeping a log while it is written"): a store that keeps logs
by the store rules ([`MemoryStore`](@ref), or anything with `claim!`,
`keep!` and `events_after`). Best effort (the default) never makes a call
wait. `required = true`: the call waits until its events are confirmed at
its start, before each tool runs, and at its end, and raises
[`JournalError`](@ref) when they are not. An event is sent at most
`1 + retries` times in a row before the writer gives up on it for now.

```julia
with_settings(journal = FunctAI.Journal(store; required = true)) do … end
FunctAI.configure!(journal = store)          # best effort, for every tree
```
"""
struct Journal
    store::Any
    required::Bool
    retries::Int
end
Journal(store; required::Bool=false, retries::Integer=2) = Journal(store, required, Int(retries))
Base.:(==)(a::Journal, b::Journal) = a.store === b.store && a.required == b.required
Base.hash(j::Journal, h::UInt) = hash(objectid(j.store), hash(j.required, h))
mode(j::Journal) = j.required ? "required" : "best-effort"
Base.show(io::IO, j::Journal) = print(io, "Journal(", j.store, j.required ? "; required = true" : "", ")")

"A journal setting: a `Journal`, a store (best effort), or `false` (no journal)."
function journal_setting(v)
    v === false && return false
    v isa Journal && return v
    v isa Bool && throw(ArgumentError("journal is a store, Journal(store; required = true), or false (no journal); not true"))
    applicable(keep!, v, nothing) || hasmethod(keep!, Tuple{typeof(v),Event}) ||
        throw(ArgumentError("journal is a store (keep!, claim!, events_after), Journal(store; …), or false; not $(repr(v))"))
    Journal(v)
end

"""
    JournalError

A call tree's journal breaking (contract/streaming.md): `code` is
`"journal-policy"` (a closer setting replaces or removes a journal the host
set, or weakens a required one: refused before the tree runs),
`"journal-scope"` (a required journal set only inside a tree),
`"journal-barrier"` (a required journal did not confirm the call's start or
a tool call: it did not go on), or `"journal-end"` (it did not confirm the
call's end). A `journal-end` error holds the call's `outcome`, as the call
record does (`(done = value,)` or `(failed = exception,)`), names its
terminal `event` by position in the log `tree` (the outermost call's id),
and says `journal`: `"refused"` or `"unknown"` (no answer:
`FunctAI.settle(store, err.tree, err.event)` reads the journal).
"""
struct JournalError <: Exception
    code::String
    msg::String
    journal::Union{Nothing,String}
    tree::Union{Nothing,String}
    event::Union{Nothing,Position}
    outcome::Union{Nothing,NamedTuple}
end
JournalError(code::AbstractString, msg::AbstractString) = JournalError(String(code), String(msg), nothing, nothing, nothing, nothing)
Base.showerror(io::IO, e::JournalError) = print(io, "JournalError [", e.code, "]: ", e.msg)

# ------------------------------------------------------------------ receivers across layers

"The journal setting of one layer: `UNSET` when it sets none, `false` for \"no journal\", or a `Journal`."
struct Unset end
const UNSET = Unset()

"""
The observers and the journal a tree gets from the layers around its
outermost call (`layers`, closest first: `(where, observers, journal)` with
`where` `:own`, `:block` or `:configure`). Observers add up, outermost
first. The closest journal setting decides, except that a program's own
setting cannot replace or remove a journal a host layer set (it may name
the same one, or make it required), and no closer layer can replace, weaken
or remove a required journal. `refused`: the layers break that
(`journal-policy`); `journal` is then the one the layers farther out than
every refused setting give, where the refused tree's log goes.
"""
function receivers(layers)
    observers = Any[o for l in Iterators.reverse(layers) for o in l.observers]
    refused = refused_layers(layers)
    if !isempty(refused)
        rest = layers[maximum(refused)+1:end]
        return (observers=observers, journal=chosen_journal(rest), refused=true)
    end
    (observers=observers, journal=chosen_journal(layers), refused=false)
end
function chosen_journal(layers)
    for l in layers
        l.journal === UNSET || return l.journal === false ? nothing : l.journal
    end
    nothing
end
same_setting(a, b) = a === false ? b === false : b !== false && a == b
function refused_layers(layers)
    setting = [(i, l.journal) for (i, l) in enumerate(layers) if l.journal !== UNSET]
    out = Set{Int}()
    for (n, (i, far)) in enumerate(setting)
        for (k, near) in setting[1:n-1]
            same_setting(near, far) && continue
            if far !== false && far.required
                push!(out, k)                   # replaces, weakens or removes a required journal
            elseif far !== false && layers[k].where === :own && layers[i].where !== :own &&
                   !(near !== false && near.store === far.store && near.required)
                push!(out, k)                   # a program replaces or removes a host's journal
            end
        end
    end
    sort!(collect(out))
end

"The receiver layers around a program's call, closest first: its own settings, each enclosing block, `configure!`'s."
function receiver_layers(own::AbstractDict{Symbol})
    layer(where, s) = (where=where, observers=Any[something(get(s, :observers, nothing), Any[])...],
                       journal=haskey(s, :journal) && s[:journal] !== nothing ? s[:journal] : UNSET)
    global_now = lock(() -> copy(GLOBAL_SETTINGS), SETTINGS_LOCK)
    [layer(:own, own); [layer(:block, b) for b in Iterators.reverse(SCOPED_LAYERS[])]; layer(:configure, global_now)]
end

# ------------------------------------------------------------------ the writer

"""
    LogWriter(journal, tree)

Appends one tree's kept events to its journal, in order, as they are made
(contract/streaming.md, "Appending"): each event is sent at most
`1 + retries` times in a row; an event the journal answers `:kept` or
`:duplicate` is confirmed; one it refuses stops the writer (it appends
nothing more); one that gets no answer is kept and sent again with the next
event. Sending runs on its own task, so a call never waits for a best-effort
journal; [`confirmation`](@ref) waits for what was sent so far (a required
journal's barrier).
"""
mutable struct LogWriter
    journal::Journal
    tree::String
    pending::Vector{Event}
    refused::Bool
    status::Symbol                  # of the latest round: :confirmed, :unanswered or :refused
    submitted::Int
    processed::Int
    queue::Channel{Event}
    cond::Threads.Condition
    warned::Bool
    task::Union{Nothing,Task}
end
function LogWriter(journal::Journal, tree::AbstractString)
    w = LogWriter(journal, String(tree), Event[], false, :confirmed, 0, 0, Channel{Event}(Inf), Threads.Condition(), false, nothing)
    w.task = @async sender(w)
    w
end

function sender(w::LogWriter)
    for e in w.queue
        status = try
            round!(w, e)
        catch err
            w.refused = true
            warn_journal(w, err)
            :refused
        end
        lock(w.cond) do
            w.status = status
            w.processed += 1
            notify(w.cond)
        end
    end
end

"One round: the event joins what is not confirmed, and the writer sends it all again, in order."
function round!(w::LogWriter, e::Event)
    w.refused && return :refused
    push!(w.pending, e)
    while !isempty(w.pending)
        answered = false
        for _ in 1:(1+w.journal.retries)
            answer = try
                keep!(w.journal.store, w.pending[1])
            catch err
                if err isa StoreRefusal
                    w.refused = true
                    warn_journal(w, err)
                    return :refused
                end
                nothing                                     # no answer: it may have been kept
            end
            answer === nothing && continue
            answered = true
            break
        end
        if !answered
            warn_journal(w, "an event got no answer")
            return :unanswered
        end
        popfirst!(w.pending)
    end
    :confirmed
end

function warn_journal(w::LogWriter, why)
    w.journal.required && return                      # a required journal says so to its caller
    w.warned && return
    w.warned = true
    store = w.journal.store
    name = store isa MemoryStore ? "MemoryStore($(repr(store.name)))" : string(nameof(typeof(store)))
    warn_once("journal:$(objectid(store))",
              "a best-effort journal ($name) did not keep an event ($(why isa Exception ? sprint(showerror, why) : why)); calls go on")
end

"Send one (kept) event to the journal: it joins the queue, in order."
function Base.push!(w::LogWriter, e::Event)
    isopen(w.queue) || return w          # the log ended: nothing follows its end (a call that outlived its tree)
    lock(w.cond) do
        w.submitted += 1
    end
    put!(w.queue, e)
    w
end

"""
    confirmation(w::LogWriter) -> :confirmed, :unanswered or :refused

Wait until every event sent so far has had its round, and say whether all
of them are confirmed.
"""
function confirmation(w::LogWriter)
    lock(w.cond) do
        while w.processed < w.submitted
            wait(w.cond)
        end
        w.refused ? :refused : isempty(w.pending) ? :confirmed : w.status
    end
end
Base.flush(w::LogWriter) = (confirmation(w); nothing)
Base.close(w::LogWriter) = close(w.queue)

"The error a caller gets when a required journal did not confirm its end, holding the call's outcome."
function end_error(status::Symbol, terminal::Event, outcome::NamedTuple)
    word = status === :refused ? "refused" : "unknown"
    JournalError("journal-end", "the journal did not confirm the call's end ($(Position(terminal))): " *
                 (word == "refused" ? "it refused it" : "no answer came; it may be kept (settle it by reading the journal)"),
                 word, terminal.tree, Position(terminal), outcome)
end

barrier_error(e::Event) = JournalError("journal-barrier", "the journal did not keep event $(e.seq)")

# ------------------------------------------------------------------ one tree's log, in this process

"""
The whole log of one call tree, in the process running it: numbered here
(writer 1), given to the streams watching calls in it (the whole form), to
each call's observers (the kept form) and to the tree's journal (the kept
form).
"""
mutable struct TreeLog
    tree::String
    writer::Int
    seq::Int
    last::Union{Nothing,Position}
    at::Float64
    lock::ReentrantLock
    streams::Vector{Any}
    keeps::Dict{String,Keep}
    programs::Dict{String,JObj}
    observers::Dict{String,Vector{Any}}             # each call's observers
    observer_last::IdDict{Any,Union{Nothing,Position}}
    failed_observers::IdDict{Any,Bool}
    journal::Union{Nothing,Journal}
    writer_task::Union{Nothing,LogWriter}
    journal_last::Union{Nothing,Position}
end
TreeLog(tree::AbstractString, journal) =
    TreeLog(String(tree), 1, 0, nothing, 0.0, ReentrantLock(), Any[], Dict{String,Keep}(), Dict{String,JObj}(),
            Dict{String,Vector{Any}}(), IdDict{Any,Union{Nothing,Position}}(), IdDict{Any,Bool}(), journal,
            journal === nothing ? nothing : LogWriter(journal, tree), nothing)

required(t::TreeLog) = t.journal !== nothing && t.journal.required
watched(t::TreeLog, call_observers) = !isempty(t.streams) || t.journal !== nothing || !isempty(call_observers)

"Number an event of a call in this tree's log (`at` never goes back)."
function number!(t::TreeLog, kind::Symbol, call_id::AbstractString, fn::AbstractString, data::JObj; native=nothing)
    t.seq += 1
    t.at = max(t.at, time())
    e = Event(kind, t.tree, t.writer, t.seq, t.last, iso(t.at), String(call_id), String(fn), data, native)
    t.last = Position(e)
    e
end

"""
Make an event and give it to its readers: the streams watching its call, its
call's observers and the journal. With `withhold` (a required journal's
last event), the streams and observers get it only once `release!` says the
journal confirmed it.
"""
function emit!(t::TreeLog, kind::Symbol, call_id, fn, data::JObj; native=nothing, withhold::Bool=false)
    lock(t.lock) do
        e = number!(t, kind, call_id, fn, data; native)
        if kind === :started
            t.programs[e.call] = data["program"]
        end
        if t.writer_task !== nothing
            k = kept_event(e, t.keeps[e.call], t.programs[e.call])
            if k !== nothing
                k = relinked(k, t.journal_last)
                t.journal_last = Position(k)
                push!(t.writer_task, k)
            end
        end
        withhold || deliver!(t, e)
        e
    end
end

"Give an event to the streams watching its call and to its call's observers."
function deliver!(t::TreeLog, e::Event)
    lock(t.lock) do
        for s in t.streams
            offer!(s, e)
        end
        obs = get(t.observers, e.call, Any[])
        isempty(obs) && return
        k = kept_event(e, t.keeps[e.call], t.programs[e.call])
        k === nothing && return
        for o in obs
            get(t.failed_observers, o, false) && continue
            linked = relinked(k, get(t.observer_last, o, nothing))
            t.observer_last[o] = Position(linked)
            try
                o isa Channel ? put!(o, linked) : o(linked)
            catch err
                t.failed_observers[o] = true
                warn_once("observer:$(objectid(o))", "an observer failed ($(sprint(showerror, err))); it is given no more events")
            end
        end
    end
end
