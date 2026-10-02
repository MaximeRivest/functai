# Keeping a log while it is written (contract/streaming.md): observers and
# journals, set where log_calls is (a function's own settings, with_settings
# blocks, configure!); how the layers combine (observers add up; one journal
# per tree, and the host's holds: `journal-policy`); a writer that appends a
# tree's kept events to its journal in order, sends again what is not
# confirmed, and, for a required journal, waits at the three barriers; and
# the per-tree log every stream, observer and journal is fed from.

"""
    Journal(store; required = false, retries = 2, timeout = 30)

Where each call tree's kept log is kept while it is written (contract/
streaming.md, "Keeping a log while it is written"): a store that keeps logs
by the store rules ([`MemoryStore`](@ref), or any [`EventStore`](@ref)).
Best effort (the default) never makes a call wait: the store's code runs on
a task of its own, as an observer's does (Julia's tasks share threads: see
[`FunctAI.drain`](@ref) for a store that holds its thread without
yielding). `required = true`: the call waits until its events are confirmed
at its start, before each tool runs, and at its end, and raises
[`JournalError`](@ref) when they are not.
An event is sent at most `1 + retries` times in a row before the writer
gives up on it for now; a send the store has not answered after `timeout`
seconds (`nothing`: no limit) counts as no answer.

```julia
with_settings(journal = FunctAI.Journal(store; required = true)) do … end
FunctAI.configure!(journal = store)          # best effort, for every tree
```
"""
struct Journal
    store::Any
    required::Bool
    retries::Int
    timeout::Union{Nothing,Float64}
end
function Journal(store; required::Bool=false, retries::Integer=2, timeout::Union{Nothing,Real}=30.0)
    retries >= 0 || throw(ArgumentError("retries is a whole number of at least 0, not $retries"))
    timeout === nothing || timeout > 0 || throw(ArgumentError("timeout is a number of seconds above 0, or nothing, not $timeout"))
    is_store(store) || throw(ArgumentError("a journal keeps logs in a store (an EventStore: keep!, claim!, events_after), not $(repr(store))"))
    Journal(store, required, Int(retries), timeout === nothing ? nothing : Float64(timeout))
end
Base.:(==)(a::Journal, b::Journal) = a.store === b.store && a.required == b.required
Base.hash(j::Journal, h::UInt) = hash(objectid(j.store), hash(j.required, h))
mode(j::Journal) = j.required ? "required" : "best-effort"
Base.show(io::IO, j::Journal) = print(io, "Journal(", j.store, j.required ? "; required = true" : "", ")")

"A store: an `EventStore`, or anything that can keep an event (`keep!(store, ::Event)`)."
is_store(v) = v isa EventStore || hasmethod(keep!, Tuple{typeof(v),Event})

"A journal setting: a `Journal`, a store (best effort), or `false` (no journal)."
function journal_setting(v)
    v === false && return false
    v isa Journal && return v
    v isa Bool && throw(ArgumentError("journal is a store, Journal(store; required = true), or false (no journal); not true"))
    is_store(v) || throw(ArgumentError("journal is a store (an EventStore: keep!, claim!, events_after), Journal(store; …), or false; not $(repr(v))"))
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
[`settle`](@ref)`(err)` reads the journal). `cause` is what the journal
met, when it met something (a store's refusal, an error, a send it never
answered); `store` is the journal's store.
"""
struct JournalError <: Exception
    code::String
    msg::String
    journal::Union{Nothing,String}
    tree::Union{Nothing,String}
    event::Union{Nothing,Position}
    outcome::Union{Nothing,NamedTuple}
    cause::Any
    store::Any
end
JournalError(code::AbstractString, msg::AbstractString; cause=nothing, store=nothing) =
    JournalError(String(code), String(msg), nothing, nothing, nothing, nothing, cause, store)
function Base.showerror(io::IO, e::JournalError)
    print(io, "JournalError [", e.code, "]: ", e.msg)
    e.cause === nothing || print(io, " (the journal met: ", e.cause isa Exception ? sprint(showerror, e.cause) : string(e.cause), ")")
end

# ------------------------------------------------------------------ receivers across layers

"The journal setting of one layer: `UNSET` when it sets none, `false` for \"no journal\", or a `Journal`."
struct Unset end
const UNSET = Unset()

"""
The observers and the journal a tree gets from the layers around its
outermost call (`layers`, closest first: `(where, observers, journal)` with
`where` `:own`, `:block` or `:configure`). Observers add up, outermost
first, less those whose own code is making the call (`OBSERVING`). The
closest journal setting decides, except that a program's own
setting cannot replace or remove a journal a host layer set (it may name
the same one, or make it required), and no closer layer can replace, weaken
or remove a required journal. `refused`: the layers break that
(`journal-policy`); `journal` is then the one the layers farther out than
every refused setting give, where the refused tree's log goes.
"""
function receivers(layers)
    busy = OBSERVING[]
    # a host layer's program_observers = false: the program's own observers are given nothing (it only removes)
    vetoed = any(l -> l.where !== :own && l.program_observers === false, layers)
    # an observer is not given the calls its own code makes (it would be given them, and make more, without end)
    observers = Any[o for l in Iterators.reverse(layers) if !(vetoed && l.where === :own) for o in l.observers if !any(x -> x === o, busy)]
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
                       journal=haskey(s, :journal) && s[:journal] !== nothing ? s[:journal] : UNSET,
                       program_observers=get(s, :program_observers, nothing))
    global_now = lock(() -> copy(GLOBAL_SETTINGS), SETTINGS_LOCK)
    [layer(:own, own); [layer(:block, b) for b in Iterators.reverse(SCOPED_LAYERS[])]; layer(:configure, global_now)]
end

# ------------------------------------------------------------------ work off the call's task

"""
Set once Julia is exiting, after FunctAI's last drain. Julia then closes its
timers and sockets, which wakes the tasks waiting on them (a store's `keep!`
asleep in `sleep`, an observer reading a socket) with an error, while it
tears itself down on the main thread; a task woken then does nothing more,
and so compiles nothing (compiling beside Julia's exit can crash it).
"""
const EXITING = Threads.Atomic{Bool}(false)

"""
Run `f` on a task of its own, on any thread, outside every call: its FunctAI
calls start trees of their own, with no enclosing block's settings (an
observer or a store that calls an AI function is not a step of the call
that gave it the event). A failure is shown, never raised into a call.
"""
function spawn_apart(f)
    task = with(CURRENT_CALL => nothing, STREAM_OPENING => nothing, SCOPED_SETTINGS => Dict{Symbol,Any}(),
                SCOPED_LAYERS => Dict{Symbol,Any}[]) do
        Threads.@spawn f()
    end
    errormonitor(task)
end

# ------------------------------------------------------------------ observers

"""
The observers whose code is running here, and those whose code gave them the
event they are handling: an observer that calls an AI function (a host's
exporter that summarises each call, say) is not given the calls it makes,
nor the calls another observer makes on seeing those, so observers never
feed themselves without end. Every other observer sees them. A Channel is
given its events by `put!`, and the task that reads them is the user's,
which this cannot recognise: a reader that calls an AI function on each
event feeds itself (the guides say so: use an observer function).
"""
const OBSERVING = ScopedValue{Vector{Any}}(Any[])     # never changed in place: a new vector per observer added
# (a Vector{Any}, not a tuple: one concrete type whatever the observers, so a new observer compiles nothing here)

"""
The most events an observer may have waiting. A slower one loses the events
beyond (contract/streaming.md: "may drop them"), and sees a loss: each
event's `after` names the event before it in its feed, given or not.
"""
const OBSERVER_BUFFER = 10_000

"One observer's events, handed to it in order from a task of its own, so a slow observer never slows a call."
mutable struct Feed
    observer::Any
    waiting::Vector{Tuple{Event,Vector{Any}}}     # each event, and the observers whose code made the call it is about
    running::Bool
end

const FEEDS = IdDict{Any,Feed}()            # the observers with events waiting, or being given one
const BROKEN_OBSERVERS = IdDict{Any,Bool}()  # observers that failed: given no more events
const FEEDS_LOCK = ReentrantLock()

"""
Give an observer an event, without waiting for it: the event waits in the
observer's feed, which one task works through in order (so an observer is
never called from two places at once, whatever calls feed it). A failed
observer gets nothing more; a full feed drops the event, once with a warning.
"""
function give!(o, e::Event)
    made_by = OBSERVING[]                  # read on the call's task: the observers whose code made this call
    lock(FEEDS_LOCK) do
        get(BROKEN_OBSERVERS, o, false) && return
        f = get!(() -> Feed(o, Tuple{Event,Vector{Any}}[], false), FEEDS, o)
        if length(f.waiting) >= OBSERVER_BUFFER
            warn_once("observer-slow:$(objectid(o))", "an observer ($(observer_name(o))) is too slow: events it has not taken " *
                      "yet are dropped (it sees a loss: an event's after names one it was not given)")
            return
        end
        push!(f.waiting, (e, made_by))
        if !f.running
            f.running = true
            spawn_apart(() -> run_feed(f))
        end
    end
    nothing
end

function run_feed(f::Feed)
    while true
        next = lock(FEEDS_LOCK) do
            if isempty(f.waiting) || get(BROKEN_OBSERVERS, f.observer, false)
                f.running = false
                get(FEEDS, f.observer, nothing) === f && delete!(FEEDS, f.observer)     # idle: forgotten until its next event
                nothing
            else
                popfirst!(f.waiting)
            end
        end
        next === nothing && return
        e, made_by = next
        try
            o = f.observer
            if o isa Channel
                put!(o, e)
            else
                # in the newest world: this task may have been started by one older than the observer's code
                with(OBSERVING => Any[made_by..., o]) do
                    Base.invokelatest(o, e)
                end
            end
        catch err
            EXITING[] && return
            lock(() -> (BROKEN_OBSERVERS[f.observer] = true; empty!(f.waiting)), FEEDS_LOCK)
            warn_once("observer:$(objectid(f.observer))",
                      "an observer ($(observer_name(f.observer))) failed ($(sprint(showerror, unwrap(err)))); it is given no more events")
        end
    end
end
observer_name(o) = o isa Channel ? "a Channel" : o isa Function ? string(nameof(o)) : string(typeof(o))

# ------------------------------------------------------------------ the writer

"""
    LogWriter(journal, tree)

Appends one tree's kept events to its journal, in order, as they are made
(contract/streaming.md, "Appending"): each event is sent at most
`1 + retries` times in a row; an event the journal answers `:kept` or
`:duplicate` is confirmed; one it refuses stops the writer (it appends
nothing more); one that gets no answer is kept and sent again with the next
event. A store that cannot be called (`MethodError`, `UndefVarError`: not
an answer, and sending again would not help) stops it too. Sending runs on a
task of its own, at most one send at a time (a send the store has not
answered is waited for again, never started again beside itself), so a call
never waits for a best-effort journal;
[`confirmation`](@ref) waits for an event's own round (a required journal's
barrier).
"""
mutable struct LogWriter
    journal::Journal
    tree::String
    waiting::Vector{Event}          # given, not yet sent
    pending::Vector{Event}          # sent, not confirmed (sent again with the next round)
    submitted::Int                  # events given to the writer
    processed::Int                  # events that had their round
    confirmed::Int                  # events confirmed, in order
    refused::Bool                   # the journal refused an event: nothing more is appended
    stopped::Bool                   # refused, or the store's own code failed: nothing more is appended
    cause::Any                      # what the journal met last
    closed::Bool
    running::Bool
    cond::Threads.Condition
    warned::Bool
    sending::Any                    # the send the store has not answered yet (a `Send`), or nothing
end
LogWriter(journal::Journal, tree::AbstractString) =
    LogWriter(journal, String(tree), Event[], Event[], 0, 0, 0, false, false, nothing, false, false, Threads.Condition(), false, nothing)

"Send one (kept) event to the journal, after those given before it; its number among them (0 once the writer is closed)."
function Base.push!(w::LogWriter, e::Event)
    lock(w.cond) do
        w.closed && return 0
        w.submitted += 1
        push!(w.waiting, e)
        if !w.running
            w.running = true
            spawn_apart(() -> sender(w))
        end
        w.submitted
    end
end

function sender(w::LogWriter)
    while true
        e = lock(w.cond) do
            isempty(w.waiting) ? (w.running = false; notify(w.cond); nothing) : popfirst!(w.waiting)
        end
        e === nothing && return
        round!(w, e)
        lock(w.cond) do
            w.processed += 1
            notify(w.cond)
        end
    end
end

"""
The exceptions that say a store cannot be called at all (a method or a name
that does not exist, a keyword it requires: its code is not there, and
sending again would not help), and an interrupt. Any other exception, a
`KeyError` or an `ArgumentError` included, may be a passing condition: it is
no answer, and the event is sent again.
"""
is_fault(err) = err isa Union{MethodError,UndefVarError,UndefKeywordError,InterruptException} ||
                (isdefined(Core, :FieldError) && err isa getfield(Core, :FieldError))

"What the journal met when a send got no answer in time."
struct NoAnswer
    msg::String
end
Base.string(x::NoAnswer) = x.msg

"One `keep!` of an event, running on a task of its own, and its answer once it has one."
mutable struct Send
    event::Event
    done::Bool
    answer::Any
    cond::Threads.Condition
end

"""
One send: the store's answer, or its exception; `NoAnswer` when it takes
longer than the journal's timeout. A send the store has not answered stays
the writer's one send: the event is sent again by waiting for it again (its
answer, when it comes, is the answer), so a store that hangs holds one task
per writer, never one per attempt.
"""
function send_once(w::LogWriter, e::Event)
    s = w.sending
    if s !== nothing && s.event !== e
        # never happens (the writer sends the first event not confirmed until it is): no second send beside it
        lock(() -> s.done, s.cond) || return NoAnswer("the store has not answered the send before")
        s = nothing
    end
    if s === nothing
        s = Send(e, false, nothing, Threads.Condition())
        w.sending = s
        store = w.journal.store
        spawn_apart() do
            answer = try
                Base.invokelatest(keep!, store, e)          # in the newest world, as the observers
            catch err
                EXITING[] && return
                unwrap(err)
            end
            lock(s.cond) do
                s.done, s.answer = true, answer
                notify(s.cond)
            end
        end
    end
    t = w.journal.timeout
    expired = Ref(false)            # set by the timer (a monotonic clock), never by reading the wall clock
    timer = t === nothing ? nothing : Timer(_ -> lock(() -> (expired[] = true; notify(s.cond)), s.cond), t)
    answered, answer = lock(s.cond) do
        while !s.done && !expired[]
            wait(s.cond)
        end
        s.done ? (true, s.answer) : (false, NoAnswer("no answer in $(t) s"))
    end
    timer === nothing || close(timer)
    answered && (w.sending = nothing)
    answer
end

"One round: the event joins what is not confirmed, and the writer sends it all again, in order."
function round!(w::LogWriter, e::Event)
    w.stopped && return
    push!(w.pending, e)
    while !isempty(w.pending)
        answered = false
        for _ in 1:(1+w.journal.retries)
            answer = send_once(w, w.pending[1])
            if answer isa StoreRefusal || (answer isa Exception && is_fault(answer))
                # a refusal is the journal's answer; a fault of the store's own code is none (it may have kept
                # the event: "unknown"), and sending again would not help: either way the writer stops
                w.refused = answer isa StoreRefusal
                w.stopped = true
                w.cause = answer
                warn_journal(w, answer)
                return
            end
            if answer === :kept || answer === :duplicate
                answered = true
                break
            end
            w.cause = answer isa Union{Exception,NoAnswer} ? answer : NoAnswer("the store answered $(repr(answer))")
        end
        if !answered
            warn_journal(w, w.cause)
            return
        end
        lock(w.cond) do
            popfirst!(w.pending)
            w.confirmed += 1
            notify(w.cond)
        end
    end
end

function warn_journal(w::LogWriter, why)
    w.journal.required && return                      # a required journal says so to its caller (JournalError, with the cause)
    w.warned && return
    w.warned = true
    store = w.journal.store
    name = store isa MemoryStore ? "MemoryStore($(repr(store.name)))" : string(nameof(typeof(store)))
    warn_once("journal:$(objectid(store))",
              "a best-effort journal ($name) did not keep an event ($(why isa Exception ? sprint(showerror, why) : string(why))); calls go on")
end

"""
    confirmation(w::LogWriter, n) -> :confirmed, :unanswered or :refused

Wait until the `n`th event given to the writer is confirmed, or has had its
round (after the writer's own resends) without being confirmed. Events given
after it (a call running beside this one) are not waited for.
"""
function confirmation(w::LogWriter, n::Integer)
    n == 0 && return :refused                          # given after the writer closed: never sent
    lock(w.cond) do
        while w.confirmed < n && w.processed < n
            wait(w.cond)
        end
        w.confirmed >= n ? :confirmed : w.refused ? :refused : :unanswered
    end
end
"Wait until every event given so far has had its round."
Base.flush(w::LogWriter) = (confirmation(w, lock(() -> w.submitted, w.cond)); nothing)
"The writer takes no more events; those given are still sent."
Base.close(w::LogWriter) = lock(() -> (w.closed = true; nothing), w.cond)
"Whether the writer has sent every event it was given (for `drain`)."
idle(w::LogWriter) = lock(() -> !w.running && isempty(w.waiting), w.cond)

"The error a caller gets when a required journal did not confirm its end, holding the call's outcome."
function end_error(status::Symbol, terminal::Event, outcome::NamedTuple; cause=nothing, store=nothing)
    word = status === :refused ? "refused" : "unknown"
    JournalError("journal-end", "the journal did not confirm the call's end ($(Position(terminal))): " *
                 (word == "refused" ? "it refused it" : "no answer came; it may be kept (settle it by reading the journal)"),
                 word, terminal.tree, Position(terminal), outcome, cause, store)
end

barrier_error(e::Event; cause=nothing, store=nothing) = JournalError("journal-barrier", "the journal did not keep event $(e.seq)"; cause, store)

"""
    settle(err::JournalError) -> :kept, :not_kept or :another_end

What the journal says of the end a `journal-end` error could not confirm:
[`settle`](@ref)`(err.store, err.tree, err.event)`.
"""
function settle(err::JournalError)
    (err.code == "journal-end" && err.store !== nothing) ||
        throw(ArgumentError("only a journal-end JournalError names an end to settle"))
    settle(err.store, err.tree, err.event)
end

"""
    FunctAI.drain(timeout = 5.0) -> Bool

Wait, at most `timeout` seconds, until every observer has been given the
events made so far and every journal writer has sent what it was given;
whether they all were. Calls never wait for them (a slow observer or a
best-effort journal does not slow a call): a script that must see them all
before it goes on, or before it exits, drains. FunctAI drains for 2 seconds
when Julia exits.

Observers and stores run on Julia tasks, which share threads and take turns
when one yields (sleeps, waits, reads or writes). Code that holds its thread
without yielding (`Libc.systemsleep`, a long computation, a C library's
blocking call) holds every task on that thread until it is done. With one
default thread (`julia --threads=1`, and a spawned call under Julia's
default of one default thread beside the interactive one) calls on that
thread then wait for it; start Julia with several threads
(`julia --threads=auto`) to keep such an observer or store off the calls'
threads, or make it yield.
"""
function drain(timeout::Real=5.0)
    done() = lock(() -> isempty(FEEDS), FEEDS_LOCK) && all(idle, lock(() -> collect(keys(WRITERS)), WRITERS_LOCK))
    timedwait(done, Float64(timeout); pollint=0.001) === :ok
end
const WRITERS = WeakKeyDict{LogWriter,Nothing}()
const WRITERS_LOCK = ReentrantLock()

# ------------------------------------------------------------------ one tree's log, in this process

"""
The whole log of one call tree, in the process running it: numbered here
(writer 1), given to the streams watching calls in it (the whole form), to
each call's observers (the kept form) and to the tree's journal (the kept
form). Nothing of a call follows its end, and nothing follows the outermost
call's end: each call waits for the calls made inside it to end first, and a
call that starts inside a call that has ended starts a tree of its own.
"""
mutable struct TreeLog
    tree::String
    writer::Int
    seq::Int
    last::Union{Nothing,Position}
    at::Float64
    lock::ReentrantLock
    cond::Threads.Condition                         # on `lock`: a call inside the tree ended
    streams::Vector{Any}
    keeps::Dict{String,Keep}
    programs::Dict{String,JObj}
    observers::Dict{String,Vector{Any}}             # each call's observers
    observer_last::IdDict{Any,Union{Nothing,Position}}
    journal::Union{Nothing,Journal}
    writer_task::Union{Nothing,LogWriter}
    journal_last::Union{Nothing,Position}
    given::Dict{Int,Int}                            # a barrier event's seq → its number among the writer's events
    ended::Bool                                     # its outermost call's end is numbered: nothing more is
    clock::Any                                      # seq -> seconds since the epoch
    tap::Any                                        # a scripted run's copy of the whole log (see `SCRIPTED`)
end
function TreeLog(tree::AbstractString, journal; clock=nothing, tap=nothing)
    lk = ReentrantLock()
    w = journal === nothing ? nothing : LogWriter(journal, tree)
    w === nothing || lock(() -> (WRITERS[w] = nothing), WRITERS_LOCK)
    TreeLog(String(tree), 1, 0, nothing, 0.0, lk, Threads.Condition(lk), Any[], Dict{String,Keep}(), Dict{String,JObj}(),
            Dict{String,Vector{Any}}(), IdDict{Any,Union{Nothing,Position}}(), journal, w, nothing, Dict{Int,Int}(), false,
            something(clock, _ -> time()), tap)
end

required(t::TreeLog) = t.journal !== nothing && t.journal.required
watched(t::TreeLog, call_observers) = !isempty(t.streams) || t.journal !== nothing || !isempty(call_observers)

"Number an event of a call in this tree's log (`at` never goes back)."
function number!(t::TreeLog, kind::Symbol, call_id::AbstractString, fn::AbstractString, data::JObj)
    t.seq += 1
    t.at = max(t.at, Float64(t.clock(t.seq)))
    e = owned_event(kind, t.tree, t.writer, t.seq, t.last, iso(t.at), call_id, fn, data)
    t.last = Position(e)
    e
end

const BARRIER_KINDS = (:started, :tool_call, :done, :failed)

"""
Make an event and give it to its readers: the streams watching its call, its
call's observers and the journal. With `withhold` (a required journal's
last event), the streams and observers get it only once `deliver!` is
called, when the journal confirmed it. `last`: the outermost call's end,
after which nothing is numbered: an event after it is a fault of this
package (every call inside a tree ends before it), raised, never dropped.
"""
function emit!(t::TreeLog, kind::Symbol, call_id, fn, data::JObj; withhold::Bool=false, last::Bool=false)
    lock(t.lock) do
        t.ended && error("FunctAI: a $kind event of call $call_id after its tree $(t.tree) ended (a fault of FunctAI, not of your code)")
        e = number!(t, kind, call_id, fn, data)
        last && (t.ended = true)
        t.tap === nothing || push!(t.tap, e)
        if kind === :started
            t.programs[e.call] = datum(e, "program")
        end
        if t.writer_task !== nothing
            k = kept_event(e, t.keeps[e.call], t.programs[e.call])
            if k !== nothing
                k = relinked(k, t.journal_last)
                t.journal_last = Position(k)
                n = push!(t.writer_task, k)
                kind in BARRIER_KINDS && (t.given[e.seq] = n)
            end
        end
        withhold || deliver!(t, e)
        e
    end
end

"""
Wait until the journal confirmed the event (every event up to it: appends
are in order), or gave up on it: `:confirmed`, `:unanswered` or
`:refused`. `:confirmed` without a journal.
"""
function confirmation(t::TreeLog, e::Event)
    t.writer_task === nothing && return :confirmed
    n = lock(() -> get(t.given, e.seq, 0), t.lock)
    confirmation(t.writer_task, n)
end

"What the tree's journal met last (a `JournalError`'s cause)."
journal_cause(t::TreeLog) = t.writer_task === nothing ? nothing : lock(() -> t.writer_task.cause, t.writer_task.cond)

"Give an event to the streams watching its call and to its call's observers (who get it on their own tasks)."
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
            linked = relinked(k, get(t.observer_last, o, nothing))
            t.observer_last[o] = Position(linked)          # a dropped event is still the one before the next
            give!(o, linked)
        end
    end
end
