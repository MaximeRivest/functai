# Where conversations are kept (contract/conversations.md, "Stores").
#
# A store keeps each conversation as an ordered list of records (JSON
# objects) and never changes one. Every store has two methods:
#
# - `append_records!(store, conversation, records; expect = nothing) -> Int`:
#   add the records at the end, all or none, numbering them (`seq`, 1 for a
#   conversation's first); with `expect`, only when the conversation holds
#   exactly that many records now, else `ConversationError("store-conflict")`.
#   Returns how many it holds after.
# - `read_records(store, conversation, after = 0)`: the records after position
#   `after`, in order.
#
# and may have `event_store(store)` (a store of call tree logs, so another
# process can watch a turn while it is written), `wait_records(store,
# conversation, after, timeout)`, `durability(store)` and `persistent(store)`.
# Two are here: `MemoryConversations` (the default: this process's memory)
# and `FolderStore` (files in a folder, locked across processes; the same
# files Python's FolderStore writes).

const CONVERSATION_FORMAT = 1
const CONVERSATION_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$"

"A conversation's id: 1 to 200 ASCII letters, digits, `.`, `_` or `-`, starting with a letter or digit (a file name everywhere, never a path)."
function check_conversation_id(id)
    (id isa AbstractString && occursin(CONVERSATION_ID, id)) ||
        throw(ConversationError("conversation-id", "a conversation's id is 1 to 200 letters, digits, '.', '_' or '-', starting " *
                                                   "with a letter or digit; not $(repr(id))"))
    String(id)
end

"""
    FunctAI.ConversationStore

Where conversations are kept (contract/conversations.md, "Stores"): a
subtype with [`append_records!`](@ref) and [`read_records`](@ref), and
optionally `event_store`, `wait_records`, `durability` and `persistent`.
[`MemoryConversations`](@ref) and [`FolderStore`](@ref) are two.
"""
abstract type ConversationStore end

"""
    FunctAI.append_records!(store, conversation, records; expect = nothing) -> Int

Add records at the end of a conversation, all or none, giving each its
`seq`; with `expect`, only when the conversation holds exactly `expect`
records (else `ConversationError("store-conflict")`). Returns how many it holds after.
"""
function append_records! end

"""
    FunctAI.read_records(store, conversation, after = 0) -> Vector

The records of a conversation after position `after`, in order (copies).
"""
function read_records end

"Wait until a record after `after` exists (or the time is up), so a reader need not ask again and again."
wait_records(store, conversation, after, timeout) = sleep(min(timeout, 0.1))
"How surely an append is kept when it returns: `\"memory\"`, `\"disk\"` (written and flushed), or the store's own word."
durability(store) = "unknown"
"Whether a store keeps records beyond this process (unknown: yes)."
persistent(store) = true
"A store's call tree logs (an `EventStore`), or `nothing`."
event_store(store) = nothing

"""
    MemoryConversations()

Conversations kept in this process's memory (lost when it ends): the store a
conversation uses when none is named. One per process by default, so
opening the same id again opens the same conversation.
"""
struct MemoryConversations <: ConversationStore
    records::Dict{String,Vector{JObj}}
    cond::Threads.Condition
    events::MemoryStore
end
MemoryConversations() = MemoryConversations(Dict{String,Vector{JObj}}(), Threads.Condition(), MemoryStore("conversations"))
durability(::MemoryConversations) = "memory"
persistent(::MemoryConversations) = false
event_store(m::MemoryConversations) = m.events
Base.show(io::IO, m::MemoryConversations) = print(io, "MemoryConversations(", lock(() -> length(m.records), m.cond), " conversations)")

function append_records!(m::MemoryConversations, conversation::AbstractString, records; expect=nothing)
    check_conversation_id(conversation)
    lock(m.cond) do
        log = get!(() -> JObj[], m.records, String(conversation))
        expect === nothing || expect == length(log) ||
            throw(ConversationError("store-conflict", "conversation $conversation holds $(length(log)) records, not $expect"))
        for r in records
            rec = LMCC.deepcopy_json(JObj(String(k) => v for (k, v) in r))
            rec["seq"] = length(log) + 1
            push!(log, rec)
        end
        notify(m.cond)
        length(log)
    end
end
read_records(m::MemoryConversations, conversation::AbstractString, after::Integer=0) =
    lock(() -> JObj[LMCC.deepcopy_json(r) for r in get(m.records, conversation, JObj[])[after+1:end]], m.cond)
function wait_records(m::MemoryConversations, conversation, after, timeout)
    deadline = time() + timeout
    lock(m.cond) do
        while length(get(m.records, conversation, JObj[])) <= after && time() < deadline
            t = Timer(_ -> lock(() -> notify(m.cond), m.cond), max(0.0, min(0.1, deadline - time())))
            wait(m.cond)
            close(t)
        end
    end
end
"The ids of the conversations a store holds."
conversations(m::MemoryConversations) = lock(() -> sort!([k for (k, v) in m.records if !isempty(v)]), m.cond)

const MEMORY_CONVERSATIONS = MemoryConversations()

# ------------------------------------------------------------------ files, locked across processes

const FILE_LOCKS = Dict{String,ReentrantLock}()
const FILE_LOCKS_GUARD = ReentrantLock()

"""
Run `f()` holding an exclusive lock on a file, across processes (`flock` on
Unix, `LockFileEx` on Windows: what Python's `FolderStore` takes on the same
file) and across this process's tasks. The lock is taken without blocking
the thread: a busy lock is asked for again after a short sleep.
"""
function with_file_lock(f, path::AbstractString)
    inner = lock(() -> get!(ReentrantLock, FILE_LOCKS, String(path)), FILE_LOCKS_GUARD)
    lock(inner) do
        io = Base.Filesystem.open(path, Base.Filesystem.JL_O_RDWR | Base.Filesystem.JL_O_CREAT, 0o600)
        try
            pause = 0.002
            while !try_lock_file(io)
                sleep(pause)
                pause = min(2pause, 0.05)
            end
            f()
        finally
            close(io)                                    # closing releases the lock
        end
    end
end

@static if Sys.iswindows()
    function try_lock_file(io)
        h = ccall(:_get_osfhandle, Ptr{Cvoid}, (Cint,), reinterpret(Cint, io.handle))
        overlapped = zeros(UInt8, 32)
        # LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, the first byte (as Python's msvcrt.locking)
        ccall((:LockFileEx, "kernel32"), stdcall, Cint, (Ptr{Cvoid}, UInt32, UInt32, UInt32, UInt32, Ptr{UInt8}),
              h, 0x2 | 0x1, 0, 1, 0, overlapped) != 0
    end
    durable_flush(io) = ccall(:_commit, Cint, (Cint,), reinterpret(Cint, io.handle))
else
    try_lock_file(io) = ccall(:flock, Cint, (Cint, Cint), reinterpret(Cint, io.handle), 2 | 4) == 0     # LOCK_EX | LOCK_NB
    durable_flush(io) = ccall(:fsync, Cint, (Cint,), reinterpret(Cint, io.handle))
end

"Append bytes to a file in one write (`O_APPEND`, `0600`), flushed to disk when `durable`."
function append_bytes(path::AbstractString, data::Vector{UInt8}; durable::Bool=true)
    io = Base.Filesystem.open(path, Base.Filesystem.JL_O_WRONLY | Base.Filesystem.JL_O_CREAT | Base.Filesystem.JL_O_APPEND, 0o600)
    try
        write(io, data)
        durable && durable_flush(io)
    finally
        close(io)
    end
    nothing
end

"A JSON lines file read incrementally: what was parsed is kept, and only what was appended since is read again."
mutable struct JSONLines
    path::String
    size::Int
    items::Vector{Any}
    lock::ReentrantLock
end
JSONLines(path) = JSONLines(String(path), 0, Any[], ReentrantLock())
function load!(l::JSONLines)
    lock(l.lock) do
        size = isfile(l.path) ? filesize(l.path) : 0
        size < l.size && (l.size = 0; empty!(l.items))          # replaced: read it all again
        if size > l.size
            data = open(l.path, "r") do io
                seek(io, l.size)
                read(io, size - l.size)
            end
            stop = something(findlast(==(UInt8('\n')), data), 0)     # a line still being written is read next time
            for raw in split(String(data[1:stop]), '\n')
                isempty(strip(raw)) && continue
                push!(l.items, try
                    LMCC.parse_json(raw)
                catch
                    JObj("unreadable" => true)
                end)
            end
            l.size += stop
        end
        l.items
    end
end

record_line(r) = Vector{UInt8}(LMCC.json_text(r) * "\n")

"""
    FunctAI.FolderEvents(folder)

Call tree logs kept in files, by the rules every store keeps
(contract/streaming.md, "The rules a store keeps"): `<folder>/<tree>.jsonl`
holds the kept events, `<tree>.writer` the last writer number a claim gave,
`<tree>.lock` is held for each claim and append (across processes).
"""
struct FolderEvents <: EventStore
    folder::String
    durable::Bool
    files::Dict{String,JSONLines}
    events::Dict{String,Vector{Event}}      # each log's events read so far (each line checked once, when first read)
    events_read::Dict{String,Int}           # how many of its lines they come from
    guard::ReentrantLock
end
function FolderEvents(folder::AbstractString; durable::Bool=false)
    mkpath(folder; mode=0o700)
    FolderEvents(abspath(folder), durable, Dict{String,JSONLines}(), Dict{String,Vector{Event}}(), Dict{String,Int}(), ReentrantLock())
end
Base.show(io::IO, s::FolderEvents) = print(io, "FolderEvents(", repr(s.folder), ")")

function tree_file(s::FolderEvents, tree::AbstractString, ext)
    occursin(CONVERSATION_ID, tree) || throw(StoreRefusal("event-malformed", "$(repr(tree)) is not a log's id"))
    joinpath(s.folder, "$tree.$ext")
end
tree_lines(s::FolderEvents, tree) = lock(() -> get!(() -> JSONLines(tree_file(s, tree, "jsonl")), s.files, String(tree)), s.guard)
"A log's events (a copy), its new lines read and checked once: a store appended to event by event reads each line once."
function tree_events(s::FolderEvents, tree)
    lines = tree_lines(s, tree)
    lock(lines.lock) do
        items = load!(lines)
        seen = get!(() -> Event[], s.events, String(tree))
        read_upto = get(s.events_read, String(tree), 0)
        if read_upto > length(items)              # the file was replaced: read it all again
            empty!(seen)
            read_upto = 0
        end
        for x in items[read_upto+1:end]
            x isa AbstractDict && !haskey(x, "unreadable") && push!(seen, Event(x))
        end
        s.events_read[String(tree)] = length(items)
        copy(seen)
    end
end
function tree_writer(s::FolderEvents, tree)
    p = tree_file(s, tree, "writer")
    isfile(p) || return 1
    something(tryparse(Int, strip(read(p, String))), 1)
end

function keep!(s::FolderEvents, x)
    keep!(s, Any[x]) === :duplicate ? :duplicate : :kept
end
function keep!(s::FolderEvents, xs::AbstractVector)
    isempty(xs) && return :duplicate
    tree = xs[1] isa Event ? xs[1].tree : xs[1] isa AbstractDict ? get(xs[1], "tree", nothing) : nothing
    tree isa AbstractString || throw(StoreRefusal("event-malformed", "not an event"))
    with_file_lock(tree_file(s, tree, "lock")) do
        log = tree_events(s, tree)
        n = length(log)
        writer = tree_writer(s, tree)
        answers = Symbol[]
        for x in xs
            try
                push!(answers, append_one!(log, writer, x, tree))
            catch err
                err isa StoreRefusal || rethrow()
                p = err.event !== nothing ? err.event : stated_position(x isa Event ? event_json(x) : x)
                throw(StoreRefusal(err.code, p, err.msg))
            end
        end
        if length(log) > n
            append_bytes(tree_file(s, tree, "jsonl"), reduce(vcat, [record_line(event_json(e)) for e in log[n+1:end]]); durable=s.durable)
        end
        all(==(:duplicate), answers) ? :duplicate : :kept
    end
end
function claim!(s::FolderEvents, tree::AbstractString)
    with_file_lock(tree_file(s, tree, "lock")) do
        log = tree_events(s, tree)
        isempty(log) && throw(StoreRefusal("event-unknown", "no log $tree"))
        is_end(log, tree) && throw(StoreRefusal("event-after-end", "the log $tree is finished"))
        w = tree_writer(s, tree) + 1
        p = tree_file(s, tree, "writer")
        tmp = p * ".tmp"
        write(tmp, string(w))
        mv(tmp, p; force=true)
        Claim(w, Position(last(log)))
    end
end
events_after(s::FolderEvents, tree::AbstractString, after) = resume(tree_events(s, tree), after)
finished(s::FolderEvents, tree::AbstractString) = is_end(tree_events(s, tree), tree)
writer_of(s::FolderEvents, tree::AbstractString) = tree_writer(s, tree)

"""
    FolderStore(folder)

Conversations kept in a folder, shared by every process that opens it (and
by Python's, TypeScript's and R's FolderStore on the same folder):

    <folder>/conversations/<id>.jsonl   the records, one per line
    <folder>/conversations/<id>.lock    held while appending
    <folder>/trees/<tree>.jsonl         each turn's call tree log (kept form)

Each append is one step across processes (a lock on the conversation),
written and flushed to disk before it returns (`durability` `"disk"`).
Files are readable by their owner only. `store = "folder/"` names one.
"""
struct FolderStore <: ConversationStore
    folder::String
    events::FolderEvents
    files::Dict{String,JSONLines}
    guard::ReentrantLock
end
function FolderStore(folder::AbstractString)
    root = abspath(expanduser(folder))
    mkpath(joinpath(root, "conversations"); mode=0o700)
    FolderStore(root, FolderEvents(joinpath(root, "trees")), Dict{String,JSONLines}(), ReentrantLock())
end
durability(::FolderStore) = "disk"
persistent(::FolderStore) = true
event_store(s::FolderStore) = s.events
Base.show(io::IO, s::FolderStore) = print(io, "FolderStore(", repr(s.folder), ")")
conversation_file(s::FolderStore, c, ext) = joinpath(s.folder, "conversations", "$(check_conversation_id(c)).$ext")
conversation_lines(s::FolderStore, c) = lock(() -> get!(() -> JSONLines(conversation_file(s, c, "jsonl")), s.files, String(c)), s.guard)

function append_records!(s::FolderStore, conversation::AbstractString, records; expect=nothing)
    with_file_lock(conversation_file(s, conversation, "lock")) do
        n = length(load!(conversation_lines(s, conversation)))
        expect === nothing || expect == n ||
            throw(ConversationError("store-conflict", "conversation $conversation holds $n records, not $expect"))
        data = UInt8[]
        for r in records
            n += 1
            rec = JObj(String(k) => v for (k, v) in r)
            rec["seq"] = n
            append!(data, record_line(rec))
        end
        isempty(data) || append_bytes(conversation_file(s, conversation, "jsonl"), data; durable=true)
        n
    end
end
read_records(s::FolderStore, conversation::AbstractString, after::Integer=0) =
    JObj[LMCC.deepcopy_json(r) for r in load!(conversation_lines(s, conversation))[after+1:end]]
function wait_records(s::FolderStore, conversation, after, timeout)
    deadline = time() + timeout
    pause = 0.02
    while time() < deadline
        length(load!(conversation_lines(s, conversation))) > after && return
        sleep(min(pause, max(0.0, deadline - time())))
        pause = min(2pause, 0.2)
    end
end
conversations(s::FolderStore) = sort!([f[1:end-6] for f in readdir(joinpath(s.folder, "conversations")) if endswith(f, ".jsonl")])

"Where `store = true` keeps conversations: `\$XDG_DATA_HOME/functai/conversations`, beside the call log's default folder."
default_conversations_folder() = joinpath(dirname(default_log_folder()), "conversations")

const FOLDER_STORES = Dict{String,FolderStore}()
const FOLDER_STORES_LOCK = ReentrantLock()

"""
The store a `store =` value names: `nothing` (this process's memory),
`true` (the default folder), a folder, or a [`ConversationStore`](@ref).
"""
function conversation_store(store)
    (store === nothing || store === false) && return MEMORY_CONVERSATIONS
    store === true && (store = default_conversations_folder())
    if store isa AbstractString
        where = abspath(expanduser(store))
        return lock(() -> get!(() -> FolderStore(where), FOLDER_STORES, where), FOLDER_STORES_LOCK)
    end
    store isa ConversationStore && return store
    (hasmethod(append_records!, Tuple{typeof(store),String,Vector{JObj}}) && hasmethod(read_records, Tuple{typeof(store),String,Int})) &&
        return store
    throw(ArgumentError("store is nothing (memory), true (the default folder), a folder, or a FunctAI.ConversationStore " *
                        "(append_records!, read_records); not $(repr(store))"))
end

"""
The kept events of a tree's log from a store, after `after`: those kept so
far, then each as it is kept, until the log's last event (or `stop()` says
so, or `timeout` seconds pass without one). Calls `f(event)` for each.
"""
function follow_store(f, events, tree::AbstractString, after=nothing; poll=0.05, stop=nothing, timeout=nothing)
    last = after
    quiet = time()
    while true
        got = try
            events_after(events, tree, last)
        catch err
            (err isa StoreRefusal && err.code == "event-unknown" && last === nothing) || rethrow()
            Event[]                            # the log has not started yet
        end
        for e in got
            last = Position(e)
            quiet = time()
            f(e)
            e.call == tree && e.kind in (:done, :failed) && return
        end
        stop !== nothing && stop() && return
        timeout !== nothing && time() - quiet > timeout && return
        sleep(poll)
    end
end
