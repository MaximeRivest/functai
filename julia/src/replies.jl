# The reply cache (contract/replies.md): a model's reply to a request, reused
# when the same request is made again.
#
#     configure!(cache_replies = :disk)       # kept across runs, in one SQLite file every process shares
#     configure!(cache_replies = true)        # in this process's memory only
#     configure(f; replicate = 2)             # a third, independent answer to the same request
#
# A reply is kept only once it was read (lmcc read it, and its values fit
# their types), so an interrupted run leaves nothing half-written. One flight
# per key: while one caller asks the model, another caller of the same
# request waits for its reply, in this process (a lock per key) and across
# processes (a claim with a lease, in the same SQLite file). A call whose
# `log_content` drops any field is never written to disk: a disk cache holds
# whole requests and replies, which is what `log_content` forbids keeping.

const REPLY_FORMAT = 1
const REPLY_LEASE = 120.0                  # seconds a claim holds a key against other processes

const REPLY_OWNER = Ref{String}("")
function reply_owner()
    pid = getpid()
    if !occursin(":$pid:", REPLY_OWNER[])
        REPLY_OWNER[] = "$(gethostname()):$pid:$(bytes2hex(rand(UInt8, 3)))"
    end
    REPLY_OWNER[]
end

"""
    FunctAI.reply_key(request, replicate = 0) -> String

The cache key of a request (contract/replies.md, "The key"): `"sha256:"` of
the canonical JSON of `{"functai_reply": 1, "request": <lm15's canonical
request>, "replicate": n}`. The same in every language, so a cache written by
one is read by another.
"""
reply_key(request::LM15.Request, replicate::Integer=0) =
    LMCC.sha256_of(LMCC.jobj("functai_reply" => REPLY_FORMAT, "request" => LM15.to_dict(request), "replicate" => Int(replicate)))

response_text(r) = LMCC.canonical_json(LM15.to_dict(r))
response_of(text::AbstractString) = LM15.from_dict(LM15.Response, LMCC.parse_json(text))

# ------------------------------------------------------------------ one flight per key, in this process

struct KeyLocks
    cond::Threads.Condition
    held::Dict{String,Task}
end
KeyLocks() = KeyLocks(Threads.Condition(), Dict{String,Task}())

"Take a key (one caller at a time per key, in this process); `cancelled()` stops the wait by throwing."
function acquire!(k::KeyLocks, key::String, cancelled=nothing)
    me = current_task()
    lock(k.cond) do
        while haskey(k.held, key) && k.held[key] !== me
            cancelled === nothing || cancelled()
            # wake within 0.1 s to look at `cancelled` again
            t = Timer(_ -> lock(() -> notify(k.cond), k.cond), 0.1)
            wait(k.cond)
            close(t)
        end
        k.held[key] = me
    end
end
release!(k::KeyLocks, key::String) = lock(k.cond) do
    get(k.held, key, nothing) === current_task() && delete!(k.held, key)
    notify(k.cond)
end

# ------------------------------------------------------------------ stores

"""
    FunctAI.ReplyStore

Where replies are kept. A store has `FunctAI.reply(store, key)` (the kept
reply, or `nothing`) and `FunctAI.keep_reply!(store, key, response)`, and
may have `claim_reply!(store, key, cancelled)` / `unclaim_reply!(store, key)`
(one flight per key) and `discard_reply!(store, key)`. Any `AbstractDict`
is a store too (`configure!(cache_replies = Dict())`): it keeps replies by
key.
"""
abstract type ReplyStore end

"""
    FunctAI.MemoryReplies(capacity = 20_000)

Replies kept in this process's memory, at most `capacity` (the least
recently used go first). `cache_replies = true` uses one per process.
"""
mutable struct MemoryReplies <: ReplyStore
    capacity::Int
    data::OrderedDict{String,Any}
    lock::ReentrantLock
    keys::KeyLocks
end
MemoryReplies(capacity::Integer=20_000) = MemoryReplies(Int(capacity), OrderedDict{String,Any}(), ReentrantLock(), KeyLocks())
durable(::MemoryReplies) = false
Base.length(m::MemoryReplies) = lock(() -> length(m.data), m.lock)
Base.show(io::IO, m::MemoryReplies) = print(io, "MemoryReplies(", length(m), " replies)")
function reply(m::MemoryReplies, key::AbstractString)
    lock(m.lock) do
        hit = get(m.data, key, nothing)
        hit === nothing && return nothing
        delete!(m.data, key)                   # most recently used last
        m.data[key] = hit
        hit
    end
end
function keep_reply!(m::MemoryReplies, key::AbstractString, response)
    lock(m.lock) do
        delete!(m.data, key)
        m.data[String(key)] = response
        while length(m.data) > m.capacity
            delete!(m.data, first(keys(m.data)))
        end
    end
    nothing
end
discard_reply!(m::MemoryReplies, key::AbstractString) = (lock(() -> delete!(m.data, key), m.lock); nothing)
Base.empty!(m::MemoryReplies) = (lock(() -> empty!(m.data), m.lock); m)
claim_reply!(m::MemoryReplies, key::AbstractString, cancelled=nothing) = (acquire!(m.keys, String(key), cancelled); reply(m, key))
unclaim_reply!(m::MemoryReplies, key::AbstractString) = release!(m.keys, String(key))

"The `cache_replies = :disk` file: the user's cache folder, `functai/replies.sqlite`."
function default_replies_path()
    base = Sys.isapple() ? joinpath(homedir(), "Library", "Caches") :
           Sys.iswindows() ? get(ENV, "LOCALAPPDATA", joinpath(homedir(), "AppData", "Local")) :
           (x = get(ENV, "XDG_CACHE_HOME", ""); isempty(x) ? joinpath(homedir(), ".cache") : x)
    joinpath(base, "functai", "replies.sqlite")
end

const REPLIES_SCHEMA = ["CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, format INTEGER NOT NULL, created TEXT NOT NULL, model TEXT, response TEXT NOT NULL)",
                        "CREATE TABLE IF NOT EXISTS claims (key TEXT PRIMARY KEY, owner TEXT NOT NULL, until REAL NOT NULL)"]

"""
    FunctAI.DiskReplies(path = default; lease = 120)

Replies kept in one SQLite file (contract/replies.md, "The file"), shared by
every process and thread that opens it, Python's, TypeScript's and R's
included: writes are transactions, so nothing is half-written; a claim with
a lease makes one flight per key across processes. The file and its folder
are readable by their owner only.
"""
mutable struct DiskReplies <: ReplyStore
    path::String
    lease::Float64
    db::Any
    pid::Int
    lock::ReentrantLock
    keys::KeyLocks
end
function DiskReplies(path=nothing; lease::Real=REPLY_LEASE)
    p = path === nothing ? default_replies_path() : abspath(expanduser(String(path)))
    any(endswith(p, x) for x in (".sqlite", ".sqlite3", ".db")) || (p = joinpath(p, "replies.sqlite"))
    mkpath(dirname(p); mode=0o700)
    fresh = !isfile(p)
    d = DiskReplies(p, Float64(lease), nothing, 0, ReentrantLock(), KeyLocks())
    with_db(d) do db
        for stmt in REPLIES_SCHEMA
            SQLite.execute(db, stmt)
        end
    end
    fresh && try
        chmod(p, 0o600)
    catch
    end
    d
end
durable(::DiskReplies) = true
Base.show(io::IO, d::DiskReplies) = print(io, "DiskReplies(", repr(d.path), ")")

"Run `f(db)` with this process's connection (one per process: a forked child opens its own), serialized in this process."
function with_db(f, d::DiskReplies)
    lock(d.lock) do
        if d.db === nothing || d.pid != getpid()
            db = SQLite.DB(d.path)
            SQLite.busy_timeout(db, 30_000)
            run_sql(db, "PRAGMA journal_mode=WAL")
            run_sql(db, "PRAGMA synchronous=NORMAL")
            d.db, d.pid = db, getpid()
        end
        f(d.db)
    end
end

"Run `f(db)` in one `BEGIN IMMEDIATE` transaction."
function transaction(f, d::DiskReplies)
    with_db(d) do db
        SQLite.execute(db, "BEGIN IMMEDIATE")
        try
            out = f(db)
            SQLite.execute(db, "COMMIT")
            out
        catch
            try
                SQLite.execute(db, "ROLLBACK")
            catch
            end
            rethrow()
        end
    end
end

"The first row of a query, as a NamedTuple of its values (`nothing`: none); the statement is finished before it returns."
function first_row(db, sql, params)
    q = DBInterface.execute(db, sql, params)
    try
        for row in q
            return NamedTuple{Tuple(propertynames(row))}(Tuple(getproperty(row, n) for n in propertynames(row)))
        end
        nothing
    finally
        DBInterface.close!(q)
    end
end

function reply(d::DiskReplies, key::AbstractString)
    row = with_db(db -> (r = first_row(db, "SELECT response, format FROM replies WHERE key = ?", (String(key),));
                         r === nothing ? nothing : (String(r.response), r.format)), d)
    (row === nothing || row[2] != REPLY_FORMAT) && return nothing     # a format this reader does not know: no reply
    try
        response_of(row[1])
    catch
        nothing                                                         # a row this reader cannot read is no reply
    end
end
function keep_reply!(d::DiskReplies, key::AbstractString, response)
    model = try
        response.model
    catch
        nothing
    end
    transaction(d) do db
        run_sql(db, "INSERT OR REPLACE INTO replies (key, format, created, model, response) VALUES (?, ?, ?, ?, ?)",
                (String(key), REPLY_FORMAT, iso(time()), model === nothing ? missing : String(model), response_text(response)))
        run_sql(db, "DELETE FROM claims WHERE key = ? AND owner = ?", (String(key), reply_owner()))
    end
    nothing
end
discard_reply!(d::DiskReplies, key::AbstractString) = (with_db(db -> run_sql(db, "DELETE FROM replies WHERE key = ?", (String(key),)), d); nothing)

"Run a statement that returns no rows, and finish it."
run_sql(db, sql, params=()) = DBInterface.close!(DBInterface.execute(db, sql, params))
function Base.empty!(d::DiskReplies)
    with_db(d) do db
        SQLite.execute(db, "DELETE FROM replies")
        SQLite.execute(db, "DELETE FROM claims")
    end
    d
end
Base.length(d::DiskReplies) = with_db(db -> Int(first_row(db, "SELECT COUNT(*) AS n FROM replies", ()).n), d)

"""
The reply to `key` if one is kept; else take the key (one flight): wait while
another task or process holds it, then take it (`nothing`: this caller asks
the model).
"""
function claim_reply!(d::DiskReplies, key::AbstractString, cancelled=nothing)
    k = String(key)
    acquire!(d.keys, k, cancelled)
    pause = 0.05
    try
        while true
            hit = reply(d, k)
            hit === nothing || return hit
            mine = transaction(d) do db
                now = time()
                row = first_row(db, "SELECT owner, until FROM claims WHERE key = ?", (k,))
                ok = row === nothing || String(row.owner) == reply_owner() || Float64(row.until) < now
                ok && run_sql(db, "INSERT OR REPLACE INTO claims (key, owner, until) VALUES (?, ?, ?)", (k, reply_owner(), now + d.lease))
                ok
            end
            mine && return nothing
            cancelled === nothing || cancelled()
            sleep(pause)
            pause = min(2pause, 0.5)
        end
    catch
        release!(d.keys, k)
        rethrow()
    end
end
function unclaim_reply!(d::DiskReplies, key::AbstractString)
    try
        with_db(db -> run_sql(db, "DELETE FROM claims WHERE key = ? AND owner = ?", (String(key), reply_owner())), d)
    finally
        release!(d.keys, String(key))
    end
end

# any AbstractDict keeps replies by key
reply(d::AbstractDict, key::AbstractString) = get(d, key, nothing)
keep_reply!(d::AbstractDict, key::AbstractString, response) = (d[String(key)] = response; nothing)
discard_reply!(d::AbstractDict, key::AbstractString) = (delete!(d, key); nothing)
durable(::AbstractDict) = false
durable(_) = true                      # a store of its own: it may keep what it is given, so it is held to log_content
claim_reply!(store, key::AbstractString, cancelled=nothing) = reply(store, key)
unclaim_reply!(store, key::AbstractString) = nothing
discard_reply!(store, key::AbstractString) = nothing

const MEMORY_REPLIES = MemoryReplies()
const DISK_REPLIES = Dict{String,DiskReplies}()
const DISK_REPLIES_LOCK = ReentrantLock()

is_reply_store(v) = v isa Union{ReplyStore,AbstractDict} || (hasmethod(reply, Tuple{typeof(v),String}) && hasmethod(keep_reply!, Tuple{typeof(v),String,Any}))

"A `cache_replies` setting, checked: `false`, `true` (memory), `:memory`, `:disk`, a folder or `.sqlite` path, or a store."
function cache_setting(v)
    v isa Bool && return v
    v isa Symbol && v in (:memory, :disk) && return v
    v isa AbstractString && !isempty(strip(v)) && return String(v)
    is_reply_store(v) && return v
    throw(ArgumentError("cache_replies is false, true (memory), :disk, a folder or .sqlite path, or a store " *
                        "(a FunctAI.ReplyStore, or a Dict); not $(repr(v))"))
end

"The store a `cache_replies` setting names, or `nothing` when off."
function reply_store(v)
    (v === nothing || v === false) && return nothing
    (v === true || v === :memory) && return MEMORY_REPLIES
    if v === :disk || v isa AbstractString
        where = v === :disk ? default_replies_path() : abspath(expanduser(v))
        return lock(DISK_REPLIES_LOCK) do
            get!(() -> DiskReplies(where), DISK_REPLIES, where)
        end
    end
    v
end

"""
    FunctAI.clear_cache(which = nothing)

Forget kept replies, so the next identical requests reach the model again:
this process's memory (`nothing`, `true`), or the store a `cache_replies`
value names (`:disk`, a path, a store).
"""
function clear_cache(which=nothing)
    store = which === nothing ? MEMORY_REPLIES : reply_store(which)
    store === nothing || empty!(store)
    nothing
end

"""
One request's turn at the cache: `hit` (a kept reply, or `nothing`: this
caller asks the model), then `keep!` once the reply was read, or `drop!`
when it could not be (a kept reply that no longer reads is forgotten);
`finish!` always.
"""
mutable struct Flight
    store::Any
    key::String
    hit::Any
    ended::Bool
end
function keep!(f::Flight, response)
    f.hit === nothing || return
    try
        keep_reply!(f.store, f.key, response)
    catch err
        err isa InterruptException && rethrow()
        warn_once("replies:$(typeof(err))", "a reply could not be kept in the cache ($(sprint(showerror, err)))")
    end
end
drop!(f::Flight) = f.hit === nothing || discard_reply!(f.store, f.key)
function finish!(f::Flight)
    f.ended && return
    f.ended = true
    unclaim_reply!(f.store, f.key)
end

"""
The cache's turn for this request under these settings, or `nothing` when no
cache applies. A durable store is skipped for a call whose `log_content`
drops a field (the memory cache serves it): it would keep what may not be kept.
"""
function begin_flight(s::AbstractDict{Symbol}, request, call)
    store = try
        reply_store(get(s, :cache_replies, nothing))
    catch err
        warn_once("replies-store:$(typeof(err))", "the reply cache cannot be opened ($(sprint(showerror, err))); replies are not cached")
        nothing
    end
    store === nothing && return nothing
    durable(store) && call !== nothing && !whole(call.keep) && (store = MEMORY_REPLIES)
    key = reply_key(request, something(get(s, :replicate, nothing), 0))
    hit = claim_reply!(store, key, call === nothing ? nothing : () -> check_cancelled(call))
    Flight(store, key, hit, false)
end
