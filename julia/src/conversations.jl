# Conversations (contract/conversations.md): a program's calls that remember
# each other.
#
#     chat = conversation(tutor, "alex"; store = "tutoring/")   # the same line tomorrow reopens it
#     chat("Hi, I'm Alex.")                                      # called like its program: one turn
#     last(turns(chat)).saw                                      # the earlier turns that answer was based on
#     other = continue_from(chat, turns(chat)[1])                # a branch: nothing is ever deleted
#     render(chat, "What is 1/2 + 1/3?")                         # the next request, nothing sent
#
# A conversation is records in a store (stores.jl), appended and never
# changed: a `turn` record before the call (the turn's id is its call's id,
# known before the model is asked), `ended` after it, `lease` records while
# it runs, and what resuming it needs (`reply`, `tool`, `waiting`,
# `approval`). Memory belongs to the conversation, never to the function:
# `tutor` is unchanged and still callable on its own.

const LEASE_SECONDS = 30.0          # how long a running turn's lease lasts
const RENEW_SECONDS = 10.0          # how often its process renews it
const STOP_POLL = 0.25              # how often a running turn looks for a stop from another process

"The clock turn states are read by (the contract's cases fix it)."
const CONVERSATION_CLOCK = Ref{Function}(time)
conversation_now() = CONVERSATION_CLOCK[]()

# ------------------------------------------------------------------ what the model sees

"""
Which earlier turns a turn is shown: every one (`last` `nothing`) or the last
`last`; `without`, inputs or outputs left out of every earlier turn. Made by
[`all_turns`](@ref) and [`last_turns`](@ref).
"""
struct ContextRule
    last::Union{Nothing,Int}
    without::Vector{String}
end
pick(r::ContextRule, ts) = r.last === nothing ? collect(ts) : r.last > 0 ? collect(ts[max(1, end - r.last + 1):end]) : empty(collect(ts))

"""
    all_turns(; without = String[])

Show the model every earlier turn of the conversation (the default). Running
out of the model's context, and being told, is better than a model that
silently misses what was said. `without`: inputs or outputs left out of
every *earlier* turn (a long document the model already answered about); the
current turn is always whole.
"""
all_turns(; without=String[]) = ContextRule(nothing, String[String(x) for x in without])

"""
    last_turns(n; without = String[])

Show the model only the last `n` earlier turns. The turns left out are still
kept, and each turn's record says which earlier turns it was shown
(`turn.saw`), so an answer can be asked again exactly as it was.
"""
function last_turns(n::Integer; without=String[])
    n >= 0 || throw(ArgumentError("last_turns takes a whole number of turns, not $n"))
    ContextRule(Int(n), String[String(x) for x in without])
end

"""
What a helper remembers in a module's conversation: `:conversation` (its own
earlier calls on this branch, in earlier turns and this one) or `:turn`
(its earlier calls in this turn only); `steps`: with their tool steps.
"""
struct Memory
    mode::Symbol
    steps::Bool
end

"""
    remember(mode = :conversation; steps = false)

What an AI function called inside a module's conversation remembers. In a
module's conversation, the module's turns remember each other, but the AI
functions it calls (its helpers) start fresh at every call unless the
conversation says otherwise: `conversation(support; remembers = Dict(answer
=> remember(:conversation; steps = true)))`. A plain `:conversation` or
`:turn` there is the same as `remember(:conversation)`.
"""
function remember(mode=:conversation; steps::Bool=false)
    Symbol(mode) in (:conversation, :turn) || throw(ArgumentError("a helper remembers :conversation or :turn, not $(repr(mode))"))
    Memory(Symbol(mode), steps)
end
memory_of(v::Memory) = v
memory_of(v) = Symbol(v) === :own ? :own : remember(v)

# ------------------------------------------------------------------ the records, read

"What the records say of one turn."
mutable struct TurnState
    record::JObj
    ended::Union{Nothing,JObj}
    lease::Union{Nothing,JObj}
    waiting::Union{Nothing,JObj}
    stops::Int
    answers::Vector{JObj}
    tools::Vector{JObj}
    replies::Vector{JObj}
    calls::Vector{JObj}
    children::Vector{String}
    attempt::Int
end
TurnState(r::JObj) = TurnState(r, nothing, nothing, nothing, 0, JObj[], JObj[], JObj[], JObj[], String[], 1)
turn_id(st::TurnState) = String(st.record["turn"])
turn_parent(st::TurnState) = get(st.record, "parent", nothing)
seq_of(r) = r === nothing ? 0 : Int(get(r, "seq", 0))
ended_outputs(st::TurnState) = st.ended === nothing ? JObj() : JObj(something(get(st.ended, "outputs", nothing), JObj()))

"""
A turn's state from its records (contract/conversations.md, "A turn's
state"): how it ended; else `waiting` after its last lease; else `running`
while its lease holds; else `interrupted` (its process stopped).
"""
function turn_state(st::TurnState, now=conversation_now())
    st.ended === nothing || return String(st.ended["state"])
    st.waiting !== nothing && seq_of(st.waiting) > seq_of(st.lease) && return "waiting"
    until = st.lease === nothing ? 0.0 : unix_of(String(get(st.lease, "until", "")))
    until >= now ? "running" : "interrupted"
end

"Which question an approval record answers: the asking call's site, the tool call's invocation, the plugin that asked."
approval_key(a) = (get(a, "site", nothing), Int(something(get(a, "invocation", nothing), 0)), String(something(get(a, "plugin", nothing), "approval")))

"The approvals a waiting turn waits for that have no answer yet."
function unanswered(st::TurnState)
    st.waiting === nothing && return Any[]
    done = Set(approval_key(a) for a in st.answers if seq_of(a) > seq_of(st.waiting))
    Any[a for a in something(get(st.waiting, "approvals", nothing), Any[]) if !(approval_key(a) in done)]
end

"Tools that started and have no result in the records: they may have run."
function unfinished(st::TurnState)
    started = OrderedDict{Any,JObj}()
    for t in st.tools
        k = (get(t, "site", nothing), get(t, "invocation", nothing))
        state = get(t, "state", nothing)
        state == "started" && (started[k] = t)
        state in ("done", "given", "rerun") && delete!(started, k)
    end
    collect(values(started))
end

"A conversation's records, applied in order (read again incrementally)."
mutable struct ConvLog
    n::Int
    turns::Dict{String,TurnState}
    order::Vector{String}
    head::Union{Nothing,String}
    request_ids::Dict{String,String}
    programs::Dict{String,JObj}
    entries::Vector{JObj}
end
ConvLog() = ConvLog(0, Dict{String,TurnState}(), String[], nothing, Dict{String,String}(), Dict{String,JObj}(), JObj[])

function apply!(log::ConvLog, records)
    for r in records
        log.n = max(log.n, Int(something(get(r, "seq", nothing), log.n + 1)))
        get(r, "functai_conversation", nothing) == CONVERSATION_FORMAT || continue    # a format it does not know: skipped
        kind = get(r, "kind", nothing)
        if kind == "program"
            haskey(log.programs, r["version"]) || (log.programs[r["version"]] = r)
            continue
        elseif kind == "entry"
            push!(log.entries, r)
            continue
        end
        tid = get(r, "turn", nothing)
        tid isa AbstractString || continue
        if kind == "turn"
            haskey(log.turns, tid) && continue
            log.turns[tid] = TurnState(r)
            push!(log.order, tid)
            p = get(r, "parent", nothing)
            p !== nothing && haskey(log.turns, p) && push!(log.turns[p].children, tid)
            rid = get(r, "request_id", nothing)
            rid isa AbstractString && !haskey(log.request_ids, rid) && (log.request_ids[rid] = tid)
            log.head = tid
            continue
        end
        st = get(log.turns, tid, nothing)
        if kind == "head"
            st === nothing || (log.head = tid)
            continue
        end
        st === nothing && continue
        if kind == "ended"
            st.ended === nothing && (st.ended = r)
        elseif kind == "lease"
            st.lease = r
            st.attempt = max(st.attempt, Int(something(get(r, "attempt", nothing), 1)))
        elseif kind == "waiting"
            st.waiting = r
        elseif kind == "stop"
            st.stops += 1
        elseif kind == "approval"
            push!(st.answers, r)
        elseif kind == "tool"
            push!(st.tools, r)
        elseif kind == "reply"
            push!(st.replies, r)
        elseif kind == "call"
            push!(st.calls, r)
        end
    end
    log
end

"The turns from the first to `turn`, in order."
function branch(log::ConvLog, turn)
    out = TurnState[]
    seen = Set{String}()
    while turn !== nothing && haskey(log.turns, turn) && !(turn in seen)
        push!(seen, turn)
        push!(out, log.turns[turn])
        turn = turn_parent(log.turns[turn])
    end
    reverse!(out)
end

"`turn` when it ended `done`, else its nearest ancestor that did: a turn that did not end `done` is never a parent."
function done_on(log::ConvLog, turn)
    while turn !== nothing && haskey(log.turns, turn)
        turn_state(log.turns[turn]) == "done" && return turn
        turn = turn_parent(log.turns[turn])
    end
    nothing
end

# ------------------------------------------------------------------ programs, described

program_key(f::AIFunction) = (f.definition.name, f.module_name)
program_key(p::AIProgram) = (p.name, p.module_name)
program_name(f::AIFunction) = f.definition.name
program_name(p::AIProgram) = p.name
program_version(f::AIFunction) = version(f)
program_version(p::AIProgram) = version(p)

"A program's fields as data (names, directions, purposes, shapes without type names or defaults): what decides whether an earlier turn can be shown whole."
function program_fields_data(f::AIFunction)
    sig = signature_now(f, effective(f.own))
    Any[LMCC.jobj("name" => x.name, "direction" => x.direction, "purpose" => x.purpose, "shape" => data_shape(no_type_names(x.shape)))
        for x in sig.fields if !(x.direction == "input" && x.purpose != "plain")]
end
program_fields_data(p::AIProgram) = Any[LMCC.jobj("name" => x["name"], "direction" => d[1:end-1], "purpose" => "plain", "shape" => data_shape(x["shape"]))
                                        for d in ("inputs", "outputs") for x in p.interface[d]]
no_type_names(shape) = shape

"A program's descriptor record (contract/conversations.md, `program`)."
function program_record(fn)
    info = program_of(fn)
    out = LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "program", "at" => iso(time()), "version" => info["version"],
                    "name" => info["name"], "program_kind" => info["kind"], "module" => info["module"],
                    "interface" => LMCC.deepcopy_json(interface(fn)), "fields" => program_fields_data(fn), "answer" => info["answer"])
    haskey(info, "signature") && (out["signature"] = info["signature"])
    out
end

"""
Refuse a program that cannot be shown its earlier turns (contract/
conversations.md, "The program changed"): it now writes an output they lack,
unless `earlier_without` names it and it is one FunctAI adds (reasoning,
tool calls); or a field changed its shape or went away.
"""
function check_conversation_signature(fn, log::ConvLog, ts, earlier_without)
    now = Dict(x["name"] => x for x in program_fields_data(fn))
    seen = Set{String}()
    for st in ts
        v = get(st.record, "program", nothing)
        (v === nothing || v in seen || !haskey(log.programs, v)) && continue
        push!(seen, v)
        was = Dict(x["name"] => x for x in something(get(log.programs[v], "fields", nothing), Any[]))
        same = length(was) == length(now) && all(k -> haskey(now, k) && same_json(was[k], now[k]), keys(was))
        same && continue
        changed = sort!([n for n in intersect(keys(was), keys(now)) if !same_json(was[n], now[n])])
        gone = sort!(collect(setdiff(keys(was), keys(now))))
        new = sort!(collect(setdiff(keys(now), keys(was))))
        allowed = Set(n for n in new if now[n]["purpose"] in ("reasoning", "tools.calls") && n in earlier_without)
        unexplained = [n for n in new if !(n in allowed)]
        isempty(changed) && isempty(gone) && isempty(unexplained) && continue
        parts = String[]
        if !isempty(unexplained)
            hidden = [n for n in unexplained if now[n]["purpose"] in ("reasoning", "tools.calls")]
            push!(parts, "it now writes $(join(unexplained, ", ")), which earlier turns lack" *
                         (isempty(hidden) ? "" : " (to go on: earlier_without = $(repr(hidden)); earlier turns are shown without them, nothing is rewritten)"))
        end
        isempty(changed) || push!(parts, "$(join(changed, ", ")) changed type")
        isempty(gone) || push!(parts, "$(join(gone, ", ")) is no longer one of its fields")
        throw(ConversationError("conversation-signature", "$(program_name(fn)): its earlier turns in this conversation were made with " *
                                                          "other inputs or outputs: $(join(parts, "; "))"))
    end
end

# ------------------------------------------------------------------ the turn running in this process

"""
One turn being run by this process: its place, what its calls are shown,
what resuming it replays, and the records it writes as it goes.
"""
mutable struct TurnRun
    conv::Any
    store::Any
    program::Any
    turn::String
    attempt::Int
    context::Dict{Symbol,Any}           # :turns, :ids, :finish, :rows, :parent, :sections, :changes, :recorded, :module_saw
    root_taken::Ref{Bool}
    helper_calls::Vector{JObj}
    usage::JObj
    later::Any                          # a resumed turn's claim on its log: (writer, after, at, requests)
    start_seq::Int
    settings::Dict{Symbol,Any}
    lock::ReentrantLock
    durable::Bool
    replies::Dict{String,Vector{Any}}   # a resumed turn's recorded replies, by key, in order
    tools::Dict{Any,JObj}               # (site, invocation) => its done (or given) tool record
    unfinished::Dict{Any,JObj}
    answers::Dict{Any,Tuple{Bool,Any,Any,Bool}}   # (site, invocation, plugin) => (allowed, reason, by, fresh)
    root::Any
    sink::Any
end
function TurnRun(conv, turn::AbstractString; attempt::Int, context::Dict{Symbol,Any}, replay=nothing, later=nothing)
    run = TurnRun(conv, conv.store, conv.program, String(turn), attempt, context, Ref(false), JObj[], JObj(), later, 0,
                  Dict{Symbol,Any}(), ReentrantLock(), is_durable(conv), Dict{String,Vector{Any}}(), Dict{Any,JObj}(),
                  Dict{Any,JObj}(), Dict{Any,Tuple{Bool,Any,Any,Bool}}(), nothing, nothing)
    r = something(replay, Dict{Symbol,Any}())
    for rec in get(r, :replies, Any[])
        push!(get!(() -> Any[], run.replies, String(rec["key"])), rec["response"])
    end
    for t in get(r, :tools, Any[])
        k = (get(t, "site", nothing), Int(something(get(t, "invocation", nothing), 0)))
        state = get(t, "state", nothing)
        if state in ("done", "given")
            run.tools[k] = t
            delete!(run.unfinished, k)
        elseif state == "rerun"
            delete!(run.tools, k)
            delete!(run.unfinished, k)
        elseif state == "started" && !haskey(run.tools, k)
            run.unfinished[k] = t
        end
    end
    last_wait = get(r, :waiting_seq, 0)
    for a in get(r, :answers, Any[])
        run.answers[approval_key(a)] = (get(a, "verdict", nothing) == "yes", get(a, "reason", nothing), get(a, "by", nothing),
                                        seq_of(a) > last_wait)
    end
    run
end

record_base(run::TurnRun, kind) = LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => kind, "at" => iso(time()),
                                            "turn" => run.turn, "attempt" => run.attempt)

"The turn's store keeps the turn's kept log, so another process (a page after a reload) can watch it."
struct StoreSink
    store::Any
    warned::Ref{Bool}
end
function (s::StoreSink)(e::Event)
    try
        keep!(s.store, e)
    catch err
        EXITING[] && return
        DEBUG_TURNS[] && push!(SINK_ERRORS, (e, err))
        s.warned[] || (s.warned[] = true; warn_once("conversation-events:$(typeof(err))",
            "a turn's events could not be kept in its store ($(sprint(showerror, unwrap(err)))); the turn goes on, and its records are kept"))
    end
    nothing
end

function add_sink!(tree::TreeLog, run::TurnRun, later)
    events = event_store(run.store)
    events === nothing && return
    sink = StoreSink(events, Ref(false))
    run.sink = sink
    lock(tree.lock) do
        push!(tree.sinks, sink)
        tree.observer_last[sink] = later === nothing ? nothing : later.after
    end
end

"The turn's own call: its record says which conversation and turn it is; the plugins' changes of the turn are on it."
function attach_turn!(run::TurnRun, call::Call)
    run.root = call
    call.conversation = LMCC.jobj("id" => run.conv.id, "turn" => run.turn, "parent" => get(run.context, :parent, nothing))
    append!(call.changes, LMCC.deepcopy_json(get(run.context, :changes, Any[])))
end

"A call of the turn ended: its tokens count in the turn's usage; a helper the conversation remembers keeps its call for later."
function call_ended!(run::TurnRun, call::Call, err)
    lock(run.lock) do
        for (k, v) in own_usage(call)
            run.usage[k] = get(run.usage, k, 0) + v
        end
    end
    call.id == run.turn && return
    (err !== nothing || call.lmcc === nothing || !(call.fn isa AIFunction)) && return
    memory = remembered(run.conv, call.fn)
    memory isa Memory || return
    name, mod = program_key(call.fn)
    rec = record_base(run, "call")
    rec["call"] = call.id
    rec["site"] = call.site
    rec["program"] = LMCC.jobj("name" => name, "module" => mod, "signature" => signature_id(call.fn))
    rec["lmcc"] = LMCC.deepcopy_json(call.lmcc)
    rec["saw"] = LMCC.deepcopy_json(call.saw)
    lock(() -> push!(run.helper_calls, rec), run.lock)
    append_records!(run.store, run.conv.id, [rec])
end

"A remembered helper's own earlier calls on this branch (and in this turn), as turns (with their steps only when remembered with them)."
function helper_context(run::TurnRun, f::AIFunction, memory::Memory)
    name, mod = program_key(f)
    found = JObj[]
    if memory.mode === :conversation
        log = read_log!(run.conv)
        for st in branch(log, get(run.context, :parent, nothing))
            turn_state(st) == "done" || continue
            final = Int(something(st.ended === nothing ? nothing : get(st.ended, "attempt", nothing), st.attempt))
            append!(found, [c for c in st.calls if Int(something(get(c, "attempt", nothing), 1)) == final])
        end
    end
    lock(() -> append!(found, run.helper_calls), run.lock)
    sig = signature_id(f)
    ts, ids = Any[], String[]
    for c in found
        (c["program"]["name"] == name && get(c["program"], "module", nothing) == mod) || continue
        t = LMCC.deepcopy_json(c["lmcc"])
        memory.steps || (t["steps"] = Any[])
        get(c["program"], "signature", nothing) == sig || (t = LMCC.jobj("inputs" => get(t, "inputs", JObj()), "outputs" => something(get(t, "outputs", nothing), JObj())))
        push!(ts, fitted(f, t))
        push!(ids, String(c["call"]))
    end
    (ts, ids)
end

# ----- resuming: what was recorded

"A reply a resumed turn recorded for this very request, or `nothing` (the turn then does something new: its events are shown from here)."
function recorded_reply(call::Call, request, s)
    run = call.turn_run
    run === nothing && return nothing
    hit = nothing
    if !isempty(run.replies)
        q = get(run.replies, reply_key(request, something(get(s, :replicate, nothing), 0)), nothing)
        q === nothing || isempty(q) || (hit = LM15.from_dict(LM15.Response, popfirst!(q)))
    end
    hit === nothing && frontier!(call.tree)
    hit
end

"A stored turn keeps every reply (a module's, or an AI function's with tools), so resuming it pays for none twice."
function note_reply(call::Call, request, response, s)
    run = call.turn_run
    (run === nothing || !run.durable) && return
    try
        rec = record_base(run, "reply")
        rec["key"] = reply_key(request, something(get(s, :replicate, nothing), 0))
        rec["response"] = LM15.to_dict(response)
        append_records!(run.store, run.conv.id, [rec])
    catch err
        err isa InterruptException && rethrow()
        warn_once("note-reply:$(typeof(err))", "a reply could not be kept in the conversation ($(sprint(showerror, err)))")
    end
end

function recorded_tool(run::TurnRun, call::Call, a::Approval)
    k = (call.site, a.invocation)
    haskey(run.tools, k) && return String(something(get(run.tools[k], "output", nothing), ""))
    haskey(run.unfinished, k) && throw(ConversationError("turn-unfinished",
        "$(a.path) started before the turn stopped, and whether it ran is not known: resume!(turn; results = Dict($(a.invocation) => " *
        "what it returned)) or resume!(turn; rerun = [$(a.invocation)])"; turn=run.turn))
    nothing
end
function tool_started!(run::TurnRun, call::Call, a::Approval)
    rec = record_base(run, "tool")
    merge!(rec, LMCC.jobj("site" => call.site, "invocation" => a.invocation, "id" => a.id, "name" => a.name,
                          "input" => logvalue(a.input), "effects" => a.effects, "state" => "started"))
    append_records!(run.store, run.conv.id, [rec])
end
function tool_done!(run::TurnRun, call::Call, a::Approval, output)
    rec = record_base(run, "tool")
    merge!(rec, LMCC.jobj("site" => call.site, "invocation" => a.invocation, "id" => a.id, "name" => a.name, "state" => "done",
                          "output" => String(output)))
    append_records!(run.store, run.conv.id, [rec])
end
recorded_approval(run::TurnRun, call::Call, a::Approval) = get(run.answers, (call.site, a.invocation, a.plugin), nothing)
function note_approval!(run::TurnRun, call::Call, a::Approval, allowed::Bool, reason, by)
    rec = LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "approval", "at" => iso(time()), "turn" => run.turn,
                    "site" => call.site, "invocation" => a.invocation, "path" => a.path, "plugin" => a.plugin,
                    "verdict" => allowed ? "yes" : "no", "by" => by, "reason" => reason)
    append_records!(run.store, run.conv.id, [rec])
end

# ------------------------------------------------------------------ what a call is shown

"A stored turn made for this function's signature, with the fingerprint lmcc checks (this plan's); by its values when no plan can be made."
function fitted(f::AIFunction, t::AbstractDict)
    haskey(t, "signature") || return t
    try
        plan = plan_for(f, effective(f.own)).plan
        out = copy(t)
        out["signature"] = LMCC.fingerprint(plan)
        out
    catch
        LMCC.jobj("inputs" => get(t, "inputs", JObj()), "outputs" => something(get(t, "outputs", nothing), JObj()))
    end
end

"An earlier turn as this plan shows it: whole when made for it (a recorded turn), else an example of its values; `nothing` when no output is left."
function fit_turn(plan, f::AIFunction, d::AbstractDict)
    if haskey(d, "signature") && haskey(d, "steps") && d["signature"] == LMCC.fingerprint(plan)
        try
            return (LMCC.load_turn(plan, d), true)
        catch err
            err isa LMCC.Refusal || rethrow()
        end
    end
    fields = plan.signature.fields
    plain = Set(x.name for x in fields if x.direction == "input" && x.purpose == "plain")
    kept = Set(x.name for x in fields if x.direction == "output" && x.purpose in ("plain", "reasoning"))
    ins = prepare_inputs(plan.signature, JObj(k => v for (k, v) in something(get(d, "inputs", nothing), JObj()) if k in plain))
    outs = JObj(k => v for (k, v) in something(get(d, "outputs", nothing), JObj()) if k in kept)
    isempty(outs) && return (nothing, false)
    try
        (LMCC.example(plan, ins, outs), false)
    catch err
        err isa LMCC.Refusal || rethrow()
        (nothing, false)
    end
end

"""
How a plan shows earlier turns, and the `saw` entries that say so: a turn
made for that plan whole, with its steps; one made for another plan as its
values alone (`without` the fields the plan no longer has); one with no
output left, not at all. An empty id: a turn no logged call made.
"""
function shown_as(plan, f::AIFunction, ts, ids)
    entries, shown = Any[], Any[]
    for (t, cid) in zip(ts, ids)
        got, whole_turn = plan === nothing ? (t, false) : fit_turn(plan, f, t)
        got === nothing && continue
        if isempty(cid)
            push!(entries, LMCC.jobj("unrecorded" => true))
        elseif whole_turn
            push!(entries, !isempty(something(get(t, "steps", nothing), Any[])) ? LMCC.jobj("call" => cid, "steps" => true) : LMCC.jobj("call" => cid))
        else
            had = union(keys(something(get(t, "inputs", nothing), JObj())), keys(something(get(t, "outputs", nothing), JObj())))
            now = got isa LMCC.Turn ? union(keys(got.inputs), keys(something(got.outputs, JObj()))) : had
            left = sort!(collect(setdiff(had, now)))
            push!(entries, isempty(left) ? LMCC.jobj("call" => cid) : LMCC.jobj("call" => cid, "without" => Any[left...]))
        end
        push!(shown, got)
    end
    (entries, shown)
end

"Internal: a rated row asked again with its context (evaluate, the optimizers), or a conversation's next request being rendered."
const REPLAYING = ScopedValue{Any}(nothing)
const RENDERING = ScopedValue{Any}(nothing)

"What a call of `f` is shown as earlier turns: `(turns, ids, finish)`, or `nothing`."
function context_for(f::AIFunction, call::Call)
    replay = REPLAYING[]
    if replay !== nothing
        found = replay_context(replay, f, call)
        found === nothing || return found
    end
    run = call.turn_run
    run === nothing && return nothing
    if call.id == run.turn
        return (run.context[:turns], run.context[:ids], run.context[:finish])
    end
    memory = remembered(run.conv, f)
    memory isa Memory || return nothing
    ts, ids = helper_context(run, f, memory)
    (ts, ids, nothing)
end

"What an AI function's call is shown of its conversation, its `saw`, worked out before it starts (the plan it would have)."
function prepare_context!(call::Call, f::AIFunction, s)
    found = context_for(f, call)
    found === nothing && return
    ts, ids, finish = found
    plan = try
        plan_for(f, s).plan
    catch
        nothing                                   # no model to plan for: the call fails there, and says why
    end
    entries, shown = shown_as(plan, f, ts, ids)
    finish === nothing || (entries = finish(entries))
    call.saw = entries
    call.context = (plan=plan === nothing ? nothing : LMCC.fingerprint(plan), turns=ts, ids=ids, shown=shown)
end

"The earlier turns a call shows, as the plan it runs on shows them."
function shown_turns(call::Call, f::AIFunction, plan)
    ctx = call.context
    ctx === nothing && return Any[]
    ctx.plan == LMCC.fingerprint(plan) && return Any[ctx.shown...]
    Any[last(shown_as(plan, f, ctx.turns, ctx.ids))...]
end

"What `render` shows as earlier turns: the conversation's next turn's, when a conversation renders it."
function rendering_turns(f::AIFunction, plan)
    r = RENDERING[]
    (r === nothing || r.program !== f) && return Any[]
    Any[last(shown_as(plan, f, r.turns, r.ids))...]
end

"The instruction sections a call is given before its own `before_call` hooks: its turn's (the conversation's context hooks), or a row's asked again."
function context_sections(call)
    replay = REPLAYING[]
    call === nothing && (r = RENDERING[]; return r === nothing ? String[] : String[r.sections...])
    replay !== nothing && return replay_sections(replay, call)
    run = call.turn_run
    (run !== nothing && call.id == run.turn) || return String[]
    String[get(run.context, :sections, String[])...]
end

"What a module's call is shown as the conversation so far (its `saw`): a module's turn, or a row asked again."
function module_saw(call::Call, p::AIProgram)
    replay = REPLAYING[]
    replay !== nothing && call.parent === nothing && return replay_module_saw(replay)
    run = call.turn_run
    (run !== nothing && call.id == run.turn) || return Any[]
    Any[get(run.context, :module_saw, Any[])...]
end

"""
    earlier() -> Vector

The conversation so far, as data: inside a module's turn, one row per
earlier turn it is shown (its inputs and outputs by name, without the
fields the conversation leaves out); `[]` outside a conversation. For a
helper that declares an input for it:

```julia
@ai function handoff(conversation::Vector{Dict{String,String}})::String
    "Summarize this support conversation for the person who takes it over."
end
@program function support(message::String)::String
    topic(message) == "other" && notify_staff(handoff(FunctAI.earlier()))
    …
end
```
"""
function earlier()
    replay = REPLAYING[]
    replay !== nothing && return replay_rows(replay)
    call = CURRENT_CALL[]
    while call !== nothing && call.up !== nothing
        call = call.up
    end
    run = call === nothing ? TURN_STARTING[] : call.turn_run
    run === nothing && return Any[]
    LMCC.deepcopy_json(Any[get(run.context, :rows, Any[])...])
end

# ------------------------------------------------------------------ the conversation

"""
    Conversation

A program's conversation: its turns, kept in a store, called like the
program. Made by [`conversation`](@ref).
"""
mutable struct Conversation
    program::Any
    id::String
    store::Any
    context::ContextRule
    earlier_without::Vector{String}
    sends::Symbol
    settings::Dict{Symbol,Any}
    remembers::Vector{Pair{Any,Any}}
    head::Any                         # FOLLOW_HEAD (opened by id), or a turn this view continues from
    exact::Bool                       # continue_from(turn): after that very turn
    delegated::Bool                   # a delegate's own conversation, inside the turn that asked
    log::ConvLog
    log_lock::ReentrantLock
    send_lock::ReentrantLock
end
struct FollowHead end
const FOLLOW_HEAD = FollowHead()

"""
    conversation(program, id = nothing; store, context, earlier_without, remembers, sends, settings...)

A conversation with an AI function or a program: called like it, each call
is a turn that sees the earlier ones. Memory belongs to the conversation;
the program is unchanged.

```julia
chat = conversation(tutor, "alex"; store = "tutoring/")
chat("Hi, I'm Alex.")
chat("What is 1/2 + 1/3?")            # sees the first turn
predict(chat, "Is it 5/6?").outputs   # one turn, everything it produced
s = stream(chat, "Why?")              # watched while it is made; s.turn is known at once
```

- `id`: letters, digits, `.`, `_`, `-`. The same id in the same store opens
  the same conversation. Default: a new one.
- `store`: `nothing` (this process's memory), a folder (`FolderStore`),
  `true` (the default folder), or a [`FunctAI.ConversationStore`](@ref).
- `context`: [`all_turns`](@ref)`()` (default) or [`last_turns`](@ref)`(10)`,
  each with `without = [...]`.
- `earlier_without`: outputs the program now writes that earlier turns lack
  (reasoning turned on, a first tool): earlier turns are shown without them.
- `remembers`: a program's helpers' memory: `Dict(answer => :conversation)`,
  `:turn`, [`remember`](@ref)`(:conversation; steps = true)`, or `:own` for a
  conversation used inside it. Helpers remember nothing otherwise.
- `sends`: two sends at once: `:queue` (default: the second waits and
  continues from the first), `:refuse` (`conversation-busy`), or `:branch`.
- other keywords are settings for every turn (`approve`, `lm`, `plugins`, …).
"""
function conversation(program, id=nothing; store=nothing, context::ContextRule=all_turns(), earlier_without=String[],
                      remembers=nothing, sends=:queue, delegated::Bool=false, settings...)
    program isa Union{AIFunction,AIProgram} || throw(ArgumentError("a conversation is with an AI function or a program, not a $(typeof(program))"))
    Symbol(sends) in (:queue, :refuse, :branch) || throw(ArgumentError("sends is :queue, :refuse or :branch, not $(repr(sends))"))
    mems = Pair{Any,Any}[]
    for (k, v) in something(remembers, Dict())
        program isa AIFunction && throw(ArgumentError("remembers is for a program's helpers; an AI function's conversation is its own memory"))
        push!(mems, k => memory_of(v))
    end
    c = Conversation(program, id === nothing ? new_id() : check_conversation_id(id), conversation_store(store), context,
                     String[String(x) for x in earlier_without], Symbol(sends), settings_dict(settings), mems, FOLLOW_HEAD, false,
                     delegated, ConvLog(), ReentrantLock(), ReentrantLock())
    check_remembers(c)
    check_content(c)
    iface = interface(program)
    opaque = [x["name"] for d in ("inputs", "outputs") for x in iface[d] if get(x, "opaque", false) === true]
    isempty(opaque) || throw(ConversationError("conversation-opaque",
        "$(program_name(program)): $(join(opaque, ", ")) may hold values with no JSON form, and a conversation keeps its turns as data " *
        "(programs.md, opaque). Give $(length(opaque) > 1 ? "them" : "it") a type"))
    log = read_log!(c)
    check_conversation_signature(program, log, branch(log, view_head(c, log)), c.earlier_without)
    c
end

Base.show(io::IO, c::Conversation) = print(io, "Conversation(", repr(c.id), " with ", program_name(c.program), ")")
function Base.show(io::IO, ::MIME"text/plain", c::Conversation)
    ts = turns(c)
    print(io, "Conversation ", c.id, " with ", program_name(c.program), ": ", length(ts), " turn", length(ts) == 1 ? "" : "s")
    for (i, t) in enumerate(ts)
        ins = join((short_text(v, 50) for v in values(t.inputs)), " ")
        out = t.state == "done" ? short_text(t.result, 70) : "[$(t.state)]"
        print(io, "\n", lpad(i, 3), ". ", ins, " → ", out)
    end
end
function short_text(v, n=60)
    text = join(split(v isa AbstractString ? v : repr(v)), " ")
    length(text) <= n ? text : first(text, n - 1) * "…"
end

function check_remembers(c::Conversation)
    isempty(c.remembers) && return
    reachable = c.program isa AIProgram ? [x for x in reaches(c.program) if x isa AIFunction] : AIFunction[]
    for (k, v) in c.remembers
        v === :own && continue
        any(f -> f === k, reachable) || throw(ArgumentError("remembers names $(k isa AIFunction ? k.definition.name : repr(k)), which " *
            "$(program_name(c.program)) does not call (its AI functions: $(isempty(reachable) ? "none" : join((f.definition.name for f in reachable), ", ")))"))
    end
end

"A conversation whose store keeps records refuses a program a `log_content` layer drops a field of (refuse, never forget)."
function check_content(c::Conversation)
    persistent(c.store) || return
    fn = c.program
    fields = fn isa AIFunction ? program_fields(fn, effective(fn.own)) :
             (inputs=[x["name"] for x in fn.interface["inputs"]], outputs=[x["name"] for x in fn.interface["outputs"]], added=String[])
    dropped = dropped_fields(fn isa AIFunction ? fn.own : Dict{Symbol,Any}(), fields)
    isempty(dropped) || throw(ConversationError("conversation-content",
        "$(program_name(fn)): a log_content setting keeps $(join(dropped, ", ")) out of every record, and this store ($(typeof(c.store))) " *
        "keeps a conversation's records. A conversation that must remember what it may not keep refuses rather than forgets: keep it in " *
        "memory (store = nothing), or let the store keep those fields"))
end

"Whether turns record what resuming needs (their replies): a program's, or an AI function's with tools."
is_durable(c::Conversation) = !(c.program isa AIFunction) || !isempty(c.program.tools)

function remembered(c::Conversation, fn)
    for (k, v) in c.remembers
        k === fn && return v
    end
    nothing
end

function read_log!(c::Conversation)
    lock(c.log_lock) do
        apply!(c.log, read_records(c.store, c.id, c.log.n))
    end
end
add_records!(c::Conversation, records; expect=nothing) = append_records!(c.store, c.id, records; expect)

"""
Where this view continues: a conversation opened by id, its head; once it
made a turn, or was made by `continue_from`, its own branch: the latest turn
that is that turn or continues it.
"""
function view_head(c::Conversation, log::ConvLog)
    c.head === FOLLOW_HEAD && return log.head
    pinned = c.head
    c.exact && return pinned
    for tid in Iterators.reverse(log.order)
        t = tid
        while t !== nothing
            t == pinned && return tid
            st = get(log.turns, t, nothing)
            t = st === nothing ? nothing : turn_parent(st)
        end
    end
    pinned
end

"Each turn as a record `saw` reads (its id and its saw)."
saw_records(log::ConvLog) = [LMCC.jobj("functai_call" => 2, "id" => tid,
                                       "saw" => Any[something(st.ended === nothing ? nothing : get(st.ended, "saw", nothing), get(st.record, "saw", Any[]))...])
                             for (tid, st) in log.turns]

# ------------------------------------------------------------------ turns, as their records say

"""
    Turn

One turn of a conversation, as its records say now: `id` (also `call`: its
call's id in the call log), `parent`, `inputs`, `outputs`, `result` (the
answer, typed), `state` (`"running"`, `"waiting"`, `"interrupted"`, `"done"`,
`"failed"`, `"stopped"`, `"abandoned"`), `saw` (the earlier turns it was
shown), `model`, `error`, `waiting` (the [`Approval`](@ref)s it waits for),
`unfinished` (tools that may have run when it stopped), `usage` (tokens over
every call inside it), `request_id`, `reads` and `made_by` (a merge). Another
output is a property too: `turn.reasoning`.
"""
struct Turn
    conv::Conversation
    st::TurnState
end
const TURN_PROPERTIES = (:id, :call, :conversation, :parent, :request_id, :inputs, :outputs, :result, :state, :model, :error,
                         :usage, :reads, :made_by, :saw, :waiting, :unfinished)
Base.propertynames(t::Turn) = (TURN_PROPERTIES..., Symbol.(keys(ended_outputs(getfield(t, :st))))...)
function Base.getproperty(t::Turn, name::Symbol)
    c, st = getfield(t, :conv), getfield(t, :st)
    ended = something(st.ended, JObj())
    name === :id || name === :call ? turn_id(st) :
    name === :conversation ? c.id :
    name === :parent ? turn_parent(st) :
    name === :request_id ? get(st.record, "request_id", nothing) :
    name === :inputs ? LMCC.deepcopy_json(something(get(st.record, "inputs", nothing), JObj())) :
    name === :outputs ? LMCC.deepcopy_json(ended_outputs(st)) :
    name === :result ? typed_answer(c, haskey(ended, "value") ? ended["value"] : get(ended_outputs(st), answer_of(c), nothing)) :
    name === :state ? turn_state(st) :
    name === :model ? something(get(ended, "model", nothing), get(something(get(st.record, "settings", nothing), JObj()), "lm", nothing), Some(nothing)) :
    name === :error ? LMCC.deepcopy_json(get(ended, "error", nothing)) :
    name === :usage ? JObj(something(get(ended, "usage", nothing), JObj())) :
    name === :reads ? String[x for x in something(get(st.record, "reads", nothing), Any[])] :
    name === :made_by ? LMCC.deepcopy_json(get(st.record, "made_by", nothing)) :
    name === :saw ? turn_saw(t) :
    name === :waiting ? Approval[approval_from(a) for a in unanswered(st)] :
    name === :unfinished ? Any[(invocation=x["invocation"], id=x["id"], name=x["name"], input=get(x, "input", nothing),
                                started=get(x, "at", nothing), site=get(x, "site", nothing)) for x in unfinished(st)] :
    haskey(ended_outputs(st), String(name)) ? ended_outputs(st)[String(name)] :
    throw(ArgumentError("a turn has no $name (its outputs: $(join(keys(ended_outputs(st)), ", ")))"))
end
function Base.show(io::IO, t::Turn)
    st = getfield(t, :st)
    ins = join(("$k=$(short_text(v))" for (k, v) in something(get(st.record, "inputs", nothing), JObj())), ", ")
    state = turn_state(st)
    print(io, "Turn(", first(turn_id(st), 8), " ", ins, state == "done" ? " → $(short_text(t.result))" : " [$state])", ")")
end

answer_of(c::Conversation) = last(interface(c.program)["outputs"])["name"]
function typed_answer(c::Conversation, v)
    v === nothing && return nothing
    c.program isa AIFunction || return v
    try
        spec = last(c.program.definition.outputs).spec
        spec isa AbstractDict ? convert_loaded(julia_type(spec), v) : fromjson(spec, v)
    catch
        v
    end
end

function turn_saw(t::Turn)
    c, st = getfield(t, :conv), getfield(t, :st)
    log = read_log!(c)
    entries = try
        saw(saw_records(log), turn_id(st))
    catch
        Any[e for e in something(st.ended === nothing ? nothing : get(st.ended, "saw", nothing), Any[]) if haskey(e, "call")]
    end
    Turn[Turn(c, log.turns[e["call"]]) for e in entries if haskey(log.turns, get(e, "call", ""))]
end

"""
    turns(chat; all = false) -> Vector{Turn}

The turns from the first to the head, in order; `all = true`: every turn of
every branch, in the order they were made.
"""
function turns(c::Conversation; all::Bool=false)
    log = read_log!(c)
    all && return Turn[Turn(c, log.turns[t]) for t in log.order]
    Turn[Turn(c, st) for st in branch(log, view_head(c, log))]
end

"The turn this conversation continues from next (`nothing`: none yet)."
function head(c::Conversation)
    log = read_log!(c)
    h = view_head(c, log)
    h === nothing || !haskey(log.turns, h) ? nothing : Turn(c, log.turns[h])
end

"""
    head!(chat, turn)

Make a turn the conversation's head, for everyone who opens it.
"""
function head!(c::Conversation, turn)
    tid = turn_id_of(c, turn)
    add_records!(c, [LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "head", "at" => iso(time()), "turn" => tid)])
    c.head, c.exact = tid, true
    c
end

"One turn, by its id, its index in `turns(chat)`, or a Turn."
turn(c::Conversation, t) = Turn(c, read_log!(c).turns[turn_id_of(c, t)])

function turn_id_of(c::Conversation, t)
    t isa Turn && return t.id
    t isa Integer && return turns(c)[t].id
    t isa AbstractString && haskey(read_log!(c).turns, t) && return String(t)
    throw(ConversationError("turn-unknown", "conversation $(c.id) has no turn $(repr(t))"))
end

"""
    continue_from(chat, turn) -> Conversation

This conversation, continuing after `turn` (a Turn, its id, or its index in
`turns(chat)`): the next turn is a new branch. Nothing is deleted.
"""
function continue_from(c::Conversation, t)
    tid = turn_id_of(c, t)
    Conversation(c.program, c.id, c.store, c.context, c.earlier_without, c.sends, c.settings, c.remembers, tid, true, c.delegated,
                 ConvLog(), ReentrantLock(), ReentrantLock())
end

# ----- calling it

"(inputs by name, settings for this turn) from a call's arguments: a keyword that is an input is an input; one that is a setting, a setting."
function split_turn_args(c::Conversation, args, kw)
    names = [x["name"] for x in interface(c.program)["inputs"]]
    settings = Dict{Symbol,Any}()
    inputs_kw = Pair{Symbol,Any}[]
    for (k, v) in kw
        String(k) in names || !(k in SETTING_NAMES) ? push!(inputs_kw, k => v) : (settings[k] = v)
    end
    given = c.program isa AIFunction ? bind_inputs(c.program, args, inputs_kw) : c.program.binder(args...; inputs_kw...)
    (given, settings_dict(settings))
end

(c::Conversation)(args...; request_id=nothing, kw...) = fetch(send_turn(c, args, kw, request_id; passive=true))

"""
    predict(chat, inputs...) -> Prediction

One turn of an AI function's conversation: the whole call (`p.call` is the turn's id).
"""
function StatsAPI.predict(c::Conversation, args...; request_id=nothing, kw...)
    s = send_turn(c, args, kw, request_id; passive=true)
    c.program isa AIFunction || return fetch(s)
    s isa StoredTurn && return fetch_prediction(s)
    prediction(s)
end

"""
    stream(chat, inputs...) -> AIStream

One turn, watched while it is made; `s.turn` is known at once (the turn is
saved before the model is asked).
"""
LM15.stream(c::Conversation, args...; request_id=nothing, kw...) = send_turn(c, args, kw, request_id)

function check_nested(c::Conversation)
    call = CURRENT_CALL[]
    outer = call !== nothing ? call.turn_run : TURN_STARTING[]
    (outer === nothing || c.delegated || (outer.conv.id == c.id && outer.conv.store === c.store)) && return
    any(p -> (first(p) === c || first(p) === c.program) && last(p) === :own, outer.conv.remembers) && return
    throw(ConversationError("conversation-nested", "conversation $(c.id) ($(program_name(c.program))) is used inside a turn of " *
        "conversation $(outer.conv.id), which does not say so: a remembering program inside another is refused unless declared " *
        "(remembers = Dict($(program_name(c.program)) => :own))"))
end

"The turn's inputs as they are bound (the record holds them); a refused input is refused before anything is recorded."
function bound_inputs(c::Conversation, given)
    fn = c.program
    if fn isa AIFunction
        s = effective(fn.own)
        inputs, refusal, _ = bind_ai_inputs(fn, given, s)
        refusal === nothing || throw(refusal)
        return (inputs, JObj(k => logvalue(v) for (k, v) in inputs))
    end
    refusal = binding_refusal(fn, given)
    refusal === nothing || throw(refusal)
    checked = check_inputs(fn.interface, given.given)
    (given, JObj(k => logvalue(v) for (k, v) in checked))
end

"The turn_start hooks: the turn's inputs as they leave them, and the changes."
function turn_start_hooks(c::Conversation, inputs, settings)
    plugins = plugins_around(c.program isa AIFunction ? c.program.own : Dict{Symbol,Any}(), Any[(:block, settings)])
    has_hook(plugins, :turn_start) || return (inputs, Any[])
    log = read_log!(c)
    given = inputs isa AbstractDict ? inputs : inputs.given
    event = TurnStartEvent(OrderedDict{String,Any}(String(k) => v for (k, v) in given), c.id, view_head(c, log), c.program, nothing, nothing)
    applied = Applied()
    names = Set(x["name"] for x in interface(c.program)["inputs"])
    run_hook(:turn_start, plugins, event, applied, function (ch)
        ch.inputs isa Union{AbstractDict,NamedTuple} || throw(ArgumentError("inputs is a Dict of the turn's inputs"))
        for (k, v) in pairs(ch.inputs)
            String(k) in names || throw(ArgumentError("$(program_name(c.program)) has no input $(repr(String(k)))"))
            event.inputs[String(k)] = v
        end
    end)
    changed = c.program isa AIFunction ? with_defaults(c.program, event.inputs) :
              (given=event.inputs, extra=inputs.extra, twice=inputs.twice, positional=inputs.positional)
    (changed, applied.items)
end

"The turn a new turn continues from, or `:busy` while it must wait."
function parent_for(c::Conversation, log::ConvLog)
    h = view_head(c, log)
    (h === nothing || !haskey(log.turns, h)) && return nothing
    st = log.turns[h]
    state = turn_state(st)
    state == "waiting" && c.sends !== :branch &&
        throw(ConversationError("conversation-busy", "conversation $(c.id): turn $h waits for a person's answer (answer it, or " *
                                                     "continue from another turn)"; turn=h))
    if state in ("running", "waiting")
        c.sends === :queue && return :busy
        c.sends === :refuse && throw(ConversationError("conversation-busy", "conversation $(c.id): turn $h is $state"; turn=h))
        return done_on(log, turn_parent(st))
    end
    done_on(log, h)
end

lease_record(tid, attempt) = LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "lease", "at" => iso(time()), "turn" => tid,
                                       "holder" => turn_holder(), "until" => iso(time() + LEASE_SECONDS), "attempt" => attempt)
const TURN_HOLDER = Ref("")
turn_holder() = (occursin(":$(getpid()):", TURN_HOLDER[]) || (TURN_HOLDER[] = "$(gethostname()):$(getpid()):$(bytes2hex(rand(UInt8, 3)))");
                 TURN_HOLDER[])

const RUNNING_TURNS = Dict{String,Any}()            # turns this process runs: id => stream
const RUNNING_LOCK = ReentrantLock()

function send_turn(c::Conversation, args, kw, request_id; passive::Bool=false)
    given, own = split_turn_args(c, args, kw)
    send_bound_turn(c, given, own, request_id; passive)
end

"One turn of inputs given by name (an AI function's with its defaults; a program's bound), with settings for this turn."
function send_bound_turn(c::Conversation, given, own::AbstractDict, request_id; passive::Bool=false)
    check_nested(c)
    check_content(c)                          # a host's rule set since the conversation was opened holds too
    turn_settings = merge(c.settings, own)
    given, start_changes = turn_start_hooks(c, given, turn_settings)
    inputs, values = bound_inputs(c, given)   # refused before anything is recorded
    local tid, context, start_seq
    while true
        busy = lock(c.send_lock) do
            log = read_log!(c)
            if request_id !== nothing && haskey(log.request_ids, string(request_id))
                return existing_turn(c, log.request_ids[string(request_id)])
            end
            parent = parent_for(c, log)
            parent === :busy && return :busy
            check_conversation_signature(c.program, log, branch(log, parent), c.earlier_without)
            tid = new_id()
            context = turn_context(c, log, parent, turn_settings)
            context[:changes] = Any[start_changes..., context[:changes]...]
            desc = program_record(c.program)
            recs = JObj[]
            haskey(log.programs, desc["version"]) || push!(recs, desc)
            rec = LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "turn", "at" => iso(time()), "turn" => tid,
                            "parent" => parent, "program" => desc["version"], "inputs" => values)
            request_id === nothing || (rec["request_id"] = string(request_id))
            get(turn_settings, :lm, nothing) isa AbstractString && (rec["settings"] = LMCC.jobj("lm" => turn_settings[:lm]))
            context[:recorded] === nothing || (rec["context"] = context[:recorded])
            isempty(context[:changes]) || (rec["changes"] = LMCC.deepcopy_json(context[:changes]))
            push!(recs, rec)
            push!(recs, lease_record(tid, 1))
            try
                start_seq = add_records!(c, recs; expect=log.n)
            catch err
                (err isa ConversationError && err.code == "store-conflict") || rethrow()
                return :retry                         # someone appended meanwhile: read again
            end
            c.head, c.exact = tid, false              # from now on, this view follows its own branch
            nothing
        end
        busy === nothing && break
        busy isa Union{AIStream,StoredTurn} && return busy
        busy === :busy && wait_records(c.store, c.id, c.log.n, 0.25)    # a turn before it is running: queue behind it
    end
    run = TurnRun(c, tid; attempt=1, context)
    run.settings = turn_settings
    run.start_seq = start_seq
    start_turn(c, run, inputs, turn_settings; passive)
end

"""
What the turn after `parent` is shown (contract/conversations.md, "What a
turn is shown"): the earlier turns the conversation's rule picks, then the
`context` hooks' changes; each as the lmcc turn it was (with its steps) or
its values, the fields left out taken out; the `saw` entries; the rows
`earlier()` gives; the sections; the changes made and the context to
record. `fixed`: the context a turn recorded when made (resuming it shows
exactly that, with no hook run again).
"""
function turn_context(c::Conversation, log::ConvLog, parent, settings=c.settings, fixed=nothing)
    picked, without, sections, changes, recorded = shown_turns_of(c, log, parent, settings, fixed)
    ts, ids, rows = Any[], String[], Any[]
    dropped = Dict{String,Vector{String}}()
    is_ai = c.program isa AIFunction
    sig = is_ai ? signature_id(c.program) : nothing
    for st in picked
        outs = ended_outputs(st)
        row = merge(JObj(something(get(st.record, "inputs", nothing), JObj())), outs)
        gone_names = get(without, turn_id(st), String[])
        push!(rows, JObj(k => v for (k, v) in row if !(k in gone_names)))
        if !is_ai
            push!(ids, turn_id(st))
            continue
        end
        desc = get(log.programs, get(st.record, "program", ""), JObj())
        t = if st.ended !== nothing && haskey(st.ended, "lmcc") && get(desc, "signature", nothing) == sig
            fitted(c.program, LMCC.deepcopy_json(st.ended["lmcc"]))
        else
            LMCC.jobj("inputs" => JObj(something(get(st.record, "inputs", nothing), JObj())), "outputs" => outs)
        end
        t, gone = without_fields(t, gone_names)
        isempty(gone) || (dropped[turn_id(st)] = gone)
        push!(ts, t)
        push!(ids, turn_id(st))
    end
    records = saw_records(log)
    finish = function (entries)
        out = Any[]
        for e in entries
            e = JObj(e)
            gone = get(dropped, get(e, "call", ""), nothing)
            if gone !== nothing
                e["without"] = Any[sort!(unique(vcat(String[x for x in get(e, "without", Any[])], gone)))...]
                delete!(e, "steps")
            end
            push!(out, e)
        end
        compress_saw(out, parent, records)
    end
    base = Dict{Symbol,Any}(:rows => rows, :parent => parent, :sections => sections, :changes => changes, :recorded => recorded)
    if !is_ai
        entries = finish(Any[LMCC.jobj("call" => i) for i in ids])
        return merge(base, Dict{Symbol,Any}(:turns => Any[], :ids => String[], :finish => (_ -> entries), :module_saw => entries))
    end
    merge(base, Dict{Symbol,Any}(:turns => ts, :ids => ids, :finish => finish))
end

"A turn with some fields left out (and so shown without its steps), and which of its fields were."
function without_fields(t::AbstractDict, names)
    had = union(keys(something(get(t, "inputs", nothing), JObj())), keys(something(get(t, "outputs", nothing), JObj())))
    gone = sort!(collect(intersect(had, names)))
    isempty(gone) && return (t, String[])
    (LMCC.jobj("inputs" => JObj(k => v for (k, v) in something(get(t, "inputs", nothing), JObj()) if !(k in gone)),
               "outputs" => JObj(k => v for (k, v) in something(get(t, "outputs", nothing), JObj()) if !(k in gone))), gone)
end

"`[{\"saw_of\": parent}, <parent's entry>]` when the entries are exactly what the parent saw, then the parent (contract/calls.md, \"Saw\")."
function compress_saw(entries, parent, records)
    (length(entries) < 2 || parent === nothing || get(entries[end], "call", nothing) != parent) && return entries
    any(r -> r["id"] == parent, records) || return entries
    before = try
        saw(records, parent)
    catch
        return entries
    end
    same_json(Any[before...], Any[entries[1:end-1]...]) ? Any[LMCC.jobj("saw_of" => parent), entries[end]] : entries
end

"""
`(the earlier turns shown, fields left out of each, sections, the changes
made, the context to record)`: the conversation's rule picks among the done
turns of the branch, then the `context` hooks change it.
"""
function shown_turns_of(c::Conversation, log::ConvLog, parent, settings, fixed)
    done = [st for st in branch(log, parent) if turn_state(st) == "done"]
    if fixed !== nothing
        by_id = Dict(turn_id(st) => st for st in done)
        picked = [by_id[t] for t in something(get(fixed, "turns", nothing), Any[]) if haskey(by_id, t)]
        without = Dict{String,Vector{String}}(String(k) => String[x for x in v] for (k, v) in something(get(fixed, "without", nothing), JObj()))
        return (picked, without, String[x for x in something(get(fixed, "sections", nothing), Any[])], Any[], nothing)
    end
    picked = pick(c.context, done)
    without = Dict{String,Vector{String}}()
    for st in picked
        fields = union(keys(something(get(st.record, "inputs", nothing), JObj())), keys(ended_outputs(st)))
        gone = sort!(collect(intersect(fields, c.context.without)))
        isempty(gone) || (without[turn_id(st)] = gone)
    end
    plugins = plugins_around(c.program isa AIFunction ? c.program.own : Dict{Symbol,Any}(), Any[(:block, settings)])
    has_hook(plugins, :context) || return (picked, without, String[], Any[], nothing)
    event = ContextEvent(ShownTurn[ShownTurn(turn_id(st), JObj(something(get(st.record, "inputs", nothing), JObj())), ended_outputs(st),
                                             get(without, turn_id(st), String[])) for st in picked], String[], c, parent, c.program, nothing, nothing)
    applied = Applied()
    branch_done = Dict(turn_id(st) => st for st in done)
    current = Ref(picked)
    run_hook(:context, plugins, event, applied, function (ch)
        if ch.keep !== nothing
            ids = String[String(x) for x in ch.keep]
            unknown = [i for i in ids if !haskey(branch_done, i)]
            isempty(unknown) || throw(ArgumentError("keep names $(first(unknown)), which is not a done turn of this branch"))
            current[] = [st for st in done if turn_id(st) in ids]
            event.turns = ShownTurn[[t for t in event.turns if t.id in ids];
                                    [ShownTurn(i, JObj(something(get(branch_done[i].record, "inputs", nothing), JObj())), ended_outputs(branch_done[i]))
                                     for i in ids if !any(t -> t.id == i, event.turns)]]
        end
        if ch.without !== nothing
            targets = ch.without isa AbstractDict ? ch.without : Dict(turn_id(st) => ch.without for st in current[])
            for (tid, names) in targets
                (names isa AbstractString || !all(n -> n isa Union{AbstractString,Symbol}, names)) &&
                    throw(ArgumentError("without names fields: a list of names, or Dict(turn => names)"))
                without[String(tid)] = sort!(unique(vcat(get(without, String(tid), String[]), String[String(n) for n in names])))
            end
        end
        ch.sections === nothing || append!(event.sections, texts_of(ch.sections))
    end)
    picked = current[]
    ids = Set(turn_id(st) for st in picked)
    without = Dict(k => v for (k, v) in without if k in ids)
    recorded = LMCC.jobj("turns" => Any[turn_id(st) for st in picked], "without" => JObj(k => Any[v...] for (k, v) in without),
                         "sections" => Any[event.sections...])
    (picked, without, String[event.sections...], applied.items, recorded)
end

# ----- plugin entries

"""
    remember!(chat, plugin, kind, data; turn = nothing)

Keep a plugin's entry in this conversation, at a turn (it then belongs to
the branches through that turn), or with no turn (every branch): what the
plugin needs later, never shown to the model by itself. A host keeps one too
(switching a mode is an entry).
"""
function remember!(c::Conversation, plugin::AbstractString, kind::AbstractString, data; turn=nothing)
    isempty(kind) && throw(ArgumentError("an entry's kind is a name"))
    rec = LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "entry", "at" => iso(time()), "plugin" => String(plugin),
                    "entry" => String(kind), "data" => logvalue(data))
    turn === nothing || (rec["turn"] = String(turn))
    add_records!(c, [rec])
    nothing
end

"""
    entries(chat, plugin, kind; branch = head) -> Vector

A plugin's entries of a kind on the branch through `branch` (default: this
view's head), oldest first: `(turn, data, at)` each.
"""
function entries(c::Conversation, plugin::AbstractString, kind::AbstractString; branch=nothing)
    log = read_log!(c)
    through = branch === nothing ? view_head(c, log) : branch
    on = Set(turn_id(st) for st in FunctAI.branch(log, through))
    Any[(turn=get(r, "turn", nothing), data=LMCC.deepcopy_json(get(r, "data", nothing)), at=get(r, "at", nothing))
        for r in log.entries if get(r, "plugin", nothing) == plugin && get(r, "entry", nothing) == kind &&
            (get(r, "turn", nothing) === nothing || r["turn"] in on)]
end

# ----- running a turn

"A turn another process runs (or ran), watched through its store: its result, its events."
struct StoredTurn
    conv::Conversation
    turn::String
end
Base.show(io::IO, s::StoredTurn) = print(io, "StoredTurn(", s.turn, ")")
Base.fetch(s::StoredTurn) = turn_outcome(s.conv, wait_turn(turn(s.conv, s.turn)))
function fetch_prediction(s::StoredTurn)
    t = wait_turn(turn(s.conv, s.turn))
    turn_outcome(s.conv, t)
    t
end
Base.close(::StoredTurn) = nothing

function turn_outcome(c::Conversation, t::Turn)
    state = t.state
    state == "done" && return t.result
    state == "waiting" && throw(Waiting("turn $(t.id) waits for a person's answer"; turn=t, approvals=t.waiting))
    err = something(t.error, JObj())
    throw(ConversationError("turn-state", "turn $(t.id) ended $state" * (isempty(err) ? "" : ": $(get(err, "type", "")): $(get(err, "message", ""))");
                            turn=t.id))
end

function existing_turn(c::Conversation, tid)
    hit = lock(() -> get(RUNNING_TURNS, tid, nothing), RUNNING_LOCK)
    hit === nothing ? StoredTurn(c, tid) : hit
end

const DEBUG_TURNS = Ref(false)
const SINK_ERRORS = Any[]
const LAST_TURN_ERROR = Ref("")
"Run a turn: its program's call, in a stream (a task), with the turn's settings; then its last record."
function start_turn(c::Conversation, run::TurnRun, inputs, settings; passive::Bool=false)
    fn = c.program
    started = time()
    s = start_stream(; passive, turn=(c, run.turn)) do
        value, err = nothing, nothing
        try
            value = with(TURN_STARTING => run) do
                isempty(settings) ? call_turn(fn, inputs) : with_settings(() -> call_turn(fn, inputs); settings...)
            end
        catch e
            DEBUG_TURNS[] && (LAST_TURN_ERROR[] = sprint(showerror, e, catch_backtrace()))
            err = unwrap(e)
        end
        try
            err = conclude_turn(c, run, value, err, started)
        finally
            lock(() -> delete!(RUNNING_TURNS, run.turn), RUNNING_LOCK)
        end
        err === nothing || throw(err)
        value
    end
    lock(() -> (RUNNING_TURNS[run.turn] = s), RUNNING_LOCK)
    errormonitor(@async heartbeat(c, run, s))
    s
end
call_turn(f::AIFunction, inputs) = predict_inputs(f, inputs)
call_turn(p::AIProgram, bound) = run_program(p, bound)

"""
Renew the turn's lease, look for a stop from another process, and stop when
another process took the turn over (its lease ran out here).
"""
function heartbeat(c::Conversation, run::TurnRun, s::AIStream)
    renewed = time()
    seen = run.start_seq
    while !s.finished
        sleep(STOP_POLL)
        s.finished && return
        try
            recs = read_records(c.store, c.id, seen)
            seen += length(recs)
            mine = [r for r in recs if get(r, "turn", nothing) == run.turn]
            any(r -> get(r, "kind", nothing) == "stop", mine) && close(s)
            if any(r -> get(r, "kind", nothing) == "lease" && Int(something(get(r, "attempt", nothing), 1)) > run.attempt, mine)
                warn_once("lease-lost:$(run.turn)", "turn $(run.turn) was taken over by another process (its lease ran out here): this one stops")
                close(s)
            end
            if time() - renewed >= RENEW_SECONDS && !s.finished
                renewed = time()
                add_records!(c, [lease_record(run.turn, run.attempt)])
            end
        catch err
            EXITING[] && return
            warn_once("heartbeat:$(typeof(err))", "a turn's lease could not be renewed ($(sprint(showerror, err)))")
        end
    end
end

"Wait until the store's copy of the turn's log caught up (its events are kept on a task of their own)."
function drain_sink(run::TurnRun; timeout=5.0)
    run.sink === nothing && return
    deadline = time() + timeout
    while time() < deadline
        busy = lock(FEEDS_LOCK) do
            f = get(FEEDS, run.sink, nothing)
            f !== nothing && (f.running || !isempty(f.waiting))
        end
        busy || return
        sleep(0.005)
    end
end

"Write the turn's last record (`ended`, or `waiting`); returns the error the caller gets."
function conclude_turn(c::Conversation, run::TurnRun, value, err, started)
    drain_sink(run)
    root = run.root
    saw = root !== nothing ? LMCC.deepcopy_json(root.saw) : Any[get(run.context, :module_saw, Any[])...]
    if err isa TurnWaiting
        a = err.approval
        rec = record_base(run, "waiting")
        rec["approvals"] = Any[approval_json(a)]
        rec["saw"] = saw
        add_records!(c, [rec])
        t = turn(c, run.turn)
        return Waiting("turn $(run.turn) waits for a person's answer: $(a.path)($(short_text(logvalue(a.input), 80)))"; turn=t, approvals=t.waiting)
    end
    rec = record_base(run, "ended")
    rec["saw"] = saw
    rec["seconds"] = round6(time() - started)
    rec["usage"] = copy(run.usage)
    if err !== nothing
        rec["state"] = err isa Cancelled ? "stopped" : "failed"
        rec["error"] = error_json(err, true)
    else
        try
            outputs, lmcc_turn, model = turn_outputs(c, run, value)
            rec["state"] = "done"
            rec["outputs"] = outputs
            rec["value"] = logvalue(value isa Prediction ? value.value : value)
            lmcc_turn === nothing || (rec["lmcc"] = lmcc_turn)
            rec["model"] = model
        catch e
            rec["state"] = "failed"
            rec["error"] = error_json(e, true)
            err = e
        end
    end
    turn_end_hooks(c, run, rec)                  # before the end is recorded: the next turn sees what they keep
    try
        add_records!(c, [rec])
    catch e
        warn_once("ended:$(typeof(e))", "a turn's end could not be kept in its store ($(sprint(showerror, e)))")
    end
    err
end

function turn_outputs(c::Conversation, run::TurnRun, value)
    root = run.root
    model = root === nothing ? nothing : (i = findlast(e -> e.response !== nothing, root.exchanges); i === nothing ? nothing : root.exchanges[i].model)
    if c.program isa AIFunction
        value isa Prediction || throw(ArgumentError("the turn's call left no prediction"))
        root === nothing && throw(ArgumentError("the turn's call left no record"))
        outputs = JObj(k => logvalue(v) for (k, v) in something(root.outputs, JObj()))
        return (outputs, root.lmcc, model)
    end
    outputs = root === nothing || root.outputs === nothing ? JObj(answer_of(c) => logvalue(value)) :
              JObj(k => logvalue(v) for (k, v) in root.outputs)
    (outputs, nothing, model)
end

"The turn_end hooks (they hear; they may keep entries at the turn). One that fails is reported, and changes nothing of the turn."
function turn_end_hooks(c::Conversation, run::TurnRun, rec)
    plugins = plugins_around(c.program isa AIFunction ? c.program.own : Dict{Symbol,Any}(), Any[(:block, merge(c.settings, run.settings))])
    has_hook(plugins, :turn_end) || return
    st = read_log!(c).turns[run.turn]
    event = TurnEndEvent(run.turn, rec["state"], JObj(something(get(st.record, "inputs", nothing), JObj())),
                         JObj(something(get(rec, "outputs", nothing), JObj())), c, turn_parent(st), nothing, nothing)
    for p in plugins, f in get(p.handlers, :turn_end, ())
        event.plugin = p
        try
            Base.invokelatest(f, event)
        catch err
            err isa InterruptException && rethrow()
            warn_once("turn_end:$(p.name):$(typeof(unwrap(err)))", "plugin $(p.name) failed in turn_end ($(sprint(showerror, unwrap(err))))")
        finally
            event.plugin = nothing
        end
    end
end

# ----- what can be done with a turn

"""
    wait_turn(turn; timeout = nothing) -> Turn

Wait until the turn is no longer running; returns it as it is then.
"""
function wait_turn(t::Turn; timeout=nothing)
    c = getfield(t, :conv)
    tid = t.id
    deadline = timeout === nothing ? nothing : time() + timeout
    while true
        st = read_log!(c).turns[tid]
        turn_state(st) == "running" || return Turn(c, st)
        deadline !== nothing && time() >= deadline && throw(ErrorException("turn $tid is still running"))
        wait_records(c.store, c.id, c.log.n, deadline === nothing ? 0.5 : max(0.0, min(0.5, deadline - time())))
    end
end

"""
    stop!(chat, turn)
    stop!(turn)

Stop a running turn, wherever it runs: in this process or another one that
opened the same store. It ends `stopped` within about a second (its stream
throws `Cancelled`). A turn that waits for an approval, or was interrupted,
has nothing running it: stopping it ends it `abandoned`. A turn that already
ended is left as it is.
"""
function stop!(c::Conversation, t)
    tid = turn_id_of(c, t)
    state = turn_state(read_log!(c).turns[tid])
    state in ("waiting", "interrupted") && return abandon!(turn(c, tid))
    state == "running" || return nothing
    add_records!(c, [LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "stop", "at" => iso(time()), "turn" => tid)])
    hit = lock(() -> get(RUNNING_TURNS, tid, nothing), RUNNING_LOCK)
    hit === nothing || close(hit)
    nothing
end
stop!(t::Turn) = stop!(getfield(t, :conv), t)

"""
    approve!(turn, approval = nothing; by = nothing, resume = true)
    approve!(stream, approval = nothing; by = nothing)

Say yes to an approval a turn waits for (`turn.waiting[1]`, its invocation
number, or `nothing` for the only one). When nothing else waits, the turn
goes on here (`resume = false`: later, [`resume!`](@ref)) and its answer is
returned. On a stream without a conversation, the call waiting in this
process goes on.
"""
approve!(t::Turn, approval=nothing; by=nothing, resume::Bool=true) = answer_turn!(t, approval, true, nothing, by, resume)

"""
    deny!(turn, approval = nothing; reason = nothing, by = nothing, resume = true)
    deny!(stream, approval = nothing; reason = nothing, by = nothing)

Say no: the model is told the person did not allow the call (and why), and may try something else.
"""
deny!(t::Turn, approval=nothing; reason=nothing, by=nothing, resume::Bool=true) = answer_turn!(t, approval, false, reason, by, resume)

approve!(s::AIStream, approval=nothing; by=nothing) = answer_stream!(s, approval, true, nothing, by)
deny!(s::AIStream, approval=nothing; reason=nothing, by=nothing) = answer_stream!(s, approval, false, reason, by)

"Answer an approval a call watched by this stream waits for, in this process."
function answer_stream!(s::AIStream, approval, allowed::Bool, reason, by)
    ct = getfield(s, :conv_turn)
    if ct !== nothing                           # a conversation's turn waits saved: answer the turn
        t = turn(ct[1], ct[2])
        t.state == "waiting" && return answer_turn!(t, approval, allowed, reason, by, true)
    end
    mine = lock(() -> copy(getfield(s, :calls)), getfield(s, :cond))
    pending = [a for a in waiting_here() if a.call in mine]
    target = if approval === nothing
        length(pending) == 1 || throw(ArgumentError("$(length(pending)) tool calls wait here: name one"))
        pending[1]
    else
        inv = approval isa Approval ? approval.invocation : Int(approval)
        call = approval isa Approval ? approval.call : nothing
        i = findfirst(a -> a.invocation == inv && (call === nothing || a.call == call), pending)
        i === nothing && throw(ArgumentError("no tool call waits here for $(repr(approval))"))
        pending[i]
    end
    answer_here(target.call, target.invocation, allowed; reason, by, plugin=target.plugin)
end

function answer_turn!(t::Turn, approval, allowed::Bool, reason, by, resume::Bool)
    c = getfield(t, :conv)
    st = read_log!(c).turns[t.id]
    waiting = unanswered(st)
    isempty(waiting) && throw(ConversationError("turn-state", "turn $(t.id) waits for no approval (it is $(turn_state(st)))"; turn=t.id))
    target = if approval === nothing
        length(waiting) > 1 && throw(ArgumentError("turn $(t.id) waits for $(length(waiting)) approvals: name one"))
        waiting[1]
    else
        inv = approval isa Approval ? approval.invocation : approval isa AbstractDict ? approval["invocation"] : Int(approval)
        site = approval isa Approval ? approval.site : approval isa AbstractDict ? get(approval, "site", nothing) : nothing
        plugin = approval isa Approval ? approval.plugin : approval isa AbstractDict ? get(approval, "plugin", nothing) : nothing
        i = findfirst(a -> Int(a["invocation"]) == inv && (site === nothing || isempty(site) || get(a, "site", nothing) == site) &&
                           (plugin === nothing || something(get(a, "plugin", nothing), "approval") == plugin), waiting)
        i === nothing && throw(ConversationError("turn-state", "turn $(t.id) waits for no approval $(repr(approval))"; turn=t.id))
        waiting[i]
    end
    add_records!(c, [LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "approval", "at" => iso(time()), "turn" => t.id,
                          "site" => get(target, "site", nothing), "invocation" => target["invocation"], "path" => get(target, "path", nothing),
                          "plugin" => something(get(target, "plugin", nothing), "approval"), "verdict" => allowed ? "yes" : "no",
                          "by" => by, "reason" => reason)])
    st = read_log!(c).turns[t.id]
    resume && isempty(unanswered(st)) && return resume!(Turn(c, st))
    nothing
end

"""
    resume!(turn; results = Dict(), rerun = []) -> the turn's answer

Go on with a turn that waits (every approval answered) or was interrupted
(its process stopped), in this process: its program runs again with the
same inputs and earlier turns, each model reply it had and each tool result
it kept are reused (nothing is paid for or run twice), and it goes on from
where it stopped. A tool that started and has no result may have run:
`results = Dict(invocation => output)` says what it returned, `rerun =
[invocation]` runs it again. Throws [`Waiting`](@ref) when it waits again.
"""
resume!(t::Turn; results=Dict(), rerun=()) = fetch(resume_turn(getfield(t, :conv), t.id, results, rerun))

function resume_turn(c::Conversation, tid::AbstractString, results, rerun)
    local start_seq, attempt
    while true
        again = lock(c.send_lock) do
            log = read_log!(c)
            st = get(log.turns, tid, nothing)
            st === nothing && throw(ConversationError("turn-unknown", "conversation $(c.id) has no turn $tid"))
            state = turn_state(st)
            state in ("waiting", "interrupted") ||
                throw(ConversationError("turn-state", "turn $tid is $state: only a waiting or interrupted turn goes on"; turn=tid))
            isempty(unanswered(st)) || throw(ConversationError("turn-state", "turn $tid still waits for $(length(unanswered(st))) approval(s)"; turn=tid))
            given = JObj[]
            for x in unfinished(st)
                inv = Int(x["invocation"])
                keys_of = (inv, string(inv), get(x, "id", nothing))
                hit = findfirst(k -> k in keys_of, collect(keys(results)))
                base = LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "tool", "at" => iso(time()), "turn" => tid,
                                 "site" => get(x, "site", nothing), "invocation" => inv, "id" => get(x, "id", nothing), "name" => get(x, "name", nothing))
                if hit !== nothing
                    push!(given, merge(base, LMCC.jobj("state" => "given", "output" => string(results[collect(keys(results))[hit]]))))
                elseif any(r -> r in keys_of, rerun)
                    push!(given, merge(base, LMCC.jobj("state" => "rerun")))
                else
                    throw(ConversationError("turn-unfinished", "turn $tid: $(x["name"]) (invocation $inv) started and may have run: " *
                                            "resume!(turn; results = Dict($inv => what it returned)) or resume!(turn; rerun = [$inv])"; turn=tid))
                end
            end
            attempt = st.attempt + 1
            try
                start_seq = add_records!(c, [given..., lease_record(tid, attempt)]; expect=log.n)
            catch err
                (err isa ConversationError && err.code == "store-conflict") || rethrow()
                return true
            end
            false
        end
        again || break
    end
    log = read_log!(c)
    st = log.turns[tid]
    later = nothing
    events = event_store(c.store)
    if events !== nothing
        try
            claim = claim!(events, tid)
            kept = events_after(events, tid, nothing)
            requests = maximum((Int(datum(e, "request", 0)) for e in kept if e.kind === :request && e.call == tid); init=0)
            later = (writer=claim.writer, after=claim.after, at=isempty(kept) ? "" : last(kept).at, requests=requests)
        catch err
            err isa InterruptException && rethrow()
            later = nothing                               # no log to continue: this writer starts none
        end
    end
    replay = Dict{Symbol,Any}(:replies => st.replies, :tools => st.tools, :answers => st.answers, :waiting_seq => seq_of(st.waiting))
    fixed = get(st.record, "context", nothing)            # what the turn was shown when it was made (context hooks ran)
    run = TurnRun(c, tid; attempt, context=turn_context(c, log, turn_parent(st), c.settings, fixed), replay, later)
    run.context[:changes] = Any[]                          # the turn's own changes are on its first record
    run.start_seq = start_seq
    settings = copy(c.settings)
    lm = get(something(get(st.record, "settings", nothing), JObj()), "lm", nothing)
    lm === nothing || (settings[:lm] = lm)
    run.settings = settings
    inputs = LMCC.deepcopy_json(something(get(st.record, "inputs", nothing), JObj()))
    given = c.program isa AIFunction ? with_defaults(c.program, OrderedDict{String,Any}(inputs)) :
            (given=OrderedDict{String,Any}(inputs), extra=0, twice=String[], positional=String[])
    called = c.program isa AIFunction ? bind_ai_inputs(c.program, given, effective(c.program.own))[1] : given
    start_turn(c, run, called, settings; passive=true)
end

"""
    abandon!(turn)

End a turn that waits or was interrupted, without going on: it ends `abandoned`.
"""
function abandon!(t::Turn)
    c = getfield(t, :conv)
    st = read_log!(c).turns[t.id]
    turn_state(st) in ("waiting", "interrupted") ||
        throw(ConversationError("turn-state", "only a waiting or interrupted turn can be abandoned; turn $(t.id) is $(turn_state(st))"; turn=t.id))
    add_records!(c, [LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "ended", "at" => iso(time()), "turn" => t.id,
                          "state" => "abandoned", "attempt" => st.attempt)])
    nothing
end

"""
    eachevent(turn; after = nothing, view = :kept, timeout = nothing)

The turn's events from its store (the kept form, or a view made from it:
`view = :outside`), after the event `after` names (a [`Position`](@ref)),
those kept so far and then each as it is kept, until its last: what another
process, a page after a reload, reads.
"""
function eachevent(t::Turn; after=nothing, view=:kept, timeout=nothing)
    c = getfield(t, :conv)
    events = event_store(c.store)
    events === nothing && return Event[]
    stop = () -> turn_state(read_log!(c).turns[t.id]) in ("waiting", "interrupted", "abandoned")
    out = Event[]
    if Symbol(view) === :kept
        follow_store(e -> push!(out, e), events, t.id, after; stop, timeout)
        return out
    end
    v = View(Symbol(view); answer_from=answer_from_of(c.program))
    waiting_for = after
    follow_store(events, t.id, nothing; stop, timeout) do e
        shown = apply_view!(v, e)
        shown === nothing && return
        if waiting_for !== nothing
            Position(shown) == waiting_for && (waiting_for = nothing)
            return
        end
        push!(out, shown)
    end
    waiting_for === nothing || throw(StoreRefusal("event-unknown", "this view of turn $(t.id) has no event $after"))
    out
end

"""
    FunctAI.calls_in(turn) -> Vector

The calls inside a turn, as its kept log says: `(call, parent, function,
invocation, ended)` each, in the order they started. `rate` takes their ids.
"""
function calls_in(t::Turn)
    c = getfield(t, :conv)
    events = event_store(c.store)
    events === nothing && return Any[]
    out = OrderedDict{String,Any}()
    for e in events_after(events, t.id, nothing)
        if e.kind === :started
            out[e.call] = Dict{Symbol,Any}(:call => e.call, :parent => datum(e, "parent"), :function => getfield(e, :fn),
                                           :invocation => datum(e, "invocation"), :ended => nothing)
        elseif e.kind in (:done, :failed) && haskey(out, e.call)
            out[e.call][:ended] = String(e.kind)
        end
    end
    Any[(; x...) for x in values(out)]
end

# ----- the rest

"""
    render(chat, inputs...; call = nothing) -> LM15.Request

The exact request the next turn would send (nothing is sent or recorded).
`call = helper`: the request that helper would get, with the memory the
conversation gives it (inputs: the helper's).
"""
function render(c::Conversation, args...; call=nothing, kw...)
    log = read_log!(c)
    parent = view_head(c, log)
    parent = parent === nothing ? nothing : done_on(log, parent)
    if call !== nothing
        memory = remembered(c, call)
        ts, ids = if memory isa Memory
            run = TurnRun(c, new_id(); attempt=1, context=Dict{Symbol,Any}(:parent => parent, :turns => Any[], :ids => String[]))
            helper_context(run, call, memory)
        else
            (Any[], String[])
        end
        return with(() -> render(call, args...; kw...), RENDERING => (program=call, turns=ts, ids=ids, sections=String[]))
    end
    c.program isa AIFunction || throw(ArgumentError("$(program_name(c.program)) is a program: render the request one of its helpers " *
                                                     "would get, render(chat, …; call = helper)"))
    ctx = turn_context(c, log, parent, c.settings)
    with(() -> render(c.program, args...; kw...), RENDERING => (program=c.program, turns=ctx[:turns], ids=ctx[:ids], sections=ctx[:sections]))
end

"""
    merge!(chat, branches, fn; inputs...) -> Turn

A turn after the conversation's head made from several branches by another
AI function (`fn`): its answer becomes this program's answer, recorded as a
turn (`made_by` fn, `reads` the branches), so the next turn sees it. Its
rating belongs to `fn`'s call. When `fn` has one input and none is given, it
is given the branches' answers (`[(model, inputs…, outputs…)]`).
"""
function Base.merge!(c::Conversation, branches::AbstractVector, fn::AIFunction; inputs...)
    c.program isa AIFunction || throw(ArgumentError("merge! makes an AI function's turn; a program's conversation cannot take one"))
    log = read_log!(c)
    read_turns = Turn[Turn(c, log.turns[turn_id_of(c, b)]) for b in branches]
    isempty(read_turns) && throw(ArgumentError("merge! needs the turns to merge"))
    parent = view_head(c, log)
    for t in read_turns
        t.parent == parent || throw(ConversationError("turn-state", "turn $(t.id) does not continue from $parent: a merge reads branches " *
                                                                    "of the turn it follows"; turn=t.id))
        t.state == "done" || throw(ConversationError("turn-state", "turn $(t.id) is $(t.state)"; turn=t.id))
    end
    given = Dict{Symbol,Any}(inputs)
    if isempty(given)
        names = input_names(fn)
        length(names) == 1 || throw(ArgumentError("$(fn.definition.name) takes $(length(names)) inputs: give them by name"))
        given[Symbol(only(names))] = Any[merge(JObj("model" => t.model), t.inputs, t.outputs) for t in read_turns]
    end
    p = predict_inputs(fn, with_defaults(fn, OrderedDict{String,Any}(String(k) => v for (k, v) in given)))
    answer = answer_of(c)
    value = p.answer
    tid = new_id()
    base = first(read_turns)
    plan = plan_for(c.program, effective(c.program.own)).plan
    lmcc_json = try
        ins = prepare_inputs(plan.signature, JObj(base.inputs))
        LMCC.turn_to_dict(LMCC.example(plan, ins, JObj(answer => logvalue(value))))
    catch err
        throw(ArgumentError("$(fn.definition.name)'s answer does not fit $(program_name(c.program))'s answer: $(sprint(showerror, err))"))
    end
    desc = program_record(c.program)
    recs = haskey(log.programs, desc["version"]) ? JObj[] : JObj[desc]
    push!(recs, LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "turn", "at" => iso(time()), "turn" => tid, "parent" => parent,
                          "program" => desc["version"], "inputs" => base.inputs, "reads" => Any[t.id for t in read_turns],
                          "made_by" => LMCC.jobj("name" => fn.definition.name, "version" => version(fn), "call" => p.call)))
    push!(recs, LMCC.jobj("functai_conversation" => CONVERSATION_FORMAT, "kind" => "ended", "at" => iso(time()), "turn" => tid, "state" => "done",
                          "outputs" => JObj(answer => logvalue(value)), "value" => logvalue(value), "lmcc" => lmcc_json, "saw" => Any[],
                          "model" => nothing, "attempt" => 1))
    add_records!(c, recs)
    c.head, c.exact = tid, false
    turn(c, tid)
end

# ------------------------------------------------------------------ rows asked again with their context (stage 5)

"""
A rated row asked again (`evaluate`, the optimizers) with what its call was
shown: the program's own call gets the row's `earlier` turns, each helper call
the earlier turns its original call was shown (in the order they were made),
and its record says `saw_of` the original. No conversation is read or written.
"""
struct RowReplay
    program::Any
    earlier::Vector{Any}
    helpers::Vector{Any}
    sections::Vector{String}
    call::Any
    lock::ReentrantLock
    helper_sections::Dict{String,Vector{String}}
end

row_meta(row, key) = (for k in Iterators.reverse(collect(keys(row)))
    lstrip(string(k), '_') == key && return row[k]
end; nothing)

function RowReplay(row::AbstractDict, program)
    RowReplay(program, Any[something(row_meta(row, "earlier"), Any[])...], Any[something(row_meta(row, "helpers"), Any[])...],
              String[something(row_meta(row, "sections"), Any[])...], row_meta(row, "call"), ReentrantLock(), Dict{String,Vector{String}}())
end

function replay_turn(f::AIFunction, t)
    sig = signature_id(f)
    if !isempty(something(get(t, "steps", nothing), Any[])) && get(t, "signature", nothing) == sig
        return fitted(f, LMCC.jobj("signature" => sig, "inputs" => JObj(something(get(t, "inputs", nothing), JObj())),
                                   "steps" => LMCC.deepcopy_json(t["steps"]), "outputs" => JObj(something(get(t, "outputs", nothing), JObj()))))
    end
    LMCC.jobj("inputs" => JObj(something(get(t, "inputs", nothing), JObj())), "outputs" => JObj(something(get(t, "outputs", nothing), JObj())))
end

function replay_context(r::RowReplay, f::AIFunction, call::Call)
    if call.up === nothing && f === r.program
        isempty(r.earlier) && return nothing
        ts = Any[replay_turn(f, t) for t in r.earlier]
        original = r.call
        return (ts, fill("", length(ts)), original === nothing ? nothing : (_ -> Any[LMCC.jobj("saw_of" => original)]))
    end
    h = lock(r.lock) do
        i = findfirst(h -> get(h, "program", nothing) == f.definition.name, r.helpers)
        i === nothing && return nothing
        h = popat!(r.helpers, i)
        r.helper_sections[call.id] = String[something(get(h, "sections", nothing), Any[])...]
        h
    end
    h === nothing && return nothing
    ts = Any[replay_turn(f, t) for t in something(get(h, "earlier", nothing), Any[])]
    isempty(ts) && return nothing
    original = get(h, "call", nothing)
    (ts, fill("", length(ts)), original === nothing ? nothing : (_ -> Any[LMCC.jobj("saw_of" => original)]))
end
replay_module_saw(r::RowReplay) = r.call !== nothing && !isempty(r.earlier) ? Any[LMCC.jobj("saw_of" => r.call)] : Any[]
function replay_sections(r::RowReplay, call::Call)
    call.up === nothing && call.fn === r.program && return copy(r.sections)
    lock(() -> copy(get(r.helper_sections, call.id, String[])), r.lock)
end
replay_rows(r::RowReplay) = Any[merge(JObj(something(get(t, "inputs", nothing), JObj())), JObj(something(get(t, "outputs", nothing), JObj()))) for t in r.earlier]

"Ask a rated row again with the context its call had: `replaying(() -> f(...), row, f)`; a row with none changes nothing."
function replaying(thunk, row::AbstractDict, program)
    has = row_meta(row, "earlier") !== nothing && !isempty(row_meta(row, "earlier")) ||
          row_meta(row, "helpers") !== nothing && !isempty(row_meta(row, "helpers")) ||
          row_meta(row, "sections") !== nothing && !isempty(row_meta(row, "sections"))
    has ? with(thunk, REPLAYING => RowReplay(row, program)) : thunk()
end

"""
    FunctAI.call_tree(turn) -> String

The calls inside a turn, as an indented tree (from its store's kept log):
the program, its helpers, the calls their tools made, and how each ended.
"""
function call_tree(t::Turn)
    calls = calls_in(t)
    kids = Dict{Any,Vector{Any}}()
    for c in calls
        push!(get!(() -> Any[], kids, c.call == t.id ? nothing : c.parent), c)
    end
    lines = String[]
    function walk(c, prefix, last_child, top)
        mark = top ? "" : (last_child ? "└─ " : "├─ ")
        state = c.ended == "done" ? "" : " [$(something(c.ended, "running"))]"
        push!(lines, prefix * mark * c.function * state)
        children = get(kids, c.call, Any[])
        for (i, k) in enumerate(children)
            walk(k, prefix * (top ? "" : (last_child ? "   " : "│  ")), i == length(children), false)
        end
    end
    for root in get(kids, nothing, Any[])
        walk(root, "", true, true)
    end
    join(lines, "\n")
end
