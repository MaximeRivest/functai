# Views (contract/streaming.md, "Views"): what one kind of reader may see of
# a call tree's log.
#
# - `:full`: every event and value (only the process running the tree has it);
# - `:kept`: what `log_content` lets be kept (a store's form);
# - `:outside`: a caller who sees only the program's boundary (a served
#   program's customer): its `started` (its program object without its file
#   and line), its answer's text as it is written, the approvals addressed to
#   it, and its `done`, or its `failed` with the error's type and code and no
#   message. Never a helper's answer, a tool call or result, a thinking, or
#   why a request was retried.
#
# A view keeps each event's `writer` and `seq` and sets `after` to the event
# before it in the view. It never alters a value it shows and never invents
# one: a module's answer shown as it is written is its `answer_from`
# helper's text, re-addressed to the module.

const VIEW_NAMES = (:full, :kept, :outside)
const OUTSIDE_PROGRAM_KEYS = ("name", "kind", "module", "version", "signature", "interface", "answer")

"A view made event by event: `apply_view!(v, e)` gives the event as the view shows it, or `nothing`."
mutable struct View
    name::Symbol
    answer_from::Union{Nothing,String}
    root::Union{Nothing,String}
    root_function::Union{Nothing,String}
    root_kind::String
    answer::String
    forwarded::Dict{String,Bool}
    requests::Int
    last::Union{Nothing,Position}
    caller_asked::Set{Tuple{String,Any}}
end
function View(name::Symbol; answer_from=nothing)
    name in VIEW_NAMES || throw(ArgumentError("a view is one of $(join(VIEW_NAMES, ", ")); not $(repr(name))"))
    from = answer_from === nothing ? nothing : answer_from isa AbstractString ? String(answer_from) :
           answer_from isa AIFunction ? answer_from.definition.name : string(nameof(answer_from))
    View(name, from, nothing, nothing, "ai", "result", Dict{String,Bool}(), 0, nothing, Set{Tuple{String,Any}}())
end

function link!(v::View, e::Event)
    out = relinked(e, v.last)
    v.last = Position(out)
    out
end

with_kind(e::Event, kind::Symbol, call, fn, data::JObj) = owned_event(kind, e.tree, e.writer, e.seq, e.after, e.at, call, fn, data)

function apply_view!(v::View, e::Event)
    v.name in (:full, :kept) && return link!(v, e)
    if v.root === nothing
        e.kind === :started || return nothing
        v.root = e.call
        v.root_function = getfield(e, :fn)
        program = something(datum(e, "program"), JObj())
        v.root_kind = String(get(program, "kind", "ai"))
        v.answer = String(something(get(program, "answer", nothing), "result"))
    end
    e.call == v.root ? view_root!(v, e) : view_inside!(v, e)
end

function view_root!(v::View, e::Event)
    data = JObj(getfield(e, :data))
    kind = e.kind
    if kind === :started
        data["program"] = JObj(k => x for (k, x) in something(get(data, "program", nothing), JObj()) if k in OUTSIDE_PROGRAM_KEYS)
        delete!(data, "invocation")
        return link!(v, with_data(e, data))
    elseif kind === :request
        v.root_kind == "ai" || return nothing
        v.requests = max(v.requests, Int(something(get(data, "request", nothing), 0)))
        return link!(v, e)
    elseif kind === :retry
        v.root_kind == "ai" || return nothing
        delete!(data, "reason")
        data["content"] = false
        return link!(v, with_data(e, data))
    elseif kind === :text
        return get(data, "answer", false) === true ? link!(v, e) : nothing
    elseif kind in (:approval, :approved)
        return view_approval!(v, e)
    elseif kind === :done
        return link!(v, e)
    elseif kind === :failed
        err = something(get(data, "error", nothing), JObj())
        data["error"] = JObj(k => x for (k, x) in err if k in ("type", "code"))
        data["content"] = false
        return link!(v, with_data(e, data))
    end
    nothing
end

function view_approval!(v::View, e::Event)
    if e.kind === :approval
        datum(e, "to") == "caller" || return nothing
        push!(v.caller_asked, (e.call, datum(e, "invocation")))
    else
        (e.call, datum(e, "invocation")) in v.caller_asked || return nothing
    end
    link!(v, with_kind(e, e.kind, v.root, v.root_function, JObj(getfield(e, :data))))
end

function view_inside!(v::View, e::Event)
    e.kind in (:approval, :approved) && return view_approval!(v, e)
    (v.root_kind == "ai" || v.answer_from === nothing) && return nothing
    if e.kind === :started
        v.forwarded[e.call] = getfield(e, :fn) == v.answer_from
        v.forwarded[e.call] || return nothing
        return as_request!(v, e)
    end
    get(v.forwarded, e.call, false) || return nothing
    e.kind in (:request, :retry) && return as_request!(v, e)
    if e.kind === :text && datum(e, "answer") === true
        data = JObj(getfield(e, :data))
        data["field"] = v.answer
        return link!(v, with_kind(e, :text, v.root, v.root_function, data))
    end
    nothing
end

"A forwarded helper's new request (or its start): the module's answer starts again, as a `request` of the module."
function as_request!(v::View, e::Event)
    v.requests += 1
    link!(v, with_kind(e, :request, v.root, v.root_function, LMCC.jobj("request" => v.requests, "model" => nothing)))
end

"""
    FunctAI.outside(events; answer_from = nothing) -> Vector{Event}

A whole form of a log as the outside view shows it (contract/streaming.md,
"Views"): what a served program's caller may see.
"""
function outside(events; answer_from=nothing)
    v = View(:outside; answer_from)
    Event[x for x in (apply_view!(v, Event(e)) for e in events) if x !== nothing]
end

"The AI function whose answer a program's answer is, as it is written (`@program answer_from = f`)."
answer_from_of(p::AIProgram) = p.answer_from
answer_from_of(_) = nothing
