# What a call was shown (contract/calls.md, "Saw"): the earlier calls it was
# given as context, read from the call log, and whether the log keeps what
# showing them again needs. Julia's calls see no earlier call yet (no
# conversation memory), so they write `saw: []`; this reads any language's.

"""
    SawUnknown

What a call saw cannot be known, or shown again (contract/calls.md,
"Saw"): `code` is `not-recorded`, `missing-call`, `unknown-key`,
`saw-cycle`, `not-kept` or `turn-invalid`, and `call` the call whose record
says so.
"""
struct SawUnknown <: FunctAIError
    code::String
    call::String
end
Base.showerror(io::IO, e::SawUnknown) = print(io, "SawUnknown [", e.code, "]: call ", e.call)

const SAW_ENTRY_KEYS = Set(["call", "steps", "without", "slot"])

records_by_id(records) = Dict{String,Any}(string(r["id"]) => r for r in records
                                          if r isa AbstractDict && get(r, "functai_call", nothing) in READ_FORMATS && haskey(r, "id"))

function expand_saw(by_id, call, following=String[])
    rec = get(by_id, call, nothing)
    (rec === nothing || !haskey(rec, "saw")) && throw(SawUnknown(isempty(following) ? "not-recorded" : "missing-call", call))
    out = Any[]
    for (i, entry) in enumerate(rec["saw"])
        entry isa AbstractDict || throw(SawUnknown("unknown-key", call))
        if haskey(entry, "saw_of")
            (i == 1 && length(entry) == 1) || throw(SawUnknown("unknown-key", call))
            target = string(entry["saw_of"])
            (target in following || target == call) && throw(SawUnknown("saw-cycle", target))
            append!(out, expand_saw(by_id, target, vcat(following, call)))
        else
            (haskey(entry, "call") && issubset(keys(entry), SAW_ENTRY_KEYS)) || throw(SawUnknown("unknown-key", call))
            push!(out, LMCC.deepcopy_json(entry))
        end
    end
    out
end

"""
    saw(records, call) -> Vector

The calls a call was given as context, in order, each entry as its record
has it (`call`, and `steps`, `without`, `slot` when present), every
`saw_of` replaced by the entries of the call it names. `records` are call
records (`FunctAI.read_log(folder)[1]`). Throws [`SawUnknown`](@ref) when
they cannot be known: `not-recorded`, `missing-call`, `unknown-key`,
`saw-cycle`.
"""
saw(records, call::AbstractString) = expand_saw(records_by_id(records), String(call))

"Whether a record keeps what the entry says it was shown: the values of the fields shown, as data; with steps, every request hash and reply."
function keeps_values(rec, entry)
    get(rec, "truncated", false) === true && return false
    left_out = Set(something(get(entry, "without", nothing), Any[]))
    sizes = get(rec, "sizes", JObj())
    shown = setdiff(union(keys(get(sizes, "inputs", JObj())), keys(get(sizes, "outputs", JObj()))), left_out)
    isempty(intersect(shown, union(described(rec, "inputs"), described(rec, "outputs")))) || return false
    if get(entry, "steps", false) === true
        return get(rec, "content", false) === true &&
               all(ex -> haskey(ex, "request_hash") && (haskey(ex, "response") || get(ex, "finish", nothing) === nothing), rec["exchanges"])
    end
    get(rec, "content", false) === true && return true
    omitted = get(rec, "omitted", nothing)
    omitted isa AbstractDict || return false                   # format 1, or no value kept
    isempty(setdiff(union(omitted["inputs"], omitted["outputs"]), left_out))
end

"""
    keeps_saw(records, call) -> nothing

Whether the log keeps what showing a call its context again needs
(contract/calls.md, "Knowing is not replaying"): every call it saw is in
the log, not truncated, with the values it was shown as data (and with
`steps`, each exchange's request hash and reply). Throws
[`SawUnknown`](@ref) naming the first call that fails: `turn-invalid` (an
entry no call can have been shown: `steps` without the tool calls),
`missing-call`, `not-kept`, or why what it saw is not known. It shows
nothing: turning a record's steps into a turn is stage 5's.
"""
function keeps_saw(records, call::AbstractString)
    by_id = records_by_id(records)
    for entry in expand_saw(by_id, String(call))
        target = string(entry["call"])
        get(entry, "steps", false) === true && "calls" in something(get(entry, "without", nothing), Any[]) &&
            throw(SawUnknown("turn-invalid", target))
        rec = get(by_id, target, nothing)
        rec === nothing && throw(SawUnknown("missing-call", target))
        keeps_values(rec, entry) || throw(SawUnknown("not-kept", target))
    end
    nothing
end
