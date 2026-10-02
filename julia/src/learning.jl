# Learning from conversations (stage 5, contract/calls.md, "Rows that keep
# their context"), and keeping a long-running log: a rated call's earlier
# turns as data its row carries, a split that keeps each conversation on one
# side, pruning old days of the call log without losing what ratings need,
# and checking a judge's evidence.

"""
The turn a `saw` entry stands for, as data: the record's inputs and outputs
without the entry's `without`; with `steps`, the record's steps and its
`program.signature` (contract/calls.md, "Saw").
"""
function turn_of_entry(rec, entry)
    left = Set(String[x for x in something(get(entry, "without", nothing), Any[])])
    out = LMCC.jobj("inputs" => JObj(k => v for (k, v) in something(get(rec, "inputs", nothing), JObj()) if !(k in left)),
                    "outputs" => JObj(k => v for (k, v) in something(get(rec, "outputs", nothing), JObj()) if !(k in left)))
    if get(entry, "steps", false) === true
        get(rec, "steps", nothing) isa AbstractVector || throw(SawUnknown("not-kept", string(get(rec, "id", ""))))
        out["steps"] = LMCC.deepcopy_json(rec["steps"])
        out["signature"] = get(something(get(rec, "program", nothing), JObj()), "signature", nothing)
    end
    out
end

"Whether `rec` runs inside the call `ancestor` (following parents)."
function runs_under(rec, ancestor, by_id)
    seen = Set{String}()
    parent = get(rec, "parent", nothing)
    while parent !== nothing && !(parent in seen)
        parent == ancestor && return true
        push!(seen, parent)
        parent = get(get(by_id, parent, JObj()), "parent", nothing)
    end
    false
end

"""
    FunctAI.earlier_of(call, records) -> Dict

What a rated call was shown before its own inputs, as data a row carries
(contract/calls.md, "Rows that keep their context"): `earlier` (the turns it
was shown, in order), `conversation` (its conversation's id, or `nothing`),
`sections` (what plugins added to its instruction, when any) and, for a
program's call, `helpers` (each call inside it that was shown earlier turns,
in the order they started: `{program, call, earlier}`). Throws
[`SawUnknown`](@ref) when the log cannot show them again (never a part).
"""
function earlier_of(call::AbstractString, records)
    by_id = records isa AbstractDict ? records : records_by_id(records)
    rec = get(by_id, call, nothing)
    rec === nothing && throw(SawUnknown("missing-call", String(call)))
    earlier = Any[]
    if !isempty(something(get(rec, "saw", nothing), Any[]))
        keeps_saw(collect(values(by_id)), call)
        earlier = Any[turn_of_entry(by_id[e["call"]], e) for e in saw(collect(values(by_id)), call)]
    end
    conv = get(rec, "conversation", nothing)
    out = LMCC.jobj("earlier" => earlier, "conversation" => conv isa AbstractDict ? get(conv, "id", nothing) : nothing)
    isempty(something(get(rec, "sections", nothing), Any[])) || (out["sections"] = Any[rec["sections"]...])
    if get(something(get(rec, "program", nothing), JObj()), "kind", nothing) == "module"
        inside = [c for c in values(by_id) if get(c, "root", nothing) == get(rec, "root", nothing) && get(c, "id", nothing) != call &&
                  runs_under(c, call, by_id) && (!isempty(something(get(c, "saw", nothing), Any[])) || !isempty(something(get(c, "sections", nothing), Any[])))]
        sort!(inside; by=c -> (string(get(c, "started", "")), string(get(c, "id", ""))))
        helpers = Any[]
        for c in inside
            keeps_saw(collect(values(by_id)), c["id"])
            h = LMCC.jobj("program" => get(c["program"], "name", nothing), "call" => c["id"],
                          "earlier" => Any[turn_of_entry(by_id[e["call"]], e) for e in saw(collect(values(by_id)), c["id"])])
            isempty(something(get(c, "sections", nothing), Any[])) || (h["sections"] = Any[c["sections"]...])
            push!(helpers, h)
        end
        out["helpers"] = helpers
    end
    out
end

"Whether a rated call was shown anything before its inputs: itself, or, for a program, a call inside it (a helper that remembers)."
function needs_context(rec, by_id)
    rec === nothing && return false
    (!isempty(something(get(rec, "saw", nothing), Any[])) || get(rec, "conversation", nothing) !== nothing ||
     !isempty(something(get(rec, "sections", nothing), Any[]))) && return true
    get(something(get(rec, "program", nothing), JObj()), "kind", nothing) == "module" || return false
    any(c -> get(c, "root", nothing) == get(rec, "root", nothing) &&
             (!isempty(something(get(c, "saw", nothing), Any[])) || !isempty(something(get(c, "sections", nothing), Any[]))) &&
             runs_under(c, rec["id"], by_id), values(by_id))
end

"""
Rows of calls that were shown earlier turns (a conversation), each with
`earlier`, `conversation` (and `helpers` for a program's, `sections` when
plugins gave some), and how many rows were left out because the log cannot
show those turns again. When no row was shown any, the rows are as they were.
"""
function with_context(rows, calls)
    by_id = records_by_id(calls)
    needs = [r for r in rows if needs_context(get(by_id, row_meta(r, "call"), nothing), by_id)]
    isempty(needs) && return (rows, 0)
    out, dropped = JObj[], 0
    meta = ("call", "version", "rating", "rated_by", "origin", "sample", "disputed")
    for r in rows
        ctx = try
            earlier_of(row_meta(r, "call"), by_id)
        catch err
            err isa SawUnknown || rethrow()
            dropped += 1
            continue
        end
        ks = collect(keys(r))
        n_meta = count(k -> lstrip(k, '_') in meta, ks[max(1, end - length(meta) + 1):end])
        rebuilt = JObj(k => r[k] for k in ks[1:end-n_meta])
        add_meta!(rebuilt, ctx)                       # the context goes before the rating's columns
        for k in ks[end-n_meta+1:end]
            rebuilt[k] = r[k]
        end
        push!(out, rebuilt)
    end
    (out, dropped)
end

"""
    train_test(rows; by = :conversation, test = 0.2, seed = 0) -> (train, test)

Two sets of rows with every group of rows on one side: `train, test =
train_test(rated(tutor))`. Turns of one conversation depend on each other: a
test row whose conversation is also in the training rows measures memory,
not the program. A row with no group (`conversation` missing) is a group of
its own. `test` is the share of groups in the test side (at least one group
each side when there are two or more). Python's `functai.split`.
"""
function train_test(rows; by=:conversation, test::Real=0.2, seed::Integer=0)
    by = String(by)
    0 < test < 1 || throw(ArgumentError("test is a share between 0 and 1, not $test"))
    items = rows_of(rows)
    key(r) = r isa NamedTuple ? (haskey(r, Symbol(by)) ? r[Symbol(by)] : throw(ArgumentError("the rows have no column $by"))) :
             r isa AbstractDict ? (haskey(r, by) ? r[by] : haskey(r, Symbol(by)) ? r[Symbol(by)] : throw(ArgumentError("the rows have no column $by"))) :
             Tables.getcolumn(r, Symbol(by))
    groups = Any[]
    of = Any[]
    for (i, r) in enumerate(items)
        g = key(r)
        g = (g === nothing || g === missing) ? (:row, i) : g
        push!(of, g)
        any(x -> isequal(x, g), groups) || push!(groups, g)
    end
    shuffle!(Xoshiro(seed), groups)
    n = length(groups) > 1 ? min(max(1, round(Int, length(groups) * test)), length(groups) - 1) : 0
    held = Set(groups[1:n])
    ([r for (r, g) in zip(items, of) if !(g in held)], [r for (r, g) in zip(items, of) if g in held])
end

# ------------------------------------------------------------------ pruning the call log

"A time before now: `\"90d\"`, `\"12w\"`, `\"36h\"`, a `Dates.Period`, a `Date` or a `DateTime` (UTC)."
function time_before(x)
    x isa Dates.DateTime && return x
    x isa Dates.Date && return Dates.DateTime(x)
    x isa Dates.Period && return Dates.now(Dates.UTC) - x
    m = x isa AbstractString ? match(r"^\s*(\d+)\s*([dwh])\s*$", x) : nothing
    m === nothing && throw(ArgumentError("older_than is a time: \"90d\", \"12w\", a Date, a DateTime or a period; not $(repr(x))"))
    n = parse(Int, m.captures[1])
    Dates.now(Dates.UTC) - (m.captures[2] == "d" ? Dates.Day(n) : m.captures[2] == "w" ? Dates.Week(n) : Dates.Hour(n))
end

"""
    FunctAI.prune_calls(older_than = "90d"; folder, keep_rated = true) -> NamedTuple

Delete the call log's day folders older than a time, keeping what ratings
need (contract/calls.md, "The folder"). Before a day goes, every rated call
in it is kept: the call, every call of its tree (a program's helpers), every
call its `saw` names (the earlier turns a row replays) and their ratings are
copied into one file at the folder's top level (`kept-<host>-<pid>-<hex>.jsonl`),
which every reader reads. A row of `rated` made before pruning is made the
same after. Returns `(days, calls, kept)`: folders deleted, calls deleted, calls kept.
"""
function prune_calls(older_than="90d"; folder=nothing, keep_rated::Bool=true)
    root = log_root(folder)
    (root === nothing || !isdir(root)) && return (days=0, calls=0, kept=0)
    first_day = Dates.format(time_before(older_than), dateformat"yyyy-mm-dd")
    old_days = sort!([joinpath(root, d) for d in readdir(root) if occursin(r"^\d{4}-\d{2}-\d{2}$", d) && isdir(joinpath(root, d)) && d < first_day])
    isempty(old_days) && return (days=0, calls=0, kept=0)
    everything, ratings = read_log(root)
    by_id = Dict(String(c["id"]) => c for c in everything if haskey(c, "id"))
    in_old = Set{String}()
    old_lines = JObj[]
    for day in old_days, f in sort(readdir(day))
        endswith(f, ".jsonl") || continue
        for raw in eachline(joinpath(day, f))
            rec = try
                LMCC.parse_json(raw)
            catch
                continue
            end
            rec isa AbstractDict || continue
            push!(old_lines, rec)
            haskey(rec, "functai_call") && push!(in_old, string(get(rec, "id", "")))
        end
    end
    keep = Set{String}()
    if keep_rated
        rated_ids = Set(string(get(r, "call", "")) for r in ratings)
        trees = Set(get(by_id[c], "root", nothing) for c in rated_ids if haskey(by_id, c))
        keep = Set(cid for (cid, c) in by_id if cid in rated_ids || get(c, "root", nothing) in trees)
        todo = collect(keep)
        while !isempty(todo)                       # every call a kept call's saw names, and theirs
            c = get(by_id, pop!(todo), nothing)
            c === nothing && continue
            for entry in something(get(c, "saw", nothing), Any[])
                entry isa AbstractDict || continue
                for k in ("call", "saw_of")
                    cid = get(entry, k, nothing)
                    if cid isa AbstractString && haskey(by_id, cid) && !(cid in keep)
                        push!(keep, cid)
                        push!(todo, cid)
                    end
                end
            end
        end
    end
    kept_lines = [r for r in old_lines if (haskey(r, "functai_call") && string(get(r, "id", "")) in keep) ||
                                          (haskey(r, "functai_rating") && string(get(r, "call", "")) in keep)]
    if !isempty(kept_lines)
        host = replace(gethostname(), r"[^A-Za-z0-9_.-]" => "_")
        path = joinpath(root, "kept-$host-$(getpid())-$(bytes2hex(rand(UInt8, 3))).jsonl")
        append_bytes(path, reduce(vcat, [record_line(r) for r in kept_lines]); durable=true)
    end
    foreach(d -> rm(d; recursive=true), old_days)
    kept_calls = count(r -> haskey(r, "functai_call"), kept_lines)
    (days=length(old_days), calls=length(setdiff(in_old, keep)), kept=kept_calls)
end

# ------------------------------------------------------------------ a judge's evidence

const SAME_CHARS = Dict{Char,String}('\u2018' => "'", '\u2019' => "'", '\u201a' => "'", '\u201b' => "'", '\u2032' => "'", '`' => "'",
                                     '\u00b4' => "'", '\u201c' => "\"", '\u201d' => "\"", '\u201e' => "\"", '\u201f' => "\"",
                                     '\u2033' => "\"", '\u00ab' => "\"", '\u00bb' => "\"", '\u2010' => "-", '\u2011' => "-",
                                     '\u2012' => "-", '\u2013' => "-", '\u2014' => "-", '\u2015' => "-", '\u2212' => "-",
                                     '\u2026' => "...", '\u00a0' => " ")
const QUOTE_EDGES = collect(" \t\n\"'.,;:!?…“”‘’«»()[]")

"Text as `quotes_found` compares it: Unicode compatibility form, quotes straight, dashes one dash, white space one space, case folded."
function plain_text(text)
    t = Base.Unicode.normalize(string(text), :NFKC)
    t = join(get(SAME_CHARS, c, string(c)) for c in t)
    casefold(strip(replace(t, r"\s+" => " ")))
end

"""
    quotes_found(text, quotes) -> Bool or Vector{Bool}

Whether each quote is in the text, word for word: a judge written as an
ordinary AI function (a score with the quotes it rests on) is only as good
as its quotes, and one that invents evidence is caught here. White space,
case, curly and straight quotes, dashes, and a quote's own surrounding
quotation marks and final punctuation do not count; any other difference
does (a changed word, a paraphrase, an invented sentence). Deterministic,
and costs nothing.

```julia
source = "The parcel left Leeds on Monday. It was delayed by snow."
quotes_found(source, ["“It was delayed by snow”", "It was lost"])    # [true, false]
```
"""
function quotes_found(text, quotes)
    haystack = plain_text(text)
    quotes isa AbstractString && return quote_in(haystack, quotes)
    Bool[quote_in(haystack, q) for q in quotes]
end
function quote_in(haystack, q)
    needle = strip(plain_text(q), QUOTE_EDGES)
    !isempty(needle) && occursin(needle, haystack)
end
