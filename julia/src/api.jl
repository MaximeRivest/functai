# The call log's face: rate calls, list them, and turn ratings into rows
# with known answers for `evaluate` and the optimizers.

log_root(folder) = folder === nothing ? folder_of(something(setting(effective(Dict{Symbol,Any}()), :log_calls), true)) : expand(folder)

"""
    rate(call, verdict; answer, outputs, note, reasons, by, origin, sample, folder)

Say whether a call's answer is right, and if not what it should have been
(contract/calls.md, "A rating record"). `call` is a [`Prediction`](@ref) or
its id; `verdict` is `:right`, `:wrong` (or `true`, `false`), or `nothing` to
withdraw your rating. A correction alone is `:wrong`:

```julia
p = predict(mood, "Arrived broken, but support was great.")
rate(p, :right)
rate(p; answer = mixed, note = "broken item, happy with the help")
rate.(predictions, :right)          # many at once
```

The rating is written to the call log folder (the `log_calls` setting, or
`folder`). `by` defaults to the caller's `user`, else your account's name.
"""
rate(p::Prediction, verdict=NOTHING_GIVEN; kw...) = rate(p.call, verdict; kw...)
function rate(call::AbstractString, verdict=NOTHING_GIVEN; answer=NOTHING_GIVEN, outputs=nothing, note=nothing, reasons=nothing,
              by=nothing, origin=:review, sample=nothing, folder=nothing)
    s = effective(Dict{Symbol,Any}())
    rec = rating_record(call, verdict, s; answer, outputs, note, reasons, by, origin, sample)
    root = log_root(folder)
    root === nothing && throw(ArgumentError("no log folder: pass folder, or turn the call log on (log_calls = true, or FUNCTAI_LOG_CALLS)"))
    append_record(root, rec)
    rec
end

log_time(s) = s isa AbstractString && length(s) >= 19 ? Dates.DateTime(first(s, min(23, length(s) - 1)), dateformat"yyyy-mm-ddTHH:MM:SS.s") : missing
getpath(d, keys...) = (for k in keys
    d isa AbstractDict || return missing
    d = get(d, k, missing)
    d === nothing && return missing
end; d)

"""
    calls(f = nothing; folder, since)

The logged calls (of `f`, when given: an AI function, a program, or a name),
oldest first, as rows (a Tables.jl table: `DataFrame(calls(mood))`):
`id`, `started`, `name`, `module`, `version`, `model`, `seconds`, `inputs`,
`outputs`, `error`, tokens, `parent`, `caller` (who called: an evaluation's
calls have `caller["evaluation"] == e.run`), `language`.
"""
function calls(f=nothing; folder=nothing, since=nothing)
    recs, _ = read_log(log_root(folder); since)
    name = f === nothing ? nothing : f isa AbstractString ? f : f isa AIFunction ? f.definition.name : string(nameof(f))
    mod = f isa AIFunction ? f.module_name : f isa AIProgram ? f.module_name : nothing
    mine = [c for c in recs if name === nothing ||
            (getpath(c, "program", "name") == name && (mod === nothing || getpath(c, "program", "module") == mod))]
    sort!(mine; by=c -> (string(get(c, "started", "")), string(get(c, "id", ""))))
    [(id=c["id"], started=log_time(get(c, "started", nothing)), name=getpath(c, "program", "name"),
      module_name=getpath(c, "program", "module"), version=getpath(c, "program", "version"), model=something(get(c, "model", nothing), missing),
      seconds=something(get(c, "seconds", nothing), missing), inputs=getpath(c, "inputs"), outputs=getpath(c, "outputs"),
      error=get(c, "error", nothing) isa AbstractDict ? c["error"]["type"] * (haskey(c["error"], "message") ? ": " * c["error"]["message"] : "") : missing,
      input_tokens=getpath(c, "usage", "input_tokens"), output_tokens=getpath(c, "usage", "output_tokens"),
      reasoning_tokens=getpath(c, "usage", "reasoning_tokens"), total_tokens=getpath(c, "usage", "total_tokens"),
      parent=something(get(c, "parent", nothing), missing), caller=something(get(c, "caller", nothing), Dict{String,Any}()),
      language=getpath(c, "process", "language")) for c in mine]
end

"""
    rated(f; by, folder, since, any_file = false)

Rows with known answers from people's ratings of `f`'s calls
(contract/calls.md, "Rows with known answers"): the inputs, the right answer
under the output's name (typed), then `call`, `version`, `rating`,
`rated_by`, `origin`, `sample` and `disputed`. Ready for [`evaluate`](@ref)
and the optimizers. Calls of format 1 and 2 are read; a call is used when
its interface is `f`'s (or its signature, for records written before
interfaces), so turning reasoning on still pools its ratings. Calls rated
under another interface, whose inputs were not all logged, or rated wrong
with no correction are left out (and counted in an `@info`).
"""
function rated(f::Union{AIFunction,AIProgram}; by=nothing, folder=nothing, since=nothing, any_file::Bool=false)
    recs, ratings = read_log(log_root(folder); since)
    # a function defined at the top level (a notebook, a script) is known by its file too: two notebooks' summarize
    # are two programs; any_file pools across files (a notebook that moved)
    name, mod, file_of = f isa AIFunction ? (f.definition.name, f.module_name, f.file) : (f.name, f.module_name, f.file)
    file = any_file || mod != "__main__" ? nothing : top_level_file(file_of)
    rows, left = rated_rows(recs, ratings; name, module_name=mod, signature=f isa AIFunction ? signature_id(f) : nothing,
                            interface=interface_signature(f), by, file)
    # a rated turn of a conversation carries what it was shown (stage 5): rows the log cannot show it again for are left out
    rows, left["no_context"] = with_context(rows, recs)
    reasons = Dict("no_answer" => "marked wrong without the right answer", "other_signature" => "made when its inputs or outputs were different",
                   "no_content" => "logged without their values", "no_context" => "shown earlier turns the log cannot show again")
    total = sum(values(left))
    total > 0 && @info "rated($name): $total rated call(s) left out: " *
                       join(("$v $(reasons[k])" for (k, v) in left if v > 0), "; ")
    f isa AIFunction ? typed_rows(f, rows) : plain_rows(rows)
end

"Rows as NamedTuples with one set of columns (absent values `missing`), values as JSON reads them."
function plain_rows(rows)
    columns = String[]
    for r in rows, k in keys(r)
        k in columns || push!(columns, k)
    end
    names = Tuple(Symbol.(columns))
    [NamedTuple{names}(Tuple(haskey(r, k) && r[k] !== nothing ? r[k] : missing for k in columns)) for r in rows]
end

"Rows as NamedTuples with one set of columns (absent values `missing`), fields read as their types."
function typed_rows(f::AIFunction, rows)
    columns = String[]
    for r in rows, k in keys(r)
        k in columns || push!(columns, k)
    end
    specs = Dict(x.name => x.spec for x in vcat(f.definition.inputs, f.definition.outputs))
    read(k, v) = v === nothing ? missing : haskey(specs, k) ? (try
        fromjson(specs[k], v)
    catch
        v
    end) : v
    names = Tuple(Symbol.(columns))
    [NamedTuple{names}(Tuple(haskey(r, k) ? read(k, r[k]) : missing for k in columns)) for r in rows]
end
