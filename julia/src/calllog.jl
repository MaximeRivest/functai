# The call log (contract/calls.md): every call of an AI function as one line
# of JSON in a folder, ratings of those calls, and the rows with known
# answers they make. The folder is the interface: Python, TypeScript, R,
# Julia and any other tool read and write the same one.

const LOG_FORMAT = 1
const MAX_LINE = 8 * 1024 * 1024
const OFF = Set(["", "0", "false", "no", "off"])
const ON = Set(["1", "true", "yes", "on"])

# ------------------------------------------------------------------ ids and times

"A UUIDv7 (RFC 9562): time-ordered, made when a call starts."
function new_id()
    b = rand(UInt8, 16)
    ms = UInt64(floor(time() * 1000))
    for i in 6:-1:1
        b[i] = UInt8(ms & 0xff)
        ms >>= 8
    end
    b[7] = (b[7] & 0x0f) | 0x70
    b[9] = (b[9] & 0x3f) | 0x80
    h = bytes2hex(b)
    "$(h[1:8])-$(h[9:12])-$(h[13:16])-$(h[17:20])-$(h[21:32])"
end

"RFC 3339 UTC with exactly six fraction digits: times in the log sort as text."
function iso(t::Float64)
    whole = floor(t)
    micro = min(round(Int, (t - whole) * 1e6), 999_999)
    Dates.format(Dates.unix2datetime(whole), dateformat"yyyy-mm-ddTHH:MM:SS") * "." * lpad(micro, 6, '0') * "Z"
end

round6(x) = round(x; digits=6)

# ------------------------------------------------------------------ where

"The contract's default folder: `\$XDG_DATA_HOME/functai/calls`, or the platform's place for data."
function default_log_folder()
    base = if Sys.isapple()
        joinpath(homedir(), "Library", "Application Support")
    elseif Sys.iswindows()
        get(ENV, "LOCALAPPDATA", joinpath(homedir(), "AppData", "Local"))
    else
        x = get(ENV, "XDG_DATA_HOME", "")
        isempty(x) ? joinpath(homedir(), ".local", "share") : x
    end
    joinpath(base, "functai", "calls")
end

expand(p::AbstractString) = abspath(expanduser(p))

"The folder calls are logged to under this setting, or `nothing` when logging is off."
function folder_of(setting)
    setting === false && return nothing
    raw = strip(get(ENV, "FUNCTAI_LOG_CALLS", ""))
    on = !(lowercase(raw) in OFF)
    env_folder = on && !(lowercase(raw) in ON) ? String(raw) : nothing
    setting === nothing && !on && return nothing
    (setting === nothing || setting === true) && return expand(something(env_folder, default_log_folder()))
    expand(setting)
end

function content_of(setting)
    setting === nothing || return Bool(setting)
    raw = lowercase(strip(get(ENV, "FUNCTAI_LOG_CONTENT", "")))
    !(raw in OFF && raw != "")
end

const WARNED = Set{String}()
function warn_once(key, message)
    first_time = lock(WARN_LOCK) do
        key in WARNED ? false : (push!(WARNED, key); true)
    end
    first_time && @warn "functai: $message"
    nothing
end

"Who is calling: `\$FUNCTAI_CALLER`, with the `caller` setting's keys over it."
function caller_of(s::AbstractDict{Symbol})
    raw = get(ENV, "FUNCTAI_CALLER", "")
    base = JObj()
    if !isempty(raw)
        try
            parsed = LMCC.parse_json(raw)
            parsed isa AbstractDict || error("not a JSON object")
            base = JObj(parsed)
        catch err
            warn_once("caller:$raw", "\$FUNCTAI_CALLER is not a JSON object ($(sprint(showerror, err))); ignored")
        end
    end
    for (k, v) in something(setting(s, :caller), Dict())
        base[String(k)] = jsonvalue(v)
    end
    base
end

# ------------------------------------------------------------------ a call

mutable struct Exchange
    model::String
    provider::Union{Nothing,String}
    started::Float64
    seconds::Float64
    cached::Bool
    request::Any
    response::Any
    error::Any
    streamed::Bool
    first_delta::Union{Nothing,Float64}
end

"One call being made: its id, its parent, its exchanges; a line in the log when it ends, if logging is on."
mutable struct Call
    const id::String
    const parent::Union{Nothing,String}
    const root::String
    const started::Float64
    const program::Function              # () -> the program's Dict: its version is computed only when needed
    const folder::Union{Nothing,String}
    const content::Bool
    const caller::JObj
    exchanges::Vector{Exchange}
    provider::Union{Nothing,String}
    inputs::Union{Nothing,JObj}
    sizes::JObj
    outputs::Any
    confidence::Any
end

const CURRENT_CALL = ScopedValue{Union{Nothing,Call}}(nothing)

function exchange!(call::Call, model, request, response, started, seconds; cached=false, error=nothing, streamed=false, first_delta=nothing)
    push!(call.exchanges, Exchange(model, call.provider, started, seconds, cached, request, response, error, streamed, first_delta))
    nothing
end

"Start a call of a program: an id, its parent (the call it runs inside), and its inputs when logging is on."
function start_call(program::Function, s::AbstractDict{Symbol}, inputs::AbstractDict)
    folder, content = nothing, true
    try
        folder = folder_of(setting(s, :log_calls))
        content = content_of(setting(s, :log_content))
    catch err
        warn_once("start:$(typeof(err))", "calls are not logged: $(sprint(showerror, err))")
    end
    parent = CURRENT_CALL[]
    id = new_id()
    call = Call(id, parent === nothing ? nothing : parent.id, parent === nothing ? id : parent.root, time(), program,
                folder, content, caller_of(s), Exchange[], nothing, nothing, JObj(), nothing, nothing)
    if folder !== nothing
        values = JObj()
        for (k, v) in inputs
            data = logvalue(v)
            values[String(k)] = data
            call.sizes[String(k)] = jsonsize(data)
        end
        content && (call.inputs = values)
    end
    call
end

"An exception as the log writes it: `{type, message?, code?}`."
function error_json(err, content::Bool)
    err = unwrap(err)
    out = JObj("type" => error_type(err))
    err isa LMCC.Refusal && (out["code"] = err.code)
    content && (out["message"] = error_message(err))
    out
end

unwrap(err) = err isa TaskFailedException ? unwrap(err.task.exception) :
              err isa CompositeException && !isempty(err.exceptions) ? unwrap(first(err.exceptions)) : err
error_type(err) = err isa LM15.LM15Error ? String(LM15.class_name(err)) : String(nameof(typeof(err)))
error_message(err) = err isa LMCC.Refusal ? err.hint : err isa LM15.LM15Error ? err.message :
                     hasproperty(err, :msg) && getproperty(err, :msg) isa AbstractString ? String(err.msg) :
                     sprint(showerror, err)

function usage_of(response)
    u = get(LM15.to_dict(response), "usage", JObj())
    JObj(String(k) => Int(v) for (k, v) in u if v isa Integer && !(v isa Bool))
end

function exchange_json(ex::Exchange, content::Bool)
    out = JObj("model" => ex.model, "provider" => ex.provider, "started" => iso(ex.started),
               "seconds" => round6(ex.seconds), "cached" => ex.cached)
    if ex.streamed
        out["streamed"] = true
        out["first_delta"] = ex.first_delta === nothing ? nothing : round6(ex.first_delta)
    end
    if ex.response !== nothing
        out["finish"] = ex.response.finish_reason
        out["usage"] = usage_of(ex.response)
    end
    ex.error === nothing || (out["error"] = error_json(ex.error, content))
    if content
        out["request"] = LM15.to_dict(ex.request)
        ex.response === nothing || (out["response"] = LM15.to_dict(ex.response))
    end
    out
end

const PROCESS = Ref{Union{Nothing,JObj}}(nothing)
function process_json()
    if PROCESS[] === nothing
        user = something(get(ENV, "USER", nothing), get(ENV, "USERNAME", nothing), try
            Sys.username()
        catch
            nothing
        end, Some(nothing))
        PROCESS[] = LMCC.jobj("host" => gethostname(), "pid" => getpid(), "user" => user, "language" => "julia",
                              "runtime" => string(VERSION), "functai" => string(FUNCTAI_VERSION))
    end
    copy(PROCESS[])
end

"The call's record (contract/calls.md, \"A call record\")."
function call_record(call::Call; returned=nothing, has_returned::Bool=false, error=nothing)
    program = call.program()
    content = call.content
    outputs = nothing
    out_sizes = JObj()
    if call.outputs !== nothing
        outputs = JObj()
        for (k, v) in pairs(call.outputs)
            data = logvalue(v)
            outputs[String(k)] = data
            out_sizes[String(k)] = jsonsize(data)
        end
    end
    if program["kind"] == "module" && error === nothing && has_returned
        data = logvalue(returned)
        outputs = LMCC.jobj("result" => data)
        out_sizes["result"] = jsonsize(data)
    end
    answered = [e for e in call.exchanges if e.response !== nothing]
    usage = JObj()
    for e in answered, (k, v) in usage_of(e.response)
        usage[k] = get(usage, k, 0) + v
    end
    rec = LMCC.jobj("functai_call" => LOG_FORMAT, "id" => call.id, "parent" => call.parent, "root" => call.root,
                    "program" => program, "started" => iso(call.started), "seconds" => round6(time() - call.started),
                    "content" => content)
    if content
        rec["inputs"] = something(call.inputs, JObj())
        rec["outputs"] = outputs
        if program["kind"] == "ai" && has_returned && error === nothing
            shown = logvalue(returned)
            (outputs === nothing || !LMCC.json_equal(get(outputs, program["answer"], nothing), shown)) && (rec["returned"] = shown)
        end
    end
    rec["sizes"] = LMCC.jobj("inputs" => call.sizes, "outputs" => out_sizes)
    rec["error"] = error === nothing ? nothing : error_json(error, content)
    rec["model"] = isempty(answered) ? nothing : last(answered).model
    rec["usage"] = usage
    rec["confidence"] = call.confidence
    rec["exchanges"] = Any[exchange_json(e, content) for e in call.exchanges]
    rec["caller"] = copy(call.caller)
    rec["process"] = process_json()
    rec
end

function log_line(rec::AbstractDict)
    text = LMCC.json_text(rec) * "\n"
    ncodeunits(text) <= MAX_LINE && return text
    rec = copy(rec)
    rec["truncated"] = true
    rec["exchanges"] = Any[JObj(k => v for (k, v) in e if !(k in ("request", "response"))) for e in rec["exchanges"]]
    text = LMCC.json_text(rec) * "\n"
    ncodeunits(text) <= MAX_LINE && return text
    for k in ("inputs", "outputs", "returned", "probabilities")
        delete!(rec, k)
    end
    LMCC.json_text(rec) * "\n"
end

const LOG_FILE = Ref{Union{Nothing,String}}(nothing)
function log_file_name()
    LOG_FILE[] === nothing && (LOG_FILE[] = "$(gethostname())-$(getpid())-$(bytes2hex(rand(UInt8, 3))).jsonl")
    LOG_FILE[]
end

const APPEND_LOCK = ReentrantLock()

"Append one record to this process's file in `folder` (one file per UTC day): one write, `O_APPEND`, `0600`."
function append_record(folder::AbstractString, rec::AbstractDict)
    line = log_line(rec)
    day = joinpath(folder, Dates.format(Dates.unix2datetime(time()), dateformat"yyyy-mm-dd"))
    lock(APPEND_LOCK) do
        isdir(day) || mkpath(day; mode=0o700)
        f = Base.Filesystem.open(joinpath(day, log_file_name()),
                                 Base.Filesystem.JL_O_WRONLY | Base.Filesystem.JL_O_CREAT | Base.Filesystem.JL_O_APPEND, 0o600)
        try
            write(f, codeunits(line))
        finally
            close(f)
        end
    end
    nothing
end

"End a call: write its line when logging is on. Never throws into the call."
function finish_call(call::Call; kw...)
    call.folder === nothing && return nothing
    try
        append_record(call.folder, call_record(call; kw...))
    catch err
        warn_once("$(call.folder):$(typeof(err))",
                  "could not log a call to $(call.folder) ($(sprint(showerror, err))); calls go on, unlogged")
    end
    nothing
end

# ------------------------------------------------------------------ reading

"`(calls, ratings)` logged in a folder, as the Dicts of their lines (a partial or unknown line is skipped)."
function read_log(folder=nothing; since=nothing)
    root = folder === nothing ? folder_of(true) : expand(folder)
    cutoff = since === nothing ? "" : since isa AbstractString ? String(since) :
             since isa Dates.DateTime ? iso(Dates.datetime2unix(since)) : since isa Dates.Date ? iso(Dates.datetime2unix(Dates.DateTime(since))) :
             throw(ArgumentError("since is a Date, a DateTime or an RFC 3339 time"))
    calls, ratings = JObj[], JObj[]
    (root === nothing || !isdir(root)) && return (calls, ratings)
    for day in sort(readdir(root))
        dir = joinpath(root, day)
        (occursin(r"^\d{4}-\d{2}-\d{2}$", day) && isdir(dir)) || continue
        !isempty(cutoff) && day < cutoff[1:10] && continue
        for file in sort(readdir(dir))
            endswith(file, ".jsonl") || continue
            text = try
                read(joinpath(dir, file), String)
            catch
                continue
            end
            for raw in split(text, '\n')
                isempty(strip(raw)) && continue
                rec = try
                    LMCC.parse_json(raw)
                catch
                    continue
                end
                rec isa AbstractDict || continue
                if get(rec, "functai_call", nothing) == LOG_FORMAT && string(get(rec, "started", "")) >= cutoff
                    push!(calls, rec)
                elseif get(rec, "functai_rating", nothing) == LOG_FORMAT && string(get(rec, "at", "")) >= cutoff
                    push!(ratings, rec)
                end
            end
        end
    end
    (calls, ratings)
end

"Whether record `a` is later than `b` by `key` (a time), then by id."
function later(a, b, key)
    ta, tb = string(get(a, key, "")), string(get(b, key, ""))
    ta != tb && return ta > tb
    string(get(a, "id", "")) > string(get(b, "id", ""))
end

"For each call, the ratings that count: each person's latest (a `null` verdict withdraws it)."
function current_ratings(ratings; by=nothing)
    latest = Dict{Tuple{Any,Any},Any}()
    for r in ratings
        by !== nothing && get(r, "by", nothing) != by && continue
        key = (get(r, "call", nothing), get(r, "by", nothing))
        had = get(latest, key, nothing)
        (had === nothing || later(r, had, "at")) && (latest[key] = r)
    end
    out = Dict{String,Vector{Any}}()
    for r in values(latest)
        get(r, "verdict", nothing) in ("right", "wrong") || continue
        push!(get!(out, string(r["call"]), Any[]), r)
    end
    for list in values(out)
        sort!(list; lt=(a, b) -> later(b, a, "at"))
    end
    out
end

function what_it_says(rating, call)
    answer = something(get(get(call, "program", JObj()), "answer", nothing), "result")
    if rating["verdict"] == "right"
        outputs = get(call, "outputs", nothing)
        return outputs isa AbstractDict && haskey(outputs, answer) ? JObj(answer => outputs[answer]) : nothing
    end
    values = JObj()
    haskey(rating, "answer") && (values[answer] = rating["answer"])
    for (k, v) in something(get(rating, "outputs", nothing), JObj())
        haskey(values, k) || (values[k] = v)
    end
    isempty(values) ? nothing : values
end

function add_meta!(row, meta)
    for (key, value) in meta
        while haskey(row, key)
            key = "_" * key
        end
        row[key] = value
    end
end

"""
Rows with known answers from rated calls (contract/calls.md, "Rows with known
answers"): `(rows, left_out)`.
"""
function rated_rows(calls, ratings; name, module_name=nothing, signature=nothing, by=nothing)
    counting = current_ratings(ratings; by)
    left = LMCC.jobj("other_signature" => 0, "no_content" => 0, "no_answer" => 0)
    rows = JObj[]
    mine = [c for c in calls if get(get(c, "program", JObj()), "name", nothing) == name &&
                                (module_name === nothing || get(c["program"], "module", nothing) == module_name)]
    sort!(mine; by=c -> (string(get(c, "started", "")), string(get(c, "id", ""))))
    for call in mine
        rs = get(counting, string(get(call, "id", "")), nothing)
        (rs === nothing || isempty(rs)) && continue
        program = call["program"]
        if signature !== nothing && get(program, "signature", nothing) != signature
            left["other_signature"] += 1
            continue
        end
        if !(get(call, "content", false) === true) || !haskey(call, "inputs")
            left["no_content"] += 1
            continue
        end
        usable = [(r, v) for (r, v) in ((r, what_it_says(r, call)) for r in rs) if v !== nothing]
        if isempty(usable)
            left["no_answer"] += 1
            continue
        end
        rating, values = last(usable)
        disputed = length(Set(r["verdict"] for r in rs)) > 1 || length(Set(LMCC.canonical_json(v) for (_, v) in usable)) > 1
        answer = something(get(program, "answer", nothing), "result")
        row = JObj(String(k) => v for (k, v) in something(call["inputs"], JObj()))
        haskey(values, answer) && (row[answer] = values[answer])
        for (k, v) in values
            k == answer || (row[k] = v)
        end
        add_meta!(row, LMCC.jobj("call" => call["id"], "version" => get(program, "version", nothing), "rating" => rating["verdict"],
                                 "rated_by" => get(rating, "by", nothing), "origin" => something(get(rating, "origin", nothing), "review"),
                                 "sample" => get(rating, "sample", nothing), "disputed" => disputed))
        push!(rows, row)
    end
    (rows, left)
end

# ------------------------------------------------------------------ rating

"A rating record (contract/calls.md, \"A rating record\"), written to the log folder."
function rating_record(call_id::AbstractString, verdict, s; answer=NOTHING_GIVEN, outputs=nothing, note=nothing,
                       reasons=nothing, by=nothing, origin=:review, sample=nothing)
    corrected = answer !== NOTHING_GIVEN || (outputs !== nothing && !isempty(outputs))
    v = if verdict === NOTHING_GIVEN
        corrected || throw(ArgumentError("say whether the answer is right: rate(p, :right), rate(p, :wrong), or rate(p; answer = …)"))
        "wrong"
    elseif verdict === nothing
        nothing
    elseif verdict === true || verdict in (:right, "right")
        "right"
    elseif verdict === false || verdict in (:wrong, "wrong")
        "wrong"
    else
        throw(ArgumentError("a verdict is :right, :wrong, true, false, or nothing (withdraw), not $(repr(verdict))"))
    end
    corrected && v != "wrong" && throw(ArgumentError("a right answer needs no correction: give answer or outputs only with :wrong"))
    origin in (:review, :edit, "review", "edit") || throw(ArgumentError("origin is :review or :edit, not $(repr(origin))"))
    who = something(by, get(caller_of(s), "user", nothing), process_json()["user"], "someone")
    rec = LMCC.jobj("functai_rating" => LOG_FORMAT, "id" => new_id(), "call" => String(call_id), "at" => iso(time()),
                    "by" => String(who), "verdict" => v)
    answer === NOTHING_GIVEN || (rec["answer"] = logvalue(answer))
    outputs === nothing || (rec["outputs"] = JObj(String(k) => logvalue(x) for (k, x) in pairs(outputs)))
    reasons === nothing || isempty(reasons) || (rec["reasons"] = Any[String(r) for r in reasons])
    note === nothing || isempty(note) || (rec["note"] = String(note))
    rec["origin"] = String(origin)
    sample === nothing || (rec["sample"] = String(sample))
    rec
end

"The marker of a keyword not given (`answer = nothing` is an answer: null)."
struct NothingGiven end
const NOTHING_GIVEN = NothingGiven()
