# Serving a program over HTTP (contract/serving.md), and using one served
# elsewhere (`remote`).
#
#     server = FunctAI.serve(team; port = 8080, keys = "keys.txt", block = false)
#     team2 = remote("http://127.0.0.1:8080"; key = ENV["TEAM_KEY"])
#     team2("I was charged twice for order B-2210.")       # logged here and there: one call tree
#
# What a caller sees is the program's boundary (the outside view): its
# answer, its answer's text as it is written, approvals addressed to it, and
# its end; never a helper's answer, a tool's input or output, a thinking, or
# an error's message. The owner watches everything in their own log.

const SERVE_FORMAT = 1
const MAX_BODY = 16 * 1024 * 1024
const UUID_TEXT = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
const LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")

"What a route answers: a status, headers, and a body (bytes) or a function that writes chunks (Server-Sent Events)."
struct Reply
    status::Int
    headers::Vector{Pair{String,String}}
    body::Any                     # Vector{UInt8}, or a function (write::Function) -> nothing
end
json_reply(status, data) = Reply(status, ["content-type" => "application/json"], Vector{UInt8}(LMCC.json_text(data)))

function error_reply(status, err; message::Bool=false)
    out = LMCC.jobj("type" => error_type(err))
    code = hasproperty(err, :code) ? getproperty(err, :code) : nothing
    code isa AbstractString && (out["code"] = code)
    err isa InterfaceError && err.field !== nothing && (out["field"] = err.field)
    message && (out["message"] = error_message(err))
    json_reply(status, LMCC.jobj("error" => out))
end

"Keys from a list, one key, or a file of one key per line (`#` comments and blank lines skipped)."
function serve_keys(keys)
    keys === nothing && return String[]
    if keys isa AbstractString && isfile(expanduser(keys))
        return [String(strip(l)) for l in eachline(expanduser(keys)) if !isempty(strip(l)) && !startswith(strip(l), "#")]
    end
    keys isa AbstractString && return [String(keys)]
    String[string(k) for k in keys]
end

"Two keys compared in time that does not depend on where they differ."
function same_secret(a::AbstractString, b::AbstractString)
    x, y = codeunits(a), codeunits(b)
    diff = UInt8(length(x) == length(y) ? 0 : 1)
    for i in 1:max(length(x), length(y))
        diff |= (i <= length(x) ? x[i] : 0x00) ⊻ (i <= length(y) ? y[i] : 0x00)
    end
    diff == 0
end

"An event's position as a Server-Sent Events id: `<writer>-<seq>`."
position_id(e::Event) = "$(e.writer)-$(e.seq)"
function parse_position(text)
    m = text === nothing ? nothing : match(r"^\s*(\d+)-(\d+)\s*$", text)
    m === nothing ? nothing : Position(parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end
sse_bytes(e::Event) = Vector{UInt8}("id: $(position_id(e))\nevent: $(e.kind)\ndata: $(LMCC.json_text(event_json(e)))\n\n")

"""
    FunctAI.Service(program; keys, store, lm, approvals = :owner)

A program as an HTTP service, independent of any server:
`FunctAI.handle(service, method, path, headers, body)` answers one request;
[`serve`](@ref) runs it on HTTP.jl's server. `program` is an AI function, a
program, or a saved folder (loaded with `FunctAI.load`).

- `keys`: bearer keys a caller must send (a list, one key, or a file of one
  per line). None: no key, which `serve` allows only on 127.0.0.1.
- `store`: where conversations are kept (as `conversation(…; store)`);
  `nothing` keeps them in this process's memory.
- `lm`: the model every call uses.
- `approvals`: who answers a tool call the program's `approve` rule asks
  about: `:owner` (in their own process; the caller sees the turn waiting),
  or `:caller`, through the approvals route.
"""
struct Service
    program::Any
    keys::Vector{String}
    store::Any
    lm::Union{Nothing,String}
    approvals::String
    lock::ReentrantLock
end
function Service(program; keys=nothing, store=nothing, lm=nothing, approvals=:owner)
    program isa AbstractString && (program = load(program))
    program isa Union{AIFunction,AIProgram} || throw(ArgumentError("serve an AI function, a program, or a saved folder; not a $(typeof(program))"))
    iface = interface(program)
    opaque = [x["name"] for d in ("inputs", "outputs") for x in iface[d] if get(x, "opaque", false) === true]
    isempty(opaque) || throw(ServeError("serve-opaque", "$(program_name(program)) cannot be served: $(join(opaque, ", ")) may hold " *
        "values with no JSON form, and only JSON crosses HTTP. Give $(length(opaque) > 1 ? "them" : "it") a type"))
    String(approvals) in ("owner", "caller") || throw(ArgumentError("approvals go to :owner or :caller, not $(repr(approvals))"))
    program isa AIFunction && lm !== nothing && (program = configure(program; lm=String(lm)))
    Service(program, serve_keys(keys), store, lm === nothing ? nothing : String(lm), String(approvals), ReentrantLock())
end
Base.show(io::IO, s::Service) = print(io, "Service(", program_name(s.program), isempty(s.keys) ? "" : ", $(length(s.keys)) keys", ")")

"The program, described (`GET /interface`): its interface with a format number, as it is served outside a saved manifest."
function describe(s::Service)
    info = program_of(s.program)
    LMCC.jobj("functai_interface" => SERVE_FORMAT, "name" => program_name(s.program), "kind" => info["kind"], "version" => info["version"],
              "interface" => LMCC.deepcopy_json(interface(s.program)), "answer" => info["answer"], "model" => s.lm, "approvals" => s.approvals)
end

"The same as OpenAPI 3.1 (`GET /openapi.json`)."
function openapi(s::Service)
    iface = interface(s.program)
    ins = LMCC.jobj("type" => "object", "properties" => JObj(x["name"] => x["shape"] for x in iface["inputs"]),
                    "required" => Any[x["name"] for x in iface["inputs"] if get(x, "optional", false) !== true])
    outs = LMCC.jobj("type" => "object", "properties" => JObj(x["name"] => x["shape"] for x in iface["outputs"]))
    schema = LMCC.jobj("type" => "object", "properties" => LMCC.jobj("inputs" => ins), "required" => Any["inputs"])
    body = LMCC.jobj("required" => true, "content" => LMCC.jobj("application/json" => LMCC.jobj("schema" => schema)))
    answer = LMCC.jobj("type" => "object", "properties" => LMCC.jobj("call" => LMCC.jobj("type" => "string"), "outputs" => outs, "value" => JObj()))
    call_ok = LMCC.jobj("description" => "the answer", "content" => LMCC.jobj("application/json" => LMCC.jobj("schema" => answer)))
    stream_ok = LMCC.jobj("description" => "Server-Sent Events", "content" => LMCC.jobj("text/event-stream" => JObj()))
    paths = LMCC.jobj("/call" => LMCC.jobj("post" => LMCC.jobj("requestBody" => body, "responses" => LMCC.jobj("200" => call_ok))),
                      "/stream" => LMCC.jobj("post" => LMCC.jobj("requestBody" => body, "responses" => LMCC.jobj("200" => stream_ok))))
    LMCC.jobj("openapi" => "3.1.0",
              "info" => LMCC.jobj("title" => program_name(s.program), "version" => describe(s)["version"], "description" => iface["description"]),
              "paths" => paths,
              "components" => LMCC.jobj("securitySchemes" => LMCC.jobj("key" => LMCC.jobj("type" => "http", "scheme" => "bearer"))),
              "security" => isempty(s.keys) ? Any[] : Any[LMCC.jobj("key" => Any[])])
end

html_escape(t) = replace(string(t), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;", "'" => "&#39;")

"A form (`GET /`), which asks for the key itself."
function form(s::Service)
    iface = interface(s.program)
    name = html_escape(program_name(s.program))
    rows = join("<label>$(html_escape(x["name"]))<br><textarea name=\"$(html_escape(x["name"]))\" rows=\"3\"></textarea></label><br>" for x in iface["inputs"])
    "<!doctype html><meta charset=utf-8><title>$name</title>" *
    "<style>body{font:16px system-ui;max-width:40em;margin:2em auto}textarea{width:100%}</style>" *
    "<h1>$name</h1><p>$(html_escape(iface["description"]))</p>" *
    "<form id=f>$rows<label>key <input name=__key type=password></label> <button>Ask</button></form><pre id=out></pre><script>" *
    "f.onsubmit=async e=>{e.preventDefault();const d=new FormData(f),inputs={};" *
    "for(const[k,v]of d)if(k!='__key'){try{inputs[k]=JSON.parse(v)}catch{inputs[k]=v}}" *
    "const r=await fetch('call',{method:'POST',headers:{'content-type':'application/json'," *
    "authorization:'Bearer '+d.get('__key')},body:JSON.stringify({inputs})});" *
    "out.textContent=JSON.stringify(await r.json(),null,2)}</script>"
end

function authorized(s::Service, headers)
    isempty(s.keys) && return true
    got = get(headers, "authorization", "")
    startswith(lowercase(got), "bearer ") || return false
    given = strip(got[8:end])
    any(k -> same_secret(given, k), s.keys)
end

"""
    FunctAI.handle(service, method, path, headers, body) -> Reply

Answer one request (`headers` a Dict of lower-case names): JSON in, JSON
out; errors `{"error": {"type", "code"?, "field"?}}`, with a message only
when it quotes the caller's own input (contract/serving.md, "Routes").
"""
function handle(s::Service, method::AbstractString, path::AbstractString, headers::AbstractDict, body=UInt8[])
    path = "/" * strip(first(Base.split(path, '?')), '/')
    headers = Dict{String,String}(lowercase(String(k)) => String(v) for (k, v) in headers)
    try
        method == "GET" && path == "/" && return Reply(200, ["content-type" => "text/html; charset=utf-8"], Vector{UInt8}(form(s)))
        authorized(s, headers) || return json_reply(401, LMCC.jobj("error" => LMCC.jobj("type" => "Unauthorized")))
        method == "GET" && path == "/interface" && return json_reply(200, describe(s))
        method == "GET" && path == "/openapi.json" && return json_reply(200, openapi(s))
        data = isempty(body) ? JObj() : try
            LMCC.parse_json(String(copy(body)))
        catch
            return json_reply(400, LMCC.jobj("error" => LMCC.jobj("type" => "BadRequest", "message" => "the body is not JSON")))
        end
        data isa AbstractDict || return json_reply(400, LMCC.jobj("error" => LMCC.jobj("type" => "BadRequest", "message" => "the body is a JSON object")))
        parent = get(headers, "functai-parent", nothing)
        parent = parent !== nothing && occursin(UUID_TEXT, parent) ? parent : nothing
        parts = [p for p in Base.split(path, '/') if !isempty(p)]
        method == "POST" && path == "/call" && return serve_call(s, data, parent)
        method == "POST" && path == "/stream" && return serve_stream(s, data, parent)
        if length(parts) >= 3 && parts[1] == "conversations" && parts[3] == "turns"
            return serve_conversation(s, method, String(parts[2]), String.(parts[4:end]), data, headers, parent)
        end
        json_reply(404, LMCC.jobj("error" => LMCC.jobj("type" => "NotFound")))
    catch err
        err = unwrap(err)
        err isa InterfaceError && return error_reply(422, err; message=err.code == "interface-input")
        if err isa ConversationError
            status = err.code == "turn-unknown" ? 404 : err.code == "conversation-id" ? 400 : 409
            return error_reply(status, err; message=true)
        end
        err isa FunctAIError && return error_reply(409, err)
        err isa KeyError && return json_reply(404, LMCC.jobj("error" => LMCC.jobj("type" => "NotFound")))
        error_reply(500, err)                       # the caller sees the type, never the message (outside view)
    end
end

"""
The request's inputs, checked against the program's interface before
anything runs (a missing, unknown or unbindable input is `interface-input`,
422), bound by name: an AI function's with its defaults, a program's as its
call binds them.
"""
function served_inputs(s::Service, data)
    inputs = get(data, "inputs", JObj())
    inputs isa AbstractDict || throw(InterfaceError("interface-input", nothing, "inputs is a JSON object of the program's inputs"))
    given = OrderedDict{String,Any}(String(k) => v for (k, v) in inputs)
    check_inputs(interface(s.program), given)
    s.program isa AIFunction && return with_defaults(s.program, given)
    (given=given, extra=0, twice=String[], positional=String[])
end

"Run `f` as a served call: its caller (`kind: api`), where approvals go, the remote caller's call as its parent, the service's model."
function in_service(f, s::Service, parent)
    caller = merge(Dict{String,Any}(caller_of(effective(Dict{Symbol,Any}()))), Dict{String,Any}("kind" => "api"))
    settings = Dict{Symbol,Any}(:caller => caller)
    s.program isa AIProgram && s.lm !== nothing && (settings[:lm] = s.lm)
    with(APPROVALS_TO => s.approvals, REMOTE_PARENT => parent) do
        with_settings(f; settings...)
    end
end

function served_outputs(s::Service, value, outputs=nothing)
    names = [x["name"] for x in interface(s.program)["outputs"]]
    outputs !== nothing && return JObj(k => logvalue(v) for (k, v) in pairs(outputs) if String(k) in names)
    length(names) == 1 && return JObj(only(names) => logvalue(value))
    value isa Union{NamedTuple,AbstractDict} ? JObj(String(k) => logvalue(v) for (k, v) in pairs(value) if String(k) in names) :
                                               JObj(last(names) => logvalue(value))
end

function serve_call(s::Service, data, parent)
    bound = served_inputs(s, data)
    if s.program isa AIFunction
        p = in_service(() -> predict_inputs(s.program, bound), s, parent)
        p === missing && throw(InterfaceError("interface-input", nothing, "an input is missing"))
        names = [x["name"] for x in interface(s.program)["outputs"]]
        outs = JObj(String(k) => logvalue(v) for (k, v) in pairs(p.outputs) if String(k) in names)
        return json_reply(200, LMCC.jobj("call" => p.call, "outputs" => outs, "value" => logvalue(p.value)))
    end
    st = in_service(() -> start_stream(() -> run_program(s.program, bound); passive=true), s, parent)
    value = fetch(st)
    json_reply(200, LMCC.jobj("call" => getfield(st, :outer), "outputs" => served_outputs(s, value), "value" => logvalue(value)))
end

"Events as Server-Sent Events, through a view; the stream is closed when the reader goes."
sse_reply(events_of::Function, closing=nothing) = Reply(200, ["content-type" => "text/event-stream", "cache-control" => "no-cache"],
    function (write)
        try
            events_of(e -> write(sse_bytes(e)))
        finally
            closing === nothing || closing()
        end
    end)

function serve_stream(s::Service, data, parent)
    bound = served_inputs(s, data)
    st = in_service(() -> s.program isa AIFunction ? start_stream(() -> predict_inputs(s.program, bound)) :
                          start_stream(() -> run_program(s.program, bound)), s, parent)
    v = View(:outside; answer_from=answer_from_of(s.program))
    sse_reply(() -> close(st)) do put
        for e in eachevent(st)
            shown = apply_view!(v, e)
            shown === nothing || put(shown)
        end
    end
end

served_conversation(s::Service, cid) = conversation(s.program, cid; store=s.store)

function turn_json(s::Service, t::Turn)
    out = LMCC.jobj("turn" => t.id, "parent" => t.parent, "state" => t.state, "inputs" => t.inputs)
    if t.state == "done"
        names = [x["name"] for x in interface(s.program)["outputs"]]
        out["outputs"] = JObj(k => v for (k, v) in t.outputs if k in names)
        out["value"] = logvalue(t.result)
    end
    if t.state == "waiting" && s.approvals == "caller"
        out["waiting"] = Any[LMCC.jobj("invocation" => a.invocation, "name" => a.name, "input" => logvalue(a.input), "path" => a.path) for a in t.waiting]
    end
    err = t.error
    err === nothing || (out["error"] = JObj(k => v for (k, v) in err if k in ("type", "code")))
    out
end

function serve_conversation(s::Service, method, cid, rest, data, headers, parent)
    chat = served_conversation(s, cid)
    if method == "POST" && isempty(rest)
        bound = served_inputs(s, data)
        after = get(data, "after", nothing)
        after === nothing || (chat = continue_from(chat, after))
        rid = get(data, "request_id", nothing)
        st = in_service(() -> send_bound_turn(chat, bound, Dict{Symbol,Any}(), rid), s, parent)
        if get(data, "wait", false) === true
            try
                fetch(st)
            catch
                # the turn's state says how it went
            end
        end
        tid = st isa StoredTurn ? st.turn : getfield(st, :conv_turn)[2]
        out = turn_json(s, turn(chat, tid))
        out["conversation"] = cid
        return json_reply(201, out)
    end
    method == "GET" && isempty(rest) && return json_reply(200, LMCC.jobj("conversation" => cid, "turns" => Any[turn_json(s, t) for t in turns(chat)]))
    isempty(rest) && return json_reply(404, LMCC.jobj("error" => LMCC.jobj("type" => "NotFound")))
    t = turn(chat, rest[1])
    method == "GET" && length(rest) == 1 && return json_reply(200, turn_json(s, t))
    if method == "GET" && rest[2:end] == ["events"]
        after = parse_position(get(headers, "last-event-id", nothing))
        return sse_reply() do put
            for e in eachevent(t; after, view=:outside)
                put(e)
            end
        end
    end
    if method == "POST" && rest[2:end] == ["stop"]
        stop!(t)
        return json_reply(202, LMCC.jobj("turn" => t.id, "stopping" => true))
    end
    if method == "POST" && length(rest) == 3 && rest[2] == "approvals"
        s.approvals == "caller" || return json_reply(403, LMCC.jobj("error" => LMCC.jobj("type" => "Forbidden", "message" => "approvals go to the owner")))
        verdict = get(data, "verdict", nothing)
        verdict in ("yes", "no") || return json_reply(400, LMCC.jobj("error" => LMCC.jobj("type" => "BadRequest", "message" => "verdict is 'yes' or 'no'")))
        inv = parse(Int, rest[3])
        in_service(s, parent) do
            verdict == "yes" ? approve!(t, inv; resume=false) : deny!(t, inv; reason=get(data, "reason", nothing), resume=false)
            again = turn(chat, t.id)
            if isempty(again.waiting)
                # goes on in the background, in the caller's scope: approvals stay addressed to it
                errormonitor(Threads.@spawn try
                    resume!(again)
                catch
                    # the turn's records say how it went
                end)
            end
        end
        return json_reply(202, LMCC.jobj("turn" => t.id, "verdict" => verdict))
    end
    json_reply(404, LMCC.jobj("error" => LMCC.jobj("type" => "NotFound")))
end

"""
    serve(program; host = "127.0.0.1", port = 8080, keys, store, lm, approvals = :owner, block = true)

Serve a program over HTTP (contract/serving.md): its interface, calls,
streams and conversations, to callers who see only its boundary.

| route | answer |
|---|---|
| `GET /interface` | the program, described (`{"functai_interface": 1, …}`) |
| `GET /openapi.json` | the same, as OpenAPI 3.1; `GET /`: a form |
| `POST /call` `{"inputs": …}` | `{"call", "outputs", "value"}`; a refused input `422` |
| `POST /stream` | Server-Sent Events of the outside view |
| `POST /conversations/<id>/turns` | a turn, saved before it runs (`after`, `request_id`, `wait`) |
| `GET /conversations/<id>/turns[/<turn>[/events]]` | the branch's turns, one turn, its events (`Last-Event-ID` resumes) |
| `POST …/turns/<turn>/stop`, `…/approvals/<invocation>` | stop it; answer an approval (`approvals = :caller`) |

Without `keys` it listens only on this machine (`serve-keys` otherwise).
`block = false` serves on a task and returns the server (`close(server)`
stops it). A caller that is a FunctAI call sends `FunctAI-Parent`: the
served call names it as its parent, so the two logs make one call tree.
"""
function serve(program; host::AbstractString="127.0.0.1", port::Integer=8080, keys=nothing, store=nothing, lm=nothing,
               approvals=:owner, block::Bool=true)
    s = program isa Service ? program : Service(program; keys, store, lm, approvals)
    isempty(s.keys) && !(host in LOCAL_HOSTS) &&
        throw(ServeError("serve-keys", "serving on $host lets anyone on the network call $(program_name(s.program)) and spend your " *
                                       "model budget: give keys (keys = \"keys.txt\"), or serve on 127.0.0.1"))
    server = HTTP.serve!(host, Int(port); stream=true, verbose=-1) do http::HTTP.Stream
        serve_http(s, http)
    end
    block || return server
    try
        wait(server)
    catch err
        err isa InterruptException || rethrow()
    finally
        close(server)
    end
    server
end

function serve_http(s::Service, http::HTTP.Stream)
    req = http.message
    n = something(tryparse(Int, HTTP.header(req, "content-length", "0")), -1)
    if n < 0 || n > MAX_BODY
        HTTP.setstatus(http, n > MAX_BODY ? 413 : 400)
        HTTP.setheader(http, "content-length" => "0")
        HTTP.startwrite(http)
        return
    end
    body = read(http)
    headers = Dict{String,String}(lowercase(String(k)) => String(v) for (k, v) in req.headers)
    r = handle(s, String(req.method), String(req.target), headers, body)
    HTTP.setstatus(http, r.status)
    for (k, v) in r.headers
        HTTP.setheader(http, k => v)
    end
    if r.body isa AbstractVector{UInt8}
        HTTP.setheader(http, "content-length" => string(length(r.body)))
        HTTP.startwrite(http)
        write(http, r.body)
        return
    end
    HTTP.startwrite(http)
    try
        r.body(chunk -> (write(http, chunk); flush(http)))
    catch err
        err isa Union{Base.IOError,EOFError,ArgumentError} || rethrow()     # the reader went away: its stream is closed
    end
    nothing
end

# ------------------------------------------------------------------ a program served elsewhere

"""
    RemoteError

The server answered with an error: `code` is its code (or `remote-<status>`),
`status` the HTTP status, `type` its error's type.
"""
struct RemoteError <: FunctAIError
    code::String
    msg::String
    status::Int
    type::String
end
Base.showerror(io::IO, e::RemoteError) = print(io, "RemoteError [", e.code, "]: ", e.msg)

function remote_request(url, key, data=nothing; timeout=120.0, parent=nothing, stream=false)
    headers = Pair{String,String}["accept" => stream ? "text/event-stream" : "application/json"]
    key === nothing || push!(headers, "authorization" => "Bearer $key")
    parent === nothing || push!(headers, "functai-parent" => parent)
    body = data === nothing ? UInt8[] : Vector{UInt8}(LMCC.json_text(data))
    data === nothing || push!(headers, "content-type" => "application/json")
    resp = HTTP.request(data === nothing ? "GET" : "POST", url, headers, body; status_exception=false, readtimeout=round(Int, timeout),
                        retry=false)
    if resp.status >= 400
        err = try
            something(get(LMCC.parse_json(String(resp.body)), "error", nothing), JObj())
        catch
            JObj()
        end
        code = String(something(get(err, "code", nothing), "remote-$(resp.status)"))
        text = String(something(get(err, "message", nothing), "$url answered $(resp.status) ($(something(get(err, "type", nothing), "error")))"))
        code in ("interface-input", "interface-output") && throw(InterfaceError(code, get(err, "field", nothing), text))
        throw(RemoteError(code, text, resp.status, String(something(get(err, "type", nothing), ""))))
    end
    LMCC.parse_json(String(resp.body))
end

"""
    remote(url; key = nothing, timeout = 120) -> AIProgram

A program served elsewhere ([`serve`](@ref), in any FunctAI language), used
like a local one: its interface is the server's (`GET /interface`), its
inputs are bound and checked here before anything is sent, and what comes
back is checked against its outputs. Each call is logged here (`program.kind`
`"remote"`, `program.remote` the URL) and there; the server's record names
this call as its parent, so the two logs make one call tree. It broadcasts
over a column and evaluates as a local program does; its conversations are
kept by its server (`POST <url>/conversations/<id>/turns`).

```julia
team = remote("https://example.org/team"; key = ENV["TEAM_KEY"])
team("I was charged twice for order B-2210.")
team.(tickets.message)
```
"""
function remote(url::AbstractString; key=nothing, timeout::Real=120.0)
    base = rstrip(String(url), '/')
    described = remote_request(base * "/interface", key; timeout)
    get(described, "functai_interface", nothing) == SERVE_FORMAT ||
        throw(RemoteError("remote-format", "$url does not describe a FunctAI program this version reads " *
                                           "(functai_interface $(repr(get(described, "functai_interface", nothing))))", 0, ""))
    iface = described["interface"]
    names = [x["name"] for x in iface["outputs"]]
    code = function (; inputs...)
        current = CURRENT_CALL[]
        got = remote_request(base * "/call", key, LMCC.jobj("inputs" => JObj(String(k) => logvalue(v) for (k, v) in inputs));
                             timeout, parent=current === nothing ? nothing : current.id)
        outputs = something(get(got, "outputs", nothing), JObj())
        length(names) == 1 ? get(outputs, only(names), nothing) : NamedTuple(Symbol(n) => get(outputs, n, nothing) for n in names)
    end
    # the server's interface, checked as its kind's: an AI function's shapes may carry lmcc's keywords
    kind = String(something(get(described, "kind", nothing), "module"))
    check_interface(iface; ai=kind == "ai", what="remote $(described["name"])")
    p = AIProgram(String(described["name"]), code; interface=Dict{String,Any}(iface), module_name="functai.remote",
                  remote=(url=base, key=key, timeout=Float64(timeout), version=String(described["version"]), kind=kind))
    p
end
