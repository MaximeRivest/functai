# Serving a program over HTTP and using it from elsewhere (contract/serving.md),
# on a fake model: a real HTTP server on this machine.

const HTTP = FunctAI.HTTP
const Sockets = FunctAI.Sockets
free_port() = (s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p)

function http_json(method, url; body=nothing, key=nothing, headers=Pair{String,String}[])
    hs = Pair{String,String}["content-type" => "application/json", headers...]
    key === nothing || push!(hs, "authorization" => "Bearer $key")
    resp = HTTP.request(method, url, hs, body === nothing ? UInt8[] : LMCC.json_text(body); status_exception=false)
    (resp.status, isempty(resp.body) ? nothing : (try
        LMCC.parse_json(String(resp.body))
    catch
        String(resp.body)
    end))
end

"The Server-Sent Events of a reply's body, as JSON events."
sse_events(text) = [LMCC.parse_json(line[7:end]) for line in Base.split(text, '\n') if startswith(line, "data: ")]

@ai function team(message::String)::String
    "Which team should answer this customer message?"
end

@testset "a served program: interface, calls, streams, keys, refusals" begin
    r = FakeRouter(; responder=(req, i) -> "<result>\nbilling\n</result>")
    port = free_port()
    base = "http://127.0.0.1:$port"
    calls_dir = mktempdir()
    server = with_settings(router=r, lm="gpt-4.1-mini", log_calls=calls_dir) do
        FunctAI.serve(team; port, keys=["k-1"], block=false)
    end
    try
        @test http_json("GET", "$base/interface")[1] == 401
        status, d = http_json("GET", "$base/interface"; key="k-1")
        @test status == 200 && d["functai_interface"] == 1 && d["name"] == "team" && d["kind"] == "ai"
        @test http_json("GET", "$base/openapi.json"; key="k-1")[2]["openapi"] == "3.1.0"
        @test occursin("<form", String(HTTP.get("$base/").body))
        with_settings(router=r, lm="gpt-4.1-mini", log_calls=calls_dir) do
            status, d = http_json("POST", "$base/call"; key="k-1", body=Dict("inputs" => Dict("message" => "charged twice")))
            @test status == 200 && d["value"] == "billing" && d["outputs"] == Dict("result" => "billing")
            status, d = http_json("POST", "$base/call"; key="k-1", body=Dict("inputs" => Dict("text" => "x")))
            @test status == 422 && d["error"]["code"] == "interface-input"
            resp = HTTP.post("$base/stream", ["authorization" => "Bearer k-1", "content-type" => "application/json"],
                             LMCC.json_text(Dict("inputs" => Dict("message" => "where is it"))))
            events = sse_events(String(resp.body))
            @test events[1]["kind"] == "started" && events[end]["kind"] == "done" && events[end]["value"] == "billing"
            @test all(e -> e["kind"] in ("started", "request", "text", "retry", "done"), events)
            # remote: the served program used like a local one, one call tree across two logs
            far = remote(base; key="k-1")
            @test far("charged twice") == "billing"
            @test FunctAI.program_name(far) == "team"
            @test FunctAI.program_of(far)["kind"] == "remote"
            @test far.(["a", "b"]) == ["billing", "billing"]
            err = try
                far(nothing)
            catch e
                e
            end
            @test err isa InterfaceError
            @test (try remote(base; key="wrong") catch e; e end) isa FunctAI.RemoteError
        end
        recs, _ = FunctAI.read_log(calls_dir)
        mine = [c for c in recs if c["program"]["kind"] == "remote"]
        served = [c for c in recs if c["program"]["kind"] == "ai" && c["parent"] !== nothing]
        @test !isempty(mine) && any(c -> c["parent"] in [m["id"] for m in mine], served)
    finally
        close(server)
    end
    @test (try FunctAI.serve(team; host="0.0.0.0", port=free_port(), block=false) catch e; e end).code == "serve-keys"
    @program function opaque_one(x)::String
        "Opaque."
        "y"
    end
    @test (try FunctAI.Service(opaque_one) catch e; e end).code == "serve-opaque"
end


@testset "a served conversation: turns, events, approvals by the caller" begin
    notes = Dict("todo.md" => "milk")
    "Write a note."
    put_note(name::String, text::String) = (notes[name] = text; "ok")
    pt = tool(put_note; effects=:changes)
    @ai tools = [pt] approve = :changes function keeper(request::String)::String
        "Keep notes."
    end
    r = FakeRouter(Any[(calls=[("c1", "put_note", (name="todo.md", text="oat milk"))],), "<result>\nDone.\n</result>"])
    port = free_port()
    base = "http://127.0.0.1:$port"
    server = with_settings(router=r, lm="gpt-4.1-mini") do
        FunctAI.serve(keeper; port, approvals=:caller, store=mktempdir(), block=false)
    end
    try
        with_settings(router=r, lm="gpt-4.1-mini") do
            status, d = http_json("POST", "$base/conversations/c1/turns"; body=Dict("inputs" => Dict("request" => "oat milk"), "wait" => true))
            @test status == 201 && d["state"] == "waiting" && d["waiting"][1]["name"] == "put_note"
            tid = d["turn"]
            resp = HTTP.get("$base/conversations/c1/turns/$tid/events")
            events = sse_events(String(resp.body))
            @test any(e -> e["kind"] == "approval" && e["to"] == "caller", events)
            @test !any(e -> e["kind"] == "tool_call", events)
            status, d = http_json("POST", "$base/conversations/c1/turns/$tid/approvals/1"; body=Dict("verdict" => "yes"))
            @test status == 202
            ok = timedwait(() -> http_json("GET", "$base/conversations/c1/turns/$tid")[2]["state"] == "done", 30.0)
            @test ok === :ok
            @test notes["todo.md"] == "oat milk"
            @test http_json("GET", "$base/conversations/c1/turns")[2]["turns"][1]["value"] == "Done."
            @test http_json("GET", "$base/conversations/c1/turns/nope")[1] == 404
        end
    finally
        close(server)
    end
end
