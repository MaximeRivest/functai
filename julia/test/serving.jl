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
            rs = stream(far, "charged twice")
            @test join(collect(rs)) == "billing" && fetch(rs) == "billing"
            @test [e.kind for e in eachevent(rs)][[1, end]] == [:started, :done]
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

@testset "baking: the examples every trainer reads, and a baked student served elsewhere" begin
    @ai function reply(message::String, guide::String)::String
        "Answer the customer."
    end
    rows = [(message="Where is my order?", guide="Be kind.", result="On its way."),
            (message="Can I return it?", guide="Be kind.", result="Yes, within 30 days.")]
    table = FunctAI.bake_examples(reply, rows; fixed=Dict("guide" => "Be kind."), validation=0.5)
    @test length(table) == 2 && table[1].messages[end]["role"] == "assistant"
    @test table[1].messages[end]["content"] == "<result>\nOn its way.\n</result>"
    @test !any(m -> occursin("Be kind.", m["content"]), table[1].messages)     # a fixed input is left out
    @test sort([r.split for r in table]) == ["train", "validation"]
    path = FunctAI.export_examples(joinpath(mktempdir(), "reply.jsonl"), reply, rows; fixed=Dict("guide" => "Be kind."))
    @test countlines(path) == 2 && LMCC.parse_json(read(path * ".meta.json", String))["functai_examples"] == 1
    @test (try FunctAI.bake_examples(reply, [(message="a", guide="x", result="b"), (message="c", guide="y", result="d")];
                                     fixed=Dict("guide" => "x")) catch e; e end).code == "bake-rows"
    # a folder baked elsewhere (baked.json format 2), its weights served by an OpenAI-compatible server
    e = FunctAI.bake_entry(reply; fixed=Dict("guide" => "Be kind."), rows=[Dict("message" => r.message, "guide" => r.guide) for r in rows])
    folder = mktempdir()
    write(joinpath(folder, "baked.json"), LMCC.json_text(Dict("functai_baked" => 2, "kind" => "generative", "name" => "reply-student",
        "student" => "Qwen/Qwen3.5-0.8B", "functions" => [FunctAI.entry_meta(e)], "weights" => Dict("form" => "merged", "path" => "model"),
        "template" => Dict("sha256" => "x", "kwargs" => Dict("enable_thinking" => false), "end" => "", "stop_token_ids" => []),
        "generation" => Dict("max_new_tokens" => 64, "max_model_len" => 512), "hashes" => Dict(), "sizes" => Dict())))
    seen = Ref{Any}(nothing)
    port = free_port()
    server = HTTP.serve!("127.0.0.1", port; verbose=-1) do req
        body = LMCC.parse_json(String(req.body))
        seen[] = body
        HTTP.Response(200, LMCC.json_text(Dict("model" => body["model"], "choices" => [Dict("message" => Dict("role" => "assistant",
            "content" => "<result>\nOn its way.\n</result>"), "finish_reason" => "stop")], "usage" => Dict("prompt_tokens" => 20, "completion_tokens" => 6))))
    end
    try
        student = FunctAI.baked(folder; url="http://127.0.0.1:$port/v1")
        fast = configure(reply; lm=student)
        @test fast("Where is my order?", "Be kind.") == "On its way."
        msgs, _ = FunctAI.student_messages(e, Dict("message" => "Where is my order?"))
        @test same(seen[]["messages"], msgs)                  # called with the messages it was trained on
        @test seen[]["chat_template_kwargs"]["enable_thinking"] == false
        @test (try fast("Where is my order?", "Be rude.") catch e; e end).code == "baked-fixed"
        changed = AIFunction("reply", "Answer the customer."; inputs=(message=String, guide=String), output=Int)
        @test (try configure(changed; lm=student)("x", "Be kind.") catch e; e end).code == "baked-changed"
    finally
        close(server)
    end
end
