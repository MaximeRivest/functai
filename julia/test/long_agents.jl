# Long agents and pipelines, setups and declared notes, on a fake model:
# tools that answer with pictures, the step hook and its built-in plugins
# (limits, keep_pictures), the call log in a store of one's own, results that
# come later, notes from code (declared and checked), the answer so far,
# what a program runs on (setup), what it may reach, and what code spent
# (contract/tools.md, plugins.md, calls.md, streaming.md, programs.md).

const LA_LM = "gpt-4.1-mini"
const LA_PNG = FunctAI.Base64.base64encode(UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x66, 0x61, 0x6b, 0x65])
la_xml(v) = "<result>\n$v\n</result>"
la_results(req) = [p for m in req.messages for p in m.parts if p isa LM15.ToolResultPart]
la_pictures(req) = sum((count(c -> c isa LM15.ImagePart, p.content) for p in la_results(req)); init=0)
la_texts(part) = [c.text for c in part.content if c isa LM15.TextPart]

"A model that asks for these tool calls, one per reply while it is offered tools, then answers."
function la_script(steps...; answer=req -> "done")
    (req, i) -> begin
        n = length(la_results(req))
        if n < length(steps) && !isempty(req.tools)
            name, args = steps[n+1]
            (calls=[("c$(n + 1)", name, args)],)
        else
            la_xml(answer(req))
        end
    end
end
la_router(steps...; kw...) = FakeRouter(Any[]; responder=la_script(steps...; kw...))
la_err(f) = try
    f()
    nothing
catch e
    e
end

"A screenshot of a page, with its title."
la_look(page::String) = ["page $page", LM15.image(; data=LA_PNG, media_type="image/png")]
const LA_LOOK = tool(la_look; name="look", effects=:reads)
@ai tools = [LA_LOOK] function la_browse(task::String)::String
    "Look at pages to do the task."
end

# ------------------------------------------------------------------ pictures from tools

@testset "a tool answers with a picture" begin
    r = la_router(("look", (page="home",)))
    with_settings(router=r, lm=LA_LM) do
        s = stream(la_browse, "find the price")
        events = collect(eachevent(s))
        @test fetch(s) == "done"
        part = only(la_results(r.requests[end]))
        @test [typeof(c) for c in part.content] == [LM15.TextPart, LM15.ImagePart]   # the model sees the picture itself
        @test part.content[2].data == LA_PNG
        res = only(filter(e -> e.kind === :tool_result, events))
        @test res.output == "page home"                                # the text, for readers that show text
        @test same(res.parts, [Dict("type" => "text", "text" => "page home"),
                               Dict("type" => "image", "media_type" => "image/png", "data" => LA_PNG)])
        keep = FunctAI.Keep(FunctAI.OrderedDict("task" => false), FunctAI.OrderedDict("result" => true))
        kept = FunctAI.kept_event(res, keep, Dict("kind" => "ai", "answer" => "result"))
        @test !haskey(kept, "parts") && !haskey(kept, "output")
    end
end

@testset "a picture from a file is given its bytes; a value is its JSON, as before" begin
    file = joinpath(mktempdir(), "shot.png")
    write(file, UInt8[0x89, 0x50, 0x4e, 0x47, 0x66, 0x69, 0x6c, 0x65])
    shot = tool(input -> LM15.image(; path=file); name="screenshot", parameters=Dict("type" => "object"), effects=:reads)
    facts = tool(input -> FunctAI.OrderedDict("a" => 1, "list" => ["x"]); name="facts", parameters=Dict("type" => "object"), effects=:reads)
    watch = AIFunction("watch", "Watch."; inputs=(task=String,), output=String, tools=[shot, facts])
    r = la_router(("screenshot", (;)), ("facts", (;)))
    with_settings(router=r, lm=LA_LM) do
        s = stream(watch, "look")
        events = collect(eachevent(s))
        @test fetch(s) == "done"
        first_result, second_result = la_results(r.requests[end])
        @test first_result.content[1].data == FunctAI.Base64.base64encode(read(file)) && first_result.content[1].path === nothing
        res = filter(e -> e.kind === :tool_result, events)
        @test res[2].output == "{\"a\":1,\"list\":[\"x\"]}" && !haskey(res[2], "parts")
    end
end

@testset "a plugin sees a picture result and changes it" begin
    seen = Any[]
    p = Plugin("caption"; tool_result=t -> (push!(seen, t.output); Change(output=[t.output..., "(the price is at the top right)"])))
    r = la_router(("look", (page="home",)))
    @test with_settings(() -> configure(la_browse; plugins=[p])("find"); router=r, lm=LA_LM) == "done"
    @test seen[1][2]["type"] == "image"
    @test [typeof(c) for c in only(la_results(r.requests[end])).content] == [LM15.TextPart, LM15.ImagePart, LM15.TextPart]
end

la_clicked = String[]
"Click, and see the screen."
la_click(where::String) = (push!(la_clicked, where); ["clicked", LM15.image(; data=LA_PNG, media_type="image/png")])
@ai tools = [tool(la_click; name="click", effects=:changes)] function la_agent(task::String)::String
    "Do it."
end

@testset "a turn keeps a picture result, and goes on with it" begin
    empty!(la_clicked)
    store = MemoryConversations()
    with_settings(router=la_router(("click", (where="buy",))), lm=LA_LM) do
        chat = conversation(la_agent, "pictures"; store, approve=["click"])
        @test la_err(() -> chat("buy it")) isa Waiting
        @test approve!(last(turns(chat))) == "done"
        @test la_clicked == ["buy"]
        done = [x for x in FunctAI.read_records(store, "pictures", 0) if get(x, "kind", "") == "tool" && get(x, "state", "") == "done"]
        @test same(only(done)["output"][2], Dict("type" => "image", "media_type" => "image/png", "data" => LA_PNG))
    end
end

# ------------------------------------------------------------------ the step hook

@testset "the step hook sees each step, and wraps up" begin
    seen = Any[]
    p = Plugin("wrap"; step=s -> begin
        push!(seen, (s.step, s.max_steps, [x.invocation for x in s.results], copy(s.tools), s.spent.requests))
        s.step == 3 ? Change(tools=String[], reminder="Answer now.") : nothing
    end)
    r = la_router((("look", (page=string(i),)) for i in 0:9)...)
    @test with_settings(() -> configure(la_browse; plugins=[p], max_steps=5)("find"); router=r, lm=LA_LM) == "done"
    @test seen == [(2, 5, [1], ["look"], 1), (3, 5, [1, 2], ["look"], 2)]
    @test isempty(r.requests[end].tools)                              # no tool offered: the model must answer
    @test last(la_texts(la_results(r.requests[end])[end])) == "Answer now."   # the reminder follows the last result
end

@testset "earlier results shown otherwise: the record says so, and the call stays replayable" begin
    p = Plugin("forget"; version="2.0.0", step=s -> begin
        old = [x for x in s.results[1:end-1] if FunctAI.pictures(x) > 0]
        isempty(old) ? nothing : Change(replace=Dict(x.invocation => "(an earlier screenshot)" for x in old))
    end)
    r = la_router((("look", (page=string(i),)) for i in 0:2)...)
    dir = mktempdir()
    @test with_settings(() -> configure(la_browse; plugins=[p])("find"); router=r, lm=LA_LM, log_calls=dir) == "done"
    @test [la_pictures(q) for q in r.requests] == [0, 1, 1, 1]       # only the newest picture is sent
    rec = only(first(FunctAI.read_log(dir)))
    changes = [c for c in rec["changes"] if c["hook"] == "step"]
    @test same(changes[1], Dict("plugin" => "forget", "version" => "2.0.0", "hook" => "step",
                                "change" => Dict("replace" => Dict("1" => "(an earlier screenshot)"))))
    @test all(ex -> haskey(ex, "request_hash"), rec["exchanges"])    # nothing was rewritten
end

@testset "a step handler stops the call; a change step cannot make is refused" begin
    p = Plugin("brake"; step=s -> s.step >= 3 ? Change(stop="too many pages") : nothing)
    err = la_err(() -> with_settings(() -> configure(la_browse; plugins=[p])("find"); router=la_router((("look", (page=string(i),)) for i in 0:9)...), lm=LA_LM))
    @test err isa PluginError && err.code == "plugin-stopped" && occursin("too many pages", sprint(showerror, err))
    bad = Plugin("bad"; step=s -> Change(tools=["fly"]))
    err = la_err(() -> with_settings(() -> configure(la_browse; plugins=[bad])("find"); router=la_router(("look", (page="a",)), ("look", (page="b",))), lm=LA_LM))
    @test err isa PluginError && err.code == "plugin-change"
    worse = Plugin("worse"; step=s -> Change(sections=["no"]))
    err = la_err(() -> with_settings(() -> configure(la_browse; plugins=[worse])("find"); router=la_router(("look", (page="a",))), lm=LA_LM))
    @test err isa PluginError && err.code == "plugin-change"
end

# ------------------------------------------------------------------ limits and keep_pictures

@testset "limits: the last step answers instead of failing; a token limit wraps up, then stops the next call" begin
    r = la_router((("look", (page=string(i),)) for i in 0:9)...)
    @test with_settings(() -> configure(la_browse; plugins=[limits(seconds=3600)], max_steps=3)("find"); router=r, lm=LA_LM) == "done"
    @test length(r.requests) == 3 && isempty(r.requests[end].tools)
    @test last(la_texts(la_results(r.requests[end])[end])) == FunctAI.LAST_STEP
    r = la_router((("look", (page=string(i),)) for i in 0:9)...)
    limit = limits(tokens=1)                                          # reached after the first request
    @test with_settings(() -> configure(la_browse; plugins=[limit])("find"); router=r, lm=LA_LM) == "done"
    @test length(r.requests) == 2 && isempty(r.requests[end].tools)
    @test startswith(last(la_texts(la_results(r.requests[end])[end])), "This task has reached its limit (1 tokens)")
    twice = AIProgram("twice", (; task) -> la_browse(task) * la_browse(task); interface=Dict(
        "description" => "", "inputs" => [Dict("name" => "task", "shape" => Dict("type" => "string"))],
        "outputs" => [Dict("name" => "result", "shape" => Dict("type" => "string"))]))
    err = la_err(() -> with_settings(() -> twice("find"); router=la_router((("look", (page=string(i),)) for i in 0:9)...), lm=LA_LM, plugins=[limit]))
    @test err isa PluginError && err.code == "plugin-stopped" && occursin("limit reached: 1 tokens", sprint(showerror, err))
end

@testset "a cost limit needs prices, and counts a model with no price as reached" begin
    @test la_err(() -> limits(cost=1.0)) isa ArgumentError
    @test la_err(() -> limits()) isa ArgumentError
    limit = limits(cost=1.0, prices=Dict("some-other-model" => (input_per_million=1, output_per_million=2)))
    err = la_err(() -> with_settings(() -> configure(la_browse; plugins=[limit])("find"); router=la_router(("look", (page="a",))), lm=LA_LM))
    @test err isa PluginError && occursin("no price is known for gpt-4.1-mini", sprint(showerror, err))   # refused up front
    flat = (input_per_million=1, output_per_million=2, cache_read_per_million=1, cache_write_per_million=1)
    priced = limits(cost=1000.0, prices=Dict(LA_LM => flat), then=:stop)
    @test with_settings(() -> configure(la_browse; plugins=[priced])("find"); router=la_router(("look", (page="a",))), lm=LA_LM) == "done"
    s = Spent(Any[("m", Dict("input_tokens" => 1_000_000, "output_tokens" => 1_000_000, "total_tokens" => 2_000_000,
                             "cache_read_tokens" => 0, "cache_write_tokens" => 0))], 1.0)
    @test FunctAI.cost(s, Dict("m" => (input_per_million=3, output_per_million=15))) == 18.0
    @test FunctAI.cost(s, Dict("other" => (input_per_million=3, output_per_million=15))) === nothing
    @test s.tokens == 2_000_000 && Spent(Any[], 0.0).tokens == 0
end

@testset "keep_pictures shows only the newest, dropping them in batches when asked" begin
    r = la_router((("look", (page=string(i),)) for i in 0:4)...)
    @test with_settings(() -> configure(la_browse; plugins=[keep_pictures(2)])("find"); router=r, lm=LA_LM) == "done"
    @test [la_pictures(q) for q in r.requests] == [0, 1, 2, 2, 2, 2]
    @test la_texts(la_results(r.requests[end])[1]) == ["page 0", FunctAI.PICTURE_GONE]
    r = la_router((("look", (page=string(i),)) for i in 0:6)...)
    @test with_settings(() -> configure(la_browse; plugins=[keep_pictures(2; at_once=3)], max_steps=10)("find"); router=r, lm=LA_LM) == "done"
    @test [la_pictures(q) for q in r.requests] == [0, 1, 2, 3, 4, 2, 3, 4]
end

# ------------------------------------------------------------------ the call log in a store of one's own

"A store of one's own, as a database keeps rows: JSON text."
struct LaListStore
    lines::Vector{String}
end
Base.push!(s::LaListStore, record) = (push!(s.lines, LMCC.json_text(record)); s)
FunctAI.records(s::LaListStore) = Any[LMCC.parse_json(x) for x in s.lines]
struct LaBrokenStore end
Base.push!(::LaBrokenStore, record) = error("database away")
FunctAI.records(::LaBrokenStore) = Any[]

@ai function la_team(message::String)::String
    "Which team answers this message."
end

@testset "calls go to a store of one's own, and are read back, rated" begin
    store = LaListStore(String[])
    with_settings(router=FakeRouter(Any[la_xml("billing")]), lm=LA_LM, log_calls=store) do
        p = predict(la_team, "I was charged twice.")
        rate(p, :right)
    end
    recs = FunctAI.records(store)
    @test [(haskey(x, "functai_call"), haskey(x, "functai_rating")) for x in recs] == [(true, false), (false, true)]
    @test length(rated(la_team; folder=store)) == 1 && length(calls(la_team; folder=store)) == 1
    @test la_err(() -> prune_calls(; folder=store)) isa ArgumentError
    # a store that fails never stops a call
    @test (@test_logs (:warn, r"could not log") match_mode=:any with_settings(() -> la_team("x"); router=FakeRouter(Any[la_xml("billing")]), lm=LA_LM, log_calls=LaBrokenStore())) == "billing"
    @test la_err(() -> configure!(log_calls=1)) isa ArgumentError
    mem = MemoryCalls()
    with_settings(router=FakeRouter(Any[la_xml("a"), la_xml("b")]), lm=LA_LM, log_calls=mem) do
        la_team("1")
        la_team("2")
    end
    @test [x["outputs"]["result"] for x in FunctAI.records(mem)] == ["a", "b"]
end

# ------------------------------------------------------------------ results that come later

const la_jobs = String[]
"Visit a site in a cloud browser; its report comes back later."
la_start_harvest(url::String) = (push!(la_jobs, url); later("harvest:$(length(la_jobs))"))
"Today's weather."
la_weather(city::String) = "sunny in $city"
@ai tools = [tool(la_start_harvest; name="start_harvest", effects=:changes), tool(la_weather; name="weather", effects=:reads)] function la_researcher(task::String)::String
    "Research the site."
end

@testset "a turn waits for a result that comes later, and goes on when it is delivered" begin
    empty!(la_jobs)
    store = MemoryConversations()
    r = la_router(("start_harvest", (url="a.com",)), ("weather", (city="Oslo",)); answer=req -> "report: " * join(la_texts(la_results(req)[1])))
    with_settings(router=r, lm=LA_LM) do
        chat = conversation(la_researcher, "later-1"; store)
        err = la_err(() -> chat("look at a.com"))
        @test err isa Waiting && [x.key for x in err.later] == ["harvest:1"] && isempty(err.approvals)
        @test last(turns(chat)).state == "waiting" && la_jobs == ["a.com"]
        @test [x.key for x in last(turns(chat)).later] == ["harvest:1"]
        asked = length(r.requests)
        # the webhook, in another process: open the conversation, deliver
        again = conversation(la_researcher, "later-1"; store)
        @test deliver!(again, "harvest:1", ["3 pages", LM15.image(; data=LA_PNG, media_type="image/png")]) == "report: 3 pages"
        @test la_jobs == ["a.com"]                                    # the tool did not run again
        @test length(r.requests) - asked == 2                         # no model answer paid twice: weather, then the answer
        @test deliver!(again, "harvest:1", "late duplicate") === nothing   # a webhook sent twice changes nothing
        kinds = [(x["kind"], get(x, "state", nothing)) for x in FunctAI.read_records(store, "later-1", 0) if x["kind"] in ("tool", "delivered")]
        @test kinds == [("tool", "started"), ("tool", "later"), ("delivered", nothing), ("tool", "done"), ("tool", "done")]
        events = FunctAI.eachevent(last(turns(again)))
        @test [(e.key, e.name, e.invocation) for e in events if e.kind === :later] == [("harvest:1", "start_harvest", 1)]
    end
end

@testset "a result delivered before the turn waits is used at once; outside a turn, refused" begin
    empty!(la_jobs)
    store = MemoryConversations()
    with_settings(router=la_router(("start_harvest", (url="b.com",)); answer=req -> "report: " * join(la_texts(la_results(req)[end]))), lm=LA_LM) do
        chat = conversation(la_researcher, "later-2"; store)
        @test deliver!(chat, "harvest:1", "fast"; resume=false) === nothing   # the webhook came first
        @test chat("look") == "report: fast"
    end
    err = la_err(() -> with_settings(() -> la_researcher("look"); router=la_router(("start_harvest", (url="c.com",))), lm=LA_LM))
    @test err isa ConversationError && err.code == "later-unavailable" && la_jobs == ["b.com", "c.com"]
    store = MemoryConversations()
    with_settings(router=la_router(("start_harvest", (url="d.com",))), lm=LA_LM) do
        chat = conversation(la_researcher, "later-3"; store)
        @test la_err(() -> chat("look")) isa Waiting
        err = la_err(() -> resume!(last(turns(chat))))
        @test err isa ConversationError && err.code == "turn-state" && occursin("harvest:3", err.msg)
    end
end

# ------------------------------------------------------------------ notes

@ai function la_write(x::String)::String
    "Write."
end

const LA_PAGE = @NamedTuple{url::String, title::String}
@program notes = ["stage" => Note(OneOf(:walking, :writing); latest=true, outside=true), "page captured" => LA_PAGE,
                  "cost so far" => Float64] function la_walk(url::String)::String
    "Walk a site."
    note("stage", "walking")
    note("page captured", (url=url, title="Home"))
    note("cost so far", 0.02)
    note("stage", "writing")
    la_write(url)
end
@program function la_pipeline(url::String)::String
    note("screen captured", Dict("screen" => 4))
    la_write(url)
end

@testset "code says what happened; declared notes are checked where they are made" begin
    with_settings(router=FakeRouter(Any[la_xml("ok"), la_xml("ok"), la_xml("ok")]), lm=LA_LM) do
        s = stream(la_pipeline, "a.com")
        @test fetch(s) == "ok"
        n = only(filter(e -> e.kind === :note, collect(eachevent(s))))
        @test (n.name, n.data, n.function) == ("screen captured", Dict("screen" => 4), "la_pipeline")
        @test note("nothing watches") === nothing                    # outside any call: nothing happens
        @test la_err(() -> note(" ")) isa ArgumentError
        # declared: in the interface (not its signature), shown outside when declared so
        iface = FunctAI.interface(la_walk)
        @test [n["name"] for n in iface["notes"]] == ["stage", "page captured", "cost so far"]
        @test iface["notes"][1]["latest"] === true && !haskey(iface["notes"][2], "outside")
        s = stream(la_walk, "a.com")
        @test fetch(s) == "ok"
        events = collect(eachevent(s))
        outside = FunctAI.outside(FunctAI.event_json.(events); notes=FunctAI.outside_notes(la_walk))
        @test [(e.kind, e.name) for e in outside if e.kind === :note] == [(:note, "stage"), (:note, "stage")]
        state = FunctAI.replay(FunctAI.event_json.(events))
        @test [n["name"] for n in state.calls[events[1].call].notes] == ["stage", "page captured", "cost so far", "stage"]
    end
    # refused where made: a name not declared, data that does not fit
    @program notes = ["stage" => OneOf(:walking, :writing)] function la_strict(x::String)::String
        note(x, "walk")
        "done"
    end
    err = la_err(() -> la_strict("stage"))
    @test err isa InterfaceError && err.code == "interface-note" && err.field == "stage"
    err = la_err(() -> la_strict("stage changed"))
    @test err isa InterfaceError && err.code == "interface-note" && err.field == "stage changed"
    # an AI function's notes are its tools': two tools that declare one name differently refuse it
    t1 = tool(input -> ""; name="a", parameters=Dict("type" => "object"), notes=["page" => String])
    t2 = tool(input -> ""; name="b", parameters=Dict("type" => "object"), notes=["page" => Int])
    @test FunctAI.interface(AIFunction("one", "One."; inputs=(x=String,), output=String, tools=[t1]))["notes"][1]["name"] == "page"
    err = la_err(() -> AIFunction("two", "Two."; inputs=(x=String,), output=String, tools=[t1, t2]))
    @test err isa InterfaceError && err.code == "interface-malformed" && err.field == "page"
    @test la_err(() -> tool(input -> ""; name="c", parameters=Dict("type" => "object"), notes=Dict("x" => String))) isa ArgumentError
end

@enum LaSeverity low medium high
@ai function la_insights(page::String)::@NamedTuple{summary::String, severity::LaSeverity}
    "Read the page."
end

@testset "the answer so far is read by its shape" begin
    r = FakeRouter(Any[la_xml("{\"summary\": \"Checkout hides fees\", \"severity\": \"medium\"}")]; piece=7)
    with_settings(router=r, lm=LA_LM) do
        s = stream(la_insights, "a page")
        @test fetch(s).severity == medium
        @test answer_so_far(s).severity == medium                     # done: the answer itself
    end
    shape = last(FunctAI.interface(la_insights)["outputs"])["shape"]
    @test same(answer_so_far("{\"summary\": \"Check", shape), Dict("summary" => "Check"))
    @test same(answer_so_far("{\"summary\": \"x\", \"severity\": \"med", shape), Dict("summary" => "x"))
    @test answer_so_far("Sure!", shape) === nothing
end

# ------------------------------------------------------------------ setups (calls.md, Setups)

"Where is the order?"
la_track_shipped(order::String) = "shipped"
"Where is the order?"
la_track_lost(order::String) = "LOST"

@testset "a setup: what a program runs on, beside what it is" begin
    a = AIFunction("track", "Track."; inputs=(order=String,), output=String, tools=[tool(la_track_shipped; name="track_order")])
    b = AIFunction("track", "Track."; inputs=(order=String,), output=String, tools=[tool(la_track_lost; name="track_order")])
    with_settings(lm=LA_LM) do
        @test version(a) == version(b) && setup(a) != setup(b)         # rewriting a tool: another setup, the same version
        @test setup_document(a)["tools"]["track_order"]["code"]["language"] == "julia"
        @test setup(a) != with_settings(() -> setup(a); lm="claude-haiku-4-5")      # another model
        # what identifies a tool beyond its text
        id1 = AIFunction("track", "Track."; inputs=(order=String,), output=String, tools=[tool(la_track_shipped; name="track_order", code_id="v1")])
        id2 = AIFunction("track", "Track."; inputs=(order=String,), output=String, tools=[tool(la_track_shipped; name="track_order", code_id="v2")])
        @test setup(id1) != setup(id2) && version(id1) == version(id2)
        # an AI function used as a tool counts by its setup
        outer = AIFunction("outer", "Outer."; inputs=(q=String,), output=String, tools=[tool(la_write)])
        @test same(setup_document(outer)["tools"]["la_write"], Dict("setup" => setup(la_write)))
        # plugins count with their settings
        @test setup(configure(a; plugins=[keep_pictures(2)])) != setup(configure(a; plugins=[keep_pictures(3)]))
        # a program names each step it reaches
        doc = setup_document(la_walk)
        @test collect(keys(doc["ai"])) == ["__main__:la_write"] && doc["reach"] == "open"
        # code a version cannot read, named by the host: in every setup worked out under it, a program's too
        @test !haskey(doc, "code_id")
        @test with_settings(() -> setup_document(la_walk)["code_id"]; code_id="sha256:aaaa") == "sha256:aaaa"
        @test with_settings(() -> setup_document(a)["code_id"]; code_id="sha256:aaaa") == "sha256:aaaa"
        @test setup(la_walk) != with_settings(() -> setup(la_walk); code_id="sha256:aaaa")
        @test with_settings(() -> setup(a); code_id="a") != with_settings(() -> setup(a); code_id="b")
        @test version(a) == with_settings(() -> version(a); code_id="a")
    end
    for bad in ("", "   ", "x"^201, 7)
        @test_throws ArgumentError with_settings(() -> 1; code_id=bad)
    end
end

@testset "reach: open, or behind a wall; a wall that lets through less refuses before any request" begin
    fetcher = tool(input -> "page"; name="fetch_page", parameters=Dict("type" => "object"),
                   needs=(domains=["API.Browser-Use.com"], secrets=["BROWSER_USE_API_KEY"]))
    f = AIFunction("walker", "Walk."; inputs=(url=String,), output=String, tools=[fetcher])
    with_settings(lm=LA_LM) do
        @test setup_document(f)["reach"] == "open"
        @test same(with_settings(() -> setup_document(f)["reach"]; wall="srt"),
                   Dict("domains" => ["api.browser-use.com"], "secrets" => ["BROWSER_USE_API_KEY"]))
    end
    r = FakeRouter(Any[la_xml("x")])
    dir = mktempdir()
    err = la_err(() -> with_settings(() -> f("a.com"); router=r, lm=LA_LM, log_calls=dir,
                                     wall=(name="bubblewrap", domains=["*.example.com"], secrets=String[])))
    @test err isa ReachError && err.code == "needs-unmet" && same(err.missing["domains"], ["api.browser-use.com"])
    @test isempty(r.requests)
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["error"]["code"] == "needs-unmet" && rec["placement"]["wall"] == "bubblewrap"
    ok = with_settings(() -> f("a.com"); router=FakeRouter(Any[la_xml("x")]), lm=LA_LM, log_calls=dir,
                       wall=(name="bubblewrap", domains=["*.browser-use.com"], secrets=["BROWSER_USE_API_KEY"]))
    @test ok == "x"
end

"Visit with a paid browser."
la_paid_visit(url::String) = (spend(0.05); "visited $url")
@ai tools = [tool(la_paid_visit; name="visit", effects=:reads)] function la_visitor(url::String)::String
    "Visit the site."
end

@testset "a record says its setup, where it ran, and what code spent; limits count it" begin
    dir = mktempdir()
    with_settings(router=la_router(("visit", (url="a.com",))), lm=LA_LM, log_calls=dir, base_url="https://gateway.example/v1") do
        @test la_visitor("a.com") == "done"
    end
    rec = only(first(FunctAI.read_log(dir)))
    @test rec["program"]["setup"] == with_settings(() -> setup(la_visitor); lm=LA_LM)
    @test same(rec["placement"], Dict("base_url" => "https://gateway.example/v1"))
    @test same(rec["spent"], Dict("usd" => 0.05))
    @test FunctAI.schema_fault("call", rec) === nothing
    flat = (input_per_million=0, output_per_million=0, cache_read_per_million=0, cache_write_per_million=0)
    twice = AIProgram("visit_twice", (; url) -> la_visitor(url) * la_visitor(url); interface=Dict(
        "description" => "", "inputs" => [Dict("name" => "url", "shape" => Dict("type" => "string"))],
        "outputs" => [Dict("name" => "result", "shape" => Dict("type" => "string"))]))
    err = la_err(() -> with_settings(() -> twice("a.com"); router=la_router(("visit", (url="a.com",)), ("visit", (url="a.com",))), lm=LA_LM,
                                     plugins=[limits(cost=0.04, prices=Dict(LA_LM => flat))]))
    @test err isa PluginError && err.code == "plugin-stopped" && occursin("cost 0.04", sprint(showerror, err))
end

@testset "a saved folder keeps each setup, and two are compared" begin
    a, b = mktempdir(), mktempdir()
    with_settings(lm=LA_LM) do
        FunctAI.save(a, la_walk)
    end
    with_settings(lm="claude-haiku-4-5") do
        FunctAI.save(b, la_walk)
    end
    m = LMCC.parse_json(read(joinpath(a, "functai.json"), String))
    @test FunctAI.schema_fault("saved", m) === nothing
    node = m["nodes"]["__main__:la_walk"]
    @test node["kind"] == "module" && node["version"] == version(la_walk) && haskey(node, "setup")
    @test haskey(m["nodes"]["__main__:la_write"], "setup")
    diff = setup_diff(a, b)
    @test [(d.program, d.field, d.before, d.after) for d in diff] == [("__main__:la_write", "model.model", "gpt-4.1-mini", "claude-haiku-4-5"),
                                                                      ("__main__:la_write", "model.provider", "openai", "anthropic")]
    # an agent is saved with its tools as data; loading it here refuses, as in every language: a tool is code
    c = mktempdir()
    with_settings(() -> FunctAI.save(c, la_browse); lm=LA_LM)
    m = LMCC.parse_json(read(joinpath(c, "functai.json"), String))
    t = only(m["nodes"]["__main__:la_browse"]["ai"]["tools"])
    @test t["tool"]["name"] == "look" && t["effects"] == "reads" && startswith(t["code"]["code"], "sha256:")
    err = la_err(() -> FunctAI.load(c))
    @test err isa LoadRefused && err.code == "saved-tools"
end
