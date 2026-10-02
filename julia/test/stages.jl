# The contract's cases for stages 1.2 to 5 and plugins (contract/cases/replies,
# conversations, tools, views, context, plugins), run through the Julia
# implementation: the same cases Python passes.

at_time(text) = FunctAI.unix_of(text)
stage_router() = FakeRouter(Any[]; provider="openai")

@testset "replies case $name" for (name, c) in cases("replies")
    request = LM15.from_dict(LM15.Request, c["request"])
    @test FunctAI.reply_key(request, c["replicate"]) == c["expect"]["key"]
end

@testset "conversations case $name" for (name, c) in cases("conversations")
    now = at_time(c["now"])
    old_clock = FunctAI.CONVERSATION_CLOCK[]
    FunctAI.CONVERSATION_CLOCK[] = () -> now
    try
        if c["kind"] == "state"
            log = FunctAI.apply!(FunctAI.ConvLog(), c["records"])
            e = c["expect"]
            @test Dict(t => FunctAI.turn_state(st) for (t, st) in log.turns) == e["states"]
            @test log.head == e["head"]
            head = log.head === nothing ? nothing : log.turns[log.head]
            state = head === nothing ? nothing : FunctAI.turn_state(head)
            nxt = e["next"]
            if haskey(nxt, "refuses")
                @test state == "waiting"
            elseif haskey(nxt, "waits")
                @test state == "running" && log.head == nxt["waits"]
            else
                @test FunctAI.done_on(log, log.head) == nxt["parent"]
            end
            waiting = Dict(t => Any[a["invocation"] for a in FunctAI.unanswered(st)] for (t, st) in log.turns if !isempty(FunctAI.unanswered(st)))
            @test waiting == e["waiting"]
            unfinished = Dict(t => Any[x["invocation"] for x in FunctAI.unfinished(st)] for (t, st) in log.turns if !isempty(FunctAI.unfinished(st)))
            @test unfinished == e["unfinished"]
        else
            tutor = AIFunction("tutor", "Tutor."; inputs=(message=String,), output=String)
            store = MemoryConversations()
            FunctAI.append_records!(store, "c", [Dict{String,Any}(k => v for (k, v) in r if k != "seq") for r in c["records"]])
            rule = c["rule"]
            with_settings(router=stage_router(), lm="gpt-4.1-mini") do
                chat = conversation(tutor, "c"; store, context=FunctAI.ContextRule(rule["last"], String[something(get(rule, "without", nothing), Any[])...]))
                log = FunctAI.read_log!(chat)
                for d in values(log.programs)       # the case's program is this one
                    d["signature"] = signature_id(tutor)
                end
                ctx = FunctAI.turn_context(chat, log, c["parent"])
                entries = Any[]
                for (t, id) in zip(ctx[:turns], ctx[:ids])
                    e = Dict{String,Any}("call" => id)
                    isempty(something(get(t, "steps", nothing), Any[])) || (e["steps"] = true)
                    push!(entries, e)
                end
                @test same(ctx[:finish](entries), c["expect"]["saw"])
                @test same(ctx[:rows], c["expect"]["rows"])
            end
        end
    finally
        FunctAI.CONVERSATION_CLOCK[] = old_clock
    end
end

@testset "tools case $name" for (name, c) in cases("tools")
    if c["kind"] == "asks"
        rule = c["rule"]
        rule = rule == "function" ? (a -> true) : rule isa AbstractVector ? String[rule...] : rule
        got = [FunctAI.asks(rule, FunctAI.Approval("c", 1, "t", a["name"], Dict(), a["effects"], a["path"], "")) for a in c["approvals"]]
        @test got == c["expect"]["asks"]
    else
        @test FunctAI.denial(c["reason"]) == c["expect"]["output"]
    end
end

@testset "views case $name" for (name, c) in cases("views")
    got = FunctAI.outside(c["events"]; answer_from=c["answer_from"])
    @test same(Any[FunctAI.event_json(e) for e in got], c["expect"]["events"])
end

@testset "context case $name" for (name, c) in cases("context")
    e = c["expect"]
    if haskey(e, "refuses")
        err = refusal_of(() -> FunctAI.earlier_of(c["call"], c["records"]))
        @test err isa FunctAI.SawUnknown && err.code == e["refuses"]
    else
        got = FunctAI.earlier_of(c["call"], c["records"])
        @test same(Dict("earlier" => got["earlier"], "conversation" => got["conversation"]), e)
    end
end

@testset "plugins case $name" for (name, c) in cases("plugins")
    if c["kind"] == "order"
        made = Dict{String,Plugin}()
        layers = Any[]
        for layer in c["layers"]
            ps = [get!(() -> Plugin(n), made, n) for n in layer["plugins"]]
            s = Dict{Symbol,Any}(:plugins => ps)
            layer["where"] == "configure" && c["program_plugins"] === false && (s[:program_plugins] = false)
            push!(layers, (Symbol(layer["where"]), s))
        end
        @test [p.name for p in FunctAI.plugins_in_order(layers)] == c["expect"]["order"]
        continue
    end
    hook, start, expect = Symbol(c["hook"]), c["start"], c["expect"]
    ran = Ref(0)
    plugins = Plugin[]
    for (i, ch) in enumerate(c["changes"])
        change = ch === nothing ? nothing : Change(; (Symbol(k) => v for (k, v) in ch)...)
        push!(plugins, Plugin("p$i"; Dict(hook => (_ -> (ran[] += 1; change)))...))
    end
    read_tool = FunctAI.tool(x -> "r"; name="read", description="Read.", parameters=Dict("type" => "object"))
    write_tool = FunctAI.tool(x -> "w"; name="write", description="Write.", parameters=Dict("type" => "object"))
    f = AIFunction("f", "Answer."; inputs=(x=String,), output=String, tools=[read_tool, write_tool])
    with_settings(plugins=plugins, router=stage_router(), lm="gpt-4.1-mini") do
        if hook === :before_call
            s = FunctAI.effective(f.own)
            s[:lm] = start["lm"]
            shaped = FunctAI.shape_call(f, Dict("x" => "1"), s, nothing, start["instruction"])
            @test shaped.sections == expect["sections"]
            @test shaped.settings[:lm] == expect["lm"]
            @test something(shaped.instruction, start["instruction"]) == expect["instruction"]
            @test shaped.tools == expect["tools"]
            for (k, v) in expect["settings"]
                @test get(shaped.settings, Symbol(k), nothing) == v
            end
        elseif hook === :context
            store = MemoryConversations()
            recs = Any[merge(FunctAI.program_record(f), Dict("version" => "v"))]
            parent = nothing
            for t in start["keep"]
                push!(recs, Dict("functai_conversation" => 1, "kind" => "turn", "at" => "2026-09-30T10:00:00.000000Z", "turn" => t,
                                 "parent" => parent, "program" => "v", "inputs" => Dict("x" => t)))
                push!(recs, Dict("functai_conversation" => 1, "kind" => "ended", "at" => "2026-09-30T10:00:00.000000Z", "turn" => t,
                                 "state" => "done", "outputs" => Dict("result" => "r", "photo" => "P", "notes" => "N")))
                parent = t
            end
            FunctAI.append_records!(store, "c", recs)
            chat = conversation(f, "c"; store)
            picked, without, sections, _, _ = FunctAI.shown_turns_of(chat, FunctAI.read_log!(chat), parent, Dict{Symbol,Any}(), nothing)
            @test [FunctAI.turn_id(st) for st in picked] == expect["keep"]
            @test sections == expect["sections"]
            @test Dict(k => v for (k, v) in without) == Dict(k => String[v...] for (k, v) in expect["without"])
        elseif hook === :turn_start
            g = AIFunction("g", "G."; inputs=(message=String, tone=String), output=String)
            got, _ = FunctAI.turn_start_hooks(conversation(g), FunctAI.OrderedDict{String,Any}(start["inputs"]), Dict{Symbol,Any}())
            @test Dict(got) == expect["inputs"]
        else
            call = (fn=f, turn_run=nothing, name="f", changes=Any[])
            a = FunctAI.Approval("c", 1, "t1", "send", Dict{String,Any}(something(get(start, "inputs", nothing), Dict())), "changes", "f/send", "f#1")
            if hook === :tool_call
                got, refused = FunctAI.plugin_tool_call(call, a, Dict{Symbol,Any}())
                if haskey(expect, "block")
                    @test got === nothing && occursin(expect["block"], refused)
                else
                    @test Dict(got) == expect["inputs"]
                end
            else
                @test FunctAI.plugin_tool_result(call, a, Dict(), start["output"]) == expect["output"]
            end
        end
    end
    @test ran[] == expect["ran"]
end

"A contract definition as a Julia AI function (as contract.jl writes one)."
function baked_case_function(d)
    field(x) = get(x, "desc", nothing) === nothing ? x["shape"] : x["shape"] => x["desc"]
    settings = Dict{Symbol,Any}()
    haskey(d["settings"], "adapter") && (settings[:adapter] = d["settings"]["adapter"])
    get(d["settings"], "module", nothing) == "cot" && (settings[:reasoning] = true)
    AIFunction(d["name"], d["description"]; inputs=[Symbol(x["name"]) => field(x) for x in d["inputs"]],
               outputs=[Symbol(x["name"]) => field(x) for x in d["outputs"]], instructions=d["state"]["instructions"], settings...)
end
function signature_without_type(sig)
    data = LMCC.signature_to_dict(sig)
    Dict("instructions" => data["instructions"],
         "fields" => [Dict(k => v for (k, v) in merge(Dict{String,Any}("purpose" => "plain"), x) if k != "type" && v !== nothing) for x in data["fields"]])
end

@testset "baked case $name" for (name, c) in cases("baked")
    f = baked_case_function(c["definition"])
    rows = [merge(Dict{String,Any}(r["inputs"]), Dict{String,Any}(r["outputs"])) for r in c["rows"]]
    b = c["bake"]
    e = FunctAI.bake_entry(f; fixed=something(get(b, "fixed", nothing), Dict()), derived=something(get(b, "derived", nothing), Dict()),
                           reasoning=get(b, "reasoning", false) === true, rows)
    want = c["expect"]
    @test same(signature_without_type(e.signature), Dict("instructions" => want["signature"]["instructions"],
                                                "fields" => [merge(Dict{String,Any}("purpose" => "plain"), x) for x in want["signature"]["fields"]]))
    @test same(e.fixed, want["fixed"]) && same(e.derived, want["derived"])
    for (r, ex) in zip(c["rows"], want["examples"])
        msgs, reply = FunctAI.student_messages(e, merge(Dict{String,Any}(r["inputs"]), Dict{String,Any}(k => v for (k, v) in something(get(b, "fixed", nothing), Dict()))), r["outputs"])
        @test same(Any[msgs..., Dict("role" => "assistant", "content" => reply)], ex["messages"])
    end
end
