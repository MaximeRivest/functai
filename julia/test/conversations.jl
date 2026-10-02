# Stages 1.2 to 5 and plugins, on a fake model: conversations, tools that ask
# first, resuming without paying twice, helpers' memory, views, the reply
# cache, learning from rated turns, plugins and the built-ins.

xmlr(pairs...) = join(("<$k>\n$v\n</$k>" for (k, v) in pairs), "\n")
const GPT = "gpt-4.1-mini"

@ai function tutor(message::String)::String
    "Tutor."
end

@testset "a conversation: turns that see the earlier ones" begin
    r = FakeRouter(Any[xmlr(:result => "Welcome, Alex!"), xmlr(:result => "First, the bottoms."), xmlr(:result => "Yes")])
    with_settings(router=r, lm=GPT) do
        store = MemoryConversations()
        chat = conversation(tutor, "alex"; store)
        @test chat("Hi, I'm Alex.") == "Welcome, Alex!"
        @test chat("What is 1/2 + 1/3?") == "First, the bottoms."
        s = stream(chat, "Is it 5/6?")
        @test s.turn.inputs == Dict("message" => "Is it 5/6?")      # known at once
        @test fetch(s) == "Yes"
        # the third request shows both earlier turns, as they were
        @test [m.role for m in r.requests[3].messages] == ["user", "assistant", "user", "assistant", "user"]
        ts = turns(chat)
        @test length(ts) == 3 && all(t -> t.state == "done", ts)
        @test [t.id for t in ts[3].saw] == [ts[1].id, ts[2].id]
        @test ts[3].parent == ts[2].id && ts[2].result == "First, the bottoms."
        # the record does not grow with the conversation: saw_of the parent, then the parent
        rec = FunctAI.read_records(chat.store, chat.id)[end]
        @test rec["kind"] == "ended" && rec["saw"] == [Dict("saw_of" => ts[2].id), Dict("call" => ts[2].id, "steps" => true)]
        # the function itself remembers nothing
        r.replies = Any[xmlr(:result => "Hello")]
        tutor("Who am I?")
        @test length(r.requests[end].messages) == 1
        # opened again by id, the same conversation
        @test length(turns(conversation(tutor, "alex"; store))) == 3
    end
end

@testset "branches, the head, the last turns, a merge, render" begin
    r = FakeRouter(; responder=(req, i) -> xmlr(:result => "answer $(i + 1)"))
    with_settings(router=r, lm=GPT) do
        chat = conversation(tutor; context=last_turns(1))
        chat("one"); chat("two"); chat("three")
        @test length(r.requests[3].messages) == 3                     # one earlier turn shown
        ts = turns(chat)
        other = continue_from(chat, ts[1])
        @test other("two, again") == "answer 4"
        @test length(turns(other)) == 2 && turns(other)[2].parent == ts[1].id
        @test length(turns(chat; all=true)) == 4
        @test length(turns(chat)) == 3                                 # this view follows its own branch
        req = render(chat, "four")
        @test req.messages[end].parts[1].text |> x -> occursin("four", x)
        @test length(req.messages) == 3
        @test isempty(r.requests[end:end]) || length(r.requests) == 4  # rendering sends nothing
        # a merge: another function's answer becomes this program's turn
        base = conversation(tutor, "merge-demo"; store=MemoryConversations())
        base("question")
        b1 = continue_from(base, turns(base)[1])
        @ai function judge_answers(answers::Vector{Dict{String,Any}})::String
            "Pick the best answer."
        end
        r.responder = (req, i) -> xmlr(:result => "picked")
        x = continue_from(base, turns(base)[1])
        a1 = fetch(stream(x, "follow-up A"))
        y = continue_from(base, turns(base)[1])
        a2 = fetch(stream(y, "follow-up B"))
        all = turns(base; all=true)
        merged = merge!(continue_from(base, turns(base)[1]), all[2:3], judge_answers)
        @test merged.result == "picked" && merged.reads == [all[2].id, all[3].id]
        @test merged.made_by["name"] == "judge_answers"
    end
end

@testset "request ids, queueing, refusing, stopping from elsewhere" begin
    gate = Channel{Nothing}(1)
    r = FakeRouter(; responder=(req, i) -> (i == 0 && take!(gate); xmlr(:result => "r$i")))
    with_settings(router=r, lm=GPT) do
        store = MemoryConversations()
        chat = conversation(tutor, "busy"; store)
        s1 = stream(chat, "first"; request_id="click-1")
        sleep(0.2)
        @test stream(chat, "first"; request_id="click-1") === s1     # a double click is one turn
        refusing = conversation(tutor, "busy"; store, sends=:refuse)
        err = try
            refusing("second")
        catch e
            e
        end
        @test err isa ConversationError && err.code == "conversation-busy"
        queued = @async chat("second")                                  # waits, then continues from the first
        sleep(0.2)
        put!(gate, nothing)
        @test fetch(s1) == "r0"
        @test fetch(queued) == "r1"
        @test turns(chat)[2].parent == turns(chat)[1].id
    end
    # stopping a turn from another view of the store (another process would append the same record)
    hold = Channel{Nothing}(1)
    r2 = FakeRouter(; responder=(req, i) -> (take!(hold); xmlr(:result => "late")))
    dir = mktempdir()
    with_settings(router=r2, lm=GPT) do
        chat = conversation(tutor, "stoppable"; store=dir)
        s = stream(chat, "a long question")
        sleep(0.3)
        elsewhere = conversation(tutor, "stoppable"; store=FolderStore(dir))
        stop!(elsewhere, s.turn.id)
        put!(hold, nothing)
        err = try
            fetch(s)
        catch e
            e
        end
        @test err isa Cancelled
        @test wait_turn(FunctAI.turn(chat, s.turn.id)).state == "stopped"
    end
end

notes_store = Dict("todo.md" => "buy milk")
"Read a note."
read_note(name::String) = notes_store[name]
"Write a note."
write_note(name::String, text::String) = (notes_store[name] = text; "ok")
const RT = tool(read_note; effects=:reads)
const WT = tool(write_note; effects=:changes)
@ai tools = [RT, WT] function gardener(request::String)::String
    "Tend the notes."
end
gardener_script() = FakeRouter(Any[(calls=[("c1", "read_note", (name="todo.md",))],),
                                   (calls=[("c2", "write_note", (name="todo.md", text="buy oat milk"))],), xmlr(:result => "Done.")])

@testset "tools that ask first" begin
    @test RT.effects == "reads" && WT.effects == "changes" && tool(read_note).effects === nothing
    notes_store["todo.md"] = "buy milk"
    # a plain call has nobody to ask: refused before the tool runs
    err = try
        with_settings(() -> gardener("oat milk"); router=gardener_script(), lm=GPT, approve=:changes)
    catch e
        e
    end
    @test err isa ApprovalError && err.code == "approval-required" && notes_store["todo.md"] == "buy milk"
    # a function decides; a refusal is an answer the model sees
    r = gardener_script()
    @test with_settings(() -> gardener("oat milk"); router=r, lm=GPT, approve=a -> "no writing today") == "Done."
    @test notes_store["todo.md"] == "buy milk"
    @test r.requests[3].messages[end].parts[1].content[1].text == "The person did not allow this call. Reason: no writing today"
    # tools that read are never asked about by :changes
    asked = String[]
    r = gardener_script()
    with_settings(() -> gardener("oat milk"); router=r, lm=GPT, approve=a -> (push!(asked, a.name); true))
    @test asked == ["write_note"] && notes_store["todo.md"] == "buy oat milk"
    # on a stream without a conversation, the call waits for the stream's answer
    notes_store["todo.md"] = "buy milk"
    with_settings(router=gardener_script(), lm=GPT, approve=:all) do
        s = stream(gardener, "oat milk")
        waits() = timedwait(() -> !isempty(FunctAI.waiting_here()), 30.0) === :ok
        @test waits()
        approve!(s)
        sleep(0.1)
        @test waits()
        deny!(s; reason="not now")
        @test fetch(s) == "Done."
        kinds = [e.kind for e in eachevent(s)]
        @test count(==(:approval), kinds) == 2 && count(==(:approved), kinds) == 2
        @test notes_store["todo.md"] == "buy milk"
    end
end

@testset "a turn waits, saved, and goes on without paying twice" begin
    notes_store["todo.md"] = "buy milk"
    r = gardener_script()
    dir = mktempdir()
    with_settings(router=r, lm=GPT) do
        chat = conversation(gardener, "notes"; store=dir, approve=:changes)
        err = try
            chat("oat milk")
        catch e
            e
        end
        @test err isa Waiting && err.code == "turn-waiting" && only(err.approvals).name == "write_note"
        t = last(turns(chat))
        @test t.state == "waiting" && length(r.requests) == 2 && notes_store["todo.md"] == "buy milk"
        # answered from "another process": a fresh store object on the same folder
        elsewhere = conversation(gardener, "notes"; store=FolderStore(dir), approve=:changes)
        @test approve!(last(turns(elsewhere)); by="ana") == "Done."
        @test length(r.requests) == 3                        # the two earlier replies were reused
        @test notes_store["todo.md"] == "buy oat milk"
        @test last(turns(chat)).state == "done"
        # the turn's log goes on as a later writer's, and every event is in the store
        events = eachevent(last(turns(chat)))
        @test [e.writer for e in events] == [fill(1, 7); fill(2, 5)]
        @test events[end].kind === :done
        @test [e.seq for e in events] == collect(1:12)
        outside = eachevent(last(turns(chat)); view=:outside)
        @test all(e -> !(e.kind in (:tool_call, :tool_result)), outside)
    end
end

@testset "a tool that may have run is never run again on its own" begin
    calls = Ref(0)
    "Send an email."
    send_email(to::String) = (calls[] += 1; "sent")
    et = tool(send_email; effects=:changes)
    @ai tools = [et] function mailer(request::String)::String
        "Send mail."
    end
    store = MemoryConversations()
    r = FakeRouter(Any[(calls=[("c1", "send_email", (to="ana",))],), xmlr(:result => "Sent.")])
    with_settings(router=r, lm=GPT) do
        chat = conversation(mailer, "m"; store)
        chat("mail ana")
        t = last(turns(chat))
        # pretend the process died after the tool started: a started record with no result, a lease that ran out
        recs = FunctAI.read_records(store, "m")
        id = t.id
        dead_id = FunctAI.new_id()
        FunctAI.append_records!(store, "m", [Dict("functai_conversation" => 1, "kind" => "turn", "at" => FunctAI.iso(time()), "turn" => dead_id,
                                                  "parent" => id, "program" => recs[1]["version"], "inputs" => Dict("request" => "again")),
                                             Dict("functai_conversation" => 1, "kind" => "lease", "at" => FunctAI.iso(time()), "turn" => dead_id,
                                                  "holder" => "h:1:a", "until" => FunctAI.iso(time() - 1), "attempt" => 1),
                                             Dict("functai_conversation" => 1, "kind" => "tool", "at" => FunctAI.iso(time()), "turn" => dead_id,
                                                  "site" => "mailer#1", "invocation" => 1, "id" => "c1", "name" => "send_email",
                                                  "input" => Dict("to" => "ana"), "state" => "started")])
        dead = FunctAI.turn(chat, dead_id)
        @test dead.state == "interrupted" && length(dead.unfinished) == 1
        err = try
            resume!(dead)
        catch e
            e
        end
        @test err isa ConversationError && err.code == "turn-unfinished"
        r.replies = Any[(calls=[("c1", "send_email", (to="ana",))],), xmlr(:result => "Sent again.")]
        before = calls[]
        @test resume!(dead; results=Dict(1 => "sent")) == "Sent again."
        @test calls[] == before                              # given, not run again
    end
end

@testset "a program's conversation: helpers remember only when told, earlier()" begin
    @ai function topic(message::String)::String
        "The topic."
    end
    @ai function answer(message::String, topic::String)::String
        "Answer the customer."
    end
    seen = Ref{Any}(nothing)
    @program function support(message::String)::String
        "Support."
        seen[] = FunctAI.earlier()
        answer(message, topic(message))
    end
    r = FakeRouter(; responder=(req, i) -> xmlr(:result => "r$i"))
    with_settings(router=r, lm=GPT) do
        chat = conversation(support, "ana"; store=MemoryConversations(), remembers=Dict(answer => :conversation))
        chat("Where is my parcel?")
        chat("And the invoice?")
        # topic remembers nothing; answer sees its own earlier call
        topic_req = r.requests[3]
        answer_req = r.requests[4]
        @test length(topic_req.messages) == 1
        @test length(answer_req.messages) == 3
        @test seen[] == [Dict("message" => "Where is my parcel?", "result" => "r1")]
        @test length(turns(chat)) == 2
        @test turns(chat)[2].usage["input_tokens"] == 20     # every call inside the turn
        # a remembering program used inside another, undeclared, is refused
        inner = conversation(tutor, "inner"; store=MemoryConversations())
        @program function outer_prog(message::String)::String
            "Outer."
            inner(message)
        end
        err = try
            conversation(outer_prog, "outer"; store=MemoryConversations())("hi")
        catch e
            e
        end
        @test err isa ConversationError && err.code == "conversation-nested"
    end
    @test FunctAI.earlier() == []
end

@testset "what a conversation refuses" begin
    @test (try conversation(tutor, "a/b") catch e; e end).code == "conversation-id"
    dir = mktempdir()
    secret = configure(tutor; log_content=false)
    @test (try conversation(secret, "s"; store=dir) catch e; e end).code == "conversation-content"
    conversation(secret, "s")                                 # memory: nothing outlives the process
    @program function opaque_prog(x)::String
        "Opaque."
        "y"
    end
    @test (try conversation(opaque_prog) catch e; e end).code == "conversation-opaque"
    # reasoning turned on: earlier turns lack it
    r = FakeRouter(; responder=(req, i) -> xmlr(:reasoning => "because", :result => "r$i"))
    with_settings(router=r, lm=GPT) do
        cstore = MemoryConversations()
        chat = conversation(tutor, "changing"; store=cstore)
        r.responder = (req, i) -> xmlr(:result => "r$i")
        chat("one")
        thinking = configure(tutor; reasoning=true)
        @test (try conversation(thinking, "changing"; store=cstore) catch e; e end).code == "conversation-signature"
        r.responder = (req, i) -> xmlr(:reasoning => "because", :result => "r$i")
        @test conversation(thinking, "changing"; store=cstore, earlier_without=["reasoning"])("two") == "r1"
    end
end

@testset "the reply cache: kept once read, one flight, replicates, on disk" begin
    r = FakeRouter(; responder=(req, i) -> xmlr(:result => "answer $i"))
    @ai function capital(country::String)::String
        "The country's capital."
    end
    with_settings(router=r, lm=GPT, cache_replies=true) do
        FunctAI.clear_cache()
        @test capital("Peru") == "answer 0"
        @test capital("Peru") == "answer 0" && length(r.requests) == 1
        @test configure(capital; replicate=1)("Peru") == "answer 1"      # another answer to the same request
    end
    # an unreadable reply is never kept
    bad = FakeRouter(; responder=(req, i) -> i == 0 ? "no tags at all" : xmlr(:result => "Lima"))
    with_settings(router=bad, lm=GPT, cache_replies=true, retries=0) do
        FunctAI.clear_cache()
        @test (try capital("Chile") catch e; e end) isa LMCC.Refusal
        @test capital("Chile") == "Lima" && length(bad.requests) == 2
    end
    # on disk: a second "run" sends nothing; the file is Python's format
    dir = mktempdir()
    path = joinpath(dir, "replies.sqlite")
    r2 = FakeRouter(; responder=(req, i) -> xmlr(:result => "disk $i"))
    with_settings(router=r2, lm=GPT, cache_replies=path) do
        @test capital("Kenya") == "disk 0"
    end
    empty!(FunctAI.DISK_REPLIES)                                # as if the process had stopped
    with_settings(router=r2, lm=GPT, cache_replies=path) do
        @test capital("Kenya") == "disk 0" && length(r2.requests) == 1
    end
    @test length(FunctAI.DiskReplies(path)) == 1
    # one flight: two calls of the same request at once ask the model once
    slow = FakeRouter(; responder=(req, i) -> (sleep(0.3); xmlr(:result => "once")))
    with_settings(router=slow, lm=GPT, cache_replies=true) do
        FunctAI.clear_cache()
        a = @async capital("Japan")
        b = @async capital("Japan")
        @test fetch(a) == fetch(b) == "once" && length(slow.requests) == 1
    end
    # a call whose log_content drops a field is not written to disk
    secret = configure(capital; log_content=false)
    r3 = FakeRouter(; responder=(req, i) -> xmlr(:result => "s$i"))
    with_settings(router=r3, lm=GPT, cache_replies=joinpath(dir, "other.sqlite")) do
        secret("Fiji")
    end
    @test length(FunctAI.DiskReplies(joinpath(dir, "other.sqlite"))) == 0
end

@testset "plugins: changes as data, recorded" begin
    seen = Ref{Any}(nothing)
    modes = Plugin("modes"; version="1.2.0")
    on!(modes, :before_call) do call
        seen[] = (call.function, call.path, call.instruction)
        Change(sections=["Answer like a pirate."], settings=Dict("temperature" => 0.5))
    end
    @test_throws PluginError on!(identity, modes, :before_cal)
    @test (try Plugin("Bad Name") catch e; e end).code == "plugin-name"
    @test (try Plugin("future"; api=2) catch e; e end).code == "plugin-api"
    dir = mktempdir()
    r = FakeRouter(Any[xmlr(:result => "Arr")])
    with_settings(router=r, lm=GPT, plugins=[modes], log_calls=dir) do
        @test tutor("hi") == "Arr"
    end
    @test seen[][1] == "tutor" && seen[][2] == "tutor"
    @test occursin("Answer like a pirate.", r.requests[1].system) && r.requests[1].config.temperature == 0.5
    rec = only(FunctAI.read_log(dir)[1])
    @test rec["changes"] == [Dict("plugin" => "modes", "version" => "1.2.0", "hook" => "before_call",
                                  "change" => Dict("sections" => ["Answer like a pirate."], "settings" => Dict("temperature" => 0.5)))]
    # a handler that throws stops the call: the change it was meant to make did not happen
    broken = Plugin("broken"; before_call=_ -> error("oops"))
    err = with_settings(router=FakeRouter(Any["x"]), lm=GPT, plugins=[broken]) do
        try tutor("hi") catch e; e end
    end
    @test err isa PluginError && err.code == "plugin-failed"
    # a change a hook cannot make
    wrong = Plugin("wrong"; before_call=_ -> Change(block="no"))
    @test (with_settings(router=FakeRouter(Any["x"]), lm=GPT, plugins=[wrong]) do; try tutor("hi") catch e; e end; end).code == "plugin-change"
    # the request hook: the escape hatch, recorded, and the call is not replayable
    hatch = Plugin("hatch"; request=e -> LM15.Request(e.request; model="gpt-4.1"))
    dir2 = mktempdir()
    r2 = FakeRouter(Any[xmlr(:result => "ok")])
    with_settings(() -> tutor("hi"); router=r2, lm=GPT, plugins=[hatch], log_calls=dir2)
    rec = only(FunctAI.read_log(dir2)[1])
    @test r2.requests[1].model == "gpt-4.1" && rec["replayable"] == false && !haskey(rec["exchanges"][1], "request_hash")
    # a tool_call hook blocks; a host's program_plugins = false drops a program's own
    block = Plugin("guard"; tool_call=t -> t.name == "write_note" ? Change(block="read only here") : nothing)
    notes_store["todo.md"] = "buy milk"
    r3 = gardener_script()
    with_settings(() -> gardener("x"); router=r3, lm=GPT, plugins=[block])
    @test notes_store["todo.md"] == "buy milk"
    @test r3.requests[3].messages[end].parts[1].content[1].text == "This call was blocked (guard): read only here"
    own = configure(tutor; plugins=[modes])
    r4 = FakeRouter(Any[xmlr(:result => "plain")])
    with_settings(() -> own("hi"); router=r4, lm=GPT, program_plugins=false)
    @test !occursin("pirate", something(r4.requests[1].system, ""))
    # a plugin from a file
    file = joinpath(mktempdir(), "shout.jl")
    write(file, "plugin = Plugin(\"shout\"; before_call = _ -> Change(sections = [\"SHOUT.\"]))\n")
    @test load_plugin(file).name == "shout"
    r5 = FakeRouter(Any[xmlr(:result => "OK")])
    with_settings(() -> tutor("hi"); router=r5, lm=GPT, plugins=[file])
    @test occursin("SHOUT.", r5.requests[1].system)
end

@testset "plugins in a conversation: context, turn_start, turn_end, entries" begin
    r = FakeRouter(; responder=(req, i) -> xmlr(:result => "r$i"))
    trim = Plugin("trim"; turn_start=e -> Change(inputs=Dict("message" => strip(e.inputs["message"]))))
    ended = String[]
    log_end = Plugin("log-end"; turn_end=e -> (push!(ended, e.state); FunctAI.remember!(e, "seen", e.turn); nothing))
    with_settings(router=r, lm=GPT) do
        chat = conversation(tutor, "plugged"; store=MemoryConversations(), plugins=[trim, log_end])
        chat("   hi   ")
        @test turns(chat)[1].inputs == Dict("message" => "hi")
        @test ended == ["done"]
        @test only(FunctAI.entries(chat, "log-end", "seen")).data == turns(chat)[1].id
        @test getfield(turns(chat)[1], :st).record["changes"][1]["plugin"] == "trim"
    end
end

@testset "compaction: a summary instead of the oldest turns" begin
    summaries = Any[]
    summarize = (earlier, rows) -> (push!(summaries, (earlier, length(rows))); "summary of $(length(rows)) turns")
    r = FakeRouter(; responder=(req, i) -> xmlr(:result => "r$i"))
    with_settings(router=r, lm=GPT) do
        chat = conversation(tutor, "long"; store=MemoryConversations(), plugins=[compaction(keep=2, every=2, summarize=summarize)])
        for i in 1:6
            chat("message $i")
        end
        @test summaries[1] == ("", 2)
        last_req = r.requests[end]
        @test occursin("summarized", last_req.system)
        # what the last turn was shown is recorded: a rated turn is asked again with the same summary
        st = getfield(last(turns(chat)), :st)
        @test haskey(st.record, "context") && !isempty(st.record["context"]["sections"])
        @test length(st.record["context"]["turns"]) <= 3
    end
end

@testset "delegation: another program as a tool, remembering per branch" begin
    @ai function research(question::String)::String
        "Look things up."
    end
    helper = delegate(research; description="Look things up in the notes.")
    @test helper.name == "research" && helper.effects == "reads"
    @ai tools = [helper] function assistant(request::String)::String
        "Help."
    end
    r = FakeRouter(Any[(calls=[("c1", "research", (question="q1",))],), xmlr(:result => "found 1"), xmlr(:result => "done 1"),
                       (calls=[("c2", "research", (question="q2",))],), xmlr(:result => "found 2"), xmlr(:result => "done 2")])
    with_settings(router=r, lm=GPT) do
        dstore = MemoryConversations()
        chat = conversation(assistant, "desk"; store=dstore)
        @test chat("first") == "done 1"
        @test chat("second") == "done 2"
        sub = conversation(research, "desk.research"; store=dstore)
        @test length(turns(sub)) == 2                           # it remembers what it was asked on this branch
        @test length(r.requests[5].messages) == 3
    end
end

@testset "learning from rated turns: earlier, asked again with them, train_test" begin
    dir = mktempdir()
    r = FakeRouter(; responder=(req, i) -> xmlr(:result => "r$i"))
    with_settings(router=r, lm=GPT, log_calls=dir) do
        chat = conversation(tutor, "lesson"; store=MemoryConversations())
        chat("one"); chat("two")
        t = last(turns(chat))
        rate(t.id, :right; folder=dir)
        rows = rated(tutor; folder=dir)
        @test length(rows) == 1
        row = only(rows)
        @test row.conversation == "lesson" && length(row.earlier) == 1 && row.earlier[1]["inputs"] == Dict("message" => "one")
        e = evaluate(tutor, rows)
        @test length(r.requests[end].messages) == 3                     # asked again with its earlier turn
        rec = FunctAI.read_log(dir)[1][end]
        @test rec["saw"] == [Dict("saw_of" => t.id)]
        # a worked example is never a rated turn of a conversation
        @test isempty(demos(labeled_few_shot(tutor, rows)))
    end
    rows = [(x=1, conversation="a"), (x=2, conversation="a"), (x=3, conversation="b"), (x=4, conversation=missing)]
    train, test = train_test(rows; test=0.34)
    @test isempty(intersect(Set(r.conversation for r in train if r.conversation !== missing),
                            Set(r.conversation for r in test if r.conversation !== missing)))
    @test length(train) + length(test) == 4
end

@testset "pruning the call log keeps what ratings need" begin
    dir = mktempdir()
    r = FakeRouter(; responder=(req, i) -> xmlr(:result => "r$i"))
    p = with_settings(() -> predict(tutor, "keep me"); router=r, lm=GPT, log_calls=dir)
    with_settings(() -> tutor("drop me"); router=r, lm=GPT, log_calls=dir)
    rate(p, :right; folder=dir)
    day = only(filter(d -> occursin(r"^\d{4}", d), readdir(dir)))
    mv(joinpath(dir, day), joinpath(dir, "2000-01-01"))
    got = prune_calls("30d"; folder=dir)
    @test got == (days=1, calls=1, kept=1)
    @test length(rated(tutor; folder=dir)) == 1
end

@testset "quotes_found" begin
    source = "The parcel left Leeds on Monday. It was delayed by snow."
    @test quotes_found(source, ["“It was delayed by snow”", "It was lost"]) == [true, false]
    @test quotes_found(source, "the PARCEL   left leeds")
    @test !quotes_found(source, "")
end
