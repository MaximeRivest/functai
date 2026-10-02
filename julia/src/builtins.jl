# Built-in plugins, made with the public hooks only (contract/plugins.md,
# "Built-in plugins"): anyone can replace them with their own.
#
#     chat = conversation(tutor, "alex"; store = "tutoring/", plugins = [compaction(keep = 20)])
#     researcher = delegate(research; description = "Look things up in the notes.")
#     @ai tools = [researcher] function assistant(request::String)::String … end

const DEFAULT_SUMMARIZER = Ref{Any}(nothing)

"The AI function compaction summarizes with by default (the same instruction and fields as Python's)."
function default_summarizer()
    DEFAULT_SUMMARIZER[] === nothing || return DEFAULT_SUMMARIZER[]
    DEFAULT_SUMMARIZER[] = AIFunction("summarize_conversation",
        "Write the summary of a conversation that whoever continues it needs:\n" *
        "what was asked, what was answered and decided, names, numbers and\n" *
        "facts that came up, and what is still open. Start from the earlier\n" *
        "summary (empty when there is none) and fold the new turns into it. Be\n" *
        "complete about facts and short about everything else.";
        inputs=(earlier_summary=String, new_turns=Vector{Dict{String,Any}}), output=String, module_name="functai.builtins")
end

"""
    compaction(; keep = 20, every = 10, summarize = nothing, lm = nothing, name = "compaction") -> Plugin

Keep a long conversation short: older turns are folded into a summary. When a
turn ends and more than `keep + every` turns of its branch are not yet
summarized, every turn but the last `keep` is folded into the summary (the
earlier summary and those turns, given to `summarize(earlier_summary,
new_turns)`, `new_turns` a vector of `Dict(input…, output…)` rows; default: an
AI function of FunctAI's, on `lm` when given). The summary is kept in the
conversation (an entry at that turn, so each branch has its own), and the
next turns are shown it, as a section of the instruction, with only the turns
after it. What a turn was shown is in its record, so a rated turn is asked
again with the same summary. `every` keeps the summary's text the same for
`every` turns: a provider's prompt cache keeps its prefix meanwhile.

```julia
chat = conversation(tutor, "alex"; store = "tutoring/", plugins = [compaction(keep = 20, every = 10)])
```
"""
function compaction(; keep::Integer=20, every::Integer=10, summarize=nothing, lm=nothing, name::AbstractString="compaction")
    (keep >= 0 && every >= 1) || throw(ArgumentError("compaction(keep = n, every = m): keep a whole number of at least 0, every at least 1"))
    latest(es) = isempty(es) ? nothing : last(es).data
    fold = function (event::TurnEndEvent)
        event.state == "done" || return nothing
        ts = turns(event)
        summary = latest(entries(event, "summary"))
        ids = [t.id for t in ts]
        through = summary === nothing ? nothing : findfirst(==(summary["through"]), ids)
        open_turns = ts[(through === nothing ? 1 : through + 1):end]
        length(open_turns) < keep + every && return nothing
        folded = open_turns[1:end-keep]
        rows = Any[merge(JObj(t.inputs), JObj(t.outputs)) for t in folded]
        fn = something(summarize, Some(nothing))
        if fn === nothing
            fn = default_summarizer()
            lm === nothing || (fn = configure(fn; lm=String(lm)))
        end
        text = with_settings(; caller=Dict("compaction" => event.turn)) do
            Base.invokelatest(fn, summary === nothing ? "" : summary["text"], rows)
        end
        (text isa AbstractString && !isempty(strip(text))) || throw(ArgumentError("a summary is a text"))
        remember!(event, "summary", LMCC.jobj("through" => last(folded).id, "text" => String(text),
                                              "turns" => (summary === nothing ? 0 : summary["turns"]) + length(folded)))
        nothing
    end
    show = function (event::ContextEvent)
        summary = latest(entries(event, "summary"))
        summary === nothing && return nothing
        log = read_log!(event.conversation)
        ids = [turn_id(st) for st in branch(log, event.parent)]
        i = findfirst(==(summary["through"]), ids)
        i === nothing && return nothing
        covered = Set(ids[1:i])
        Change(keep=[t.id for t in event.turns if !(t.id in covered)],
               sections=["Earlier in this conversation ($(summary["turns"]) turns, summarized):\n$(summary["text"])"])
    end
    Plugin(name; version="1.0.0", description="compaction: summarize all but the last $keep turns", turn_end=fold, context=show)
end

"""
    delegate(program; name, description, remember = true, effects) -> AITool

Another program as a tool: an assistant hands part of its work to it. Asked
inside a conversation's turn, the program answers in a conversation of its
own (`<conversation>.<name>`, in the same store), which follows the branch of
the turn that asked: asked again later on that branch, it remembers what it
was asked before; on another branch, it does not. Its calls are in the
asking turn's call tree, under the tool call. Outside a conversation, or
with `remember = false`, it is called plainly. `effects` defaults to
`:reads` when every tool it has only reads, else unknown (counts as `:changes`).

```julia
researcher = delegate(research; description = "Look things up in the notes.")
@ai tools = [researcher] function assistant(request::String)::String
    "Help with the request."
end
```
"""
function delegate(program::Union{AIFunction,AIProgram}; name=nothing, description=nothing, remember::Bool=true, effects=nothing)
    tool_name = String(something(name, program_name(program)))
    run = function (input)
        given = OrderedDict{String,Any}(String(k) => v for (k, v) in input)
        call = CURRENT_CALL[]
        turn_run = call === nothing ? nothing : call.turn_run
        if !remember || turn_run === nothing
            return program isa AIFunction ? (p = predict_inputs(program, with_defaults(program, given)); p === missing ? missing : p.value) :
                                            program(; (Symbol(k) => v for (k, v) in given)...)
        end
        outer = turn_run.conv
        sub_id = "$(outer.id).$tool_name"
        length(sub_id) > 200 && (sub_id = "$(first(outer.id, 150)).$(first(LMCC.sha256_hex(sub_id), 16))")
        sub = conversation(program, sub_id; store=outer.store, delegated=true)
        previous = entries(outer, "delegate", tool_name; branch=turn_run.turn)
        if isempty(previous)
            sub.head, sub.exact = nothing, true            # the first delegation on this branch starts afresh
        else
            sub = continue_from(sub, last(previous).data["turn"])
        end
        bound = program isa AIFunction ? with_defaults(program, given) : (given=given, extra=0, twice=String[], positional=String[])
        s = send_bound_turn(sub, bound, Dict{Symbol,Any}(), nothing; passive=true)
        value = fetch(s)
        remember!(outer, "delegate", tool_name, LMCC.jobj("turn" => getfield(s, :conv_turn)[2]); turn=turn_run.turn)
        value
    end
    eff = effects === nothing ? (program isa AIFunction ? effects_of(program) : nothing) : effects_value(effects)
    program_tool(interface(program), tool_name, something(description, Some(nothing)), run, eff)
end
