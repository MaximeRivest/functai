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
    program_tool(interface(program), tool_name, something(description, Some(nothing)), run, eff, program)
end

# ------------------------------------------------------------------ long tool loops (the step hook)

const LIMIT_REACHED = "This task has reached its limit (%s). Do not ask for any more tools: give your answer now, from what you have found so far."
const LAST_STEP = "This is the last step this task may take. Do not ask for any more tools: give your answer now, from what you have found so far."
const PICTURE_GONE = "(A picture was here. It is no longer shown, to save space.)"
const COST_UNKNOWN = "a request's cost is not known from its token counts (a price without cache_read_per_million and " *
                     "cache_write_per_million is exact only for requests that report their cache counts; give both, equal to " *
                     "input_per_million when the provider does not charge them apart)"

"A limit as the model is told it: a whole number without a fraction."
limit_number(x) = isinteger(x) ? string(Int(x)) : @sprintf("%g", x)

"""
Which limits a tree's spending reached (contract/plugins.md, *limits*): each
`(limit, why)`, the limit as the model is told it (`"cost 5"`) and why it
counts as reached when its count is not known (else `nothing`). A count that
is not known cannot be checked, and counts as reached.
"""
function limits_reached(spent::Spent; tokens=nothing, cost=nothing, seconds=nothing, prices=nothing)
    out = Tuple{String,Union{Nothing,String}}[]
    if tokens !== nothing
        got = spent.tokens
        got === nothing ? push!(out, ("$(limit_number(tokens)) tokens", "a request did not say how many it used")) :
                          got >= tokens && push!(out, ("$(limit_number(tokens)) tokens", nothing))
    end
    if cost !== nothing
        got = FunctAI.cost(spent, prices)
        if got === nothing
            unknown = sort!(unique(String[m for (m, _) in spent.exchanges if pricing(prices, m) === nothing]))
            push!(out, ("cost $(limit_number(cost))", isempty(unknown) ? COST_UNKNOWN : "no price is known for $(join(unknown, ", "))"))
        elseif got >= cost
            push!(out, ("cost $(limit_number(cost))", nothing))
        end
    end
    seconds !== nothing && spent.seconds >= seconds && push!(out, ("$(limit_number(seconds)) seconds", nothing))
    out
end

"The reason a `limits` stop gives: each limit reached, and why when its count is not known."
stopped_by(hit) = "limit reached: " * join((why === nothing ? label : "$label ($why)" for (label, why) in hit), "; ")

"A price in `limits`' config: its fields by lm15's names, without those at lm15's defaults (an unset rate, USD, no tiers)."
function price_config(p::LM15.InferencePricing)
    d = LM15.to_dict(p)
    JObj(String(k) => v for (k, v) in d if v !== nothing && !(k == "currency" && v == "USD") && !(k == "tiers" && isempty(v)))
end

"""
    limits(; tokens, cost, seconds, prices, then = :answer, name = "limits") -> Plugin

Stop a call tree (an agent, a pipeline) when it has used enough: tokens,
money or time. Nothing is limited by default; this plugin is opt-in.

Counted over the whole tree a call starts (every AI function inside a
program, every tool's own calls), from the requests a provider answered in
this process (replies from the reply cache, and those a resumed turn
reuses, cost nothing now and are not counted), plus what code said it
spent ([`spend`](@ref)). Checked before each AI function's call and before
each step of a tool loop, so one request may go past a limit; the next does
not start.

- `tokens`: every token read and written (`total_tokens`), summed.
- `cost`: the list price of the requests, from `prices` (needed with it).
  Exact or nothing: a request to a model with no price, or whose counts do
  not settle its price, counts as reaching the limit.
- `seconds`: since the tree's outermost call started.
- `prices`: a model's name (as you ask for it) => an lm15
  `InferencePricing`, or its fields (`(input_per_million = 3,
  output_per_million = 15, cache_read_per_million = 0.3,
  cache_write_per_million = 3.75)`). FunctAI ships no price list: prices
  change, and a wrong price would mislead.
- `then`: `:answer` (the default): a tool loop that reaches a limit is
  offered no more tools and told to answer with what it has; so is a loop
  at its last step (`max_steps`), instead of failing with `StepLimit`.
  `:stop`: the call stops. Either way a call that would start past a limit
  stops ([`PluginError`](@ref), code `plugin-stopped`).

```julia
prices = Dict("claude-opus-5-5" => (input_per_million = 5, output_per_million = 25))
with_settings(plugins = [limits(cost = 3.0, seconds = 600, prices)]) do
    concept_test(url)
end
```
"""
function limits(; tokens=nothing, cost=nothing, seconds=nothing, prices=nothing, then=:answer, name::AbstractString="limits")
    String(then) in ("answer", "stop") || throw(ArgumentError("then is :answer or :stop, not $(repr(then))"))
    cost !== nothing && (prices === nothing || isempty(prices)) &&
        throw(ArgumentError("a cost limit needs prices: Dict(model => (input_per_million = …, output_per_million = …)); FunctAI ships none"))
    for (label, v) in (("tokens", tokens), ("cost", cost), ("seconds", seconds))
        v === nothing || (v isa Real && !(v isa Bool) && isfinite(v) && v > 0) || throw(ArgumentError("$label is a number above 0, not $(repr(v))"))
    end
    tokens === nothing && cost === nothing && seconds === nothing && throw(ArgumentError("limits needs at least one of tokens, cost, seconds"))
    priced = prices === nothing ? nothing : Dict{String,LM15.InferencePricing}(String(k) => pricing(Dict(String(k) => v), String(k)) for (k, v) in pairs(prices))
    config = JObj()
    tokens === nothing || (config["tokens"] = tokens)
    cost === nothing || (config["cost"] = cost)
    seconds === nothing || (config["seconds"] = seconds)
    priced === nothing || (config["prices"] = JObj(k => price_config(p) for (k, p) in sort!(collect(priced); by=first)))
    config["then"] = String(then)
    limit = (tokens=tokens, cost=cost, seconds=seconds, prices=priced)
    before = function (call)
        hit = limits_reached(call.spent; limit...)
        if isempty(hit) && cost !== nothing && call.lm isa AbstractString && pricing(priced, call.lm) === nothing
            hit = [("cost $(limit_number(cost))", "no price is known for $(call.lm)")]
        end
        isempty(hit) ? nothing : Change(stop=stopped_by(hit))
    end
    step = function (s)
        hit = limits_reached(s.spent; limit...)
        !isempty(hit) && String(then) == "stop" && return Change(stop=stopped_by(hit))
        if String(then) == "answer" && !isempty(s.tools) && (!isempty(hit) || s.last_step)
            return Change(tools=String[], reminder=isempty(hit) ? LAST_STEP :
                                                   replace(LIMIT_REACHED, "%s" => join((label for (label, _) in hit), ", ")))
        end
        nothing
    end
    Plugin(name; version="1.0.0", description="stop a call tree when it used enough tokens, money or time", config,
           before_call=before, step)
end

"""
Which earlier tool results `keep_pictures` shows otherwise, and as what
(contract/plugins.md): the pictures before the newest `last` are replaced by
a short text, once at least `at_once` of them are due. `invocation => result`.
"""
function pictures_to_drop(results; last::Integer, at_once::Integer=1)
    seen = 0
    drop = Dict{Int,Vector{Int}}()
    for r in Iterators.reverse(results)
        r.output isa AbstractString && continue
        for i in length(r.output):-1:1
            get(r.output[i], "type", nothing) == "image" || continue
            seen += 1
            seen > last && push!(get!(() -> Int[], drop, r.invocation), i)
        end
    end
    sum(length, values(drop); init=0) < at_once && return Dict{Int,Any}()
    out = Dict{Int,Any}()
    for r in results
        haskey(drop, r.invocation) || continue
        gone = Set(drop[r.invocation])
        out[r.invocation] = Any[i in gone ? LMCC.textpart(PICTURE_GONE) : LMCC.deepcopy_json(p) for (i, p) in enumerate(r.output)]
    end
    out
end

"""
    keep_pictures(last = 3; at_once = 1, name = "keep-pictures") -> Plugin

Show a long tool loop only its newest pictures: before each step, the
pictures tool results gave earlier than the newest `last` are replaced by a
short text (the results' own text stays). A browsing agent that takes a
screenshot at every click otherwise sends every screenshot again with every
request.

`at_once`: drop old pictures only once this many are due, all together. A
provider's prompt cache keeps the start of a request only while it stays
the same, and every drop changes it: with caching on, `at_once = 5` drops
five pictures every fifth step instead of one at each. The turn and the
call log keep every picture as the tool gave it; each request in the call
log holds what it sent.

```julia
agent = configure(browse; plugins = [keep_pictures(3)], max_steps = 200)
```
"""
function keep_pictures(last::Integer=3; at_once::Integer=1, name::AbstractString="keep-pictures")
    (last isa Bool || last < 0) && throw(ArgumentError("last is how many of the newest pictures stay shown (0 or more)"))
    (at_once isa Bool || at_once < 1) && throw(ArgumentError("at_once is 1 or more"))
    step = function (s)
        drop = pictures_to_drop(s.results; last, at_once)
        isempty(drop) ? nothing : Change(replace=drop)
    end
    Plugin(name; version="1.0.0", description="show a tool loop only its newest pictures",
           config=LMCC.jobj("last" => Int(last), "at_once" => Int(at_once)), step)
end
