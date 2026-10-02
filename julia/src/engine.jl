# One call of an AI function: lay it out (lmcc), send it (lm15), read the
# reply (lmcc), re-ask when it cannot be read, run tools until the model
# answers (contract/functions.md, "When the reply cannot be read"). No prompt
# text is written here except the one re-ask sentence the contract gives;
# every other byte of the request comes from the plan.

"The tool loop reached `max_steps` without an answer."
struct StepLimit <: Exception
    msg::String
    turn::Any
end
Base.showerror(io::IO, e::StepLimit) = print(io, "StepLimit: ", e.msg)

"A stream was closed before its call ended."
struct Cancelled <: Exception
    msg::String
end
Cancelled() = Cancelled("the stream was closed")
Base.showerror(io::IO, e::Cancelled) = print(io, "Cancelled: ", e.msg)

"""
    Prediction

Everything one call produced: `value` (what calling the function returns),
`answer` (the answer output's value), `outputs` (every output, typed, as a
`NamedTuple`), `call` (the call log's id: rate it with `rate(p, :right)`),
`turn` (lmcc's record of the exchange) and `responses` (lm15's replies).
"""
struct Prediction
    value::Any
    outputs::NamedTuple
    answer_name::String
    call::String
    turn::Any
    responses::Vector{Any}
    repairs::Vector{Any}
    probabilities::JObj
end
function Base.getproperty(p::Prediction, name::Symbol)
    name === :probabilities && return typed_probabilities(p)
    name === :answer && return getproperty(getfield(p, :outputs), Symbol(getfield(p, :answer_name)))
    name === :response && return isempty(getfield(p, :responses)) ? nothing : last(getfield(p, :responses))
    name === :attempts && return length(getfield(p, :responses))
    getfield(p, name)
end
Base.propertynames(::Prediction) = (:value, :answer, :outputs, :call, :turn, :response, :responses, :attempts, :repairs, :probabilities)

"An answer as the probabilities spell it: text for a text or choice answer, JSON otherwise."
answer_json(T, k) = T <: Union{AbstractString,Symbol,Enum} ? k : (try
    LMCC.parse_json(k)
catch
    k
end)

"""
The probabilities the model measured for its own answers, by output, with
the answers as their types: `p.probabilities.result[billing] == 0.93`. Empty
when the model measures none: only TypeSafe (`typesafe:jev-latest`) and
servers that score tokens do; other providers give an answer, not a
distribution (a model's own words about its confidence are not one).
"""
function typed_probabilities(p::Prediction)
    raw = getfield(p, :probabilities)
    outputs = getfield(p, :outputs)
    names = Symbol[]
    dists = Any[]
    for (field, dist) in raw
        haskey(outputs, Symbol(field)) || continue
        T = typeof(outputs[Symbol(field)])
        typed = OrderedDict{Any,Float64}()
        for (k, v) in dist
            key = try
                fromjson(T, answer_json(T, k), field)
            catch
                k
            end
            typed[key] = Float64(v)
        end
        push!(names, Symbol(field))
        push!(dists, all(k -> k isa T, keys(typed)) ? OrderedDict{T,Float64}(typed) : typed)
    end
    NamedTuple{Tuple(names)}(Tuple(dists))
end
function Base.show(io::IO, p::Prediction)
    print(io, "Prediction(")
    show(io, p.value)
    print(io, "; call = \"", p.call, "\")")
end

# ------------------------------------------------------------------ watching

"Whether calls through `router` can be streamed: a method of `LM15.stream(router, request)`."
can_stream(router) = hasmethod(LM15.stream, Tuple{typeof(router),LM15.Request})

"A piece of an output's text (never empty: a piece holds text)."
function emit_text(call::Call, field::AbstractString, text::AbstractString)
    isempty(text) && return nothing
    emit!(call, :text, LMCC.jobj("field" => String(field), "answer" => field == call.program["answer"], "text" => String(text)))
end
emit_thinking(call::Call, text) = isempty(text) ? nothing : emit!(call, :thinking, LMCC.jobj("text" => String(text)))
emit_retry(call::Call, reason, wait) = emit!(call, :retry, LMCC.jobj("reason" => String(reason), "wait" => wait))

# ------------------------------------------------------------------ a job

struct Job
    name::String
    plan::Any
    past::Vector{Any}
    inputs::AbstractDict
    settings::Dict{Symbol,Any}
    router::Any
    model::String
    tools::Vector{Any}
    call::Call
    typed::Function                  # the reply's values (JSON) → typed, or a parse-value refusal
    asked::Vector{Any}               # every tool call the model asked for, across steps (outputs.calls)
end
Job(name, plan, past, inputs, settings, router, model, tools, call, typed) =
    Job(name, plan, past, inputs, settings, router, model, tools, call, typed, Any[])

"Sleep `seconds`, waking to stop when a stream watching the call closes."
function cancellable_sleep(call, seconds)
    deadline = time() + seconds
    while time() < deadline
        check_cancelled(call)
        sleep(min(0.1, max(0.0, deadline - time())))
    end
end

"The call begins a request to a model (every exchange is one); a resumed turn replaying a request it made before shows none."
function begin_request!(call::Call, model)
    call.tree.replaying && return
    call.requests += 1
    emit!(call, :request, LMCC.jobj("request" => call.requests, "model" => model))
end

"Only the fields a stream shows: an output's text (tool calls are shown whole)."
function show_events(job::Job, batch)
    for ev in batch
        get(ev, "kind", nothing) == "field_delta" && ev["field"] != "calls" && emit_text(job.call, ev["field"], ev["text"])
    end
end

"A reply that came whole, shown as one text piece per field (streaming.md, law 6)."
function replay(job::Job, response)
    s = LMCC.stream(job.plan)
    shown = JObj[]
    try
        for e in LM15.response_to_events(response)
            e isa LM15.StreamDeltaEvent && append!(shown, LMCC.feed!(s, LM15.to_dict(e.delta)))
        end
        append!(shown, LMCC.finish!(s, response.finish_reason).events)
    catch
        return                                   # an unreadable reply: the re-ask will say so
    end
    texts = OrderedDict{String,String}()
    for e in shown
        get(e, "kind", nothing) == "field_delta" && e["field"] != "calls" && (texts[e["field"]] = get(texts, e["field"], "") * e["text"])
    end
    for (field, text) in texts
        emit_text(job.call, field, text)
    end
end

"The seconds to wait before re-sending after a transient error: the provider's advice, else backoff with jitter."
function backoff(err, attempt)
    after = err isa LM15.LM15Error ? (try
        err.retry_after
    catch
        nothing
    end) : nothing
    after isa Real && after > 0 && return Float64(after)
    min(30.0, 2.0^attempt) * (0.5 + rand())
end

"""
One request: through the router, re-sent after transient errors; streamed
when the call is watched. Each attempt is a `request` event and an exchange
(law 8); `request_hash` is lmcc's hash of the request as it renders it.
"""
function send(job::Job, request, request_hash; hit=nothing)
    call = job.call
    if hit !== nothing
        # a reply already known (the reply cache, or a turn being resumed): an exchange like any other
        begin_request!(call, job.model)
        exchange!(call, job.model, request, hit, time(), 0.0; cached=true, request_hash)
        watched(call) && replay(job, hit)
        return hit
    end
    retries = max(0, job.settings[:api_retries])
    streams = wants_pieces(call) && can_stream(job.router)
    attempt = 0
    while true
        check_cancelled(call)
        begin_request!(call, job.model)
        started = time()
        t0 = time_ns()
        elapsed() = (time_ns() - t0) / 1e9
        first_delta = nothing
        try
            response = if streams
                events = Any[]
                view = LMCC.stream(job.plan)
                viewing = true
                thinking_read = any(f -> f.purpose == "reasoning", job.plan.signature.fields)
                source = LM15.stream(job.router, request)
                try
                    for e in source
                        cancelled(call) && throw(Cancelled())
                        push!(events, e)
                        e isa LM15.StreamDeltaEvent || continue
                        first_delta === nothing && (first_delta = elapsed())
                        d = e.delta
                        d isa LM15.ThinkingDelta && !thinking_read && emit_thinking(call, d.text)
                        if viewing
                            # the view (lmcc's stream reader) never decides the call: a piece it
                            # cannot read stops the view, and the whole reply is read below
                            try
                                show_events(job, LMCC.feed!(view, LM15.to_dict(d)))
                            catch
                                viewing = false
                            end
                        end
                    end
                finally
                    applicable(close, source) && close(source)
                end
                if viewing
                    i = findlast(e -> e isa LM15.StreamEndEvent, events)
                    try
                        show_events(job, LMCC.finish!(view, i === nothing ? nothing : events[i].finish_reason).events)
                    catch
                    end
                end
                LM15.materialize_response(events, request)
            else
                r = LM15.complete(job.router, request)
                watched(call) && replay(job, r)
                r
            end
            exchange!(call, job.model, request, response, started, elapsed(); streamed=streams, first_delta, request_hash)
            return response
        catch err
            exchange!(call, job.model, request, nothing, started, elapsed(); error=err, streamed=streams, request_hash)
            cancelled(call) && throw(Cancelled())
            (LM15.retryable(err) && attempt < retries) || rethrow()
            wait = backoff(err, attempt)
            emit_retry(call, "the provider failed ($(error_type(err))); sending again in $(round(wait; digits=1)) s", wait)
            cancellable_sleep(call, wait)
            attempt += 1
        end
    end
end

asked_again(r::LMCC.Refusal) = r.code == "parse-truncated" ?
    "the reply was cut off; asking again with a larger token budget" : "the reply could not be read ($(r.hint)); asking again"

unreadable(r::LMCC.Refusal) = startswith(r.code, "parse-") || r.code == "format-read-error"

const LMCC_TRUNCATED_ADVICE = "; raise max_tokens or ask for less"

"""
    cut_off(refusal, response, limit) -> LMCC.Refusal

A `parse-truncated` refusal that says what happened (functions.md, "When the reply cannot be
read"): how much went to thinking, whose limit it was and whether it can be raised, and what
lm15 changed in the request. `limit`: the `max_tokens` this request set, or `nothing`.
"""
function cut_off(r::LMCC.Refusal, response, limit)
    hint = endswith(r.hint, LMCC_TRUNCATED_ADVICE) ? r.hint[1:end-length(LMCC_TRUNCATED_ADVICE)] : r.hint
    thought = something(response.usage.reasoning_tokens, 0)
    total = something(response.usage.output_tokens, 0)
    if thought > 0
        hint *= total >= thought ? "; the model spent $thought of its $total output tokens thinking" :
                                   "; the model spent $thought tokens thinking"
    end
    notes = response.adaptations
    i = findfirst(a -> a.field == "config.max_tokens" && a.action == "defaulted", notes)
    if limit !== nothing
        hint *= "; raise max_tokens (it was $limit) or ask for less"
    elseif i !== nothing
        hint *= "; no max_tokens was set, and lm15 sent $(notes[i].applied), the most it knows this model to allow: " *
                "lower the reasoning effort or ask for less"
    else
        hint *= "; no max_tokens was set, so the provider used its own maximum: lower the reasoning effort or ask for less"
    end
    other = ["$(a.field) $(a.action): $(a.reason)" for a in notes if a.field != "config.max_tokens"]
    isempty(other) || (hint *= " (lm15 adapted the request: " * join(other, "; ") * ")")
    return LMCC.Refusal(r.code, hint; fix=r.fix, partial=r.partial)
end

"""
One model call; after an unreadable reply, up to `retries` follow-ups that
send the reader's hint back. Before each send: the `request` hook (the escape
hatch), then a reply already known (a resumed turn's recorded one, or the
reply cache's); a reply is kept in the cache only once it was read.
"""
function complete_once(job::Job, rendered, responses)
    s = job.settings
    call = job.call
    request = LMCC.lm15_request(rendered; model=job.model, config=config_of(s))
    first_rendered = LMCC.request(rendered)
    first_hash = LMCC.sha256_of(first_rendered)
    rendered_request, request_hash = first_rendered, first_hash      # what the next exchange sends, and its hash
    budget = nothing
    retries = max(0, s[:retries])
    attempt = 0
    while true
        # the escape hatch: a plugin may replace the provider request; no one can rebuild that request, so its
        # exchange has no request_hash and the call is not replayable
        sent, replaced = plugin_request(request, call, job.name)
        sent_hash = replaced ? nothing : request_hash
        hit = recorded_reply(call, sent, s)
        replayed = hit !== nothing                       # a reply its turn recorded before: kept already
        flight = hit === nothing ? begin_flight(s, sent, call) : nothing
        flight === nothing || (hit = flight.hit)
        try
            response = send(job, sent, sent_hash; hit)
            push!(responses, response)
            replayed || note_reply(call, sent, response, s)     # a stored turn keeps every reply
            try
                reading = LMCC.read(job.plan, response)
                calls = get(reading.values, "calls", nothing)
                asks_tools = calls isa AbstractVector && !isempty(calls)   # a tool step: the answer comes later
                typed = asks_tools ? nothing : job.typed(reading.values)
                flight === nothing || keep_flight!(flight, response) # only a reply that was read is kept
                return (response, reading, typed)
            catch err
                err isa LMCC.Refusal || rethrow()
                flight === nothing || drop!(flight)                  # a kept reply that no longer reads is forgotten
                refusal = err
                # the budget this request set (functions.md: a cut reply is re-sent with twice it, only when one was set)
                limit = something(budget, setting(s, :max_tokens), Some(nothing))
                refusal.code == "parse-truncated" && (refusal = cut_off(refusal, response, limit))
                (attempt < retries && unreadable(refusal)) || throw(refusal)
                refusal.code == "parse-truncated" && limit === nothing && throw(refusal)   # nothing larger to give
                if refusal.code == "parse-truncated"
                    # asked again from the start, with a larger budget: the first request's messages, and its hash
                    budget = 2 * limit
                    request = LMCC.lm15_request(rendered; model=job.model, config=config_of(s; max_tokens=budget))
                    rendered_request, request_hash = first_rendered, first_hash
                else
                    words = "Your reply could not be read: $(refusal.hint). Reply again, in exactly the form the instructions give."
                    again = LM15.user(words)
                    request = LM15.Request(request; messages=(request.messages..., response.message, again))
                    # the request as lmcc writes it, the reply and the re-ask after its messages: what this exchange sent
                    rendered_request = LMCC.deepcopy_json(rendered_request)
                    rendered_request["messages"] = Any[rendered_request["messages"]..., LM15.to_dict(response.message),
                                                       LMCC.jobj("role" => "user", "parts" => Any[LMCC.textpart(words)])]
                    request_hash = LMCC.sha256_of(rendered_request)
                end
                emit_retry(job.call, asked_again(refusal), nothing)
                attempt += 1
            end
        finally
            flight === nothing || finish!(flight)
        end
    end
end

"Run one tool call; its result as the model sees it (text), or its error when `tool_errors` is `:report`."
function run_tool(tools, call, errors::Symbol)
    name = string(get(call, "name", ""))
    i = findfirst(t -> t.name == name, tools)
    if i === nothing
        errors === :raise && throw(ArgumentError("the model called unknown tool $(repr(name))"))
        return "error: there is no tool named $(repr(name))"
    end
    out = try
        invoke_tool(tools[i], something(get(call, "input", nothing), JObj()))
    catch err
        errors === :raise && rethrow()
        # what stops the call is never a tool's answer: a journal's barrier or scope (a call the tool made),
        # a closed stream, an interrupt
        unwrap(err) isa Union{JournalError,Cancelled,InterruptException,TurnWaiting,ConversationError,ApprovalError} && throw(unwrap(err))
        err = unwrap(err)
        return "error: $(error_type(err)): $(error_message(err))"
    end
    out isa AbstractString ? String(out) : LMCC.json_text(logvalue(out))
end

"Call the model (and run tools until it answers): the typed outputs, the turn, the replies."
function run_job(job::Job)
    plan = job.plan
    values = prepare_inputs(plan.signature, job.inputs)
    isempty(job.tools) || (values["tools"] = Any[tool_data(t) for t in job.tools])
    turn = LMCC.new_turn(plan, values)
    responses = Any[]
    steps = max(1, job.settings[:max_steps])
    for _ in 1:steps
        rendered = LMCC.render(plan, turn; turns=job.past)
        response, reading, typed = complete_once(job, rendered, responses)
        turn = LMCC.step(rendered, response)
        calls = get(reading.values, "calls", nothing)
        calls === nothing || append!(job.asked, calls)
        if calls === nothing || isempty(calls)
            turn = LMCC.finish(turn)
            return (typed, turn, responses, reading)
        end
        for c in calls
            output = one_tool!(job, c)
            turn = LMCC.tool(turn, string(c["id"]), output)
        end
    end
    throw(StepLimit("$(job.name): no answer after $steps model steps", turn))
end

"""
One tool call of the call in progress (contract/tools.md, contract/plugins.md):
numbered (its invocation), shown, given to the `tool_call` hooks (which may
change its input, block it, or ask a person: `approve =` is one of them),
kept before and after it runs when it changes things (a stored turn, a
required journal's barrier), run, its result given to the `tool_result`
hooks, and shown. A turn being resumed gets the result a tool that ran
before returned: no tool runs twice.
"""
function one_tool!(job::Job, c)
    call = job.call
    id, name = string(c["id"]), string(c["name"])
    input = something(get(c, "input", nothing), JObj())
    call.invocations += 1
    n = call.invocations
    i = findfirst(t -> t.name == name, job.tools)
    t = i === nothing ? nothing : job.tools[i]
    effects = t === nothing ? "reads" : effects_of(t)            # an unknown tool runs nothing
    asked = emit!(call, :tool_call, LMCC.jobj("id" => id, "name" => name, "input" => input, "invocation" => n))
    approval = Approval(call.id, n, id, name, input, effects, approval_path(call.site, name), call.site)
    tool_input, refused = plugin_tool_call(call, approval, job.settings)
    output = if refused !== nothing
        refused
    else
        run = call.turn_run
        known = run === nothing ? nothing : recorded_tool(run, call, approval)
        if known !== nothing
            known                                                  # a turn resumed: this tool ran before
        else
            frontier!(call.tree)
            if effects != "reads"
                # a required journal keeps the tool call before a tool that changes things runs
                t_log = call.tree
                if required(t_log)
                    confirmation(t_log, asked) === :confirmed ||
                        throw(barrier_error(asked; cause=journal_cause(t_log), store=t_log.journal.store))
                end
                run === nothing || tool_started!(run, call, with_input(approval, tool_input))
            end
            check_cancelled(call)
            out = with(TOOL_INVOCATION => n) do
                run_tool(job.tools, LMCC.jobj("id" => id, "name" => name, "input" => tool_input), Symbol(job.settings[:tool_errors]))
            end
            # the result as the model is shown it; a turn resumed later reuses it, hooks and all
            out = plugin_tool_result(call, approval, tool_input, out)
            run === nothing || tool_done!(run, call, approval, out)
            out
        end
    end
    emit!(call, :tool_result, LMCC.jobj("id" => id, "name" => name, "output" => output, "invocation" => n))
    output
end
