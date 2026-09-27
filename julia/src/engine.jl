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
    name === :answer && return getproperty(getfield(p, :outputs), Symbol(getfield(p, :answer_name)))
    name === :response && return isempty(getfield(p, :responses)) ? nothing : last(getfield(p, :responses))
    name === :attempts && return length(getfield(p, :responses))
    getfield(p, name)
end
Base.propertynames(::Prediction) = (:value, :answer, :outputs, :call, :turn, :response, :responses, :attempts, :repairs, :probabilities)
function Base.show(io::IO, p::Prediction)
    print(io, "Prediction(")
    show(io, p.value)
    print(io, "; call = \"", p.call, "\")")
end

# ------------------------------------------------------------------ watching

"What watches a call as it is made (a stream); the engine tells it what happens."
abstract type Watch end
const WATCHING = ScopedValue{Union{Nothing,Watch}}(nothing)
watch_closed(::Nothing) = false
watch_check(::Nothing) = nothing
on_text(::Nothing, call, field, text) = nothing
on_thinking(::Nothing, call, text) = nothing
on_tool_call(::Nothing, call, id, name, input) = nothing
on_tool_result(::Nothing, call, id, name, output) = nothing
on_retry(::Nothing, call, reason, wait) = nothing

"Whether calls through `router` can be streamed: a method of `LM15.stream(router, request)`."
can_stream(router) = hasmethod(LM15.stream, Tuple{typeof(router),LM15.Request})

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
    watch::Union{Nothing,Watch}
    typed::Function                  # the reply's values (JSON) → typed, or a parse-value refusal
end

"Sleep `seconds`, waking to stop when the watching stream closes."
function cancellable_sleep(watch, seconds)
    deadline = time() + seconds
    while time() < deadline
        watch_closed(watch) && throw(Cancelled())
        sleep(min(0.1, max(0.0, deadline - time())))
    end
end

"Only the fields a stream shows: an output's text (tool calls are shown whole)."
function show_events(job::Job, batch)
    for ev in batch
        get(ev, "kind", nothing) == "field_delta" && ev["field"] != "calls" && on_text(job.watch, job.call, ev["field"], ev["text"])
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
        on_text(job.watch, job.call, field, text)
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

"One request: through the router, re-sent after transient errors; streamed when watched."
function send(job::Job, request)
    call, watch = job.call, job.watch
    retries = max(0, job.settings[:api_retries])
    streams = watch !== nothing && can_stream(job.router)
    attempt = 0
    while true
        watch_check(watch)
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
                        watch_closed(watch) && throw(Cancelled())
                        push!(events, e)
                        e isa LM15.StreamDeltaEvent || continue
                        first_delta === nothing && (first_delta = elapsed())
                        d = e.delta
                        d isa LM15.ThinkingDelta && !thinking_read && !isempty(d.text) && on_thinking(watch, call, d.text)
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
                watch === nothing || replay(job, r)
                r
            end
            exchange!(call, job.model, request, response, started, elapsed(); streamed=streams, first_delta)
            return response
        catch err
            exchange!(call, job.model, request, nothing, started, elapsed(); error=err, streamed=streams)
            watch_closed(watch) && throw(Cancelled())
            (LM15.retryable(err) && attempt < retries) || rethrow()
            wait = backoff(err, attempt)
            on_retry(watch, call, "the provider failed ($(error_type(err))); sending again in $(round(wait; digits=1)) s", wait)
            cancellable_sleep(watch, wait)
            attempt += 1
        end
    end
end

asked_again(r::LMCC.Refusal) = r.code == "parse-truncated" ?
    "the reply was cut off; asking again with a larger token budget" : "the reply could not be read ($(r.hint)); asking again"

unreadable(r::LMCC.Refusal) = startswith(r.code, "parse-") || r.code == "format-read-error"

"One model call; after an unreadable reply, up to `retries` follow-ups that send the reader's hint back."
function complete_once(job::Job, rendered, responses)
    s = job.settings
    request = LMCC.lm15_request(rendered; model=job.model, config=config_of(s))
    budget = nothing
    retries = max(0, s[:retries])
    attempt = 0
    while true
        response = send(job, request)
        push!(responses, response)
        try
            reading = LMCC.read(job.plan, response)
            calls = get(reading.values, "calls", nothing)
            asks_tools = calls isa AbstractVector && !isempty(calls)   # a tool step: the answer comes later
            return (response, reading, asks_tools ? nothing : job.typed(reading.values))
        catch err
            err isa LMCC.Refusal || rethrow()
            refusal = err
            thought = response.usage.reasoning_tokens
            if refusal.code == "parse-truncated" && thought !== nothing && thought > 0
                refusal = LMCC.Refusal(refusal.code, "$(refusal.hint) (the model spent $thought of its tokens thinking first; raise max_tokens)";
                                       fix=refusal.fix, partial=refusal.partial)
            end
            (attempt < retries && unreadable(refusal)) || throw(refusal)
            if refusal.code == "parse-truncated"
                budget = 2 * something(budget, setting(s, :max_tokens), 1024)
                request = LMCC.lm15_request(rendered; model=job.model, config=config_of(s; max_tokens=budget))
            else
                again = LM15.user("Your reply could not be read: $(refusal.hint). Reply again, in exactly the form the instructions give.")
                request = LM15.Request(request; messages=(request.messages..., response.message, again))
            end
            on_retry(job.watch, job.call, asked_again(refusal), nothing)
            attempt += 1
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
        if calls === nothing || isempty(calls)
            turn = LMCC.finish(turn)
            return (typed, turn, responses, reading)
        end
        for c in calls
            id, name = string(c["id"]), string(c["name"])
            on_tool_call(job.watch, job.call, id, name, something(get(c, "input", nothing), JObj()))
            output = run_tool(job.tools, c, Symbol(job.settings[:tool_errors]))
            on_tool_result(job.watch, job.call, id, name, output)
            turn = LMCC.tool(turn, id, output)
        end
    end
    throw(StepLimit("$(job.name): no answer after $steps model steps", turn))
end
