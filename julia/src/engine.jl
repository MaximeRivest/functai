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
function send(job::Job, request, request_hash)
    call = job.call
    retries = max(0, job.settings[:api_retries])
    streams = watched(call) && can_stream(job.router)
    attempt = 0
    while true
        check_cancelled(call)
        call.requests += 1
        emit!(call, :request, LMCC.jobj("request" => call.requests, "model" => job.model))
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

"One model call; after an unreadable reply, up to `retries` follow-ups that send the reader's hint back."
function complete_once(job::Job, rendered, responses)
    s = job.settings
    request = LMCC.lm15_request(rendered; model=job.model, config=config_of(s))
    first_rendered = LMCC.request(rendered)
    first_hash = LMCC.sha256_of(first_rendered)
    rendered_request, request_hash = first_rendered, first_hash      # what the next exchange sends, and its hash
    budget = nothing
    retries = max(0, s[:retries])
    attempt = 0
    while true
        response = send(job, request, request_hash)
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
                # asked again from the start, with a larger budget: the first request's messages, and its hash
                budget = 2 * something(budget, setting(s, :max_tokens), 1024)
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
        unwrap(err) isa Union{JournalError,Cancelled,InterruptException} && throw(unwrap(err))
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
            id, name = string(c["id"]), string(c["name"])
            tool_called!(job.call, LMCC.jobj("id" => id, "name" => name, "input" => something(get(c, "input", nothing), JObj())))
            output = run_tool(job.tools, c, Symbol(job.settings[:tool_errors]))
            emit!(job.call, :tool_result, LMCC.jobj("id" => id, "name" => name, "output" => output))
            turn = LMCC.tool(turn, id, output)
        end
    end
    throw(StepLimit("$(job.name): no answer after $steps model steps", turn))
end
