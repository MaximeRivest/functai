# A fake lm15 router: answers each request from a script, records every
# request. No network. `Whole(router)` has no stream: replies arrive whole.

using LM15

mutable struct FakeRouter
    replies::Vector{Any}
    responder::Any
    provider::String
    piece::Int
    requests::Vector{Any}
end
FakeRouter(replies=Any[]; responder=nothing, provider="openai", piece=3) = FakeRouter(Any[replies...], responder, provider, piece, Any[])

function FunctAI.route(r::FakeRouter, model::AbstractString)
    i = findfirst(':', model)
    i === nothing ? (r.provider, String(model)) : (model[1:i-1], model[i+1:end])
end

"A reply: text, or (text = …, calls = [(id, name, input)], finish = …)."
function reply!(r::FakeRouter, request)
    i = length(r.requests)
    push!(r.requests, request)
    spec = r.responder === nothing ? (isempty(r.replies) ? error("the fake router has no more replies") : popfirst!(r.replies)) :
           r.responder(request, i)
    spec isa Exception && throw(spec)
    spec = spec isa AbstractString ? (text=spec,) : spec
    parts = Any[]
    haskey(spec, :text) && push!(parts, LM15.TextPart(; text=spec.text))
    for (id, name, input) in get(spec, :calls, ())
        push!(parts, LM15.ToolCallPart(; id, name, input=Dict{String,Any}(String(k) => v for (k, v) in pairs(input))))
    end
    finish = get(spec, :finish, isempty(get(spec, :calls, ())) ? "stop" : "tool_call")
    LM15.Response(; model=request.model, message=LM15.Message(; role="assistant", parts=Tuple(parts)), finish_reason=finish,
                  usage=LM15.Usage(; input_tokens=10, output_tokens=5, reasoning_tokens=get(spec, :reasoning_tokens, nothing)))
end

LM15.complete(r::FakeRouter, request::LM15.Request) = reply!(r, request)

function LM15.stream(r::FakeRouter, request::LM15.Request)
    response = reply!(r, request)
    out = Any[]
    for e in LM15.response_to_events(response)
        if e isa LM15.StreamDeltaEvent && e.delta isa LM15.TextDelta && length(e.delta.text) > r.piece
            t = e.delta.text
            idx = collect(eachindex(t))
            for k in 1:r.piece:length(idx)
                piece = t[idx[k]:idx[min(k + r.piece - 1, length(idx))]]
                push!(out, LM15.StreamDeltaEvent(; delta=LM15.TextDelta(; text=piece, part_index=e.delta.part_index)))
            end
        else
            push!(out, e)
        end
    end
    out
end

struct Whole
    router::FakeRouter
end
FunctAI.route(w::Whole, model::AbstractString) = FunctAI.route(w.router, model)
LM15.complete(w::Whole, request::LM15.Request) = reply!(w.router, request)
