# Streaming, tools and programs

## Streaming

A stream is the same call, watched while it is made: it asks the same thing, retries the same way, runs the same tools, logs the same line, and ends with the same value or error.

```julia
for piece in stream(haiku, "the first snow")     # the answer's text, piece by piece
    print(piece)
end

s = stream(support, "Where is order A-1042?")
for e in eachevent(s)                             # everything, in order
    println(e.kind)                               # :started, :text, :thinking, :tool_call, :tool_result, :retry, :done, :failed
end
fetch(s)                                          # the typed answer, the same as calling
FunctAI.text(s)                                   # the answer so far (provisional)
close(s)                                          # cancel: the call ends with Cancelled

answer = stream(print, haiku, "the first snow")   # the do form: each piece to a function, then the answer
```

Each [`Event`](@ref) has `kind`, `call`, `function` and the fields of its kind (`e.text`, `e.field`, `e.name`, `e.input`, `e.output`, `e.value`, `e.error`); `FunctAI.event_json(e)` is its contract JSON. A retry voids the text shown since the previous one. A reply that can't be read in pieces (the `:json` layout) is shown when it's complete; nothing is invented.

## Tools

A tool is a Julia function the model may ask for. FunctAI runs it and sends the result back; the model then answers, or asks again, up to `max_steps` requests.

```julia
"Look up an order's delivery status by its number, like A-1042."
lookup_order(order::String) = …

@ai tools = [lookup_order] function support(message::String)::String
    "Answer the customer, looking up their order."
end
```

The docstring is the tool's description; its one method's argument names and types are its parameters (typed like an AI function's inputs). For another description or a hand-written schema: `tool(f; name, description, parameters)`. A tool that throws is reported to the model as `error: <type>: <message>` (`tool_errors = :raise` throws instead). The tool's name and result are in the call log, with the request that asked for it.

## Programs

A program is ordinary Julia code that calls AI functions, followed as one call: in the call log it is the parent of every call it made, and a stream of it shows them all.

```julia
@program function reply(ticket::String)
    t = triage(ticket)
    t.minutes > 60 ? escalate(ticket) : support(ticket)
end

reply("I was charged twice.")
reply.(tickets)                           # concurrent over a column
s = stream(reply, "I was charged twice.") # watch it and every call inside it
version(reply)                            # its code, and the version of every AI function it names
```

A program's version covers its own code and the AI functions and programs it names; a plain Julia function it calls is not followed.
