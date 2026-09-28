# Streaming, tools and programs

## Streaming

A stream is the same call, watched while it is made: it asks the same thing, retries the same way, runs the same tools, logs the same line, and ends with the same value or error.

```julia
for piece in stream(haiku, "the first snow")     # the answer's text, piece by piece
    print(piece)
end

s = stream(support, "Where is order A-1042?")
for e in eachevent(s)                             # everything, in order
    println(e.seq, " ", e.kind)                   # :started, :request, :text, :thinking, :tool_call, :tool_result, :retry, :done, :failed
end
fetch(s)                                          # the typed answer, the same as calling
FunctAI.text(s)                                   # the answer so far (provisional)
close(s)                                          # cancel: the call ends with Cancelled

answer = stream(print, haiku, "the first snow")   # the do form: each piece to a function, then the answer
```

Each [`Event`](@ref) has `kind`, `call`, `function` and the fields of its kind (`e.text`, `e.field`, `e.name`, `e.input`, `e.output`, `e.value`, `e.error`, `e.request`), and says where it is in its call tree's log: `tree` (the outermost call's id), `writer` and `seq` (its [`FunctAI.Position`](@ref)), `after` (the event before it in what this reader is shown) and `at`. `FunctAI.event_json(e)` is its contract JSON (format 2), what a page or another process reads. Each request to the model is a `:request` event; it, and a retry, void the text shown so far. A reply that can't be read in pieces (the `:json` layout) is shown when it's complete; nothing is invented.

A tree has one log, numbered once: a stream opened on a call inside a program shows that call's events with the tree's numbers, its first `after` being `nothing`. Events are data another process can follow: `FunctAI.replay(events)` is what a watcher saw, and a [`FunctAI.Follower`](@ref) follows a log live, across a writer that took over after a restart (contract/streaming.md).

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
@program function reply(ticket::String; tone::String = "kind")::String
    "Answer a support ticket."
    t = triage(ticket)
    t.minutes > 60 ? escalate(ticket) : support(ticket)
end

reply("I was charged twice.")
reply.(tickets)                           # concurrent over a column
s = stream(reply, "I was charged twice.") # watch it and every call inside it
version(reply)                            # its code, and the version of every AI function it names
```

A program's version covers its own code, its interface, and the AI functions and programs it names; a plain Julia function it calls is not followed.

### Interfaces

Every program has an interface, the same JSON in every FunctAI language: what it takes and gives, so it can be described, checked and called from elsewhere without running it. `FunctAI.interface(reply)` shows it. A program's is read from its declaration: each argument is an input typed by its Julia type (an untyped argument, `Any`, or a type with no JSON form is *opaque*: never checked); an argument with a default may be left out (a default that is data, a literal or a constant, is written in the interface; any other default is Julia code; either way the code makes it anew on each call, as Julia does); the return type is its one output, `result`, or `outputs = (team = String, minutes = Int)` before `function` declares several, returned as a `NamedTuple`. Every input may also be given by name. Every call checks its inputs before the code runs and what it returns, and throws [`InterfaceError`](@ref) naming the field (`interface-input`, `interface-output`); a refused call is still a call, with its events and its record, and its code does not run. The error's message says what the value is (a text of 14 characters), never the value, since a message goes where the value may not be kept.

An AI function's interface is its definition's inputs and outputs. An input with a default (`tone::String = "kind"`) is optional: its default is written in the interface and sent to the model whenever the input is left out, so it is data, the same for every call: a literal or a constant (`const TONE = "kind"`). A default that uses another input or is computed (`at::Float64 = time()`) is refused when the function is defined, rather than computed once and silently shared; give such a value at each call. Each call gets its own copy of the default. A default that does not fit its type refuses the definition (`InterfaceError` `interface-malformed`).

A call made inside a program is a step of it, even on a task the program starts: the program's call ends once every call made inside it has (its log's last event is its own end). Work meant to outlive the call is started with `FunctAI.detached() do … end`, whose calls are trees of their own.

```julia
FunctAI.interface(reply)["inputs"][2]      # {"name": "tone", "shape": {"type": "string", "default": "kind"}, "type": "String", "optional": true}
FunctAI.interface_signature(reply)         # what its logged data looks like: calls with one signature pool
```

## Keeping a call tree's log: observers and journals

Events can be given, as they happen, to receivers set like any setting (a function's own, `with_settings`, `configure!`), whether or not anyone streams the call. They get the *kept* form: what `log_content` keeps, nothing more.

```julia
with_settings(observers = [e -> println(FunctAI.event_json(e))]) do     # watching: best effort, never in the way
    reply("I was charged twice.")
end
FunctAI.drain()                                                        # wait until the observers have had them all

store = FunctAI.MemoryStore()
FunctAI.configure!(journal = FunctAI.Journal(store; required = true))  # keeping: one journal per tree
```

Observers add up over the layers. Each gets its events in order on a task of its own (never from two places at once), so a slow, blocked or failing observer never slows a call: the call does not wait for it, and `FunctAI.drain()` waits for them when a script must. An observer that falls 10,000 events behind loses the rest, and sees the loss (an event's `after` names one it was not given); one that fails is warned about once and given nothing more.

A tree has one journal: a program's own setting cannot replace or remove the journal a host set around it, and nothing closer can weaken a required one (the tree is refused, [`JournalError`](@ref) `journal-policy`). A best-effort journal never makes a call wait: its writer sends on a task of its own. A required one makes it wait at its start, before each tool runs, and at its end, each time for its own events: when the start or a tool call is not kept, the call stops (`journal-barrier`, the tool does not run); when the end is not confirmed, the caller gets `JournalError` `journal-end` holding the outcome (`err.outcome.done` or `err.outcome.failed`) and what the journal met (`err.cause`), and `FunctAI.settle(err)` says later whether it was kept. A send the store has not answered in `Journal(store; timeout = 30)` seconds counts as no answer; a store whose own code fails (a `MethodError`, say) stops the writer rather than being sent again. A store is a `FunctAI.EventStore`: `keep!`, `claim!` and `events_after`, by the contract's store rules.
