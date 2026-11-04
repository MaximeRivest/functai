# Long agents and pipelines

An agent that browses a site for two hundred steps, a pipeline whose steps run for minutes, a page that shows them while they run: tools that answer with pictures, steering a long tool loop, limits on what a run spends, notes from plain code (declared and checked), results that come later, the call log somewhere other than a folder, and what each program runs on (its *setup*), so a score belongs to exactly one thing that ran (contract/tools.md, plugins.md, streaming.md, calls.md).

```@setup agents
using FunctAI
```

## Tools that answer with pictures

A tool returns text, a value (shown as its JSON), a picture, a sound, a video or a document (an lm15 part), or a list holding one: a picture with its caption.

```julia
import LanguageModel15 as LM15

"A screenshot of a page, with its title."
look(page::String) = ["page $page", LM15.image(; path = screenshot(page))]

@ai tools = [tool(look; effects = :reads)] function browse(task::String)::String
    "Look at pages to do the task."
end
```

The model is shown the picture itself. A part read from a file is given its bytes, so the result is whole wherever it is read again (a resumed turn, another process). The `tool_result` event keeps `output` as the result's text, and adds `parts` (the whole result) when there is more than text; a `tool_result` plugin sees and may change the parts.

## Steering a long tool loop

Before each request after the first, `step` handlers may change what the next request offers and shows ([Plugins](plugins.md), *The step hook*). Two are built in:

```julia
agent = configure(browse; max_steps = 200,
                  plugins = [keep_pictures(3; at_once = 5),               # only the newest pictures, dropped five at a time
                             limits(cost = 3.0, seconds = 600, prices)])   # answer with what it has, at a limit
```

- [`keep_pictures`](@ref)`(last; at_once)`: pictures earlier than the newest `last` are shown as a short text. `at_once` drops several together: every change to an earlier result breaks a provider's prompt cache from that point on.
- [`limits`](@ref)`(; tokens, cost, seconds, prices, then = :answer)`: opt-in. Counted over the whole tree (every AI function in a program, every tool's own calls) from what providers answered in this process, plus what code said it spent ([`spend`](@ref)). With `then = :answer` a loop at a limit, or at its last step, is told to answer instead of failing with `StepLimit`; a call that would start past a limit stops ([`PluginError`](@ref) `plugin-stopped`). Exact or nothing: a model with no price, or a request whose counts do not settle its price, counts as reaching a cost limit. FunctAI ships no price list.

```@example agents
prices = Dict("claude-sonnet-5" => (input_per_million = 3, output_per_million = 15,
                                    cache_read_per_million = 0.3, cache_write_per_million = 3.75))
s = Spent([("claude-sonnet-5", Dict("input_tokens" => 200_000, "output_tokens" => 10_000, "total_tokens" => 210_000,
                                    "cache_read_tokens" => 150_000, "cache_write_tokens" => 0))], 30.0)
(s.tokens, FunctAI.cost(s, prices))
```

## Notes from plain code

[`note`](@ref) says what plain code did, so a page can show a pipeline's progress between model calls:

```julia
@program function walk(url::String)::String
    "Walk a site and say what is wrong with it."
    for page in pages(url)
        note("page captured", (url = page.url, title = page.title))
    end
    write_insights(url)
end
```

A note is an event, never a value of the call: the call log does not keep it, and a kept form whose content is not whole keeps its name without its data. A program may **declare** its notes; then each is checked where it is made (a name it does not declare, or data that does not fit what a program around it declares, throws [`InterfaceError`](@ref) `interface-note`), and those declared `outside` are shown to a caller who sees only the program's boundary (a served program's caller, the `:outside` view):

```@example agents
@program notes = ["stage" => Note(OneOf(:walking, :writing); latest = true, outside = true),
                  "page captured" => @NamedTuple{url::String, title::String}] function walk(url::String)::String
    "Walk a site."
    note("stage", "walking")
    "done"
end
FunctAI.interface(walk)["notes"]
```

An AI function's notes are its tools': `tool(f; notes = ["page captured" => Page])`. Notes are not in an interface's signature: declaring one never splits a program's records or ratings.

## The answer so far

[`answer_so_far`](@ref) reads a stream's answer by its shape: the text for a text answer, the JSON read so far for a record or a list. A choice shows once it is whole, never `"med"` on its way to `"medium"`; a number once a character after it ends it. A page that knows the program reads a call's `text` events the same way:

```@example agents
shape = Dict("type" => "object", "properties" => Dict("summary" => Dict("type" => "string"),
                                                      "severity" => Dict("enum" => ["low", "medium", "high"])))
(answer_so_far("{\"summary\": \"Checkout hi", shape), answer_so_far("{\"summary\": \"x\", \"severity\": \"med", shape))
```

## Results that come later

A tool that starts something ending elsewhere (a job, a cloud browser, a person's reply by mail) returns [`later`](@ref)`(key)`. In a conversation's turn, the step's other tool calls run, then the turn waits, saved ([`Waiting`](@ref), its `later` the results awaited); [`deliver!`](@ref)`(chat, key, result)` from any process records the result and the turn goes on, the tool never run again. A result delivered before the turn waits is used at once; delivering a key twice changes nothing (a webhook may be sent twice). Outside a conversation's turn nothing could wait for it: `later-unavailable`.

```julia
"Visit a site in a cloud browser; its report comes back later."
start_harvest(url::String) = later("harvest:$(start_job(url))")

chat = conversation(researcher, "visits"; store = "chats/")
chat("look at a.com")                              # Waiting: turn.later == [(key = "harvest:8f2c", …)]
# in the webhook's handler, any process:
deliver!(conversation(researcher, "visits"; store = "chats/"), "harvest:8f2c", report)   # the turn's answer
```

## The call log in a store of one's own

A machine that keeps no files (a serverless function, a container that goes away) logs calls to a **store**: any object with `push!(store, record)` and [`FunctAI.records`](@ref)`(store)`. Every reader takes it where it takes a folder (`calls`, `rated`, `rate`); `prune_calls` refuses one (the store prunes as it chooses). [`MemoryCalls`](@ref) keeps them in memory, for tests.

```julia
struct TableCalls; db; end
Base.push!(s::TableCalls, record) = (insert!(s.db, record); s)
FunctAI.records(s::TableCalls) = rows(s.db)
configure!(log_calls = TableCalls(db))
```

## What a program runs on

A [`version`](@ref) says what a program *is*. Its [`setup`](@ref) says what it runs on: the model each step asks, the settings that change what the model is asked or what comes back, its plugins with their config, its tools' code, and what it may reach. Two runs of one version on two models have two setups, so a score, a comparison or a release belongs to exactly one thing that ran; every call records its setup (`program.setup`), and where it ran (`placement`: the wall, the server address). Ratings still pool by interface.

```@example agents
@ai function mood(review::String)::OneOf(:happy, :unhappy, :mixed)
    "How does the customer feel about what they bought?"
end
with_settings(lm = "openai:gpt-4.1-mini") do
    setup_document(mood)
end
```

- **A tool's code** counts in the setup, not the version, so an agent loaded in another language keeps its version: Julia reads each of the tool's methods as written in its file (its lowered code when the file cannot be read), never the code it calls or what a closure captured; `tool(f; code_id = "v3")` says what identifies it beyond that.
- **Reach.** A tool says what its code reaches, `tool(f; needs = (domains = ["api.example.com"], secrets = ["EXAMPLE_API_KEY"]))` (names, never values), and a program its own (`@program needs = (…)`). The host says what wall it runs behind with the `wall` setting: none (reach is `"open"`: a run on a laptop is another setup than the same program behind a wall), a name (`"srt"`: made from the needs), or `(name, domains, secrets, limits)`. A program that needs what its wall does not let through is refused before its first call ([`ReachError`](@ref) `needs-unmet`).
- **What code spent**: [`spend`](@ref)`(0.05)` from a tool or a program's code; the call's record keeps the sum (`spent`), and `limits` counts it.
- **Saved folders** keep each program's setup where it was saved; [`setup_diff`](@ref)`(a, b)` says what differs, step by step, between two programs or two folders (a table: `DataFrame(setup_diff(a, b))`).
