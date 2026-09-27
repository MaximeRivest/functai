# Settings, models and sign-in

```@setup settings
using FunctAI
@enum Mood happy unhappy mixed
@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end
```

## Three places, one order

```julia
FunctAI.configure!(lm = "gpt-6-luna", temperature = 0)          # the session (the `!`: it changes something)

with_settings(lm = "claude-haiku-4-5", log_calls = true) do      # a block, and every task it starts
    mood.(reviews)
end

@ai lm = "gemini:gemini-3.8-flash" function mood(review::String)::Mood … end   # the function's own
fast = configure(mood; lm = "groq:openai/gpt-oss-120b")            # a copy with other settings
```

A function's own settings win over `with_settings`, which wins over `configure!`, which wins over the defaults. A setting given as `nothing` goes back to what the next layer says. An unknown setting is refused with the names that exist:

```@example settings
try
    configure(mood; modle = "gpt-6-luna")
catch err
    showerror(stdout, err)
end
```

The settings a function sets itself:

```@example settings
settings(configure(mood; temperature = 0, retries = 2))
```

## The settings

| Setting | What it does |
|:--|:--|
| `lm` | the model: `"gpt-6-luna"`, `"claude-haiku-4-5"`, `"gemini:gemini-3.8-flash"`, `"groq:openai/gpt-oss-120b"`, … |
| `temperature`, `max_tokens`, `top_p`, `stop`, `seed` | sampling, sent to the provider |
| `config` | any other lm15 `Config` field: `config = (reasoning = LM15.Reasoning(effort = "off"),)` |
| `adapter` | the layout: `:xml` (default, any model), `:chat` (DSPy's sections), `:json` (a schema the provider enforces) |
| `template` | a chat template: `[:system => "…", :turns, :user => "Review: {review}"]` |
| `reasoning` | `true`: the model writes its reasoning before the answer (Python's `module = "cot"`) |
| `include_name` | `false`: leave the function's name out of the instruction |
| `capabilities` | facts about the model that replace the table's: `Dict("native_reasoning" => false)` |
| `retries` | re-asks after an unreadable reply (default 1) |
| `api_retries` | re-sends after a transient provider error (default 3) |
| `max_steps` | model requests per tool loop (default 8) |
| `tool_errors` | `:report` (the model sees a tool's error, default) or `:raise` |
| `concurrency` | calls in flight over a column (default 8) |
| `log_calls` | the call log: a folder, `true` (the default folder) or `false` |
| `log_content` | `false`: log sizes, times and tokens, never values |
| `caller` | who is calling, added to the log: `Dict("kind" => "notebook")` |
| `router` | the lm15 router calls go through (a gateway, a fake in tests) |

## Models and keys

A model name is `provider:model` or a name lm15 recognizes (`"gpt-6-luna"`, `"claude-haiku-4-5"`). The key comes from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …). With no model named, FunctAI uses the first provider it finds a key for.

Subscriptions and saved keys go through lm15's sign-in, shared by every lm15 and FunctAI language:

```julia
FunctAI.login("claude")                 # a browser, or a link over SSH
FunctAI.login("openai"; key = "sk-…")   # save a key
FunctAI.logins()                        # what this machine is signed in to
@ai lm = "claude:claude-sonnet-5" function …
```

## What a model can do

What a model can do (native tool calls, reasoning, stop sequences, enforced JSON) is declared in the contract's table, never guessed, and decides how the layout is written:

```@example settings
model_capabilities("anthropic", "claude-haiku-4-5")
```

A setting a model doesn't take (a `temperature` of 0 on a reasoning model that runs only at 1) is left out of its requests, with one warning per provider, rather than failing every call.

## Layouts and templates

The same function, three ways of writing it for the model:

```@example settings
FunctAI.prompt(configure(mood; adapter = :chat), "It broke.")
```

A template is your own wording; `{name}` places an input:

```@example settings
FunctAI.prompt(configure(mood; template = [:system => "You read reviews.", :user => "Review: {review}"]), "It broke.")
```
