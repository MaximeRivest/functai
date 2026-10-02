# Conversations and memory

An AI function remembers nothing: each call is on its own. A **conversation** is a program's calls that remember each other. Each call is a **turn**, shown the turns before it; the memory belongs to the conversation, never to the function, which stays the same and callable on its own (contract/conversations.md).

```@setup conversations
using FunctAI
```

## A conversation

```julia
@ai function tutor(message::String)::String
    "Tutor a student in arithmetic, one small step at a time."
end

chat = conversation(tutor, "alex"; store = "tutoring/")
chat("Hi, I'm Alex.")
chat("What is 1/2 + 1/3?")          # sees the first turn
chat("Is it 5/6?")                  # sees both
```

A conversation is called like its program: the same inputs, the same answer. `predict(chat, …)` gives the whole call, `stream(chat, …)` watches it; a stream's `s.turn` is known at once, because the turn is saved before the model is asked.

- **`id`**: letters, digits, `.`, `_`, `-` (a file name everywhere). The same id in the same store opens the same conversation: the line above, run again tomorrow, reopens Alex's. Leave it out for a new one.
- **`store`**: `nothing` (this process's memory, the default), a folder (a [`FolderStore`](@ref): files, locked across processes, flushed before each append returns), `true` (the default folder, `~/.local/share/functai/conversations`), or a store of your own (a [`FunctAI.ConversationStore`](@ref) with `append_records!` and `read_records`).
- Other keywords are settings for every turn (`lm`, `approve`, `plugins`, …); a setting given with a turn's inputs is that turn's only: `chat("Why?"; lm = "claude-sonnet-4-5")`.

A folder store is the same files Python's `FolderStore` writes: a conversation started in Python continues here, and the other way round.

## Turns

```julia
ts = turns(chat)                    # the turns from the first to the head, in order
t = last(ts)
t.result, t.inputs, t.outputs       # the answer (typed), and every value by name
t.state                             # "done", "failed", "stopped", "running", "waiting", "interrupted", "abandoned"
t.saw                               # the earlier turns this answer was based on
t.usage                             # tokens, over every call inside the turn
t.model                             # who answered
```

Every turn records what it was shown (`saw`), so an answer can be asked again exactly as it was. The record does not grow with the conversation: a turn that saw what its parent saw, then its parent, says so in two entries.

## What the model sees

Every earlier turn, by default: running out of the model's context, and being told, is better than a model that silently misses what was said.

```julia
chat = conversation(tutor, "alex"; context = last_turns(10))                         # the last ten
qa = conversation(reader, "paper"; context = all_turns(without = ["document"]))      # earlier turns without a bulky input
```

`render(chat, "Is it 5/6?")` is the exact request the next turn would send, nothing sent or recorded.

## Branches

Nothing is ever deleted. Continuing from an earlier turn makes a **branch**:

```julia
again = continue_from(chat, turns(chat)[1])     # after the first turn
again("What is 2/3 + 1/6?")                     # a new branch; the old one is still there
turns(chat; all = true)                         # every turn of every branch
FunctAI.head!(chat, turns(again)[2])            # make that turn the head, for everyone who opens it
```

A conversation opened by id follows its head; a view made by `continue_from` follows its own branch. A **merge** makes one turn from several branches, by another AI function:

```julia
x = stream(continue_from(chat, t), "Explain with pizza."); y = stream(continue_from(chat, t), "Explain with money.")
best = merge!(continue_from(chat, t), [x.turn, y.turn], pick_the_clearest)
```

The merged answer becomes this program's turn (`made_by` names the function that made it, `reads` the branches); its rating belongs to that function's call.

## Two sends at once, stopping, a process that died

- **Two sends at once** queue (the default `sends = :queue`: the second waits, then continues from the first), refuse (`:refuse`: [`ConversationError`](@ref) `conversation-busy`), or branch (`:branch`: beside it).
- **The same `request_id` twice is one turn**: a double click gets the first turn, its result once it ends.
- **Stopping from anywhere**: `stop!(chat, t)` appends a `stop` record; the process running the turn sees it within a second, and the turn ends `stopped` (its stream throws `Cancelled`).
- **A process that died**: a running turn renews a lease every 10 seconds. A turn whose lease ran out is `interrupted`; [`resume!`](@ref) goes on with it in this process.

## A program's conversation: helpers remember only when told

A `@program`'s turns remember each other, but the AI functions it calls start fresh at every call unless the conversation says otherwise:

```julia
@program function support(message::String)::String
    "Answer the customer."
    answer(message, topic(message))
end

chat = conversation(support, "ana"; remembers = Dict(answer => :conversation))
```

`:conversation` shows `answer` its own earlier calls on this branch, in earlier turns and this one; `:turn`, its earlier calls in this turn only; [`remember`](@ref)`(:conversation; steps = true)` with their tool calls and results. [`earlier`](@ref)`()` is the conversation so far as data (one row per earlier turn), for a helper that takes it as an input:

```julia
@ai function handoff(conversation::Vector{Dict{String,Any}})::String
    "Summarize this support conversation for the person who takes it over."
end
@program function support(message::String)::String
    "Answer the customer."
    topic(message) == "other" && notify_staff(handoff(FunctAI.earlier()))
    answer(message, topic(message))
end
```

A conversation used inside another conversation's turn is refused (`conversation-nested`), unless the outer one declares it: `remembers = Dict(inner => :own)`.

## What a conversation refuses

- `conversation-content`: a store that keeps records, for a program a `log_content` setting keeps values of out of the log. The host said never keep them, and a conversation must remember them: it refuses rather than forgets. Keep it in memory, or let the store keep those fields.
- `conversation-opaque`: a program with an untyped (opaque) input or output: a turn is kept as data.
- `conversation-signature`: the program now writes an output the earlier turns lack, or a field changed or went away. Turning reasoning on (or adding a first tool) goes on with `earlier_without = ["reasoning"]`: earlier turns are shown without it, and nothing is rewritten. The model may change freely: each turn records who answered.

## Learning from rated turns

A rated turn is a row of [`rated`](@ref) like any call, with what it was shown: `earlier` (its earlier turns, as data), `conversation` (its id) and, for a program's turn, `helpers` (what each helper was shown). [`evaluate`](@ref) and the optimizers ask each such row again with its own earlier turns, recording nothing in any conversation; the optimizers never take one as a worked example, since it answers its conversation. Keep a conversation on one side of a split, or the test measures memory:

```@example conversations
rows = [(message = "1/2 + 1/3?", conversation = "alex"), (message = "Is it 5/6?", conversation = "alex"),
        (message = "2 + 2?", conversation = "ben"), (message = "Hi", conversation = missing)]
train, test = train_test(rows; test = 0.34)
test
```
