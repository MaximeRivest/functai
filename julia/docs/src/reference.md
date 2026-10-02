# Reference

Every exported name, and the few used qualified (`FunctAI.save`, …), by task. `?name` in the REPL shows the same text.

```@meta
CurrentModule = FunctAI
```

```@docs
FunctAI
```

## Writing AI functions

```@docs
@ai
AIFunction
OneOf
FunctAI.@ai_str
tool
AITool
@program
AIProgram
FunctAI.interface
FunctAI.interface_signature
InterfaceError
```

## Calling, and what a call returns

```@docs
predict(::AIFunction, ::Vararg)
Prediction
problems
render(::AIFunction, ::Vararg)
FunctAI.prompt
```

## What a function is

```@docs
instructions
demos
version
signature_id
FunctAI.signature
FunctAI.shape_of
FunctAI.julia_type
```

## Settings and models

```@docs
configure!
with_settings
configure(::AIFunction)
settings
model_capabilities
FunctAI.login
FunctAI.logins
FunctAI.logout
```

## Streaming

```@docs
stream(::AIFunction, ::Vararg)
AIStream
eachevent
Event
FunctAI.event_json
FunctAI.Position
FunctAI.text(::AIStream)
Cancelled
```

## Measuring

```@docs
evaluate
Evaluation
compare
exact_match
score_interval
FunctAI.scores
normalize_text
casefold
```

## Improving

```@docs
with_demos
with_instructions
labeled_few_shot
bootstrap_few_shot
random_search
instruction_search
gepa
```

## Models: fit, formulas, MLJ

```@docs
AIModel
AIModelFit
fit(::AIModel, ::Any, ::AbstractVector)
predict(::AIModelFit, ::Any)
```

## The call log

```@docs
rate
calls
rated
LogContentError
FunctAI.written_record
FunctAI.saw
FunctAI.keeps_saw
FunctAI.SawUnknown
```

## Saving and loading

```@docs
FunctAI.save
FunctAI.load
FunctAI.to_manifest
FunctAI.from_manifest
FunctAI.describe
LoadRefused
```

## Conversations

```@docs
conversation
Conversation
Turn
turns
FunctAI.turn
FunctAI.head
FunctAI.head!
continue_from
all_turns
last_turns
remember
earlier
render(::Conversation, ::Vararg)
predict(::Conversation, ::Vararg)
stream(::Conversation, ::Vararg)
merge!(::Conversation, ::AbstractVector, ::AIFunction)
wait_turn
stop!
FunctAI.calls_in
FunctAI.call_tree
MemoryConversations
FolderStore
FunctAI.ConversationStore
FunctAI.append_records!
FunctAI.read_records
```

## Tools that ask first

```@docs
Approval
approve!
deny!
resume!
abandon!
Waiting
ApprovalError
```

## Plugins

```@docs
Plugin
on!
Change
load_plugin
compaction
delegate
FunctAI.ask
FunctAI.entries
FunctAI.remember!
PluginError
```

## Serving and remote programs

```@docs
serve
remote
FunctAI.Service
FunctAI.handle
FunctAI.outside
FunctAI.RemoteError
```

## Long runs

```@docs
FunctAI.clear_cache
FunctAI.reply_key
FunctAI.ReplyStore
FunctAI.MemoryReplies
FunctAI.DiskReplies
quotes_found
prune_calls
train_test
FunctAI.earlier_of
inspect_history
phistory
```

## Baking

```@docs
FunctAI.bake_examples
FunctAI.export_examples
FunctAI.bake_entry
FunctAI.baked
FunctAI.BakeError
```

## Errors

```@docs
FunctAIError
ConversationError
StepLimit
```

## Datasets

```@docs
FunctAI.tickets
FunctAI.field_notes
FunctAI.refunds
```
