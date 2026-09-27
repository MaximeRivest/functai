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
```

## Saving and loading

```@docs
FunctAI.save
FunctAI.load
FunctAI.to_manifest
FunctAI.from_manifest
LoadRefused
```

## Errors

```@docs
StepLimit
```

## Datasets

```@docs
FunctAI.tickets
FunctAI.field_notes
FunctAI.refunds
```
