# Reference: a call tree's log

A call tree's events as data (contract/streaming.md, format 2): replaying a
form of a log, following it live across writers, its kept form, and keeping
it while it is written (journals, stores). See the guide *Streaming, tools
and programs* for how a call makes them.

```@meta
CurrentModule = FunctAI
```


```@docs
FunctAI.replay
FunctAI.replay!
FunctAI.LogState
FunctAI.CallState
FunctAI.resume
FunctAI.Follower
FunctAI.receive!
FunctAI.recover!
FunctAI.kept_event
FunctAI.kept_log
FunctAI.Journal
JournalError
FunctAI.MemoryStore
FunctAI.claim!
FunctAI.keep!
FunctAI.events_after
FunctAI.settle
FunctAI.StoreRefusal
FunctAI.LogWriter
FunctAI.confirmation
```

