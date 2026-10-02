# Long runs

Running a function over thousands of rows, or for weeks: replies kept and reused, a run that resumes by being run again, a progress line, a log kept small without losing what ratings need, a judge's evidence checked, and a second model for the answers the first is unsure of.

```@setup long
using FunctAI
```

## The reply cache

```julia
FunctAI.configure!(cache_replies = :disk)      # kept across runs and processes, in one SQLite file
FunctAI.configure!(cache_replies = true)       # this process's memory only
configure(mood; replicate = 2)                 # a third, independent answer to the same request
FunctAI.clear_cache(:disk)                     # forget them
```

`cache_replies` is off by default. On, a request already answered is answered again from the cache (an exchange with `cached: true`, `seconds: 0`), and the model is not paid twice.

- **Only a reply that was read is kept**: once lmcc read it and its values fit their types. An unreadable reply, a failed or cancelled request is never kept, so an interrupted run leaves nothing half-written: running it again sends only what has no kept reply. That is how a long run resumes.
- **One flight per request**: while one caller asks the model, a second caller of the same request waits and gets its reply, in this process and across processes (a claim with a 120-second lease in the same file).
- **`:disk`** is `~/.cache/functai/replies.sqlite`, the file Python uses too (format 1, contract/replies.md): a reply Python kept answers Julia's identical request. A folder or a `.sqlite` path names another; any `Dict` is a store (`cache_replies = Dict()`), and so is a [`FunctAI.ReplyStore`](@ref) of your own.
- **What may be written**: a call whose `log_content` keeps any field out of the log is never written to disk (the memory cache still serves it): a kept reply holds the whole request and reply.
- **`replicate`** is part of the key and nothing else: five samples of the same request on purpose are five replies, never one.

## A progress line

Over a column (`f.(xs)`, `map`), a line on stderr says rows done, failures, tokens and the time left, updated in place: on when stderr is a terminal (or a notebook), else with `progress = true`, off with `progress = false`.

## Pruning the call log

```julia
FunctAI.prune_calls("90d")      # (days = 12, calls = 3400, kept = 41)
```

Keeping the log small is deleting whole day folders. Before a day goes, every rated call in it is kept: the call, every call of its tree (a program's helpers), every call its `saw` names (the earlier turns a rated turn is asked again with) and their ratings are copied into one file at the folder's top level, which every reader reads. A row of `rated` made before pruning is made the same after. `keep_rated = false` deletes them too.

## Checking a judge's evidence

A judge written as an ordinary AI function (a score with the quotes it rests on) is only as good as its quotes; one that invents evidence is caught by checking each quote is really in the text:

```@example long
source = "The parcel left Leeds on Monday. It was delayed by snow."
quotes_found(source, ["“It was delayed by snow”", "It was lost"])
```

White space, case, curly and straight quotes, dashes, and a quote's own quotation marks and final punctuation do not count; any other difference does. Deterministic, and costs nothing.

## Escalation

```julia
@ai lm = "typesafe:jev-latest" escalate_to = "gpt-5.4" escalate_below = 0.8 function team(message::String)::Team
    "Which team should answer this customer message?"
end
```

When the first model is less sure of its answer than `escalate_below` (0.9 by default), `escalate_to` answers instead: a model name, a baked model, or another AI function (which follows only its own `escalate_to`). The call's stream shows a retry saying why, and its record says `escalated`. It needs a first model that measures its confidence (TypeSafe's Jev, a classifier): providers' chat models give an answer, not a probability, and the call says so rather than guess.

## The last requests

`inspect_history(5)` gives the last five requests of this process, with lm15's request and response (exactly what went to the provider and came back); `phistory(5)` prints them as a conversation. The last 500 are kept in memory; the call log keeps every call on disk.
