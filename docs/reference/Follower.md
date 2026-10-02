# Follower { #functai.Follower }

```{.python .no-run}
Follower(live=False)
```

A reader that follows a form of logs live, one state per tree
(streaming.md, *Following a log*).

``receive(event)`` says what it did with each: ``"kept"`` (the next
event), ``"duplicate"``, ``"stale"`` (a writer the log has left behind),
``"rewind"`` (a later writer continued the log from an event this reader
holds: it drops what it has after that event, keeps the rest, and takes
this one), ``"loss"`` (events were lost: ``resume`` from a source) or
``"unknown-format"`` (it stops following). It keeps the events it holds,
since it may have to rewind; ``forget(tree)`` lets a tree go.

``live=True``: it follows a live form (the whole log, or a view of it that
may show values the kept form lacks), given each event by the process of
the event's writer; such a form it resumes in place only from the process
that gave it every event it holds. The kept form (the default) is the same
from every source.

## Methods

| Name | Description |
| --- | --- |
| [can_resume_from](#functai.Follower.can_resume_from) | Whether reading ``source`` after its last event keeps what it holds |
| [events](#functai.Follower.events) | The events it holds of a tree, in order (copies). |
| [forget](#functai.Follower.forget) | Let a tree go: its events and state (a long-lived reader forgets |
| [resume](#functai.Follower.resume) | Read a tree again from ``source`` (anything with ``read(tree, after)``): |
| [state](#functai.Follower.state) | The replay of what it holds of a tree: ``{"calls", "finished"}``. |

### can_resume_from { #functai.Follower.can_resume_from }

```{.python .no-run}
Follower.can_resume_from(tree, source)
```

Whether reading ``source`` after its last event keeps what it holds
true: the source can give every event it holds, as it holds it
(streaming.md, *Resuming*). ``source.writer`` is the writer whose
process it is, or None for a store.

### events { #functai.Follower.events }

```{.python .no-run}
Follower.events(tree)
```

The events it holds of a tree, in order (copies).

### forget { #functai.Follower.forget }

```{.python .no-run}
Follower.forget(tree)
```

Let a tree go: its events and state (a long-lived reader forgets
the trees it is done with).

### resume { #functai.Follower.resume }

```{.python .no-run}
Follower.resume(tree, source)
```

Read a tree again from ``source`` (anything with ``read(tree, after)``):
after its last event when it may resume in place, from the beginning
when it may not or the source says ``event-unknown``. Returns its reads,
each ``(after, the events read or the EventRefused)``. It stops at an
event of a format it does not know, as when following.

### state { #functai.Follower.state }

```{.python .no-run}
Follower.state(tree)
```

The replay of what it holds of a tree: ``{"calls", "finished"}``.