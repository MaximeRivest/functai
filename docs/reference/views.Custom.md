# views.Custom { #functai.views.Custom }

```{.python .no-run}
views.Custom(calls=None, shows=None)
```

A view of your own (contract/streaming.md, *Views*): what a served
program's callers see (``serve(program, view=...)``), or what you read of
a stream (``s.events(view=...)``)::

    thoughts = views.Custom(shows=lambda e: e if e["kind"] in ("thinking", "text") else None)

``calls(started)``: whether a call is shown, given its ``started`` event
(as JSON); default every call. ``shows(event)``: each event of a shown
call as the view shows it: the same, a copy with values left out, or
None; default the event. A call it shows always shows its ``started``,
``request``, ``retry``, ``done`` and ``failed`` (with the values
``shows`` leaves them), and what ``shows`` gives is cut to the keys the
contract knows. Never alter a value, nor invent one: a reader takes what
a view shows as what happened.