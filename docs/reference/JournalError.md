# JournalError { #functai.JournalError }

```{.python .no-run}
JournalError(
    code,
    message,
    *,
    outcome=None,
    event=None,
    journal=None,
    store=None,
    tree=None,
)
```

A journal kept the call from going on, or could not confirm its end.

``code`` is one of:

- ``"journal-policy"``: the settings around the call break the journal
  policy (a program's own setting replacing or removing a host's
  journal; a closer layer replacing, weakening or removing a required
  one). Raised before the call runs.
- ``"journal-scope"``: a required journal set only around a call inside
  a tree (it cannot keep part of a tree).
- ``"journal-barrier"``: a required journal did not confirm the call's
  start, or a tool call, so the code or the tool did not run.
- ``"journal-end"``: a required journal did not confirm the call's end.
  The call itself ended as ``outcome`` says (its value, or its error):
  the journal changes nothing of that. ``journal`` says ``"refused"``
  (it did not keep the end) or ``"unknown"`` (no answer came: it may
  have), ``event`` names the end by its position. ``settle()`` asks
  the journal which it was.

## Methods

| Name | Description |
| --- | --- |
| [settle](#functai.JournalError.settle) | For ``journal-end`` with ``journal == "unknown"``: read the journal |

### settle { #functai.JournalError.settle }

```{.python .no-run}
JournalError.settle(claim=False)
```

For ``journal-end`` with ``journal == "unknown"``: read the journal
and say what became of the end, ``"kept"``, ``"not-kept"`` (the log is
unfinished) or ``"another-end"`` (another writer ended the log, so this
outcome is not the log's).

``"not-kept"`` may still change while the writer keeps sending. With
``claim=True`` the log is claimed first, which fences the writer, so
the answer is final; the log then waits for whoever ends it.