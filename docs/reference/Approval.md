# Approval { #functai.Approval }

```{.python .no-run}
Approval(
    call,
    invocation,
    id,
    name,
    input,
    effects,
    path,
    site='',
    plugin='approval',
    question=None,
)
```

One tool call waiting for a person's answer.

``call``: the id of the AI function's call that asked for it;
``invocation``: the tool call's number in that call; ``id``: the id the
model gave it; ``name`` and ``input``: the tool and what it would be
given; ``effects``: what the tool says it does; ``path``: where it is,
by names (``support/answer/refund``), for rules written before any call.