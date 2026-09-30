# earlier { #functai.earlier }

```{.python .no-run}
earlier()
```

The conversation so far, as data: inside a module's turn, one row per
earlier turn it is shown (its inputs and outputs by name); ``[]`` outside
a conversation.

For a helper that declares an input for it:

```{.python .no-run}
@ai
def handoff(conversation: list[dict[str, str]]) -> str:
    """Summarize this support conversation for the person who takes it over."""

@module
def support(message: str) -> str:
    if topic(message) == "other":
        notify_staff(handoff(functai.earlier()))
        ...
```