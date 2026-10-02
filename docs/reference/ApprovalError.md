# ApprovalError { #functai.ApprovalError }

```{.python .no-run}
ApprovalError(message, *, approval=None)
```

A tool call needs a person's answer and nobody can be asked (code
``approval-required``): a plain call, with a rule and no function to ask,
no stream to answer on and no conversation to wait in. The tool did not
run. ``approval`` is what would have been asked.