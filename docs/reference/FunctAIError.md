# FunctAIError { #functai.FunctAIError }

```{.python .no-run}
FunctAIError(code, message)
```

The base of every error FunctAI raises with a code: ``err.code``.

The code is a short, stable word for what went wrong
(``"interface-input"``, ``"approval-required"``, ``"turn-waiting"``...),
the same in every language FunctAI is written in, and the one the call
log records. Branch on it rather than on the message, which may change:

```{.python .no-run}
try:
    chat("Refund my order, please.")
except functai.FunctAIError as err:
    if err.code == "turn-waiting":
        ...                    # a person has to approve a tool call first
    else:
        raise
```

The subclasses say where the error comes from: ``InterfaceError``,
``LogContentError``, ``SawError``, ``EventRefused``, ``JournalError``,
``ConversationError``, ``Waiting``, ``ApprovalError``, ``ServeError``,
``PluginError``, and ``RemoteError`` (``from functai.remote import
RemoteError``). A reply the model wrote that cannot be read is lmcc's
``Refusal``, with lmcc's codes (see *When the model gets it wrong*).