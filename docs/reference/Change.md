# Change { #functai.Change }

```{.python .no-run}
Change(
    instruction=None,
    sections=None,
    lm=None,
    settings=None,
    tools=None,
    keep=None,
    without=None,
    inputs=None,
    block=None,
    output=None,
    reminder=None,
    replace=None,
    stop=None,
)
```

What a hook changes. Each hook accepts some fields (HOOKS); a field it
does not accept refuses ``plugin-change``.

- ``instruction``: the instruction itself, replaced for this call (a
  mode's own system prompt) (``before_call``);
- ``sections``: text added to the instruction, in order (``before_call``,
  ``context``);
- ``lm``: the model for this call; ``settings``: lm15 settings for it
  (``reasoning``, ``temperature``, ``max_tokens``...) (``before_call``);
- ``tools``: the names of the tools offered to the model, from the
  function's own (``before_call``);
- ``keep``: the ids of the earlier turns shown (``context``); ``without``:
  fields left out of earlier turns, a list for every turn or ``{turn id:
  [names]}`` (``context``);
- ``inputs``: inputs replaced, by name (``turn_start``: the turn's;
  ``tool_call``: the tool's);
- ``block``: the tool call may not run, and why (``tool_call``);
- ``output``: the tool's result as the model is shown it: text, or a list
  of lm15 parts (a picture with a caption) (``tool_result``);
- ``reminder``: text the model is shown after this step's tool results,
  in this request and the ones after (``step``);
- ``replace``: earlier tool results, by invocation, shown as this from
  this request on: text, or a list of lm15 parts (``step``);
- ``stop``: the call stops here, and why (``before_call``, ``step``): it
  raises ``PluginError`` (``plugin-stopped``).