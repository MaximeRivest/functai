# ProgramState { #functai.ProgramState }

```{.python .no-run}
ProgramState(instructions=None, demos=())
```

What an optimizer tunes in one AI function: its instruction and its
worked examples (``fn.state()``).

``instructions``: the instruction sent, or None for the one written from
the code (the docstring). ``demos``: the worked examples sent before
every call (lmcc turns, or ``{"inputs", "outputs"}`` dicts).
``to_dict()`` and ``ProgramState.from_dict(...)`` turn it into JSON data
and back; ``fn.load_state(state)`` makes a function use it.