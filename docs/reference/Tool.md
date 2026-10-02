# Tool { #functai.Tool }

```{.python .no-run}
Tool(fn, *, effects=None, name=None, description=None)
```

A function the model may call, with what it does to the world. Made by
``functai.tool``.

``effects`` is ``"reads"`` (it only looks), ``"changes"`` (it writes,
sends or pays) or None (it says nothing, which every approval rule treats
as ``"changes"``). Called directly, it is the function; its name and
docstring (or the ``name`` and ``description`` given) are what the
model is told.