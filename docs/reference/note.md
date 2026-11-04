# note { #functai.note }

```{.python .no-run}
note(name, data=None)
```

Say that something happened, from code inside a call (a module's body,
a tool): ``functai.note("screen captured", {"screen": 4, "key": "s4.jpg"})``.

Whatever watches the call sees a ``note`` event (a stream, an observer, a
journal: contract/streaming.md), in its place among the call's events.
``data`` is any JSON value (a record is given as its members); the kept
form of a call whose content is not whole keeps the note's ``name`` and
leaves ``data`` out. A note is not a value of the call: the call log does
not keep it. Outside any call it does nothing. A resumed turn does not
show again the notes it made before it stopped.

A program may declare its notes (``@module(notes=...)``,
``@functai.tool(notes=...)``: contract/programs.md, *Notes*). Then, inside
its calls, a note it does not declare, or whose data does not fit what a
program around it declares, raises ``InterfaceError`` (code
``interface-note``) here, whether or not anything watches.