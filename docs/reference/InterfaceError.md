# InterfaceError { #functai.InterfaceError }

```{.python .no-run}
InterfaceError(code, field, message)
```

A program's interface refused what it was given or what it gave back.

``code`` is ``"interface-input"`` (a call given what the interface does
not take), ``"interface-output"`` (the code returned what the interface
does not give), ``"interface-note"`` (the code made a note its program
does not declare, or whose data does not fit the declaration: contract
programs.md, *Notes*) or ``"interface-malformed"`` (the interface itself
breaks the rules, when the program is defined or read from a folder).
``field`` names the field (or note) at fault, or is None when the fault
is no field's.

It is a ``TypeError`` (as Python's own wrong arguments are) and a
``ValueError`` (a value that does not fit).