# Note { #functai.Note }

```{.python .no-run}
Note(type=None, *, desc=None, latest=False, outside=False)
```

A note a program declares (contract/programs.md, *Notes*): what its
data holds (a type, as a field's; ``functai.JSON`` or nothing: any JSON
value), words about it, whether only its latest value matters
(``latest=True``: a stage, a count so far), and whether a caller who sees
only the program's boundary may see it (``outside=True``).

    @functai.module(notes={"stage": functai.Note(Literal["walking", "writing"], latest=True, outside=True),
                           "page captured": Page})