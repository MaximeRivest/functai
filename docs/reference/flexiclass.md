# flexiclass { #functai.flexiclass }

```{.python .no-run}
flexiclass(cls)
```

Make a plain annotated class a dataclass, as ``@ai`` does for types.


Convert `cls` to a dataclass IN PLACE, giving UNSET defaults to
any annotated field that doesn't already have a default.

Usages:
    @flexiclass
    class Person: name: str; age: int; city: str = "Unknown"

    # or
    class Person: ...
    flexiclass(Person)

## Returns {.doc-section .doc-section-returns}

| Name   | Type      | Description                                       |
|--------|-----------|---------------------------------------------------|
|        | dataclass | The same class object, mutated to be a dataclass. |