# docments { #functai.docments }

`docments`

Docments: documentation harvested from code (inline comments, docstrings),
plus ``flexiclass`` and the ``UNSET`` sentinel.

Nothing here talks to a model. ``functai.signature`` uses these helpers to
turn comments into instructions and field descriptions.

## Functions

| Name | Description |
| --- | --- |
| [docments](#functai.docments.docments) | The documentation of a function's parameters and return, read from its comments. |
| [docstring](#functai.docments.docstring) | Get cleaned docstring for functions and classes. |
| [extract_docstrings](#functai.docments.extract_docstrings) | Return mapping {name: (docstring, paramlist)} for top-level symbols in code. |
| [flexiclass](#functai.docments.flexiclass) | Make a plain annotated class a dataclass, as ``@ai`` does for types. |
| [get_dataclass_source](#functai.docments.get_dataclass_source) | Get source code for dataclass s. |
| [get_source](#functai.docments.get_source) | Get source code for string, function object, class, or dataclass. |
| [isdataclass](#functai.docments.isdataclass) | Check if s is a dataclass *class* (not an instance). |
| [parse_docstring](#functai.docments.parse_docstring) | Split a numpy-style docstring into its parts. |
| [sig2str](#functai.docments.sig2str) | Generate a function signature string with inline docments comments. |

### docments { #functai.docments.docments }

```{.python .no-run}
docments.docments(
    elt,
    full=False,
    args_kwargs=False,
    returns=True,
    eval_str=False,
)
```

The documentation of a function's parameters and return, read from its comments.


Generate comment docs for functions or classes.

For functions: returns {param_name: comment, 'return': comment?}
For classes:   returns {field_name: comment}
If full=True, each value becomes {'anno': ..., 'default': ..., 'docment': ...}.

### docstring { #functai.docments.docstring }

```{.python .no-run}
docments.docstring(sym)
```

Get cleaned docstring for functions and classes.

### extract_docstrings { #functai.docments.extract_docstrings }

```{.python .no-run}
docments.extract_docstrings(code)
```

Return mapping {name: (docstring, paramlist)} for top-level symbols in code.

### flexiclass { #functai.docments.flexiclass }

```{.python .no-run}
docments.flexiclass(cls)
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

#### Returns {.doc-section .doc-section-returns}

| Name   | Type      | Description                                       |
|--------|-----------|---------------------------------------------------|
|        | dataclass | The same class object, mutated to be a dataclass. |

### get_dataclass_source { #functai.docments.get_dataclass_source }

```{.python .no-run}
docments.get_dataclass_source(s)
```

Get source code for dataclass s.

### get_source { #functai.docments.get_source }

```{.python .no-run}
docments.get_source(s)
```

Get source code for string, function object, class, or dataclass.

### isdataclass { #functai.docments.isdataclass }

```{.python .no-run}
docments.isdataclass(s)
```

Check if s is a dataclass *class* (not an instance).

### parse_docstring { #functai.docments.parse_docstring }

```{.python .no-run}
docments.parse_docstring(sym)
```

Split a numpy-style docstring into its parts.


Parse a subset of numpy-style docstrings:

####   Parameters {.doc-section .doc-section---parameters}

  name : type
      description...

####   Returns {.doc-section .doc-section---returns}

  type
      description...
Returns dict with 'param:<name>' and 'return' keys when found.

### sig2str { #functai.docments.sig2str }

```{.python .no-run}
docments.sig2str(func)
```

Generate a function signature string with inline docments comments.