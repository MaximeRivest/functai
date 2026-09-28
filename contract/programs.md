# Programs and their interfaces

A **program** is an AI function or a module (code that calls AI
functions). Every program has an **interface**: the inputs it takes and
the outputs it gives, named and typed as JSON. The interface is what lets
a program be described, checked, served, conversed with and called from
another language without running it, and what its calls are recorded
against. An AI function's interface comes from its definition; a module
declares its own, because its code cannot be read for one in every
language.

`cases/programs/*.json` pin the interface's signature, which interfaces
are refused, and how values are checked against one.
`schema/saved.schema.json` (`$defs/interface`) checks its form.

## The interface

```json
{"description": "Answer a customer's message.",
 "inputs": [{"name": "message", "shape": {"type": "string"}},
            {"name": "tone", "shape": {"type": "string", "default": "kind"}, "optional": true}],
 "outputs": [{"name": "result", "shape": {"type": "string"}}]}
```

| key | meaning |
|---|---|
| `description` | what the program does, in words (Python: the docstring). May be empty. |
| `inputs` | in order, each field `{name, shape, desc?, type?, opaque?, optional?}`. |
| `outputs` | in order, at least one, each field `{name, shape, desc?, type?, opaque?}`. The **last is the answer** (`program.answer` in the call log). A program with one output usually names it `result`. |

A field:

| key | meaning |
|---|---|
| `name` | an ASCII identifier, used once among the program's inputs and outputs. |
| `shape` | JSON Schema (draft 2020-12), as [functions.md](functions.md) *Shapes* says. `{}` is any JSON value. A `default` in an input's shape is the value it takes when it is left out, and must fit the shape. |
| `desc` | words about the field. |
| `type` | the host language's name for the type (`str`, `pd.DataFrame`), for people; never compared across languages. |
| `opaque` | `true` when the field's values have no JSON form in the language that declared it (a data frame, a file handle, an unannotated Python argument). Its shape is `{}`. Its values are never checked, and are written in the log as descriptions (`$type`, `$repr`), not as data: a boundary that needs data (serving, a conversation store: stages 2 and 3) cannot carry the field, and says so before any call. |
| `optional` | inputs only: `true` when a caller may leave the input out (*Checking values*). |

**Its signature** is lmcc's signature fingerprint (kernel §3a) of the
fields: `"sha256:"` and the SHA-256 of the canonical JSON of the list of
the inputs then the outputs, each `{"direction", "name", "purpose":
"plain", "shape", "type": ""}`. It says what the program's data looks
like: calls of two programs with the same signature record the same
fields in the same shapes, so their records and ratings pool (the call
log's `program.interface`). It does not say which calls a program
accepts: `description`, `desc`, `type`, `opaque` and `optional` are not
in it, so a required input and an optional one of the same shape have
one signature. To know that two programs accept the same calls, compare
their interfaces, leaving out `description`, `desc` and `type`. A
`default` is part of its shape, and so of the signature (as lmcc
fingerprints shapes): changing a default gives a new signature.

## Interfaces that are refused

A module's declared interface, and any interface read from a saved
folder, is refused `interface-malformed` when:

- it has no output (the refusal names no field); or
- a field's name is not an ASCII identifier, or is used twice among the
  inputs and outputs; its shape is not valid JSON Schema; its `default`
  does not fit its shape; it is `opaque` with a shape other than `{}`; or
  it is an output marked `optional`.

The refusal names the first field at fault, inputs then outputs, in
order. It comes when the module is defined, or the folder read, before
any call. An AI function's fields are checked by lmcc when it is defined
(`signature-malformed`).

## How each program has one

- **An AI function**: its definition's `description`, `inputs` and
  `outputs` ([functions.md](functions.md), *A definition*), with
  `optional` where the language lets an input be left out. The fields
  FunctAI adds to its signature (`reasoning` with `module: "cot"`, the
  tool fields) are not in it: they are how the model answers, not what a
  caller gives or gets. Its signature equals the call log's
  `program.signature` when the function has neither reasoning nor tools.
- **A module** declares it. Python derives it from the function's
  parameters and return annotation: a parameter with a default is
  optional, its default in the shape when it has a JSON form (a default
  of `None` makes the shape nullable); an unannotated parameter, or one
  whose type has no JSON form, is opaque; `*args` is one optional input,
  a list with default `[]`, and `**kwargs` one, an object with default
  `{}`. TypeScript takes it as data (`module(name, { input, output |
  outputs, uses }, run)`, `run` given one object of the inputs). Julia
  derives it from `@program`'s typed arguments and return type.
- **Outputs.** A program's return type is one output, whatever its shape:
  a record (a Python dataclass or `TypedDict`, a TypeScript object, a
  Julia `NamedTuple`) is one output of object shape, named `result`
  unless the declaration names it. A module has several outputs only when
  it declares them as several (TypeScript `outputs`; each language says
  how), and returns them as one record by name.

## Checking values against it

A module's call checks its inputs before its code runs and its outputs
when its code returns, as an AI function's call checks what it sends and
reads. A value **fits** a field when the field is opaque, or when the
value has a JSON form (calls.md, *Values*) and that form is valid against
the field's shape as JSON Schema (draft 2020-12, `default` being only an
annotation). A value with no JSON form (written in the log as a
description) fits only an opaque field.

1. **Inputs.** Every input the call gives must be one the interface has
   (the first that is not refuses), then, in the interface's order: a
   given input must fit its field; a required input left out refuses; an
   optional input left out takes its shape's `default` when the shape has
   one, and otherwise **stays left out**: the program's own default
   applies (the module's code default, which may have no JSON form), and
   the call's record has no value for it. Left out is never null: `null`
   is a value, given or not, and is checked like any other. A refusal is
   `interface-input`, naming the input, and the module's code does not
   run.
2. **Outputs.** What the code returned is the program's outputs. One
   output: the value. Several: a record holding each output by name and
   nothing else. A value that is not a record when there are several
   refuses, naming the first output; so does a record missing an output
   (naming the first missing, in order), or holding a key the interface
   does not name (naming it: a returned key that is not an output is a
   mistake, as an input the interface lacks is). Each output must then
   fit, in order. A refusal is `interface-output`, naming the field.

The error is `{"type": "InterfaceError", "code": "interface-input" |
"interface-output", "message"}` in the call log, and the call's `outputs`
is `null`. The check runs on every call, wherever the module is called
from: an interface that is not enforced is only documentation. A module
whose fields are all opaque is refused only for its names, never for its
values.

## Where it is written

- **A saved folder**: every program node has `interface`
  ([saved.md](saved.md)), and an AI node's is checked against its
  signature. Folders written before 2026-09-28 have none: an AI node's is
  then read from its `signature` (the fields whose `purpose` is `plain`,
  the instruction as its description), and a module's is not known.
- **The call log**: every call's `program.interface` is its interface's
  signature; a module's call names its inputs and outputs as its
  interface does; a module's version includes its interface (calls.md,
  *Versions*).
- **A stream**: a module's `started` event's `inputs` are named as its
  interface names them.
- **In the program**: each language shows it as data in this JSON form
  (Python `support.interface`, TypeScript `support.interface`, Julia
  `FunctAI.interface(support)`, R `ai_interface(support)`).
