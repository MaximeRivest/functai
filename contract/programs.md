# Programs and their interfaces (format 1)

A **program** is an AI function or a module (code that calls AI
functions). Every program has an **interface**: the inputs it takes and
the outputs it gives, named and typed as JSON. The interface is what lets
a program be described, checked, served, conversed with and called from
another language without running it, and what its calls are recorded
against. An AI function's interface comes from its definition; a module
declares its own, because its code cannot be read for one in every
language.

`cases/programs/*.json` pin the interface's id and how values are checked
against it. `schema/saved.schema.json` (`$defs/interface`) checks its form.

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
| `inputs` | in order, each field `{name, shape, desc?, type?, optional?}`. |
| `outputs` | in order, each field `{name, shape, desc?, type?}`. The **last is the answer** (`program.answer` in the call log). A program with one output usually names it `result`. |

A field:

| key | meaning |
|---|---|
| `name` | an ASCII identifier, unique among the program's inputs and outputs. |
| `shape` | JSON Schema, as [functions.md](functions.md) *Shapes* says. `{}` is a type with no JSON shape the language knows (an unannotated argument, a data frame): any value, described only by `type`. A `default` in the shape is the value used when the input is left out. |
| `desc` | words about the field. |
| `type` | the host language's name for the type (`str`, `pd.DataFrame`), for people; never compared across languages. |
| `optional` | inputs only: `true` when a caller may leave the input out. It then gets its shape's `default`, or `null` when the shape has none. |

**Its id** is lmcc's signature fingerprint (kernel §3a) of the fields, as
[calls.md](calls.md) computes `program.signature`: `"sha256:"` and the
SHA-256 of the canonical JSON of the list of the inputs then the outputs,
each `{"direction", "name", "purpose": "plain", "shape", "type": ""}`.
Descriptions, `desc`, `type` and `optional` are not part of it: two
programs with the same id take and give the same data.

## How each program has one

- **An AI function**: its definition's `description`, `inputs` and
  `outputs` ([functions.md](functions.md), *A definition*), with
  `optional` where the language knows an input may be left out. The
  fields FunctAI adds to its signature (`reasoning` with `module: "cot"`,
  the tool fields) are not in it: they are how the model answers, not
  what a caller gives or gets. Its id
  equals `program.signature` when the function has neither reasoning nor
  tools.
- **A module** declares it. Python derives it from the function's
  parameters and return annotation (a parameter with a default is
  optional, its default in the shape when it has a JSON form; an
  unannotated one has shape `{}`; `*args` is one optional input, a list
  with default `[]`, and `**kwargs` one, an object with default `{}`); TypeScript takes it as data
  (`module(name, { input, output, uses }, run)`, `run` given one object
  of the inputs); Julia derives it from `@program`'s typed arguments and
  return type. A module with one output names it `result` unless its
  declaration names it; a module with several returns them as one record
  (a dict or dataclass, an object, a NamedTuple), and its call record's
  `outputs` has each by name.

## Checking values against it

A module's call checks its inputs before its code runs and its outputs
when its code returns, as an AI function's call checks what it sends and
reads. A value **fits** a shape when its JSON form (calls.md, *Values*)
is valid against the shape as JSON Schema (draft 2020-12, `default` being
only an annotation).

1. **Inputs.** An input the call does not give takes its shape's
   `default`, or `null`, when it is `optional`; otherwise the call fails
   `interface-input`, naming the input. An input the interface does not
   have fails `interface-input`. Every input's value must fit its shape,
   or the call fails `interface-input`, naming the input. The module's
   code does not run.
2. **Outputs.** What the code returned is the program's outputs (one
   output: the value; several: the record's values by name, each
   present, and keys the interface does not name left out). A record
   missing an output, or a value that is not a record when there are
   several, fails `interface-output`, naming the (first missing)
   output. Each must fit its shape, or the call fails
   `interface-output`, naming the output.

The error is `{"type": "InterfaceError", "code": "interface-input" |
"interface-output", "message"}` in the call log, and the call's `outputs`
is `null`. A shape of `{}` accepts every value, so a module that declares
no types is never refused, and cannot be served or kept in a conversation
store either: those need JSON (they refuse `interface-untyped` before any
call, naming the field).

## Where it is written

- **A saved folder**: a module node has `interface` ([saved.md](saved.md)).
  An AI node's interface is read from its `signature`: the fields whose
  `purpose` is `plain`, and the instruction as its description.
- **The call log**: a module's call names its inputs and outputs as its
  interface does, and its `program.signature` is the interface's id; a
  module's version includes its interface (calls.md, *Versions*).
- **A stream**: a module's `started` event's `inputs` are named as its
  interface names them.
- **In the program**: each language shows it as data in this JSON form
  (Python `support.interface`, TypeScript `support.interface`, Julia
  `FunctAI.interface(support)`, R `ai_interface(support)`).
