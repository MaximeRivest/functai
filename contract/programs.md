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
`schema/interface.schema.json` checks its form.

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
| `shape` | JSON Schema (draft 2020-12), as [functions.md](functions.md) *Shapes* says; a module's uses only the keywords *Checking values* lists. `{}` is any JSON value. A `default` in an input's shape is the value it takes when it is left out, and must fit the shape. |
| `desc` | words about the field. |
| `type` | the host language's name for the type (`str`, `pd.DataFrame`), for people; never compared across languages. |
| `opaque` | `true` when the field's values may have no JSON form in the language that declared it (a data frame, a file handle, an unannotated Python argument, Python's `Any`). Its shape is `{}`. Its values are never checked. It says what the field accepts and where it may go, not how a value is written: the log writes each value by what it is (JSON when it has a JSON form, else a description: calls.md, *Values*). A boundary that needs data (serving, a conversation store: stages 2 and 3) cannot carry the field, and says so before any call. |
| `optional` | inputs only: `true` when a caller may leave the input out (*Checking values*). |

An interface and its fields have only the keys above. A later contract
may add some (kinds of data for `log_content` and views, question 1 of
design/08); until a reader knows a key, it refuses the interface
(`interface-malformed`) rather than ignore it, since a key it does not
know may narrow what the program accepts or say how a field must be
kept.

**Its signature** is lmcc's signature fingerprint (kernel §3a) of the
fields: `"sha256:"` and the SHA-256 of the canonical JSON of the list of
the inputs then the outputs, each `{"direction", "name", "purpose":
"plain", "shape", "type": ""}`, where `shape` is the field's shape
without its own `default` key. It says what the program's data looks
like: calls of two programs with the same signature record the same
fields in the same shapes, so their records and ratings pool (the call
log's `program.interface`). It does not say which calls a program
accepts, nor what it does with them: `description`, `desc`, `type`,
`opaque`, `optional` and defaults are not in it, so a required input and
an optional one of the same shape have one signature, and so do two
versions whose default differs (a default computed when the program is
loaded, such as today's date, must not split a program's records). A
default is behaviour: a module's is in its version (calls.md,
*Versions*); an AI function's is bound before the request, and the
record holds the value it took. To know that two programs accept the same
calls, compare their interfaces, leaving out `description`, `desc` and
`type`. Values written as descriptions are not data even when the
signatures match: `rated` and replay leave them out (calls.md).

## Interfaces that are refused

A module's declared interface, and any interface read from a saved
folder, is refused `interface-malformed` when:

- the interface is not an object with `description` (text), `inputs` (a
  list) and `outputs` (a list of at least one), and no other key (the
  refusal names no field); or
- a field has a key the tables above do not name, `desc` or `type` that
  is not text, or `opaque` or `optional` that is not `true`; its name is
  not an ASCII identifier, or is used twice among the inputs and
  outputs; its shape is not an object, or (in a module's interface) uses
  a keyword *Checking values* does not list, or a listed keyword with a
  value of the wrong kind; its `default` does not fit its shape; it is
  `opaque` with a shape other than `{}`; it is an output marked
  `optional`; or it is an AI function's optional input with no `default`.

Form and meaning are checked together, field by field: the refusal names
the first field at fault, inputs then outputs, in order, whatever its
fault. It comes when the module is defined, or the folder read, before
any call. A saved folder's manifest is checked against its schema first
([saved.md](saved.md)): a fault the schema sees there refuses
`saved-malformed`. An AI function's shapes are lmcc's, checked by lmcc
when it is defined (`signature-malformed`).

## How each program has one

- **An AI function**: its definition's `description`, `inputs` and
  `outputs` ([functions.md](functions.md), *A definition*), with
  `optional` where the language lets an input be left out. A model is
  sent every input, so an AI function's optional input always has a
  `default` in its shape, the value sent when it is left out: the
  language's default (it must have a JSON form), or `null` when the
  language gives none (TypeScript's optional inputs; the shape then
  accepts null). The fields FunctAI adds to its signature (`reasoning`
  with `module: "cot"`, the tool fields) are not in it: they are how the
  model answers, not what a caller gives or gets. Its signature equals
  the call log's `program.signature` when the function has neither
  reasoning nor tools.
- **A module** declares it. Python derives it from the function's
  parameters and return annotation: a parameter with a default is
  optional, its default in the shape when it has a JSON form (a default
  of `None` makes the shape nullable); an unannotated parameter, one
  annotated `Any` or `object`, or one whose type has no JSON form, is
  opaque (`functai.JSON` says "any JSON value", shape `{}`); `*args` is
  one optional input named after it, a list with default `[]`, holding
  the call's extra positional arguments in order, and `**kwargs` one, an
  object with default `{}`, holding its extra keyword arguments by name
  (a call from data gives them so, and the module's code receives them
  as Python passes them). A shape Python writes with keywords the
  vocabulary lacks (a pydantic model's `pattern`) refuses at definition;
  declare the field opaque, or use a type the vocabulary can say.
  TypeScript takes it as data (`module(name, { input, output | outputs,
  uses }, run)`, `run` given one object of the inputs). Julia derives it
  from `@program`'s typed arguments and return type (`Any` is opaque).
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
value has a JSON form (calls.md, *Values*) and that form fits the
field's shape. A value with no JSON form (written in the log as a
description) fits only an opaque field.

A module's shapes use only these keywords, read as JSON Schema draft
2020-12 reads them, so that every language checks alike without a full
JSON Schema validator:

- `type`: one of `null`, `boolean`, `integer`, `number`, `string`,
  `array`, `object`, or a list of them. An integer is a number with no
  fraction (`5.0` is one); every integer is a number; `true` is not a
  number.
- `enum` (a list), `const`: equal as canonical JSON (calls.md), so `1`
  and `1.0` are equal and `true` is not `1`.
- `anyOf` (a list of shapes): fits one of them.
- For arrays: `items`, `prefixItems`, `minItems`, `maxItems`,
  `uniqueItems` (unique as canonical JSON).
- For objects: `properties`, `required`, `additionalProperties` (a
  shape, or `false`).
- For strings: `minLength`, `maxLength`, counted in Unicode code points.
- For numbers: `minimum`, `maximum`, `exclusiveMinimum`,
  `exclusiveMaximum`.
- `$defs`, and `$ref` of the form `#/$defs/<name>` naming one of the
  shape's own (a record by reference, as pydantic writes one).
- Words that are never checked: `title`, `description`, `default`,
  `examples`, `format`, `$comment`, `deprecated`, `readOnly`,
  `writeOnly`.

Any other keyword (`pattern`, whose regular expressions differ between
languages; `oneOf`; `multipleOf`, whose floating point differs) refuses
`interface-malformed`: an interface either says what every language
checks, or is refused. A later contract may add keywords (lmcc's
`media`, stage 3's boundaries).

When several names are at fault, the one named is the first in
code-point order of the names: a map's order is not the same in every
language.

1. **Inputs.** Every input the call gives must be one the interface has
   (else the first in code-point order that is not refuses), then, in the
   interface's order: a given input must fit its field; a required input
   left out refuses; an optional input left out takes its shape's
   `default` when the shape has one, and otherwise **stays left out**:
   the program's own default applies (the module's code default, which
   may have no JSON form), and the call's record has no value for it.
   Left out is never null: `null` is a value, given or not, and is
   checked like any other. A refusal is `interface-input`, naming the
   input, and the module's code does not run.
2. **Outputs.** What the code returned is the program's outputs. One
   output: the value. Several: a record holding each output by name and
   nothing else. A value that is not a record when there are several
   refuses, naming the first output; so does a record holding a key the
   interface does not name (naming the first in code-point order: a
   returned key that is not an output is a mistake, as an input the
   interface lacks is). Then, in the interface's order, each output must
   be there and fit. A refusal is `interface-output`, naming the field.

The error is `{"type": "InterfaceError", "code": "interface-input" |
"interface-output", "message"}` in the call log, and the call's `outputs`
is `null`. The check runs on every call, wherever the module is called
from: an interface that is not enforced is only documentation. A module
whose fields are all opaque is refused only for its names, never for its
values.

## Where it is written

- **A saved folder**: every program node has `interface`
  ([saved.md](saved.md)), and an AI node's is checked against its
  signature; loading an AI function takes its optional inputs and their
  defaults from it. Folders written before 2026-09-28 have none: an AI node's is
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
