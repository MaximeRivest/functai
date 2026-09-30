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
are refused, and how values are bound and checked against one.
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
| `name` | an ASCII identifier (a letter or `_`, then letters, digits or `_`), matched whole (nothing after it, not even a newline), used once among the program's inputs and outputs. |
| `shape` | JSON Schema (draft 2020-12), as [functions.md](functions.md) *Shapes* says, read by the keywords *Checking values* lists: a module's uses no other; an AI function's may carry others, which are lmcc's. `{}` is any JSON value. The shape's own `default` (its top-level key, in an input's shape) is the value the input takes when it is left out, and must fit the shape. A `default` inside the shape (a member's, as pydantic writes one for a model's defaulted field) is a word: never checked, never filled in. A shape with `properties` and no `additionalProperties` is a **record**, closed: it holds the members it names and no other (*Checking values*). |
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
**without any `default`**: its own, and every `default` keyword inside
it (a member's, an item's, a `$defs` entry's, an `anyOf` option's). Only
keywords go: a member *named* `default` (a key of `properties`) stays,
and so does any value inside `enum`, `const` or `examples`. It says what
the program's data looks like: calls of two programs with the same
signature record the same fields in the same shapes, so their records
and ratings pool (the call log's `program.interface`). It does not say
which calls a program accepts, nor what it does with them:
`description`, `desc`, `type`, `opaque`, `optional` and defaults are not
in it, so a required input and an optional one of the same shape have
one signature, and so do two versions whose default differs (a default
computed when the program is loaded, such as today's date, or a record
type whose field defaults to it, must not split a program's records). A
default is behaviour: it is in the program's version, by its logic
(calls.md, *Versions*), and a call's record holds the value it took. To know that two programs accept the same
calls, compare their interfaces, leaving out `description`, `desc` and
`type`. Values written as descriptions are not data even when the
signatures match: `rated` and replay leave them out (calls.md).

## Interfaces that are refused

Every interface is checked by these rules: a module's when the module is
defined, an AI function's when the function is defined, and any
interface read from a saved folder. It is refused `interface-malformed`
when:

- the interface is not an object with `description` (text), `inputs` (a
  list) and `outputs` (a list of at least one), and no other key (the
  refusal names no field); or
- a field has a key the tables above do not name, `desc` or `type` that
  is not text, or `opaque` or `optional` that is not `true`; its name is
  not an ASCII identifier, or is used twice among the inputs and
  outputs; its shape is not an object, uses a keyword *Checking values*
  lists with a value of the wrong kind, has a reference that comes back
  to itself without passing into a value, or (in a module's interface)
  uses a keyword *Checking values* does not list; its shape's own
  `default` does not fit its shape (an interface holds a bound default:
  *Binding a call's inputs*); it is `opaque` with a shape other
  than `{}`; it is an
  output marked `optional`; or it is an AI function's optional input with
  no `default`.

Form and meaning are checked together, field by field: the refusal names
the first field at fault, inputs then outputs, in order, whatever its
fault. It comes when the program is defined, or the folder read, before
any call. A saved folder's manifest is checked against its schema first
([saved.md](saved.md)): a fault the schema sees there refuses
`saved-malformed`. An AI function's definition is checked by lmcc first
(`signature-malformed`, which asks only that each shape is an object),
then its interface by the rules above (its shapes may carry lmcc's other
keywords: *Checking values*). So what one language lets an AI function
be defined and saved with, every language can load and describe: an
interface no language would read back is refused where it is written,
not in another process later (`programs/19`; a pydantic `Field(ge=10)`
with a default of 5, which pydantic does not check, refuses there).

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

## Binding a call's inputs

Every program's call, an AI function's and a module's, **binds** each
input it is given to its field before anything else: the value is
converted toward the field's shape when its meaning is clear, then
checked (*Checking values*). A caller gives what it has (a number read
from a table, a date, a row); the program gets its declared types, and
the call's record holds the bound values, so records and the rows made
from them hold one type per field. Only inputs are bound: what a
module's code returns and what a model replies are checked as they are
(a program's outputs are its promise; a model's reply that does not fit
is asked again).

A value is first taken as its JSON form (calls.md, *Values*), when it
has one. Then, by the shape (after following `$ref`):

| the shape wants (`type`) | a value is converted when it is | and refused when it is |
|---|---|---|
| `string` | a number: its canonical JSON text (`42`, `2.5`); a boolean: `true` or `false`; an array or object: its JSON, indented by two spaces as *Worked examples* write it ([functions.md](functions.md)); a value with no JSON form whose type defines its own text (a data frame, a date): that text | a value with no JSON form whose only text is the language's default for any object (`<object at 0x…>`) |
| `integer` | a number with no fraction (`5.0` is `5`); text that reads as a JSON number with no fraction, white space around it trimmed (`"5"`, `" 5 "`, `"5.0"`) | a number with a fraction, other text, a boolean |
| `number` | text that reads as a JSON number, white space around it trimmed (`"2.5"`) | other text, a boolean |
| `boolean` | never: `true` and `false` only | anything else (`"yes"`, `1`) |
| `array` | each item bound by `prefixItems`, then `items` | |
| `object` | each member bound by its shape in `properties`, else `additionalProperties`; a **record** (a shape with `properties` and no `additionalProperties`) keeps only the members it names, in the value's order: the others are dropped | |
| a list of types | the first type in the list the value binds to and fits; `null` stays `null` | |
| none (`{}`), or an opaque field | never: the value as it is | |

- **Choices** (`enum`, `const`) bind by the shape's `type` when it has
  one (`{"enum": [1, 2], "type": "integer"}` given `"2"` is `2`), and
  then must match exactly.
- **`anyOf`**: `null` stays `null`; any other value takes the first
  option it binds to and fits; when none, it is refused as the first
  option refuses it.
- **Missing values** are `null`: Python `None` and a float not-a-number
  (pandas' gap), R `NULL` and `NA`, Julia `nothing` and `missing`,
  TypeScript `null` and `undefined` given explicitly. `null` given to an
  **optional** input that `null` does not fit is that input **left out**:
  it takes its default (a table's gap, mapped, gives the program's
  default). To a required input that `null` does not fit, it is refused.
  Inside a value, a missing value is `null` and is checked as any.
- A number is never text-rounded: `2.5` for an integer is refused, not
  `2`; not-a-number and the infinities are not numbers.
- An **optional input's default** is bound the same way when the program
  is defined, before its interface is made (Julia's `1.0` for an integer
  input is `1`; Python's `"5"` for one is `5`): the interface holds the
  bound value, and a default that does not bind refuses
  `interface-malformed`. An interface read as data (a saved folder) is
  not bound again: its defaults must fit.
- A value refused by binding refuses `interface-input`, naming the input,
  before any request is sent or any of the module's code runs. The
  message follows *The message*, below.

A language with no type annotation on an AI function's input (Python
`def f(text)`) declares it text (`{"type": "string"}`): binding then
gives the model the value's text. A module's unannotated input is opaque:
code, unlike a model, can use any value.

## Checking values against it

A module's call checks its bound inputs before its code runs and its
outputs when its code returns, as an AI function's call checks what it
sends and reads. A value **fits** a field when the field is opaque, or
when the value has a JSON form (calls.md, *Values*) and that form fits
the field's shape. A value with no JSON form (written in the log as a
description) fits only an opaque field.

Values are checked by these keywords, read as JSON Schema draft 2020-12
reads them, so that every language checks alike without a full JSON
Schema validator:

- `type`: one of `null`, `boolean`, `integer`, `number`, `string`,
  `array`, `object`, or a list of them, each once. An integer is a
  number with no fraction (`5.0` is one); every integer is a number;
  `true` is not a number.
- `enum` (a list of at least one value), `const`: equal as canonical
  JSON (calls.md), so `1` and `1.0` are equal and `true` is not `1`.
- `anyOf` (a list of at least one shape): fits one of them.
- For arrays: `items` (a shape), `prefixItems` (a list of at least one
  shape), `minItems`, `maxItems`, `uniqueItems` (a boolean; unique as
  canonical JSON).
- For objects: `properties` (shapes by name), `required` (names, each
  once), `additionalProperties` (a shape, `true` or `false`). A shape
  with `properties` and no `additionalProperties` is a **record** and is
  **closed**: a member it does not name does not fit, as if
  `additionalProperties` were `false` (lmcc sends records to providers
  so, D-57). A map (`additionalProperties` a shape) or an explicit
  `additionalProperties: true` is open as it says. A model's reply with a
  member its record does not name is unreadable (`parse-value`,
  [functions.md](functions.md)); a module's returned record with one
  refuses `interface-output`; an input drops it (*Binding*).
- For strings: `minLength`, `maxLength`, counted in Unicode code points.
- For numbers: `minimum`, `maximum`, `exclusiveMinimum`,
  `exclusiveMaximum` (numbers).
- `$defs` (shapes by name), and `$ref` naming one of the shape's own (a
  record by reference, as pydantic writes one). A `$ref` is
  `#/$defs/<name>`, where `<name>` is ASCII letters, digits, `_`, `.` and
  `-` (every name pydantic writes fits), matched whole and read as written:
  nothing after it (not even a newline: a regular expression's `$` may
  match before one, so implementations match the whole text), no JSON
  Pointer escapes (`~1`) and no percent-encoding (`%20`). A `$defs` entry
  whose name has other characters cannot be referred to
  (`programs/20`).
- Words that are never checked, each of its kind: `title`,
  `description`, `format`, `$comment` (text), `deprecated`, `readOnly`,
  `writeOnly` (booleans), `examples` (a list), `default` (any value;
  only a field's own must fit, as *The interface* says: one inside the
  shape is never checked nor filled in, and is left out of the
  signature, as the field's own is: `programs/21`).

A count (`minItems`, `maxItems`, `minLength`, `maxLength`) is an integer
of at least 0, by the rule above: `2.0` is `2`, since some languages
cannot tell them apart once read. A keyword with a value of another kind
refuses `interface-malformed`.

**References end.** Checking a value follows `$ref` and `anyOf` without
moving into the value, and moves into it through `items`, `prefixItems`,
`properties` and `additionalProperties`. A `$defs` entry that comes back
to itself by `$ref` and `anyOf` alone (`{"$defs": {"A": {"$ref":
"#/$defs/A"}}, …}`) would never end, and refuses `interface-malformed`,
whether or not anything names it. A record holding a list of itself is
checked one level of the value at a time, and ends.

**A module's shapes** use only these keywords. Any other (`pattern`,
whose regular expressions differ between languages; `oneOf`;
`multipleOf`, whose floating point differs) refuses
`interface-malformed`: a module's interface either says what every
language checks, or is refused. A later contract may add keywords
(lmcc's `media`, stage 3's boundaries).

**An AI function's shapes** are lmcc's, and may carry other keywords
(`pattern`, `oneOf`, `media`), which lmcc passes on to formats and to
the provider untouched. FunctAI checks an AI function's values (a
default here; a served call's inputs in stage 3) by the keywords above
alone, and never reads the others: a default that a `pattern` would
refuse fits, the same in every language. The keywords above still refuse
when their value is of the wrong kind, and references must end, as for
a module.

When several names are at fault, the one named is the first in
code-point order of the names: a map's order is not the same in every
language.

1. **Inputs.** Every input the call gives must be one the interface has
   (else the first in code-point order that is not refuses), then, in the
   interface's order: a given input is bound (*Binding a call's inputs*)
   and must then fit its field; a required input left out refuses; an
   optional input left out takes its shape's `default` when the shape
   has one, and otherwise **stays left out**: the program's own default
   applies (the module's code default, which may have no JSON form), and
   the call's record has no value for it. `null` is a value, and is
   checked like any other, except that `null` given to an optional input
   that `null` does not fit is that input left out (*Binding*). A refusal
   is `interface-input`, naming the input, and the module's code does not
   run.
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
is `null`. The check runs on every call, wherever the program is called
from: an interface that is not enforced is only documentation. A module
whose fields are all opaque is refused only for its names, never for its
values. An AI function's refusal is recorded as a module's is: a call
with that error and no exchange.

**The message** names the program, the field and what the field wants,
and quotes the value at fault: its canonical JSON (calls.md), or its
description's `$repr` for a value with no JSON form, cut after 80 code
points and ended with `…` when longer. When a `log_content` layer in
effect for the call drops the field (calls.md, *Content*), the message
names the field and what it wants, and never the value: an exception
travels further than the log (a traceback, an error tracker, a served
program's reply). The words are each language's; cases pin only whether
the value is quoted.

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
