# Writing AI functions

An AI function has a name, typed inputs, typed outputs, and a description. The model writes its body every time it is called.

```@setup functions
using FunctAI
FunctAI.configure!(lm = "gpt-4.1-mini")
```

## The parts of `@ai`

```@example functions
@ai function triage(ticket::String; product::String = "the shop")
    """
    Read a support ticket.

    # Arguments
    - `ticket`: the customer's own words
    """
    summary::String = ai"one sentence, no names"
    minutes::Int    = ai"minutes to fix"
end
```

- **Inputs** are the arguments, positional or keyword, as in any Julia function. A default makes an input optional; it is written in the function's interface and sent to the model whenever the input is left out, so it is data: a literal or a constant, the same for every call (a computed default, or one that uses another input, is refused when the function is defined).
- **The description** is the first string of the body. A docstring's `# Arguments` list describes the inputs, and those words go to the model as "Parameter guidance".
- **Outputs** are `name::Type = ai"words"` lines. The words go to the model as "Output guidance". The last output is the answer (the one ratings are about). With several, calling returns them all as a `NamedTuple`.
- With **no output lines**, the one output is `result`, of the return type (`String` when none is written).

What the model reads is always one call away, and costs nothing:

```@example functions
FunctAI.prompt(triage, "I was charged twice for one order.")
```

`FunctAI.instructions(f)` is the instruction alone; `render(f, args...)` is the lm15 request itself.

## Types

A type is a promise the answer keeps. FunctAI writes each type as the JSON Schema the model is held to, the same schema Python writes for the same type, and reads the reply back as the type:

| You write | The model is asked for | You get |
|:--|:--|:--|
| `String`, `Int`, `Float64`, `Bool` | text, an integer, a number, true/false | that type |
| an `@enum` | one of its names | the enum value |
| `OneOf(:a, :b)`, `OneOf("x", "y")`, `OneOf(list)` | one of these | the `Symbol` or `String` |
| `Union{T,Missing}`, `Union{T,Nothing}` | a `T`, or null | `missing` / `nothing` for null |
| `Vector{T}`, `Set{T}` | a list | a `Vector{T}` / `Set{T}` |
| `Dict{String,T}` | a map | a `Dict` |
| a struct, a `NamedTuple` type | a record, every field required | your struct / `NamedTuple` |
| `Any` | any JSON | JSON values (`Dict`, `Vector`, …) |
| a `Dict` (a JSON Schema) | that schema | JSON values |

```@example functions
struct Person
    name::String
    age::Union{Int,Missing}
end
print(FunctAI.LMCC.json_text(FunctAI.shape_of(Person)))    # the schema the model is held to
```

A reply that doesn't fit (a choice outside the list, a missing field, a fraction for an `Int`) is unreadable: FunctAI asks again, with the reason, up to `retries` times (default 1), then throws. For a type with fields that may be missing, the `:json` layout (`adapter = :json`) holds the model to the exact schema.

## Code of your own

Code after the outputs runs on them, with the inputs in scope, and its value is what calling returns:

```@example functions
@ai function price(item::String)::Float64
    "Estimate the price in US dollars."
    usd::Float64 = ai"the price"
    round(usd; digits = 2)
end
```

The call log records both what the model answered (`usd`) and what the function returned. Code of your own is part of the function's [`version`](@ref), by its parsed form: reformatting or editing comments doesn't change it.

## Without the macro

The same function, as data, for when names and types come from elsewhere (a config file, a table's columns):

```@example functions
@enum Mood happy unhappy mixed
mood = AIFunction("mood", "How does the customer feel about what they bought?";
                  inputs = (review = String => "the customer's own words",), output = Mood)
```

`inputs` and `outputs` are `NamedTuple`s, `Dict`s or vectors of `name => spec`; a spec is a type, `Type => "words"`, `OneOf(...)` or a JSON Schema `Dict`.

## `missing`

A `missing` input is not sent: the call returns `missing`, for free. That makes a column with holes safe to broadcast over. An answer the model may leave empty is a different thing: declare it `Union{T,Missing}`.

## Help

`?mood` shows the function's signature and description, as for any documented function.
