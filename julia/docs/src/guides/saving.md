# Saving, loading, and other languages

```@setup saving
using FunctAI
```

## Save and load

```@example saving
@enum Mood happy unhappy mixed
@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end
taught = with_demos(mood, ["Broke in a day." => unhappy, "Love it!" => happy])

dir = joinpath(mktempdir(), "mood")
FunctAI.save(dir, taught)
readdir(dir)
```

`functai.json` holds the signature, the instruction, the worked examples, the layout and the settings, with the fingerprints of the requests it renders. It is indented JSON, to read, diff and keep in git.

```@example saving
loaded = FunctAI.load(dir; types = (result = Mood,))
version(loaded) == version(taught)
```

Loading checks, before any call, that the function sends exactly what it sent when it was saved; a difference refuses with [`LoadRefused`](@ref) (`saved-differs`) rather than running another function under this one's name. A folder holds JSON shapes, not Julia types, so without `types` a choice comes back as `String`, a record as a `NamedTuple`; `types` gives fields their Julia types back, when the shapes agree.

## Across languages

A function saved in Python, TypeScript or R loads here and sends the same bytes; one saved here loads there. What only the saving language can run is refused, with the reason:

| Refusal | Why |
|:--|:--|
| `saved-code` | code of its own beside the model, in the saving language |
| `saved-tools` | tools: a tool is code |
| `saved-not-ai` | a module (a program) is code |
| `saved-model` | a baked model, or another function as a setting |
| `saved-format` | a format this loader doesn't know |
| `saved-malformed` | not a manifest |
| `saved-differs` | it would send something else than was saved |

For the same reason, a Julia function with code of its own or tools can't be saved yet: the folder carries no Julia code.

```julia
python_mood = FunctAI.load("saved/mood")        # saved by functai.save in Python
python_mood("Arrived late but fine.")
```
