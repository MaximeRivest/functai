# Plugins

A **plugin** is a named, versioned set of hooks: functions FunctAI calls at fixed points of a call or a conversation's turn. A hook either changes something, by returning a [`Change`](@ref) (data), or only hears of something. Every change is recorded in the call's record, so a rated call is asked again as it was (contract/plugins.md). FunctAI's own features are plugins written with these hooks and nothing private: `approve =` is the `approval` plugin, [`compaction`](@ref) and [`delegate`](@ref) are built from them, and you can replace any of them.

```@setup plugins
using FunctAI
```

## A plugin

```@example plugins
modes = Plugin("modes"; version = "1.0.0")
on!(modes, :before_call) do call
    Change(sections = ["Answer carefully. Cite the file you read."])
end
guard = Plugin("no-deletes"; tool_call = t -> t.name == "delete_file" ? Change(block = "deleting files is not allowed here") : nothing)
(modes, guard)
```

Use them where settings go: `configure!(plugins = [guard])` for every call, `with_settings(plugins = […])` for a block, `@ai plugins = [modes] function …` for one function, `conversation(f; plugins = […])` for a conversation, or a file that defines `plugin` (`plugins = ["plugins/modes.jl"]`, [`load_plugin`](@ref); loading runs the file: load only code you trust).

| hook | when | a `Change` of |
|---|---|---|
| `turn_start` | a conversation's turn is sent, before it is recorded | `inputs` |
| `context` | its earlier turns are chosen | `keep`, `without`, `sections` |
| `before_call` | an AI function is about to be asked (helpers too) | `instruction`, `sections`, `lm`, `settings`, `tools` |
| `request` | a provider request is about to be sent | returns another `LM15.Request` (the escape hatch) |
| `tool_call` | a tool is about to run | `inputs`, `block`; or `FunctAI.ask(event)` a person |
| `tool_result` | a tool ran | `output` |
| `turn_end` | a turn ended (hears only; may `FunctAI.remember!` entries) | |

A handler is given an event with the fields of its hook (`call.function`, `call.inputs`, `call.instruction`, `call.path`, `call.tools`; `t.name`, `t.input`, `t.effects`, `t.path`; …) and returns a `Change` or `nothing`. A field its hook does not take refuses `plugin-change` ([`PluginError`](@ref)), and so does a value that does not fit (a tool the function does not have, a turn not on the branch). A hook that does not exist refuses `plugin-hook` when it is registered: a misspelt hook would otherwise never run.

## Order

The program's own plugins run first, then each layer around the call from the innermost out (a conversation's, each enclosing block), then `configure!`'s: the host's handlers see what the program's did, and have the last word. A plugin set in several layers runs once, in its outermost layer. `sections` add up; `instruction`, `lm`, `tools` and an input are replaced by a later change; `settings` merge. In `tool_call`, the first `block` ends the hook, and the `approval` plugin runs last, on the input the others left. A host's `program_plugins = false` drops a program's own.

## When a handler fails

A handler that throws stops the call (`plugin-failed`): the change it was meant to make (a redaction, a guard) did not happen, and the call must not go on as if it had. Except `tool_call`, where a handler that throws blocks the tool (the model is told a check failed), and `turn_end`, which hears only (a failure is reported). A plugin is code the host trusts, run in the call's own task.

## What is recorded

A call's record keeps `changes` (each change, in order: plugin, version, hook, the change as data; without the change when its content is not whole), `sections` (what its conversation's `context` hooks gave it), and `replayable = false` when a `request` handler replaced a request (no one could rebuild it). A turn keeps what its `context` hooks showed it, and resuming it shows exactly that.

A rated call is asked again (`evaluate`, the optimizers) with what it was shown of its conversation, never with recorded `before_call` changes: how the host shaped it is the environment's, so the program under test is what is measured, its improved instruction included.

## Entries

A plugin keeps what it needs later as **entries** of a conversation, at a turn (they then belong to the branches through that turn): `FunctAI.remember!(event, kind, data)` in a hook, `FunctAI.entries(event, kind)` to read them on the branch. A host keeps one too: `FunctAI.remember!(chat, "modes", "mode", "careful")`.

## Built in

- [`compaction`](@ref)`(keep = 20, every = 10)`: keeps a long conversation short. When more than `keep + every` turns of a branch are not summarized, every turn but the last `keep` is folded into a summary (by an AI function, or `summarize = (earlier_summary, rows) -> text`), kept as an entry at that turn; the next turns are shown the summary as a section of the instruction, and only the turns after it. Each branch has its own.
- [`delegate`](@ref)`(program)`: another program as a tool; asked inside a turn, it answers in a conversation of its own (`<conversation>.<tool>`, same store) that follows the branch of the turn that asked.

```julia
chat = conversation(tutor, "alex"; store = "tutoring/", plugins = [compaction(keep = 20, every = 10)])
researcher = delegate(research; description = "Look things up in the notes.")
@ai tools = [researcher] function assistant(request::String)::String
    "Help with the request."
end
```

## Plugins and layouts

`instruction` and `sections` change the instruction lmcc is given; a template that never writes `{instruction}`, or a baked student (it reads only the message it was trained on), would drop them while the record says they were sent, so such a call refuses `plugin-change` before anything is sent. The program's version is computed without plugins: they change calls, not programs.
