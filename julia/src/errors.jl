# The errors FunctAI defines by a contract code (contract/README.md, "Refusal
# codes FunctAI defines"). Each carries `code`, the contract's word for what
# went wrong, so a program can branch on it and the call log records it, the
# same in every language. lmcc's own refusals (`LMCC.Refusal`) carry lmcc's.

"""
    FunctAIError

The supertype of every error FunctAI throws with a contract code: `err.code`
is a short, stable word for what went wrong (`"interface-input"`,
`"turn-waiting"`, `"approval-required"`, …), the same in every language
FunctAI is written in, and the one the call log records. Branch on it, not
on the message:

```julia
try
    chat("Refund my order, please.")
catch err
    err isa FunctAI.FunctAIError && err.code == "turn-waiting" || rethrow()
    # a person has to approve a tool call first
end
```

Its subtypes say where it comes from: [`InterfaceError`](@ref),
[`LogContentError`](@ref), [`JournalError`](@ref), `StoreRefusal`,
`SawUnknown`, `LoadRefused`, [`ConversationError`](@ref),
[`Waiting`](@ref), [`ApprovalError`](@ref), [`PluginError`](@ref),
`ServeError`, `RemoteError`.
"""
abstract type FunctAIError <: Exception end

"""
    FilesUnavailable

A function reads files (an input typed `Artifacts`: Attachments' `att()`
results, contract/functions.md), and Julia has no Attachments: code
`files-unavailable`, raised when such a function is loaded. Nothing was
sent. Call it where it was saved, or serve it there and call it with
[`remote`](@ref).
"""
struct FilesUnavailable <: FunctAIError
    code::String
    msg::String
    field::String
end
FilesUnavailable(msg, field) = FilesUnavailable("files-unavailable", String(msg), String(field))
Base.showerror(io::IO, e::FilesUnavailable) = print(io, "FilesUnavailable [", e.code, "]: ", e.msg)

"""
    ConversationError

A conversation, or one of its turns, refused what was asked
(contract/conversations.md). `code` is one of `conversation-id`,
`conversation-content` (a store that keeps records, for a program a
`log_content` setting keeps values of out of the log), `conversation-signature`
(the program changed in a way earlier turns cannot be shown with; name an
added `reasoning` in `earlier_without`), `conversation-opaque`,
`conversation-busy`, `conversation-nested`, `turn-unknown`, `turn-state`,
`turn-unfinished` or `store-conflict`; `turn` names the turn at fault.
"""
struct ConversationError <: FunctAIError
    code::String
    msg::String
    turn::Union{Nothing,String}
end
ConversationError(code, msg; turn=nothing) = ConversationError(String(code), String(msg), turn === nothing ? nothing : String(turn))
Base.showerror(io::IO, e::ConversationError) = print(io, "ConversationError [", e.code, "]: ", e.msg)

"""
    Waiting

A turn stopped to wait (code `turn-waiting`): for a person's answer (a tool
call its `approve` rule, or a plugin, asks about, with no function to ask),
or for results that come later (tools that returned [`later`](@ref)).
`turn` is the [`Turn`](@ref), `approvals` the questions ([`approve!`](@ref)
or [`deny!`](@ref)) and `later` the results awaited (`(key, invocation, id,
name, input, site)` each: [`deliver!`](@ref) one), from any process that
opens the conversation; once nothing waits, the turn goes on.
"""
struct Waiting <: FunctAIError
    code::String
    msg::String
    turn::Any
    approvals::Vector{Any}
    later::Vector{Any}
end
Waiting(msg::AbstractString; turn=nothing, approvals=Any[], later=Any[]) =
    Waiting("turn-waiting", String(msg), turn, Any[approvals...], Any[later...])
Base.showerror(io::IO, e::Waiting) = print(io, "Waiting [turn-waiting]: ", e.msg)

"""
    ApprovalError

A tool call needs a person's answer and nobody can be asked (code
`approval-required`): a plain call, with a rule and no function to ask, no
stream to answer on and no conversation to wait in. The tool did not run.
`approval` is what would have been asked.
"""
struct ApprovalError <: FunctAIError
    code::String
    msg::String
    approval::Any
end
ApprovalError(msg::AbstractString; approval=nothing) = ApprovalError("approval-required", String(msg), approval)
Base.showerror(io::IO, e::ApprovalError) = print(io, "ApprovalError [approval-required]: ", e.msg)

"""
    PluginError

A plugin refused or failed (contract/plugins.md): `code` is `plugin-api`,
`plugin-hook` (no such hook), `plugin-name`, `plugin-change` (a change this
hook cannot make, or a value that does not fit), `plugin-failed` (its
handler threw: the call stops, since the change it was meant to make did
not happen) or `plugin-load`; `plugin` and `hook` name where.
"""
struct PluginError <: FunctAIError
    code::String
    msg::String
    plugin::Union{Nothing,String}
    hook::Union{Nothing,String}
end
PluginError(code, msg; plugin=nothing, hook=nothing) =
    PluginError(String(code), String(msg), plugin === nothing ? nothing : String(plugin), hook === nothing ? nothing : String(hook))
Base.showerror(io::IO, e::PluginError) = print(io, "PluginError [", e.code, "]: ", e.msg)

"""
    ServeError

A program cannot be served as asked (contract/serving.md): `serve-opaque`
(an input or output with no JSON form cannot cross HTTP) or `serve-keys`
(listening beyond this machine with no keys).
"""
struct ServeError <: FunctAIError
    code::String
    msg::String
end
Base.showerror(io::IO, e::ServeError) = print(io, "ServeError [", e.code, "]: ", e.msg)

"""
    ReachError

A program needs to reach what the wall it runs behind does not let through
(code `needs-unmet`: contract/calls.md, *Reach*), refused before its first
call: it would fail later, halfway, for a reason no score would show.
`missing` is `{"domains": [...], "secrets": [...]}`, what it needs and is
not let through.
"""
struct ReachError <: FunctAIError
    code::String
    msg::String
    missing::Dict{String,Any}
end
ReachError(msg::AbstractString, missing) = ReachError("needs-unmet", String(msg), Dict{String,Any}(missing))
Base.showerror(io::IO, e::ReachError) = print(io, "ReachError [needs-unmet]: ", e.msg)

"""
A turn stops to wait, for a person or for results that come later (inside
the call that asked; the turn's
end turns it into [`Waiting`](@ref)). Code inside a call never treats it as
an ordinary failure: no call it passes through ends, and nothing is recorded
as failed.
"""
struct TurnWaiting <: Exception
    approval::Any                   # the approval it waits for, or nothing
    later::Vector{Any}              # the results it waits for: {"site", "invocation", "id", "name", "key", "path"} each
end
TurnWaiting(approval) = TurnWaiting(approval, Any[])
Base.showerror(io::IO, e::TurnWaiting) = e.approval !== nothing ?
    print(io, "TurnWaiting: ", e.approval.path, " waits for a person's answer") :
    print(io, "TurnWaiting: waits for ", join((repr(x["key"]) for x in e.later), ", "))
