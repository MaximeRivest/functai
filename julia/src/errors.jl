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

A turn stopped to wait for a person's answer (code `turn-waiting`): a tool
call its `approve` rule (or a plugin) asks about, with no function to ask.
`turn` is the [`Turn`](@ref) and `approvals` what waits;
[`approve!`](@ref) or [`deny!`](@ref), from any process that opens the
conversation, answers and resumes it.
"""
struct Waiting <: FunctAIError
    code::String
    msg::String
    turn::Any
    approvals::Vector{Any}
end
Waiting(msg::AbstractString; turn=nothing, approvals=Any[]) = Waiting("turn-waiting", String(msg), turn, Any[approvals...])
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
A turn stops to wait for a person (inside the call that asked; the turn's
end turns it into [`Waiting`](@ref)). Code inside a call never treats it as
an ordinary failure: no call it passes through ends, and nothing is recorded
as failed.
"""
struct TurnWaiting <: Exception
    approval::Any
end
Base.showerror(io::IO, e::TurnWaiting) = print(io, "TurnWaiting: ", e.approval.path, " waits for a person's answer")
