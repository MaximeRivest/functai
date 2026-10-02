# Tools that ask first

A tool says what it does to the world, and a person can be asked before it runs: at once (a function you give), or later, from any process (the turn waits, saved). A turn that stopped goes on without paying for a model answer twice and without running a tool twice (contract/tools.md).

## What a tool does

```julia
"The text of one note."
read_note(name::String) = read(joinpath("notes", name), String)
"Replace a note's text."
write_note(name::String, text::String) = (write(joinpath("notes", name), text); "ok")

@ai tools = [tool(read_note; effects = :reads), tool(write_note; effects = :changes)] function gardener(request::String)::String
    "Tend the notes."
end
```

`effects = :reads`: it only looks. `:changes`: it writes, sends or pays. Left out, it is unknown, and every rule treats it as `:changes`: forgetting to declare is safe. An AI function used as a tool reads, unless one of its own tools changes things; a `@program` used as a tool counts as changing things unless it says `:reads`. [`delegate`](@ref) makes another program a tool that remembers per branch.

## `approve`

`approve` is a setting, so it works wherever settings do: `@ai`, `configure`, `with_settings`, `configure!`, a conversation, one turn.

```julia
with_settings(approve = a -> a.name == "write_note" && a.input["name"] != "todo.md" ? "only todo.md" : true) do
    gardener("Tidy the notes.")
end
```

- **A function** is asked at once about each tool call that changes things (or says nothing), with an [`Approval`](@ref) (`name`, `input`, `effects`, `path`, `invocation`): it answers `true`, `false`, or a reason to refuse.
- **A rule** is answered by a person: `:changes` (tools that change things or say nothing), `:all`, or a list of tool names and approval paths (`"support/answer/refund"`, or its ending `"answer/refund"`): names, not ids, so a host writes rules before any call exists.

A refusal is an answer the model sees: the tool's result is `The person did not allow this call.`, then the reason; the model may try something else.

## Who answers a rule

- **In a conversation's turn**, the turn stops and waits, saved: the call throws [`Waiting`](@ref) (`turn-waiting`), holding the approvals. Any process that opens the conversation answers, and the turn goes on there:

  ```julia
  chat = conversation(gardener, "notes"; store = "notes-chats/", approve = :changes)
  try
      chat("Merge groceries.md into todo.md.")
  catch err
      err isa Waiting || rethrow()
  end
  t = last(turns(chat))
  t.waiting                                  # what waits: [Approval(gardener/write_note #2, changes)]
  approve!(t; by = "ana")                    # from this process or any other: the turn goes on, and its answer is returned
  deny!(t; reason = "not today")             # or no
  ```

- **On a stream** without a conversation, the call waits in this process for the stream's answer: `approve!(s)` or `deny!(s; reason)`.
- **A plain call** has nobody to ask: it throws [`ApprovalError`](@ref) (`approval-required`) before the tool runs.

## Going on without paying twice

A turn records, as it runs, each model reply (for a program's turns and an AI function's with tools) and each tool that changes things, before it runs (`started`) and after (`done`, with its result). [`resume!`](@ref) runs the turn's program again with the same inputs and earlier turns: a request it made before gets the reply recorded for it (nothing paid twice), a tool that ran gets its recorded result (nothing run twice), and the turn goes on from where it stopped, at any depth. Its log goes on as a later writer's (`writer = 2`), and only what it had not done before is shown again.

A tool whose last record is `started` **may have run** (the process died while it ran). It is never run again on its own: `t.unfinished` lists it, and the turn goes on once a person says what it returned, or asks for it to run again:

```julia
t = last(turns(chat))
t.state                                      # "interrupted"
t.unfinished                                 # [(invocation = 1, name = "send_email", input = …)]
resume!(t; results = Dict(1 => "sent"))      # it did run: this is what it returned
resume!(t; rerun = [1])                      # it did not: run it again
abandon!(t)                                  # or end the turn without going on
```

A program whose code is not deterministic given its inputs and replies makes new requests when resumed: they cost, never mislead.

## The journal waits only before tools that change things

A required journal (`FunctAI.Journal(store; required = true)`) keeps a tool call before a tool that changes things runs; a tool that only reads runs without waiting for it. Each tool call is numbered within its call (`invocation`, 1, 2, … across every step: the model's ids repeat across replies), and every call a tool makes carries that number.
