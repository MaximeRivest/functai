# 11 — Plugins: the core the trust-and-improve loop needs

*Status, 2026-10-02: built and tested on the branch `plugins`, in Python;
the contract is `contract/plugins.md`. Builds on `10` (stages 1.2 to 5) and
on Maxime's ask (2026-10-02): hooks over tools, context and model choice,
changes recorded as data so replays and ratings stay trustworthy, approval
rebuilt on them, compaction and delegation as built-ins: what lets
Chattering's agents run on FunctAI. Not a Pi clone: only the hooks real
plugins use.*

## Evidence first

The hooks come from what extensions actually do, not from Pi's list:

- Maxime's 22 (14 in `~/.pi/agent/extensions`, 8 in Chattering) use about
  ten of Pi's ~30 hooks: session start, input, prompt sections and active
  tools, the raw request, the tool call, the context sent, model choice,
  the end of a turn.
- A friend's 34 packages (`tryingET/pi-extensions`, 735 files) use the
  same set plus vetoes before branching and compaction, and a great deal of
  host interface (notifications, the editor, custom screens, shortcuts),
  which is not FunctAI's.

So API 1 has seven hooks (`turn_start`, `context`, `before_call`,
`request`, `tool_call`, `tool_result`, `turn_end`). A hook is added when a
real plugin needs it.

## Decisions

1. **Changes are data.** A hook returns a `Change` (sections, instruction,
   model, settings, tools, turns kept, fields left out, inputs, a block, an
   output), never a rewritten message list. *Why:* FunctAI's promise is that
   a rated call can be asked again as it was; data can be recorded and
   applied again, arbitrary code cannot. Pi itself steers its authors to
   structured prompt sections for the same reason.
2. **The escape hatch is kept, and marked.** `request` may replace the
   provider request: the call says `replayable: false` and that exchange
   has no `request_hash`. *Why:* without it, people fork; with it unmarked,
   ratings lie.
3. **What is replayed: the conversation, not the host.** A rated call is
   asked again with what it was shown of its conversation (earlier turns,
   the sections `context` hooks gave: a summary). How the host shaped it
   (`before_call`, `request`) is not given back: the evaluating process's
   own plugins apply. *Why:* a host mode that replaced the instruction,
   replayed, would hide the very instruction an optimizer improves; the
   program under test must be what is measured.
4. **Order: the program's own first, the host's last.** Then each layer
   from the innermost out; in a layer, as listed; a plugin set twice runs
   once, in its outermost layer; a host may drop a program's own
   (`program_plugins=False`). The `approval` plugin runs after every other,
   on the input they left. *Why:* the host's policy must judge what will
   actually happen.
5. **Failing closed.** A hook that changes things and raises stops the call
   (`plugin-failed`): a redaction that did not happen must not pass as if
   it had. A `tool_call` handler that raises blocks the tool only. A
   `turn_end` that raises is reported and changes nothing. *Why:* the safe
   failure differs by what the hook guards.
6. **Approval is a plugin.** `approve=` drives the built-in `approval`
   plugin; any plugin's `tool_call` may ask a person (`tool.ask(reason)`),
   with the same waiting, resuming and recording. Questions are addressed by
   site, invocation and plugin, so two plugins can ask about one tool call.
   *Why:* Pi's test of an extension system: built-ins use no private hook.
7. **Entries belong to branches.** A plugin keeps what it needs as
   conversation entries at a turn; it reads them on the branch it runs in.
   Mode switches, summaries and delegations follow the branch. *Why:* the
   Pi rule ("tool state in tool results follows the branch") generalized.
8. **A turn's context is recorded when a hook shaped it**, and resuming the
   turn shows exactly that, with no hook run again; `turn_end` runs before
   the turn's end is recorded, so the next turn (which waits for that end)
   sees what it kept. A tool's result is recorded after the `tool_result`
   hooks and replayed as is.
9. **Compaction** is a rolling summary: `turn_end` folds all but the last
   `keep` turns once `keep + every` are open (so not one call per turn), and
   `context` shows it as a section with the turns after it. Per branch.
   **Delegation** is a tool: the delegate's own conversation
   (`<conversation>.<tool>`), continued from the delegation recorded on this
   branch.
10. **The word is "plugin".** lm15's `extensions` setting already exists
    (provider-specific settings), and every lm15 setting is a FunctAI
    setting by the same name. *Rejected:* dropping lm15's setting (breaks a
    documented field); one name meaning two things by the value's type.
11. **API version 1, and the contract's cases.** A plugin says which API it
    targets; FunctAI refuses one it does not implement rather than run it
    wrongly. Adding a hook or a field is not a new version; changing what one
    means is.

## Proven on real extensions

Six were written again as plugins (`python/examples/plugins/`, run by
`tests/test_ported_plugins.py`): modes, session-prompt, auto-inject,
prompt-capture, checkpoints, image-budget (as a budget on any bulky field).
Writing them changed the design twice: `before_call` may replace the
instruction (modes and session-prompt replace, not only add), and it is
given the instruction as it stands; tool hooks may keep entries
(checkpoints). Delegation and compaction are built in.

## Trade-offs taken

- **Plugins are less powerful than Pi's extensions** by design: no hook
  rewrites messages except the marked escape hatch.
- **They run in the process, trusted, with no time limit.** A plugin is code
  the host chose; isolating untrusted plugins (in another process, over a
  protocol, which would also let one plugin serve every language) is not
  built.
- **A plugin that fails stops the call** (except a tool check, which blocks
  the tool). Safer, and a buggy plugin is loud.
- **`turn_end` adds to the turn's time** when compaction summarizes (rarely:
  every `every` turns), because it runs before the turn's end is recorded.
- **Only `tool_call` can ask a person.** Choosing, typing a value, notifying,
  and where a host shows them, are not in API 1.
- **Compaction's summary is shown to AI functions only**: a module's code
  gets the turns after it through `earlier()`, not the summary.
- **No host features**: commands, shortcuts, status lines, editor text,
  custom screens are the host's. A plugin offers functions
  (`modes.switch`) the host binds.
- **Python only**, as for stages 1.2 to 5.

## Not done

- Asking a person anything but "may this tool run" (choose, type, notify),
  through the host.
- Plugins in another process (isolation, and one plugin for four
  languages).
- Starting a turn from a plugin (Pi's `sendUserMessage`), with a loop
  guard; vetoes before branching and compaction (the friend's rewind).
- A JSON-lines command that runs a turn and streams its events (sub-agents
  as processes, in any language).
- TypeScript, R, Julia.
