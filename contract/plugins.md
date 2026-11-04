# Plugins (API 1)

A **plugin** is a named, versioned set of **hooks**: functions FunctAI calls
at fixed points of a call or a conversation's turn. A hook either
**changes** something, by returning a change as data, or only **hears** of
something. Every change is recorded, so a rated call is asked again as it
was. FunctAI's own features (approval, compaction, delegation) are plugins
written with these hooks and nothing private: anyone can replace them
(Python `python/functai/plugins.py`, `builtins.py`). `cases/plugins/` pin
the order and how changes combine.

The word is *plugin*, not *extension*: lm15's request settings already have
`extensions` (provider-specific settings), and every lm15 setting is a
FunctAI setting by the same name.

## A plugin

`name` (lower-case letters, digits, `_`, `-`, at most 64, starting with a
letter: its changes and entries are named by it), `version` (its own,
recorded with each change), `api` (the plugin API it was written for: `1`;
another refuses `plugin-api`), and its handlers by hook. Registering a
handler for a hook that does not exist refuses `plugin-hook`: a misspelt
hook would otherwise never run.

## Hooks

| hook | when | changes | what a handler is given |
|---|---|---|---|
| `turn_start` | a conversation's turn is sent, before it is recorded | `inputs` | the inputs, the conversation, the turn it continues |
| `context` | the turn's earlier turns are chosen ([conversations.md](conversations.md), *What a turn is shown*) | `keep`, `without`, `sections` | the earlier turns the conversation's rule picked, the sections so far, the conversation |
| `before_call` | an AI function's call is about to be asked (each call, helpers included) | `instruction`, `sections`, `lm`, `settings`, `tools`, `stop` | the function, its bound inputs, its instruction, model, settings, the tools offered and its own, its place in the tree, its conversation and turn, what its tree spent |
| `step` | a call with tools is about to ask the model again, after the tools of its last reply ran (*The step hook*) | `tools`, `reminder`, `replace`, `stop` | the function, its place in the tree, the step, `max_steps`, every tool result so far as shown now, the tools offered now and at its start, its model, what its tree spent, its conversation and turn |
| `request` | a provider request is about to be sent | the request itself (the escape hatch) | the lm15 request |
| `tool_call` | a tool is about to run | `inputs`, `block`; or ask a person | the tool, its input, effects, approval path, invocation |
| `tool_result` | a tool ran | `output` | the tool, its input, its output (a text or a list of parts: [tools.md](tools.md), *What a tool returns*) |
| `turn_end` | a conversation's turn ended, before its end is recorded | (hears) | the turn's outcome, the conversation; it may keep entries |

A handler returns a change (Python `functai.Change(...)`) or nothing (no
opinion). A change names only fields its hook takes; another refuses
`plugin-change`, as does a value that does not fit (a tool the function
does not have, a turn not done on the branch, an input the program does
not take).

**The fields.**

- `instruction`: the instruction itself, for this call (a mode's own system
  prompt); `sections`: text added to the instruction, after it, in order.
- `lm`: the model for this call; `settings`: lm15 settings for it.
- `tools`: the names of the tools offered, from the function's own.
- `keep`: the ids of the earlier turns shown (done turns of the branch);
  `without`: fields left out of earlier turns, a list for every turn or
  `{turn id: [names]}`.
- `inputs`: inputs replaced, by name (the turn's, or the tool's).
- `block`: the tool may not run; the reason.
- `output`: the tool's result as the model is shown it: a text or a list
  of lm15 parts.
- `stop`: the call stops here, and why (a text): it raises `PluginError`
  `plugin-stopped`, naming the plugin and the reason; the change is kept
  in the call's record first. The handlers after it do not run.
- `reminder`, `replace` (`step`): *The step hook*.

## Order

The plugins around a call run in this order: the program's own (its
`plugins` setting), then each layer around it from the innermost out (a
conversation's, each enclosing block), then the process's (`configure`).
Within a layer, plugins run in the order it lists them. A plugin set in
several layers runs once, in its outermost layer. Within a plugin,
handlers run in the order they were registered. So the host's
handlers see what the program's did, and have the last word.

- A host layer's `program_plugins: false` drops the program's own.
- Each handler is given the event as the changes before it left it.
- `sections` accumulate; `instruction`, `lm`, `tools`, an input are
  replaced by a later change; `settings` merge, a later key winning;
  `keep` replaces the turns shown, and `without` adds to the fields left
  out (of the turns shown in the end: a turn not shown has none).
- `tool_call`: the first `block` ends the hook (no later handler runs);
  so does a person's refusal. The `approval` plugin, which the `approve`
  setting drives ([tools.md](tools.md)), runs after every other handler,
  on the input they left: a host's rule judges what will run.

## The step hook

A call with tools asks the model, runs the tools of its reply, and asks
again, up to `max_steps` requests ([functions.md](functions.md)). Before
each request after the first, its `step` handlers are given:

- `step`: the number of the request about to be made (2 for the first
  after tools), `max_steps`, and whether it is the last the call may make;
- `results`: every tool result of the call so far, oldest first, as the
  model is shown it now (after earlier `replace` and `reminder` changes),
  each with its `invocation`, `name` and `input`;
- `tools`: the names offered now; `all_tools`: those the call was offered
  when it started (after `before_call`); the model; what its tree spent
  (*What a tree spent*); its place in the tree, conversation and turn.

Changes:

- `tools`: the tools offered from this request on, from `all_tools`
  (`[]`: none: the model must answer).
- `reminder`: a text the model is shown after this step's last tool
  result (a text part added to its end), in this request and every one
  after.
- `replace`: `{invocation: result}`: earlier tool results shown as this
  from this request on (a picture taken out, a long text cut).
- `stop`: the call stops (*The fields*).

What the model is shown changes; what happened does not. The call's lmcc
turn, its `tool_result` events and a conversation's `tool` records keep
every result as the tool gave it. Each exchange of the call log holds the
request it sent, and its `request_hash` is lmcc's hash of that request,
rendered from the turn as the changes show it: the changes are recorded
(*Records*), so the call stays replayable, unlike a `request` change. A
resumed turn runs its `step` handlers again: when one decides otherwise
(a time limit), the request differs, and a new one is sent (it costs,
never misleads).

**What a tree spent.** `before_call` and `step` handlers are given what
the call tree spent so far in this process: each request a provider
answered, as its model (as asked) and lm15's token counts; replies from
the reply cache and replies a resumed turn reuses cost nothing now and
are not in it. From it: the number of requests, the counts summed as a
call's are ([calls.md](calls.md), *Token counts*), the tokens
(`total_tokens` summed; unknown when a request did not report it), the
seconds since the tree's outermost call started, and the cost at prices
the host gives: each request's lm15 `InferencePricing.estimate`, summed,
exact or unknown (a model with no price, or a request whose counts do not
settle its price). A model named `provider:model` is priced by its own
name when the prices do not name it whole. FunctAI ships no prices: they
change, and a wrong one would mislead.

## Plugins and layouts

Plugins work on data before lmcc lays a call out (its layout: an adapter
or a template, [functions.md](functions.md)), and on the provider request
after (`request`). They never read or write the layout itself, with these
consequences:

- `instruction` and `sections` change the instruction lmcc is given, and
  land where the layout writes `{instruction}` (the system message, for
  the built-in layouts). A layout that never writes it (a template without
  `{instruction}`), or a baked model (it reads only the message it was
  trained on), would drop them while the record says they were sent: the
  call refuses `plugin-change` instead, before anything is sent. A
  template that writes `{instruction}` gets them there.
- Earlier turns (`keep`, `without`) are written by the layout's turn slot,
  as lmcc writes any turn; a layout with no turn slot refuses them
  (lmcc's `turns-unplaced`), as it does without plugins.
- `tools` changes the tools the request offers; how they are offered is the
  layout's.
- The program's version is computed without plugins: they change calls,
  not programs.
- Sections change the start of the request: a provider's prompt cache keeps
  its prefix only while they stay the same (compaction changes its summary
  every `every` turns, not every turn).

## When a handler fails

- A `stop` change is not a failure: the handler decided, and the call
  stops with `plugin-stopped` and the handler's reason.
- A handler of a hook that changes things that raises stops the call
  (`plugin-failed`, naming the plugin and the hook): the change it was
  meant to make (a redaction, a guard) did not happen, and the call must
  not go on as if it had.
- Except `tool_call`: a handler that raises **blocks the tool** (the model
  is told a check failed) and the call goes on.
- `turn_end` hears only: a handler that raises is reported, and changes
  nothing of the turn.

A handler runs in the call's own thread, in the process: a plugin is code
the host trusts, with all its power. No time limit is enforced (a running
function cannot be stopped safely in every language); a handler that waits
on something should watch the call being closed.

## Records

A call's record ([calls.md](calls.md)) keeps:

- `changes`: each change made to it, in order: `{"plugin", "version",
  "hook", "change"}`, the change as data (a `request` change is
  `{"request": "replaced"}`: the request is in the exchange; a `step`
  change's `replace` is keyed by the invocation as text, `{"3": ...}`). The changes
  of the turn's `turn_start` and `context` hooks are on its turn's call.
  When content is not whole, each keeps its plugin and hook, not what it
  changed.
- `sections`: the sections its conversation's `context` hooks gave it
  (what it was shown of its conversation). Only when content is whole.
- `replayable: false`: a `request` handler replaced a request. That
  exchange has no `request_hash` (no one could rebuild it).

A turn's record ([conversations.md](conversations.md)) keeps its
`changes` (turn_start, context) and, when a `context` hook ran, the
`context` it was shown: `{"turns": [ids], "without": {id: [names]},
"sections": [...]}`. Resuming the turn shows exactly that, with no hook
run again.

## Asking a turn again

A rated call is asked again (evaluating, optimizing) with what it was
shown **of its conversation**: its earlier turns and its context
`sections` (*Rows that keep their context*, [calls.md](calls.md)). No
plugin's `turn_start` or `context` runs again. How the **host shaped** it
(`before_call`, `request`) is the environment's: asked again, the call is
shaped by the plugins of the process that asks, never by recorded ones.
An instruction a mode replaced is not given back: the program under test
is what is measured, its improved instruction included.

## Entries

A plugin keeps what it needs later as **entries** of a conversation: an
`entry` record `{"plugin", "entry" (its kind), "data", "turn"?}` at a turn
(it then belongs to the branches through that turn), or with no turn (to
every branch). A hook reads its plugin's entries on the branch of the
turn it runs in, and `turn_end`, `before_call`, `tool_call` and
`tool_result` may keep entries at that turn. A host keeps one too (Python
`chat.remember(plugin, kind, data)`): switching a mode is an entry. An
entry is never shown to a model by itself; a `context` hook makes it a
section when it should be (compaction's summary). A tool's own state
belongs in its results, which follow the branch.

## Asking a person

A `tool_call` handler may ask a person whether the tool may run (Python
`tool.ask(reason)`), or decide in a person's place (`decide=`). The
question is addressed by the asking call's site, the invocation and the
plugin, and recorded and answered as [tools.md](tools.md) says (a turn
waits, saved, and goes on when someone answers). Other kinds of question
(choose one of several, type a value, notify) and where a host shows them
are not in API 1.

## Built-in plugins

- **`approval`** (the `approve` setting): its `tool_call` asks a person,
  or the function given, about the tool calls the rule selects.
- **compaction** (`compaction(keep, every, summarize)`): its `turn_end`
  folds every turn but the last `keep` into a summary once `keep + every`
  are open, with an AI function, and keeps it as an entry at that turn;
  its `context` shows the summary as a section and only the turns after
  it. Each branch has its own summary.
- **delegation** (`delegate(program)`): a tool, not a hook: asked in a
  turn, the program answers in a conversation of its own
  (`<conversation>.<tool name>`, same store), continued from the
  delegation recorded on this branch (an entry of `delegate`), so it
  remembers what it was asked earlier on this branch only. Its calls are
  under the tool call in the asking turn's tree.
- **limits** (`limits(tokens, cost, seconds, prices, then)`, named
  `limits`, version 1.0.0): opt-in limits on a call tree. Nothing is
  limited by default. At least one of `tokens`, `cost`, `seconds` (each a
  number above 0); `cost` needs `prices` (a model's name → its
  `InferencePricing`); `then` is `answer` (the default) or `stop`. A limit
  is **reached** when the tree's tokens, cost or seconds are at or above
  it, or when its count is unknown (it cannot be checked). Each reached
  limit is named `<n> tokens`, `cost <n>`, `<n> seconds` (a whole number
  without a fraction), in that order, with, when its count is unknown,
  why: `a request did not say how many it used`; `no price is known for
  <models, sorted, joined by ", ">`; or `a request's cost is not known
  from its token counts (a price without cache_read_per_million and
  cache_write_per_million is exact only for requests that report their
  cache counts; give both, equal to input_per_million when the provider
  does not charge them apart)`. Its `stop` reason is `limit reached: `
  and each limit, `<limit> (<why>)` when there is a why, joined by `; `.
  - `before_call`: a limit reached stops the call; so does a cost limit
    when the call's model is named and has no price (why: `no price is
    known for <model>`), before anything is sent.
  - `step`: a limit reached with `then` `stop` stops the call. With
    `answer`, while tools are offered, a limit reached changes `tools`
    to `[]` and `reminder` to `This task has reached its limit
    (<limits, joined by ", ">). Do not ask for any more tools: give your
    answer now, from what you have found so far.`; and at the last step
    (`step` = `max_steps`), with no limit reached, the same with
    `reminder` `This is the last step this task may take. Do not ask for
    any more tools: give your answer now, from what you have found so
    far.`, so the call ends with an answer instead of `StepLimit`.
  Limits are checked between requests: one request may go past one.
- **keep_pictures** (`keep_pictures(last, at_once)`, named
  `keep-pictures`, version 1.0.0; `last` 0 or more, default 3; `at_once`
  1 or more, default 1): its `step` counts the image parts of the
  results as shown now, from the newest (the last part of the last
  result) back; every one after the first `last` is due to go. When at
  least `at_once` are due, it changes `replace`: each result holding one
  is shown with each such part replaced by the text part `(A picture was
  here. It is no longer shown, to save space.)`, its other parts as they
  are. A provider's prompt cache keeps a request's start only while it
  stays the same, and every replacement changes it: `at_once` drops
  several at a time.

`cases/agents/` pin `limits` (`limits-*`), what a tree spent
(`spent-*`) and `keep_pictures` (`pictures-*`).

## Versions

API 1 is this document. A change to what a hook gives, takes or means is a
new API number; a plugin says which it was written for, and a FunctAI
refuses one it does not implement rather than run it wrongly. Adding a
hook or a field is not a new number (a plugin that does not use it is
unchanged).
