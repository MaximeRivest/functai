# Conversations (format 1)

A **conversation** is a program's calls that remember each other: each
call is a **turn**, which is shown the turns before it. Memory belongs to
the conversation, never to the program: the program is unchanged, and
still callable on its own. Nothing is ever deleted: continuing from an
earlier turn makes a **branch** (Python `python/functai/conversations.py`,
`python/functai/stores.py`). `cases/conversations/` pin a turn's state,
the head, where a new turn goes, and what a turn is shown;
`schema/conversation.schema.json` checks a record.

## Words

- **Turn**: one call of the conversation's program, made in the
  conversation. Its id is its call's id ([calls.md](calls.md)), minted
  when the turn is recorded, before the call starts: a page knows where an
  answer goes before it is written.
- **Parent**: the turn a turn continues. A turn has at most one parent, so
  the turns are a tree. A merge reads other turns (`reads`) without making
  them parents.
- **Branch**: the turns from the first to one turn, following parents.
- **Head**: the conversation's latest turn, or the turn a `head` record
  names, whichever comes last.
- **Store**: where a conversation's records are kept (*Stores*).

## Records

A conversation is an ordered list of records, appended and never
changed. Every record is a JSON object with `functai_conversation: 1`, a
`kind`, `at` (the call log's time format), and `seq`, its position (1 for
the first), which the store gives it. A reader skips a record of a
format or a kind it does not know, and keys it does not know.

| kind | written | keys |
|---|---|---|
| `program` | before the first turn of a program version | `version`, `name`, `program_kind`, `module`, `signature` (AI functions), `interface` (whole), `fields` (each field's `name`, `direction`, `purpose`, `shape` without defaults), `answer`: what a reader needs to show that version's turns. Keyed by the version, never by the interface's signature alone. |
| `turn` | when a turn is sent, before its call | `turn`, `parent` (a turn, or null), `program` (its version), `inputs` (bound, as the call record holds them, after the `turn_start` hooks), `request_id`?, `settings`? (`lm` when the turn chose one), `reads`? and `made_by`? (a merge), `context`? (what it was shown, when a `context` hook ran) and `changes`? ([plugins.md](plugins.md), *Records*) |
| `lease` | with the `turn` record, then every 10 seconds while it runs, and when a process resumes it | `turn`, `holder` (`<host>:<pid>:<random>`), `until` (the lease's end, 30 seconds on), `attempt` (1, then one more for each resumption) |
| `ended` | when the turn's call ends | `turn`, `state` (`done`, `failed`, `stopped`, `abandoned`), `outputs` (done: every output, the fields FunctAI adds included), `value` (what the code returned), `lmcc` (an AI function: the lmcc turn its call made, with its steps), `saw`, `error`, `model`, `usage` (tokens, summed over every call inside), `seconds`, `attempt` |
| `waiting` | when the turn stops to wait for a person ([tools.md](tools.md)) | `turn`, `approvals`, `saw`, `attempt` |
| `approval` | a person's answer | `turn`, `site`, `invocation`, `plugin` (the plugin that asked; absent: `approval`), `path`, `verdict` (`yes`, `no`), `by`, `reason` |
| `entry` | a plugin keeps something ([plugins.md](plugins.md), *Entries*) | `plugin`, `entry` (its kind), `data`, `turn`? (its branches; absent: every branch) |
| `tool` | before a tool that changes things runs (`started`), after a tool ran (`done`), or when a person says what a tool that may have run did (`given`) or asks it to run again (`rerun`) | `turn`, `site`, `invocation`, `id`, `name`, `state`, `input`? (started), `effects`?, `output` (done, given) |
| `reply` | each model reply of a turn that can be resumed (a module's, or an AI function's with tools) | `turn`, `key` ([replies.md](replies.md)), `response` (lm15's canonical JSON), `attempt` |
| `call` | when a helper the conversation remembers ends a call in a turn | `turn`, `call`, `site`, `program` (`name`, `module`, `signature`), `lmcc`, `saw`, `attempt` |
| `stop` | anyone asks a running turn to stop | `turn` |
| `head` | the head is moved to a turn | `turn` |

`site` is a call's place in its tree by names and order: the outermost
call's name, then each child's, each with its occurrence among its
parent's children of that name (`support#1/answer#1`). Resuming a turn
runs its program again, and finds what each call did by its site.

## A turn's state

From its records, in order:

- an `ended` record: its `state`;
- else a `waiting` record after its last `lease`: `waiting`;
- else its last lease's `until` not before now: `running`;
- else `interrupted`: its process stopped (its lease ran out) before the
  turn ended.

A turn's **unanswered** approvals are those of its last `waiting` record
that no later `approval` names (by `site`, `invocation` and `plugin`). Its
**unfinished** tools are those whose last `tool` record for a `site` and
`invocation` is `started`: they may have run.

## Where a new turn goes

- A conversation opened by id follows its head. A view made by
  "continue from this turn" continues from that very turn; once it made
  a turn, it follows its own branch: the latest turn that is its last one
  or continues it (a turn another process adds after it is followed; a
  branch made elsewhere is not).
- The new turn's parent is where the view continues, when that turn ended
  `done`; else its nearest ancestor that did (a failed, stopped or
  interrupted turn has no answer to show, and is never a parent).
- When that turn is `running`, two sends at once follow the
  conversation's `sends` setting, which is the host's policy: `queue`
  (the default: wait until it ends, then continue from it: the second
  send sees the first), `refuse` (`conversation-busy`), `branch`
  (continue beside it, from its parent). When it is `waiting`, a send is
  refused (`conversation-busy`) unless `branch`: a person's answer is
  pending.
- The `turn` record is appended conditionally on the conversation's
  length (*Stores*): when another turn was recorded meanwhile, the
  writer reads again and decides again.
- **The same `request_id` twice is one turn**: a send whose `request_id`
  a turn already has gets that turn (watched while it runs, its result
  once it ended), and records nothing.

## What a turn is shown

A turn is shown the turns of its branch before it that ended `done`,
picked by the conversation's **context rule**: every one (the default),
or the last *n*; each without the fields the rule's `without` names (a
bulky input, left out of earlier turns). Then the `context` hooks of its
plugins may change which turns are shown, leave out more fields, and add
sections to the instruction ([plugins.md](plugins.md)); when one ran, the
turn records what it was shown (`context`), and resuming it shows exactly
that. The turn's call records them in
its `saw` ([calls.md](calls.md), *Saw*):

- an AI function's earlier turn made for the same `program.signature` is
  shown whole, with its steps (the `lmcc` turn its `ended` record keeps):
  `{"call": <turn>, "steps": true}`;
- one made for another signature is shown by its values, the fields the
  program still has: `{"call": <turn>}`, with `without` naming the fields
  it had that were left out;
- one with a field the rule leaves out is shown by its values, without
  its steps: `{"call": <turn>, "without": [...]}`;
- when the entries are exactly what the parent saw, then the parent, the
  turn writes `[{"saw_of": <parent>}, <the parent's entry>]`, so a record
  does not grow with the conversation.

A **module**'s turn records the same entries (without steps): it was
given the conversation so far. Its code reads it as data with
`earlier()`: one row per earlier turn shown, its inputs and outputs by
name, without the rule's fields.

**The program changed.** A conversation whose program now has a field
its earlier turns' program did not, whose field changed its shape, or
that lost one, refuses `conversation-signature`, except that an output
FunctAI adds (`reasoning`, `calls`) that earlier turns lack is accepted
when named in `earlier_without`: earlier turns are shown without it, and
nothing is rewritten. The model may change freely, per conversation or
per turn; each turn records the model that answered.

## Helpers' memory (stage 3)

Inside a module's turn, the AI functions it calls remember nothing, unless
the conversation says so, per helper:

- `"conversation"`: its own calls in the turns of this branch (the `call`
  records of those turns' last attempts), then its earlier calls in this
  turn;
- `"turn"`: its earlier calls in this turn only;
- either, with steps (`remember("conversation", steps=True)`): shown with
  their tool steps; without, shown by their inputs and outputs.

A helper's call records what it was shown in its own `saw`. A
conversation used inside another conversation's turn is refused
(`conversation-nested`), unless the outer conversation declares it
(`remembers={inner: "own"}`): it then keeps its own turns.

## What a conversation refuses

- `conversation-id`: an id that is not 1 to 200 ASCII letters, digits,
  `.`, `_` or `-`, starting with a letter or digit (it is a file name
  everywhere, never a path).
- `conversation-content`: a store that keeps records beyond the process,
  for a program a `log_content` layer drops a field of: the host said
  never keep it, and a conversation must remember it. It refuses; it
  never forgets silently.
- `conversation-opaque`: a program with an opaque field
  ([programs.md](programs.md)): a turn is kept as data.
- `conversation-signature`, `conversation-busy`, `conversation-nested`
  (above); `turn-unknown`, `turn-state`, `turn-unfinished`
  ([tools.md](tools.md)).

## Stopping, leases, the lease that ran out

- A running turn's process renews its lease every 10 seconds, for 30.
- **Stopping from anywhere** is appending a `stop` record. The process
  running the turn reads its conversation at least every second and
  stops the call (it ends `stopped`, its call's error `Cancelled`).
- A turn whose lease ran out is `interrupted`. Only then may another
  process go on with it (resuming, [tools.md](tools.md)), by appending a
  `lease` record conditionally: the first claimant holds it until its own
  lease runs out.

## Stores

A store keeps each conversation's records. It has:

- `append(conversation, records, expect=None)`: adds the records at the
  end, all or none, giving each its `seq`; with `expect`, only when the
  conversation holds exactly `expect` records, else it refuses
  `store-conflict`. Returns how many it holds after. This compare-and-set
  is the one step every rule above needs (a parent, a lease, a claim).
- `read(conversation, after=0)`: the records after position `after`, in
  order.

and may have:

- `events`: a store of call tree logs by [streaming.md](streaming.md)'s
  rules (*The rules a store keeps*), where each turn's call tree keeps its
  kept form while it runs, so another process (a page after a reload)
  follows it;
- `wait(conversation, after, timeout)`: returns when a record after
  `after` exists (or the time is up), so a reader need not ask again and
  again;
- `durability`: how surely an append is kept when it returns (`memory`,
  `disk`: written and flushed);
- `persistent`: whether it keeps records beyond the process (unknown:
  yes).

The folder store keeps `<folder>/conversations/<id>.jsonl` (one record a
line), locks `<id>.lock` for each append (across processes), flushes
before it returns, and keeps each turn's log in
`<folder>/trees/<tree>.jsonl` (its claims' writer number in
`<tree>.writer`). Files are readable by their owner only.

A reader that resumes a turn's events after a reload names its last event
by its position (writer and seq); a server's resume token (an SSE
`Last-Event-ID`) is `<writer>-<seq>` ([serving.md](serving.md)).

## Merging

A merge (vignette 9) is a turn of the conversation's program whose answer
another AI function made from branches of the turn it follows: its
`turn` record has `reads` (the branches) and `made_by` (the merging
program's name, version and call), and its `ended` record the merged
answer as the program's answer (and the lmcc turn of its values). The
next turn sees it as the program's own earlier turn. Its rating belongs
to the merging program's call.
