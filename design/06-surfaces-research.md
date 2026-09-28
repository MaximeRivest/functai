# 06 — New surfaces: what we know before designing them

*Status, 2026-09-28: research brief. Nothing here is decided. It collects
what the code, the contract, Chattering, lmcc, dspy-session, ProgramIR and
our records say, so that sessions, run records, tools, packaging, serving
and distribution can be designed as one set of surfaces that fit together.
Sections 10 and 11 list what only Maxime can decide.*

Sources read for this brief (all on lambda):

- FunctAI: `contract/*.md`, `contract/schema/event.schema.json`,
  `python/functai/{core,module,streaming,calllog,saved,config}.py`,
  `ts/src/{index,fn,engine,stream,settings}.ts`, `r/NAMESPACE`,
  `julia/src/{FunctAI,settings}.jl`, `design/01-many-languages.md`, the
  website's language table (`docs/index.md` at `HEAD`).
- lmcc: `GUIDE.md` §8, `plans/12-turns.md` (turns, slots, replay, F1–F14).
- Chattering: `ai-programs.js`, `pirouter.js`, `programs-live.js`,
  `pisdk-runtime.js`, `sandbox.js`, `keyproxy.js`, `conversation-tree.js`,
  `design/66-one-tree-one-head.md`, `design/74-ai-programs.md`,
  `design/76-programs-you-ship.md`.
- dspy-session: `~/Projects/dspy_session/README.md` (docs),
  `~/Projects/dspy-community-org/dspy_session/` (code, tests,
  `docs/design-notes.md`, `docs/v2-recursive-sessions.md`,
  `docs/sessionify-vs-with-memory.md`).
- ProgramIR: `~/Projects/programir-contract/spec/{SCOPE,trust,placement,manifest}.md`.
- Records: *Live AI* (01a0e730), *Branch bug* (01a0e745), both
  2026-09-28; project memory of functai and chattering (AI-written,
  unreviewed).

---

## 1. What "composable" and "delightful" must mean here

Measurable, so a design can fail:

- **Composable.** Every new surface takes and returns things that already
  exist (an AI function, a module, a `Prediction`, a stream event, a call
  id, a saved folder), or a new thing that the other new surfaces also
  accept. No surface needs its own copy of another's concept (one "run",
  not a session run, a serve run and a stream run).
- **One execution path.** The same program retries, runs tools, streams
  and logs the same way whether called in a notebook, in a session, by
  `evaluate`, or behind an endpoint (the rule `streaming.md` already sets
  for streams: "a view, never a second behaviour").
- **Layered.** A beginner needs four verbs: define, call, remember (a
  session), save. Durability, trust, bindings and serving appear only when
  asked for, and never change what the first four do.
- **Native.** Each language spells it its own way (Python decorator,
  TypeScript values and `AbortSignal`, R copies and withr, Julia `!` and
  ScopedValues). The contract fixes data and behaviour, not syntax
  (`design/01`).
- **The words predict the behaviour.** lmcc's D-40 method: give the
  glossary and three examples to someone new, ask what each name does;
  every wrong guess is a naming bug.

## 2. Who uses these surfaces, and for what

| Who | Job | Evidence |
|---|---|---|
| Analyst or scientist in a notebook | a chat-like helper over their data; turn good conversations into test rows | FunctAI intent memory; dspy-session README examples 4–5 |
| App developer (Chattering itself first) | one program, many users; stream; survive refreshes and restarts; branch and continue | *Branch bug* 01a0e745 #69–#140; `design/66` |
| Agent builder | a tool loop with permissions, nested helpers with their own or shared memory, recovery after a crash | 01a0e745 #140; dspy-session v2 doc §1 |
| Person improving a program | rate calls, build an answer key with context, compare versions, publish only if better | `design/74` (as built) |
| Person shipping a program | save, download, serve, call from a phone or a script, keys, limits | `design/76` |
| Stranger running someone's program | know what it can do before running it; run it walled off | *Live AI* 01a0e730 #549, #580; ProgramIR `trust.md` |
| Family member, no code | a form on a phone | `design/76` "The job" |

Chattering is the first and heaviest user, but FunctAI is meant for all of
them (FunctAI intent: "an independent, open-source framework").

## 3. The vocabulary we extend (today's grammar)

Nouns already in the contract, with fixed meanings (`calls.md`, "Words";
`streaming.md`, "Words"):

- **program**: an AI function (`kind: "ai"`) or a module (`kind: "module"`).
- **call**: one call of a program, inputs to answer or error; a tree
  (`id`, `parent`, `root`, UUIDv7).
- **exchange**: one model request and its reply inside a call.
- **rating**: one person's judgement of one call.
- **version**: fingerprint of what a program sends besides its inputs;
  **not** the model.
- **stream**: one call, watched; events `started`, `text`, `thinking`,
  `tool_call`, `tool_result`, `retry`, `done`, `failed`.
- **saved folder**: `functai.json` + `code/` + locks + `files/` + `models/`.

lmcc's nouns (`GUIDE.md` §8): **turn** = one call of one signature, as
values (inputs, steps, outputs); **steps** = model replies and tool
results; **turn slots** in a template (`lmcc.turns()`, named slots, the
reserved `steps`). There is no "history slot": history is a choice of
turns made above lmcc (01a0e745 #175–#177).

Conventions every new surface must follow:

- Settings layer: the function's own > a block (`with configure`,
  `withSettings`, `with_ai_config`, `with_settings`) > `configure` >
  defaults. Same order in all four languages.
- **Improving returns a copy** (`fn.opt`, `labeled_few_shot`, R
  `with_demos`); `using(...)` makes a copy with other settings.
- Refusals before money: an error names its code and a fix, before any
  model is called.
- Plain words in docs and errors ("answer", "right", "asked again").
- JSON keys `snake_case`; each language's API in its own case.
- Unreleased APIs may break when that makes them better (FunctAI intent).

The word to settle first: the contract's **call** is one program call;
Chattering and the proposals say **run** for "everything after Send". A
session turn in Chattering can be several root calls (a module, then a
tool that calls another program). Decide whether a new noun is needed or
whether "call" (with its tree) is enough.

## 4. Facts about today's code that constrain the design

Checked in the code on 2026-09-28.

| Fact | Where | Consequence |
|---|---|---|
| Python memory is `fn.history`, a list on the function object; `stateful=True`, `state_window=5`; `reset()` | `core.py` 391, 600, 674, 757 | memory is per function, not per user; shared across users on a server |
| Reading history happens outside the lock; only the append is locked | `core.py` `_past` vs `_run` | two concurrent calls can see the same history and both append |
| The window deletes old turns (`del self.history[:-window]`) | `core.py` 761 | what the model sees and what is kept are the same list |
| TypeScript, R and Julia have no memory at all | language table; `ts/src/fn.ts` `pastTurns` reads demos only | sessions are new in three languages |
| Past turns reach lmcc only through `demos` + history, rendered into one `turns` list | `fn.ts` `pastTurns`, `engine.ts` `plan.render(turn, { turns: [...job.past] })` | no per-call context argument, no named slots |
| A TypeScript `module()` records inputs as `arg0`, `arg1`… and has no `predict`, `stream` or typed interface | `ts/src/index.ts` `module` | a composed program has no declared inputs or outputs in TS |
| Python `@module` keeps its last 100 calls in memory (`self.history`) | `module.py` `_invoke_original` | a second, unrelated meaning of "history" |
| Stream events carry `kind`, `call`, `function` and their fields; no sequence number, no time | `schema/event.schema.json`, `streaming.py` `Event` | a client cannot resume from "event 57" |
| Events live in memory in the process that made the call | `stream.ts`, `streaming.py` | another process cannot watch a call mid-flight (01a0e730) |
| The call log writes one line when a call **ends** | `calls.md` "Never in the way" | nothing on disk while a call runs; a crash loses the call |
| A tool is `{name, description, parameters, run}`; its result is text in events | `engine.ts` `Tool`, `streaming.md` | no effects, timeouts, approval, or media in results |
| The tool loop runs a tool as soon as the model asks | `engine.ts` `run` | no place to pause for permission or save before acting |
| Closing a stream cancels the call (`Cancelled`) | `streaming.md` "Closing" | cancellation exists; it is tied to a stream object |
| A saved folder's AI functions load in every language; tools and code are refused across languages | `saved.md` | a TS server can host one-step programs only |
| Python `load(path, trust=True)` is the only trust control | `saved.py` 835 | one switch, no facts |
| `requirements.lock` is "as installed here", no hashes, one platform | 01a0e730 #549 | not reproducible elsewhere |
| A saved function stores one model name (`settings.lm`) | `saved.md` manifest | no way to say "tested on X, bind Y at deploy" |
| `rated` rows are inputs + the right answer; no conversation context | `calls.md` "Rows with known answers" | conversation turns make bad rows today |
| `log_content` is all or nothing per call | `calls.md` | Chattering logs whole transcripts as sizes only, by input size |

## 5. What Chattering needs (from its code, not only its plans)

Workarounds in Chattering's code are requirements FunctAI did not meet:

- **Raw reply text while streaming.** Programs whose whole reply is the
  answer (the 13 editor commands) read the text through an
  `AsyncLocalStorage` side channel into the Pi router, because FunctAI's
  reader shows that reply only whole (`ai-programs.js`, `rawText`).
- **Per-input content control.** Chattering logs a call's values only when
  all inputs together are under 128 KiB, else sizes only
  (`ai-programs.js` `contentLimit`). It wants "log everything but this
  transcript input".
- **Caller isolation.** It sets every caller key explicitly (even to
  `undefined`) so nothing leaks from the server's own `$FUNCTAI_CALLER`.
- **One-message routers.** Pi takes one message per run, so earlier turns
  are folded into one quoted message (`pirouter.js` `piMessage`). A
  session on a one-message router silently changes the prompt's shape.
- **Live watching is rebuilt outside FunctAI** (`programs-live.js`:
  snapshots, ops, flush, per-viewer filtering), because events have no
  cursor and no persistence.
- **FunctAI is vendored** (`vendor/functai/0.1.0/functai.mjs`): not on npm.

What Chattering's agent side uses from Pi today (`pisdk-runtime.js`,
`pisdk-custom.js`), the parity list if FunctAI sessions ever run
Chattering's agents: `prompt`, `followUp`, `abort`, `waitForIdle`,
`subscribe`, `setModel`, thinking levels, `compact`, `fork`,
`navigateTree`, `newSession`/`switchSession`, `sendCustomMessage`,
`appendCustomEntry`, active tools by name, extensions (artifacts,
delegation, image budget, modes, prompt capture, records), session names.
Around it Chattering built: a conversation tree with one head per person
and parallel answers (`design/66`), git checkpoints per run
(`checkpoint-*.js`), failure classification and recovery
(`agent-recovery.js`), delegation trees (`delegation*.js`), a bubblewrap
sandbox and a key proxy that injects credentials (`sandbox.js`,
`keyproxy.js`).

From the designs: the Programs pages need, per call, the signature's
plain data (`design/74` trade-off: allowed answers are guessed from the
prompt today); `design/76` needs `contract/serve.md`, whole-program
interfaces, deploy-time models, keys and budgets, sandboxed hosting, and
pulling call logs from external hosts.

## 6. What lmcc already gives sessions

- **A turn is JSON** in every language, typed only against a signature
  (F8), and carries its signature fingerprint; a mismatch refuses
  `signature-mismatch`. Instructions are not in the fingerprint, so
  improving a docstring does not orphan recorded turns.
- **Adapter switching**: the current adapter writes past turns; recorded
  replies are reused when the plan reads them back into the same values
  (`replay: "recorded" | "values"`, adapter-level). Verbatim replay is
  not a kernel promise.
- **Slots**: `lmcc.turns()` and named slots, as messages or as text, with
  guards for empty slots; `steps` for the current turn's own work.
- **Refusals at bind**: `turns-drift`, `turns-unplaced`,
  `turns-double-placed`, `turn-incomplete`, …
- **Storage size**: a model step keeps the reply and a hash of the
  request, not the request (F7: storing requests grows quadratically).
- **Tool steps hold child turns** (RLM sub-calls), rendered one level
  deep only.
- **Out of lmcc, on purpose**: choosing, windowing, summarizing turns;
  running loops; executing tools. That is our layer.
- Still open in lmcc: live-provider checks of verbatim replay and id
  qualification (`plans/12-turns.md`, "Still open").

## 7. What dspy-session teaches

Keep (its best ideas):

- **Program separate from memory**: one blueprint, one state per user,
  bound for a block (`use_state`), serializable (`to_dict`).
- **The context each turn saw is recorded** (`history_snapshot`), so
  training rows are faithful.
- **Filtering what enters memory** (`exclude_fields`, `history_input_fields`).
- **Nested policies**: `isolation` (own or shared), `lifespan`
  (persistent, episodic, stateless), a consolidator for long memory,
  include/exclude/where for which children.
- **Training data from conversations**: `to_examples` with a metric,
  `min_score`, `strict_trajectory` (drop a bad turn and everything after).
- **Replay without touching state** (`history_policy="override"`).
- **Hot-swap the program, keep the state** (`update_module`).
- v2 invariant: *one turn per user-facing call; everything inside is a
  step* (`v2-recursive-sessions.md` §2).

Change (its weaknesses, seen in its code):

- Turns are recorded **after** they finish; there is no identity while a
  turn runs, so streaming and crashes have no place (01a0e745 #123).
- Forks deep-copy the turn list; snapshots store the whole history per
  turn (the quadratic growth lmcc F7 measured).
- A consolidator failure only logs a warning (`_finalize_macro_turn`),
  so memory silently stops updating.
- The same `forward` code is duplicated for sync and async.
- A shared blueprint without `use_state` mixes users silently
  (`sessionify-vs-with-memory.md`, caveat 1): the safe path is opt-in.
- Its README examples leak an API key (`dspy-community-org/dspy_session/README.md`):
  revoke and remove it if it is real.

## 8. What ProgramIR teaches (from `programir-contract/spec`)

- **Facts, not a trust switch**: origin (`builtin`, `packaged`,
  `authored`), authority (declared effects `pure | reads_env | network |
  stateful`, credential slots), exposure (computed: can model output reach
  it?).
- **Postures belong to the receiver**: `workbench` (own machine, no
  ceremony), `reviewed` (one summary at load), `hardened` (refuse unsigned
  code). One aggregated report, never a warning per piece; decisions at
  export and load, never mid-run.
- **Trust deficit is paid with isolation**; an ordered isolation scale
  (`none < namespace < fork < fork_cgroup < fork_ratchet < sandbox <
  remote`); floors set by the author, envelopes by the host, never under
  the floor.
- **A broker injects credentials** so secrets never enter untrusted code
  (Chattering's key proxy is this already).
- Pools of models, adapters and tools referenced by name; model identity
  separate from where it runs; a versions block with loud refusals;
  whole-module signatures (D-036); checks that run no code.
- Its own scope defers signing and sidecar wire formats; its node-set
  language (742 lines of spec) is a large, separate project.

## 9. Prior art to study before designing

Not verified in this session (no web access here; from general knowledge,
which may be out of date). Each line says what to look for.

| Project | Look at | Why |
|---|---|---|
| OpenAI Agents SDK, *Sessions* | a four-method memory protocol (get, add, pop, clear items) with pluggable stores | the smallest store interface that works |
| Pydantic AI | `message_history=` per call; `agent.iter()`; durable execution via Temporal/DBOS | per-call context without hidden state |
| LangGraph | checkpointers, threads, `interrupt()` for human approval, time travel and forks | durable state and approval, and where it gets heavy |
| Vercel AI SDK | UI message streams, resumable streams, tool approval in `useChat` | the browser side of streaming and branching |
| ellmer (R) | `chat$chat()`, `get_turns()`, `set_turns()` | what an R user expects a conversation to be |
| Temporal, Restate, DBOS, Inngest | idempotency keys, activity retries, "effectively once" | the honest limits of durable tools |
| MCP | tool annotations (read-only, destructive, idempotent, open-world hints) | an existing vocabulary for tool effects |
| Hugging Face Hub | revisions, `trust_remote_code`, model cards, safetensors | what distribution users already know |
| Replicate Cog, BentoML, Modal | typed predictor → HTTP + OpenAPI; warm workers | the serving surface to beat |
| PEP 751, uv | `pylock.toml`, hashes, platforms | reproducible installs |
| Sigstore, SLSA | signing and provenance | if signing ever ships |
| OpenTelemetry GenAI | span and event conventions | exporters, not the primary record (`design/74`) |

## 10. Contradictions found in the records

1. **Who owns hosting and sandboxing.** FunctAI's project memory lists as
   a non-goal "implementing Chattering's dashboard, hosting, billing, or
   sandboxing inside FunctAI". `design/76` D1 recommends that the server
   belongs to FunctAI; 01a0e730 #549 has a FunctAI server start programs
   "in a sandbox"; and the backlog drawn up on 2026-09-28 put trust
   enforcement and sandbox policies on FunctAI's side. A consistent split is possible
   (FunctAI declares requirements, serves, and checks; the host
   enforces walls, billing and budgets) but it has not been agreed.
2. **Pi's future.** Chattering's memory says "Pi remains the agent
   foundation while alternatives are explored"; 01a0e745 #69 says Maxime
   is "considering moving away from Pi". Whether FunctAI sessions must
   reach Pi parity (section 5) changes their scope by a lot.
3. **"Session" already means three things**: Python `stateful`, a Pi
   session (a conversation file), and in R docs "the R session". The new
   noun must not collide.
4. **"history" already means three things** in Python (a stateful
   function's turns, a module's last 100 calls, `phistory()` = requests
   sent).
5. **`design/74` Principle 2 says rat computes evaluation**; a portable
   answer key checked at server start would run evaluation inside the
   server. Decide whether that is a check (fine) or a second evaluator.

## 11. Decisions only Maxime can make

1. The boundary in contradiction 1.
2. Whether FunctAI sessions are meant to run Chattering's coding agents
   (Pi parity), or conversations with programs only, for now.
3. Order of languages: TypeScript and Python first (Chattering and
   notebooks), R and Julia after, with the contract written first either
   way?
4. Where session state lives by default: a folder FunctAI writes (like
   the call log), a store interface the host implements, or both.
5. Whether session records and call records are one log or two.
6. How far to go on trust now: facts and a load report only, or
   enforcement too.
7. ProgramIR: borrow its ideas into FunctAI's contract, or adopt its
   manifest as FunctAI's.
8. Priorities among: sessions, run records and durability, tool
   permissions, serving, Hugging Face.

## 12. What is still missing, and how to get it

| Missing | How to get it |
|---|---|
| Whether people understand the names | the D-40 word test with 3 people new to FunctAI (household members, for example) |
| Real multi-turn data to test against | Chattering's Pi conversations, words replaced (the `test/fixtures/trees` method from `design/66`) |
| Real single-step programs | Chattering's 37 programs (`ai-programs.js`) as a corpus |
| Cost of storing sessions | measure Pi session files: size per turn, with and without request hashing |
| Live behaviour of replayed turns across providers | lmcc's open item: one real request per provider (cents) |
| How R and Julia users expect conversations to look | ellmer's API; one Julia user's opinion |
| Prior art, current versions | a web session to confirm section 9 |
| Crash behaviour of tools | a fault-injection test: kill the process between "tool asked" and "tool result saved" |
| Latency budget | Chattering's measurements (`design/60-response-speed.md`) |

## 13. How the design should be made

1. **Vignettes first**, as lmcc plan 12 did: the smallest real programs
   (a chat helper in a notebook, Chattering's send, a served classifier,
   an agent that edits a file with permission) written in the proposed
   API of each language before any code.
2. **The word test** on the vignettes' names.
3. **Contract text and cases next**: session record, event cursor,
   turn context, tool declarations, manifest additions, `serve.md`;
   cases written by scripts from the rules (`contract/cases/make.py`).
4. **Prototype in TypeScript and Python** against the cases; move
   Chattering's own programs onto them as the first real use.
5. Every trade-off stated in the design note, as `design/01` does.
