# 18 — Setup: what a program runs on, and what its tools need

*Status, 2026-11-04: built in Python and TypeScript on the branch
`setup-and-events`, with the contract (calls.md *Setups*, *Reach*,
*Spending*; plugins.md, tools.md, saved.md) and its cases (`setup/`,
`agents/spent-04` to `06`). R and Julia do not have it yet. What was built,
and where it differs from the proposal below, is at the end (*What was
built*). The proposal is kept as it was written.*

## Why

FunctAI names a program three ways today, each computed from data, before
any call, the same in every language:

| name | covers | answers |
|---|---|---|
| `program.interface` | what goes in and what comes out | "can I put this program where that one was?"; ratings pool by it |
| `program.version` | the instruction, worked examples, layout, the program's own code, defaults, tools' names, descriptions and parameters | "is my new instruction better than the old one?" |
| `program.saved` | the bytes of the `functai.json` it was loaded from | "is this the folder I shipped?" |

None of them names **what the program runs on**: which model answers each
step, with which settings, with which tool code, in which surroundings.
calls.md says so on purpose: *"The model and its sampling settings are not
part of a version: they are where a version runs, and the record says
which."* The record says so after the call, one call at a time. Nothing
says it before, for the whole program.

That gap has now shown up three times:

1. **Bigspin's Quiver (2026-11-04, Chattering 01a1a681).** The same
   pipeline on Claude Haiku 5.5 and on GPT-6 Luna had one version
   (`insight_finder` `sha256:b2fdf426…` on both) and behaved nothing alike
   (11 actions and 6 problems found, against 27 actions and 2). Quiver's
   release rule ("a position moves to a new board only if it is not worse
   on held-out examples") needs a score to belong to exactly one thing that
   ran. Quiver started building its own id from a bundle hash, its
   `program.json` and `functai.json`.
2. **Studio (design/16).** *"A model change is not a new version (the
   contract's rule), so runs made with the old model stay current; the
   dashboard notices and offers to run them again."* Studio patched the
   gap on its own side.
3. **Tool code.** In Python, changing what a tool returns (`"shipped"` to
   `"LOST"`) leaves the agent's version where it was (`sha256:947e6a5d…`,
   checked 2026-11-04). The version counts a module's code and a
   function's own code, never a tool's.

Two products building their own answer, differently, is the sign that it
belongs here.

## The rule, borrowed from ProgramIR

ProgramIR (`~/Projects/programir-contract`, `spec/placement.md`, ratified
2026-08-12) settled this for DSPy: **identity is baked, location is bound;
identity changes detach scores, placement changes do not** (PIR-010).
Applied to FunctAI:

- **What** a program is told: the **version** (unchanged, portable).
- **What** runs it: the **setup** (new). A change here means old scores
  may not hold.
- **Where** and **behind which wall** it runs: recorded on each call,
  never part of a name. A change here must not change what the program
  does; when it does, the thing that changed is in the wrong column (see
  *The sandbox*, which is the case that proved it).

So the three nest: **interface ⊂ version ⊂ setup**.

## What the setup counts

The setup is `"sha256:"` and the SHA-256 of the canonical JSON of one
document, like the version. It is computed where the program runs, from
the settings in force there (a program's own, `configure`, the process),
before its first call and without the network.

For an **AI function**: its version, and

- **the model**, as it will be asked: the `lm` setting resolved to
  provider and model name (`anthropic:claude-haiku-5-5`). A baked model
  counts by its weights' hash, not its name. A backup model
  (escalation) counts the same way.
- **the settings that change what the model sees, what it is asked, or
  what comes back**: lm15's sampling configuration (temperature, top_p,
  max_tokens, seed, stop, reasoning), `capabilities`, `max_steps`,
  `retries` (asking again after a malformed reply), `tool_errors`,
  `on_unreadable`. Written in lm15's canonical JSON, under each setting's
  contract name, so `maxTokens` and `max_tokens` are one setting. A setting
  left unset is absent, not the provider's default (FunctAI cannot know
  it; the record says what the provider did).
- **the plugins that shape the run**: each plugin's name, version and
  configuration (`keep_pictures(3)`, `limits(...)`, a `step` hook). A
  limit that makes the agent answer early changes what it answers.
- **its tools** (next section).
- **what its tools may reach** (*The sandbox*).

For a **module**: its version, and the setup of every program it uses, by
key. The pipeline on Luna and the pipeline on Haiku are two setups.

**Not counted**, recorded on each call instead: the server address
(`base_url`), keys and accounts, network retries and waits (`api_retries`,
`api_max_wait`), timeouts of the connection, the reply cache, logging
(`log_content`, the store), concurrency, and how the wall is built.

## Tools

What a tool is counts in two places:

- **In the version** (as today): its name, description and parameters.
  That is what the model is told, the same in every language.
- **In the setup** (new): its code, as `{"language", "code": "sha256:…"}`,
  read as each language reads a program's own code. Python follows what
  the tool reaches (the graph `check` and `save` follow); TypeScript reads
  the function's text only, so a helper it imports is not seen (stated in
  each language's docs; a host that bundles, as Quiver does, fingerprints
  the bundle).

Why the code goes in the setup and not the version: **portability**. A
Python tool and a TypeScript tool that do the same thing never have the
same hash. In the version, the same agent would have a different version in
each language, and the loader planned in saved.md (*"a later format may let
a loader bind tools by name"*) could never pass its check that a loaded
program has the version it was saved with. In the setup, the agent loaded
elsewhere keeps its version and gets its own setup: which is true.

The trade-off: rewriting what a tool does makes a new setup, not a new
version. A program's own code and a module's code stay in the version, as
now; they never cross languages anyway.

Tools that are not code of this program count by what names them:

| tool | counts as |
|---|---|
| a function in this program | its code (above) |
| another FunctAI program used as a tool | its setup |
| a tool run by the model's provider (lm15's `BuiltinTool`: web search, code execution) | its name and `builtin_config`, like a model |
| an MCP server's tool, or a program served elsewhere | the server's name and the tool's declared schema; what runs behind it can change without notice, as a hosted model can, so each call records what answered |

FunctAI supports neither provider-run tools nor MCP tools yet; the rule is
written now so they fit when they come.

## The sandbox

ProgramIR's placement rule says a stronger wall must not change what a
program does. That is true of **how the wall is built** and false of
**what the wall lets through**. A tool that is refused returns an error
where it returned data; the model reads it and does otherwise; an
instruction improved on runs that could install packages, or fetch pages
directly, can be wrong where they cannot. ProgramIR's own `trust.md` agrees
in practice: a hardened program's *"shipped scores detach"*. So the
sandbox splits in two:

| | examples | changes what the program does? | |
|---|---|---|---|
| **what it may reach** | domains, secrets (by name), folders it may write, programs it may run, memory, CPU and time limits | yes | **in the setup** |
| **how the wall is built**, for the same reach | bubblewrap, a container, a virtual machine, another machine | must not; if it does, that is a bug in the wall or the tool | recorded on each call |

A tool declares what it needs, as data, beside its effects:

```json
{"effects": "reads", "needs": {"domains": ["api.browser-use.com", "*.browser-use.com"],
                               "secrets": ["BROWSER_USE_API_KEY"]}}
```

The setup counts the program's **reach**: the union of its tools' needs,
and its limits, **when it runs walled** (everything not declared is
refused, as srt does). Then the declaration is the whole world the
program sees. A run with **no wall** (a laptop, a notebook) has reach
`"open"`: anything not declared may also work, so it is another setup.
That is the honest answer to a real failure: a prompt improved on a laptop,
where the agent could install what it liked, failing in Quiver's sandbox.

**Who does what.** FunctAI declares, records and checks; the host builds
the wall. FunctAI refuses to run a program whose host says it cannot give
what the program declares, records on each call which wall it ran behind,
and never holds a secret: it names them, and the host (or a proxy that adds
them on the way out, as Chattering's key proxy and srt's proxy do) supplies
them. This settles design/06's open contradiction ("who owns sandboxing")
the way design/06 suggested and ProgramIR ratified. Trust (whose code
runs, how hard to wall it) stays the host's policy, never something a
saved file claims (ProgramIR `trust.md`, *postures*).

## Records and saved files

- **Every call record** gets `program.setup`, and the call's surroundings:
  `placement` (`{"wall": "srt" | "none" | …, "endpoint"?}`), beside the
  `model` and `exchanges` it already has. Ratings keep pooling by
  interface; comparisons, scores and Quiver's boards group by setup.
- **`functai.json`**: TypeScript saves agents (each tool as `{"tool": {name,
  description, parameters}, "code": …, "effects", "needs"}`) instead of
  refusing; module nodes write their `version`; each program node writes
  the `setup` it had when saved, so two folders can be compared line by
  line ("`write_insights`: model `claude-haiku-5-5` → `gpt-6-luna`"). A
  loader computes its own setup where it runs, and says when it differs
  from the saved one; it never refuses for that alone (a different model
  is a choice, not a fault).
- **`functai setup-diff A B`** (and the same function in each language):
  what differs between two setups, by program and field. Studio and Quiver
  show it.

## Other things about tools, decided now or later

From ProgramIR, MCP's tool annotations, and the research on prompt
injection.

**Now, in this change:**

- **Needs** (above): domains and secrets. Folders, programs and limits use
  the same object when a host needs them.
- **A tool may report what it spent.** Browser Use's charges were $0.05 of
  a $0.06 run at Bigspin, invisible to `limits(cost=…)` and to every cost
  total, which count model tokens only. A tool returns `spent: {"usd":
  …}` beside its result (or calls `spend(usd)`), recorded on its step and
  summed with the model's cost.

**Next, each its own small design:**

- **`open` effects: text from outside.** FunctAI's effects are `reads` and
  `changes`. MCP adds `idempotentHint` and `openWorldHint`. The one that
  matters here is the second: a tool that brings in text someone else
  wrote (a web page, an email) is where prompt injection enters. With it,
  FunctAI can warn when, in one turn, text from an `open` tool can reach a
  model that may call a `changes` tool. That is ProgramIR's *integrity
  dual* and the core idea of CaMeL (Debenedetti et al., 2026,
  doi:10.1109/satml68715.2026.00040): keep untrusted data from steering
  what runs. FunctAI's typed pipelines already allow the safe shape from
  *Design Patterns for Securing LLM Agents against Prompt Injections*
  (Beurer-Kellner et al., 2025, doi:10.48550/arxiv.2506.08837): code
  decides the steps, the model that reads the web has no tool that
  changes anything, and its answer is typed data. Bigspin's
  `insight_finder` has that shape; its one-agent `insight_walker` does
  not. Asking a model to *detect* injections is not a defence to build on
  (doi:10.48550/arxiv.2507.05630).
- **`idempotent`**: safe to run again. tools.md's *Keeping what may have
  happened* asks a person when a `changes` tool may have run; an
  idempotent one could simply run again.
- **Session tools** (a browser, a terminal): state held across calls,
  opened and closed with the call that uses them, their steps recorded so
  replay never needs them. ProgramIR's `call | session` split. Needed when
  a walker drives its own browser instead of Browser Use's.
- **MCP tools and provider-run tools**, with the identities above.
- **Improving tool descriptions.** They are prompt (in the version);
  `gepa` and `instruction_search` could propose better ones from failed
  tool calls.

## Portability, stated

- The **version** stays portable, as now.
- The **setup** is portable for what is data: the same function, model,
  settings and declared needs give one setup in every language. It is
  not portable for code: a tool or plugin written in another language is
  another setup. That is true: it is another thing running.
- Ratings pool by interface, as now: nothing here splits them.

## Trade-offs, stated

- **Leaving the server address out** cannot tell apart two servers that
  serve different models under one name. The record of which model
  answered is the guard; a host flags when it changes for one setup.
- **Limits and plugins count.** Two runs that differ only in a limit get
  two setups, and their scores do not pool, even when the limit never
  mattered. Pooling wrongly is worse than splitting.
- **TypeScript sees only a tool's own text**, not what it imports. A host
  that needs more fingerprints its bundle.
- **Every program gets a setup once**: nothing that exists changes its
  version.
- **Four languages.** By the rule of the repository (a contract change
  lands in every implementation in one commit), R and Julia come with it,
  not after.

## Open questions for Maxime

1. **The name**: `setup` (`fn.setup`, `program.setup`)? Alternatives:
   `runtime`, `deployment`, `configuration` (ProgramIR's word).
2. **Plugins in the setup**: all plugins, or only those that change what
   the model sees or when it stops? (Recommended: all with a version; a
   plugin that only logs declares itself neutral.)
3. **Tool spending now** (`spent`), or with the `open` effects later?
4. **Order**: this note's contract and cases first, then Python and
   TypeScript, then R and Julia, all on one branch merged once.

## What was built

Maxime decided (2026-11-04): the name `setup`; tool code in the setup,
not the version (portability); the sandbox split in two (what a program
may reach counts, how the wall is built does not, and a run with no wall
is `"open"`); all of it in Python and TypeScript now, R and Julia later.

**As proposed**: the setup document of an AI function and of a module,
`program.setup` and `placement` on every record, `needs` on tools and
modules, the `wall` setting, `ReachError` (`needs-unmet`) before a
program's code runs, `spend(usd)` and `spent`, plugins' `config`,
`setup_diff` / `setupDiff` and `functai setup-diff`, saved folders with
each program's setup and each module's version, TypeScript agents saved
with their tools as data.

**Decided while building, each against the reason that would reject it:**

- **Every plugin counts; no "neutral" flag.** What only watches is an
  observer (the `observers` setting), never a plugin: every plugin hook can
  change the call. A flag a plugin sets on itself would let a plugin that
  changes behaviour leave the setup.
- **`spend(usd)` has no label, and makes no event.** Nothing would keep a
  label; a live reader that wants a running cost declares a note
  (`cost so far`). Trade-off: a page cannot show a tool's spending live
  without one.
- **`codeId` (Python `code_id`).** Found by the tests: two tools made by one
  factory have one text and two behaviours, and TypeScript reads only a
  function's text. A tool's author (or a host that bundles, as Quiver does)
  says what identifies it beyond that.
- **A model lm15 cannot route** is `{"provider": null, "model": name}`:
  named, not routed; the call fails and says why.
- **A refused reach is a recorded call**, as an input the interface
  refuses is: the record says `ReachError`, and no request was sent.
- **A price in `limits`' config** is written without lm15's defaults
  (`currency` USD, no tiers), so one price is one JSON in both languages
  (TypeScript's lm15 writes `3.0` as a `RawNumber`; canonical JSON writes
  `3` either way).
- **Python could not save a tool made with `functai.tool`** (it was "not a
  function"). It is saved now as `{"node": key, effects, needs, notes,
  code_id}` and loaded back as the same tool, with the version and setup it
  had.
- **The first call of a module read every installed package** (about 13
  seconds here; `packages_distributions()` twice per first call). The
  studio branch had cached it; the same change is on this branch, so both
  merge as one.

**Checked:** the contract's 333 cases; Python 1,000+ tests and TypeScript
590+ (each with tests of every part above through calls with a fake
model); `tools/crosslang.py` (Python, TypeScript, R and Julia reading
each other's records and folders: 58 checks).

**Not built, and why:** MCP tools and tools a provider runs (the setup
says how they would count; FunctAI runs neither yet); a wall's limits
checked by FunctAI (the host holds them: FunctAI records them); the model
a server address really serves (the record of which model answered is the
guard). R and Julia: they refuse an interface with `notes` (a key they do
not know) and skip `program.setup` (a record key they do not know), as the
contract asks of a reader, until they catch up.
