# 10 — Stages 1.2 to 5, built in Python first

*Status, 2026-09-30: built and tested on the branch `stages2-5-python`
(from `stage1.1`), not reviewed, not merged; Python only, by Maxime's decision: "all stages two to five,
starting in Python; the other languages after, not in this session". The
contract text and the cases change with it, so TypeScript, R and Julia
have something exact to build to; their harnesses read only the folders
they know, so the new cases wait for them (contract/README.md, "Which
cases each language passes"). Builds on `07` (the vignettes and
Maxime's answers there, which bind), `08` (what later stages must
provide) and `09` (the leanings for later stages).*

This changes the way of working of stages 1 and 1.1 in one respect: the
rule "a contract change and every implementation in one commit" is
suspended for these stages. The contract still changes first and the
cases are still written by scripts from the rules; only Python passes
them today. When a language takes a stage, it passes that stage's cases.

## What each stage delivers

| stage | what | where (Python) | contract |
|---|---|---|---|
| 1.2 | replies kept on disk, one flight per request, `replicate`; `map` with a progress line, resume by running again; `quotes_found`; `prune_calls` (ratings outlive a cleanup, F3) | `replies.py`, `engine.py`, `evaluation.py`, `judges.py`, `calllog.py` | `replies.md` |
| 2 | conversations: turns, branches, what the model sees, stores (memory, folder, your own), a turn's place known before the call, `request_id`, queueing, leases, stopping from another process, watching a turn's events from a store | `conversations.py`, `stores.py`, `core.py`, `calllog.py` | `conversations.md` |
| 3 | calls inside calls: `remembers`, `earlier()`, nested conversations refused unless declared; views (`full`, `kept`, `outside`), a module's answer as it is written (`answer_from`); serving (`functai serve`, `Service`, ASGI), `remote` | `conversations.py`, `views.py`, `serving.py`, `remote.py` | `conversations.md`, `streaming.md` (*Views*), `serving.md` |
| 4 | tools that say what they do (`effects`), `approve` (a function or a rule), waiting turns, a tool invocation id, the journal's barrier only before tools that change things, resuming a turn after a restart without paying for a model answer twice | `tools.py`, `engine.py`, `conversations.py` | `tools.md`, `streaming.md` |
| 5 | a record keeps its turn's steps; `rated` gives `earlier`, `conversation` (and `helpers` for modules); evaluating and optimizing on such rows ask each row again with its own earlier turns; `split` by conversation | `calllog.py`, `evaluation.py`, `optimizers.py` | `calls.md` |

Removed (07, decision 8; no users): `stateful=`, `state_window=`,
`fn.history`, `fn.reset()`, `module.history`. A conversation replaces
them.

## Decisions (the leanings of `07` and `09`, taken; the others new)

Each line: the decision, and why an expert would not choose otherwise.
"Leaning" marks a question `09` left open with a leaning, taken as it was.

### Stage 1.2

1. **The reply cache is a store behind one interface.** `cache_replies`
   is `False`, `True` (memory, as before), `"disk"` (one SQLite file in
   the user's cache folder), a path (a folder, or a `.sqlite` file), or an
   object with `get(key)` and `put(key, reply)`. SQLite, not a file per
   reply: many writers, atomic writes, thousands of entries without a
   folder of thousands of files. *Why not the conversation store:* a
   reply cache is keyed by content and may be shared by every program on
   the machine; a conversation store is keyed by conversation. They keep
   different things under different retention.
2. **The key** is `sha256` of the canonical JSON of
   `{"functai_reply": 1, "request": <lm15's canonical request>,
   "replicate": n}`: lm15's canonical request is the same in every
   language, so a cache written by one is read by another.
3. **Only replies that were read are kept.** The reply goes in once lmcc
   read it (and its values fit their types); an unreadable one never
   does. So an interrupted run leaves nothing half-written and nothing
   "started".
4. **One flight per key**, in a process (a lock per key) and across
   processes (a claim row with a lease in the same SQLite file; a claim
   whose lease ran out is taken over). The second caller waits and gets
   the first one's reply.
5. **What may be written.** A call any `log_content` layer drops a field
   of is not written to disk (the memory cache still serves it): a disk
   cache holds whole requests and replies, which is what `log_content`
   says must not be kept.
6. **`replicate`** (a setting: `fn.using(replicate=3)`) is the n-th
   independent answer to the same request; it is part of the key and
   nothing else. *Not* `sample`: `rate(sample=...)` already means a
   random draw of calls.
7. **`map`** takes `threads=` (the name `vectorize` and `unpack` use;
   `num_threads=` is still read, as `evaluate` and the optimizers keep
   it: renaming 677 uses across the docs is its own change) and
   `progress=` (a line on stderr, updated in place: rows done, errors,
   tokens, time left; on by default when stderr is a terminal or a
   notebook). Resuming is running it again with a disk cache.
8. **`quotes_found(text, quotes)`**: whether each quote is in the text,
   with white space, case, curly quotes and dashes made the same; a
   small deterministic check, and a recipe for a 0–10 judge written as
   an ordinary AI function. No rubric framework.
9. **Ratings outlive a cleanup (F3).** `prune_calls(older_than=...)`
   deletes whole day folders, after copying every rated call, its
   ratings and every call its `saw` needs into the folder's top level
   (`kept-<host>-<pid>-<hex>.jsonl`), which every reader already reads
   (calls.md, *The folder*). The layout does not change.

### Stage 2: conversations and stores

10. **A conversation is records in a store, append-only**
    (`conversations.md`): `turn` (created before the call: its id is the
    turn's call id, minted then; its parent turn; its inputs;
    `request_id`), `ended` (outputs, the lmcc turn with its steps, `saw`,
    error), `lease`, `stop`, `head`, `call` (a helper's call, for
    memory), `reply` and `tool` (what resuming needs, stage 4),
    `approval`. Nothing is rewritten; a branch is a turn whose parent
    has another child. *Why records, not a mutable tree:* a tree
    rewritten in place cannot be followed from another process or kept
    by a store that only appends; lmcc F7 (references, not copies).
11. **A store has two required methods** (`append(conversation, records,
    expect=)`, conditional on how many records it holds, and
    `read(conversation, after=)`), and may have `events` (an event log
    store, *The rules a store keeps*, for watching a turn from another
    process) and `durability`. `append` with `expect` is the one
    compare-and-set every rule needs (a turn's parent, a lease, a
    claim). *Not* vignette 3's two unconditional methods: `08` showed a
    conversation cannot queue or fence without a conditional append.
12. **Stores given:** `store=None` keeps the conversation in this
    process's memory (one memory per process, so the same id opens the
    same conversation); `store="folder/"` or `FolderStore(path)` keeps it
    in files (one JSONL per conversation, an event log per turn, locked
    with `flock`, `fsync`ed: durability `"disk"`); `store=True`, the
    default folder `$XDG_DATA_HOME/functai/conversations`; any object
    with the two methods.
13. **What the model sees: every earlier turn by default** (Maxime's
    answer 8). `context=functai.last_turns(10)` keeps the last ten;
    `without=["photo"]` leaves bulky inputs out of past turns (the `saw`
    entry's `without`). A turn that no longer fits the model fails with
    the provider's error and the fix in its message.
14. **A turn's `saw`** is `[{"saw_of": parent}, {"call": parent, …}]`
    when it saw exactly what its parent saw plus its parent, else the
    entries in full: the record does not grow with the conversation.
15. **Two sends at once to one conversation queue** (07's proposal;
    Maxime's answer 10 makes it a setting: `sends="queue"` (default),
    `"refuse"` or `"branch"`). A conversation opened by id follows its
    head: its next turn's parent is the head at the moment the turn is
    created, once that head has ended. A view made by
    `continue_from(turn)` follows its own branch. So three views on one
    turn answer in parallel (vignette 9), and two page requests on one
    conversation queue (vignette 2).
16. **The same `request_id` twice is one turn**: the second send gets
    the first turn (waits for it, or watches its events from the store).
17. **Leases, not heartbeats alone** (leaning B6: the first claimant
    holds a running turn until its lease expires). A running turn's
    process renews a `lease` record every 10 s for 30 s. A turn whose
    lease ran out without `ended` is `interrupted`. Only then may another
    process claim it (a `lease` record appended conditionally).
18. **Stopping from another process** is a `stop` record; the running
    process sees it within a second and the turn ends `stopped` (the
    call's error is `Cancelled`, as closing a stream).
19. **A host that never keeps transcripts refuses a stored
    conversation** (leaning B2): a store that keeps records (any but the
    process's memory) refuses `conversation-content` when a
    `log_content` layer drops a field of the program. Refuse, never
    forget: a conversation that silently forgets what it was told is
    worse than one that says it cannot start.
20. **Changing what the program writes mid-conversation** (leaning C7,
    done in FunctAI, not lmcc): a turn made with another signature is
    shown with the fields the program still has; a program that now
    writes an output the earlier turns lack (reasoning turned on, a
    first tool) refuses `conversation-signature` unless
    `earlier_without=["reasoning"]` names it; a changed type or a
    removed input refuses. Models change freely, per conversation or per
    turn (`chat(..., lm=...)`), and each turn records who answered.
21. **A merge** (leaning C8) is a turn whose parent is the question,
    made by another AI function from the branches (`reads`), and
    recorded also as the conversation's program's turn (`made_by`), so
    the next turn sees it. Its rating goes to the merge function.

### Stage 3: calls inside calls, views, serving

22. **Helpers remember nothing unless told** (07 decision 5):
    `remembers={answer: "conversation"}` (its own earlier calls on this
    branch, in earlier messages and this one), `"turn"` (this message
    only), `functai.remember("conversation", steps=True)` (with its tool
    steps). A helper's memory is found through the store's `call`
    records under the turns from the root to the head, never copied
    into turns.
23. **`functai.earlier()`** inside a module is the conversation so far
    as data (a list of `{input…, output…}` rows), for a helper that
    declares an input for it (vignette 6). `[]` outside a conversation.
24. **A remembering program called inside another is refused**
    (leaning C11, `conversation-nested`) unless the outer conversation
    declares it: `remembers={inner_chat: "own"}`.
25. **Views** (`streaming.md`): `full` (the process's whole log),
    `kept` (the log_content form), `outside` (a caller who sees only the
    program's boundary: its `started` with a program object that names
    no file, its answer's text, approvals addressed to the caller, its
    `done`, or `failed` with the error's type and code and no message).
    `s.events(view="outside")` gives the JSON a server sends.
26. **A module's answer as it is written:** `@module(answer_from=fn)`
    says the answer text of `fn`'s last call is the module's answer; the
    outside view shows it as the module's text (each call of `fn` a new
    `request`, so a second call empties the first's text).
27. **Serving** is one transport-free `Service` (routes, keys, views),
    with two front ends: the standard library's threaded HTTP server
    (`functai serve`, no dependency) and ASGI (`service.asgi`, to mount
    in FastAPI or run under uvicorn). Keys are bearer tokens from a file;
    without keys it listens on 127.0.0.1 only. Programs with an opaque
    input or output are refused (`serve-opaque`): no JSON reaches them.
    The interface is served with a format number (leaning B10b:
    `{"functai_interface": 1, …}`).
28. **`remote(url, key=)`** is a program again: it binds and checks its
    inputs here, is logged here (kind `remote`), maps over tables,
    evaluates, streams; the server's call names the caller's call as its
    `parent` (the `FunctAI-Parent` header), so the two records are one
    call tree across two logs.

### Stage 4: tools that ask first

29. **`@functai.tool(effects="reads" | "changes")`**; a tool that says
    nothing has unknown effects and counts as `"changes"` for rules (07
    decision 6). An AI function used as a tool reads, unless its own
    tools change things.
30. **`approve`** is a setting (so it works in `configure`, `using`,
    `@ai`, a conversation, a turn): a function (asked at once, returns
    `True`, `False` or a reason to refuse) or a rule (`"changes"`,
    `"all"`, a list of tool names or approval paths such as
    `"support/answer/refund"`). No approval by default.
31. **A refusal is an answer the model sees**: the tool result is
    `The person did not allow this call.` and the reason.
32. **A rule with no function to ask waits.** In a conversation with a
    store the turn ends `waiting` (the call raises `Waiting`, holding the
    approvals); `turn.approve(...)` or `turn.deny(...)`, from any
    process, records the answer and resumes the turn. On a stream
    without a conversation, the stream waits for `s.approve()` or
    `s.deny()`. A plain call cannot be answered, and refuses
    `approval-required` before the tool runs.
33. **Resuming replays what was recorded.** A turn in a store records,
    as it runs, each model reply (`reply`, by request hash) and each
    tool that changes things before and after it runs (`tool`). Resuming
    runs the turn's program again with the same inputs: a request made
    before gets the recorded reply (no answer paid twice), a tool that
    ran gets its recorded result, and the call goes on from where it
    stopped. This works at any depth (vignette 11's pause three levels
    down) because a module's code is run again, not restored. *Why not
    restore lmcc's turn alone:* it covers only an AI function at the
    root; replay covers modules and nested helpers with one rule. *Cost:*
    a module whose code is not deterministic given its inputs and
    replies makes new requests on resume (paid for, never wrong).
34. **After a crash**, a tool that started and has no result is
    `unfinished` ("may have run"). `turn.resume(results={…})` gives what
    it returned, `rerun=[…]` runs it again, `turn.abandon()` ends the
    turn. It is never run again on its own.
35. **The journal's barrier comes only before tools that change
    things** (leaning B3): a tool that says `effects="reads"` runs
    without waiting for the journal.
36. **A tool invocation id** is the number of the tool call within its
    call (1, 2, …, across every step), on `tool_call` and `tool_result`
    events (`invocation`) and on the `started` event and record of every
    call a tool makes: lmcc's ids repeat across replies.
37. **A call continued by a later writer** (resumed) keeps its id; its
    record says `writer`; a reader takes, for one id, the record of the
    highest writer (its exchanges are that writer's).

### Stage 5: learning from conversations

38. **A record keeps its turn's steps** when the call ran tools and
    content is whole (`steps`, lmcc's turn steps), each model step's
    `request` equal to an exchange's `request_hash`: showing a call
    again with its steps is reading them, checked, never rebuilding
    them from replies (which would need every language to agree on how
    retries, the cache and escalation become steps).
39. **`rated`** gives `earlier` (the turns the call was shown, each
    `{"inputs", "outputs"}`, and `steps` when shown with them) and
    `conversation` (the conversation's id, or null); a module's row
    also `helpers` (each helper call's earlier turns, in the order they
    were made). A rated call whose earlier turns the log cannot show
    again is left out and counted (`no_context`), never shown in part.
40. **Evaluating and optimizing ask each row again with its own earlier
    turns**, nothing recorded in any conversation, the call's `saw`
    `[{"saw_of": <the rated call>}]`. Optimizers take only rows with no
    earlier turns as worked examples (leaning C4) and say how many they
    skipped.
41. **`functai.split(rows, by="conversation")`** keeps each
    conversation on one side.

## How it was checked

- Python's offline suite: 738 tests pass (104 new: `test_conversations`,
  `test_tools`, `test_serving`, `test_learning`, `test_long_runs`,
  `test_contract_stages`), the old `stateful` tests rewritten as
  conversations. A real HTTP server and `remote` run in the tests; so
  does a turn that waits, is answered from "another process" (a fresh
  store object on the same folder) and resumes with one new model call.
- The contract: 238 cases (34 new, in `replies/`, `conversations/`,
  `tools/`, `views/`, `context/`), written by `cases/stages.py` from the
  rules; every record, event and call record the new code writes passes
  the schemas (`conversation.schema.json` is new; `call` and `event`
  gained optional keys).
- Not run: the other languages' checks, the live runs (real models) and
  the documentation notebooks (`docs/articles/memory.md` was rewritten
  for conversations; its outputs are written when it is run).

## Trade-offs taken

- **Every conversation turn runs in a thread of its own** (a stream),
  even `chat(x)`: that is what lets a turn be stopped from another
  process and watched while it runs. A breakpoint in a tool stops in
  that thread. A turn called plainly is not streamed from the provider
  (its replies arrive whole); only a live reader asks for streaming.
- **A resumable turn keeps every model reply in its store** (a module's,
  or an AI function's with tools): resuming pays for nothing twice, at
  the cost of store size.
- **Resuming re-runs a module's code**: code that is not deterministic
  given its inputs and replies makes new (paid) requests on resume.
- **The default memory store lives as long as the process**: a notebook
  that opens thousands of conversations holds them all.
- **A record keeps its turn's steps** when made in a conversation or with
  tools: a larger log, in exchange for showing a call again without
  rebuilding it.
- **`/interface` needs the key** when a service has keys (a private
  program's interface is not public); `GET /` (the form) does not.
- A remote program's `stream()` is not logged on the caller's side
  (its calls are).
- A conversation used inside another's turn joins the outer turn's call
  tree: it has no event log of its own in its store.

## Not done here

- Stage 1.1's pull request (this branch is built on it) and the reviews
  by two outside reviewers that each stage had before merging.
- TypeScript, R and Julia (each takes the stages' cases later).
- Images and audio (stage 7 of `07`'s plan): tool results stay text.
- `functai push` and Hugging Face; the answer key saved with a program.
- A token-based context rule (`fits(model)`): turns only.
- Summaries of old turns (C5): an input the program declares.
- Notebook outputs in `docs/`: the pages that show conversations are
  written, and run when Maxime asks (they call real models).
