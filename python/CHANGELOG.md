# Changelog

## Unreleased

### lm15 2: one meaning per token count

FunctAI now requires lm15 2 (`lm15>=2.0,<3`) and lmcc 0.8.8, the lmcc made for it. lm15 2 gives each token
counter one meaning on every provider: `input_tokens` includes cached
tokens, `output_tokens` includes reasoning (contract/calls.md, *Token
counts*). In lm15 1, Anthropic's input left the cache out and Gemini's and
xAI's output left the thinking out, so the same column meant different
things depending on the model.

- **A sum of token counts is exact or absent.** `pred.usage`, a call
  record's `usage`, a turn's, an evaluation row's and bake's teacher
  totals include a counter only when every model call reported it. Before,
  a counter one call did not report was skipped, so the sum was a part of
  the total that looked like all of it.
- **Counts lm15 1 wrote are shown only when they cannot mislead.**
  `functai.calls()` reads a record's counts with the lm15 that wrote it
  (`process.lm15`): from lm15 1 they are shown when `total_tokens` is
  `input_tokens + output_tokens` and no cache count is above 0, else null.
  Records are never rewritten.
- **The reply cache writes rows of format 2** (contract/replies.md). Rows
  lm15 1 wrote are still used, with the same rule for their counts, so no
  reply is asked for again. Keys are unchanged. An older FunctAI skips the
  new rows and asks the model again.
- **Bake's teacher cost is unknown when tokens are.** A row whose tokens
  were not all reported made the dollar figure an undercount; it is now
  left out, with a note, and the report says "tokens not reported". The
  cost is still plain input and output list prices (a cached prompt's
  discount is not taken off).

Pictures and files, checked against TypeScript, R and Julia (contract:
`functions.md`, *Pictures, sounds, documents and files*; cases
`functions/19`–`23` and `files/`).

- **A picture function has a version of its own.** A media input
  (`ImagePart`, `AudioPart`, …) had no sample value: its sample render was
  refused, so every function with a picture input had the one version
  `{"request": "refused:value-invalid"}`, whatever its instruction, layout or
  worked examples, and their ratings pooled. Its sample is now a picture by
  web address (contract/calls.md, *Versions*). Every picture function gets a
  new version once; cached replies are unaffected (they are keyed by the
  request).
- **Lists of pictures are sent as pictures** (lmcc 0.8.7, D-64, now
  required): `list[ImagePart]` sends each picture as an image part, and
  `Optional[ImagePart]` left out is the text `null`. The warning that a list
  of pictures was written as base64 text is gone. A dataclass holding a
  picture is refused before anything is sent (`no-format`) instead of
  being sent as base64 text.
- **A function reading files, loaded from a folder another language wrote,
  reads them.** Loaded from data, an `Artifacts` input was not bound to
  Attachments (lmcc binds by the Python type, which a saved signature names
  only as text), so the sample sent `[]` as JSON and the load refused
  `saved-differs`; calls would have sent the artifacts as JSON text.
- **Files are a whole input.** `Optional[Artifacts]`, `list[Artifacts]` and
  a record holding `Artifacts` sent the artifacts as JSON text, pictures as
  base64; they are refused when the function is defined
  (`interface-malformed`). For no files, give `att([])`: the model sees
  `(no content)`.
- **A failed file in a served call is refused.** Artifacts given in their
  wire form (a served call, a replay) were not checked for files that failed
  to read; they are refused before any request, as `att()`'s result is, and so
  is a value Attachments' `from_wire` refuses (`interface-input`).
- **`FilesUnavailable`** (code `files-unavailable`, an `ImportError`): a
  function reading files where Attachments is not installed.
- The `Artifacts` input's shape no longer carries a description (it named a
  file Attachments has since moved); its signature id changes once.

What the transcripts of real projects and the coding-agent evaluation
(`evals/skill/findings.md`) showed was wrong or slow.

- **A bake's longest example is one row's prompt and answer.** The plan added
  the longest prompt of the data to its longest answer, which usually belong
  to two different rows: long-text bakes were refused for memory they did not
  need (an example of 26,000 tokens was counted as 38,000).
- **Baking from an Examples file works.** `bake(examples)` stopped with
  `'Plan' object has no attribute 'to_label'` before planning anything.
  Examples made without `student=` (or for another student) are tokenized
  by the bake itself instead of refused; examples tokenized for a student
  train that student (the default list was tried instead); and the plan
  counts their validation rows (it said "0 validation").
- **Bake's estimates match real runs, and runs correct them.** From a
  StatLM run (Qwen3.5-4B, examples up to 87,119 tokens, an RTX PRO 6000):
  - The RTX PRO 6000 matched nothing in the speed table and was taken for
    a 30 TFLOPS card: the plan said 3.7 days, the run took 9.3 hours. The
    table knows it (500), the B300 and the RTX PRO 5000/4500/4000; a card
    it does not know is named in the plan as a rough guess.
  - Students with linear-attention layers (Qwen3.5) need about 1.7 times
    the memory a token the formula gave: the plan said 16-bit weights fit
    (58.9 / 95 GB); they ran out of memory. Their margin comes from that
    run: 16-bit does not fit, 4-bit does, as it happened.
  - 4-bit base weights are counted about a third slower.
  - Every run on a GPU measures the memory a token of its largest batch
    takes (from batches of 8k tokens or more) and the speed it trains at;
    later plans of that student on that machine use them, and say so
    (`~/.cache/functai/bakes/measured.json`). A speed is reused only for
    examples of a similar length.
- **A run says what it decided on its own.** Retrying with smaller batches
  or with 4-bit base weights after running out of memory was a line in
  `log.txt`; it is now printed while waiting (`NOTE: …`), kept in
  `run.notices`, recorded in the model (`training.notices`), and shown in
  the model's repr ("trained on 4bit base weights").
- **The plan lists what matters.** With `where=` given, places that cannot
  train the bake are left out of the plan (Tinker and Prime were listed
  with their missing keys under every plan); a missing `causal-conv1d` for
  a Qwen3.5 student is named with how to install it.
- **Long replies are not cut by the client.** A request to a baked
  student's server waited 10 minutes at most, less than a 50,000-token
  reply takes on a small GPU; it now waits at least as long as 10 tokens a
  second would need.
- **Serving on a full GPU says so.** `baked.on("vllm")` on a GPU without
  room for the weights started vLLM, which failed with a long trace; it now
  says how much is free and how much is needed before starting.
- **Words written in the code reach the model** (`functai/guidance.py`). A
  comment after each choice of a `Literal[...]` written over several lines,
  or after each member of an `Enum`, now describes that choice; a comment
  after the closing `]` of a multi-line annotation describes the field;
  `Annotated[T, "words"]` on a record's field, a record's own docstring,
  and every record nested in what a function reads or writes
  (`list[Analysis]`, `Optional[Obs]`, a record given as an input) are listed
  with their described fields; a comment after the first parameter on the
  `def` line is read. All of these were silently dropped: StatLM's schema
  had 1 of its 65 comments in the prompt (2026-10-04); of its 80 today, 73
  are (the other 7 are section banners and group labels, correctly left
  out). A wrapped comment's next line
  is no longer taken for the following field's. Prompts that were already
  complete are unchanged; a function whose prompt gains words gets a new
  version, so its cached replies are asked again once.
- **Tool pragmas are not words for the model.** `# noqa: …`, `# type: ignore`,
  `# pyright: …` and the like were read as guidance (a `# noqa` after a return
  annotation became "Return guidance: noqa: F722"); they are left out now.
- **The coding-agent skill is proportional and tested.** Everything common is
  on its first page with the exact names of what comes back; checks scale
  with the run's cost; every example runs in `tests/test_skill.py`. Agents
  using it took 8% more steps than DSPy agents (was 35%).
- **The skill's examples need only what `functai[data]` installs**: they read
  results with dpyr's `to_dicts()`, not `to_pandas()` (pandas is not in the
  extra; found by running them in a clean environment from PyPI).
- **`functai[data]` needs dpyr ≥ 1.12** (`unnest`, which the skill and the
  docs use for lists of records; 1.10 and 1.11 lack it).
- **A function defined with no source says so.** Comments come from the
  source; code run from standard input (`python - <<EOF`) or `exec()` (and
  `python -c` before Python 3.13) has none, so its comments never reached the
  model, nor were its `name: T = _ai` outputs asked for, without a word.
  FunctAI now warns once per function, naming both.
- **A shape written as JSON Schema is kept.** `picture: {"media": "image"}`
  was rewritten as text, so the picture went into the prompt as base64 and
  the model guessed (3 of 12 right in the evaluation). It is now the shape
  lmcc's own frontend reads, sent as an image part (12 of 12).
- **Requests to OpenAI ask not to be stored.** OpenAI's Responses API keeps
  every request and its reply for 30 days unless the request says `store:
  false`; FunctAI now sends that by default to the providers the contract's
  `models.json` lists under `no_storage` (OpenAI), unless the settings give
  `store` (contract/functions.md, "Requests the provider would keep").
  Measured 2026-10-06 on four models, kept requests were also slower (median
  0.88–1.36 s against 0.69–0.91 s, and a long tail: p90 2.6 s against 0.9 s).
  Every OpenAI request changes, so replies already in a disk cache are asked
  again once.
- **`api_max_wait`** (default 60 seconds). A provider asking to wait longer
  before a re-send (a subscription's spent window asked for 3 h 7 min) is
  no longer waited in silence: the error is raised at once, saying how long,
  until when (UTC), and the setting that would wait for it. Per-minute
  limits are still waited.
- **A reply the provider stopped is never an answer and is not asked
  again.** lmcc 0.8.6 refuses it with the code `parse-filtered` (a finish
  reason `content_filter`, a safety filter or the model declining, and a
  model's refusal in its own words); FunctAI raises it at once. It was asked
  again as "a reply that could not be read", which sent the empty reply back
  (a request Claude refuses). Requires lmcc ≥ 0.8.6.
- **A reply that ended in an error is sent again as it was** (lmcc's
  `parse-interrupted`): the same request, without the "could not be read"
  correction that would have sent the broken half reply back.
- **Files as inputs, through Attachments** (`functai/files.py`, design/13):
  `def summarize(report: attachments.Artifacts)`, called with
  `att("report.pdf")` (or a spreadsheet, slides, a web page, a folder, a
  picture, several files): the model gets Attachments' own parts: text,
  and pictures as image parts (a PDF's page pictures, with `[images: true]`,
  each after its page's text). The call log, the
  reply cache, conversations and served calls keep Attachments' wire form,
  and rebuild it on replay. A file `att()` could not read refuses the call
  before any request (`InterfaceError`, naming the file and why). Importing
  functai never imports attachments: the binding is made when attachments is
  imported, or when a program that reads files is planned. `pip install
  "functai[files]"` (attachments ≥ 1.0.0b2, a beta on PyPI).
- **Pictures as `lm15.types.ImagePart` inputs** (lmcc 0.8.6 binds lm15's
  media parts): `def shape(picture: ImagePart)`, called with
  `image(path="cat.png")`, `image(data=...)` or `image(url=...)`.
- **A picture given by its path is read when the call is made**
  (contract/programs.md, *Binding*): the request, the reply cache's key and
  the call log hold the picture, not its name. A file changed under the same
  name was answered from the cache with the old picture's answer. A path that
  cannot be read refuses the call, naming the file.
- **A list of pictures is warned about.** `list[ImagePart]` is written as
  base64 text the model cannot see (lmcc#7); functai warns once per function
  and points to one input per picture, or one `Artifacts` input.
- **`compare` says what differs** when two evaluations' rows are not the
  same (`column 'rules' only in the second`), and how to compare instead.
- **The coding-agent skill ships in the package** (`functai/skill/`):
  `functai skill` says where it is and which agents have it; `functai skill
  --install` copies it into Claude Code's, Pi's and Codex's skills folders.
- **A call that kept an unreadable reply (`on_unreadable="record"`) records no
  `returned`**: it has no answer, and no code changed one (contract/functions.md,
  "Keeping an unreadable reply"). Its record had `returned: null`.
- **Choosing a device for a baked head skips a GPU too full to ask**: with every
  GPU held by other jobs, asking one its free memory fails (CUDA out of memory),
  and running a baked head failed with it; it now runs on a GPU that answers, else
  the CPU.
- **Python loads what TypeScript, R and Julia save**, and `tools/crosslang.py`
  now checks it: the same version, the same request, and the settings saved with
  them. (It always could; nothing checked it, and the other languages' READMEs
  said it could not.)
- **`bake(rows, plan_only=True)` on a function with a fixed set of answers
  describes the head bake that `bake(rows)` does** (answers, splits, labels,
  where), instead of planning a generative student that refused an encoder.
  `method="sft"` still asks for a generative plan.

## 1.2.0 (2026-10-03)

Conversations, tools that ask first, plugins, serving, the reply cache on disk,
learning from rated conversation turns, and baking redesigned (Upgrading, in
the documentation, lists what changed from 1.1).

### Fixes since the stages were merged

- **Baking on the CPU no longer fails when the machine's GPUs are full.**
  A bake with `device="cpu"` left the GPUs visible to its training
  process, and torch's optimizer asked one for its stream at every step:
  with every GPU's memory taken by another job, the training ended
  `failed` (`CUDA error: out of memory`). The training process now sees no
  GPU when it trains on the CPU.

- **A served AI function refuses a wrong input with `422`** (`interface-input`,
  naming the field), as `contract/serving.md` says and as a served module
  already did: a missing, unknown or unbindable input is checked against the
  interface before anything runs, on `/call`, `/stream` and a conversation's
  turns. It was a `500` (`TypeError`).
- **A program saved while no model was configured verifies.** `save(...,
  record=[...])` keeps the model FunctAI picked by default, and `verify`
  replays with it; it failed with "no model configured", since a replay
  has no logins to pick one from.

- **Baking, rebuilt** (`design/12-bake.md`, `contract/baked.md`). Bake had no
  users, so nothing of the old generative API is kept.
  - **One line, decided from the data.** `fn.bake(rows)` picks a head model
    when every output is finite, else a generative student; the student
    (Qwen3.5 4B, 2B or 0.8B: the largest that trains where it is going
    within a day), LoRA's rank, the learning rate (Thinking Machines' fitted
    rule), the passes, the batch size by tokens, the lengths (no more 4,096
    caps) and the schedule (warmup, constant, decay: checkpoints before the
    decay are fair points of a quality-against-data curve).
  - **The plan before anything runs.** `fn.bake(rows, plan_only=True)` (or
    `functai.bake.plan`) prints the rows, tokens, student, where it trains,
    how long, what the teacher's answers and a service would cost, and which
    kernels are missing; nothing is spent.
  - **Where it trains.** `where="here"` trains with TRL's `SFTTrainer` on
    functai's own tokens: 16-bit weights (4-bit base weights, QLoRA, when
    those do not fit), batches built by tokens (packed without padding when
    a flash-attention kernel and an all-attention model allow it; grouped by
    length otherwise, as for Qwen3.5, whose linear-attention layers would
    carry state across packed examples), every free GPU through `torchrun`,
    the loss on answer tokens only. `where="tinker"` trains on Tinker with
    the same tokens and exact per-example weights, without PyTorch here.
    `where="prime"` dispatches Prime Intellect's hosted SFT. `where="export"`
    writes the examples with a TRL script and an Axolotl config. `"auto"`:
    here when a GPU here can train it, else the cheapest service set up;
    `configure(bake_where=...)` sets a preference.
  - **Runs you can leave.** A bake is a folder and a process of its own:
    `metrics.jsonl` as it goes, checkpoints, `run.stop()`/`resume()`,
    `functai.bake.runs()`, any checkpoint as a model; an out-of-memory first
    batch retries smaller; running the same bake again resumes it.
  - **What is trained is what is called.** Examples are the exact requests
    the layout writes, tokenized with the student's template; the call sends
    the same tokens (in-process, vLLM's completions endpoint, Tinker's
    sampler). `functai.bake.examples(fn, rows, student=...)` is that table
    for any trainer; `functai.bake.adopt(folder, fn, examples=...)` takes a
    model trained elsewhere after checking its template writes those tokens.
  - **Several functions, a whole program, fixed inputs.** `bake({f: rows,
    g: rows})`; `bake(program, inputs, teacher=...)` keeps every AI call
    inside as an example; `fixed={"input": value}` and
    `derived={"input": "other"}` leave inputs out of every example and
    call, and refuse calls the student never learned. `tags=` and
    `weights=` columns.
  - **Judged with your metric.** Open answers are scored with `metric=`
    (any `evaluate` metric or AI judge), else readability and samples; the
    teacher is no longer re-run on the test rows unless asked
    (`compare_teacher=True`, now off by default for heads too).
  - `baked.on("transformers" | "vllm" | "tinker" | url)`; `baked.download()`
    brings Tinker weights here. `baked.json` is format 2 (one entry per
    function); format 1 folders are refused with the advice to bake again.
  - Extras: `functai[bake]` adds TRL (pinned to 1.14), Accelerate, datasets
    and bitsandbytes; `functai[tinker]` and `functai[fast]` (flash-attention
    hub kernel, flash-linear-attention, Liger) are new.

- **A reply cut off at the token limit** (`contract/functions.md`): it is sent again
  with twice `max_tokens` only when one was set. Without one it already had the
  model's whole limit (lm15's default, or the provider's own), and the old re-send
  with 2048 (a guessed 1024, doubled) only shrank it: it now raises at once. The
  refusal's hint says how many tokens went to thinking, what the limit was and
  whether it can be raised, and what lm15 changed in the request (a dropped
  thinking budget, the usual reason `thinking_budget` does not bound the thinking).

**Plugins** (`design/11-plugins.md`, `contract/plugins.md`): hooks over
turns, context, calls, requests and tools, whose changes are data and
recorded, so rated calls are still asked again as they were.

- `functai.Plugin("name", version=...)` with seven hooks: `turn_start`
  (a turn's inputs), `context` (which earlier turns are shown, fields left
  out, sections), `before_call` (instruction, sections, model, settings,
  tools offered), `request` (the escape hatch: the call is then not
  replayable), `tool_call` (change the input, block, or `tool.ask()` a
  person), `tool_result`, `turn_end` (hears; may keep entries). A handler
  returns `functai.Change(...)` or None.
- `plugins=[...]` in `configure`, a block, `@ai`, `@module`, a
  conversation or one turn; the program's own run first, the host's last;
  `program_plugins=False` lets a host drop a program's own.
  `functai.load_plugin(path)` reads one from a file.
- Records keep each change (`changes`), the sections a call's conversation
  gave it (`sections`), and `replayable: false` after a request was
  replaced; a turn keeps what its context hooks showed it, and resuming
  shows exactly that. Rated rows carry `sections`; evaluating asks them
  again with their conversation, shaped by the evaluating process's
  plugins, never by recorded ones.
- Conversations keep plugins' **entries** per branch: `chat.remember(...)`,
  `chat.entries(...)`.
- **Approval is a plugin** (`approve=` drives it); any plugin may ask a
  person, and approvals name the plugin that asked (`plugin`, `question`).
- Built-ins: `functai.compaction(keep=20, every=10)` (a rolling summary per
  branch) and `functai.delegate(program)` (a tool that runs another program
  in its own conversation, per branch).
- `examples/plugins/`: six real Pi and Chattering extensions as plugins.

### Stages 1.2 to 5

Stages 1.2 to 5 (`design/10-stages-1.2-to-5-python.md`), in Python first;
the contract says each (`contract/replies.md`, `conversations.md`,
`tools.md`, `serving.md`, and additions to `streaming.md` and `calls.md`).

- **Conversations** (breaking: `stateful=`, `state_window=`, `fn.history`,
  `fn.reset()` and `module.history` are gone). `chat = fn.conversation(
  "alex", store="tutoring/")` is called like the function; each call is a
  turn that sees the earlier ones (every one by default, or
  `context=functai.last_turns(10)`, with `without=[...]` for bulky inputs).
  `chat.turns`, `turn.saw`, `chat.render(...)` (the next request, nothing
  sent), `chat.continue_from(turn)` (a branch: nothing is ever deleted),
  `chat.merge(branches, fn)`. A turn is saved before its call starts
  (`chat.stream(...).turn`), a `request_id` sent twice is one turn, two
  sends at once queue (`sends="refuse"` or `"branch"`), a turn is stopped
  from any process (`chat.stop(turn)`), and a turn whose process died is
  `interrupted` (a lease), and can be resumed. Stores: this process's
  memory (default), a folder (`functai.FolderStore`, locked across
  processes), or any object with `append` and `read`. A stored
  conversation refuses a program whose `log_content` drops a field
  (`conversation-content`), and one whose outputs changed
  (`conversation-signature`; `earlier_without=["reasoning"]` goes on after
  turning reasoning on).
- **Helpers inside a module's turn** remember nothing unless told:
  `support.conversation(id, remembers={answer: "conversation"})`
  (`"turn"`, `functai.remember("conversation", steps=True)`).
  `functai.earlier()` is the conversation so far, as data. A conversation
  used inside another's turn is refused unless declared (`"own"`).
- **Tools that ask first**: `@functai.tool(effects="reads" | "changes")`;
  `approve=` a function (asked at once) or a rule (`"changes"`, `"all"`,
  tool names or approval paths such as `"support/answer/refund"`). A
  refusal is shown to the model. In a conversation, a rule makes the turn
  wait (`functai.Waiting`), and `turn.approve()` / `turn.deny()` from any
  process resume it: every model reply and tool result it had is reused,
  so nothing is paid for or run twice; a tool that may have run when a
  process died is never run again on its own (`turn.resume(results=...)`
  or `rerun=[...]`). On a stream, `s.approve()` / `s.deny()`. Tool calls
  are numbered (`invocation`), and so are the calls a tool makes. A
  required journal waits only before tools that change things.
- **Views and serving**: `s.events(view="outside")` is what a caller who
  sees only the program's boundary may see; `@module(answer_from=fn)` shows
  a helper's answer as the module's while it is written.
  `functai.serve(program, keys=...)` and `functai serve folder/ --lm ...`
  serve a program (interface, OpenAPI, calls, streams, conversations,
  approvals); `functai.Service(program).asgi` mounts it in another app.
  `functai.remote(url, key=...)` is a served program used like a local
  one (logged here, `program.kind` `"remote"`; the served call names the
  caller's as its parent).
- **Long runs**: `cache_replies="disk"` (or a path) keeps replies in one
  SQLite file across runs and processes; only replies that were read are
  kept; one flight per request; `replicate=n` asks for another answer to
  the same request. `fn.map(rows, threads=8)` shows a progress line and,
  with the disk cache, resumes by being run again. `functai.quotes_found(
  text, quotes)` checks a judge's evidence. `functai.prune_calls("90d")`
  deletes old day folders and keeps what ratings need.
- **Learning from conversations**: `functai.rated(tutor)` gives a rated
  turn's `earlier` turns and its `conversation` (a module's row also its
  helpers' memory); `evaluate` and the optimizers ask each such row again
  with them (optimizers never show one as a worked example);
  `functai.split(rows, by="conversation")`. A record keeps its turn's
  `steps` when it ran tools or was made in a conversation.

### Stage 1.1

Stage 1.1 (`design/09-stage1.1-decisions.md`): decisions on stage 1's open
questions, and three fixes to how ratings become data.

- **Inputs are bound** (breaking): every call, of an AI function or a
  module, converts each input to its declared type when the meaning is
  clear, and refuses it otherwise (`InterfaceError`, `interface-input`,
  before anything is sent; an AI function's refused call is logged as a
  module's is). Text takes a number (`42` is `"42"`), `true`/`false`, a
  list or record (JSON indented by two spaces), or a value whose type has
  a text of its own (a data frame, a date); `<object at 0x…>` is refused.
  An integer takes `5.0`, `"5"`; a number takes `"2.5"`; a boolean only
  `True`/`False`. A record input keeps only the members it declares. A
  missing value (`None`, `NaN`, pandas' `NA`) for an optional input is
  that input left out: it takes its default. The call log holds the bound
  values. A default is bound when the function is defined (`n: int = "5"`
  is `5`). Outputs are checked as they are, never converted.
- **Records are closed**: a record type holds only its fields; a model's
  reply or a module's return with another member does not fit.
- **Error messages quote the value** at fault (its JSON, cut after 80
  characters), except a value a `log_content` setting drops.
- **Defaults count in `fn.version` by what is written**: `day=today()` is
  one version every day, `tone="kind"` to `"formal"` a new one (every
  function with a default has a new version once). A saved folder keeps a
  computed default's code (a node's `defaults`). `program.signature` and
  `program.interface` leave out every default, one inside a record type
  included, so a record whose field defaults to today no longer splits a
  function's ratings.
- **The call log**: each exchange's `request_hash` is the hash of what it
  sent (a re-ask's included); with tools, `outputs.calls` holds every tool
  call of the call, not the last step's (usually empty) list; `returned` is
  kept only when nothing is dropped; dropping `calls` no longer drops
  `reasoning` (nor the other way round); `process.lmcc` and `process.lm15`
  name the libraries that made the record.
- **`configure(program_observers=False)`**: the observers a program sets
  for itself get no event; the host's still do.
- **Ratings**: a rating with no person named (`by=`, or the caller's
  `user`) is made under the computer's `account`, and is kept on its own:
  on a shared account, one person's rating no longer replaces another's,
  and a disagreement shows as `disputed`. A program defined in a notebook
  or a script is known by its file too (`program.file`: the notebook, not
  the kernel's cell file), so `rated` no longer pools two notebooks'
  `summarize`; `rated(fn, any_file=True)` pools across files.
- A loaded function whose input's shape is a list of types (`["string",
  "null"]`) no longer crashes when its version is computed.

Stage 1 of the contract (`design/08-stage1-foundations.md`): logs every
language keeps and reads alike, and programs that say what they take.

- **Interfaces.** Every program has `.interface`, its inputs and outputs
  as data. A `@module`'s is derived from its function (`Any`, `object`
  or no annotation: opaque; `functai.JSON`: any JSON; a default makes an
  input optional; `outputs={...}` declares several; `interface={...}`
  declares it whole) and checked on every call: `InterfaceError`
  (`interface-input`, `interface-output`) before its code runs or when it
  returns. A missing or unknown input is `interface-input` naming it,
  derived or declared (`InterfaceError` is a `TypeError`, as Python's own
  wrong arguments are, and a `ValueError`); a value given under a name
  the interface lacks is never recorded. Interfaces every language would
  refuse are refused when the program is defined (`interface-malformed`),
  an AI function's included (an optional input whose default does not
  fit, or has no JSON form), and so is a module whose annotations name a
  type not defined yet (define the type first; a type defined earlier in
  the enclosing function is found). A positional-only parameter (before
  `/`) is refused where the program is defined, `@ai` or `@module`: a
  program's inputs are given by name. `@module` keeps the function's
  parameter and return types for type checkers, as `@ai` does; its `fn`
  is positional-only. Only modules check their inputs as the interface
  does (`InterfaceError`); an AI function's wrong arguments are Python's
  `TypeError`, as before.
  `Annotated[int, Field(ge=10)]` now puts its constraint in the shape;
  `x: str = None` is `Optional[str]`. A module's version includes its
  interface (every module's version changes once).
- **The call log is format 2** (both formats are read): `program.interface`
  on every record, `program.signature` for AI functions only, `saw` (the
  earlier calls a stateful function was shown, by id), `described` (values
  written as descriptions), `request_hash` on exchanges, `journal`.
  `functai.calllog.saw` and `check_kept` read what a call saw.
- **`log_content` per field**: `{"transcript": False}`, `{"*": False,
  "question": True}`. It only removes: a block's or `configure`'s `False`
  beats a function's own `True`, `FUNCTAI_LOG_CONTENT=0` beats everything.
  Dropping any field drops the reasoning and tool calls FunctAI adds, and
  every request, reply and error message. A misspelt name in a function's
  own map is `LogContentError`. `@module` takes `log_content`, `log_calls`,
  `caller`, `observers` and `journal`.
- **Stream events are format 2**: `tree`, `writer`, `seq`, `after`, `at`
  on every event (`event.position`), and a `Request` event that begins
  every request; a stream opened inside a tree shows the tree's numbers.
- **Observers and journals**: `observers=[...]` get the kept form of every
  event, each its own copy (they add up over layers, and never slow a
  call). A list is given each event as it is made, so it is complete when
  the call returns; a function is given them from a thread that runs
  only while events wait for it (`functai.flush()` waits until it caught
  up; at exit FunctAI waits at most 2 seconds for observers and journals
  to catch up), and nothing holds an observer once its trees have ended,
  an observer that failed included (it is given no more events).
  `journal=` keeps whole trees in a store while they run, best effort or
  `functai.Journal(store, required=True, timeout=30, retries=2,
  backoff=0.05)`: the call waits at its start, before each tool and at
  its end, each at most `timeout` (`JournalError` `journal-barrier`, or
  `journal-end` holding the outcome, with `err.settle()`); closing a
  stream ends its wait. A store answers `"kept"` or `"duplicate"`; any
  other answer is a refusal (`functai.Store` says the protocol); a store
  with `extend` is sent every event through it, what waits as one batch,
  except a subclass that overrides `append` and not `extend`, which is
  sent every event through its `append` (`batches = True` or `False` says so outright). A program
  cannot replace a host's journal (`journal-policy`), and when it names
  the host's journal again, the host's `timeout`, `retries` and `backoff`
  apply. An event made after its tree's last one (a call running on in a
  thread after the outermost call returned) goes to no receiver, with a
  warning. In a process forked inside a call, calls start a tree of their
  own, seen by that call's observers and kept in its journal; the tree it
  was forked from is written only by the process that started it. If an event of a tree cannot be
  put in its kept form, the kept log stops there (a required journal then
  refuses) rather than guess. Any observer or journal makes a call
  watched, so its requests stream (the reply is the same).
  `functai.MemoryStore` keeps logs by the contract's store rules (an
  event must pass the event schema); `functai.Follower` follows them, and
  stops at a format it does not know, when following, resuming or
  replaying; `follower.forget(tree)` lets a tree go.
- **Saved folders** carry every program's interface; `functai.describe(path)`
  reads it without loading; a folder another language wrote loads from
  its data (`functai.load`, `functai.saved.from_manifest`), optional inputs
  and their defaults included, bound from the interface as data (an
  optional input may come before a required one; an input may be named
  `class`: `fn(**{"class": ...})`). Every manifest is checked against the
  whole saved schema before anything else (`saved-malformed`; FunctAI
  carries the contract's schemas and checks them itself), every probe
  must have its fingerprint, in any folder, before any saved code runs,
  and `load(..., trust=True)` checks every interface before running saved
  code and compares what the loaded code accepts with what was saved.
  `load(path, node=...)` works for Python folders too. A module with
  `outputs={...}` saves and loads.
- **A function loaded from data is the function that was saved**: its
  interface is the one source of its inputs' names, order, requiredness
  and defaults, for calls, `using` copies, row demos, `map`, `evaluate`,
  optimizers, `bake` and `vectorize`, so each sends what the original
  sends; used as another function's tool, its tool schema is its
  interface's shapes, records whole. `inspect.signature` and `help` show
  its interface as Python can write it (inputs out of Python's order are
  keyword-only there, and a name such as `class` is given through `**`,
  named `inputs`, or `inputs_1`, ... when an input has that name). Its
  signature is the saved one: `module=` other than its own, and tools,
  are refused. It has no Python source, so `save` and `check` refuse it
  (`loaded-from-data`): keep the folder it came from.
- **What a call saw** is what it was shown: the turns captured when the
  call starts, as its plan shows them (a turn made for another signature
  is shown, and recorded, as its values alone), each known by the turn
  itself and by what it held when its call made it (a turn put in
  `history` by hand, or changed since, even inside its values, is
  `{"unrecorded": true}`). The turns shown are copied when the call is
  prepared (a turn holding a value that cannot be copied, from its JSON
  form), so changing `history` meanwhile changes nothing it is shown.
- **A tool's answer that is a record** (a pydantic model, a dataclass)
  is sent to the model as its JSON, no longer as its Python `str()`.

Breaking (see *Upgrading* in the documentation): the API says what it
does, and a type checker follows it.

- `@ai` is typed: an AI function keeps its parameters and return type
  for Pyright, mypy and editors, and its methods (`predict`, `using`,
  `opt`, `stream`...) are known. Wrong arguments are errors before any
  call; on a table's columns (`team(col.message)`) it is a column.
  `_ai` is typed as the value it stands for, so `return _ai` type-checks
  everywhere; the documentation writes `...` after the docstring, which
  Pyright (and so VS Code) accepts and mypy does not (`empty-body`): with
  mypy, write `return _ai`. `tests/typing/api.py` pins all of it
  (checked with basedpyright).
- `fn.predict(...)` replaces `fn(..., all=True)`; an input may be called `all`.
- The reply cache (`cache_replies=True`) no longer keeps a reply that
  could not be read, so with `retries=0` one bad reply no longer fails
  that input until the cache is cleared; and the optimizers' teacher
  (GEPA, InstructionSearch, synthesis), which samples on purpose, is never
  answered from it.
- Async: `await fn.acall(...)`, `await fn.apredict(...)`, and `@ai async
  def` for a function whose body is the model call. Calls run in a worker
  thread (the engine itself is synchronous), with the caller's settings.
- Improving returns a copy: `better = fn.opt(rows, ...)` (the data is the
  first argument; `trainset=` is gone), and the function is unchanged;
  `undo_opt`, `programs` and `latest_program` are gone. A `@module`'s
  `opt` returns a copy running with the improved states; the AI functions
  it calls are unchanged (`better.state()`, `save`, and `load`, which
  returns a copy). `fn.trials`: a search's candidates.
- By name, as in R and TypeScript: `functai.labeled_few_shot(fn, rows,
  k=)`, `functai.bootstrap_few_shot(fn, rows, teacher=)`,
  `functai.gepa(fn, rows, teacher=, selection=)`.
- `from functai import *` brings the API only; helpers (`flexiclass`,
  `docments`, `sig2str`, ...) are `functai.<name>`.
- `GEPA`: the instruction rewritten from the function's mistakes, read by
  a `teacher` model with feedback in words, over a Pareto pool of
  candidates (Agrawal et al., 2025), written in functai, with changes for
  one function: the teacher sees what it tried that failed, a proposal
  that copies an input is dropped, two candidates right on different rows
  are combined, ties go to the shorter instruction, and no row runs twice
  for one instruction (`design/04-gepa.md`). `f.opt(trainset=...,
  optimizer=GEPA(budget=300, teacher="gpt-6-sol"))`; `.trials` holds the
  search. Live, it took `gpt-5.4-nano` from 72% to 88% on refund
  decisions it never saw. One AI function at a time (a `@module` is
  refused).
- `evaluate()`'s table and `calls()` have `reasoning_tokens` and
  `total_tokens`: Gemini's `output_tokens` leave its hidden reasoning out,
  so a cost is `total_tokens - input_tokens` at the output price.
- `bake(..., log=False)` is silent (it raised).
- Eight tutorials (`docs/tutorials/`), from a first function to decision
  models with Jev, choosing a model by cost, and baking a model you own.

- `functai.datasets.refunds()`: 120 refund requests to the shop of
  `tickets`, with the facts its order system knows and the decision its
  refund rules give (the same table as R's `refunds`).

- Models that run only at temperature 1 (GPT-6, and Claude Opus, Sonnet,
  Fable and Mythos 5) no longer fail when a session sets `temperature=0`:
  the setting is left out of their requests, with one warning (the new
  `fixed_sampling` table in `contract/models.json`). A temperature of 1 is
  sent as is. GPT-6 models are declared reasoning models.

## 1.1.0 (2026-09-27)

New:

- **Streaming.** `fn.stream(...)` makes the same call as `fn(...)` (same
  retries, tools, call log line and value) and lets you watch it being
  written: `for piece in s` gives the answer's text as it arrives,
  `s.show()` prints it (labelling a reasoning, tool calls and retries),
  `s.events()` gives everything (every output's text, thinking, tool calls
  and results, retries, and the calls inside a module), `s.partial` a
  record or list as it fills in, and `s.result` the value. Works with
  `async for` and `await s`; closing the stream (or leaving a `with`
  block, or Ctrl-C) stops the call. Modules stream too (`module.stream`).
  Events have `.to_dict()` for web apps; the contract is
  `contract/streaming.md`. Logged calls record whether each request was
  streamed and how long its first piece took.
- **The call log.** `functai.configure(log_calls=True)` (or
  `FUNCTAI_LOG_CALLS=1` in the environment) writes every call of an AI
  function or module to `~/.local/share/functai/calls`, one line of JSON
  each: the typed inputs and outputs, every request and reply, tokens,
  time, the version, who called, and the call it ran in. Off unless asked
  for; writing never slows or breaks a call. `log_content=False` keeps only
  sizes, times and tokens (for programs that see secrets).
- **Right or wrong.** `functai.rate(prediction, "right")`, or `"wrong"` with
  the right `answer=`, writes a rating next to the call.
  `functai.rated(fn)` turns ratings into rows with known answers, ready for
  `evaluate` and `.opt`; `functai.calls(fn)` is the whole log as a table.
- **`fn.version`**: a fingerprint of everything a function sends besides
  its inputs (instruction, worked examples, layout, tools), and of its code
  when code of its own runs beside the model; the same for a program and
  its saved-and-loaded copy, and for the same function written in another
  language. Modules have one too. A call's `signature` leaves out how a
  language spells types, so ratings pool across languages.
  `prediction.call_id` names the call that produced a prediction.
- **The contract** (`contract/`): the log's format, JSON Schemas and test
  cases, so FunctAI in other languages and other tools (Chattering) read
  and write the same folder.
- **A default model.** With no model configured, functai picks one this
  machine can use (API keys first: `gpt-4.1-mini`, `claude-haiku-4-5`,
  `gemini-2.5-flash`, Groq, OpenRouter; then Claude, ChatGPT or Copilot
  subscriptions) and says which, once. Choosing one explicitly is unchanged.
- **`expected=`** in `evaluate` and `.opt`: the column holding the right
  answers, when it isn't named like the output (`expected="category"`), or
  a dict per output or answer field.
- **Records, field by field.** When the answer is a record (dataclass,
  pydantic, TypedDict) and the data has columns named like its fields,
  `evaluate` scores each field (`species_match`, …) as well as the whole
  (`exact_match`), and the run table has `pred_<field>` columns instead of
  one `pred_result` holding the record.
- **`fn.unpack(col.x)`**: one column per field of a record answer, for
  `table.mutate(**fn.unpack(col.note))`, still one model call per row.
- **`functai.datasets`**: `tickets()` (80 support messages, labelled by
  team and order number) and `field_notes()` (60 bird survey notes,
  labelled by species, count and behaviour), to learn with.
- `fn.state()` prints the instruction and the worked examples readably.
- The documentation website, with three ways in (a table of text, notes
  and documents, a prompt you already have).

Changed:

- **The repository holds every language.** The Python package moved into
  `python/`; the contract, the documentation and the design notes stay at
  the top, shared by the TypeScript, R and Julia implementations to come.
  Nothing changes for `pip install functai`. Installing from git names the
  folder: `pip install "functai @ git+https://github.com/MaximeRivest/functai#subdirectory=python"`.
  Python releases are tagged `python-v<version>` (e.g. `python-v1.1.0`).

Fixed:

- **`adapter="json"` with a record answer** (a dataclass, a pydantic
  model) was refused by OpenAI and Anthropic, whose strict schema mode
  wants every nested object closed. lmcc 0.8.4 closes them; functai now
  requires it.
- **A bare `_ai` is always the answer.** After a named output
  (`reasoning: str = _ai["..."]`), `return round(_ai, 2)`, `return
  _ai.upper()` and `return critique, _ai` used the named output instead of
  asking for the answer, silently. The answer is now its own output (typed
  by the return annotation, or by its place in a returned tuple).
- **`label = _ai` is the output `label`.** A plainly named output used later
  in the body (`return label if confidence > 0.7 else ...`) gave the
  answer's value instead of its own. Such lines are now bound the way
  `_ai["..."]` is.
- **Values are checked against their types all the way down.** A choice
  outside a `Literal` inside a record (`behaviour="swimming"`), text where a
  number goes, or a value outside an `Enum` is now an unreadable reply: the
  model is asked again once with what was wrong, then it raises. Before,
  the first slipped through silently and the last raised without a retry.

## 1.0.1

- An AI function used as a metric (a judge) gets plain data: typed
  parameters (`row: dict, prediction: dict`) are written into its prompt as
  JSON. Before, the prediction arrived as a `Prediction` object and every
  row failed with "Prediction is not JSON data".
- AI functions on columns work in modules with `from __future__ import
  annotations`: the return type is resolved, not read as the text `'str'`.
- Documentation: a runnable tutorial (`docs/tutorial.md`), every example
  rewritten for 1.0 and rendered with real outputs, and `tests/docs_live.py`
  to check that the README's code runs and to re-render them.

## 1.0.0

FunctAI no longer depends on DSPy. It is built on lmcc (layout: how values are
written into prompts and read back) and lm15 (one wire for every provider).
The public API is kept: `@ai`, `_ai`, `configure`, `all=True`, `stateful`,
`tools`, `module="cot"`, `.opt`, `undo_opt`, `@module`, `phistory`, docments.

- Chat templates in the decorator: `@ai(template=[system(...), turns(), user(...)])`,
  with lmcc's template language. The reply pattern in a template is also its
  parser; a template with no pattern and one output reads the whole reply.
- Layouts by name: `adapter=None|"xml"` (tags), `"chat"` (DSPy's sections),
  `"json"` (provider-enforced JSON), or any `lmcc.Adapter`.
- `module="cot"` uses a model's own thinking channel where it has one, a
  written reasoning section otherwise.
- Tools run in a tool loop (native tool calls where the model has them, text
  calls otherwise) instead of switching the program to ReAct; `max_steps`,
  `tool_errors`, `StepLimit`.
- Memory is kept as lmcc turns (`fn.history`, `fn.reset()`).
- Own optimizers: `LabeledFewShot`, `BootstrapFewShot`,
  `BootstrapFewShotWithRandomSearch`, `InstructionSearch` (MIPRO-style).
  Optimizers change only instructions and demos; `fn.state()`,
  `fn.save()`/`fn.load()`, `@module` save/load.
- Data is rows, results are tables (no `Example` class). A dataset is a list of
  dicts or any table dpyr reads (parquet, CSV, pandas/polars, Hugging Face);
  columns named like the parameters are the inputs. `evaluate(program, data,
  metric)` returns an `Evaluation`: `.score` (0 to 1), `.summary` (per metric,
  with a 95% interval: Wilson for 0/1 scores, Student's t otherwise), `.table`
  (one row per example: data, `pred_*`, metrics, error, seconds, tokens,
  model, run). A metric is `metric(row, prediction)` or a dpyr expression;
  several go in a list or dict. Failed rows are null in the table and count 0
  in the score. `compare(before, after)` pairs examples; `log=` and
  `functai.runs(folder)` keep runs as parquet files; `fn.map(table)`.
  Tables need `pip install "functai[data]"` (dpyr); scores do not.
- AI functions on columns: `df.mutate(topic=classify(col.text))`, with any
  mix of columns and constants as arguments, in `mutate()` and `filter()`
  (dpyr's `vectorize`). One model call per distinct input, 8 at a time,
  remembered for the session; displays only run the shown rows; the column
  is pinned to the prompt in use when it was written. `fn.vectorize(threads=,
  errors=, dtype=)` for options; @modules too (their return annotation types
  the column).
- Settings resolve at call time with a thread-safe cascade (`using` >
  function > `with configure` block > `configure`); unknown settings raise;
  any lm15 `Config` field is a setting.
- Reliability: one re-ask after an unreadable reply (`retries=1`), backoff on
  transient provider errors (`api_retries=3`), an opt-in in-memory reply
  cache (`cache_replies=True`; off by default), misspelled layouts repaired and reported.
- Accounts: `functai.login("claude" | "chatgpt" | "copilot" | "grok" | "kimi" |
  "openrouter" | "<provider>", key=...)`, `logins()`, `logout()`, `LoginRequired`;
  model prefixes `claude:`, `chatgpt:`, `copilot:`, `kimi:`. Saved lm15 logins are
  used automatically (after an explicit `api_key`, before the environment);
  `configure(auth=False | path)`. Claude Code / Codex CLI logins are used in place.
  Subscription providers get their API's abilities (native tools, thinking).
- Settings a model refuses are left out with one warning: `temperature`/`top_p`
  for OpenAI reasoning models, those and `max_tokens` for the ChatGPT backend;
  no stop sequences for xAI.
- Model, connection and layout on the program: `fn.lm`, `fn.adapter`, `fn.template`
  setters and `fn.using(lm=, client=, adapter=, template=)`; an adapter replaces a
  template and back (a copy's `adapter=` was ignored when the function had a
  template); `None` in `using` means inherit. `client=` takes an lm15 router or one
  provider's LM (`OpenAILM(api_key=...)`, `ClaudeCodeLM(...)`); `lm=` takes a model
  name or an lm15 `BoundClient`. Bad layouts, templates, models and clients are
  refused where they are written (a DSPy adapter now at definition).
- Saving programs with their dependencies: `functai.check(program)` (the dependency
  graph and every problem with its fix), `save(program, folder, record=[...])` (code,
  settings, demos, data files, pinned requirements and a full lock; refuses while
  there are errors; all or nothing), `verify(folder, trust=True)` (a fresh uv
  environment from the lock alone: byte-identical rendered requests, and recorded
  recordings replayed to the same results, no model called), `load(folder, trust=True)`
  (hash, package and prompt checks first), `functai.file("data/x.txt")`,
  `@ai(requires=...)`, `@module(requires=...)`, `python -m functai verify <folder>`.
- Baking (`functai[bake]`): `fn.bake(rows, student=..., teacher=..., labels=...)` trains
  a head model (outputs with a fixed set of answers; calibrated probabilities) or
  a generative student (`method="sft"`, LoRA when needed) and reports accuracy with
  its interval, top-3, calibration, the confident-share curve, speed, the teacher's
  accuracy on the same rows and break-even. `fn.using(lm=baked)` runs the function on
  the weights (the model keeps its training layout; a changed function is refused;
  concurrent calls are batched). `escalate_to` / `escalate_below` send unsure answers
  to a bigger model; `prediction.confidence`, `.escalated`, `.first`. `baked.serve()`
  serves a generative student with vLLM. Saved programs carry their weights
  (`models/`), pin torch/transformers, and verify what the weights answer.
- Prime Intellect (`functai[prime]`): the `functai-verifiers` harness, the
  `functai-rows` taskset, `functai.bake.prime.env_package` and `.config`;
  `on_unreadable="record"`, `prediction.refusal`.
- A spent or forbidden key (HTTP 403) is no longer reported as a missing login.
- `@module` finds AI functions called under another name or through helper
  functions (it missed them before).
- Notebooks: a dataclass's field comments now reach the prompt (they were lost for
  classes defined in cells).
- Inspection: `phistory()`, `inspect_history()`, `fn.render()`, `fn.explain()`.
- Types: tuples, sets, TypedDicts, `Any`, `Annotated[T, "description"]`;
  JSON integers read into `float` fields become floats.
- Breaking: automatic instruction writing (`autoinstruct`) and refinement are
  now opt-in and run at the first call, not at definition; `to_dspy()` is
  removed; `fn.signature` is an lmcc signature; DSPy adapters, modules and
  optimizers are refused with a pointer to their replacement.
- Requires Python 3.11+ (lmcc's type registry fails on 3.10's generic aliases).
- Fixes: `_ai` declarations in functions defined inside other functions are
  found; a user `__post_init__` error in a `flexiclass` is no longer swallowed.

## 0.12.0

- Includes the Pydantic compatibility fix and structured output/input improvements introduced after 0.11.0.
- See 0.11.1 notes below for details.

## 0.11.1

- Fix: do not convert Pydantic `BaseModel` subclasses to dataclasses when building signatures; avoids corrupting Pydantic internals.
- Feature: coerce JSON/dict model outputs back into declared output types (Pydantic models, dataclasses, or lists thereof).
- Feature: accept Pydantic v1 inputs by using `.dict()` when `.model_dump()` is unavailable.

## 0.11.0

- Auto-instruction: generate a first-pass system instruction from function code and types when creating an AI function. Controlled by `autocompile`/`autoinstruct`; enabled by default.
- Instruction refinement: improve the instruction over the first N calls using recent inputs/outputs as noisy hints (not gold). Configure via `instruction_autorefine_calls` and `instruction_autorefine_max_examples`; disable by calling `.freeze()`.
- Teacher modes: support `teacher` and/or `teacher_lm` on `@ai(...)` and `.opt(...)` to synthesize examples with `n_synth` for optimization. Accepts teacher programs (`FunctAIFunc`) or bare LMs.
- Example ingestion: `.opt(...)` now accepts DSPy `Example`s, JSON-like dicts, or `(inputs, outputs)` pairs; auto-coerces into a trainset.
- Default metric: when an optimizer requires a `metric` and none is provided, fall back to a conservative exact-match metric over all output fields.
- Optimization logs: record `.opt(...)` runs, including example counts and whether synthesis was used; retrieve via `optimization_runs()`.
- Bespoke instruction override: runtime overrides are honored in signatures; docstring-derived appendix is skipped when an override is set.
- History helper: new `phistory(n=1)` to quickly print `dspy.inspect_history()` for recent calls.
- Docs and examples: README/specs updated to use `phistory`; added `examples/typing_and_extraction` showing type-directed extraction patterns.

## 0.10.0

- Type-directed outputs: preserve function return annotations and variable annotations (including typing generics like `list[int]`, `dict[str, int]`), and pass them intact to DSPy.
- Clear precedence: when returning a variable from `_ai`, prefer that variable's annotation; otherwise inherit the function's return annotation; mismatch raises a helpful error with actionable fixes.
- Sentinel returns: `return _ai` / `return ...` uses the function's return annotation for the primary `result` output; auxiliary `_ai` variables become extra outputs.
- Extras typing: unannotated extra outputs default to `str`; annotated ones are preserved.
- Instruction text: keep output and field docs but avoid naming base types in instructions; only include class/dataclass field docs when meaningful.
- Schema friendliness: auto-convert plain classes-with-annotations to dataclasses and ensure nested types are schema-friendly.

## 0.8.1

- Adapters: accept duck-typed adapters and classes; allow per-call `@ai(adapter=...)` by scoping DSPy settings during invocation.
- _ai returns: resolve bare `_ai` placeholders inside tuples/lists/dicts (e.g., `return (id, email)`) to concrete values using return-order mapping.

## 0.8.0

- Add `flexiclass` to convert classes with annotations into dataclasses in place.
- Introduce `UNSET` sentinel; defaults use schema-safe `None` with post-init flip to `UNSET`.
- Harvest inline comments (docments) for:
  - Function parameters and return annotations
  - `_ai` output declarations (including comments after `_ai`)
  - Class/dataclass fields, with robust source fallbacks
- Append harvested guidance to signature instructions:
  - Parameter guidance, Output guidance, Return guidance
  - Qualified field guidance (e.g., `Account.id: User ID`)
- Auto-convert return/extra output classes (and nested types) to dataclasses to satisfy Pydantic schema generation.
- Normalize inputs: accept Pydantic BaseModel and dataclass instances as structured inputs (converted to dicts).
