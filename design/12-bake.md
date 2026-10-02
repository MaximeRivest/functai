# 12 — Bake: a model of your own for an AI function

*Status, 2026-10-02: implemented in Python; see
**As built** at the end for what differs from the proposal and what is
not yet proven. Proposed 2026-10-01. Bake has no users yet, so nothing here keeps the old API. Written for any functai user, on any machine: a laptop without a GPU, a Mac, a free Colab notebook, a workstation, a cluster, or only a cloud account. The first long-text attempt (a team distilling a 10k-token rewriting function, feedback of 2026-10-01) showed what the current code lacks; its twelve points are cited below as "item N", but it is one user among many. Numbered 12: 11 is plugins (`11-plugins.md`), merged first.*

## The goal

FunctAI should be the easiest place to train a model specialised on an AI function. "Easiest" means three things at once:

1. **One line in the common case.** `fast = fn.using(lm=fn.bake(rows))` does the right thing, with defaults read from the data, the function and whatever machine or account the user has.
2. **Every choice can be taken back.** Any default can be overridden, any layer used alone, and any good outside tool plugged in: TRL, Unsloth or Axolotl on your own GPUs, Tinker (Thinking Machines) or Prime Intellect in the cloud, or a trainer we have never heard of.
3. **What is trained is what is called.** The student sees in training the exact tokens it will see when the function calls it, whoever trained it and wherever it runs. This is the one guarantee nobody else can give, because only functai knows how the function lays out its calls.

We are few. So functai **owns only what only functai can do** and borrows everything else from projects with whole teams behind them. Today `bake/sft.py` has its own training loop. That loop is the cause of five of the twelve items (fp32 weights, fixed-row batches, no checkpoints, no resume, one GPU), and every one of them is already solved in TRL. The loop goes.

| functai owns | functai borrows |
|---|---|
| turning rows into the exact requests and replies the layout writes | the training loop: TRL + PEFT + Accelerate locally, Tinker, Prime |
| where the examples come from: rows, a teacher, rated calls, a whole program's traces | packing, padding-free attention, QLoRA, multi-GPU, checkpoints, schedules |
| several functions → one student, each through its own layout | fast kernels (flash attention, Liger, Unsloth), when installed |
| inputs fixed at bake time and left out of the prompt | serving at scale: vLLM, Tinker sampling, Prime deployments |
| judging the student with the function's own metrics and judges | experiment tracking: a metrics file any dashboard reads (W&B if asked) |
| plugging the result back: `fn.using(lm=baked)`, with checks | |
| the plan: what will happen, how long, what it costs, before anything runs | |

## 1. The one line

```python
baked = summarize.bake(rows)
fast = summarize.using(lm=baked)
```

Every default is decided from the data, the function and the place it will train, and printed in the plan before anything is spent (§2).

| Choice | Default | Why |
|---|---|---|
| method | `"head"` when every output is finite (Literal, Enum, bool), else `"sft"` | a classifier trains in seconds, even on a CPU |
| labels | the rows' own outputs; rows without them are answered by the teacher | |
| teacher | the function's own model (its `lm`) | it is the model being replaced; the plan shows what its answers will cost |
| where | see *Where it trains* below | |
| student | from a short table of tested models, the largest that trains where it is going at the data's lengths in reasonable time | a beginner should not have to know model names |
| weights | LoRA on all linear layers; 4-bit base (QLoRA) when the base does not fit at 16 bits and the GPU supports it; full weights when the model is small and they fit | item 1 |
| precision | bf16 where supported; fp16 with fp32 master weights on older GPUs (Colab's T4); fp32 on CPU | |
| lengths | from the data: the longest rendered example and the longest answer; the student's context is the only cap, and rows beyond it are left out with a note | item 5 |
| batches | by tokens, not rows; packed without padding when the kernels allow it (§3.4), else grouped by length | item 2 |
| schedule | warmup, constant, then a short decay (WSD): checkpoints taken before the decay are fair "quality versus data" points | item 10 |
| passes | 1 for large sets, up to 3 for small ones; early stop on validation loss | |
| validation | 2% of rows (at least 16, at most 500); loss every ~5% of the run | item 3 |
| checkpoints | every ~10% of the run and at the end; a run resumes from the last one | item 3 |
| GPUs | every free one visible, data parallel | item 7 |
| report | readability (the share of replies the layout reads back into the output type), held-out loss, 20 side-by-side samples; a **score** only with `metric=`, or exact match for finite outputs | item 4: exact match on open text measures nothing |
| teacher on the test rows | off; `compare_teacher=True` turns it on | it costs a full teacher pass over the test rows |

### Where it trains

`where="auto"` (the default) chooses, in this order:

1. **The user's preference**, when there is one: `where=` on the call, or `functai.configure(bake_where=...)` once. A list is an order of preference: `where=["here", "tinker", "prime"]` takes the first that can train this bake.
2. **Here**, when this machine has a GPU the student fits on at the data's lengths (CUDA, or Apple silicon for small students). The CPU only for head models and tiny generative ones: anything bigger would take days.
3. **A cloud service the user has set up.** A service counts as set up when its trainer's `check()` passes: its key or login is present, the account may train (Prime's hosted SFT is per-account), and it offers the student at the data's lengths. A key alone is not enough: a Prime key made for inference does not make Prime a trainer.
   - **One** passes: it is used.
   - **Several** pass: the cheapest estimate is used, the faster one on a tie. Cheapest, because the services train the same student on the same tokens, so the result should not depend on which one ran it; only price and time differ, and the plan shows both.
4. **None**: it stops before anything runs and prints the plan with every option, what each is missing (a key, a package, room), and the one word to add (`where="tinker"`, `student=`).

The plan always says which place was chosen, why, and what the others would have cost and taken. The choice is written in `plan.json`, so resuming a run never moves it to another service: its checkpoints live where it started.

### Money

No limits: users decide. Every paid step (the teacher's answers, training and judging on a service) is estimated in the plan from token counts and the provider's price table, and printed before it starts; `plan_only=True` shows it without running anything. What was actually spent is written in `run.json` and the report, next to the estimate. Time on the user's own hardware is shown, not priced.

### What to install

| Install | Gives | Needs |
|---|---|---|
| `functai[bake]` | head models, and generative students trained here (TRL, PEFT, Accelerate) | PyTorch |
| `functai[tinker]` | training and running on Tinker | the `tinker` SDK and a tokenizer; **no PyTorch** |
| `functai[prime]` | training and RL on Prime Intellect | the `prime` CLI |
| `functai[fast]` | packing kernels and Liger, where the platform has wheels | CUDA |

A laptop user with no GPU can bake a generative student through a cloud service without installing PyTorch at all.

## 2. The plan, before anything runs

```python
plan = summarize.bake(rows, plan_only=True)      # or functai.bake.plan(...)
print(plan)
```

```
bake summarize → Qwen/Qwen3.5-2B (sft, LoRA r32, bf16)
  rows       4,800 (4,704 train, 96 validation) · 1,200 answered by claude-sonnet-5 ≈ $3.10
  tokens     9.4M per pass (prompts 1.6k median / 4.1k p99; answers 220 / 610)
  longest    4,890 tokens · fits the student's context
  where      here: 1× RTX 4090, packed 16k-token batches · ≈ 50 min · fits (11.2 / 24 GB)
             tinker: ≈ $7 · prime: not set up
  kernels    flash attention ✓  flash-linear-attention ✓
  run        ~/.cache/functai/bakes/summarize-qwen3.5-2b-7f3a2c (resumes if it exists)
```

(The numbers are illustrative.) The plan is an object with every resolved setting. `plan.run()` starts it; `plan.using(where="tinker")` changes a setting. It is saved as `plan.json` in the run folder, which is what makes resuming exact.

## 3. The layers, each usable alone

```
 sources ──► Examples ──► Plan ──► Trainer ──► Run ──► Weights ──► Baked ──► fn.using(lm=baked)
 rows, teacher,  (functai)   (settings)  here/tinker/    folder,     merged,    layout +     judged with
 rated calls,                            prime/yours     metrics,    LoRA, or   signature +  evaluate()
 program traces                                          checkpoints remote     runner
```

### 3.1 Examples: what functai owns

```python
ex = functai.bake.examples(summarize, rows, student="Qwen/Qwen3.5-2B")
ex                       # a dpyr table, one row per training conversation
ex.save("train.parquet") # or .jsonl, or ex.to_hf() (a datasets.Dataset)
```

Columns:

- `messages`: the chat as the layout writes it (system, demos, the call, the reply). TRL, Axolotl, Unsloth and prime-rl read this directly.
- `prompt`, `completion`: the same chat split at the reply, for trainers that take that form.
- `input_ids`, `loss_mask`: only when `student=` is given: the exact tokens under that student's chat template, and which of them are the answer. This is the faithful form: TRL takes it as already-tokenised data (with `labels`), Tinker takes nothing else.
- `function`, `tag`, `weight`, `row_id`: which function, any label given to the row (which teacher wrote it, which source it came from), its weight, its source row.
- `prompt_tokens`, `answer_tokens`: so long rows can be filtered with dpyr before any training.

Sources, all producing the same table:

```python
functai.bake.examples(fn, rows)                             # rows with answers
functai.bake.examples(fn, rows, teacher="claude-opus-5.5")  # a teacher answers the rest
functai.bake.examples(fn, functai.rated(fn))                # what people marked right in use
functai.bake.examples(my_pipeline, inputs, teacher=...)     # a whole program: every AI call inside it,
                                                            #   traced while the teacher runs it
```

The table's format goes in the contract (`contract/baked.md`, *the examples*), with cases: R, Julia and TypeScript must write the same `messages` for the same rows, long before they train anything.

### 3.2 Several functions, one student

```python
baked = functai.bake.bake({extract: extract_rows, summarize: summarize_rows})
# or: functai.bake.bake(my_pipeline, inputs, teacher=...)   (a program)
extract.using(lm=baked)
summarize.using(lm=baked)
```

`baked.json` lists every function it answers, each with its signature and layout; `fn.using(lm=baked)` picks the function's entry, and refuses a function the model was not trained for. Item 6.

### 3.3 Inputs fixed at bake time

```python
baked = reply.bake(rows, fixed={"style_guide": STYLE_GUIDE})
```

Fixed inputs are given **by value**, not by name. They are left out of the student's prompt (the layout renders the call without them), and `baked.json` keeps a hash of each value. At call time the function is called exactly as before, with every argument; the baked model checks that the fixed ones still have their baked values and refuses, with a clear reason, when they do not: it never learned to read a different guide. Long constant instructions stop costing tokens in every training example and every call. Item 8.

`fixed=` refuses an input whose value differs between rows. An input that differs but is decided by another input (a per-section guidance chosen by the section's name) is named with `derived={"guidance": "section_name"}`: bake checks in the data that one determines the other, and leaves it out too.

### 3.4 Trainers: where training happens

One small protocol:

```python
class Trainer(Protocol):
    name: str
    def check(self, plan) -> list[str]                 # what is missing (keys, packages, room), before spending
    def estimate(self, plan) -> Estimate               # seconds and dollars
    def start(self, plan, examples, folder) -> Run     # returns at once
```

| `where=` | Trains with | Token fidelity | Result |
|---|---|---|---|
| `"here"` | TRL `SFTTrainer` on our tokens, PEFT, Accelerate for several GPUs; Liger, Unsloth and fast kernels when installed | exact (our tokens) | a LoRA adapter, and a merged HF folder |
| `"tinker"` | Tinker's `forward_backward`, our tokens, answer-only weights | exact (our tokens) | a `tinker://` checkpoint, downloadable |
| `"prime"` | Prime hosted SFT (`prime train`, `messages` column) | Prime's renderer: checked against our tokens when its `renderers` package is installed, else a warning | a checkpoint on a Prime volume |
| `"export"` | nothing: writes the examples and a ready config or script for TRL, Axolotl and Unsloth | exact with `input_ids`; template-dependent with `messages` | a folder to train from |
| any `Trainer` | yours | yours | yours |

Training here runs in its own process (`python -m functai.bake run
<folder>`), so a long run survives a closed notebook or a dropped SSH
session, and Accelerate can own the GPUs. In a notebook where that is not possible (some hosted notebooks kill child processes), it runs in-process and says so.

**Packing safely.** Putting several examples in one row is only correct when every layer knows where each example ends. Attention layers need a flash-attention kernel (the `kernels-community/flash-attn2` hub kernel needs no compiling). Hybrid models with linear-attention layers (Qwen3.5, and others coming) also need their kernel (flash-linear-attention): the slow fallback ignores the boundaries, so examples would leak into each other. TRL warns only about the first and not at all about the second, so functai checks both. When either is missing it does not pack: batches are still built by tokens from examples of similar length, with a little padding, and the plan says so. Item 9.

Facts behind the table, as of 2026-10-01: TRL 1.14 computes vocabulary scores only on answer tokens (`chunked_nll`), packs best-fit-decreasing and trains padding-free. Tinker trains Qwen3.5-4B and 9B at 64k context for $0.74 and $1.46 per million training tokens, and serves checkpoints through an OpenAI-compatible endpoint (beta, for testing) or its sampling client. Prime's shared LoRA service stops taking new runs on 2026-10-05; its hosted SFT is a per-account closed beta, full fine-tuning only.

### 3.5 Runs: long jobs you can leave

A run is a folder:

```
plan.json            every resolved setting, the data and code fingerprints
examples.parquet     the training conversations
metrics.jsonl        one line per log step: step, tokens seen, train loss, lr, validation loss, seconds
checkpoints/00012/   adapter + optimizer state (here), or the service's checkpoint id
run.json             running / stopped / done / failed, where, process or remote id
```

```python
run = summarize.bake(rows, wait=False)   # returns at once
run.metrics()                            # a dpyr table of metrics.jsonl, live
run.wait(); run.stop(); run.resume()
functai.bake.runs()                      # every run on this machine, with its state
functai.bake.run(folder)                 # reattach after a restart
baked = run.checkpoint(12).baked()       # any checkpoint as a usable model
```

Running the same bake again with the same plan resumes it. With `wait=True` (the default) it shows a progress line and returns the `Baked`. Item 3.

### 3.6 Baked: the model, wherever it runs

`Baked` keeps what it answers (functions, signatures, layouts, fixed values' hashes) apart from where its weights run:

```python
baked = functai.bake.load(folder)                          # a bake folder
baked = functai.bake.adopt("my-hf-folder", summarize, examples=ex)   # trained elsewhere (item 12)
baked.on("transformers")            # in-process (the default for small models)
baked.on("vllm")                    # a vLLM server started here
baked.on("tinker")                  # Tinker's sampling client
baked.on("http://host:8000/v1")     # any OpenAI-compatible server already running it
```

`adopt` is the bridge for models trained by hand: it checks that the folder's chat template writes the same tokens as `ex` did, then records the layout. The runner sends **token ids** wherever the server accepts them (vLLM's completions endpoint, Tinker's `ModelInput.from_ints`), so call-time tokens are the training tokens, not a re-rendering; elsewhere it sends messages and records that fidelity rests on the template. Lengths for serving (`--max-model-len`, `max_tokens`) come from the data.

### 3.7 Judging

```python
report = functai.bake.judge(baked, test_rows, metric=my_judge)   # or bake(..., metric=...)
```

`judge` is `functai.evaluate` run on `fn.using(lm=baked)`, plus the student-only facts: readability, speed, cost per call next to the teacher's. Any metric `evaluate` takes works: a function, an AI judge, a dict per output. `report=False` skips judging. Item 4.

## 4. What changes on disk (the contract)

`baked.json` format 2:

- `functions`: one entry per function answered: `name`, `signature`, `layout`, `fingerprint`, `fixed` (`{input: sha256}`).
- `weights`: `{"form": "merged" | "lora", "base": "<hf id>", "path": "model/"}`, or `{"form": "remote", "uri": "tinker://…"}`.
- `template`: the chat template's hash and the kwargs the tokens were made with.
- `run`: the plan and the run folder it came from.

`contract/baked.md` fixes this file and the examples table, with cases written by `contract/cases/make.py`. Head models keep their content, moved under `functions[0]`.

## 5. Four users, the same line

| User | `fn.bake(rows)` does |
|---|---|
| A support classifier, 3,000 labeled tickets, a laptop | a head model on the CPU in about a minute; exact-match accuracy with a range |
| A summarizer, 5,000 rows without answers, free Colab (T4) | asks the function's model for the answers (≈ $3, shown in the plan), LoRA in fp16 on a 0.8B or 2B student, about an hour, saved to Drive if asked |
| The same, on a MacBook, with a Tinker key | trains on Tinker (the only place set up), no PyTorch installed; runs through Tinker, or downloads and runs here |
| Two long-text rewriting functions sharing a 10k-token brief, two GPUs | trains here, packed, on both GPUs, resumable, judged with the team's own metric; the plan shows it takes about a day here, or ≈ $90 on Tinker (`where="tinker"`). With `fixed=` and the two functions as one bake, every example is shorter |
| A user with both a Tinker key and Prime training access, no GPU | the cheaper estimate of the two, named in the plan with the other's price; `configure(bake_where="prime")` makes Prime the choice from then on |

## 6. Order of work

Each stage ships on its own and is useful before the next.

| Stage | What | For | Size |
|---|---|---|---|
| A | `examples()` (all columns; rows and teacher sources), `adopt()`, lengths from the data, the report without exact match on open text | anyone training by hand, today | 1–2 days |
| B | the plan, choosing where (preferences, `check()`, the cheapest estimate), the run folder, training here on TRL (precision by hardware, QLoRA, packing with the kernel check, several GPUs, WSD, checkpoints, resume, own process), `judge` | anyone with a GPU, Colab included | 4–5 days |
| C | Tinker as a trainer and a runner, `functai[tinker]` without PyTorch | anyone without a GPU | 2 days |
| D | several functions, `fixed=`/`derived=`, tags and weights, examples from rated calls and from program traces | real applications | 3 days |
| E | Prime (when hosted SFT is open), `export` configs for Axolotl and Unsloth | | 2 days |
| F | `contract/baked.md` format 2 and the examples cases; the other languages write examples | | 2 days |

Training here comes before Tinker: it is free, needs no sign-up, and is what most people fine-tuning today use (often on Colab). Tinker comes right after because it is the only path for the many users without a GPU.

## 7. Trade-offs taken

- **TRL becomes a dependency of `functai[bake]`**, pinned to the tested version: it changes fast. The alternative, our own loop, is what failed.
- **Weights by repetition here, exact on Tinker.** TRL has no per-example loss weight; `weight=2` duplicates the row (fractional weights sample). Tinker takes per-token weights.
- **Fast kernels are optional.** Some must be compiled and have no wheels for every platform; without them training is correct, only slower, and the plan says which is missing.
- **Prime comes after Tinker.** Its open LoRA service closes on 2026-10-05 and its SFT is a closed beta whose renderer cannot take our tokens. The RL path to Prime (`bake/prime.py`) stays.
- **`where="auto"` uses a cloud service on its own** when this machine cannot train the student and a service is set up. There is no spending limit; the plan prints the price before anything starts, and `plan_only=True` previews it.
- **Several services: the cheapest wins.** Estimates come from price tables that can be out of date, and a user may prefer a service for other reasons (data rules, credits); one `configure(bake_where=...)` settles it for good.
- **"Here" wins over the cloud whenever it fits**, even when the cloud would be much faster. Free beats fast by default; the plan shows both.
- **`fixed=` trusts the data.** It checks that a value never varies; it cannot know whether the student does well without it. `judge` is the check.
- **The default student list is short** (currently Qwen3.5 0.8B, 2B, 4B, 9B) and must be kept up to date as models come out; any Hugging Face chat model still works by name.
- **`method="head"` is unchanged** apart from moving into format 2: it is small, fast and works.

## 8. Decided

- 2026-10-01, Maxime: no spending limit ("users are not kids"); the plan shows every price before it is spent.
- Several services set up: a stated preference first, else the cheapest estimate (this note's proposal).


## As built (2026-10-02)

Python, `python/functai/bake/`:

| module | layer |
|---|---|
| `functions.py` | `Entry`: one function as the student reads it (layout with replies from values, fixed and derived inputs, no demos) |
| `sources.py` | rows, the teacher (cost estimated before, counted after), a program's traces |
| `template.py` | messages → the student's tokens; the template's hash, end and stop tokens |
| `dataset.py` | `Examples`: the table, Parquet/JSONL with its metadata |
| `students.py`, `hardware.py`, `prices.py`, `recipe.py` | what a student is (from config files, no weights), what this machine has, prices (models.dev, Tinker's table, cached a day), the default settings |
| `planning.py` | `Plan`: the decisions, the printed plan, `run()` |
| `running.py`, `__main__.py` | `Run`: the folder, its own process, stop/resume |
| `trainers/` | `here` (+ `here_worker`, TRL), `tinker`, `prime`, `export` |
| `finish.py`, `baked.py`, `runners.py`, `judging.py` | the folder, the model, where it answers, the report |
| `contract/baked.md`, `cases/baked/`, `schema/baked.schema.json` | the examples and the folder, for every language |

Differences from the proposal, each for a reason:

- **No spending limit** (decided with Maxime): every price is printed in
  the plan; `plan_only=True` previews.
- **Packing is bake's own** (about 60 lines: best-fit decreasing into
  token budgets, positions restarting per example), not TRL's: TRL packs
  but has no token-budgeted mode for unpacked batches, which hybrid models
  need, and one path for both is simpler to trust. TRL still brings the
  loss (`chunked_nll`), checkpoints, resuming, DDP and the schedule.
- **Liger is off unless `liger=True`**: TRL's chunked loss already keeps
  the vocabulary's scores off prompt tokens, and Liger's patches are
  tested per model family.
- **Prime trains all weights** (its hosted SFT has no LoRA) and tokenizes
  with its own renderer; the plan says so. Its weights stay on a Prime
  volume (its CLI has no download): `baked.download()` explains how to
  adopt them.
- **Tinker's adapter format is not PEFT's**: `baked.download()` uses
  `tinker-cookbook`'s converter, which is not a dependency (it needs
  PyTorch); running on Tinker needs nothing more.
- **The contract pins messages, not tokens**: tokens are each student's
  template's rule; `adopt` checks them against the examples instead.
- The teacher's cost before labeling counts prompt tokens as characters /
  3.6 and answers at the mean of the rows that have them; unknown answers
  are said to be unknown.

Proven, and how:

- Training here end to end on a real Qwen3.5-0.8B (hybrid: grouped, not
  packed; LoRA; merged; called in-process; 20/20 on its test rows), and on
  a tiny model on the CPU in the tests (stop, resume from a checkpoint,
  the call's tokens equal to the training tokens, adopt refusing another
  template, export).
- Packing's correctness on a flash-attention hub kernel: a packed pair
  gives the same logits as each example alone (difference 0.0); with SDPA
  or eager attention it does not (0.5), which is why packing needs the
  kernel.

Not yet proven:

- **Several GPUs** (`torchrun`): written and reviewed, not run (the second
  GPU here was taken by another program).
- **Tinker**: run against a fake service client only (no key here); the
  SDK calls follow Tinker 0.31's documented API.
- **Prime hosted SFT**: closed beta, and the CLI here is 0.7.3; the
  commands follow Prime's documentation for 0.9.
- QLoRA on a model that needs it (9B on 24 GB), and vLLM serving of a
  merged Qwen3.5 student.
