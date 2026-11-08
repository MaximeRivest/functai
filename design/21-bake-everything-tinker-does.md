# 21 — Bake with everything Tinker does

*Proposed 2026-11-08. Nothing here is built. It extends design 12 (bake)
and design 14 (bake in a real project), and uses design 20 (measures)
for rewards. Sources: the Bigspin "EyeTrack" conversation (Chattering
`01a1b0f8ba04`, 2026-11-06 to 11-08, messages #839–#1331), its scripts
in `~/Projects/gaze-models/tinker/` (1,112 lines), Tinker's documentation
read on 2026-11-08 (SDK 0.39, tinker-cookbook 0.6.1, tml-renderers
0.4.1), and `python/functai/bake/` on master (29a450f).*

## Why

An agent working for Bigspin used Tinker to teach Inkling 2 Small to
predict where people's eyes go on a screen. In two days it ran nine
training runs for about $180, and the model beat the published state
of the art (SeekUI, as the agent re-ran it) on 10 of the 12 measures of
the authors' own test and tied on the other 2 (ScanMatch 0.380 against
0.343). It did not use FunctAI. It wrote 1,112
lines of scripts on Tinker's SDK, and most of those lines do things
bake should do for everyone:

| What the agent wrote by hand | Lines | Should be |
|---|---|---|
| pictures into the model's tokens, with a workaround for a Tinker bug (widths that are a multiple of 40 px are refused) | ~40 | bake: functions with picture inputs |
| a held-out split by screen, scored with the task's own metric before, during and after training | ~120 | bake + design 20 |
| keeping only the best checkpoint by that metric, deleting the rest (each copy is 17 GB, storage is billed) | ~40 | bake |
| choosing the sampling temperature (greedy decoding looped after light training; 0.7 was best) | — | bake |
| stage 2: reinforcement learning with a reward (GRPO, 4 answers per screen) | ~170 | bake, `method="rl"` |
| a reward on a *group* of answers (the energy score), against *all* the people who searched that screen | ~150 | bake + a reward helper |
| writing 5 answers in one reply, each different from the ones before | ~220 | a different program, trained by the same `rl` |
| cost per run from token counts, rate-limit probing | ~60 | bake's plan |
| resuming from a stage: the first stage-1 run kept no training state, so stage 2 could not start from it and stage 1 ran again ($10.20) | — | bake: a finished run can always be continued |

FunctAI's Tinker trainer today (`bake/trainers/tinker.py`, 243 lines)
does only the first row's simpler cousin: supervised training on text,
judged by loss. It refuses pictures (`functions.py`: "images and files
are not trained yet"), and it has **never run against the real service**
(design 12, *Not yet proven*: "run against a fake service client only").
Its runner samples at temperature 0, which is the setting that looped.

Tinker can do much more than supervised training. This note says how
bake supports all of it while keeping its one promise (design 12): *what
is trained is what is called*.

## What Tinker offers (2026-11-08)

| Capability | Tinker's API | In bake today |
|---|---|---|
| Supervised training (LoRA, any rank) | `forward_backward(data, "cross_entropy")`, `optim_step` | yes, text only, untested live |
| Pictures (and audio on Inkling and Inkling-Small) | `ImageChunk` in `ModelInput`; the cookbook's renderers turn chat messages with image parts into tokens | refused |
| Inkling models | need `tml-renderers` (needs PyTorch ≥ 2.10); a *thinking effort* from 0.0 to 0.99 is part of the prompt | no: bake uses Hugging Face chat templates only |
| Reinforcement learning | sample with `num_samples`, then `forward_backward` with `importance_sampling`, `ppo`, `cispo`, `dro`, `gspo` or `stable_reinforce`; KL to a reference as part of the reward; *inflight* sampling for long rollouts | no (Prime only, through an environment package) |
| Rewards on a whole group of answers | the cookbook's `EnvGroupBuilder.compute_group_rewards` | no |
| Multi-turn rollouts with tools | the cookbook's `MessageEnv` (`initial_observation`, `step`) and `tool_use` | no |
| Preferences (DPO) | `forward_backward_custom` with the DPO loss; cookbook `preference` | no |
| On-policy distillation | the teacher's log-probabilities on the student's own samples (`compute_logprobs`, top-k) | no (only off-policy: the teacher writes answers) |
| Decision models | one forward pass, the probability of each allowed answer (`target_prompt_logprobs`); trained on hard labels or on probabilities (Brier, any loss) | no (bake's `head` is a local encoder, which cannot read pictures) |
| Any loss written over log-probabilities | `forward_backward_custom` (1.5× the compute) | no |
| Checkpoints | `save_state` (to continue training; 50 GB on Inkling 2 Small), `save_weights_for_sampler` (17 GB), `save_weights_external` (to download), a TTL on each, `RestClient` to list and delete | partly: state saved, never deleted, no TTL |
| Storage | free to 100 GB *and* 100 checkpoints, after which **training stops**; paid storage bills all of it at $0.10 per GB-month | not checked |
| Spending | `RestClient.get_billing_usage` per session; `user_metadata` on runs and checkpoints | estimated only |
| Serving | the native sampler (exact tokens); an OpenAI-compatible and an Anthropic-compatible endpoint (beta, chat rendered with the model's default template); serverless Inkling | native sampler, at temperature 0 |

Prices that matter here: Inkling 2 Small $0.30 per million prompt
tokens ($0.03 when cached), $1.00 per million generated, $1.50 per
million trained; Qwen3.5-4B $0.33 / $1.005 / $0.737. Throughput measured
by the agent: 15 to 20 requests a second from 32 at once upwards, none
refused at 256.

## The design in one picture

```
            what is learned (method)                    where (trainer)
  rows ─┐   sft       imitate known answers           here   TRL (sft today; dpo, grpo, gkd later)
ratings ┼─► rl        a reward from a criterion   ──► tinker the SDK; the cookbook's RL loop
  pairs ┤   prefer    pairs people preferred          prime  verifiers environments (exists)
teacher ┘   distill   a teacher's own probabilities   export files for any trainer
            decision  a probability per choice
                                  │
                                  ▼
             criteria (design 20) score it during and after, choose the checkpoint
                                  │
                                  ▼
                  Baked: weights + how its prompts are rendered + how it samples
                                  │
                                  ▼
                     fn.using(lm=baked)   ·   bake(..., start=baked) continues
```

Three ideas carry it:

1. **`method` is what the student learns from; `where` is where.** Today
   bake has `method="head" | "sft"`. It gains `rl`, `prefer`, `distill`
   and `decision`. Each trainer says which methods it supports. Tinker
   supports all of them; "here" supports `sft` now and the others later
   through TRL, which has a trainer for each.
2. **A reward is a criterion.** Design 20 already says what a good answer
   is: a `Criterion` with a judge (code or an AI function), asked by name
   for what it needs. The same object scores `evaluate`, picks the best
   checkpoint, and is the reward in `rl`. One way to say "good", used
   everywhere.
3. **Stages chain.** `bake(..., start=earlier)` continues from an earlier
   bake's training state. Stage 1 imitates, stage 2 reinforces, as in
   SeekUI's paper and the agent's runs.

## What the eye-tracking job becomes

```python
from lm15.types import ImagePart
from functai import ai, _ai, Criterion
import functai.bake
from functai.bake import rewards

Point = tuple[int, int]

@ai
def scanpath(screen: ImagePart, width: int, height: int, target: str) -> list[Point]:
    """Where a person's eyes stop, in order, while searching this screen
    for the element whose text is the target."""
    why: str = _ai          # what they look for first, and why
    return _ai

# one row per person: screen, width, height, target, why (written by a teacher), result (their path)
match = Criterion("scanmatch", "score", scale=(0, 1),
                  judge=lambda answer, expected, width, height: scanmatch(answer, expected, width, height))

s1 = scanpath.bake(rows, student="thinkingmachines/Inkling-2-Small", where="tinker",
                   criteria=[match], held_out_by="screen")          # stage 1: imitate
s2 = scanpath.bake(rows, method="rl", start=s1, reward=match)       # stage 2: one person's path
crowd = scanpath.bake(rows, method="rl", start=s1,                  # or: match the crowd's spread
                      reward=rewards.energy(path_distance), group=8)

functai.evaluate(scanpath.using(lm=s2), test_rows, criteria=[match], repeats=10)
```

About 30 lines in place of 1,112. `scanmatch` and `path_distance` stay
the user's: they are the task. Everything else the agent did by hand,
bake does, and shows in the plan before spending anything.

## 1. Pictures, and how a prompt becomes tokens

**Rendering.** Today a student's tokens come from its Hugging Face chat
template (`bake/template.py`: `Template` holds the template's hash and
its keyword arguments). Tinker's models are rendered by the cookbook's
renderers, and Inkling's *only* by `tml-renderers`. `Template` becomes
one of two kinds:

- `{"kind": "hf", "sha256": …, "kwargs": {"enable_thinking": false}}`, as
  now;
- `{"kind": "tinker", "renderer": "tml_v0", "kwargs": {"effort": 0.0},
  "versions": {"tinker-cookbook": "0.6.1", "tml-renderers": "0.4.1"}}`.

Training and calling go through the same renderer with the same keyword
arguments, so the promise holds. The default for a student on Tinker is
the renderer Tinker recommends for it
(`model_info.get_recommended_renderer_name`).

**Thinking effort.** A function's own reasoning setting chooses it,
through the presets Tinker documents (none 0.0, minimal 0.1, low 0.2,
medium 0.7, high 0.9, xhigh 0.99). A student is trained at **0.0** by
default, the equivalent of today's `enable_thinking=False`. The agent
measured why: with thinking, one call spent 6,000 hidden tokens before a
short answer, about 20 times the cost of a training example. A function
that wants thinking to be learned declares a reasoning output (`why`
above). It is visible, trained and judged, like everything else.

**Pictures in the examples.** The examples table (`contract/baked.md`)
gains picture parts in `messages`: `{"type": "image", "media_type",
"sha256", "path"}`. The pictures are stored next to the table, named by
hash, so the agent's 1,362 paths over 646 screenshots would store each
screenshot once. This is a contract change, in every language, with
cases. Training here on pictures (TRL with a processor) comes later
(stage F). Until then, `where="here"` refuses with that reason.

**Tinker's picture bug.** tml-renderers 0.4.1 counts one token column
too many when a picture's width is a multiple of 40 px, and the server
refuses the request. Bake pads such pictures with one black column on
the right (no coordinate moves). The fix lives in one function, a test
pins it, and it goes once a fixed release is out. Report it upstream.

## 2. Judged by its criteria while it trains (design 14, item 18)

On Tinker this is cheap: `save_weights_and_get_sampling_client()` gives
a sampler for the current weights that costs no storage. Every
`eval_every` (default: ten times per run, and before the first step):

- `evaluate` the function on the held-out cases with its `criteria`
  (`metric` works too), at the temperatures in `temperatures=` (default
  `(0, 0.7, 1.0)`);
- the results go to `run.metrics()` next to the loss;
- when a check is the best so far, it is kept as a sampler checkpoint
  and the previous best is deleted.

The baked model is the best check, not the last step, and it **records
the temperature that scored best**. Calls use that temperature unless
the request sets one. The agent's pilot is why: greedy decoding scored
0.084 with 86 of 143 answers looping, and temperature 0.7 scored 0.281.

**Held out.** Design 20's split (`split` column, or the hash of the case
key) decides it, the same in every language, so a case `evaluate` calls
held out is never trained on. `held_out_by="screen"` names a column
whose values stay on one side together. Otherwise the two people who
searched one screen for different targets would put that screen on both
sides, and the scores would flatter the model.

## 3. `method="rl"`: learning from a reward

```python
scanpath.bake(rows, method="rl", start=s1, reward=match, group=4,
              loss="importance_sampling", kl=0.0, temperature=1.0, epochs=2)
```

**The reward** is a criterion (or a plain function, read as a score
criterion). Its judge is asked by name, as in design 20, with two new
names:

| it asks for | it gets | example |
|---|---|---|
| `answer`, `outputs`, `expected`, inputs by name | one answer, and this row's known answer | ScanMatch against the person |
| `references` | **every** known answer of the case: all rows with the same inputs (the same case key) | the distance to each of the people who searched this screen |
| `answers` | the whole group of answers to the case; it returns one value per answer | the energy score's share for each answer |

A judge that asks for `answers` makes a *group reward*. A judge that asks
for `references` uses all of a case's people, which made round 2's
signal much less noisy than round 1's.

**`rewards.energy(distance)`** is the one group reward bake ships. It
rewards a group of answers for being spread the way the references are
spread: no more and no less. It is the proper scoring rule for matching
a distribution (Gneiting & Raftery 2007). Each answer's reward is
`−mean d(answer, reference) + mean d(answer, other answers)`. The spread
term has weight 1, not ½, because both answers of a pair come from the
model. The agent first wrote ½, and its run drifted back to one path; a
shipped helper with a test removes that trap. Honest note: on the eye
data, energy-score training did *not* beat stage 1. The careful
measurement showed stage 1 was already within noise of real people,
with 1 to 5 people per screen. The helper is a tool, not a promise.

**The loop is Tinker's cookbook's**, not ours (`tinker_cookbook.rl.train`):
it already handles groups, advantages centred per group, dropping
groups whose answers all scored the same (no signal; an option bake
turns on), the KL penalty, the six losses, async sampling, retries and
its own checkpoints. Bake
supplies two pieces and nothing else:

- an **environment per case** (`MessageEnv`): its first observation is
  the function's layout rendered for that case; `step` parses the reply
  with the layout into typed outputs (an unreadable reply scores the
  worst value, as a failed call does in design 20) and returns the
  reward;
- an **evaluator** that runs section 2's checks.

**What the report adds, always.** Design 20's *self-agreement* (how alike
a setup's answers to the same case are), before and after, on the
held-out cases. Reinforcing "match this one person" made the eye model's
paths nearly identical (0.41 → 0.78 alike; real people 0.31), and the
reward rose the whole time. Self-agreement shows that collapse on every
`rl` run, whatever the task, without anyone thinking to look.

**Measuring what was trained on.** When the reward is also the criterion
in the report, the report says so ("trained on: scanmatch") and suggests a
second criterion. Rows marked held out are refused for training, as for
the optimizers (design 20).

**Programs, not only functions (stage E).** A `@module` or an agent with
tools is the same environment with more steps: each model turn is
sampled and trained, and tool results are part of the prompt, never
trained on. The reward judges the program's final outputs. Tools run
unattended during training, so bake refuses tools with
`effects="changes"` unless `allow=` names them (design 17's approval
rules, applied to training).

**Several answers in one reply** (the agent's run B) needs no new method.
It is another function (`-> list[list[Point]]`) with a group-style judge
(`answers` of one reply are its items). The agent found it trains
reliably but did not beat independent samples. That is worth knowing,
but it is a choice of program, not of bake.

## 4. `method="prefer"`, `"distill"`, `"decision"`, and any loss

**`prefer`** (DPO): the pairs come from design 20's `prefer()` ratings
(`functai.preferred`), or from a table of `chosen`/`rejected` answers.
Defaults are Tinker's: β = 0.1, learning rate 1e-5. The reference is the
starting weights.

**`distill`, on-policy**: the student answers, and the teacher scores
every token the student wrote. The student learns where the teacher
disagrees (reverse KL, per token). This needs the teacher's log
probabilities on the student's tokens, so the two must share a
tokenizer. Bake checks this and refuses otherwise, saying why and
offering the off-policy route, which exists today and works across
families: the teacher writes answers and the student imitates them. The
agent proposed exactly that: Inkling 2 Small's skill passed to a 4B
model that runs on the DGX Spark. With `fixed=` (design 12, §3.3) and
`teacher="self"`, the teacher is the student's own starting weights
*shown* the fixed inputs. This is context distillation, Tinker's
"prompt distillation" and SDFT, done the way bake already leaves fixed
inputs out.

**`decision`**: for a function whose output is finite (Literal, Enum,
bool), one forward pass gives a probability for each allowed answer.
This is Tinker's decision-model recipe, the same idea as bake's `head`
but on a language model, so it reads pictures and long context.
Training uses the right answer (cross-entropy), or a probability per
answer when several people rated the case (design 20 `label` ratings
give the distribution), fitted with the Brier score. A call returns the
answer, and the probabilities in its record, as `head` does.

**Any loss**: `loss=` takes Tinker's names for `rl`, or a function
`(data, logprobs) -> (loss, metrics)` for `forward_backward_custom`, as
the way out for research.

## 5. The plan, money and storage

Before anything runs, the plan says (`plan_only=True` shows it and
spends nothing; the numbers are the agent's stage 2):

```
bake scanpath → thinkingmachines/Inkling-2-Small (rl from s1, LoRA r32, effort 0.0)
  cases      1,219 paths on 582 screens · held out: 143 paths on 64 screens, by "screen"
  reward     scanmatch, against the case's person · group 4 · importance_sampling · KL 0
  tokens     ≈ 2.6M sampled, 3.1M trained per pass (answers ≈ 250 tokens, measured on 20 cases)
  money      ≈ $19 (sampling $5.10 · training $13.90) · checks ≈ $0.40 each
  storage    best sampler 17 GB, final state 50 GB: $6.70 a month at Tinker's $0.10/GB-month
             you use 67 of 100 free GB: this run needs paid storage, or training stops at 100 GB
  your data  646 screenshots are sent to Thinking Machines
  runs only  on Tinker: Inkling 2 Small's weights are not published (594B parameters; ≈300 GB at 4 bits)
```

- **Answer lengths are measured, not guessed.** The first check is also
  the cost probe: before training, bake samples 20 cases and prices from
  their real lengths.
- **What was spent is read back** from Tinker's billing records for the
  run's session (bake tags every session and checkpoint with
  `user_metadata`: the functions, the plan's fingerprint and the run
  folder). `run.json` and the report show estimate and actual side by
  side. No spending limit (design 12, *Decided*).
- **Storage is checked first.** Training stops when the free tier is
  full, so the plan reads current usage and says what this run adds.
  Intermediate saves get a TTL (one day). The final training state is
  always kept, so `start=` can continue from any finished bake: the agent
  lost $10.20 to a stage-1 run that kept none. `keep_state=False` skips
  it. `functai.bake.storage()` lists every checkpoint bake made on the
  account, with its size, monthly cost and the run it belongs to.
  `functai.bake.storage(remove=…)` deletes them.
- **Errors that are really waiting.** A 402 ("billing status") in the
  first minutes after paying is retried with a note, because Tinker takes
  a minute to see a payment.
- **Concurrency**: 32 requests at once by default (the agent measured
  that more only queues), `num_samples` for a case's group (the prompt
  is billed once, at the cached price after the first).
- **Pipelined steps**: each step's next `forward_backward` is sent before
  the current one is awaited. On Tinker's clock cycles, a naive loop
  uses two or three cycles per step.

## 6. The baked model, and calling it

`baked.json` (format 2) gains, for a Tinker student:

- `template` of kind `tinker` (section 1);
- `sampling`: `{"temperature": 0.7}`, chosen by the checks;
- `weights`: the best sampler's `tinker://` path, the final state's,
  and the trained-on criteria with their held-out scores.

`fn.using(lm=baked)` calls the native sampler with the exact rendered
tokens. The other routes, honestly:

| From | How | Same tokens as training? |
|---|---|---|
| Python | native sampler | yes |
| TypeScript, R, Julia | `functai serve` the function in Python and call it (serving exists) | yes |
| TypeScript, R, Julia, any tool | Tinker's OpenAI-compatible endpoint | only if Tinker's default chat template renders the layout as the renderer did: **unverified for Inkling**; bake checks on the examples and says which |
| here, after download | `save_weights_external` and its signed URLs (replacing the cookbook converter for models that support it), then `baked.on("vllm")` | the same check as `adopt` (design 12) |

A student that cannot leave Tinker (Inkling today, and anything too big
to run anywhere else) is lock-in, and FunctAI's principle is no lock-in.
The plan says so in its last line. The way out it offers is `distill` to
an open model that runs here.

## 7. Order of work

Each stage ships alone and is accepted by **reproducing the agent's
measured results with FunctAI**: same data, same test, and scores within
the agent's sampling ranges.

| Stage | What | Accepted when | Live cost |
|---|---|---|---|
| A | Tinker trainer made real: SDK 0.39 (async), renderer templates, Inkling and effort, pictures in examples (contract, all languages write them), the picture workaround, checks by criteria while training, best checkpoint, temperature chosen, state always kept, storage and billing, `held_out_by` | `scanpath.bake(rows, …)` (5 passes) scores ScanMatch 0.315–0.334 on the 250 test cases (agent: 0.323) | ≈ $12 |
| B | `method="rl"` through the cookbook loop; rewards as criteria with `references` and `answers`; `rewards.energy`; self-agreement in the report | stage 2 from A scores 0.375–0.385 (agent: 0.380), and the report shows self-agreement collapsing without being asked | ≈ $20 |
| C | `decision` and `prefer` | a finite-output function with pictures, against `head` and against the untrained model, with ranges | ≈ $5 |
| D | `distill`: on-policy (same tokenizer), off-policy from a Tinker teacher, `teacher="self"` with `fixed=` | Inkling 2 Small → Qwen3.5-4B, run here, against the 4B trained alone | ≈ $15 |
| E | `rl` on modules and agents with tools | a two-step module and a tool-using agent improve on held-out cases | ≈ $20 |
| F | the same methods `where="here"` (TRL: DPO, GRPO, GKD), pictures here | the stage C and D tasks on lambda's GPUs | free |

## Trade-offs taken

- **The RL loop is borrowed from tinker-cookbook**, pinned to a tested
  version. It is a research codebase that changes fast (0.6.1 today), and
  we follow its changes. The alternative, our own ~200-line loop, is what
  the agent wrote, and the subtle parts (off-policy corrections, KL,
  retries, async) are exactly where a small team gets it wrong. The
  supervised loop stays ours: it is 40 lines with nothing subtle, and it
  already exists.
- **`functai[tinker]` now needs PyTorch.** Design 12 promised it would
  not. Inkling's renderer and the cookbook require PyTorch ≥ 2.10. A
  laptop can still train on Tinker. It just installs more.
- **Training on the criterion you report** inflates the report unless
  the cases are held out. Bake enforces the held-out side and labels
  trained-on criteria. It cannot know if the criterion itself can be
  gamed: only a second criterion, or people (design 20), can.
- **Students learn without hidden thinking (effort 0.0) by default.**
  About 20 times cheaper per example. A task that needs long thinking
  may lose quality, and the function's own reasoning output is the
  visible, judged alternative. `reasoning=` changes it.
- **Calls sample at the temperature the checks chose**, not greedily.
  The same input can get different answers, and FunctAI's records keep
  each one, so replay is unaffected. Greedy stays one argument away.
- **The final training state is always kept** (Inkling 2 Small: 50 GB,
  $5 a month). Losing it costs a rerun; keeping it costs money shown in
  the plan. Intermediate saves expire after a day.
- **Decision models are an explicit method**, not chosen by `auto`. For
  finite outputs `auto` stays `head` for text. For pictures, which a head
  cannot read, it is open below.
- **One reward helper only** (`energy`). Others are a line of the user's
  code. Shipping many would suggest they help, and the only one measured
  (here) did not.

## Open

1. **Whose Tinker account tests this?** The working key is Bigspin's,
   paid by Bigspin's card, in `~/Projects/bigspin/.env`. FunctAI's
   development (stages A to E, about $70) should not spend a client's
   credit without its say. Use your own account, or get Bigspin's
   agreement. The eye-tracking data is Bigspin's work product too: can
   FunctAI's acceptance tests use it, or only the public VSGUI10K part
   (SeekUI's released data, which is all the agent trained on)?
2. Should `auto` choose `decision` for finite outputs with pictures (a
   head cannot read them), or keep asking?
3. `references` and `answers`: the right names? (Design 20's judge
   already gets `expected`, `answer`, `outputs`, `item`, `first`,
   `second`.)
4. Does the OpenAI-compatible endpoint render Inkling prompts exactly as
   tml-renderers does? If yes, the other languages can call Tinker
   students directly. To check in stage A, on the examples.
5. Does the cookbook's RL loop take picture observations from a
   `MessageEnv` as its sampling examples do? Likely, since both use the
   same renderer, but it is unverified. To check first in stage B.
6. The agent's last finding: variety needs many people per screen, not
   a cleverer reward. That is data collection for Bigspin, not FunctAI.
   FunctAI's part is design 20's draws and blind ratings, which already
   exist.
