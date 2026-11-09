# 21 — Bake with everything Tinker does

*Proposed 2026-11-08. Nothing here is built. It extends design 12 (bake)
and design 14 (bake in a real project), and uses design 20 (measures)
for rewards. Sources: the Bigspin "EyeTrack" conversation (Chattering
`01a1b0f8ba04`, 2026-11-06 to 11-08, messages #839–#1331), its scripts
in `~/Projects/gaze-models/tinker/` (1,112 lines), Tinker's documentation
read on 2026-11-08 (SDK 0.39, tinker-cookbook 0.6.1, tml-renderers
0.4.1), and `python/functai/bake/` on master (29a450f).*

*Revised the same day after studying what needs PyTorch and how
TypeScript can use Tinker (section 0, evidence in `21-evidence/`). Three
decisions of the first draft were reversed: FunctAI renders tokens from
the models' published files instead of Tinker's renderers, writes its
own training loops instead of borrowing the cookbook's, and builds
TypeScript first, for Quiver. Neither language needs PyTorch.*

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

## 0. No PyTorch, and Tinker from TypeScript

Nothing runs locally on Tinker, so why would anything need PyTorch?
Measured (`21-evidence/`):

| Piece | Needs PyTorch? | Why |
|---|---|---|
| `tinker`, the SDK | **no** (optional: used only if installed) | it sends JSON and token numbers over HTTP |
| `tml-renderers`, Inkling's renderer | **yes, to import it** | only audio uses it (sound becomes feature tensors, "DMEL"), but `chat.py` imports it on its first line and the compiled part looks for PyTorch's library in memory at load time |
| `tinker-cookbook` | **yes**, a declared dependency | its renderers, its custom losses (autograd), weight conversion |

The agent's environment was **5.7 GB**, 4 GB of it NVIDIA libraries for a
GPU it never used. With CPU-only PyTorch the same environment is
**0.84 GB**. Without the cookbook and `tml-renderers` it is the SDK and
its dependencies alone.

What a client really has to do, beyond HTTP:

1. **Turn a request into the model's tokens.** Text: the model's
   tokenizer and chat template. Pictures: Tinker's servers do all the
   picture processing. The client sends the PNG or JPEG bytes (base64 in
   JSON) with the number of tokens the picture will take, which is
   arithmetic: `ceil(w/40) × ceil(h/40)` for Inkling, with no resizing
   (an 8000×8000 picture is 40,000 tokens); Qwen's resize rule from its
   published `preprocessor_config.json`. The 40-px bug the agent hit is
   the Python client's arithmetic (`floor(w/40)+1`), not the server's.
2. **Losses that are not built in** (DPO, a Brier loss on decision
   probabilities): Tinker computes the model's log-probabilities, the
   client computes the loss's derivative with respect to them, Tinker
   does the rest. The cookbook uses PyTorch autograd for that derivative;
   for the losses bake needs, it is a short closed formula.

**The decisive test.** Inkling-Small (first generation) publishes its
`tokenizer.json`, `chat_template.jinja` and picture settings on Hugging
Face. Rendered in **Node**, with `@huggingface/tokenizers` and
`@huggingface/jinja` (about 1 MB together, no Python), they give the
cookbook's **exact** training tokens, prompt tokens and trained-token mask
on **366 of 366** conversations (code, Chinese, Cyrillic, emoji, tabs;
three thinking efforts; 0.6 seconds). The only difference from the
cookbook's prompt is by design: its prompt stops before
`<|message_model|>`, which the model writes itself.

**Inkling 2 Small is the exception, for now.** Its tokenizer (h200k)
is not published. Its vocabulary can be read back from `tml-renderers`,
but its merge rule cannot: the best rebuild tokenizes 93 of 2,334 texts
differently (runs of punctuation). Not exact, so not usable. Thinking
Machines said on 5 November that the weights come out "next week" under
Apache 2.0, which publishes these files; the first generation shows
they reproduce the renderer exactly. Until then:

- Python renders Inkling 2 with `tml-renderers` on CPU-only PyTorch
  (`functai[tinker-inkling2]`, 0.84 GB), the only place PyTorch remains;
- TypeScript trains on every other Tinker model (Qwen3.5 4B/9B,
  Qwen3.6-35B-A3B, Inkling-Small, and the rest with public files), and
  on Inkling 2 once the files are out. A bridge (TypeScript asking a
  small Python helper for Inkling 2's tokens) would be throwaway work
  for about a week.

**Talking to Tinker from TypeScript.** There is no JavaScript SDK (none
on npm, none in the thinking-machines-lab organisation). The Python SDK
is open source (Apache 2.0) and generated from an OpenAPI description
that is not public. The protocol is plain HTTPS:

- about 15 endpoints in use (`create_session`, `session_heartbeat` every
  10 seconds, `create_model`, `forward_backward`, `optim_step`,
  `asample`, `save_weights`, `save_weights_for_sampler`, `load_weights`,
  `retrieve_future(s)`, checkpoints, billing), JSON both ways;
- work is asynchronous: a request returns a future, collected by
  polling; "not yet" is HTTP 408 with the queue's state;
- `forward_backward` is sent as protobuf (zstd-compressed) by the Python
  SDK, a speed choice. The `.proto` is recoverable from the SDK (23
  messages). The server read a JSON `forward_backward` too (a probe with
  no session answered "Invalid session", past the format check): very
  likely accepted, to confirm with one real call (a fraction of a cent);
- the server sends feature flags (`client/config`); none forces an SDK
  version today.

So FunctAI TypeScript gets its own small Tinker client (about 700
lines). The cost is ours to carry: the protocol is not documented and
can change under flags. The mitigation is three things: recorded
requests and replies of the Python SDK as test fixtures, a live smoke
test that runs one cent of work, and asking Thinking Machines for a
TypeScript SDK or the OpenAPI file (their generator, Stainless, writes
TypeScript SDKs).

**One protocol detail worth using.** A training example can point at
tokens the server already sampled (`sampled_tokens` provenance) instead
of sending them again. For reinforcement learning on screenshots, this
avoids uploading every picture twice.

## The design in one picture

```
            what is learned (method)                    where (trainer)
  rows ─┐   sft       imitate known answers           here   TRL (sft today; dpo, grpo, gkd later)
ratings ┼─► rl        a reward from a criterion   ──► tinker its API; FunctAI's loops (TypeScript, Python)
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

**Rendering.** A student's tokens come from its published files, as
Python's bake already does (`bake/template.py`): the tokenizer, the chat
template and its keyword arguments, now with a **picture rule** (how
many tokens a picture takes, and what replaces the template's picture
placeholder). `Template` records the hashes of the tokenizer and the
template, the keyword arguments (`{"reasoning_effort": 0.0}` for
Inkling, `{"enable_thinking": false}` for Qwen) and the picture rule's
name (`inkling-40px`, `qwen-smart-resize`). Python renders with
`tokenizers` and `jinja2`, TypeScript with `@huggingface/tokenizers` and
`@huggingface/jinja`. A contract case set (`contract/cases/tokens/`:
messages and pictures → tokens, for each model family's files) holds
the two languages to the same tokens. The cookbook's renderers are not
the reference. The model's own published template is, and for Inkling
they agree exactly. For Qwen they differ in two known details (a
line break, an empty thinking block), which matters only if training and
calling used different ones, and FunctAI does both. The one exception
is Inkling 2 until its files are published (section 0).

The cookbook's prompt for Inkling stops before `<|message_model|>`; the
published template adds it. FunctAI follows the template on both sides.

**Pictures are sent as they are.** PNG stays PNG. The cookbook
re-encodes every picture as JPEG, which blurs text and thin lines,
exactly what screenshots are made of.

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
refuses the request. FunctAI computes the count itself
(`ceil(w/40) × ceil(h/40)`), so it never sends the wrong number and
needs no padding; a contract case pins it. Python's fallback through
`tml-renderers` (Inkling 2, until its files are out) pads with one black
column on the right instead (no coordinate moves). Report it upstream.

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

**The loop is FunctAI's, in the contract.** The first draft borrowed
the cookbook's loop. That made PyTorch a dependency and left TypeScript
without one, and the parts that looked subtle turn out to be Tinker's
job, not the client's: the off-policy correction is the server's loss
(`importance_sampling`, `ppo`, `cispo`, `stable_reinforce` take the
sampler's log-probabilities), and a KL penalty is a reference model's
log-probabilities (`compute_logprobs`) folded into the reward. What is
left is short and written down once, in `contract/training.md`, with
cases every language reproduces:

1. a step's cases: `group` answers each, from a sampler of the current
   weights (`num_samples`, so the prompt is billed once);
2. each answer parsed with the function's layout into typed outputs (an
   unreadable reply scores the worst value, as a failed call does in
   design 20), then rewarded;
3. advantages: reward minus the group's mean (divided by its standard
   deviation when `normalize=True`, as GRPO does); groups where every
   answer scored the same are dropped, since they teach nothing;
4. one `forward_backward` with the chosen loss, the next step's
   requests sent before this one is awaited (Tinker's clock cycles);
5. section 2's checks every `eval_every`.

The cases fix numbers, not code: given these rewards, these advantages;
given these tokens and advantages, this request. Python's and
TypeScript's loops are each about 250 lines.

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
`(data, logprobs) -> (loss, gradient with respect to logprobs, metrics)`,
as the way out for research. FunctAI then asks Tinker for the
log-probabilities and sends the gradient back as per-token weights,
which is what the SDK's `forward_backward_custom` does after running
PyTorch's autograd. DPO's and the Brier score's gradients are written
out in the contract, so neither language needs autograd.

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
| Python, TypeScript | native sampler, each language's own client | yes |
| R, Julia | `functai serve` the function in Python or TypeScript and call it (serving exists) | yes |
| any tool | Tinker's OpenAI-compatible endpoint (beta; Tinker says testing, with a waitlist for production) | the endpoint renders with the model's default template, which for Inkling is the published one (section 0); text only is proven |
| here, after download | `save_weights_external` and its signed URLs (plain HTTPS: TypeScript can fetch them), then `baked.on("vllm")` | the same check as `adopt` (design 12) |

A student that cannot leave Tinker (Inkling today, and anything too big
to run anywhere else) is lock-in, and FunctAI's principle is no lock-in.
The plan says so in its last line. The way out it offers is `distill` to
an open model that runs here.

## 7. Order of work

Each stage ships alone and is accepted by **reproducing the agent's
measured results with FunctAI**: same data, same test, and scores within
the agent's sampling ranges.

TypeScript comes first: it is what Quiver runs. Python follows each
stage with the same contract cases (it has the official SDK, so it
needs no client of its own).

| Stage | What | Accepted when | Live cost |
|---|---|---|---|
| A0 | the contract: picture parts in the examples, `contract/cases/tokens/` (Inkling-Small, Qwen3.5, pictures), `contract/training.md` with its cases; TypeScript's Tinker client, tested on recorded Python SDK traffic and by one live call (JSON `forward_backward` confirmed or not) | both languages write the same tokens for every case; the live call trains one step | < $0.10 |
| A | supervised training on Tinker from TypeScript: rendering from published files, picture rules, checks by criteria while training, best checkpoint, temperature chosen, state always kept, storage and billing, `held_out_by`; Python's trainer moved to the same rendering | `scanpath` (5 passes) on Qwen3.5-4B and on Inkling 2 Small once its files are out, ScanMatch 0.315–0.334 on the 250 test cases for Inkling 2 (agent: 0.323) | ≈ $15 |
| B | `method="rl"` with FunctAI's loop; rewards as criteria with `references` and `answers`; `rewards.energy`; self-agreement in the report | stage 2 from A scores 0.375–0.385 (agent: 0.380), and the report shows self-agreement collapsing without being asked | ≈ $20 |
| Q | Quiver: weight optimization beside GEPA (section 8) | a Poser AI function improved on Tinker, its draft measured on held-out cases and deployed by the existing rule | Quiver's |
| C | `decision` and `prefer` | a finite-output function with pictures, against `head` and against the untrained model, with ranges | ≈ $5 |
| D | `distill`: on-policy (same tokenizer), off-policy from a Tinker teacher, `teacher="self"` with `fixed=` | Inkling 2 Small → Qwen3.5-4B, run here, against the 4B trained alone | ≈ $15 |
| E | `rl` on modules and agents with tools | a two-step module and a tool-using agent improve on held-out cases | ≈ $20 |
| F | the same methods `where="here"` (TRL: DPO, GRPO, GKD), pictures here | the stage C and D tasks on lambda's GPUs | free |

## 8. In Quiver: weight optimization beside GEPA

Quiver already has the frame (its README, *Improving*): an optimizer
proposes a **draft** version, Quiver measures it, and a person decides
what goes live. GEPA is the one optimizer today, and it rewrites an AI
function's instruction. Quiver also already names **weights** ("a
fine-tuned or custom model a version uses… stored on its own, by
fingerprint; a version only points to it"). Weight optimization is a
second optimizer in that frame, not a new frame:

```ts
// what Quiver's worker runs (FunctAI TypeScript)
const plan = await bake(fn, trainingCases, {
  student: "Qwen/Qwen3.5-4B", where: "tinker", criteria, heldOutBy: "site", planOnly: true,
});                                   // cost, storage, data sent: shown on the page first
const baked = await plan.run();       // an hour or more; metrics stream to the run's page
const rl = await bake(fn, trainingCases, { method: "rl", start: baked, reward: criteria[0] });
```

- **The same data rule as GEPA.** Only training cases are sent; held-out
  cases never are, and they decide the deploy. Cases with a known answer
  train by imitation (`sft`). Cases with only a judge train by reward
  (`rl`, the judge program as the criterion). That is the item Quiver
  lists as "not yet: improving against a judge program", which Poser's
  programs need.
- **The draft** is a board whose `program.json` points at the weights
  (`weights:<sha256>`), and the weights record is `baked.json`: the
  `tinker://` path, the template and picture rule, the chosen
  temperature, the criteria it was trained on. Tinker's price table
  gives the model its price, because Quiver offers no model without one.
  Its setup differs from the parent's, so its answers are measured apart.
- **The deploy rule is unchanged**: the draft must not be worse on
  held-out cases, compared case by case.
- **GEPA, then weights.** GEPA is cheap and fast, so it runs first. Bake
  starts from GEPA's instruction, and with `fixed=` (design 12) the long
  instruction can be left out of the student's prompt entirely, so
  every call is shorter.
- **What Quiver adds**: the Tinker key as a provider key (whose
  account pays), Tinker's host in the worker's reach, a run that
  outlives a worker restart (the run folder and Tinker's saved state let
  it resume), and the plan's lines on the page before anything is spent,
  including "your data is sent to Thinking Machines".
- **Where the weights answer** in production is open question 6.

## Trade-offs taken

- **FunctAI writes its own training loops**, in both languages, written
  down in the contract. We give up the cookbook's recipes and its future
  improvements, and we carry about 250 lines per language and per
  method. We gain: no PyTorch, TypeScript at all, one behaviour in every
  language, and loops that call functions through their layouts. The
  agent's one real bug (the spread weight) is in a reward, which we ship
  with a test either way.
- **TypeScript speaks an undocumented protocol.** Thinking Machines may
  change it; our client then breaks until we follow. Recorded fixtures
  and a one-cent live test catch it the same day; an official SDK or the
  OpenAPI file would remove the risk, and we ask for them.
- **Inkling 2 Small waits for its published files in TypeScript** (said
  to be days away). Python can train it today through `tml-renderers`
  with CPU-only PyTorch (0.84 GB, not 5.7). If the files are late, a
  Python helper that only renders tokens for TypeScript is the stopgap.
- **No PyTorch anywhere else.** `functai[tinker]` is the SDK, `tokenizers`
  and `jinja2` in Python; two packages of about 1 MB in TypeScript. Audio
  (which Inkling reads) waits: its features need tensor maths.
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
4. Does Tinker accept a JSON `forward_backward` all the way (the probe
   got past the format check, not to a trained step)? One call in stage
   A0. If not, TypeScript sends protobuf from the recovered `.proto`
   (one more dependency, `@bufbuild/protobuf`).
5. Ask Thinking Machines: a TypeScript SDK or the OpenAPI file; Inkling
   2 Small's tokenizer files before the weights, if the weights slip;
   the 40-px count bug; and making PyTorch an import only audio needs.
6. Production serving. Tinker calls its OpenAI-compatible endpoint a
   testing service, with a waitlist for production, and the agent
   measured 15 to 20 requests a second. A Poser study at Bigspin's
   volume on a Tinker-only model (Inkling 2) depends on that waitlist;
   open models (Qwen) can be downloaded and served on lambda or the
   Spark instead.
7. The agent's last finding: variety needs many people per screen, not
   a cleverer reward. That is data collection for Bigspin, not FunctAI.
   FunctAI's part is design 20's draws and blind ratings, which already
   exist.

## As built (2026-11-08)

Branch `bake-tinker`. Every test runs against `tools/fake_tinker.py`,
which the official SDK checks (below); the integration test against
Tinker itself is at the end of this section.

| piece | Python | TypeScript |
|---|---|---|
| a student's tokens from its files, pictures, reading replies | `bake/tokens.py` (`tokenizers`, Hugging Face's `render_jinja_template`) | `students.ts` (`@huggingface/tokenizers`, `@huggingface/jinja`, optional peers) |
| the arithmetic (datums, advantages, KL, energy, DPO, decisions, choosing) | `bake/training.py` | `training.ts` |
| the client | the official SDK, `tinker` 0.39 | `tinker.ts`, `protowire.ts` (written from the SDK's protocol; protobuf by hand) |
| plan, run, methods, checks, the model | `bake/service.py` (`bake(..., where="tinker")` and the new methods route there) | `train.ts` (`bake`, `planBake`, `bakedOnTinker`, `rewards`) |
| the contract | `contract/training.md`, `contract/students.json`, `contract/cases/training/` (19 cases) | |

Differences from the proposal above, each for a reason:

- **Checks of `rl` score the reward over a group of answers per case**, not
  one answer per case: at temperature 1 one answer is a coin flip, and a
  lucky one before training became an unbeatable best (found by a test).
  Explicit `criteria` still decide through `evaluate`.
- **The model keeps the training state of the weights it answers with**
  (the best check's), saved when a check becomes the best, so `start`
  continues from exactly them; a run kept the final state at first, which
  was not the model's weights when an earlier check won (found by a test).
- **Samples are seeded** (the bake's seed, the step, the case; a call during
  a check by its request), so a run chooses the same way again.
- **A check never reads the reply cache**, and names its deciding metric
  (rows that know their answers add exact match first, which a first
  version read instead of the criterion).
- **Generation stops at the end of the student's turn**, the last special
  token of what the template writes after a reply (Inkling's
  `<|content_model_end_sampling|>`), and a reply is read after the last
  `lead`: a thinking part before the answer does not cut it. Python's earlier
  rule stopped at the first token after the reply.
- **Python's Tinker trainer** (never run against the service) is gone;
  `trainers/tinker.py` keeps the plan's estimate and `download()`.
- **The heartbeat is never retried** (TypeScript): a retried one kept a
  process alive forever once the service was unreachable (found as a test
  file that would not exit).
- **A run stops when asked** (`signal` in TypeScript, `stop` in Python) at the end of
  the step under way, its training state saved at that step: a host that must stop a
  long run (a server restarting) loses no step. Intermediate states are kept a day: a
  run left stopped longer than that cannot load its state any more, and is baked anew
  (a new folder).
- **Not built**: `rl` on modules and agents with tools (stage E); the new
  methods trained here with TRL (stage F); weight optimization in Quiver
  (section 8: that repository's work, next).

Proven offline, and how:

- **The official SDK runs against the fake**: training learns, greedy
  sampling reproduces, prompt log-probabilities, save and load with the
  optimizer, samplers from paths, an RL step, checkpoints listed, given a
  TTL and deleted, a picture's token count enforced as Tinker enforces it.
- **The contract's 19 training cases** pass in both languages.
- **Real files**: Qwen3.5-4B's and Inkling-Small's give the same tokens in
  Python and TypeScript (56 conversations each, 29 pictures;
  `tools/crosslang_training.py`, in `./check`); Inkling-Small's published
  files give the cookbook's exact training tokens (366 of 366,
  `21-evidence/`); Inkling 2 Small through `tml-renderers` with
  Inkling-Small's template gives Thinking Machines' renderer's prompts on 99
  of 100 conversations, the other being that renderer's picture-count bug.
- **End to end against the fake** (10 tests per language): the plan prices
  and spends nothing; sft learns and calls answer on Tinker; criteria decide
  by name; a failed run resumes from its saved state; `rl` takes a rewarded
  answer from 4% to 83% of samples; the energy score runs; DPO moves the
  student to the chosen answer; a decision model answers the likeliest
  choice with probabilities summing to 1; distillation refuses a teacher
  with another tokenizer; a picture function trains and is called.

### The integration test (2026-11-08, Maxime's own account)

`ts/tools/tinker-live.ts` and `python/tests/tinker_live.py` against the
real service, Qwen/Qwen3.5-4B: the client alone (a training step, a
sample), then `sft`, `rl` from it, and `decision`, each called through
`fn.using`, then every checkpoint deleted. Both pass. A few cents in all;
Tinker's billing records were not in hours later, so the exact amount is
not known yet.

What the live service showed that the fake did not, each now fixed and
made part of the fake, so the offline tests hold it:

- **Sampling needs `X-Tinker-Sampling-Backpressure: 1`**: without it every
  sample is refused, "Tinker SDK is outdated" (400). Asking for it is a
  promise: a busy server answers 429 and the client waits (1 s, 5 s for a
  prompt over 128 KB, one backoff for every sampler) and sends again, with
  no limit, as the SDK does. The TypeScript client had given up after six
  tries. The fake refuses a sample without the header and can answer busy
  (`busy`).
- **A sparse request is answered sparse**: `target_prompt_logprobs` asked
  in CSR (decisions) comes back in CSR. TypeScript refused it; Python's SDK
  makes it dense only through PyTorch, so a decision model failed in
  Python without PyTorch too. Both now read CSR themselves (NaN where no
  value is listed); the Python Tinker tests run as if PyTorch were absent.
- **A check whose calls failed scored those cases 0** (an error in a call
  counts 0 in `evaluate`) and training went on, blind: a decision run
  checked 0.000 at every step. A call that fails in Tinker or on the way
  now stops the bake with its error (the run resumes); a reply the
  function cannot read still counts 0, as before.
- **The bill**: `spent.billed` summed the whole account's usage in the
  run's hours, and wrote 0 (and, in Python, the text "None") before
  Tinker's records came in. It is now the run's own sessions (kept in
  `run.json`, one per process), `null` until records come, and
  `billed_through` the end of the last hour billed. Tinker's
  `cost_data_through` is not that: it separates invoice prices from list
  prices.
- **Records named no model** for a model trained on Tinker (TypeScript
  named no baked model at all), and every Tinker model's name would have
  been the same (`hashes` is empty when the weights are not in the
  folder). calls.md now names weights on a service by their URI too; four
  languages.

Not proven yet:

- Learning on real models: the live tasks (capitals, a two-way mood) are
  ones Qwen3.5-4B already gets right, so the best check was the untrained
  step 0 in every run. The test proves the plumbing, not that training
  improves a model; that takes a task the base model fails (cents to a few
  dollars).
- That Tinker's server computes Qwen's picture token count by the rule the
  cookbook uses (we follow the cookbook; Inkling's is measured). No picture
  was sent live.
- Inkling models live (Inkling 2 Small from TypeScript waits for its files).
- The billed amount read back: the code is tested against the fake's
  records, the real ones had not come in.
