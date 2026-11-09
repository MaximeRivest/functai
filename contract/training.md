# Training on a service

What every language must agree on when it trains a student on a service
that takes tokens (Tinker) and calls it there: **the student's tokens**
(made from the model's published files, pictures included), **the training
examples** sent for each method, the **arithmetic** of the training loops,
and the folder's new keys. Weights need not match between languages; the
numbers below must, to 1e-9 for arithmetic and exactly for tokens.

[baked.md](baked.md) says what a student learns (the conversations); this
document says how those conversations become what a service trains on.
Design 21 explains why; `cases/training/` holds the cases, written by
`cases/training.py` from these rules.

## 1. A student's files

A student is named by the service's id (Tinker's: `Qwen/Qwen3.5-4B`,
`thinkingmachines/Inkling-2-Small:peft:262144`). Its files are the
Hugging Face repository of the same name up to the first `:`:

- `tokenizer.json` (with `tokenizer_config.json`): how text becomes tokens;
- the chat template: `chat_template.jinja`, else `tokenizer_config.json`'s
  `chat_template`;
- the picture configuration of its family ([students.json](students.json)):
  `processor_config.json` (Inkling) or `preprocessor_config.json` (Qwen).

A student whose files are not published cannot be rendered from them; an
implementation says so before anything is spent (Python may render
Inkling 2 Small through `tml-renderers` until its files are out: design
21, section 0).

**The template's keyword arguments.** `{"enable_thinking": false}` when
the template's text contains `enable_thinking`; `{"reasoning_effort": E}`
when it contains `reasoning_effort` (E = 0.0 unless the bake sets
`effort`). Both may apply. A student answers directly: a reasoning
output, when the bake keeps it, is part of the reply it learns.

## 2. Conversations with pictures

The chat messages of a call ([baked.md](baked.md), *The examples*) keep
pictures where the layout puts them. A message whose parts are all text
has its text as `content` (as before). A message with a picture has a
list: adjacent text parts joined into one `{"type": "text", "text"}`,
each picture `{"type": "image"}`, in order. Pictures are numbered from 0
in the order they appear in the conversation; their bytes travel beside
the messages.

In the examples table, a picture part is `{"type": "image", "sha256":
"sha256:<hex of the bytes>", "media_type"}` and the bytes are stored once
per hash (`pictures/<hex>` beside the table). Only PNG and JPEG are
trained on a service (Tinker takes nothing else): another picture format
is refused with the advice to convert it.

**A picture's size** is read from its bytes: a PNG's `IHDR` (width, then
height, 32-bit big-endian at bytes 16 to 24); a JPEG's first
start-of-frame segment (markers `C0`–`CF` except `C4`, `C8` and `CC`:
height, then width, 16-bit big-endian, 3 bytes after the segment's
length). Anything else is not a picture a service takes.

## 3. Tokens

For messages `M`, a reply `R` (absent for a call), the template `T` with
its keyword arguments `K` and the tokenizer:

1. `prompt = encode(T(M, add_generation_prompt = true, K))`;
   `full = encode(T(M + [{"role": "assistant", "content": R}],
   add_generation_prompt = false, K))`. `encode` is the tokenizer on the
   whole text with no added special tokens (special tokens written by the
   template are read as themselves).
2. When `full` does not begin with `prompt` (a template that writes the
   last reply differently from a generation prompt),
   `full = prompt + encode(R + end)` (`end`: section 5).
3. **Pictures.** Each occurrence of the family's picture placeholder
   token (one token) in `prompt` and `full` is, in order, one picture: it
   is replaced by that picture's chunk, which stands for `n` positions
   (section 4). The number of placeholders must equal the number of
   pictures, else the conversation is refused.
4. The result is a list of **chunks**: `{"text": [token ids]}` and
   `{"picture": k, "tokens": n}`, adjacent text merged. A chunk list's
   **length** counts a picture as its `n` positions. The **reply starts**
   at the length of the prompt's chunks.

## 4. Picture tokens

By the family's rule ([students.json](students.json)), from the picture's
width `w` and height `h`:

- **grid** (Inkling; `P` = `image_processor.size.width` of
  `processor_config.json`, 40): `ceil(w / P) × ceil(h / P)`. Pictures are
  not resized.
- **qwen** (Qwen3.5 and 3.6; from `preprocessor_config.json`: `p` =
  `patch_size`, `m` = `merge_size`, `min` = `size.shortest_edge`, `max` =
  `size.longest_edge`; `f = p·m`). Refused when `max(w, h) / min(w, h) >
  200`. `H = round(h / f)·f`, `W = round(w / f)·f`, **round half to even**.
  When `H·W > max`: `b = sqrt(h·w / max)`, `H = max(f, floor(h / b / f)·f)`,
  `W = max(f, floor(w / b / f)·f)`. Else when `H·W < min`:
  `b = sqrt(min / (h·w))`, `H = ceil(h·b / f)·f`, `W = ceil(w·b / f)·f`.
  Tokens: `(H / f)·(W / f)`.

## 5. Where a reply starts and ends

From the template alone, with `S` = `"\u2063R\u2063"` and the probe
`[{"role": "user", "content": "x"}]`:
`P = T(probe, add_generation_prompt = true, K)`,
`F = T(probe + [assistant S], add_generation_prompt = false, K)`.

- `lead` is what `F` writes between `P` and `S` (`""` when `F` does not
  begin with `P`): Inkling's `<|content_text|>`, Qwen's nothing.
- `end` is what `F` writes after `S`: Inkling's
  `<|end_message|><|content_model_end_sampling|>`, Qwen's `<|im_end|>\n`.
- **The stop token** is the last special token in `encode(end)`; when
  there is none, the tokenizer's end-of-sequence token. Inkling stops on
  `<|content_model_end_sampling|>`, which ends the whole turn (a thinking
  part before the text also ends with `<|end_message|>`); Qwen on
  `<|im_end|>`.
- **The end's head** is `end` up to that stop token's text (Inkling's
  `<|end_message|>`, Qwen's `""`).

**Reading a reply.** Decode the generated tokens, the stop token left
out, keeping special tokens. When `lead` is not empty and occurs, keep
what follows its last occurrence. When the end's head is not empty and
occurs in what is kept, cut at its first occurrence. What remains is the
reply text the layout reads.

## 6. Training examples (Tinker's "datums")

A datum is `model_input` (chunks) and `loss_fn_inputs` (lists, one value
per position of `model_input`). From chunks of length `N` (the last chunk
text; trailing pictures dropped first):

- `model_input`: the chunks without their last token (an emptied text
  chunk dropped), length `N - 1`;
- `target_tokens[i]`: the token at position `i + 1`; `0` where that
  position is a picture.

By method:

- **sft**: `weights[i]` = the example's weight when position `i + 1` is
  text at or after the reply start, else `0`.
- **rl** (an answer sampled for a prompt, with the sampler's
  log-probabilities): the chunks are the prompt's then the sampled tokens;
  `logprobs[i]` = the sampler's log-probability of the token at position
  `i + 1` when it was sampled, else `0`; `advantages[i]` = the answer's
  advantage (section 7) on those same positions, else `0`. No `weights`
  (Tinker refuses extra keys on RL losses).
- **weighted** (any loss written as per-token weights: DPO, decisions;
  section 9): as sft, with a weight per position given by the method;
  with `K` targets per position (decisions), `target_tokens` and
  `weights` are `N - 1` rows of `K`, every row `0` but the scored one.

## 7. Groups and advantages

A step asks `G` answers for each of its prompts (one request with
`num_samples = G`), and rewards each.

- An answer that cannot be read (the layout refuses the reply, or the
  sampler stopped at its length limit) gets the reward's worst value: 0
  for a criterion (its values lie in [0, 1], design 20), or the group
  reward's own worst.
- **Advantages.** `A_i = r_i − mean(r)`; with `normalize` (GRPO's),
  divided by `std(r) + 1e-6` (the population standard deviation). A
  group whose rewards differ by less than `1e-9` teaches nothing: it is
  left out of the step, and counted.
- **KL to the starting weights** (`kl = β > 0`): for each sampled token,
  `δ` = the sampler's log-probability minus the starting weights'
  (Tinker's `compute_logprobs` on a sampler of the starting weights); the
  token's advantage becomes `A + β·(mean δ − δ)`, the mean over every
  sampled token of the step.

## 8. The energy score

A reward for groups whose answers should be spread the way the right
answers are (Gneiting & Raftery 2007). For answers `x_1…x_n` (the
readable ones, `I`), references `y_1…y_m` and a distance `d` (the
user's):

- `a_i = mean_k d(x_i, y_k)`; `s_i = mean_{j ∈ I, j ≠ i} d(x_i, x_j)`.
- reward `r_i = −a_i + s_i` (the spread term has weight 1: both answers
  of a pair come from the model, so this is the gradient of the expected
  score); with one readable answer, `r_i = −a_i`.
- the score `ES = mean_i a_i − ½·mean_{i ≠ j} d(x_i, x_j)` (lower is
  better; `mean_i a_i` with one readable answer).
- an unreadable answer: reward `−1` (the worst for a distance in
  [0, 1]), left out of the others' spread.

## 9. Losses as weights

Tinker computes the model's log-probability of each target token; a loss
written over those log-probabilities is trained by sending its gradient
back as cross-entropy weights: `weight = −∂L/∂logprob`.

- **DPO** (a chosen and a rejected reply to one prompt; `π` = the summed
  log-probabilities of a reply's trained tokens now, `ρ` = under the
  starting weights, `β` = 0.1 by default):
  `z = β·((π_c − ρ_c) − (π_r − ρ_r))`, `L = log(1 + e^(−z))`. Every
  trained token of the chosen reply gets weight `β·σ(−z) / P`, of the
  rejected one `−β·σ(−z) / P`, `P` the number of pairs in the step.
- **Decisions** (the log-probabilities `ℓ_1…ℓ_K` of the choices' tokens
  at the answer's position; `p = softmax(ℓ)`):
  - a known answer `y`: `L = −log p_y`; `weight_k = δ_ky − p_k`;
  - a target distribution `t` (several people's ratings): the Brier
    score `L = Σ_k (p_k − t_k)²`;
    `weight_k = −2·p_k·((p_k − t_k) − Σ_j p_j·(p_j − t_j))`.

## 10. Choosing what to keep

**Held out.** The cases a run is checked on are given, or held out by
[measures.md](measures.md)'s rule (default share 0.1 for a bake). With a
column that groups rows (`held_out_by`: every row of one screen on one
side), the side is that rule's for a row whose only key is the
conversation id `<column>:<canonical JSON of the row's value>`. Held-out
cases are never trained on.

A run is checked before its first step, every `eval_every` steps and at
its end: each check samples the held-out cases at each of the bake's
`temperatures` (default `[0.0, 0.7, 1.0]`) and scores them with the
deciding criterion: the first of `criteria` (each case answered once, as a
call is), else for `rl` the reward itself, as training sees it (each case
answered by a group of `group` answers, the mean of their rewards; a group
reward's own score, negated when lower is better). Samples are seeded from
the bake's seed, the step and the case (or the request), so a run chooses
the same way again. A check never reads the reply cache. The
baked model is the best (check, temperature) pair: the highest mean, the
earlier check and then the lower temperature on a tie. Its temperature is
the one calls use (`baked.json` `sampling.temperature`) unless a call
sets one.

## 11. The folder

A student trained on a service adds to `baked.json` ([baked.md](baked.md)):

| key | |
|---|---|
| `weights` | `{"form": "remote", "service": "tinker", "uri": <the sampler checkpoint>, "state_uri": <the training state, to continue>, "base", "lora_rank"}` |
| `template` | as before (`sha256` of the template, `kwargs`, `end`, `stop_token_ids`), and `source` (`"files"` or `"tml-renderers"`), `repo`, `tokenizer_sha256`, `lead`, `picture` (the family's rule and its numbers) |
| `sampling` | `{"temperature"}`, chosen by the checks |
| `training` | as before, and `method`, `criteria` (`{name: held-out score}`), `trained_on` (the criteria used as rewards), `spent` (`{"estimated", "billed", "billed_through"}`: the plan's estimate in dollars; what Tinker billed the run's own sessions, `null` until its records come in (hours later); the end of the last hour billed, `YYYY-MM-DDTHH:00:00Z`, or `null`) |

A call on such a model renders its request with the template's rules
above, samples with `sampling.temperature` (unless the call sets one)
and stops on `stop_token_ids`, and reads the reply as section 5 says.
