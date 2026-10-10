# 14 — Bake in a real project: what StatLM's student needed

*Written 2026-10-10 from the StatLM conversation (Chattering `01a0fe70`,
2026-10-02 to 2026-10-10, messages #1503–#1885) and its follow-up
(`01a1257f`). The goal of bake (design 12) is one line in the common case.
This is what stood between that line and a trained student when a real
project used it, what was fixed, and what is still open.*

## The job

Four AI functions that read a scientific paper into linked records
(`read_setup`, `read_findings`, `read_claims`, `read_table`), a long fixed
guide (`fixed={"guide": GUIDE}`), 922 papers and 2,048 tables answered by
a two-model teacher: 4,814 examples, 64M tokens, examples up to 87,119
tokens, answers up to 53,486. One Qwen3.5-4B student, LoRA. lambda's two
RTX 3090s were busy or too small, so it trained on a rented RTX PRO 6000
(96 GB, Nebius, about $1.80 an hour): 175 steps, 9.3 hours, about $20.

What the project wrote to get there, besides its data:
`bake_plan.py`, `build_bake_data.py`, `rent/up.sh`, `rent/down.sh`,
`rent/train_student.py`, `rent/test_remote.sh`, `watch.py`,
`eval_student.py`, and a dashboard. Only the first two should have been
needed.

## What went wrong, and where it stands

| # | What happened | Cause | Status |
|---|---|---|---|
| 1 | The plan refused to train: the longest example "needs 26 GB" (it was 26k tokens, counted as 38k) | longest prompt + longest answer, from different rows | fixed (`ddc154f`) |
| 2 | `bake(examples)` crashed: `'Plan' object has no attribute 'to_label'` | an attribute set on one path only | fixed (`7e7b2f6`) |
| 3 | Examples saved without `student=` were refused; the rented GPU waited while they were rebuilt | tokens required up front | fixed: bake tokenizes them |
| 4 | Examples tokenized for one student were planned against the default list | the student was not taken from the examples | fixed |
| 5 | The plan said "4,718 train, 0 validation" (96 existed) | the count was not filled for Examples | fixed |
| 6 | Time: plan 3.7 days, real 9.3 hours | "RTX PRO 6000 Blackwell" matched nothing in the speed table: 30 TFLOPS assumed | fixed: the card is known; unknown cards are named as a guess; runs measure their speed |
| 7 | Memory: plan "fits (58.9 / 95 GB)" in 16-bit; out of memory at once | the formula is about 1.7× low for Qwen3.5 (linear-attention layers) | fixed: margin from this run; runs measure their memory |
| 8 | The run switched to 4-bit weights on its own; only `log.txt` said so | `_shrink` set a phase, shown only before the first step | fixed: `NOTE:` while waiting, `run.notices`, the model's record and repr |
| 9 | Every plan listed Tinker's and Prime's missing keys, even with `where="here"` | alternatives always shown | fixed: with `where=`, places that cannot train it are left out |
| 10 | `functai[bake]` did not bring the linear-attention kernels; `causal-conv1d` had to be compiled by hand | extras | partly: the plan names `causal-conv1d` and how to install it. Open: `functai[bake]` should bring `flash-linear-attention` on Linux (the default students need it); a lock change, left for a separate commit |
| 11 | Install on the rented machine failed: lmcc not at `../../lmcc/python` | the development checkout's `[tool.uv.sources]` | open, developer setup only (PyPI installs are not affected) |
| 12 | Requests to the student's vLLM timed out after 10 minutes | a fixed 600 s client timeout; a 50k-token reply at 60 tokens a second takes 14 | fixed: at least as long as 10 tokens a second needs |
| 13 | vLLM failed to start with a long trace on a GPU another job filled | no check before starting | fixed: says what is free and what is needed |
| 14 | Each checkpoint turned into a model is a full merged copy (8 GB); ten would fill the disk | `checkpoint_model` merges | open (below) |
| 15 | Small-model replies ran to the length limit (one function's runaway limited by the longest answer of *any* function: 66,858 tokens) | one `generation.max_new_tokens` for all functions | open (below) |
| 16 | The student invented values outside the schema (`"LSD_0.05"`, `"p_value"`) and wrote broken JSON | free generation | open (below) |
| 17 | Bake could not rent, watch, copy back or delete a remote machine | not bake's job yet | open (below) |
| 18 | The loss fell smoothly while the student wrote 0 readable records out of 10 (step 18) | loss is not the goal; judging only at the end | open (below) |
| 19 | The plan said about 148 steps; the run made 175 | the plan counts examples per step; the run groups by length | open, small |

Not bake's: the agent stopped a family service it took for its own
leftover, moved the checkpoint tests to the end of training without
asking, and its own clean-up did not restart the service it had paused.

## Measured numbers kept in the code

- Qwen3.5-4B, LoRA r64, 87,119-token batches, 95 GB GPU: 16-bit weights ran
  out of memory with 92.7 GiB allocated and 0.83 GiB more asked (at least
  0.96 MiB a token); 4-bit weights fit. The hybrid margin (2.45) puts the
  plan's decision where the run's was (any margin from 2.37 to 2.55 does).
  One run of one student: a run on another hybrid student will correct it
  through `measured.json`, which is the point of measuring.
- Speed: 1,730 tokens a second on the RTX PRO 6000 with 4-bit weights and
  examples of 13.6k tokens on average, about 15% of the card's peak. The
  formula now says about 6.5 hours for that run (efficiency 0.3, 4-bit
  ×0.75): still optimistic by about 1.4×, until the first run measures.

## Open: proposals

**Training on a machine you rent (17).** A trainer `where="ssh://user@host"`:
copy the examples and the plan, install `functai[bake]` there (the same
version), start `python -m functai.bake supervise` in its own session,
bring back metrics and checkpoints (adapters only) as they appear, and
the final model at the end, checked by hash. Renting and deleting the
machine stay outside (every cloud differs), with a hook when the run ends
(`on_done=`) so a script can delete it. This replaces `rent/test_remote.sh`
and most of `watch.py`.

**Judging checkpoints while training (18).** `bake(..., test=rows,
metric=..., judge_every="checkpoint", judge_on="cuda:1" | url)`: each new
checkpoint is served (where training is not) and judged with the same
metric as the final report, and the curve is in `run.metrics()` beside
the loss. The training GPU is full, so judging needs another device or an
endpoint; the plan should say which.

**A reply limit per function (15).** Each function's entry in `baked.json`
gets its own `max_new_tokens` (1.25× its longest answer); a call uses its
function's. This changes the contract (`baked.md`), and every language's
router must know which function a request is for, which none does today
(they read `generation.max_new_tokens`). It must not become the request's
`max_tokens`: the contract re-sends a cut-off reply with twice the budget
when the request set one (`functions.md`), and a student that runs past
1.25× its longest answer is repeating itself, so more room only doubles
the time.

**Checkpoints as adapters (14).** Serve a checkpoint's adapter on the base
weights instead of merging (vLLM `--enable-lora`; LoRA on Qwen3.5's
linear-attention layers in vLLM is unverified), or keep one merged
checkpoint at a time.

**Replies that cannot leave the schema (16).** vLLM's structured outputs
can hold a reply to a grammar. The grammar is the layout's (tags around a
JSON value of the output's schema); lmcc would have to write it.

**Leaving out what does not fit (from the 3090 plan).** On a 24 GB card,
509 of 922 `read_findings` examples were too long, and the project
filtered them by hand from token counts it computed itself. A
`max_example_tokens=` option, with the plan saying how many rows of each
function it leaves out, would make that one argument.
