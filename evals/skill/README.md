# FunctAI for coding agents: does the skill help, and how does it compare with DSPy?

`python/functai/skill/` (shipped in the package) teaches coding agents to use
FunctAI. This folder measures two things on the same five tasks, in fresh
isolated workspaces, graded by code:

- **the skill**: the same agent with and without it (`--arms functai-skill functai`);
- **FunctAI vs DSPy**: (A) an agent doing each task with FunctAI and its skill,
  or with DSPy (which models know from training); (B) a hand-written expert
  solution in each library (`references.py`).

| task | what the agent must do | the trap it comes from |
|---|---|---|
| `forms` | fill a lab codebook's nested form for 10 study descriptions | definitions written as comments that never reach the model (StatLM) |
| `measure` | score a classifier with its 95% range, then judge two new rules with a paired interval | numbers without ranges; unpaired comparisons |
| `export` | write the exact training conversations for a small model, leaving a constant style guide out | rebuilding prompts from functai internals (Scholar's Reading List) |
| `photos` | name shapes and colours in 12 pictures, sent as real images | finding how to send an image (species pilot) |
| `files` | answer 5 questions from a PDF report and a spreadsheet; two answers are only in a chart's picture | DSPy reads images only; FunctAI reads any file through Attachments (design/13) |

What each check means and why: `rubric.md`.

## Setup (once)

```bash
cd evals/skill
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install -e "../../python[data]" pandas pillow pydantic scipy
uv venv --python 3.13 .venv-dspy
VIRTUAL_ENV=.venv-dspy uv pip install "dspy==3.4.0" pandas pillow pydantic scipy
nix build nixpkgs#mitmproxy --out-link .mitm                 # the recording proxy
mkdir -p .mitm-conf && timeout 5 .mitm/bin/mitmdump --set confdir=$PWD/.mitm-conf -p 18999   # makes its CA
cat .mitm-conf/mitmproxy-ca-cert.pem /etc/ssl/certs/ca-certificates.crt > .mitm-conf/bundle.pem
for v in .venv .venv-dspy; do cp sitecustomize.py $($v/bin/python -c "import site;print(site.getsitepackages()[0])")/; done
```

**The recording proxy.** Every request a task's code sends to a provider goes
through mitmproxy with `recorder.py`, whatever the library: the
`sitecustomize.py` hook routes an environment's Python through it when
`$BENCH_PROXY` is set (the agent itself is not routed). The graders read only
this record, so "what the model was sent", "a re-run sends nothing" and "the
picture went as a picture" are judged the same way for FunctAI and DSPy.

Provider keys are read from `~/Projects/lm15-dev/.env` (or
`$FUNCTAI_SKILL_EVAL_KEYS`); the tasks use `gpt-4.1-mini` (a few cents per
run). The agents run in Pi; the default agent is `openai-codex/gpt-6-astra`
through the ChatGPT login. (Pi's Claude login refuses headless runs as a
"third-party app" billed to extra usage, so it is not the default.)

## Run

```bash
.venv/bin/python run.py --reps 3 --parallel 6                         # FunctAI + skill vs DSPy
.venv/bin/python run.py --arms functai-skill functai --reps 3        # does the skill help?
.venv/bin/python run.py --tasks forms --arms dspy --model openai-codex/gpt-6-luna
.venv/bin/python report.py runs/<stamp>         # the table again
.venv/bin/python run.py --collect runs/<stamp>  # grade every run again from its workspace (after fixing a grader)
```

Results go to `runs/<stamp>/` (ignored by git): `report.md`, and per run
`result.json` (grade, time, tokens, cost, tool calls, the task's model calls)
and the agent's whole `session.jsonl`. Workspaces stay in
`~/.cache/functai-skill-eval/runs/<stamp>/` for inspection.

## Expert solutions (benchmark B, and the graders' own check)

Each task has a hand-written solution per library: `tasks/<task>/reference/`
(FunctAI) and `tasks/<task>/reference.dspy/` (DSPy, written in DSPy's own
idiom: typed Signatures, `Predict.batch`, `dspy.Evaluate`, `dspy.Image`,
`ChatAdapter.format_finetune_data`). Every one must score 1.0, and
`references.py` compares them: correctness, requests and tokens sent, time,
lines of code.

```bash
.venv/bin/python references.py                                  # both libraries, 3 rounds
.venv/bin/python references.py --tasks forms --libraries dspy --reps 1
```

## Fairness

- Every arm: the same task text (with the library's name), the same agent model
  and thinking level, no extensions, no other skills, no AGENTS.md, no
  Chattering records. FunctAI arms may add `--skill <copy of python/functai/skill>`.
  DSPy has no skill: models already know it from training; FunctAI is new.
- The export task names the call DSPy will use (`dspy.Predict(Simplify.delete("style_guide"))`)
  because DSPy has no built-in way to leave a constant input out of a trained
  model's prompt; FunctAI's task relies on its `fixed=`. Each is graded against
  its library's own training format.
- Both environments hold pandas, pillow, pydantic and scipy; a run whose agent
  installs packages is flagged.
- Each run has its own caches (`XDG_CACHE_HOME`, `DSPY_CACHEDIR`) and its own
  recording proxy, so no run reuses another's replies.
- Workspaces live outside this repository; agents are told not to open
  `functai/evals` or `functai/skills`, and a run that does is marked
  contaminated. Both may read the rest of functai (its docs and source), as
  agents in real projects do.

## Adding a task

`tasks/<name>/task.md` (the prompt), `files/` (the workspace), `grade.py` with
`grade(work, env, python, out) -> dict` (use `common.summarize`), and a
`reference/` solution that scores 1.0. Base it on a mistake seen in a real
transcript, and make the success criteria checkable by code.

## Results so far (GPT-6 Astra, thinking medium, 3 runs per cell)

| round | skill | with: succeeded | without: succeeded | with vs without: list-price cost | tool calls | time |
|---|---|---|---|---|---|---|
| 1 (`runs/20261006T140317Z`) | first version, 2,300 words | 12/12 | 10/12 | −13% | −23% | −21% |
| 2 (`runs/20261006T142219Z`, measure only) | + "change the function, not the data" | 3/3 | 3/3 | +34% | +13% | +33% |
| 3 (`runs/20261006T142622Z`) | lean core (1,000 words) + side files | 12/12 | 11/12 | **−31%** | −11% | −5% |

By task in round 3 (medians, with vs without): export $0.16 vs $0.36; photos
$0.21 vs $0.40 and 3/3 vs 2/3 right; forms and measure about even (measure
+$0.03 and 3 more tool calls: the reading cost of the skill on a task the
agent already handles).

Limits: three runs per cell shows direction, not precise effects; one agent
model (Pi's Claude login refuses headless runs); small tasks of a few minutes,
so the rules about subscriptions' rate limits, safety filters and long runs
come from the transcripts and are not exercised here; agents without the
skill could read FunctAI's docs and source, as in real projects. Findings
about FunctAI itself: `findings.md`.

## FunctAI vs DSPy (2026-10-06; FunctAI 1.2.0, DSPy 3.4.0, gpt-4.1-mini, 3 runs per cell)

**A. An agent doing the task** (GPT-6 Astra; FunctAI with its skill, DSPy without
one; `runs/20261006T145753Z`):

| task | succeeded (FunctAI / DSPy) | agent cost | agent tool calls | agent time | tokens the finished program sent |
|---|---|---|---|---|---|
| export | 3/3 · 3/3 | $0.16 · $0.11 | 9 · 6 | 0.7 · 0.6 min | 0 · 0 |
| forms | 3/3 · 3/3 | $0.30 · $0.28 | 13 · 13 | 3.3 · 1.9 min | 22,348 · 24,486 |
| measure | 3/3 · 3/3 | $0.40 · $0.32 | 11 · 8 | 1.8 · 2.6 min | 16,096 · 47,456 |
| photos | 3/3 · 3/3 | $0.18 · $0.11 | 9 · 4 | 1.3 · 1.0 min | 3,480 · 4,920 |
| **all 12** | **12/12 · 12/12** | **$3.27 · $2.31** | **126 · 100** | **21 · 22 min** | |

**B. Each library used well** (hand-written solutions, `runs/references-20261006T145755Z`): all
24 runs correct in both libraries.

| task | input tokens sent (FunctAI / DSPy) | output tokens | seconds | lines of code |
|---|---|---|---|---|
| export | 0 · 0 | 0 · 0 | 0.4 · 1.3 | 7 · 10 |
| forms | 7,683 · 8,773 | 487 · 746 | 4.5 · 5.9 | 42 · 49 |
| measure | 14,096 · 36,336 | 1,440 · 1,923 | 27.7 · 19.5 | 27 · 41 |
| photos | 2,124 · 4,488 | 204 · 228 | 4.7 · 4.8 | 27 · 15 |

Reading it: both libraries let an agent get every task right. Writing the
code was cheaper with DSPy (−29% agent cost, −21% tool calls): models know it
from training, while a FunctAI agent first reads the skill. Running the code
was cheaper with FunctAI: its prompts are shorter (−61% input tokens on the
short-input task, −53% on pictures, −12% on long inputs), which grows with the
number of rows. DSPy has a native image type (15 lines against 27 with
FunctAI's `lmcc.format` registration) and its evaluation ran faster here.
Limits: one agent model, three runs per cell, small tasks, and both expert
solutions written by the same author (FunctAI's side).

## FunctAI vs DSPy, second round (2026-10-08; FunctAI unreleased 1.3 on lmcc 0.8.6 with Attachments 1.0.0a2, DSPy 3.4.0)

What changed since the first round: requests to OpenAI say `store: false`;
pictures are `lm15.types.ImagePart` inputs; files are Attachments inputs; the
skill teaches both. A fifth task, `files` (a PDF whose chart holds two answers,
and a spreadsheet). Both environments have the same document tools.

**A. An agent doing the task** (GPT-6 Astra, thinking medium, 3 runs per cell;
`runs/20261008T120740Z`):

| task | succeeded (FunctAI / DSPy) | agent cost | agent tool calls | agent time | tokens the finished program sent |
|---|---|---|---|---|---|
| export | 3/3 · 3/3 | $0.18 · $0.15 | 10 · 7 | 0.7 · 0.7 min | 0 · 0 |
| files | 3/3 · 3/3 | $0.24 · $0.14 | 8 · 6 | 1.0 · 1.1 min | 22,567 · 44,712 |
| forms | 3/3 · 3/3 | $0.36 · $0.29 | 18 · 16 | 2.1 · 2.2 min | 27,731 · 42,030 |
| measure | 3/3 · 3/3 | $0.35 · $0.25 | 12 · 9 | 1.6 · 2.1 min | 16,096 · 51,939 |
| photos | 3/3 · 3/3 | $0.17 · $0.10 | 9 · 3 | 0.9 · 0.7 min | 3,384 · 4,956 |
| **all 15** | **15/15 · 15/15** | **$3.98 · $2.70** | **168 · 124** | **20 · 21 min** | |

**B. Each library used well** (hand-written solutions, 3 rounds, `runs/references-20261008T122232Z`):
all 30 runs correct.

| task | input tokens sent (FunctAI / DSPy) | seconds | lines of code |
|---|---|---|---|
| export | 0 · 0 | 0.9 · 1.7 | 7 · 10 |
| files | 22,148 · 32,640 | 6.0 · 7.7 | 14 · 28 |
| forms | 7,683 · 8,773 | 4.3 · 10.4 | 42 · 49 |
| measure | 14,096 · 36,336 | 17.9 · 18.9 | 27 · 41 |
| photos | 2,124 · 4,488 | 3.9 · 6.0 | 16 · 15 |

Reading it: both libraries let an agent get every task right, the new one
included (DSPy's agents drew the PDF pages with PyMuPDF; FunctAI's used
Attachments with `[images: true]`). Programs written with FunctAI send fewer
tokens on every task (about half on `files` and `measure`) and now run faster
than DSPy's everywhere, the evaluation included (it was 28 s against 20 s
before `store: false`). Writing them still costs an agent more (+47% list
price, +35% tool calls): it reads FunctAI's skill, and models know DSPy from
training. Limits as before: one agent model, three runs per cell, small
tasks; the machine was loaded during these runs (both arms equally); the
expert solutions are written by FunctAI's side; DSPy's `files` solution draws
pages at 150 dpi and Attachments at its default (2000 px at most), so their
picture tokens differ by choice, not by library.
