# Does the functai skill make agents faster, cheaper and right?

`skills/functai/` (at the repository root) teaches coding agents to use
FunctAI. This folder measures whether it helps: the same agent gets the same
four tasks with and without the skill, in fresh isolated workspaces, and code
grades the result.

| task | what the agent must do | the trap it comes from |
|---|---|---|
| `forms` | fill a lab codebook's nested form for 10 study descriptions | definitions written as comments that never reach the model (StatLM) |
| `measure` | score a classifier with its 95% range, then judge two new rules with a paired interval | numbers without ranges; unpaired comparisons |
| `export` | write the exact training conversations for a small model, leaving a constant style guide out | rebuilding prompts from functai internals (Scholar's Reading List) |
| `photos` | name shapes and colours in 12 pictures, sent as real images | finding how to send an image (species pilot) |

What each check means and why: `rubric.md`.

## Setup (once)

```bash
cd evals/skill
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install -e "../../python[data]" pandas pillow pydantic
```

Provider keys are read from `~/Projects/lm15-dev/.env` (or
`$FUNCTAI_SKILL_EVAL_KEYS`); the tasks use `gpt-4.1-mini` (a few cents per
run). The agents run in Pi; the default agent is `openai-codex/gpt-6-astra`
through the ChatGPT login. (Pi's Claude login refuses headless runs as a
"third-party app" billed to extra usage, so it is not the default.)

## Run

```bash
.venv/bin/python run.py                         # 4 tasks × with/without × 1
.venv/bin/python run.py --reps 3 --parallel 6   # what produced the results below
.venv/bin/python run.py --tasks forms --conditions with --model openai-codex/gpt-6-luna
.venv/bin/python report.py runs/<stamp>         # the table again
.venv/bin/python run.py --collect runs/<stamp>  # grade every run again from its workspace (after fixing a grader)
```

Results go to `runs/<stamp>/` (ignored by git): `report.md`, and per run
`result.json` (grade, time, tokens, cost, tool calls, the task's model calls)
and the agent's whole `session.jsonl`. Workspaces stay in
`~/.cache/functai-skill-eval/runs/<stamp>/` for inspection.

## Check a grader

Each task has a hand-written solution in `tasks/<task>/reference/`, written the
way the skill teaches. A grader must give it full marks:

```bash
.venv/bin/python try_reference.py forms tasks/forms/reference/extract.py
```

## Fairness

- Both conditions: same task text, same model and thinking level, no
  extensions, no other skills, no AGENTS.md, no Chattering records. The only
  difference is `--skill <copy of skills/functai>`.
- Each run has its own functai disk cache and call log (`XDG_CACHE_HOME`,
  `FUNCTAI_LOG_CALLS`), so no run reuses another's replies.
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
