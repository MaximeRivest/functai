---
name: functai
description: Use whenever a task involves FunctAI (Python `functai`, `@ai`, `_ai`, `functai.configure`), or when the user wants a language model to label, extract, classify, rewrite or judge many rows (papers, messages, images) as typed data, measure how well it does, or prepare data to train a small model on it. Covers writing AI functions whose guidance actually reaches the model, running them over tables with caching and resume, honest measurement, provider failures, training-data export and images.
---

# FunctAI: getting it right the first time

FunctAI turns a typed Python function into a model call: the **name, docstring,
argument names, types and some comments are the prompt**; the answer comes back
as the declared type. Distilled from real projects; every rule here cost
someone hours. Read the side files only when the task needs them:

- `recipes.md`: complete tested scripts (nested extraction, two models at once, measure-and-compare, training export, **images**). Images need that recipe: functai has no image type, and an annotation like `{"media": "image"}` is silently sent as base64 *text* the model cannot see
- `failures.md`: refusals, safety filters, rate limits on subscriptions, overloads, logins
- `training.md`: baking a small model, exporting training conversations

## 1. Setup checks (one command each)

```bash
PY=.venv/bin/python          # the project's environment
$PY -c "import functai, lmcc; print(functai.__version__, lmcc.__file__)"
$PY -c "import functai; [print(l) for l in functai.logins()]"     # what can pay for calls (reads files only)
```

- `lmcc has no attribute 'default_registry'` or an `lmcc.__file__` of `None`:
  a folder named `lmcc` next to the script shadows the library (a `/tmp/lmcc`
  link did on Maxime's machine until 2026-10-06): **never run scripts saved directly in `/tmp`**; put
  probes in the project (`scratch/probe.py`). Only if the path is right, re-sync
  (`uv sync`). Never patch functai from a user project; pin a version
  (`functai==1.2.0`) for work that must be reproducible.
- Choose the model explicitly: `functai.configure(lm="gpt-4.1-mini")`. Subscriptions:
  `"claude:claude-opus-5-5"` (= `claude-code:`), `"chatgpt:gpt-6-astra"` (= `openai-codex:`).
  Reasoning: `reasoning="low"` (`"off"`, `"medium"`, `"high"`).

## 2. Make sure your guidance reaches the model

```python
from typing import Literal
from pydantic import BaseModel, Field
import functai
from functai import ai, _ai

class Analysis(BaseModel):
    method: Literal["t_test", "anova", "other"] = Field(description="t_test: two groups; anova: three or more")

class Study(BaseModel):
    units: int | None = Field(description="independent units per group; null when not stated")
    analyses: list[Analysis]

@ai
def read_study(methods: str) -> Study:
    """Fill in the form from this methods section. Never guess: use null."""
    quotes: list[str] = _ai  # sentences copied word for word that support the answers
    return _ai
```

What reaches the prompt (measured on functai 1.2.0):

| you write | reaches the model? |
|---|---|
| the docstring; `_ai["description"]`; a comment after an `_ai` output | yes |
| a one-line comment after an argument, or after a field of the **returned** dataclass | yes |
| pydantic `Field(description=...)` and pydantic class docstrings, **at any depth** | yes |
| a long text passed as an input (a codebook, a guide) | yes |
| comments on each choice of a multi-line `Literal[...]`, on `Enum` members, or after the closing `]` | **no** |
| comments on fields of a dataclass **nested** in the result (`list[Analysis]`) | **no** |
| a dataclass docstring, `Annotated[..., "text"]` | **no** |
| **any** comment when the code runs from `python - <<EOF`, `python -c`, or `exec(...)` | **no** |

So: nested records → pydantic with `Field(description=...)`; menu choices →
define them in the field's description or the docstring; test from a file, not
a heredoc. **Before paying, look:** `print(read_study.render(<one input>).system)`
(no call is made) and check every definition you wrote is there.

Also: `Literal` for fixed answers; `X | None` + "null when not stated" instead of
guesses; outputs are written in declared order (put `quotes`/`reasoning` before
the answer); the body runs after the answer, so plain Python can check it;
several small functions beat one giant form; `fn.using(lm=..., max_tokens=...)`
is a copy with other settings.

## 3. Many rows

```python
from dpyr import read, col
functai.configure(lm="gpt-4.1-mini", cache_replies="disk")
papers = read("papers.csv").mutate(**read_study.unpack(col.methods))     # one column per output
papers.select("paper_id", "units").write_csv("out/studies.csv")
papers.select("paper_id", "analyses").unnest("analyses").write_csv("out/analyses.csv")
```

- AI function on a column → a column (`mutate(label=fn(col.text))`); a record →
  `fn.unpack(...)`; lists → `.unnest(...)`. 8 rows at a time; each distinct input asked once.
- `fn.map(rows, threads=8)` returns the run as a table: `pred_<field>` columns,
  `error`, `seconds`, tokens. Failed rows are rows, not exceptions: no `try/except`.
- `cache_replies="disk"`: a re-run is free and an interrupted run **resumes by
  running it again**. Don't hand-roll thread pools, retry loops or JSON caches.
- `configure(log_calls="out/calls")` keeps every request and reply for audit.
- Scale the process to the stakes: for an expensive or long run, try 2 rows
  first (`.slice_head(n=2)`) and read `print(functai.phistory(1))`; run it in the
  background (`nohup $PY run.py > run.log 2>&1 &`) and check `tail -2 run.log`
  instead of sleep-polling. For a run that costs cents, just run it.

## 4. Measuring honestly

```python
before = functai.evaluate(team, tickets, expected="category", num_threads=8)   # score + 95% range
after = functai.evaluate(team_v2, tickets, expected="category", num_threads=8)
functai.compare(before, after)      # paired: diff, low, high, better, worse
```

- **Put the change in the function, never in the data**: the new version is a
  second `@ai` function (new rules in its docstring) or `fn.using(lm=...)`,
  evaluated on **the very same table**. `compare` refuses ("needs the same
  examples in the same order") if you added a column for one version.
  `evaluate` takes `@ai` functions or `@module`s, not plain wrappers.
- `before.summary` has `low`/`high`; `compare(...)` gives the paired t interval.
  If it includes 0, the change may be luck: say so.
- Evidence: return `quotes` and check them free with `functai.quotes_found(text, quotes)`.
- Never score on rows used for examples or optimization; generated answers are not gold labels.

## 5. Scripts people can read

For scientists: top-to-bottom like a notebook; a docstring saying what it does,
how to run it, where results go; settings as constants at the top; `# ====`
section headers; the run under `if __name__ == "__main__":`. Hard-code what is
fixed; no clever abstractions; no `try/except` around model calls.

## 6. Before you say "done"

- [ ] `render(...)` shows every definition you meant the model to have.
- [ ] Outputs checked by eye on a few rows; failures counted separately.
- [ ] Re-running makes no new calls (`cache_replies="disk"`).
- [ ] Accuracy claims carry their range and row count; comparisons are paired.
- [ ] You told the user honestly what was not checked.
