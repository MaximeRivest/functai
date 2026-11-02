# FunctAI recipes: what SKILL.md does not cover

The recipes here hand tables to pandas (`.to_pandas()`): `pip install pandas` with `functai[data]`.

Complete scripts (the evaluation tasks in `evals/skill/` hold tested versions). Copy, then change the
names. Read `SKILL.md` first for the rules behind them.

## Extraction: one paper → a record with a list inside

Nested records use pydantic, so the field descriptions reach the model at every
depth (dataclass comments only work at the top level).

```python
"""Read the methods of each paper and fill in a form: the design, and every analysis.

Run it:   .venv/bin/python extract.py
Results:  out/designs.csv (one row per paper), out/analyses.csv (one row per analysis)
Running it again costs nothing: the model's replies are kept on disk.
"""
from typing import Literal

import os

import functai
from dpyr import read, col
from functai import ai, _ai
from pydantic import BaseModel, Field

# =============================================================================
# Settings
# =============================================================================
functai.configure(lm="gpt-4.1-mini", cache_replies="disk")
PAPERS = "data/papers.csv"            # columns: paper_id, methods
os.makedirs("out", exist_ok=True)


# =============================================================================
# The form
# =============================================================================
class Analysis(BaseModel):
    """One statistical test or model whose result the paper reports."""
    outcome: str = Field(description="the measured response, as the paper names it")
    method: Literal["t_test", "anova", "mixed_model", "regression", "other"] = Field(
        description="t_test: two groups; anova: three or more groups or factors; "
                    "mixed_model: any model with random effects; regression: continuous predictors")
    pseudoreplicated: bool = Field(
        description="true when repeated measurements of the same unit are treated as independent")


class Study(BaseModel):
    design: Literal["randomized_experiment", "observational", "not_stated"] = Field(
        description="randomized_experiment: the researchers assigned treatments at random")
    independent_units: int | None = Field(description="how many independent units; null when not stated")
    analyses: list[Analysis]


@ai
def read_study(methods: str) -> Study:
    """Fill in the form from this methods section. Report what the authors did,
    not what they should have done. Never guess: use not_stated or null."""
    quotes: list[str] = _ai  # sentences copied word for word that support the answers
    return _ai


# =============================================================================
# The run
# =============================================================================
if __name__ == "__main__":
    papers = read(PAPERS)
    print(read_study.render(papers.pull(col.methods)[0]).system)       # check the prompt once

    papers = papers.mutate(**read_study.unpack(col.methods))     # one column per output: quotes, design, ...
    papers.select("paper_id", "design", "independent_units").write_csv("out/designs.csv")
    papers.select("paper_id", "analyses").unnest("analyses").write_csv("out/analyses.csv")   # one row per analysis
```

To keep failed rows instead of stopping at the end, use the run table:
`run = read_study.map(papers, threads=8).to_pandas()`. A record answer comes
back as one `pred_<field>` column per field (`pred_design`, `pred_analyses`,
plus `pred_quotes` for the extra output), with `error`, `seconds` and token
columns; integer fields with gaps arrive as floats. Check quotes with
`functai.quotes_found(row["methods"], list(row["pred_quotes"]))`.

## Two models on the same rows, at the same time

```python
from concurrent.futures import ThreadPoolExecutor

MODELS = {"astra": dict(lm="chatgpt:gpt-6-astra", reasoning="low"),
          "opus": dict(lm="claude:claude-opus-5-5", reasoning="off", max_tokens=32000)}

def run_one(name):
    out = identify.using(**MODELS[name]).map(rows, threads=8).to_pandas()
    out["model"] = name
    return out

with ThreadPoolExecutor(len(MODELS)) as pool:        # the only thread pool you need: one per model
    tables = list(pool.map(run_one, MODELS))
```

For probes on a subscription, add `api_retries=0` so a multi-hour rate limit
fails at once instead of sleeping.

## Pictures and documents: beyond SKILL.md §4

**Before functai 1.3 / lmcc 0.8.6** `ImagePart` was refused as an input type.
There, annotate `picture: {"media": "image"}` (1.3 only) or register a type:
`lmcc.format(Photo, shape={"media": "image"}, write=lambda p: [dict(p, type="image")])`
with `class Photo(dict)`, given `Photo(media_type=..., data=<base64>)`.

More on Attachments inputs:

- The model gets Attachments' own parts: text and page images as image parts,
  page by page. Its JSON form (Attachments' wire form) is what the call log,
  the reply cache and replays hold.
- File names are shown (as `att()` prints them). When a name could give the
  answer away (a classifier over `cat_01.png`), use `ImagePart` above instead.
- A file `att()` could not read (missing, or a reader not installed:
  `pip install "attachments[pdf]"`) refuses the call before any request
  (`InterfaceError`, naming the file and why). `att(...).errors` lists them.
- **A PDF sends its text only** unless `[images: true]`: a chart's numbers, a
  scan or a figure are in the pictures (by default Attachments draws pages
  only when a PDF has no text). Pictures cost tokens: ask for the pages you need.
- PDF pages are drawn at most 2000 px wide by default (`[max_dim: 1568]`,
  `[image_format: jpeg]` to send less); `att(...).estimate_tokens()` counts
  the pictures too.
- Ask questions about many documents with `map`:
  `summarize.map([{"report": att(f)} for f in files], threads=8)`.

## Agents that browse for many steps, and steps that finish elsewhere

A tool may answer with a picture: return `image(...)` (from `lm15.types`),
or a list of texts and parts. Then steer the loop with plugins, which are
recorded as data and keep the call replayable:

```python
from lm15.models import InferencePricing
from lm15.types import image

@functai.tool(effects="changes")
def click(target: str):
    """Click the element, then see the screen."""
    return [f"clicked {target}", image(data=screenshot_b64(), media_type="image/png")]

PRICES = {"claude-sonnet-5": InferencePricing(input_per_million=3, output_per_million=15,
                                              cache_read_per_million=0.3, cache_write_per_million=3.75)}
browse = explore.using(max_steps=200, plugins=[
    functai.keep_pictures(3, at_once=5),                     # old screenshots out, 5 at a time (prompt cache)
    functai.limits(cost=3.0, seconds=900, prices=PRICES),   # then the loop answers with what it has
])
```

- Never write your own `request` hook to drop pictures or nudge the model:
  the `step` hook does it (`tools`, `reminder`, `replace`, `stop`) and the
  call stays replayable.
- A cost limit with a price that lacks cache rates is reached at once for
  providers that do not report cache counts: give all four rates.
- Progress of plain code: `functai.note("screen captured", {"n": 4})`.
- Work that finishes elsewhere (a cloud browser, a job, a webhook): the tool
  returns `functai.later(f"harvest:{job_id}")`, the agent runs as a
  conversation's turn (`agent.conversation(run_id, store=...)`), and the
  webhook calls `chat.deliver(f"harvest:{job_id}", result)`. Code in a
  module body is not recorded: anything that must not run twice belongs in
  a tool that `changes` things.
- No disk (serverless): `configure(log_calls=MyStore())`, any object with
  `append(record)` and `read()`.
