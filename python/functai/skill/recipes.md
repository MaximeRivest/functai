# FunctAI recipes (read SKILL.md first)

Complete scripts, each run against functai 1.2.0 (the evaluation tasks in
`evals/skill/` hold tested versions). Copy, then change the
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

## Measure, change, measure again

```python
import functai
from functai import ai
from dpyr import read
from typing import Literal

functai.configure(lm="gpt-4.1-mini", cache_replies="disk")
tickets = read("tickets.csv")                    # columns: message, category

@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

@ai
def team_v2(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message? An item that arrived
    broken is shipping; any request for money back is billing."""
    ...

before = functai.evaluate(team, tickets, expected="category", num_threads=8)
after = functai.evaluate(team_v2, tickets, expected="category", num_threads=8)
print(before, after, functai.compare(before, after), sep="\n")
before.table.to_pandas()      # every row: message, category, pred_result, exact_match, error, tokens
```

`compare` gives `before`, `after`, `diff`, `low`, `high`, `better`, `worse`,
`same`. A `low..high` that includes 0 means "can't tell yet".
`functai.evaluation.interval(values)` gives `(mean, low, high)` for any list of
0/1 or numeric scores.

## Training conversations for a small model

```python
import functai, functai.bake
from dpyr import read
from simplify import simplify          # the @ai function the student will answer

GUIDE = open("style_guide.md").read()
rows = read("rows.csv")                 # inputs + the answer in a column named like the output ("result")

ex = functai.bake.examples(simplify, rows, fixed={"style_guide": GUIDE})
print(ex.stats())
ex.save("train.jsonl")                  # one JSON object per line: "messages" (the reply last), "split", ...
```

No model call is made when every row has its answer. Rows without one are
answered by `teacher=` (priced first). `student="Qwen/Qwen3.5-0.8B"` adds
`input_ids` and `answer_start` under that model's chat template.

## Images

```python
import base64, io
from pathlib import Path
from typing import Literal

import functai
from functai import ai
from PIL import Image

functai.configure(lm="gpt-4.1-mini", cache_replies="disk")


def picture(path):
    """A picture as an image part's data: PNG, no EXIF or file name (they can give the answer away)."""
    image = Image.open(path).convert("RGB")
    image.info.clear()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"media_type": "image/png", "data": base64.b64encode(buffer.getvalue()).decode()}


@ai
def shape(picture: {"media": "image"}) -> Literal["circle", "square", "triangle"]:  # noqa: F821
    """The main shape in the picture."""
    ...


files = sorted(Path("images").glob("*.png"))
run = shape.map([{"file": f.name, "picture": picture(f)} for f in files], threads=8)   # pred_result per file
```

Check once that the request holds an image part:
`[p.type for p in shape.render(picture(files[0])).messages[0].parts]` → contains `"image"`.
Use a model that reads images (gpt-4.1-mini, Claude, Gemini, GPT-6). Tested:
12 of 12 colours right on the evaluation's pictures. Linters and type checkers object to a
dict annotation (ruff F821, pyright); `# noqa: F821` / `# type: ignore` quiet
them, and functai leaves such pragmas out of the prompt. lmcc is to map lm15's
`ImagePart` to a picture input, which a checker accepts (lmcc#3).

**Before functai 1.3** the dict annotation was turned into text (the picture
went as base64 the model could not see: 3 of 12 right). There, register a type:

```python
import lmcc

class Photo(dict):
    """An image part's data."""

lmcc.format(Photo, shape={"media": "image"}, write=lambda photo: [dict(photo, type="image")])
# then: def shape(picture: Photo) -> ...;  picture(path) returns Photo(**{...})
```

Other traps seen in real runs: a custom lm15 client that bolts the image onto
the request works but hides it from functai's log, cache key and replay.
