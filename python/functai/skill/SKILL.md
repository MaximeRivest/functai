---
name: functai
description: Use whenever a task involves FunctAI (Python `functai`, `@ai`, `_ai`, `functai.configure`), or when the user wants a language model to label, extract, classify, rewrite or judge many rows (papers, messages, images) as typed data, measure how well it does, or prepare data to train a small model on it. Covers writing AI functions whose guidance actually reaches the model, running them over tables with caching and resume, honest measurement, training-data export, pictures, and documents (PDF, Word, Excel, slides, web pages, folders) through Attachments.
---

# FunctAI: getting it right the first time

FunctAI turns a typed Python function into a model call: the **name, docstring,
argument names, types and comments are the prompt**; the answer comes back as
the declared type. Everything common is on this page, with the exact names of
what comes back: write the script from it, run it, look at the output once.
Side files, only for what this page does not cover:

- `recipes.md`: two models at once, a nested extraction with quotes checked, more on Attachments (sizes, errors, many documents), older versions
- `failures.md`: an error you do not understand (safety filters, rate limits, logins, unreadable files)
- `training.md`: training a small model itself (`bake`), beyond exporting its data

The project's Python is `.venv/bin/python`. Choose the model explicitly:
`functai.configure(lm="gpt-4.1-mini")`; subscriptions are `"claude:claude-opus-5-5"`,
`"chatgpt:gpt-6-astra"`; reasoning `reasoning="low"` (`"off"`, `"medium"`, `"high"`).

## 1. A function: its words reach the model

```python
from dataclasses import dataclass
from typing import Literal
import functai
from functai import ai, _ai

@dataclass
class Analysis:
    """One statistical test whose result the paper reports."""
    method: Literal[
        "t_test",   # two groups
        "anova",    # three or more groups
        "other",
    ]  # the test, as the codebook names it

@dataclass
class Study:
    units: int | None  # independent units per group; None when not stated
    analyses: list[Analysis]

@ai
def read_study(methods: str) -> Study:
    """Fill in the form from this methods section. Never guess: use None."""
    quotes: list[str] = _ai  # sentences copied word for word that support the answers
    return _ai
```

The docstring, and a comment after a parameter, a field, an `_ai` output, each
choice of a multi-line `Literal`, an `Enum` member, or a closing `]` all reach
the model, in nested records too; so do pydantic `Field(description=...)` and
`Annotated[T, "words"]`. A long text the model must follow (a codebook) can
also be an input. Not sent: section-banner comments, `# noqa`-style pragmas,
and, **when the code is run from `python - <<EOF` or `exec()`, every comment
and every `name: T = _ai` output** (both come from the source file; FunctAI
warns). Write scripts to files.

`Literal` for fixed answers; `X | None` with "None when not stated" instead of
guesses; outputs are written in declared order (`quotes` before the answer);
several small functions beat one giant form; `fn.using(lm=...)` is a copy with
other settings. To see the exact prompt without paying:
`print(fn.render(<one input>).system)`; worth one look when a function has
nested records or many described choices, not for a simple one.

## 2. Many rows

```python
from dpyr import read, col
functai.configure(lm="gpt-4.1-mini", cache_replies="disk")

papers = read("papers.csv")                                         # csv, parquet, pandas, list of dicts
papers = papers.mutate(**read_study.unpack(col.methods))            # one column per output: quotes, units, analyses
papers.select("paper_id", "units").write_csv("out/studies.csv")
papers.select("paper_id", "analyses").unnest("analyses").write_csv("out/analyses.csv")   # one row per analysis

run = read_study.map(papers, threads=8).to_dicts()                  # or: the run, one dict per row with
# the inputs, pred_<output> (pred_result for a single answer), error, seconds, *_tokens
```

Every table here also has `.to_pandas()` / `.to_polars()` (pandas needs
`pip install pandas`; `functai[data]` does not bring it). Each distinct input is asked once, 8 at a time. A failed row is a row with an
`error`, not an exception: no `try/except`. `cache_replies="disk"` makes a
re-run free and an interrupted run resume by running it again; no thread
pools, retry loops or JSON caches of your own. A long or costly run: try 2 rows
first (`papers.slice_head(n=2)`), then run it in the background with a log.

## 3. Measuring, and whether a change helps

```python
tickets = read("tickets.csv")                                        # message, category
before = functai.evaluate(team, tickets, expected="category", num_threads=8)
after = functai.evaluate(team_v2, tickets, expected="category", num_threads=8)

before.score                                                         # share right
s = before.summary.to_dicts()[0]                                     # metric, mean, low, high (95%), n, failed
c = functai.compare(before, after, margin=0.05).to_dicts()[0]        # before, after, diff, low, high (paired 95%),
                                                                     # better, worse, same, n, method, fragility
t = before.table.to_dicts()                                          # the rows: your columns, pred_result, exact_match, error, call_id
c["outcome"]                                                         # better, worse, same, not_worse, not_better or unclear
```

- **The change goes in the function, never in the data**: `team_v2` is a second
  `@ai` function (the new rules in its docstring) or `team.using(lm=...)`,
  evaluated on the same cases. `compare` pairs rows by their inputs (a column
  that is not an input changes nothing) and says how many cases only one side
  has. `evaluate` takes `@ai` functions or `@module`s, not plain wrappers.
- Read `outcome`, never the means alone: 30 cases right for both versions is
  `unclear` (the range is about ±0.11), not "no difference". More cases, or
  `repeats=3`, narrow it; `reuse=log` asks only what the log lacks.
- **No right answers?** Rate items: `real = functai.Criterion("real_problem", "label",
  items="/insights", answers={"yes": 1, "no": 0, "unsure": None})`; people rate
  a random draw (`functai.draw(ids, 20, seed)`, `rate(id, "yes", criterion=real,
  item=3, sample=seed, blind=True)`); a judge rates everything
  (`evaluate(..., criteria=[real.with_judge(judge)])`); `functai.scoreboard(prog,
  [real])` combines them and `functai.agreement(prog, real, "people", judge)` says
  whether the judge can be trusted. Hold out with `functai.split_of(row, prog)`;
  optimizers refuse `split="held_out"` rows.
- Metrics other than exact match: `functai.evaluate(fn, rows, {"name": metric})`,
  `metric(row, prediction) -> number`; or a judge `@ai` function.
- Quotes as evidence: `functai.quotes_found(text, quotes)` checks them, free.
- Never score on rows used for examples; a model's answers are not gold labels.
- **A score belongs to a setup**, not a version: `fn.setup` changes with the model,
  its settings, plugins, tools' code and the wall; `fn.version` does not. Compare
  two runs only with the setup each ran on (every record has `program.setup`);
  `functai.setup_diff(a, b)` says what differs.

## 4. Pictures and files

```python
from typing import Literal
from lm15.types import ImagePart, image
from attachments import Artifacts, att            # pip install "functai[files]"

@ai
def shape(picture: ImagePart) -> Literal["circle", "square", "triangle"]:
    """The main shape in the picture."""
shape(image(path="cat.png"))                      # a picture, not its name; image(data=..., media_type=...) too
run = shape.map([{"file": f.name, "picture": image(path=f)} for f in files], threads=8).to_dicts()

@ai
def answer(question: str, document: Artifacts) -> str:
    """Answer the question from the document, in a word or a number."""
answer("Which region sold most?", att("report.pdf[images: true]"))   # each page's text, then its picture
answer("Total spent?", att("budget.xlsx"))                           # sheets as tables
```

- A PDF sends its **text only** unless `[images: true]`: charts, scans and
  figures need the pictures. Several files are one value: `att(["a.md", "b.png"])`.
- Several pictures in one input: `list[ImagePart]` sends each as a picture.
  A dataclass holding a picture is refused (no layout can show it): give the
  picture its own input.
- Attachments shows file names; when a name could give the answer away, use `ImagePart`.
- A file `att()` could not read (missing, or a reader not installed) refuses
  the call before anything is sent, saying which and why.

## 5. Training data for a small model

```python
import functai.bake
rows = read("rows.csv")                          # the inputs, and the wanted answer in a column named like
                                                 # the output ("result" for a single answer); at least 10 rows
ex = functai.bake.examples(simplify, rows, fixed={"style_guide": open("style_guide.md").read()}, log=False)
ex.save("train.jsonl")                           # per line: messages [system, user, assistant] exactly as
                                                 # functai will send them, plus split, row_id, prompt, completion
```

`fixed=` leaves an input that is the same in every row out of the
conversations (the trained model learns it, and a call with another value is
refused). No model is called when every row has its answer. Use the public
API only: never rebuild prompts from functai's internals. Training itself:
`training.md`.

## 6. Scripts people can read, and when you are done

For scientists: top-to-bottom like a notebook; a docstring saying what it
does, how to run it and where results go; settings as constants at the top;
the run under `if __name__ == "__main__":`; no `try/except` around model calls.

Done means: the script ran, you looked at its output file once, failures are
counted, and accuracy claims carry their range and row count. For a run that
costs real money or hours, also check that running it again makes no new calls.
Tell the user what you did not check.

If `import functai` fails with `lmcc has no attribute ...`, a folder named
`lmcc` next to the script shadows the library (never run scripts saved in
`/tmp`); otherwise `uv sync`. Needs functai ≥ 1.3, lmcc ≥ 0.8.6, attachments ≥
1.0.0b2 for section 4's pictures and files.
