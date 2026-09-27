# FunctAI in eight tutorials

*Functions whose body is a language model, used like any other Python
function: on columns of a table, measured with intervals, chosen by
cost, trusted with decisions, baked into a model you own, and kept
honest in use.*

Each tutorial starts from a question about real data, builds the answer
one small step at a time, and ends with what it cost. Each runs top to
bottom in a fresh Python session, on models current in September 2026:
`gpt-6-luna` for everyday work; `gpt-6-sol`, `claude-sonnet-5`,
`claude-haiku-4-5`, `gemini-3.8-flash`, `gemini-3.1-flash-lite` and
`gpt-5.4-nano` where a comparison needs them; and TypeSafe's
`jev-latest`, a model built for decisions. Tables are
[dpyr](https://github.com/MaximeRivest/dpyr) data frames (dplyr's
grammar in Python); plots are matplotlib. Every output on these pages is
from a real run.

| | Tutorial | You will | Cost of a run |
|---|---|---|---|
| 1 | [Your first AI function](01-first-function.md) | sort 80 customer messages into teams, check them, improve them | under 1¢ |
| 2 | [Answers you can compute with](02-types.md) | turn bird-survey notes into typed columns: literals, counts that may be missing, dataclasses, lists | about 1¢ |
| 3 | [Is it right?](03-is-it-right.md) | measure with intervals, baselines, a confusion matrix, and run-to-run variation | under 1¢ |
| 4 | [Making it better without fooling yourself](04-making-it-better.md) | improve a refund decision with rules, examples and a teacher, on three piles of rows | about 2¢ |
| 5 | [Choosing a model](05-choosing-a-model.md) | compare eight models (TypeSafe's Jev among them) on accuracy, cost and speed, with paired comparisons and a rule | about 35¢ |
| 6 | [Decision models](06-decisions.md) | approve, deny or ask a person: costs of mistakes, rules in Python, calibrated probabilities from Jev, escalation, a decision tree | about 5¢ |
| 7 | [A model you own](07-a-model-you-own.md) | bake a 17M-parameter model that answers your function for free, and escalate only when it's unsure (needs a GPU in practice) | about 25¢ |
| 8 | [Living with it](08-living-with-it.md) | tools, the call log, people's corrections, versions, saving | under 1¢ |

## Before you start

You need Python 3.11 or later and a key for at least one model provider
(OpenAI's, for most of the series) in your environment:

```{.python .no-run}
pip install "functai[data]" matplotlib scikit-learn     # tutorial 7 also needs "functai[bake]"
```

Tutorial 1 assumes you know Python and have seen a data frame. Nothing
else is assumed; each tutorial says at the top what it covers, with a
short check so you can skip what you already know.

The same series exists [for R](../r/index.md), on the same datasets,
where tidymodels takes tutorial 7's place. The two follow the same
contract, so a function's version, its call log and its ratings are
shared between the languages.

## How these were made

The series was designed after reading the tutorials people recommend
most (R for Data Science, tidymodels' Get Started, Tidy Modeling with R,
Supervised Machine Learning for Text Analysis, Advanced R, and the LLM
packages' own guides, among others) for how they open, teach, and what
they leave out. The notes are in
[`design/02-r-tutorials.md`](https://github.com/MaximeRivest/functai/blob/master/design/02-r-tutorials.md).
To run them yourself from a checkout:
`python/.venv/bin/python tools/docs.py run docs/tutorials/*.md`, which
writes every output back into these pages.
