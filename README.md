# functai

**Write a function's signature and one sentence saying what it does. A
language model writes the body. You measure how well.**

functai turns a typed function into a call to a language model. The
function's name, description and types say what you want; the answer comes
back as the type you asked for. Then you run it on a whole table, find out
how often it is right, and make it better. It exists in Python, TypeScript,
R and Julia, each written the way that language writes functions:

```python
@ai
def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
    """Which team should answer this customer message?"""
    ...

team("I was charged twice for order B-2210, please fix this.")    # 'billing'
functai.evaluate(team, tickets, expected="category")              # exact_match 0.97 [0.91, 0.99]
```

```r
team <- ai(team ~ message, "Which team should answer this customer message?",
  team = choice("shipping", "billing", "product", "account"))

tickets |> mutate(team = team(message))                           # a factor column
evaluate(team, tickets, expected = category)                      # exact_match: 0.96 (95% interval 0.90 to 0.99)
```

The [website](https://maximerivest.github.io/functai/) shows the same
function in all four languages, with the answers of a real run.

## Languages

| Language | Folder | Install | Learn it |
|---|---|---|---|
| Python | [`python/`](python/) | `pip install "functai[data]"` ([PyPI](https://pypi.org/project/functai/), 1.2.0) | [get started](https://maximerivest.github.io/functai/get-started.html), [8 tutorials](https://maximerivest.github.io/functai/tutorials/index.html), [reference](https://maximerivest.github.io/functai/reference/index.html) |
| TypeScript / JavaScript | [`ts/`](ts/) | not on npm yet (0.1.0, from a checkout) | [guide](https://maximerivest.github.io/functai/ts/index.html), [API reference](https://maximerivest.github.io/functai/ts/api/index.html) |
| R | [`r/`](r/) | `remotes::install_github("MaximeRivest/functai", subdir = "r")` (0.1.0, not on CRAN yet) | [8 tutorials](https://maximerivest.github.io/functai/r/index.html), [manual](https://maximerivest.github.io/functai/r/manual/index.html) |
| Julia | [`julia/`](julia/) | `Pkg.add(url = …)` (0.1.0, not registered yet; see [julia/](julia/#install)) | [8 tutorials](https://maximerivest.github.io/functai/julia/index.html), [manual](https://maximerivest.github.io/functai/julia/manual/index.html) |

Every language follows the same [contract](contract/): the same function
has the same version everywhere, a call logged in one can be rated in
another, and a function saved in Python loads in the other three and sends
the same request. What each language has, and does not have yet, is in the
table on the [website's home page](https://maximerivest.github.io/functai/#what-each-language-has).
[design/01-many-languages.md](design/01-many-languages.md) is the plan.

## This repository

| Path | What it holds |
|---|---|
| [`contract/`](contract/) | what every implementation must agree on: formats, schemas, cases |
| [`python/`](python/) | the Python package, its tests and examples |
| [`ts/`](ts/) | the TypeScript package and its tests |
| [`r/`](r/) | the R package and its tests |
| [`julia/`](julia/) | the Julia package (`FunctAI.jl`), its tests and its manual |
| [`docs/`](docs/), [`tools/`](tools/) | the website: runnable notebooks, and the tools that run and build them; `tools/crosslang.py` checks the four languages against each other |
| [`design/`](design/) | design notes |
| [`check`](check) | one command: every implementation against the contract |

The website is built by [`.github/workflows/docs.yml`](.github/workflows/docs.yml)
on every push to `master`: the pages in `docs/` (their outputs already in
them) with Zensical, and each language's own manual beside them (Python's
reference from its docstrings, TypeDoc for TypeScript, pkgdown for R,
Documenter for Julia). No model is called to build it.

MIT licensed.
