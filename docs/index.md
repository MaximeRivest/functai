---
rat:
  project: ../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# functai

You already know how to write a good function: say what goes in and what comes out, say in one sentence what it does, write a few examples with the answers you expect, then write the body and check it against those examples. Teachers call this the *design recipe*. functai keeps every step but one. Your types are the **signature**, your docstring is the **purpose statement**, your table of trusted answers is the **examples**, and a language model writes the **body**. The answer comes back as the type you asked for, so you can **test** it like any other function.

Here is the recipe, one step at a time. The **signature** and the **purpose statement** are the first two lines you write, and the ellipsis `...` is the **body** you leave to the model:

```python
from typing import Literal
import functai as f
f.login('openai-codex')

@f.ai(lm='openai-codex:gpt-6-sol')
def mood(review: str) -> Literal['happy', 'unhappy', 'mixed']:
    """How does the customer feel about what they bought?"""
    ...
```

The **examples** are answers you trust, written as plain rows:

```python
examples = [
    {'review': 'Arrived early and works perfectly.', 'mood': 'happy'},
    {'review': 'Broke after two days. Never again.', 'mood': 'unhappy'},
    {'review': 'Great sound, but the strap snapped.', 'mood': 'mixed'},
]
```

The answer comes back as the type you asked for, so you **test** it the way you test any function, one call at a time or on all the examples at once:

```python
mood('The box was dented, but everything inside is fine.')
```

```python
f.evaluate(mood, examples, expected='mood')
```

The first number `evaluate` gives is how often it was right; the range says how sure that number is. That loop (**write, run, measure, improve**) is what functai is for.

The docstring is the whole instruction, so it can be anything. Here is one that ignores its input on purpose:

```python

@f.ai(lm = 'openai-codex:gpt-6-sol')
def my_ai_program(my_input):
  """
  No matter what, you are only telling dad's jokes.
  """
  ...

my_ai_program('Who are you?')
```

```output
Already signed in: Login(ChatGPT: saved: use CLI login, ready, until unknown). Use functai.login('openai-codex', again=True) to sign in again.
"I'm a dad-joke bot. I used to be a banker, but I lost interest."
```

functai exists in Python, TypeScript, R and Julia, each written the way
that language writes functions. Here is one function, `team`, in all four:

=== "Python"

    ```python
    from typing import Literal
    from dpyr import col
    import functai
    from functai import ai

    functai.configure(lm="gpt-6-luna")

    @ai
    def team(message: str) -> Literal["shipping", "billing", "product", "account"]:
        """Which team should answer this customer message?"""
        ...

    team("I was charged twice for order B-2210, please fix this.")    # 'billing'

    tickets = functai.datasets.tickets()                    # 80 labelled support messages
    tickets.mutate(team=team(col.message))                  # a new column
    functai.evaluate(team, tickets, expected="category")    # exact_match 0.97 [0.91, 0.99]
    ```

    [Start with Python](python.md){ .md-button }

=== "TypeScript"

    ```ts
    import { ai, t, evaluate, configure } from "functai";

    configure({ lm: "gpt-6-luna" });

    const team = ai("team", {
      description: "Which team should answer this customer message?",
      input: { message: t.string() },
      output: t.enum("shipping", "billing", "product", "account"),
    });

    await team("I was charged twice for order B-2210, please fix this.");   // "billing"

    // tickets: the same 80 messages, as rows { message, category }
    await team.map(tickets.map((row) => row.message));   // every answer, in order
    String(await evaluate(team, tickets, { expected: "category" }));
    // "exact_match: 0.95 (95% range 0.88 to 0.98), n=80"
    ```

    [Start with TypeScript](ts/index.md){ .md-button }

=== "R"

    ```r
    library(functai)
    library(dplyr)

    ai_config(lm = "gpt-6-luna")

    team <- ai(team ~ message, "Which team should answer this customer message?",
      team = choice("shipping", "billing", "product", "account"))

    team("I was charged twice for order B-2210, please fix this.")
    #> [1] billing
    #> Levels: shipping billing product account

    tickets |> mutate(team = team(message))          # a factor column
    evaluate(team, tickets, expected = category)
    #> <evaluation of team> 80 rows
    #>   exact_match: 0.96  (95% interval 0.90 to 0.99)
    ```

    [Start with R](r/index.md){ .md-button }

=== "Julia"

    ```julia
    using FunctAI, DataFrames

    FunctAI.configure!(lm = "gpt-6-luna")

    @enum Team shipping billing product account

    @ai function team(message::String)::Team
        "Which team should answer this customer message?"
    end

    team("I was charged twice for order B-2210, please fix this.")   # billing::Team

    tickets = DataFrame(FunctAI.tickets())           # 80 labelled support messages
    tickets.team = team.(tickets.message)             # the whole column, 8 calls at a time
    evaluate(team, tickets; expected = :category)     # exact_match 0.96 (95% range 0.90 to 0.99)
    ```

    [Start with Julia](julia/index.md){ .md-button }

Each ran for real, on the same 80 messages with the same model, and the
comments are what came back. The four scores differ by a message or two:
a model does not answer the same way every time, which is why every score
comes with its range, and why the four ranges overlap.

## One function, four languages

The four implementations follow one
[contract](https://github.com/MaximeRivest/functai/tree/master/contract),
checked on every change:

- **The same function has the same version.** `team` above is
  `sha256:99ae724a…` in all four. The version names everything the
  function sends besides its inputs, so it changes when the function
  does, and only then.
- **One call log.** Turn it on, and every call is a line of JSON in one
  folder that all four write and read. A call made in Julia can be rated
  by a person in R, and the rating becomes a row Python improves with.
- **One saved form.** A function saved to a folder in one language loads
  in another and sends the same request, byte for byte, when it has no
  code of its own. TypeScript, R and Julia load folders saved by any of
  the four; Python loads its own, and the others' next.

## What each language has

Python came first and has the most. The others add what their ecosystem
expects (tidymodels in R, MLJ and formulas in Julia, compile-time types
in TypeScript) and catch up on the rest.

| | Python | TypeScript | R | Julia |
|---|:-:|:-:|:-:|:-:|
| Typed AI functions: answers checked against their type, asked again when they don't fit | ✓ | ✓ | ✓ | ✓ |
| A choice whose answers each say what they mean, sent the same way in every language | comments | `t.choice` | `choice()` | `OneOf` |
| Run on a whole table | [dpyr](https://github.com/MaximeRivest/dpyr): pandas, polars, files, databases | arrays of rows | dplyr verbs | broadcasting, DataFrames |
| How often is it right, with a 95% interval | ✓ | ✓ | ✓ | ✓ |
| Two versions compared row by row | ✓ | ✓ | ✓ | ✓ |
| Worked examples chosen from your rows | ✓ | ✓ | ✓ | ✓ |
| The instruction rewritten from its mistakes (GEPA) | ✓ | ✓ | ✓ | ✓ |
| Instruction search, random search | ✓ | ✓ | ✓ | ✓ |
| A model among statistical models | – | – | tidymodels | MLJ, formulas |
| Tools the model calls | ✓ | ✓ | ✓ | ✓ |
| Streaming | ✓ | ✓ | ✓ | ✓ |
| Your code around several AI functions, logged as one call | `@module` | `module()` | `ai_program()` | `@program` |
| Call log, and people's ratings as data | ✓ | ✓ | ✓ | ✓ |
| Loads functions saved in other languages | ✓ | ✓ | ✓ | ✓ |
| Reply cache (off unless you turn it on), kept on disk across runs | ✓ | ✓ | ✓ | ✓ |
| Conversations: turns that remember, branches, kept in a store | ✓ | ✓ | ✓ | ✓ |
| Tools that ask a person first; a waiting turn goes on later, paying for nothing twice | ✓ | ✓ | ✓ | ✓ |
| Plugins: hooks over turns, calls and tools, every change recorded | ✓ | ✓ | ✓ | ✓ |
| Serve a program over HTTP, and use one served elsewhere | ✓ | ✓ (any `fetch` host: Node, Deno, Bun, Workers) | ✓ (one request at a time) | ✓ |
| Escalation to a bigger model when the first is unsure | ✓ | ✓ | ✓ | ✓ |
| Sign in with a Claude, ChatGPT or Copilot subscription | ✓ | ✓ | ✓ | ✓ |
| A function's own key, or your own model server (vLLM, Ollama) | ✓ | ✓ | ✓ | ✓ |
| Bake it into a small model you own | ✓ | training data, and runs a baked model that writes text | training data, and runs a baked model that writes text | training data, and runs a baked model that writes text |
| The instruction written and refined by the model as calls come in; training rows a teacher model writes | ✓ | – | – | – |
| Check that a saved program still sends what it sent (`verify`), what a program depends on (`check`), past evaluations (`runs`) | ✓ | – | – | – |
| Released | [PyPI](https://pypi.org/project/functai/) | not yet on npm | not yet on CRAN | not yet registered |

Where a language has no ✓, a key in the environment, your own code, or
Python does the job for now.

## Pick your language

<div class="journey" markdown>

- **[Python](python.md)**
  `pip install "functai[data]"`. Three ways in, [eight tutorials](tutorials/index.md), articles, examples and the [reference](reference/index.md).
- **[TypeScript](ts/index.md)**
  From a checkout for now. A [guide](ts/index.md), and the [API reference](ts/api/index.html) with every type.
- **[R](r/index.md)**
  `remotes::install_github("MaximeRivest/functai", subdir = "r")`. [Eight tutorials](r/index.md) and the [manual](r/manual/index.html), with the vignettes.
- **[Julia](julia/index.md)**
  `Pkg.add(url = …)`. [Eight tutorials](julia/index.md) and the [manual](julia/manual/index.html), with guides and the reference.

</div>

The tutorials follow the same path in Python, R and Julia, on the same
data: a first function, types, is it right, making it better, choosing a
model, decisions, and living with it in use.

## Getting help

Something doesn't work the way this site says? Please
[open an issue](https://github.com/maximerivest/functai/issues) with a
short example, and say which language.
