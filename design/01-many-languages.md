# FunctAI in many languages

*Status, 2026-09-27: agreed and under way. The repository is laid out
for it; the contract covers functions, scores, the call log and saved
functions; TypeScript 0.1.0, R 0.1.0 and Julia 0.1.0 pass all of it and
are checked against Python itself. None of the new packages is published
yet.*

*2026-10-02: Julia passes stages 1.2 to 5, plugins and the baked
examples too (every case Python's harness reads), and is checked against
Python on conversations, the disk reply cache and serving
(`tools/crosslang.py`). It writes training examples and calls students
trained elsewhere; it does not train yet (see* Training in every
language*).*

*2026-10-02: R passes them too (every case Python's harness reads, and
the streaming contract's `events/` cases), and is checked against Python
on conversations, the disk reply cache and serving. It writes training
examples and calls students trained elsewhere; it does not train. What R
does differently because it runs one thing at a time is in
`r/README.md`,* Where R differs, stated*.*

## What we want

FunctAI should exist natively in **Python, TypeScript/JavaScript, R and
Julia**. Each one should cover the whole loop in its own language:

1. write a typed AI function,
2. run it on a table,
3. measure how often it is right,
4. record calls and have people rate them,
5. improve it,
6. save it and ship it,
7. and in the long run, train its own weights ("bake").

The implementations are not wrappers around the Python one. An R user
should never need Python installed, and a web page should never need a
server running Python.

## What already exists underneath

FunctAI stands on two libraries:

- **lmcc** decides how values are written into a prompt and how the
  answer is read back.
- **lm15** talks to the model providers and handles sign-ins.

Both are built the way this plan proposes: one contract and native
implementations that are checked against it.

| | lm15 (providers, sign-ins) | lmcc (prompt layout, reading answers) | FunctAI |
|---|---|---|---|
| Python | reference | reference | 1.0.1 released, 1.1.0 not yet |
| TypeScript | passes the full contract | passes the same cases as Python, byte for byte (not on npm yet) | **0.1.0, in `ts/`** |
| Julia | at parity with the others since 2026-09-26 | passes the corpus (lmcc D-56) | **0.1.0, in `julia/`** |
| R | passes the full contract; reading keys from the environment fixed 2026-09-27 | passes the corpus (lmcc D-56) | **0.1.0, in `r/`** |

**TypeScript can start now.** Julia and R need lmcc in their language
first. That is the real prerequisite, and it should be built in the lmcc
repository against lmcc's existing cases, not inside FunctAI.

## The rule: one contract, native implementations

`contract/` is the authority. Every implementation passes its cases, and
when an implementation disagrees with the contract, the implementation
is wrong. No language folder imports another; they meet only through
the contract.

**Why not wrap Python?** A wrapper would use reticulate for R, PyCall or
PythonCall for Julia, and Pyodide for the web. It would be less work at
first, but it breaks what each audience came for:

- An R user would have to manage a Python environment.
- CRAN reviewers resist packages that depend on Python.
- A browser would download a Python runtime.
- Types would come back looking like Python's.

lm15 and lmcc already chose native implementations and proved that one
set of cases keeps them in agreement. **The cost is about four times the
code.** The contract is what keeps that from becoming four times the
bugs.

## What the contract holds, and what it doesn't

The contract fixes everything that has to be **the same across
languages**:

- data one language writes and another reads,
- numbers people compare,
- bytes a model sees.

It fixes nothing about how code looks in each language.

| Area | In the contract? | Today | Still to write |
|---|---|---|---|
| Call log, ratings, "rows with known answers" | yes | `calls.md`, schemas, 11 cases | nothing |
| An AI function: its signature, layout, model facts, worked examples, request, version | yes | `functions.md`, `layouts/`, `models.json`, 11 cases | nothing |
| Streaming events | yes | `streaming.md`, schema | cases: a scripted fake model that every language can run |
| **A saved AI function** (`functai.json`) | yes | `saved.md`, schema, 11 cases | tools bound by name (today a tool refuses) |
| Scores and their ranges | yes | `scores.md` (white space and case folding spelled out, `unicode/casefold.json`), 17 cases | nothing |
| Improving (optimizers) | the meaning, not the bytes | Python only | decide below |
| Baked models | the folder and the training data | `baked.json`, safetensors weights, tokenizer (`python/functai/bake/`) | `contract/baked.md`. See *Training* |
| Prompt layout and reading answers | no: it is lmcc's | lmcc corpus | nothing here |
| Providers, keys, sign-ins | no: it is lm15's | lm15 contract; saved sign-ins are shared between languages | nothing here |

**What travels between languages** is an AI function: its signature,
instruction, worked examples, settings and version. A saved *program*
also contains the ordinary code between its AI calls, such as
`code/*.py` for Python. That code belongs to one language. The saved
format should label each piece of code with its language. A loader in
another language should then refuse, before any call is made and with
a clear reason, to load a program that needs code it cannot run. It
must not guess.

## What each language decides for itself

This is how a function is written, how tables work, and how names are
spelled.

JSON field names in the contract stay `snake_case`. The API in each
language follows that language's own habits, such as `camelCase` in
TypeScript.

These are sketches, not commitments:

```python
@ai                                                    # Python: types and a docstring
def mood(review: str) -> Literal["happy", "unhappy", "mixed"]:
    """How does the customer feel about what they bought?"""
```

```ts
// TypeScript: types vanish when the code runs, so the shape is a value
const mood = ai("mood", {
  description: "How does the customer feel about what they bought?",
  input: { review: z.string() },
  output: z.enum(["happy", "unhappy", "mixed"]),
});
```

```r
# R: a model formula, what comes out ~ what goes in; an AI function
# takes whole columns, so it works inside dplyr::mutate()
mood <- ai(mood ~ review, "How does the customer feel about what they bought?",
           mood = choice("happy", "unhappy", "mixed"))
reviews |> mutate(mood = mood(review))
```

```julia
# Julia: a macro and real types; broadcasting runs it over a column
@ai function mood(review::String)::Mood
    "How does the customer feel about what they bought?"
end
df.mood = mood.(df.review)
```

Tables differ by language:

- **Python** uses dpyr.
- **R** uses the real dplyr. dpyr is a port of it, so R users get the
  original.
- **Julia** uses the Tables.jl interface, which covers DataFrames and
  every other Julia table.
- **TypeScript** uses arrays of objects. Arrow and DuckDB-Wasm can be
  added later.

## Training in every language (the long run)

Every language should eventually train its own weights, in its own
ecosystem:

| Language | Run a model | Train | GPU |
|---|---|---|---|
| Python | transformers | torch + peft (today) | CUDA, ROCm, MPS |
| TypeScript | transformers.js, onnxruntime-web | jax-js (JAX-style, WebGPU); young | WebGPU in the browser |
| R | torch for R (libtorch, no Python), `tok` for tokenizers, `hfhub` for downloads | torch for R | CUDA through libtorch |
| Julia | Transformers.jl, SafeTensors.jl | Lux.jl or Flux.jl; Reactant.jl | CUDA.jl: writing GPU code directly |

The contract should fix three things about baking.

- **What goes in.** The training rows are rendered by lmcc, so they are
  the same bytes in every language. The model is trained on exactly
  what it will see at run time, whichever language trained it.
- **What comes out.** A folder: `baked.json`, weights as safetensors,
  and the tokenizer. Every one of the four languages can read
  safetensors. A model baked in Julia should load in Python, and the
  other way round.
- **The gate.** Every bake reports its held-out score with a range.
  Loading the folder in another language must give a score inside that
  range on the same held-out rows.

The weights themselves are **not** required to match bit for bit
between languages. GPU arithmetic isn't exactly reproducible even
within one language.

The risk is uneven maturity: parameter-efficient fine-tuning and
tokenizer support are strongest in Python and thinnest in the browser.
Each language can offer baking as soon as it passes the gate, not
before.

## The repository

```
contract/     the authority: what every language must agree on
design/       notes like this one
docs/         the website (Python notebooks today)
tools/        builds the website
python/       the Python package: pyproject, uv.lock, .venv, tests, examples,
              README (PyPI's page), CHANGELOG, LICENSE
ts/           the TypeScript package (src, tests, tools)
r/  julia/    the R and Julia packages
check         one command: every language against the contract
AGENTS.md     the map, for people and agents working here
```

Each language folder is self-contained: its own manifest, lock file,
environment, tests, README, changelog and license copy. That way each
registry sees an ordinary package.

**One repository rather than one per language, as lm15 does.** The
contract still changes every day. The call log format is one day old
and streaming was added to it today. In one repository, a rule change
and every implementation's fix land in the same commit. lm15 splits by
language because its contract is frozen at 1.0, each language has its
own releases, and it needs a "contract pin" to hold them together.
FunctAI can split later if that becomes true here too. Splitting a
repository is easy; joining repositories back together is not.

## Releases

Each language has its own version and its own tag:

| Language | Tag | Registry |
|---|---|---|
| Python | `python-v1.1.0` | PyPI `functai` |
| TypeScript | `ts-v0.1.0` | npm `functai` |
| R | `r-v…` | CRAN `functai` |
| Julia | `julia-v…` | General registry, `FunctAI` (a package in a subfolder) |

- **Tags.** `.github/workflows/release.yml` publishes Python on
  `python-v*` tags. It refuses a bare `v1.1.0` tag, saying why, instead
  of guessing.
- **Versions.** They don't have to match: Python is at 1.x and
  TypeScript will start at 0.x. Instead, each implementation states
  which contract formats it reads and writes (`functai_call: 1`, …).
- **Names.** All four registry names were free on 2026-09-27. npm is
  the only one where a name can be taken by someone else before we
  publish.
- **CRAN rules** that shape the R design from the first day:
  - tests may not use the network;
  - a package may not write to the user's files unless the user asks.
    The call log is already off until turned on; when it is on, its
    folder must come from `tools::R_user_dir()`.

**Installing from git** now names the folder:
`pip install "functai @ git+https://github.com/MaximeRivest/functai#subdirectory=python"`.

## Documentation

Today the site is Python only. Every page pins `python/` as its rat
project, so it runs in `python/.venv`. When TypeScript has something to
show, the plan is the one the Polars user guide uses:

- one set of concept pages (the design recipe, measuring, improving,
  shipping), with each code example in a tab per language;
- a reference per language.

rat already runs R and Julia kernels, so the notebooks can stay
runnable in every language.

## Order of work

0. **Layout.** Done on 2026-09-27.
1. **Contract gaps TypeScript needed first.** Done on 2026-09-27:
   functions, scores and saved functions, with cases Python passes. Still
   open: streaming cases (a scripted fake model every language can run).
2. **TypeScript** (`ts/`, on `lmcc` and `@lm15/lm15`). 0.1.0 on
   2026-09-27: functions, tools, streaming, the call log, evaluation,
   `labeledFewShot` and `bootstrapFewShot`, loading what Python saved.
   It passes every case, `tools/crosslang.py` checks it against Python,
   and it answered live through OpenAI, Anthropic and Gemini. Not yet:
   `InstructionSearch`, stateful memory, escalation, the reply cache,
   baking. **To publish it**, lmcc's TypeScript kernel goes to npm first
   (and lmcc 0.8.4 to PyPI, so both languages run the same kernel).
3. **lmcc for Julia** (done, lmcc D-56), then **FunctAI.jl**: 0.1.0 on
   2026-09-27, below.
4. **lmcc for R** (done, lmcc D-56), then **the R package**: 0.1.0 on
   2026-09-27, below.
5. **Baking**, per language, each behind the gate above.

## Decisions still open

1. ~~A hand-written function's version includes its source text.~~
   **Decided 2026-09-27:** only code that runs beside the model is
   hashed; a function the model writes whole is versioned by its request
   alone. A call's `program.signature` leaves out host type names. The
   same function now has one version and one signature in every language.
2. ~~TypeScript's shapes.~~ **Decided:** `t` builders (no dependency) in
   the documentation, and zod 4 schemas accepted, read as the JSON Schema
   Python writes for the same type (no `$schema`, no safe-integer bounds,
   nullable as `anyOf`, a top-level description as the field's words).
3. **Improving in other languages.** The algorithms use randomness and
   model calls. The contract could fix only their meaning (which
   examples are eligible, what gets scored). Or it could fix exact
   choices, given a recorded set of replies and a seeded random
   generator that every language implements.
   - *Recommendation:* start with meaning. Move to exact choices only
     if users compare improvement runs across languages.
4. ~~The R face.~~ **Decided 2026-09-27**, see *R and the tidyverse*.
5. **Integrations tied to one ecosystem**, such as Prime Intellect's
   verifiers in `functai_verifiers` and `functai_rows`, stay in their
   language. They are not part of the contract.

## R and the tidyverse (0.1.0, 2026-09-27)

The R package pairs with the tidyverse rather than wrapping the Python
API:

- **An AI function is a vectorised R function** whose arguments are its
  inputs (`mood(review)`), so it is a column in any dplyr verb. One call
  per row, up to `concurrency` (8) in flight over curl: lm15 for R
  exposes its request builder and response reader, and the rows'
  requests share one curl pool. 80 rows took 9 s live.
- **A function is written as a model formula** (`team ~ message`), its
  fields as a codebook: a sentence describes a field, a type types it
  (`integer()`, `choice()`, `record()`, `list_of()`, `optional()`,
  `described()`), and `.data` gives fields their columns' types, as
  `lm()` reads them. See `03-r-formula.md`. Answers come back as those types; several
  outputs, and records, are tibble columns (`mutate(triage(x))` splices
  them in; `tidyr::unpack()` spreads a record).
- **`NA` in, `NA` out, without a call; failures are `NA` and one
  warning** (`ai_problems()`), as readr reports parsing problems. A
  single failing call is an error.
- **Evaluation speaks broom** (`tidy()`, `glance()`, `augment()`, columns
  `estimate`, `conf.low`, `conf.high`), and predictions speak tidymodels
  (`predict()`/`augment()` with `.pred`).
- **Immutable functions**: improving returns a copy (`fn |>
  labeled_few_shot(train)`), as do `with_demos()`, `with_instructions()`
  and `update()`. Settings follow withr (`with_ai_config()`,
  `local_ai_config()`).
- **It is also a tidymodels model** (`ai_model()`, a parsnip model type
  with the engine `"functai"`): the language model is the engine;
  fitting reads the outcome's levels and picks worked examples, and calls
  nothing; `examples` is tunable. Probabilities: OpenAI, Anthropic and
  Gemini measure none (lm15 MAP-14), so `samples = k` answers a row `k`
  times at temperature 1, asked for and paid for explicitly; without it
  they are refused, or `NA` through parsnip (whose `augment()` always asks).
  `evaluate()` takes any model with `predict()`. The vignette
  `tidymodels` teaches this with real results.
- Names avoid masking: `described()` (testthat has `describe`),
  `model_capabilities()` (base has `capabilities`), `read_ai()`/`write_ai()`
  (base has `load`/`save`).

Stated trade-offs: the call log's default folder is the contract's
(`~/.local/share/functai/calls`), not `tools::R_user_dir()`, so that every
language shares it; it is written only when asked, which is what CRAN's
policy requires. Program names default to the module `"__main__"`, as a
Python notebook's, so ratings pool with Python's. Streaming and modules
are not in 0.1.0 (both came on 2026-10-02, with stages 1.2 to 5).

## Julia and its ecosystem (0.1.0, 2026-09-27)

Julia's sketch above held: a macro and real types. What was decided on the way:

- **`@ai function mood(review::String)::Mood … end`**: the body is the
  description (a docstring's `# Arguments` list describes the inputs, as
  Julia documents functions), then `name::T = ai"words"` outputs, then code
  of your own. Types are Julia's (`@enum`, `OneOf(:a, :b)`, structs,
  `NamedTuple`s), written as the JSON Schema Python writes for the same
  type, so `mood` has Python's version (checked by `tools/crosslang.py`).
  Several outputs return all of them as a `NamedTuple` (Python returns the
  last); the call log still names the last as the answer.
- **One way over a column, many front doors.** DataFrames' `ByRow`,
  DataFramesMeta, Tidier and StatsModels' function terms all come down to
  broadcasting or `map`, so an AI function specializes those two (8 calls in
  flight, in order) and every table tool is concurrent without code for it.
  A failed row is `missing` with one warning, as R's `NA` is.
- **A model, three faces**: `AIModel` is `fit`/`predict` (StatsAPI), a
  formula (`fit(AIModel("…"), @formula(team ~ message), data)`, a
  StatsModels extension) and an MLJ model (`machine(AIModel("…"), X, y)`).
  Fitting calls nothing. Predictions are deterministic: no provider but
  TypeSafe measures class probabilities (lm15 MAP-14).
- **Settings** follow Julia: `configure!` (it mutates), `with_settings`
  over ScopedValues (it reaches the tasks a block starts), `configure(f; …)`
  a copy. `reasoning = true` replaces `module = "cot"`: `module` is a
  keyword Julia cannot parse as a name.
- **Code of its own is versioned by its parsed form** (layout and comments
  are not code), so it never shares a version with another language's code;
  a function the model writes whole has every language's version.

Stated trade-offs: MLJModelInterface is a dependency (MLJ's rule for
packages that define models; it is small), StatsModels and
CategoricalArrays are extensions. A program's version follows the AI
functions and programs it names, not plain Julia functions it calls.
Saving code of its own or tools from Julia is refused (a folder carries no
Julia code yet). Not in 0.1.0: the reply cache, stateful memory,
escalation, and baking. Teaching it: eight tutorials in `docs/julia/` and a
Documenter manual in `julia/docs/`, designed in `05-julia-tutorials.md`.

## Found on the way

- **The `json` layout with a record answer failed at OpenAI and
  Anthropic** (their strict schema mode wants `additionalProperties:
  false` on every object; lmcc's `json_object` reader closed only the
  outer one). Found live on 2026-09-27, **fixed the same day in lmcc**
  (`reader/json_object` 0.2.1, lmcc D-57: every record in the requested
  schema is closed), in all four lmcc kernels. Both FunctAI languages run
  on lmcc's checkout until lmcc 0.8.4 is released; Python's dependency is
  `lmcc>=0.8.4`, so releasing FunctAI for Python waits for that release.

- **lm15 for R could not read API keys from the environment** (the
  `Sys.getenv()` value kept its `Dlist` class, and the key check refused
  it): fixed in lm15-r on 2026-09-27, with a regression test.
- **The first call of a Julia session compiled for about a minute.** A
  precompile workload in FunctAI.jl brought it to about ten seconds; the
  rest is lm15's HTTP and TLS code, which only a workload in lm15 (against
  a local server) can compile ahead.
- **Python cannot load an AI function saved from TypeScript, R or Julia**: its
  `load` runs saved Python code. A data-only loader, like the other two
  languages have, is the missing piece for "improve in R, run in Python".
