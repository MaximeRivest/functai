# FunctAI in many languages

*Status, 2026-09-27: the direction is agreed and the repository is laid
out for it. Python is the only implementation. The next step is
TypeScript.*

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
| TypeScript | passes the full contract | passes the same cases as Python, byte for byte | **next** |
| Julia | at parity with the others since 2026-09-26 | **does not exist** | after lmcc |
| R | pinned to the same contract commit (its README says it passes); not yet in lm15's parity table | **does not exist** | after lmcc |

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
| Versions (what makes two calls the same program) | yes | rules in `calls.md` | cases: sample input → rendered request → version. Needs lmcc parity, so it is also a good cross-language test |
| Streaming events | yes | `streaming.md`, schema | cases: a scripted fake model that every language can run |
| **A saved AI function** (`functai.json`) | should be | described only in `python/functai/saved.py` | `contract/saved.md`, a schema and cases. This is what makes "improve it in Python, run it in R" true |
| Scores and their ranges | should be | Wilson interval for 0/1 scores, Student's t otherwise (`evaluation.py`) | cases: the same rows give the same numbers, to a stated precision |
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
const mood = ai("How does the customer feel about what they bought?", {
  inputs: { review: z.string() },
  output: z.enum(["happy", "unhappy", "mixed"]),
});
```

```r
# R: no type annotations; the signature is written out, and an AI
# function takes whole columns, so it works inside dplyr::mutate()
mood <- ai("How does the customer feel about what they bought?",
           review = character(), .returns = c("happy", "unhappy", "mixed"))
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
ts/           next
r/  julia/    later
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
1. **Contract gaps TypeScript needs first:**
   - the saved AI function format (`contract/saved.md`, schema, cases);
   - version cases;
   - score cases;
   - a scripted fake model for streaming cases.
   Each is written from the Python behaviour and checked by the Python
   tests first.
2. **TypeScript** (`ts/`, on `lmcc` and `@lm15/lm15`), in this order:
   - write, run and read an AI function;
   - the call log;
   - evaluation;
   - load a function saved in Python;
   - streaming;
   - improvement.
3. **lmcc for Julia** (in lmcc), then **FunctAI.jl**.
4. **lmcc for R** (in lmcc), then **the R package**.
5. **Baking**, per language, each behind the gate above.

## Decisions still open

1. **A hand-written function's version includes its source text**
   (`calls.md`, *Versions*). So the same AI function written in Python
   and in TypeScript gets two versions, and ratings won't pool across
   languages.
   - *Recommendation:* hash only code that runs beside the model. A
     function whose body is left to the model (`...`) has no code in
     its version, only its rendered request.
   - The call log isn't released yet (1.1.0 is unpublished). **Decide
     before 1.1.0 goes out**, when changing it breaks no one.
2. **TypeScript's shapes.**
   - *Recommendation:* zod in the documentation, since TypeScript
     developers already know it; zod 4 converts to JSON Schema. Also
     lmcc's own `t` builders, for people who want no dependencies.
3. **Improving in other languages.** The algorithms use randomness and
   model calls. The contract could fix only their meaning (which
   examples are eligible, what gets scored). Or it could fix exact
   choices, given a recorded set of replies and a seeded random
   generator that every language implements.
   - *Recommendation:* start with meaning. Move to exact choices only
     if users compare improvement runs across languages.
4. **The R face.** Named arguments (sketched above) or R's formula
   style (`ai(mood ~ review, …)`)? To be decided with R users, when the
   R package starts.
5. **Integrations tied to one ecosystem**, such as Prime Intellect's
   verifiers in `functai_verifiers` and `functai_rows`, stay in their
   language. They are not part of the contract.
