# FunctAI: the function is the prompt

FunctAI turns typed Python functions into calls to a language model.

> **The function definition *is* the prompt, and the function body *is* the program.**

Docstrings are instructions, type hints are the output contract, and
variables assigned from `_ai` are extra outputs (chain of thought, several
answers). Behind the decorator, FunctAI stands on two small libraries:

| layer | question it answers | library |
|---|---|---|
| wire | what bytes go to which provider | [**lm15**](https://github.com/lm15-dev/lm15-python) (OpenAI, Anthropic, Gemini, Groq, OpenRouter, Ollama, … no SDKs) |
| layout | how each value is written into the prompt and read back | [**lmcc**](https://github.com/MaximeRivest/lmcc) (templates, typed readers, reasoning, tools) |
| program | your function, its tools, memory, evaluation and optimization | **functai** |

New here? The [tutorial](docs/tutorial.md) builds a small program step by
step, with real outputs; the [examples](examples/) go deeper on one topic
each. This README is the reference.

- [1. Getting started](#1-getting-started)
- [2. Core concepts](#2-core-concepts)
- [3. Types are the contract](#3-types-are-the-contract)
- [4. Configuration](#4-configuration)
- [4b. Signing in: subscriptions and keys](#4b-signing-in-subscriptions-and-keys)
- [5. Chat templates and layouts](#5-chat-templates-and-layouts)
- [6. Reasoning, several outputs, tools](#6-reasoning-several-outputs-tools)
- [7. Memory](#7-memory)
- [8. Evaluation and optimization](#8-evaluation-and-optimization)
- [9. Modules: programs of several AI functions](#9-modules-programs-of-several-ai-functions)
- [9b. Saving a program with its dependencies](#9b-saving-a-program-with-its-dependencies)
- [9c. Baking a function into weights you own](#9c-baking-a-function-into-weights-you-own)
- [10. Inspection](#10-inspection)
- [11. When the model gets it wrong](#11-when-the-model-gets-it-wrong)
- [12. Upgrading from 0.x](#12-upgrading-from-0x)
- [13. A real pipeline](#13-a-real-pipeline)

------------------------------------------------------------------------

## 1. Getting started

```bash
pip install functai            # Python 3.11+
pip install "functai[data]"    # + result tables for evaluation (dpyr: polars and duckdb)
```

Keys come from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, `GROQ_API_KEY`, …). Or sign in once with your Claude,
ChatGPT, GitHub Copilot or xAI subscription (§4b):

<!-- skip: opens a browser to sign in -->
```python
import functai
functai.login("claude")
```

```python
from functai import ai, _ai, configure

configure(lm="gpt-4.1-mini", temperature=0)

@ai
def summarize(text: str, focus: str = "key points") -> str:
    """Summarize the text in one concise sentence,
    concentrating on the specified focus area."""
    return _ai

summarize("FunctAI bridges the gap between Python's expressive syntax and the dynamic "
          "capabilities of LLMs. It allows developers to focus on logic rather than boilerplate.",
          focus="developer benefits")
```

    "FunctAI benefits developers by enabling them to concentrate on logic instead of boilerplate code through its integration of Python's syntax with LLM capabilities."

When you call `summarize`, FunctAI builds the signature (inputs `text`,
`focus`; output `result: str`; the docstring as instruction), lays the call
out with lmcc, sends it with lm15, reads the reply back into a `str`, and
returns it.

## 2. Core concepts

### The `@ai` decorator

`@ai` reads the function's parameters, return type, docstring and body.
A body that is only a docstring, `...`, or `return _ai` means “the model's
answer is the return value”.

```python
@ai
def sentiment(text: str) -> str:
    """Analyze the sentiment. Return 'positive', 'negative', or 'neutral'."""
```

### The `_ai` sentinel

`_ai` stands for the model's output inside the body. It behaves like the
value it stands for, so you can post-process it with plain Python:

```python
@ai
def sentiment_score(text: str) -> float:
    """Returns a sentiment score between 0.0 (negative) and 1.0 (positive)."""
    score = _ai
    return max(0.0, min(1.0, float(score)))

sentiment_score("I think that FunctAI is amazing!")
```

    0.9

The variable's name becomes the output's name (`score` here): name
outputs the way you would name them for a colleague.

## 3. Types are the contract

Scalars, lists, dicts, tuples, sets, `Optional`, `Literal`, `Enum`,
dataclasses, TypedDicts and pydantic models all work, as inputs and
outputs. Structured values travel as JSON; the model is shown their schema.

```python
@ai
def get_keywords(article: str) -> list[str]:
    """Extract 5 key terms from the article."""
    keywords: list[str] = _ai
    return [k.lower() for k in keywords]

get_keywords("FunctAI excels at extracting structured data. Python type hints "
             "serve as the contract between your code and the LLM.")
```

    ['functai', 'structured data', 'python type hints', 'contract', 'llm']

```python
from dataclasses import dataclass

@dataclass
class ProductInfo:
    name: str
    price: float
    features: list[str]
    in_stock: bool

@ai
def extract_product(description: str) -> ProductInfo:
    """Extract product information from the description."""
    return _ai

extract_product("iPhone 15 Pro - $999, 5G, titanium design, available now")
```

    ProductInfo(name='iPhone 15 Pro', price=999.0, features=['5G', 'titanium design'], in_stock=True)

A plain class with annotations is made a dataclass for you
(`flexiclass`). Comments document fields:

```python
class Person:
    name: str   # full name, as written
    age: int    # in years
```

Restricted choices:

```python
from enum import Enum
from typing import Literal

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@ai
def classify_priority(issue_description: str) -> TicketPriority:
    """Analyzes the issue and classifies its priority level."""

classify_priority("The main database is unresponsive.")   # <TicketPriority.HIGH: 'high'>

@ai
def categorize(text: str) -> Literal["sport", "fashion"]: ...
```

Comments on parameters and on the return line become guidance in the
instruction:

```python
@ai
def translate(
    text: str,        # English, informal
    register: str,    # "formal" or "casual"
) -> str:             # French, same length as the input
    """Translate the text to French."""
```

## 4. Configuration

### The cascade

Settings are resolved at every call, innermost first:

1. `fn.using(...)`: a copy of the function with other settings
2. the function: `@ai(temperature=0.1)`, or `fn.temperature = 0.1`
3. a block: `with configure(temperature=0.1):` (this thread/context only)
4. process-wide: `configure(temperature=0.1)`

```python
import functai

functai.configure(lm="gpt-4.1-mini", temperature=0.5)

@ai(temperature=0.0, lm="claude-haiku-4-5")      # this function
def legal_analysis(document): ...

with functai.configure(lm="gpt-4.1"):             # this block
    summarize("...")

summarize.using(lm="gemini-2.5-flash")("...")    # this call
```

`functai.settings.lm` reads the effective value.

### Changing a function's model, connection and layout

Everything can be set when the function is defined, changed on it later, or
changed on a copy (`using`), which leaves the original alone:

| | at definition | afterwards | on a copy |
|---|---|---|---|
| model | `@ai(lm="gpt-4.1")` | `fn.lm = "gpt-4.1"` | `fn.using(lm="gpt-4.1")` |
| connection | `@ai(client=...)` | `fn.using(client=...)` | `fn.using(client=...)` |
| layout by name, lmcc adapter, or saved adapter JSON | `@ai(adapter="chat")` | `fn.adapter = my_adapter` | `fn.using(adapter=my_adapter)` |
| chat template | `@ai(template=[...])` | `fn.template = [...]` | `fn.using(template=[...])` |

- An adapter replaces the function's template, and a template replaces its
  adapter. `template=None` goes back to the adapter setting (or the default).
- In `using`, a setting given as `None` is no longer set by the copy: it comes
  from `configure` or the defaults.
- A bad value (an unknown layout name, a template lmcc cannot read, an
  object that is not an lmcc adapter, a connection object where a model name
  belongs) is refused where you write it, before anything changes.

**The model and the connection are separate.** `lm=` is the model's name.
`client=` is how to reach it, when you build that yourself with lm15: a
router, or one provider's LM, which does not know which model to use.

<!-- skip: needs a second OpenAI key and a Claude Code login -->
```python
import lm15

@ai(lm="gpt-4.1-mini", client=lm15.OpenAILM(api_key=OTHER_KEY))
def f(text: str) -> str: ...

f.using(lm="claude:claude-haiku-4-5", client=lm15.ClaudeCodeLM.from_claude_code(credentials_path=...))
```

A model prefix naming another provider than the client's is refused;
a bare model name is sent as written. lm15's `BoundClient` (a login plus one
model) goes in `lm=`, since it carries both; a `client=` set more widely (in
`configure`) does not apply to it, and giving both in one place is refused.
You do not need any of this for the usual cases: model names, `api_key=`,
and `functai.login(...)` cover them.

### Models

Any [lm15](https://github.com/lm15-dev/lm15-python) model string:
`"gpt-4.1-mini"`, `"claude-haiku-4-5"`, `"gemini-2.5-flash"`,
`"groq:openai/gpt-oss-120b"`, `"openrouter:qwen/qwen3-32b"`,
`"ollama:qwen3:8b"`. The litellm spelling `"openai/gpt-4o"`,
`"anthropic/claude-sonnet-4-5"`, `"groq/openai/gpt-oss-120b"` is read the way
lm15 reads it.

| setting | meaning |
|---|---|
| `lm` | the model |
| `api_key`, `base_url` | for the provider `lm` routes to (an explicit key beats everything) |
| `auth` | saved logins: default on; a path for another credentials file; `False` to never use them |
| `client` | the lm15 connection, when you build it yourself (below) |
| `temperature`, `max_tokens`, `seed`, `top_p`, `stop`, … | any lm15 `Config` field |
| `adapter` | the layout (§5); a chat template is given per function, `@ai(template=[...])` |
| `module` | `"predict"`, `"cot"` (§6), `"react"` (tools) |
| `tools`, `max_steps`, `tool_errors` | the tool loop (§6) |
| `stateful`, `state_window` | memory (§7) |
| `retries`, `api_retries`, `cache_replies` | reliability (§11) |
| `capabilities` | what the model can do, when you know better than functai's table |
| `optimizer`, `teacher`, `teacher_lm` | optimization defaults (§8) |
| `debug` | print a line per call |

An unknown setting is an error, not a silent no-op.

## 4b. Signing in: subscriptions and keys

<!-- skip: opens a browser to sign in -->
```python
import functai

functai.login("claude")        # your Claude Pro/Max subscription
functai.login("chatgpt")       # ChatGPT Plus/Pro (the Codex backend)
functai.login("copilot")       # GitHub Copilot
functai.login("grok")          # xAI
functai.login("openrouter")    # approve in the browser; OpenRouter mints a key
functai.login("openai")        # asks for an API key and saves it
functai.login("groq", key="gsk-...")
functai.login()                # asks which
```

Sign in once; every later session (and every tool built on lm15) uses it.
Then name the model with the account's prefix:

<!-- skip: needs those logins -->
```python
functai.configure(lm="claude:claude-sonnet-4-5")
functai.configure(lm="chatgpt:gpt-5.5")
functai.configure(lm="copilot:gpt-4.1")
```

`functai.logins()` shows everything you can use right now, and a model to try:

    provider        how                            status                            try
    ──────────────  ─────────────────────────────  ────────────────────────────────  ────────────────────────
    Claude          saved login                    ready until 2026-09-26 19:02 UTC  claude:claude-sonnet-4-5
    GitHub Copilot  saved login                    ready until 2026-09-27 11:02 UTC  copilot:gpt-4.1
    ChatGPT         Codex CLI                      found                             chatgpt:gpt-5.5
    openai          environment ($OPENAI_API_KEY)  found                             gpt-4.1-mini

How it works, and what to expect:

- **A Claude Code or Codex CLI already signed in on this machine** is used as
  is: no new login, nothing copied. `functai.login("claude")` just records
  that choice.
- **Which credential a call uses**: an explicit `api_key=` first; then the
  saved login for that provider; then the environment and CLI logins. A saved
  key therefore beats the same provider's environment variable.
- **Logins renew themselves.** A saved login that expired and cannot renew
  raises `functai.LoginRequired` with the command to type
  (`functai.login('claude')`); it never switches to a paid key on its own. A
  missing key raises the same error, saying which variable to set.
- `functai.login("claude")` when you are already signed in says so and does
  nothing; `again=True` signs in again. `functai.logout("claude")` forgets the
  saved login; an environment key or CLI login for that provider, if you have
  one, is used again afterwards (except xAI, which lm15 blocks on purpose).
- A browser opens by itself when the machine has a screen; over SSH the link
  (or device code) is printed for you to open anywhere.
- Some sign-ins (GitHub Copilot, Kimi Code, the Claude and ChatGPT browser
  logins) are not yet certified by lm15; functai says so before starting.
  **Whether a provider allows its subscription to be used this way, and how it
  bills it, is the provider's decision.**
- Logins are saved in lm15's credentials file (`~/.config/lm15/credentials.json`,
  or `$LM15_CREDENTIALS_PATH`); `configure(auth="path/to/file.json")` uses
  another one, `configure(auth=False)` none.
- The ChatGPT backend takes no `temperature`, `top_p` or `max_tokens`, and
  OpenAI's reasoning models (o-series, GPT-5) take no `temperature`: functai
  leaves them out of those requests and warns once, so one `configure(...)`
  works across models.

## 5. Chat templates and layouts

By default the model sees the instruction and a reply pattern in the
system message, earlier turns as messages, and the inputs in tags:

    System:  Function: summarize
             Summarize the text in one concise sentence, ...
             Reply in exactly this form:
             <result>
             ...
             </result>
    User:    <text>
             ...
             </text>

To write the conversation yourself, put a chat template in the decorator.
It uses lmcc's template language: `{instruction}`, `{input_name}`, loops over
`inputs` and `outputs`, and `turns()` for where examples and the
conversation so far go.

```python
from functai import ai, system, user, turns, assistant

@ai(template=[
    system("You are a helpful pirate. {instruction}"),
    user("Text: {text}"),
])
def pirate_summarize(text: str) -> str:
    """Summarize in 10 words."""

pirate_summarize("Foundation models are now mature enough to be used in real-world applications.")
```

    'Foundation models mature, now usable in practical, real-world applications.'

With one output and no reply pattern in the template, the whole reply is
the value. With several outputs, spell the pattern; **the same pattern is
the parser**, so the prompt and the reader cannot drift apart:

```python
@ai(template=[
    system("{instruction}\n\nAnswer in this form:\n"
           "{% for f in outputs %}{f.name}: {f.value}\n{% endfor %}"),
    turns(),
    user("Review: {review}"),
])
def rate(review: str) -> int:
    """Rate the review from 1 to 5 stars."""
    verdict: str = _ai["One short sentence."]
    return _ai

dict(rate("Great tacos, loud music. I'll be back.", all=True))
```

    {'verdict': 'Positive and concise review with a clear intention to return.', 'result': 4}

The prompt that was sent:

    System message:

    Function: rate

    Rate the review from 1 to 5 stars.

    Output guidance:
    - verdict: One short sentence.

    Answer in this form:
    verdict: ...
    result: (integer)

    User message:

    Review: Great tacos, loud music. I'll be back.

More in templates:

- `{% if context %}…{% endif %}` shows a block only when an input has a value.
- A last `assistant("<answer>")` is a **prefill**: sent to models that
  continue it, read as the start of the reply either way.
- OpenAI-style dicts work too: `template=[{"role": "system", "content": "..."}, ...]`.
- Without `turns()`, examples and memory go right before the last user message.
- A template that cannot be read back is refused when the function is
  defined or first bound, before any model call.

### Shipped layouts

| `adapter=` | layout |
|---|---|
| `None` / `"xml"` | tagged sections (the default above) |
| `"chat"` | `[[ ## name ## ]]` sections, ending with `[[ ## completed ## ]]` |
| `"json"` | one JSON object the provider enforces with a schema (models with native structured output) |
| an `lmcc.Adapter` | any lmcc adapter, including one loaded from a JSON artifact |

## 6. Reasoning, several outputs, tools

### Chain of thought

Declare the reasoning in the body; it is written before the answer:

```python
@ai
def solve_math_problem(question: str) -> float:
    """Solves a math word problem and returns the numerical answer."""
    reasoning: str = _ai["Step-by-step thinking process to reach the solution."]
    return _ai
```

Or ask for it with `@ai(module="cot")`: models with a thinking channel
(o-series, GPT-5, Claude 4.x, Gemini 2.5+) use it; the others write a
`reasoning` section first. Same program either way.

### Everything, with `all=True`

```python
p = solve_math_problem("If a train travels 120 miles in 2 hours, what is its speed?", all=True)
p.reasoning     # 'To find the speed ... Speed = 120 miles / 2 hours = 60 miles per hour'
p.result        # 60.0
p.usage         # tokens, summed over every model call
p.turn          # the lmcc turn: inputs, every model and tool step, outputs
```

### Several outputs

```python
@ai
def critique_and_improve(text: str) -> tuple[str, str]:
    """Analyze the text, criticize it constructively, and improve it."""
    critique: str = _ai["Constructive criticism focusing on clarity and tone."]
    improved_text: str = _ai["The improved version of the text."]
    return critique, improved_text

critique, improved = critique_and_improve("U should fix this asap, it's broken.")
```

### Tools

Tools are typed Python functions. With tools, a call runs the loop: ask
the model, run the tools it calls, give it the results, until it answers
(at most `max_steps`, default 8).

```python
def search_web(query: str) -> str:
    """Searches the web for information."""
    return f"Mock search results for {query}."

def calculate(expression: str) -> float:
    """Performs mathematical calculations."""
    return eval(expression)   # a demo: never eval untrusted text

@ai(tools=[search_web, calculate])
def research_assistant(question: str) -> str:
    """Answer questions using available tools to gather data and perform calculations."""

research_assistant("What is the result of (15 * 23) + 10?")
```

    'The result of (15 * 23) + 10 is 355.'

Native tool calls where the model has them, fenced text calls otherwise:
the prompt style never changes because you added a tool. A tool that
raises is reported to the model (`tool_errors="raise"` to stop instead).

## 7. Memory

```python
@ai(stateful=True)
def assistant(message):
    """A friendly AI assistant that remembers the conversation history."""

assistant("Hello, my name is Alex.")   # 'Hello Alex! How can I assist you today?'
assistant("What is my name?")          # 'Your name is Alex.'
```

The conversation is kept as lmcc turns in `assistant.history` (the last
`state_window`, default 5) and written through the function's own layout.
`assistant.reset()` forgets it.

## 8. Evaluation and optimization

Data is rows: a list of dicts, or any table. Columns named like the
function's parameters are its inputs; the others are the expected outputs
and whatever else you want to keep (a category, an id). Evaluation results
are a table too, so the whole Python data stack applies to them: filter the
failures, group by category, join two runs, save to parquet. Tables come
from [dpyr](https://github.com/MaximeRivest/dpyr) (dplyr verbs over polars
and duckdb): `pip install "functai[data]"`.

```python
from functai import ai, _ai, evaluate
from dpyr import col

@ai
def classify_intent(user_query: str) -> str:
    """Classify user intent as 'booking', 'cancelation', or 'information'."""
    return _ai

dev = [
    {"user_query": "I need to reserve a room.", "result": "booking", "lang": "en"},
    {"user_query": "How do I get there?", "result": "information", "lang": "en"},
    {"user_query": "Annuler ma réservation.", "result": "cancelation", "lang": "fr"},
]
# or: dev = "dev.parquet", a pandas/polars dataframe, a Hugging Face dataset, ...

ev = evaluate(classify_intent, dev, num_threads=8)
ev                  # Evaluation(classify_intent, 3 examples: exact_match 0.67 [0.21, 0.94])
ev.score            # 0.67, the first metric's mean
ev.summary          # one row per metric: mean, 95% interval (low, high), n, failed
ev.table            # one row per example: the data, pred_result, exact_match, error,
                    # seconds, input_tokens, output_tokens, model, run

ev.table.filter(col.exact_match == 0)                                  # read the misses
ev.table.group_by(col.lang).summarize(acc=col.exact_match.mean())      # accuracy by language
ev.write("today.parquet")
```

The interval matters: on 30 examples, 80% means "somewhere between 63% and
90%". **A metric** is `metric(row, prediction) -> float | bool` (the row as
a dict, the prediction with attribute access), or a dpyr expression over the
table's columns; give several as a list or a dict. The default is exact
match (case and spacing ignored) when the data has a column named like an
output. A metric can itself be an AI function:

```python
@ai
def judge(row: dict, prediction: dict) -> float:
    """Between 0 and 1: how well the prediction matches the row's result."""

ev = evaluate(classify_intent, dev, {
    "exact": col.pred_result == col.result,        # computed on the whole table at once
    "judge": judge,                                # one model call per row
    "short": lambda row, pred: len(pred.result) < 20,
})
```

A row whose run fails (a provider error, an unreadable reply) keeps its
message in `error`; its metrics are null in the table and count 0 in the
score. A missing input column, an unknown metric signature or a column name
the table would reuse is refused before any model is called.

**Comparing two versions** of a prompt pairs the examples, which detects a
real change with far fewer examples than two separate scores would:

```python
from functai import compare

train = [
    {"user_query": "Book me a double for Friday.", "result": "booking"},
    {"user_query": "Is parking included?", "result": "information"},
    {"user_query": "Please cancel booking #4411.", "result": "cancelation"},
    {"user_query": "Je voudrais réserver une chambre.", "result": "booking"},
]

before = evaluate(classify_intent, dev)
classify_intent.opt(trainset=train)
after = evaluate(classify_intent, dev)
compare(before, after)   # per metric: before, after, diff with its 95% interval, better/worse/same
```

`evaluate(..., log="runs/")` writes each run to `runs/<run>.parquet`;
`functai.runs("runs/")` reads them all back as one table (columns lined up by
name).

**AI functions on columns.** Called with a dpyr column instead of a value,
an AI function is a column expression, usable in `mutate()` and `filter()`
like any other; columns and constants mix freely:

```python
from dpyr import read, col, n

@ai
def reply(user_query: str, tone: str) -> str:
    """A one-sentence reply to the guest, in the given tone."""

@ai
def is_complaint(user_query: str) -> bool:
    """Is the guest unhappy about something?"""

inbox = read([                                  # or read("inbox.parquet"), a dataframe, ...
    {"user_query": "Book me a room for two nights."},
    {"user_query": "The room was dirty and nobody answered the phone!"},
    {"user_query": "Cancel my stay, the noise was unbearable."},
])
(inbox
    .mutate(intent=classify_intent(col.user_query),
            answer=reply(col.user_query, tone="formal"))
    .filter(is_complaint(col.user_query))
    .group_by(col.intent)
    .summarize(n=n()))
```

Nothing runs when the line is written, but the column's type (the return
annotation; text when there is none) is checked. Each distinct input is
sent to the model once, 8 at a time, and the answers are remembered for the
session; a displayed dataframe only asks for the rows it shows. A row that
fails raises after every row ran, and running again retries only the
failures. The column uses the prompt the function had when the line was
written, so optimizing it later never mixes old and new answers. Options:
`classify.vectorize(threads=16, errors="null")(col.text)`. `fn.map(table)`
returns the whole run table instead (tokens, errors, timing per row).

**Optimization** tunes what the function sends besides its inputs: the
**instruction** and the **demos** (worked examples). It never edits your
code, types or layout. It happens in place; `undo_opt()` reverts.

```python
classify_intent.opt(trainset=train)    # BootstrapFewShot by default; any metric above works
classify_intent.undo_opt()
```

| optimizer | what it does |
|---|---|
| `LabeledFewShot(k=16)` | labeled examples as demos |
| `BootstrapFewShot(metric, max_bootstrapped_demos=4, max_labeled_demos=16, teacher=None)` | runs the program on examples; the runs the metric accepts become demos, whole turns included (reasoning, tool calls) |
| `BootstrapFewShotWithRandomSearch(metric, num_candidate_programs=8)` | many demo sets, keeps the best on `valset` |
| `InstructionSearch(metric, num_candidates=6, num_trials=12, prompt_lm=None)` | instructions proposed by a model × demo sets, tried on minibatches, finalists scored on `valset` (random then greedy search) |

```python
from functai import InstructionSearch

opt = InstructionSearch(num_candidates=3, num_trials=5, max_bootstrapped_demos=0,
                        max_labeled_demos=0, prompt_lm="gpt-4.1-mini")
classify_intent.opt(trainset=train, metric=judge, optimizer=opt)
read(opt.trials)       # every trial as a row: which instruction and demos, and its score
```

The [translator example](examples/optimizing_translator/) runs this on
English to Québécois French, with an AI judge, before and after.

More:

- `teacher_lm="gpt-4.1"` (or `teacher=`): a stronger model produces the demos.
- `n_synth=20` with a teacher: synthesize training examples first.
- `fn.state()`, `fn.instructions`, `fn.demos`: what is in use; `fn.programs()`:
  every state optimization produced; `fn.optimization_runs()`: the log.
- `fn.save("f.json")` / `fn.load("f.json")`: the instruction and demos as JSON.
- `@ai(examples=[("I love it", "positive"), ...])`: demos by hand (pairs, or
  rows like `{"text": "I love it", "result": "positive"}`).

## 9. Modules: programs of several AI functions

```python
from functai import ai, module

@ai
def generate_query(claim: str, key_facts: list[str]) -> str:
    """Produce a follow-up search query from a claim and current key facts."""

@ai
def append_notes(claim: str, key_facts: list[str], new_docs: list[str]) -> list[str]:
    """Extend key facts with new learnings extracted from new_docs."""

def search(query: str) -> list[str]:
    """Your retriever: a vector store, a search API, ..."""
    return [f"A document about {query}."]

@module
def research_hop(claim: str, hops: int = 2) -> list[str]:
    key_facts: list[str] = []
    for i in range(hops):
        query = generate_query(claim, key_facts)
        key_facts = append_notes(claim, key_facts, search(query))
    return key_facts

def found_something(row, prediction):
    return len(prediction.result) > 0

claims = [{"claim": "The Eiffel Tower is in Paris."}, {"claim": "K2 is in Nepal."}]
research_hop.opt(trainset=claims, metric=found_something, call_defaults=dict(hops=1))
```

The metric sees `Prediction(result=<what the module returned>)` (in the
table: `pred_result`), and the table's tokens add up every inner call. Bootstrapping
records every inner call; a run the metric accepts gives a demo to each AI
function it went through.

## 9b. Saving a program with its dependencies

A functai program is code plus a contract, like a Spark UDF: its inputs and
outputs are typed, and everything it depends on must be known to ship it.
functai reads the code to find all of it.

<!-- skip: fact_check is a program spread over several files -->
```python
import functai

functai.check(fact_check)
```

    fact_check  @module  [prog]
    ├── generate_query  AI function (claim: str → str)  [prog]
    │   └── tool search  function  [prog]
    │       ├── clean  function  [helpers]
    │       │   ├── SPACES = __import__('re').compile('\\s+')
    │       │   └── file data/stop.txt
    │       └── json  (stdlib)
    ├── judge  AI function (evidence: Evidence → Verdict)  [prog]
    │   ├── Evidence  class  [kinds]
    │   │   └── dataclasses  (stdlib)
    │   └── Verdict  class  [kinds]
    │       └── enum  (stdlib)
    ├── Evidence  (see above)
    ├── search  (see above)
    ├── Verdict  (see above)
    └── DEFAULT = Verdict.FALSE

    requirements: functai==1.0.0
    no problems: ready to save

`check` follows every name the code reaches: AI functions and `@module`s
(also called under another name, or through helper functions), their tools,
your functions and classes (in files or notebook cells), the types in the
signatures, constants, and data files read with `functai.file("...")`. Each
dependency is one of:

| found | saved as |
|---|---|
| an AI function or `@module` | its code, settings, instruction and demos; followed the same way |
| a function or class from your code | its source, verbatim |
| a module or name from an installed package | a pinned requirement |
| the standard library | nothing |
| a constant (numbers, text, lists, dicts, Enum members, compiled patterns, lmcc adapters) | its value |
| a file read with `functai.file("data/x.txt")` | a copy |

And what stops a clean save, each with its fix:

| problem | example | fix |
|---|---|---|
| `hidden-state` | a tool writes `CACHE[q] = ...` into a global | pass it in, return it, or `save(allow=["hidden-state"])` to save its current value |
| `untyped-input`, `untyped-output` | `def f(text):`, a `@module` with no return type | annotate it |
| `unsaveable-value` | a global client, lock or open file | create it inside the function, or pass it in |
| `lambda`, `no-source`, `name-conflict` | a lambda tool; two nested `def f` | a named `def`; distinct names |
| `local-import-inside` | `import helpers` inside a function | import it at the top of the module |
| warnings | `getattr(module, name)`, `eval`, a `Path` global, an `api_key` or `client=` (never saved) | reported; `verify` catches what they hide |

### Save, verify, load

<!-- skip: fact_check is a program spread over several files -->
```python
functai.save(fact_check, "fact_check/", record=[{"claim": "Paris is the capital of France"}])
functai.verify("fact_check/", trust=True)        # verified in a fresh environment
fact_check = functai.load("fact_check/", trust=True)
```

The saved folder is readable and diffable:

    fact_check/
      functai.json          entry; each AI function's settings, instruction, demos,
                            signature and fingerprints; versions; file hashes
      code/prog.py          the code the program reaches, one file per module
      code/helpers.py       (a notebook or script becomes code/main.py): functions
      code/kinds.py         and classes verbatim, constants by value, imports
      files/data/stop.txt   data files read with functai.file(...)
      requirements.txt      the packages the code reaches, pinned
      requirements.lock     those and everything they pull in, as installed here
      recordings.json       model replies recorded with save(record=...)

- **`save`** refuses while `check` finds errors, and writes the folder whole or
  not at all. Credentials and connections (`api_key=`, `client=`, logins) are
  never saved: the loading machine's own are used.
- **`verify`** is the proof: it builds a new environment with `uv` from
  `requirements.lock` alone, loads the program there from an empty folder (so
  nothing from your project can leak in), and checks two things. First, every
  AI function renders byte-identical requests (instruction, layout, demos,
  tools). Second, each recording replays to the same result against its
  recorded model replies, tools and helpers included. No model is called;
  it takes about a second once uv's cache is warm. `fresh=False` checks in the
  current environment instead (weaker).
- **`load`** checks before it runs anything: file hashes (catching accidental
  edits), missing packages, and afterwards that every AI function still renders
  the requests it rendered when saved; `check_env="warn"` loads anyway.
- **`trust=True`** is required by `load` and `verify` because they run the saved
  Python code. The hashes catch accidents, not an attacker who edits both the
  code and `functai.json`.

Prompt layout (`adapter`, `module`, `include_fn_name_in_instructions`) is saved
with its effective value, even when it came from `configure()`, because it is
part of what the program means. Model and sampling settings are saved only when
the function sets them itself; otherwise the loading program uses `configure()`.

What reading code cannot see: names looked up at run time (`getattr`,
`importlib`, `eval`), functions passed in as arguments, and data files not read
through `functai.file`. `check` points at them, `@ai(requires=["numpy>=2"])` or
`save(requires=[...])` declares packages by hand, `save(include=["myproject"])`
saves an editable-installed project as code, and `verify` with recordings
catches anything still missing.

## 9c. Baking a function into weights you own

A function answered by a big model can be *baked*: a small model is trained
to answer it, and the same function then runs on those weights. The function
does not change; what executes it does. Needs `pip install "functai[bake]"`.

```python
from typing import Literal
from functai import ai

@ai
def intent(text: str) -> Literal["card_arrival", "card_delivery_estimate", ...]:   # 77 intents
    """The customer's intent."""

baked = intent.bake(rows, student="jhu-clsp/ettin-encoder-17m")    # rows carry a "result" label
print(baked.report)
fast = intent.using(lm=baked)
fast("my card still hasn't arrived")            # 'card_arrival'
fast("...", all=True).probabilities             # {'result': {'card_arrival': 0.93, ...}}
```

    Baked intent: jhu-clsp/ettin-encoder-17m (16.9M parameters)
      trained on 8,994 rows (the data's labels), validated on 999, tested on 3,076 labeled rows
      training: 6 passes (best 5), 31 s on cuda:1 (bf16), inputs up to 56 tokens

      on the test rows            student
      accuracy                    90.8% (89.8%–91.8%)
      top-3                       97.1%
      calibration error (ECE)     0.011 (was 0.048; temperature 1.58)

      answering only when sure:  most confident share → accuracy (confidence at the cut)
          50% → 99.6%  (≥ 0.99)
          80% → 98.0%  (≥ 0.89)
          90% → 95.9%  (≥ 0.65)
         100% → 90.8%  (≥ 0.17)
        for 95% accuracy: escalate_below=0.58 keeps 92% of rows

      speed on cuda:1: 18,707 rows/s batched (tokenizing included), 3.7 ms for one row

That is banking77, all of it measured: it reproduces the Ettin-17M result of
the September 2026 experiments (91.5% recorded, 97.1% top-3, ECE 0.011).

**Two kinds of student.**

| | `method="head"` (default) | `method="sft"` |
|---|---|---|
| for | outputs with a fixed set of answers: `Literal`, `Enum`, `bool`, a dataclass of them | any output: text, numbers, structures |
| model | an encoder (Ettin, ModernBERT) or a decoder with a new answer layer | a small chat model (Qwen3.5-0.8B) |
| reads | the input alone, no prompt | the function's full prompt, in its lmcc layout |
| gives | a probability for every answer, calibrated | the reply, read back by lmcc |
| speed (one 3090) | 16,000–43,000 rows/s | 12 rows/s in-process, 82 rows/s with `baked.serve()` (vLLM) |

**Labels.** A column named like an output is a label; a `<output>__probs`
column (`{answer: probability}`) is a soft label. Or a teacher labels the
rows: `teacher="jev-latest"` (Jev measures a probability for every answer, so
the student learns from the full distribution), or any model or AI function
(its answers). `labels="teacher"` relabels every training row, keeping the
data's labels for testing, which is how to measure what a teacher is worth:

    on the test rows            student                 teacher (jev-latest)
    accuracy                    75.2% (71.2%–78.8%)     80.4%
    teacher labels: 2,000 rows from jev-latest in 6 s (1,852 tokens a row, $0.09)
    break-even in money: after 2,016 rows
    note: result: the student (75.2%) is below its teacher (80.4%); trained on teacher labels, it can
          at best match it. Human labels lifted the same kind of student from 77% to 91.5% ...

The report always says which kind of labels the numbers rest on, and says it
loudly when a student is capped by its teacher: the strongest finding of those
experiments was that a small model trained on human labels (91.5%) beat every
teacher's labels (77–82%).

**Escalation.** The confidence is what the model measured, so unsure answers
can go to a bigger model:

```python
cut = baked.report.threshold(0.95)["threshold"]
safe = intent.using(lm=baked, escalate_to="claude-opus-5.5", escalate_below=cut)
p = safe("...", all=True)
p.escalated, p.first.confidence       # True, 0.41 when Opus answered
```

On 500 banking77 test questions, the Jev-taught student alone scored 75.2%;
escalating its unsure two thirds to Claude Opus 5.5 scored 90.2%, against 92%
for Opus on everything. `escalate_to` takes a model name, a baked model, or an
AI function (which follows its own `escalate_to`, never a global one).

**Details that matter.**

- The input a head reads is written by lmcc (the inputs alone: one bare, several
  in tags), and the model keeps that layout: whatever the function's template
  says later, the baked model reads what it was trained on. A generative
  student keeps its training layout the same way.
- A function whose inputs, outputs or answers changed since baking is refused.
- Training follows what measured best: the whole model, AdamW, warmup then
  cosine, early stopping on held-out rows, a temperature fitted on held-out
  human labels when there are some. Defaults per model size; all overridable.
- The GPU with the most free memory is used when it has room; memory held by
  other programs is never taken. A chat model that does not fit is trained as
  a LoRA adapter, merged into the saved weights.
- Concurrent calls to a head are batched automatically (1,500 calls/s through
  functai from threads, on a laptop CPU with a 4M model); `baked.predict(rows)`
  is the fast path for tables.
- A baked model is a folder (`baked.json`, the weights, the tokenizer, file
  hashes, the report). `functai.bake.load(folder)`; `baked.save(folder)`.
- `functai.save(program)` copies a program's baked weights into `models/`,
  pins torch and transformers, and fingerprints what the weights answer;
  `functai.verify` installs the lock in a fresh environment and checks the
  weights answer the same there (13 s for the banking77 cascade).

**Prime Intellect.** For reinforcement learning or distilling on Prime's
hosted training, `functai.bake.prime.env_package(fn, rows, folder, name=...)`
writes a verifiers environment holding the saved program and its rows, and
`functai.bake.prime.config(fn, env=..., model=..., loss="rl" | "sft", teacher=...)`
writes the `prime train` TOML. Rollouts run through the `functai-verifiers`
harness (installed with `functai[prime]`): the model is called with the
function's lmcc layout, past replies are replayed verbatim so a multi-turn
rollout stays one training sample, unreadable replies are recorded with no
values, and a thinking teacher's reasoning is dropped before the trace.
`functai-rows` is a ready taskset for a function and a table:

    vf-eval functai-rows --env.taskset.program mytask:classify --env.taskset.data rows.jsonl \
        --env.agent.harness.id functai-verifiers --env.agent.harness.program mytask:classify

## 10. Inspection

```python
import functai

print(functai.phistory())          # the last call: every message sent, and the reply
functai.inspect_history(3)         # the last 3 as records (lm15 Request and Response)
summarize.render("some text")      # the exact lm15 request, without sending it
print(summarize.explain())         # the layout: reader, transports, formats
functai.signature_text(summarize)  # 'Signature: summarize | Inputs: text:str, focus:str | Outputs: result*'
summarize.signature                # the lmcc signature
```

## 11. When the model gets it wrong

- **Misspelled layout** (`<Result>` for `<result>`, `**answer**`): read
  anyway, by one rule, and reported in `prediction.repairs`.
- **Unreadable reply**: asked again once with the reader's hint (`retries=1`;
  `retries=0` raises `lmcc.Refusal` with `.code` and `.hint`). A reply cut
  at the token limit is re-sent with twice the budget.
- **Transient provider errors** (rate limit, 5xx, timeout): re-sent with
  backoff (`api_retries=3`).
- **Reply cache (off by default):** with `cache_replies=True`, an identical
  request is answered from memory, so re-running a notebook cell or an
  evaluation costs nothing. `functai.clear_cache()` empties it.
- **Impossible layouts** (several outputs in a template with no pattern, a
  JSON layout on a model without structured output) are refused before any
  request is sent.

## 12. Upgrading from 0.x

1.0 keeps the API (`@ai`, `_ai`, `configure`, `all=True`, `stateful`,
`tools`, `module="cot"`, `.opt`, `undo_opt`, `@module`, `phistory`, the
docments utilities); what runs underneath is new (lmcc and lm15), and so
are data, metrics and optimizers.

| 0.x | 1.0 |
|---|---|
| `configure(lm=<an LM object>)` | `configure(lm="gpt-4.1")`: a model name (litellm spellings work) |
| training data as `Example(...).with_inputs(...)` | a dict per row, or a table: columns named like the parameters are the inputs |
| `metric(example, pred, trace=None)` | `metric(row, prediction)`, or a dpyr expression |
| optimizer classes from other libraries | `functai.BootstrapFewShot`, `functai.InstructionSearch`, ... |
| an evaluator returning a percentage | `functai.evaluate(program, data, metric)`: `.score` (0 to 1, with an interval), `.table` |
| `adapter="json"`, `adapter="chat"` | same names, now lmcc layouts |
| custom adapter classes | `template=[system(...), turns(), user(...)]` or an `lmcc.Adapter` |
| tools switched the program to an agent module | tools run in a tool loop; the prompt does not change |
| `fn.signature` | an lmcc `SignatureCore` |
| exporting the program to another framework | removed; `fn.state()` / `fn.save(path)` |
| `stateful` history in a separate object | lmcc turns in `fn.history` |

Behavior changes:

- **Automatic instruction writing is opt-in.** In 0.x every new function
  asked the model to rewrite its own instruction (`autoinstruct`), and the
  first calls refined it again. That spent money at import time and made
  prompts change by themselves. Now `@ai(autoinstruct=True)` or
  `@ai(instruction_autorefine_calls=2)` turns them on; they run at the first
  call, not at definition.
- Prompts are laid out by lmcc, so their text differs from 0.x.
- Unknown settings raise instead of being ignored.

## 13. A real pipeline

```python
from dataclasses import dataclass
from functai import ai, _ai, configure

configure(lm="gpt-4.1-mini", temperature=0.0)

@dataclass
class Invoice:
    invoice_number: str
    vendor_name: str
    total: float
    items: list[str]

@ai
def extract_invoice(document_text: str) -> Invoice:
    """Extract invoice information from the document text.
    Parse all relevant fields accurately. Convert amounts to float."""
    thought_process: str = _ai["Where each field is in the document."]
    return _ai

@ai
def validate_invoice(invoice: Invoice) -> bool:
    """Is the invoice complete and reasonable? The total must be positive."""

@ai
def summarize_invoice(invoice: Invoice) -> str:
    """Create a brief, human-readable summary of the invoice."""

document = """
INVOICE
Vendor: TechCorp Inc.
Invoice #: INV-2025-101
Items: 5x Laptops, 2x Monitors
Total: $5600.00
"""

invoice = extract_invoice(document)
# Invoice(invoice_number='INV-2025-101', vendor_name='TechCorp Inc.', total=5600.0,
#         items=['5x Laptops', '2x Monitors'])
if validate_invoice(invoice):
    print(summarize_invoice(invoice))
```

------------------------------------------------------------------------

Development: `uv sync`, then `uv run pytest` (offline, a fake provider).
Live checks against real models: `tests/live.py` (costs cents).
