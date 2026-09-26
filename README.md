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

Version 1.0 no longer depends on DSPy. See [Migrating from 0.x](#12-migrating-from-0x).

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
- [10. Inspection](#10-inspection)
- [11. When the model gets it wrong](#11-when-the-model-gets-it-wrong)
- [12. Migrating from 0.x](#12-migrating-from-0x)
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
- A bad value (an unknown layout name, a template lmcc cannot read, a DSPy
  adapter, a connection object where a model name belongs) is refused where you
  write it, before anything changes.

**The model and the connection are separate.** `lm=` is the model's name.
`client=` is how to reach it, when you build that yourself with lm15: a
router, or one provider's LM, which does not know which model to use.

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
`"ollama:qwen3:8b"`. The litellm/DSPy spelling `"openai/gpt-4o"`,
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
| `"chat"` | DSPy's `[[ ## name ## ]]` sections, ending with `[[ ## completed ## ]]` |
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

    [Tool executing: Calculating '(15 * 23) + 10']
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
ev.write("runs/today.parquet")
```

The interval matters: on 30 examples, 80% means "somewhere between 63% and
90%". **A metric** is `metric(row, prediction) -> float | bool` (the row as
a dict, the prediction with attribute access), or a dpyr expression over the
table's columns; give several as a list or a dict. The default is exact
match (case and spacing ignored) when the data has a column named like an
output. A metric can itself be an AI function:

```python
@ai
def judge(row, prediction) -> float:
    """Between 0 and 1: how close the prediction is to the row's result."""

ev = evaluate(translator, dev, {
    "exact": col.pred_result == col.result,        # computed on the whole table at once
    "judge": judge,                                # one model call per row
    "short": lambda row, pred: len(pred.result) < 80,
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

reviews = read("reviews.parquet")
(reviews
    .mutate(topic=classify(col.text),
            reply=answer(col.question, context=col.doc, tone="formal"))
    .filter(is_complaint(col.text))
    .group_by(col.topic)
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
| `InstructionSearch(metric, num_candidates=6, num_trials=12, prompt_lm=None)` | proposed instructions × demo sets, searched on minibatches, finalists scored on `valset` (MIPRO-style; random/greedy search, not Bayesian) |

```python
from functai import InstructionSearch

translator.opt(trainset=trainset, metric=judge,
               optimizer=InstructionSearch, num_candidates=3, num_trials=5,
               max_bootstrapped_demos=0, max_labeled_demos=0, prompt_lm="gpt-4.1-mini")
```

On a 5-example Québécois-French task with `gpt-4.1-nano` and the `judge`
above, this took the score from 0% to 80% by rewriting the instruction
(with 5 examples, an interval that wide is worth checking on more data).
The search's history is kept as rows: `dpyr.read(opt.trials)` (or
`opt.candidates` for the random search).

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

@module
def research_hop(claim: str, hops: int = 2):
    key_facts: list[str] = []
    for i in range(hops):
        query = generate_query(claim, key_facts)
        key_facts = append_notes(claim, key_facts, search(query))
    return key_facts

research_hop.opt(trainset=trainset, metric=metric, call_defaults=dict(hops=2))
```

The metric sees `Prediction(result=<what the module returned>)` (in the
table: `pred_result`), and the table's tokens add up every inner call. Bootstrapping
records every inner call; a run the metric accepts gives a demo to each AI
function it went through.

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

## 12. Migrating from 0.x

1.0 keeps the API (`@ai`, `_ai`, `configure`, `all=True`, `stateful`,
`tools`, `module="cot"`, `.opt`, `undo_opt`, `@module`, `phistory`, the
docments utilities) and replaces DSPy underneath.

| 0.x | 1.0 |
|---|---|
| `configure(lm=dspy.LM("openai/gpt-4.1"))` | `configure(lm="gpt-4.1")` (litellm strings still work; a DSPy LM's `.model` is read) |
| `dspy.Example(...).with_inputs(...)` | a dict per row, or a table: columns named like the parameters are the inputs |
| `metric(example, pred, trace=None)` | `metric(row, prediction)`, or a dpyr expression |
| `optimizer=dspy.BootstrapFewShot` / `dspy.MIPROv2` | `functai.BootstrapFewShot` / `functai.InstructionSearch` |
| `dspy.Evaluate(...)` → a percentage | `functai.evaluate(program, data, metric)` → `.score` (0 to 1, with an interval), `.table` |
| `adapter="json"`, `adapter="chat"` | same names, now lmcc layouts |
| custom DSPy adapter classes | `template=[system(...), turns(), user(...)]` or an `lmcc.Adapter` |
| tools switch the program to `dspy.ReAct` | tools run in a tool loop; the prompt does not change |
| `fn.signature` (a DSPy Signature) | an lmcc `SignatureCore` |
| `fn.to_dspy()` | removed; `fn.state()` / `fn.save(path)` |
| `stateful` history in `dspy.History` | lmcc turns in `fn.history` |

Behavior changes:

- **Automatic instruction writing is opt-in.** In 0.x every new function
  asked the model to rewrite its own instruction (`autoinstruct`), and the
  first calls refined it again. That spent money at import time and made
  prompts change by themselves. Now `@ai(autoinstruct=True)` or
  `@ai(instruction_autorefine_calls=2)` turns them on; they run at the first
  call, not at definition.
- Prompts are laid out by lmcc, so their text differs from DSPy's.
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
