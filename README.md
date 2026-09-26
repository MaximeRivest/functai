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
pip install functai          # Python 3.11+
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
| `router` | any lm15 router, for full control |
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

An optimizer tunes what the function sends besides its inputs: the
**instruction** and the **demos** (worked examples). It never edits your
code, types or layout. Optimization happens in place; `undo_opt()` reverts.

```python
from functai import ai, _ai, Example, evaluate, exact_match

@ai
def classify_intent(user_query: str) -> str:
    """Classify user intent as 'booking', 'cancelation', or 'information'."""
    return _ai

trainset = [
    Example(user_query="I need to reserve a room.", result="booking").with_inputs("user_query"),
    Example(user_query="How do I get there?", result="information").with_inputs("user_query"),
    Example(user_query="I want to cancel my reservation.", result="cancelation").with_inputs("user_query"),
]

evaluate(classify_intent, trainset, exact_match, num_threads=4)   # EvaluationResult(score=..., n=3)
classify_intent.opt(trainset=trainset)                           # BootstrapFewShot by default
classify_intent.undo_opt()
```

Training data can be `Example`s, dicts (`{"user_query": ..., "result": ...}`),
`(inputs, outputs)` pairs, or DSPy `Example`s. A metric is
`metric(example, prediction[, trace]) -> float | bool`; it can itself be an
AI function:

```python
@ai
def judge(example, prediction) -> float:
    """Between 0 and 1: how close the prediction is to the example's result."""
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
above, this took the score from 0% to 80% by rewriting the instruction.

More:

- `teacher_lm="gpt-4.1"` (or `teacher=`): a stronger model produces the demos.
- `n_synth=20` with a teacher: synthesize training examples first.
- `fn.state()`, `fn.instructions`, `fn.demos`: what is in use; `fn.programs()`:
  every state optimization produced; `fn.optimization_runs()`: the log.
- `fn.save("f.json")` / `fn.load("f.json")`: the instruction and demos as JSON.
- `@ai(examples=[("I love it", "positive"), ...])`: demos by hand.

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

The metric sees `Prediction(result=<what the module returned>)`. Bootstrapping
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
| `dspy.Example(...)` | `functai.Example(...)` (DSPy Examples are still accepted as data) |
| `optimizer=dspy.BootstrapFewShot` / `dspy.MIPROv2` | `functai.BootstrapFewShot` / `functai.InstructionSearch` |
| `dspy.Evaluate(...)` | `functai.evaluate(...)` or `functai.Evaluate(...)` |
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
