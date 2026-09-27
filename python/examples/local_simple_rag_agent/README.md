---
rat:
  python:
    dependencies: ["-e .[data]"]
---

# A research agent: tools that read the web, and a fact checker


An agent that answers a question by reading a web page, and a second AI
function that checks the answer against its source. It runs on any
model; at the end, the same agent on a local model.

Every output below is a real reply. This page is a notebook: open it in
Chattering and run it, or run it all with `python/.venv/bin/python tools/docs.py run python/examples/local_simple_rag_agent/README.md`.

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)

from functai import ai, _ai
```

## A tool that reads a page

Any typed Python function with a docstring is a tool. This one fetches a
page and keeps its text:

```python
import html.parser
import urllib.request

class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0
    def handle_starttag(self, tag, attrs):
        self._skip += tag in ("script", "style")
    def handle_endtag(self, tag):
        self._skip -= tag in ("script", "style")
    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())

def read_page(url: str) -> str:
    """The text of a web page (at most 15,000 characters)."""
    req = urllib.request.Request(url, headers={"User-Agent": "functai-example"})
    with urllib.request.urlopen(req, timeout=20) as page:
        parser = _Text()
        parser.feed(page.read().decode("utf-8", "replace"))
    return " ".join(parser.parts)[:15000]
```

## The agent

The agent reads pages with the tool, and returns its answer together
with the sentence that supports it:

```python
from dataclasses import dataclass

@dataclass
class Sourced:
    answer: str
    quote: str   # the sentence from the page that supports the answer, verbatim

@ai(tools=[read_page])
def research(question: str) -> Sourced:
    """Answer the question from the pages you read."""
    ...

found = research(
    "Which castle did the physician David Gregory inherit? "
    "Look it up on https://en.wikipedia.org/wiki/David_Gregory_(physician)"
)
found
```

```output
Sourced(answer='David Gregory inherited Kinnairdy Castle.', quote='He inherited Kinnairdy Castle in 1664.')
```

## Checking the answer

A second AI function checks the claim against the quote, reasoning
first:

```python
@ai
def fact_check(claim: str, passage: str) -> bool:
    """Does the passage support the claim?"""
    reasoning: str = _ai["What in the passage supports or contradicts the claim."]
    return _ai
```

A step that must always happen belongs in code, not in a request to the
model. A `@module` is plain Python that calls AI functions, and is
evaluated, optimized and saved as one program:

```python
from functai import module

@module
def checked_answer(question: str) -> str:
    found = research(question)
    if fact_check(found.answer, found.quote):
        return found.answer
    return f"Unverified: {found.answer}"

checked_answer("When was the physician David Gregory born? "
               "See https://en.wikipedia.org/wiki/David_Gregory_(physician)")
```

```output
'David Gregory (physician) was born on 20 December 1625.'
```

(When the model should decide whether to check, give the AI function as
a tool instead: `@ai(tools=[read_page, fact_check])`.)

## What happened

`functai.inspect_history()` has every model call, whichever function
made it:

```python
for record in functai.inspect_history(3):
    parts = record.response.message.parts
    print(f"{record.function:<10} {record.model:<13}",
          ", ".join(f"calls {p.name}" if p.type == "tool_call" else "answers" for p in parts))
```

```output
research   gpt-4.1-mini  calls read_page
research   gpt-4.1-mini  answers
fact_check gpt-4.1-mini  answers
```

## On a local model

Only the model name changes. With [Ollama](https://ollama.com) running
and a model pulled (`ollama pull qwen3:8b`):

```python
research.lm = "ollama:qwen3:8b"
research("Which castle did the physician David Gregory inherit? ...")
```

Or any OpenAI-compatible server (vLLM, llama.cpp, LM Studio):

```python
functai.configure(lm="openai:Qwen/Qwen3-8B", base_url="http://localhost:8000/v1", api_key="none")
```

Models without native tool calling get tool calls written as text; the
prompt otherwise stays the same.
