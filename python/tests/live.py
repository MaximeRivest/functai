"""functai against real models (costs cents). Not run by pytest.

    set -a; source ~/Projects/lm15-dev/.env; set +a
    cd python && .venv/bin/python tests/live.py [model ...]
"""

import dataclasses
import enum
import sys
from typing import Literal

import functai
from functai import _ai, ai, system, user


class Priority(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclasses.dataclass
class Invoice:
    invoice_number: str
    vendor_name: str
    total: float
    items: list[str]


@ai
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""
    return _ai


@ai
def extract_invoice(document_text: str) -> Invoice:
    """Extract invoice information from the document text."""
    thought_process: str = _ai["Where each field is in the document."]
    return _ai


@ai
def priority(issue: str) -> Priority:
    """Classify the priority of the issue."""


@ai(module="cot")
def solve(problem: str) -> int:
    """Solve the word problem."""


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"Sunny and 22C in {city}."


@ai(tools=[get_weather])
def weather(question: str) -> str:
    """Answer the question. Use a tool when you need facts you do not have."""


@ai(template=[system("You are a helpful pirate. {instruction}"), user("Text: {text}")])
def pirate(text: str) -> str:
    """Summarize in 10 words."""


@ai(stateful=True)
def chat(message: str) -> str:
    """A friendly assistant that remembers the conversation."""


@ai(adapter="chat")
def categorize(text: str) -> Literal["sport", "fashion", "tech"]:
    """Categorize the text."""


@ai
def sentiment_score(text: str) -> float:
    """Returns a sentiment score between 0.0 (negative) and 1.0 (positive)."""
    score = _ai
    return max(0.0, min(1.0, float(score)))


DOC = "INVOICE\nVendor: TechCorp Inc.\nInvoice #: INV-2025-101\nItems: 5x Laptops, 2x Monitors\nTotal: $5600.00"


def run(model: str) -> list:
    failures = []
    functai.configure(lm=model, max_tokens=2000, temperature=0)

    def check(name, fn):
        try:
            out = fn()
            print(f"  ok   {name}: {out!r}"[:200])
        except Exception as exc:  # noqa: BLE001
            failures.append((model, name, exc))
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}"[:300])

    print(model)
    check("summarize", lambda: summarize("LMCC is the calling convention for language models: it lays out a call "
                                         "and reads the reply back into typed values."))
    check("invoice", lambda: extract_invoice(DOC))
    check("enum", lambda: priority("The main database is down for every customer."))
    check("cot", lambda: (solve.predict("Ana buys 7 pens at 3 each and pays with 50. How much change?").result))
    check("tools", lambda: weather("What's the weather in Montreal?"))
    check("pirate template", lambda: pirate("Foundation models are now mature enough for real-world use."))
    chat.reset()
    check("stateful", lambda: (chat("Hi, my name is Alex."), chat("What is my name?"))[1])
    check("chat adapter", lambda: categorize("The new phone has a faster chip."))
    check("post-processing", lambda: sentiment_score("I think FunctAI is amazing!"))
    return failures


def optimize(model: str) -> None:
    functai.configure(lm=model, max_tokens=500, temperature=0)

    @ai
    def classify_intent(user_query: str) -> str:
        """Classify user intent as 'booking', 'cancelation', or 'information'."""
        return _ai

    train = [{"user_query": q, "result": r} for q, r in [
        ("I need to reserve a room.", "booking"), ("How do I get there?", "information"),
        ("I want to cancel my reservation.", "cancelation"), ("Is breakfast included?", "information")]]
    before = functai.evaluate(classify_intent, train, num_threads=4)
    classify_intent.opt(train)
    after = functai.evaluate(classify_intent, train, num_threads=4)
    print(f"  opt  bootstrap: {before!r} → {after!r}; demos={len(classify_intent.demos)}")


if __name__ == "__main__":
    models = sys.argv[1:] or ["gpt-4.1-mini", "claude-haiku-4-5", "groq:openai/gpt-oss-120b"]
    failures = []
    for m in models:
        failures += run(m)
    optimize(models[0])
    print(functai.phistory(1))
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)
