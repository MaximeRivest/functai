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


@ai
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


@functai.tool(effects="changes")
def send_note(text: str) -> str:
    """Send a short note to the team."""
    return "sent"


@ai(tools=[send_note])
def notifier(request: str) -> str:
    """Do what the user asks, using the tool when it helps. Say what you did."""


def _cached_twice(model: str) -> str:
    """The same request twice through a disk cache: the second is not sent."""
    import tempfile
    with functai.configure(cache_replies=tempfile.mkdtemp()):
        first = summarize("A disk cache keeps replies across runs.")
        n = len(functai.inspect_history(500))
        second = summarize("A disk cache keeps replies across runs.")
        assert first == second and functai.inspect_history(1)[0].cached, "the second reply was not from the cache"
        return f"{first!r} (cached after {n} records)"


def _approval() -> str:
    """A turn waits for a person's yes, then goes on."""
    chat = notifier.conversation(approve="changes")
    try:
        chat("Send the team a note saying the build is green.")
    except functai.Waiting as w:
        return w.turn.approve()
    raise AssertionError("the model did not call the tool, so nothing waited")


def _plugin_sections() -> str:
    """A plugin's section reaches the model and is recorded as a change."""
    shout = functai.Plugin("shout")
    shout.before_call(lambda call: functai.Change(sections=["Answer in CAPITAL LETTERS only."]))
    with functai.configure(plugins=[shout]):
        out = summarize("Plugins change calls through hooks, as data.")
    assert out == out.upper(), out
    return out


def _compaction() -> str:
    """A long conversation is summarized, and the summary keeps the facts."""
    chat = chat_fn.conversation(plugins=[functai.compaction(keep=1, every=2)])
    chat("My name is Alex and my cat is called Miso.")
    chat("I live in Montréal.")
    chat("I like fractions.")
    summary = chat.entries("compaction", "summary")[-1]["data"]["text"]
    assert "Miso" in summary, summary
    answer = chat("What is my cat called?")
    assert "Miso" in answer, answer
    return answer


@ai
def lookup(question: str) -> str:
    """Answer the question about our shop: it opens at 9 and closes at 17, Monday to Saturday."""


@ai(tools=[functai.delegate(lookup, name="shop_facts")])
def front_desk(request: str) -> str:
    """Answer the customer. Ask shop_facts for anything about the shop."""


def _delegation() -> str:
    answer = front_desk.conversation()("When do you close on Saturday?")
    assert "17" in answer or "5" in answer, answer
    return answer


chat_fn = chat


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
    conversation = chat.conversation()
    check("conversation", lambda: (conversation("Hi, my name is Alex."), conversation("What is my name?"))[1])
    check("disk cache", lambda: _cached_twice(model))
    check("approval", lambda: _approval())
    check("plugin sections", lambda: _plugin_sections())
    check("compaction", lambda: _compaction())
    check("delegation", lambda: _delegation())
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
