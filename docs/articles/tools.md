---
rat:
  project: ../../python
  python:
    dependencies: ["-e .[data]", "pandas"]
---

# Tools

*Give an AI function plain Python functions to call: lookups, searches, calculations.*

```python
import functai
functai.configure(lm="gpt-4.1-mini", temperature=0)   # the model behind every output on this page
from functai import ai, _ai
```

A model only knows what is in its prompt. To let it look things up or
compute exactly, give it **tools**: ordinary typed Python functions. When
a function has tools, a call becomes a loop. The model asks for a tool,
functai runs it and gives back the result, and so on until the model
answers.

## A first tool

A tool is a typed function with a docstring. Its name, parameters and
docstring are what the model reads, exactly as for an AI function.

```python
ORDERS = {
    "A-1042": {"item": "kettle", "status": "stuck at carrier", "shipped": "2026-09-02"},
    "B-2210": {"item": "toaster", "status": "delivered"},
}

def lookup_order(order_id: str) -> dict:
    """The order's item and shipping status."""
    return ORDERS.get(order_id, {"error": f"no order {order_id}"})

@ai(tools=[lookup_order])
def answer(question: str) -> str:
    """Answer the customer's question. Look the order up first."""
    ...

answer("Where is my order A-1042?")
```

```output
'Your order A-1042, which is a kettle, is currently stuck at the carrier. It was shipped on September 2, 2026.'
```

`phistory()` shows the whole loop: the tool call, its result, the answer.

```python
print(functai.phistory())
```

```output
[2026-10-02T10:01:51] answer → gpt-4.1-mini

System message:

Function: answer

Answer the customer's question. Look the order up first.

Reply in exactly this form:
<result>
...
</result>


User message:

<question>
Where is my order A-1042?
</question>


Assistant message:

[tool call lookup_order({"order_id": "A-1042"})]

Tool message:

[tool result call_4TTubMWC0TVAIsJM7rFWMikd] {"item": "kettle", "status": "stuck at carrier", "shipped": "2026-09-02"}

Tools: lookup_order

Response:

<result>
Your order A-1042, which is a kettle, is currently stuck at the carrier. It was shipped on September 2, 2026.
</result>

(finish: stop; tokens in 139, out 39)
```

## Several tools

```python
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression, like '(15 * 23) + 10'."""
    return eval(expression, {"__builtins__": {}})   # a demo: never eval untrusted text

def today() -> str:
    """Today's date, as YYYY-MM-DD."""
    return "2026-09-26"

@ai(tools=[lookup_order, calculate, today])
def assistant(question: str) -> str:
    """Answer the question, using the tools for facts and arithmetic."""
    ...

assistant("How many days ago was order A-1042 shipped?")
```

```output
'Order A-1042 was shipped 24 days ago.'
```

## How the loop behaves

- **Native or text.** Models with native tool calling use it; others get
  the same tools as text, in the same layout. Adding a tool never changes
  how the rest of the prompt is written.
- **At most `max_steps` model calls** (8 by default). A loop that runs out
  raises `functai.StepLimit`: `@ai(tools=[...], max_steps=20)` to allow
  more.
- **A tool that raises** is reported to the model, which can try again
  differently. `@ai(tool_errors="raise")` stops the call instead.
- **All of it is recorded**: `fn.predict(...)` gives the whole exchange in
  `p.turn`, and the tokens of every step are added up in `p.usage`.

```python
p = answer.predict("Has my toaster, order B-2210, been delivered?")
p.result, p.usage["input_tokens"]
```

```output
('Yes, your toaster from order B-2210 has been delivered.', 220)
```

## Tools that change things

A tool that looks something up is harmless; one that refunds, sends or
deletes is not. Say which a tool is, with `functai.tool`:

```python
REFUNDS, LOOKUPS = [], []

@functai.tool(effects="reads")
def find_order(order_id: str) -> dict:
    """The order's item, price paid and shipping status."""
    LOOKUPS.append(order_id)
    return {"A-1042": {"item": "kettle", "paid": 39.0, "status": "lost by the carrier"}}.get(order_id, {})

@functai.tool(effects="changes")
def refund(order_id: str, amount: float) -> str:
    """Refund an amount to the customer's card."""
    REFUNDS.append((order_id, amount))
    return f"refunded {amount:.2f} for {order_id}"

@ai(tools=[find_order, refund])
def desk(message: str) -> str:
    """Help the customer. Refund lost orders in full."""
    ...
```

`effects="reads"` (it only looks) or `"changes"` (it writes, sends,
pays). A tool that says nothing counts as `"changes"`: forgetting to
declare is safe. Then `approve=` says who decides before a tool runs.

**A function, asked at once.** It gets each tool call that changes things
(`functai.Approval`: the tool's `name`, its `input`, its `path`) and
answers `True`, `False`, or no with a reason. A refusal is not an error:
the model is told, and answers with that in mind.

```python
def manager(call):
    print("asked:", call.name, call.input)
    return "Refunds over 20.00 need a manager. Offer a replacement instead."

desk.using(approve=manager)("Order A-1042 never arrived. Refund it, please.")
```

```output
asked: refund {'order_id': 'A-1042', 'amount': 39}
'The order A-1042 was lost by the carrier. Since refunds over $20 require manager approval, I can offer you a replacement for the kettle instead. Would you like me to proceed with sending a replacement?'
```

**A rule, answered later by a person.** `approve="changes"` asks about
every tool that changes things (`"all"`: every tool; a list: those
tools, by name or by path such as `"support/refund"`, the calls from the
outermost one down to the tool). Nobody can be asked during a plain call,
so it refuses before the tool runs:

```python
try:
    desk.using(approve="changes")("Order A-1042 never arrived. Refund it, please.")
except functai.ApprovalError as error:
    print(error.code, "·", error)
REFUNDS
```

```output
approval-required · desk/refund: this tool call needs a person's answer (approval), and a plain call has nobody to ask. Give approve= a function, stream the call and answer with s.approve(...), or use a conversation, where the turn waits.
[]
```

In a [conversation](memory.md), the turn **waits** instead: it is saved,
`functai.Waiting` is raised, and anyone who opens the conversation, in
this process or another, tomorrow, answers it:

```python
import tempfile
store = tempfile.mkdtemp()
LOOKUPS.clear()

chat = desk.conversation("sam", store=store, approve="changes")
try:
    chat("Order A-1042 never arrived. Refund it, please.")
except functai.Waiting as waiting:
    print(waiting.turn.state, "·", [(a.name, a.input) for a in waiting.approvals])
```

```output
waiting · [('refund', {'order_id': 'A-1042', 'amount': 39})]
```

Later, somewhere else:

```python
turn = desk.conversation("sam", store=store).turns[-1]
turn.approve(by="maria")
```

```output
'The order A-1042 for the kettle, which was lost by the carrier, has been fully refunded with an amount of $39.00.'
```

```python
REFUNDS, LOOKUPS
```

```output
([('A-1042', 39)], ['A-1042'])
```

The turn went on where it stopped: the model replies it had were reused,
not asked for again, and `find_order` ran once, before the wait, not
again after it. `turn.deny(reason=...)`
says no (the model is told why), and `turn.abandon()` ends it without an
answer. While a turn waits, the conversation's next send is refused
(`conversation-busy`): answer it first.

On a stream, the call waits in its own process instead:
`s = desk.using(approve="changes").stream(...)`, then `s.approve()` or
`s.deny(...)` while it runs. A [served program](serving.md#tools-that-ask-first)
asks its owner, or its caller. Approval is a [plugin](plugins.md#asking-first),
so a plugin of your own can ask a person too, with a question of its own.

## When the process dies

A tool that changes things is recorded as started before it runs, and as
done after. If the process dies in between (a crash, a deploy, a laptop
lid), nobody knows whether it ran, so FunctAI never runs it again on its
own: a person says what happened.

Here a separate process sends an email, then dies before the tool returns:

```python
import subprocess, sys, textwrap
script = textwrap.dedent('''
    import functai, os, sys
    functai.configure(lm="gpt-4.1-mini", temperature=0)

    @functai.tool(effects="changes")
    def send_email(to: str, text: str) -> str:
        """Send an email."""
        open(sys.argv[2], "a").write(f"to {to}: {text}\\n")
        os._exit(1)                                   # the power goes out, right after sending

    @functai.ai(tools=[send_email])
    def assistant(message: str) -> str:
        """Do what the user asks, with the tools."""

    assistant.conversation("crash", store=sys.argv[1])("Email sam@example.com: the meeting moves to 3pm.")
''')
sent = store + "/sent.txt"
subprocess.run([sys.executable, "-c", script, store, sent]).returncode, open(sent).read()
```

```output
(1, 'to sam@example.com: The meeting moves to 3pm.\n')
```

The turn is `running` until its lease runs out (a running turn renews it
every 10 seconds; 30 seconds without a renewal, and it is `interrupted`):

```python
import time
time.sleep(31)

@functai.tool(effects="changes")
def send_email(to: str, text: str) -> str:
    """Send an email."""
    open(sent, "a").write(f"to {to}: {text}\n")
    return "sent"

@ai(tools=[send_email])
def assistant(message: str) -> str:
    """Do what the user asks, with the tools."""
    ...

turn = assistant.conversation("crash", store=store).turns[-1]
turn.state, [(u["name"], u["invocation"]) for u in turn.unfinished]
```

```output
('interrupted', [('send_email', 1)])
```

`turn.unfinished` lists the tools that may have run. Resuming refuses
until you say what each one returned:

```python
try:
    turn.resume()
except functai.ConversationError as error:
    print(error.code, "·", error)
```

```output
turn-unfinished · turn 01a0fcec-2250-7543-b602-1197ea45d6e5: send_email (invocation 1) started and may have run: resume(results={1: <what it returned>}) or resume(rerun=[1])
```

The email did go out, so:

```python
turn.resume(results={1: "sent"}), open(sent).read().count("\n")
```

```output
('The email to sam@example.com has been sent with the message: "The meeting moves to 3pm."', 1)
```

One email, not two. `rerun=[1]` runs the tool again instead, when you
know it did not run. A tool that only reads is safe to run again, so it
is never listed.

## Tools and optimization

A tool loop is a normal AI function: it can be [evaluated](accuracy.md)
and [optimized](improving.md). The worked examples an optimizer picks
keep their tool calls, so the model sees how a good run used the tools.
