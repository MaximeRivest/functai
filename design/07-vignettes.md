# 07 — Vignettes: the new surfaces, written before they exist

*Status, 2026-09-28: proposal. Nothing below is built. These are small, realistic programs written in the API we are considering, in Python, TypeScript, R and Julia, so the design can be judged by reading code before anyone writes the library. Builds on `06-surfaces-research.md`.*

How to read this: every example uses today's FunctAI where it can, and proposed names where it must. The proposed names are listed first, with their spelling in each language. Each vignette ends with what it tests and what writing it revealed. The last sections collect the decisions this forced and the questions still open.

A vignette is not a spec. Where one language shows something another does not, the reason is given; the missing piece is not an oversight.

---

## The proposed names

| Idea | Python | TypeScript | R | Julia |
|---|---|---|---|---|
| a conversation with a program (the remembered state; a tree of turns with a head) | `fn.conversation(id, store=…)` | `await fn.conversation({ id, store })` | `conversation(fn, id =, store =)` | `Conversation(fn; id, store)` |
| its turns, from the first to the head | `chat.turns` | `chat.turns` | `turns(chat)` (a tibble) | `turns(chat)` (a Tables.jl table) |
| continue from an earlier turn (a new branch; nothing is deleted) | `chat.continue_from(turn)` | `chat.continueFrom(turn)` | `continue_from(chat, turn)` | `continue_from(chat, turn)` |
| which earlier turns the model sees | `context=functai.last_turns(10)` | `context: lastTurns(10)` | `context = last_turns(10)` | `context = last_turns(10)` |
| the turns a given answer was based on | `turn.saw` | `turn.saw` | the `saw` column | `turn.saw` |
| where conversations are kept | `store="folder/"`, `True`, or an object | `store: "folder/"` or `{ append, read }` | `store = "folder/"` | `store = "folder/"` |
| memory of the AI functions a module calls | `remembers={answer: "conversation"}` | `remembers: { answer: "conversation" }` | (R has no modules yet) | `remembers = (answer = :conversation,)` |
| the conversation so far, as data, inside a module | `functai.earlier()` | `({ … }, { earlier })` | — | `earlier()` |
| what a tool does to the world | `@functai.tool(effects="changes")` | `tool(name, { effects: "changes" }, run)` | `ai_tool(f, …, .effects = "changes")` | `tool(f; effects = :changes)` |
| ask a person before a tool runs | `stream(…, approve=ask)` | `stream(…, { approve: "changes" })` | `approve = ask` | `approve = ask` |
| the same send twice is one turn | `request_id=` | `requestId` | `request_id =` | `request_id =` |
| a program served elsewhere, used like a local one | `functai.remote(url, key=…)` | `await remote(url, { key, input, output })` | `remote_ai(url, key =)` | `FunctAI.remote(url; key)` |
| the rows it is judged by, saved with it | `save(fn, path, answer_key=rows)` | `save(fn, path, { answerKey })` | `write_ai(fn, path, answer_key =)` | `FunctAI.save(path, fn; answer_key)` |
| command line | `functai check`, `verify`, `serve`, `push` | `npx functai …` (same commands) | — | — |

Removed by this proposal: Python's `stateful=True`, `state_window`, `fn.history` and `fn.reset()` (a conversation replaces them), and the unrelated `module.history` (a module's last 100 calls; the call log keeps them).

---

## Vignette 1 — A tutor that remembers (a notebook)

*Alex is learning fractions. The tutor should remember what Alex said, let the teacher see what the model was given, try another path without losing the first, and pick the conversation up tomorrow.*

### Python

```python
import functai
from functai import ai

functai.configure(lm="gpt-4.1-mini")

@ai
def tutor(message: str) -> str:
    """Tutor a student in fractions. One small step at a time, then ask
    them to try the next step themselves."""
    ...

# Kept in tutoring/; the same line tomorrow opens it again.
# tutor.conversation() alone keeps it in memory only.
chat = tutor.conversation("alex", store="tutoring/", context=functai.last_turns(10))

chat("Hi, I'm Alex. I never understood fractions.")
chat("What is 1/2 + 1/3?")
chat("Is it 2/5?")
# 'Not quite: 2/5 adds the tops and the bottoms. First make the bottoms equal: ...'

chat                          # the transcript, one turn per line
chat.turns[-1].saw            # [turn 1, turn 2]: what that answer was based on
chat.render("Why can't I just add the bottoms?")    # the exact next request, nothing sent

# What if Alex had answered 5/6? Continue after the second turn instead.
other = chat.continue_from(chat.turns[1])
other("Is it 5/6?")
len(chat.turns), len(other.turns)    # (3, 3): both paths are kept

# The teacher judges an answer; the call log keeps it, as for any call.
functai.rate(other.turns[-1], "right")
```

Tomorrow, in another notebook:

```python
chat = tutor.conversation("alex", store="tutoring/")
chat("I'm back. Can we do 3/4 - 1/2?")     # sees what came before
```

### TypeScript

```ts
import { ai, t, lastTurns, rate } from "functai";

const tutor = ai("tutor", {
  description: "Tutor a student in fractions. One small step at a time, then ask them to try the next step themselves.",
  input: { message: t.string() },
  output: t.string(),
});

const chat = await tutor.conversation({ id: "alex", store: "tutoring/", context: lastTurns(10) });

await chat("Hi, I'm Alex. I never understood fractions.");
await chat("What is 1/2 + 1/3?");
await chat("Is it 2/5?");

chat.turns.at(-1)?.saw;                            // what that answer was based on
chat.render("Why can't I just add the bottoms?");  // the exact next request, nothing sent

const other = chat.continueFrom(chat.turns[1]!);
await other("Is it 5/6?");
rate(other.turns.at(-1)!, "right");
```

### R

```r
library(functai)
ai_config(lm = "gpt-4.1-mini")

tutor <- ai(reply ~ message,
  "Tutor a student in fractions. One small step at a time, then ask them to try the next step themselves.")

chat <- conversation(tutor, id = "alex", store = "tutoring/", context = last_turns(10))

chat("Hi, I'm Alex. I never understood fractions.")
chat("What is 1/2 + 1/3?")
chat("Is it 2/5?")

turns(chat)
#> # A tibble: 3 × 5
#>    turn message                                      reply                            saw       call
#>   <int> <chr>                                        <chr>                            <list>    <chr>
#> 1     1 Hi, I'm Alex. I never understood fractions.  Welcome, Alex! Let's start with… <int [0]> 0192…
#> 2     2 What is 1/2 + 1/3?                           Good question. First, can you f… <int [1]> 0192…
#> 3     3 Is it 2/5?                                   Not quite: 2/5 adds the tops an… <int [2]> 0192…

ai_render(chat, "Why can't I just add the bottoms?")

other <- continue_from(chat, turn = 2)
other("Is it 5/6?")
rate(turns(other)$call[3], "right")
```

### Julia

```julia
using FunctAI
FunctAI.configure!(lm = "gpt-4.1-mini")

@ai function tutor(message::String)::String
    "Tutor a student in fractions. One small step at a time, then ask them to try the next step themselves."
end

chat = Conversation(tutor; id = "alex", store = "tutoring/", context = last_turns(10))

chat("Hi, I'm Alex. I never understood fractions.")
chat("What is 1/2 + 1/3?")
chat("Is it 2/5?")

turns(chat)[end].saw                 # what that answer was based on
render(chat, "Why can't I just add the bottoms?")

other = continue_from(chat, turns(chat)[2])
other("Is it 5/6?")
rate(turns(other)[end], :right)
```

**What it tests.** Memory belongs to a conversation, not to the function: `tutor` is unchanged and still callable on its own. A conversation is called like its program. Branching never deletes. What the model saw is recorded as references to turns, not copies (lmcc found copies grow quadratically). A turn can be rated like any call.

**What writing it revealed.**
- `last` is taken in R (dplyr) and in Julia (`Base.last`), so the name is `last_turns` everywhere.
- R has no objects with methods in this package, so the conversation is a function that remembers (a closure), and everything else is a plain function (`turns`, `continue_from`, `ai_render`), as the rest of the R package works. That means a conversation has reference semantics in R, unlike AI functions, which are copied when improved. Stated, because R users will not expect it; ellmer's chats behave the same way.
- Opening a conversation reads the store, so in TypeScript it is always `await` (even in memory, so the code does not change when a store is added). The others open synchronously.
- Reopening tomorrow: the head is the most recent turn. A program that needs another head (Chattering: one per person) says so with `continue_from`.

---

## Vignette 2 — One assistant, many people (a web app)

*A shop's assistant, answering many customers at once. Each customer has their own conversation; the program is shared.*

### Python (FastAPI, streaming)

```python
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from shop.auth import current_user        # your own sign-in
from shop.programs import assistant       # an @ai function or an @module

app = FastAPI()
STORE = "/var/lib/shop/conversations"     # or an object with append() and read()


class Message(BaseModel):
    text: str
    request_id: str                       # made by the page; a double click sends it twice


@app.post("/chats/{chat_id}/messages")
def send(chat_id: str, m: Message, user=Depends(current_user)):
    if not chat_id.startswith(f"{user.id}-"):        # who may use which conversation is the app's rule
        raise HTTPException(404)
    chat = assistant.conversation(chat_id, store=STORE)
    s = chat.stream(m.text, request_id=m.request_id)  # the turn is saved before the model is asked
    return StreamingResponse(s, media_type="text/plain")
```

### R (Shiny)

```r
library(shiny)
library(functai)

assistant <- read_ai("programs/assistant/")

ui <- fluidPage(
  uiOutput("transcript"),
  textInput("text", NULL, width = "100%"),
  actionButton("send", "Send")
)

server <- function(input, output, session) {
  chat <- conversation(assistant, id = session$token, store = "conversations/")
  shown <- reactiveVal(turns(chat))

  observeEvent(input$send, {
    chat(input$text)
    shown(turns(chat))
    updateTextInput(session, "text", value = "")
  })

  output$transcript <- renderUI({
    t <- shown()
    tagList(Map(function(m, r) tagList(p(strong("You: "), m), p(r)), t$message, t$reply))
  })
}

shinyApp(ui, server)
```

### Julia (Oxygen)

```julia
using Oxygen, HTTP, FunctAI

const assistant = FunctAI.load("programs/assistant")
const store = "conversations/"

@post "/chats/{id}/messages" function (req::HTTP.Request, id::String)
    body = json(req)
    chat = Conversation(assistant; id, store)
    (; reply = chat(body.text; request_id = body.request_id))
end

serve(port = 8080)
```

**What it tests.** One program, one conversation per person, no shared memory by accident: a conversation exists only when opened by id. The store is a folder by default and anything with two methods otherwise. Who may open which conversation stays the app's decision.

**What writing it revealed.**
- **Two sends to one conversation at once.** Proposed: they queue; the second sees the first. Parallel answers (Chattering's columns) are asked for explicitly, with `continue_from` on the same turn.
- **Shiny serves every visitor from one R process**, so while one answer is written the others wait. Shiny's `ExtendedTask` with a background process fixes that, and works only because the conversation reopens by id from the store in the other process. The store must therefore lock a conversation across processes, not only across threads.
- `session$token` changes when the page reloads, which is right for a demo and wrong for a real shop (use the signed-in user's id).
- R has no streaming yet, so the Shiny page shows each answer whole.

---

## Vignette 3 — Chattering: continue from an earlier message, survive a refresh

*Today's bug (conversation 01a0e745): after "Continue here", the answer streamed at the end of the old branch until the page was reloaded. With FunctAI conversations the answer's place is known before it is written.*

### TypeScript (Chattering's server)

```ts
import { conversationStore } from "functai";
import { assistant } from "./programs.ts";

// Chattering keeps its own files; a store is two methods over ordered JSON records.
const store = conversationStore("/home/maxime/.local/share/chattering/conversations");

// POST /api/send  { conversation, after, text, requestId }
export async function send(body: SendBody, person: Person) {
  const chat = await assistant.conversation({ id: body.conversation, store });
  const s = chat.continueFrom(body.after).stream(body.text, {
    requestId: body.requestId,           // a second click returns the same turn, no second call
    caller: { user: person.name },
  });
  return { turn: s.turn.id, after: body.after };   // saved already: the page places the answer under `after` now
}

// GET /api/turns/:id/events  (Server-Sent Events; a reconnect sends Last-Event-ID)
export function events(req: Request, turnId: string): Response {
  const after = Number(req.headers.get("last-event-id") ?? 0);
  const text = new TextEncoder();
  const body = new ReadableStream({
    async start(out) {
      for await (const e of store.watch(turnId, { after })) {     // saved events first, then live ones
        out.enqueue(text.encode(`id: ${e.seq}\ndata: ${JSON.stringify(e)}\n\n`));
      }
      out.close();
    },
  });
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

// The stop button: stops the call wherever it runs; the turn ends as "stopped".
export async function stop(turnId: string) {
  await store.stop(turnId);
}
```

A store Chattering writes itself:

```ts
import type { ConversationStore } from "functai";

const store: ConversationStore = {
  async append(conversation, records) { /* write them in order, all or none */ },
  async *read(conversation, { after }) { /* yield the records after position `after` */ },
};
```

**What it tests.** A turn has an id and a parent before the model is asked, so the page knows where the answer goes. Events carry a sequence number (`seq`) and are saved, so a reload or a second device reads them from where it stopped. A repeated request is one turn. Stopping works from any process.

**What writing it revealed.**
- Stream events need two new fields: `seq` (a position in the turn) and `turn`. That is a contract change (`streaming.md`, `event.schema.json`).
- `store.watch` across processes needs the store to say when records arrive. A folder can be watched; a custom store needs a third, optional method (`subscribe`), or `watch` falls back to asking every second.
- Stopping from another process means the running process must learn it from the store too. Same mechanism as `watch`.
- This vignette covers a program that answers. Chattering's coding agents also compact, change model mid-conversation, run extensions and delegate (the Pi parity list in `06`, section 5). None of that is shown, and none of it is decided (`06`, decision 2).
- Chattering's Pi router folds earlier turns into one message (`pirouter.js`). A conversation on such a router would change the prompt's shape without saying so. The router must declare that it takes one message, and a conversation must refuse it or say what it does instead.

---

## Vignette 4 — Tools that ask first

*An assistant that tidies notes. Reading is fine; changing a file needs a yes. In a notebook the yes is typed; on a server it is a button, maybe after a restart.*

### Python (a notebook: the question is asked right away)

```python
from pathlib import Path

import functai
from functai import ai

NOTES = Path("~/notes").expanduser().resolve()

def inside(name: str) -> Path:
    path = (NOTES / name).resolve()
    if NOTES not in path.parents:                  # the model chooses the name: keep it in the folder
        raise ValueError(f"{name} is outside the notes folder")
    return path

@functai.tool(effects="reads")
def read_note(name: str) -> str:
    """The text of one note."""
    return inside(name).read_text()

@functai.tool(effects="changes")
def write_note(name: str, text: str) -> str:
    """Replace a note's text."""
    inside(name).write_text(text)
    return f"wrote {len(text)} characters to {name}"

@ai(tools=[read_note, write_note])
def gardener(request: str) -> str:
    """Tidy the user's notes as they ask. Say what you changed."""
    ...

def ask(call) -> bool:
    print(f"\n{call.name}: {call.input['name']}\n{call.input['text'][:400]}")
    return input("Allow? [y/n] ").strip() == "y"

gardener.stream("Merge groceries.md into todo.md", approve=ask).show()
```

### TypeScript (a server: the turn waits, and survives a restart)

```ts
import { ai, t, tool } from "functai";
import { readFile, writeFile } from "node:fs/promises";
// inside(folder, name): the same check as Python's inside(), keeping the model's names in the folder

const readNote = tool("read_note", {
  description: "The text of one note.",
  input: { name: t.string() },
  effects: "reads",
}, ({ name }) => readFile(inside(notes, name), "utf8"));

const writeNote = tool("write_note", {
  description: "Replace a note's text.",
  input: { name: t.string(), text: t.string() },
  effects: "changes",
}, async ({ name, text }) => {
  await writeFile(inside(notes, name), text);
  return `wrote ${text.length} characters to ${name}`;
});

const gardener = ai("gardener", {
  description: "Tidy the user's notes as they ask. Say what you changed.",
  input: { request: t.string() },
  output: t.string(),
  tools: [readNote, writeNote],
});

const chat = await gardener.conversation({ id: conversationId, store });
const s = chat.stream(request, { approve: "changes", requestId });
for await (const e of s.events()) {
  if (e.kind === "approval") notify(person, e);    // { id, name, input }; the turn is saved as waiting
}

// The person answers, perhaps after the server restarted:
const turn = await chat.turn(turnId);
await turn.approve(callId);                        // or: await turn.deny(callId, "not that file")

// After a crash while write_note was running:
turn.state;         // "interrupted"
turn.unfinished;    // [{ id, name: "write_note", input, started: "2026-09-28T…" }]: it may have happened
await turn.resume({ results: { [callId]: "wrote 1204 characters to todo.md" } });   // the person checked: it did
// or: await turn.resume({ rerun: [callId] })   or: await turn.abandon()
```

R and Julia take the notebook form (`approve = ask`, a function that returns `TRUE`/`true`). The waiting form needs conversations and streaming; R has no streaming yet.

**What it tests.** A tool says what it does to the world. `approve` takes either a function (ask now) or a rule (`"changes"`: wait for a yes). A waiting turn is saved, so the yes can come from another process. After a crash, a tool that started but whose result was not saved is shown as uncertain, and a person decides; it is never re-run on its own.

**What writing it revealed.**
- **An undeclared tool** (every plain function today) has unknown effects. Proposed: `approve="changes"` asks for it too, so forgetting to declare is safe; `effects="reads"` is how to stop being asked.
- **The effect words.** `reads` and `changes` are enough here. Sending an email or paying also "changes", but a public program may want to tell them apart (`design/76`: "tools that act on the world"). MCP's hints (read-only, destructive, idempotent, open world) are the vocabulary to compare against before settling.
- **A denial is an answer the model sees** ("the person said no: not that file"), so the model can ask or try something else.
- **The path check (`inside`) is the author's job.** Nothing here makes a tool safe; `approve` only makes a person see it first. A sandbox is the host's (Chattering's) wall.
- `turn.resume({ results })` continues the model from its saved steps (lmcc keeps the current turn's steps), so no model answer is paid for twice.

---

## Vignette 5 — From conversations to an answer key

*The tutor ran all week; a teacher judged some answers in Chattering. Those judgements should test the next version, and Alex's conversation should carry on with it.*

### Python

```python
import functai
from dpyr import col

rows = functai.rated(tutor)
rows.columns
# ['message', 'result', 'earlier', 'conversation', 'call', 'version', 'rating', 'rated_by', ...]
#  earlier: the turns the model was given before that answer, as data

# Turns of one conversation depend on each other: keep each conversation on one side.
held_out = rows.distinct(col.conversation).slice_sample(n=10, seed=1)
test = rows.semi_join(held_out, on=col.conversation)
train = rows.anti_join(held_out, on=col.conversation)

better = functai.gepa(tutor, train, teacher="gpt-6-sol")
functai.compare(functai.evaluate(tutor, test), functai.evaluate(better, test))
# each row is asked again with its own earlier turns; no conversation changes

# Alex's conversation continues with the better version. A conversation is
# data: any program with the same inputs and outputs can carry it on.
chat = better.conversation("alex", store="tutoring/")
```

### R

```r
library(functai)
library(rsample)

rows <- rated(tutor)          # a tibble; `earlier` is a list column
split <- group_initial_split(rows, group = conversation)

better <- tutor |> gepa(training(split), teacher = "gpt-6-sol")
evaluate(better, testing(split))

chat <- conversation(better, id = "alex", store = "tutoring/")
```

### Julia

```julia
using FunctAI, DataFrames, Random

rows = DataFrame(rated(tutor))
held_out = Set(shuffle(Xoshiro(1), unique(rows.conversation))[1:10])
test  = filter(r -> r.conversation in held_out, rows)
train = filter(r -> !(r.conversation in held_out), rows)

better, trials = gepa(tutor, train; teacher = "gpt-6-sol")
compare(evaluate(tutor, test), evaluate(better, test))

chat = Conversation(better; id = "alex", store = "tutoring/")
```

TypeScript reads the same rows (`rated(tutor).rows`, each with `earlier`), for the same `evaluate` and optimizers.

**What it tests.** A rated turn is a row that keeps its context, so the "What is my name?" problem (`06`, section 4) disappears. Rows from one conversation stay together when splitting. Improving and continuing need no special "swap the program" call.

**What writing it revealed.**
- **`earlier` as a worked example does not work.** A worked example is one turn placed before the live question; an example that needs three earlier turns would look like part of the live conversation. Proposed: such rows are for measuring; optimizers use them as worked examples only when `earlier` is empty, and say how many they skipped. To decide.
- **Splitting by conversation matters and is easy to forget.** rsample's `group_initial_split` exists for exactly this; Python and Julia need it written out. A helper (`functai.split(rows, by="conversation")`) may be worth it.
- **Continuing with an improved version** works when the signature is unchanged (lmcc's turn fingerprint leaves the instruction out). A new input or output refuses with lmcc's `signature-mismatch` and a fix; moving old turns to a new signature is a separate, explicit step.
- dspy-session's "drop a bad turn and everything after it" is a filter on these rows (by conversation and turn order), not a new function. To confirm with a real case before adding one.

---

## Vignette 6 — Helpers with their own memory

*A support program: one AI function sorts the message, one answers, one writes a hand-over for a person. The sorter was measured without memory and must stay that way; the answerer should remember the conversation; the hand-over reads the whole conversation once.*

### Python

```python
from typing import Literal

import functai
from functai import ai, module

@ai
def topic(message: str) -> Literal["billing", "shipping", "product", "other"]:
    """What is the customer writing about?"""
    ...

@ai
def answer(message: str, topic: str) -> str:
    """Answer the customer kindly, in at most three sentences."""
    ...

@ai
def handoff(conversation: list[dict[str, str]]) -> str:
    """Summarize this support conversation for the person who takes it over:
    the problem, what was tried, and what the customer wants."""
    ...

@module
def support(message: str) -> str:
    kind = topic(message)
    if kind == "other":
        notify_staff(handoff(functai.earlier()))     # the conversation so far, as data
        return "A person from our team will write to you shortly."
    return answer(message, kind)

chat = support.conversation(customer_id, store=STORE, remembers={answer: "conversation"})
```

### TypeScript

```ts
const support = module("support", {
  input: { message: t.string() },
  output: t.string(),
  uses: [topic, answer, handoff],
}, async ({ message }, { earlier }) => {
  const kind = await topic(message);
  if (kind === "other") {
    await notifyStaff(await handoff({ conversation: earlier }));
    return "A person from our team will write to you shortly.";
  }
  return answer({ message, topic: kind });
});

const chat = await support.conversation({ id: customerId, store, remembers: { answer: "conversation" } });
```

### Julia

```julia
@program function support(message::String)::String
    kind = topic(message)
    if kind == other
        notify_staff(handoff(earlier()))
        return "A person from our team will write to you shortly."
    end
    answer(message, string(kind))
end

chat = Conversation(support; id = customer_id, store, remembers = (answer = :conversation,))
```

R has no modules yet (`r/README.md`, "Not here yet").

**What it tests.** One turn per message the customer sends; the helper calls are steps inside it (dspy-session v2's first invariant). Memory is chosen per helper: `"conversation"` (its own earlier calls in this conversation), `"turn"` (only within this message, for a helper called in a loop), or nothing (the default).

**What writing it revealed.**
- **Sharing the outer conversation cannot be done with turns.** lmcc renders only turns of the function's own signature; `handoff` has another. So sharing is an input the helper declares, filled with `earlier()`. It is explicit, typed, and works with every layout. dspy-session's "shared" mode has no direct equivalent, on purpose.
- **Helpers remember nothing unless told.** `topic` was measured on single messages; giving it memory by default would make it behave differently from what was measured. dspy-session defaults the other way. To decide.
- **TypeScript modules must declare inputs and outputs** to be conversed with, served or described (today they record `arg0`, `arg1`). This is the "whole-program interface" item, and a breaking change to `module()`.
- `remembers` is keyed by the function in Python (safe when renamed), by name in TypeScript and Julia (the natural key there). A name must then be unique among a module's helpers.
- Python passes `earlier` through a context variable; TypeScript as a second argument to the module's code, because an argument is easier to type and to see than ambient state.

---

## Vignette 7 — Ship it, and use it from anywhere

*The support team's classifier is good enough. Save it with the answers it is judged by, serve it, call it from a script, a phone shortcut, an R report. And, separately: someone else's program, found on Hugging Face.*

### Save (Python), check, verify, serve (command line)

```python
functai.save(team, "team/", answer_key=functai.rated(team))
```

```bash
functai check team/
#   team: one step, data only (no code runs when it loads)
#   answers: shipping, billing, product, account
#   tested on: gpt-4.1-mini, right 92% (95% range 84–96%), 60 answers judged
#   needs: a model; OPENAI_API_KEY for gpt-4.1-mini

functai verify team/ --lm gpt-4.1-mini --answers 20
#   20 answers from the answer key, asked again: 19 right (18 when saved)

functai serve team/ --lm gpt-4.1-mini --keys keys.txt --port 8080
#   serving team at http://0.0.0.0:8080  (a form at /, the API at /openapi.json)

functai push team/ hf://maxime/team
```

### Use it from anywhere

```python
import os

team = functai.remote("https://lambda.tail69222b.ts.net/programs/team", key=os.environ["TEAM_KEY"])
team("I was charged twice for order B-2210.")      # 'billing'
tickets.mutate(team=team(col.message))              # a column, as with a local function
```

```ts
const team = await remote("https://lambda.tail69222b.ts.net/programs/team", {
  key: process.env.TEAM_KEY!,
  input: { message: t.string() },
  output: t.enum("shipping", "billing", "product", "account"),   // checked against the server's description
});
await team("I was charged twice for order B-2210.");
```

```r
team <- remote_ai("https://lambda.tail69222b.ts.net/programs/team", key = Sys.getenv("TEAM_KEY"))
tickets |> mutate(team = team(message))
```

```julia
team = FunctAI.remote("https://lambda.tail69222b.ts.net/programs/team"; key = ENV["TEAM_KEY"])
df.team = team.(df.message)
```

### Someone else's program

```python
triage = functai.load("hf://someone/ticket-triage@3f2a91c")
```

```text
LoadRefused: hf://someone/ticket-triage at 3f2a91c runs code its author wrote.
  code: code/support.py, 140 lines (not signed)
  tools: send_email, which changes things; the model decides when to call it
  models: tested on gpt-4.1-mini, right 88% (40 answers judged by the author)
  needs: OPENAI_API_KEY, SMTP_PASSWORD
To run it here: functai.load(..., trust=True). To run it walled off: functai serve hf://someone/ticket-triage@3f2a91c --sandbox
```

**What it tests.** One command per question: what does it need (`check`, runs nothing), does it still answer as it did (`verify`, costs money, asked for explicitly), serve it, publish it. A served program is a program again on the other side (`remote`): it maps over a column, evaluates, rates. A stranger's program is refused with one summary, not a warning per piece.

**What writing it revealed.**
- **Verifying against the answer key costs money**, so it is its own command and never happens when serving starts.
- **`--lm` binds the model at deploy time**, and the summary says which model the scores came from. A score measured on another model is shown as that, not carried over.
- **A remote call is logged twice**: by the caller (a call whose exchange went to a URL) and by the server. The two records must share an id to be one call on the Programs pages. To design with the server contract.
- **TypeScript needs the shapes in the code** to type the result; they are checked against what the server describes when opened, so a mismatch fails before any call.
- In R, `remote` is a common word (the remotes package), so the name follows `read_ai`/`write_ai`: `remote_ai`.
- "Judged by the author" matters: an answer key that travels is a claim until someone else runs `verify`.

---

## Vignette 8 — Another model, and reasoning, halfway through

*Alex's tutoring (vignette 1) ran on a small model. A hard question comes; the teacher wants a stronger model for one answer, then turns on "reason first" for the rest of the conversation.*

### Python

```python
chat = tutor.conversation("alex", store="tutoring/")

# One answer from another model: a setting for this turn only.
chat("Why does dividing by 1/2 make a number bigger?", lm="claude-sonnet-4-5")
chat.turns[-1].model          # 'claude-sonnet-4-5': every turn records who answered
chat("Can you give me one to try?")                      # back to the conversation's model

# From now on, reasoning first. `using` makes a copy, as it does for any AI function.
thinking = tutor.using(module="cot")
chat = thinking.conversation("alex", store="tutoring/")
chat("I got 6. Is that right?")
```

```text
ConversationRefused: tutor now writes `reasoning` before `result`, and the 5 earlier turns have no `reasoning`.
  The model would see 5 answers written without reasoning, then be asked for one with it.
  To continue: tutor.using(module="cot").conversation("alex", store="tutoring/", earlier_without="reasoning")
  (earlier turns are shown without reasoning; nothing is invented or rewritten in the store)
```

```python
chat = thinking.conversation("alex", store="tutoring/", earlier_without="reasoning")
chat("I got 6. Is that right?")
chat.turns[-1].reasoning      # 'Alex computed 3 ÷ 1/2 ...'
```

### TypeScript

```ts
const chat = await tutor.conversation({ id: "alex", store: "tutoring/" });
await chat("Why does dividing by 1/2 make a number bigger?", { lm: "claude-sonnet-4-5" });

const thinking = tutor.using({ module: "cot" });
const later = await thinking.conversation({ id: "alex", store: "tutoring/", earlierWithout: ["reasoning"] });
await later("I got 6. Is that right?");
```

### R

```r
chat <- conversation(tutor, id = "alex", store = "tutoring/")
chat("Why does dividing by 1/2 make a number bigger?", .lm = "claude-sonnet-4-5")

thinking <- update(tutor, module = "cot")
chat <- conversation(thinking, id = "alex", store = "tutoring/", earlier_without = "reasoning")
chat("I got 6. Is that right?")
```

### Julia

```julia
chat = Conversation(tutor; id = "alex", store = "tutoring/")
chat("Why does dividing by 1/2 make a number bigger?"; lm = "claude-sonnet-4-5")

thinking = configure(tutor; reasoning = true)
chat = Conversation(thinking; id = "alex", store = "tutoring/", earlier_without = (:reasoning,))
chat("I got 6. Is that right?")
```

**What it tests.** A model is a setting, per conversation or per turn, and each turn records which model answered. Changing what the program writes (reasoning, a tool, a new output) changes its signature; the conversation says so before any call and offers one explicit way on.

**What writing it revealed.**
- **Adding reasoning is a signature change** (lmcc's turn fingerprint includes every field). Today it would refuse `signature-mismatch` with no way on. Proposed: `earlier_without` names fields the new signature has and the old turns lack; old turns are shown without them (lmcc already writes nothing for an absent hidden field). Removing a field, renaming one or changing a type still refuses. This needs one rule in lmcc (accept a turn whose signature differs only by named, absent hidden outputs) or a conversion step in FunctAI. To decide.
- **Adding the first tool** is the same kind of change (a `tool_calls` output appears) and would use the same option.
- **Hidden reasoning from the old model is dropped when the provider changes.** lm15 keeps it as continuation state labelled by provider; a turn answered by Anthropic cannot hand its sealed thinking to OpenAI. The turn keeps it in the store; the next request leaves it out, and `chat.render(...)` shows that it did. To confirm in lm15 that a foreign continuation is dropped, not sent.
- **A smaller model may not fit the conversation.** The context rule (`last_turns`) is counted in turns, not tokens; a model with a small window can refuse a conversation the previous model took. A token-based rule (`fits(model)`) is a candidate.
- R and Julia spell "a copy with other settings" as they already do (`update`, `configure(f; …)`), so nothing new is needed there.

---

## Vignette 9 — Three answers, then one

*Chattering's parallel answers: one question to three models, side by side, then a merge that keeps the best of each. Every answer stays; the conversation continues from the merge.*

### TypeScript (Chattering)

```ts
const chat = await assistant.conversation({ id: conversationId, store });
const question = chat.head;                        // the turn the person continues from

// Three answers to the same message, each on its own branch, in parallel.
const models = ["gpt-5.4", "claude-sonnet-4-5", "gemini-3.1-pro"];
// `predict` gives the whole call, as for any AI function; in a conversation it also names its turn.
const answers = await Promise.all(models.map((lm) =>
  chat.continueFrom(question).predict(message, { lm, requestId: `${requestId}:${lm}` })));

// The merge is an AI function like any other: its inputs are the answers.
const merge = ai("merge", {
  description: "Combine these answers into one: keep what each got right, drop what one got wrong, and say where they disagree.",
  input: { question: t.string(), answers: t.list(t.object({ model: t.string(), text: t.string() })) },
  output: t.string(),
});

// Recorded as the next turn after the question, reading the three branches.
const branches = answers.map((p) => p.turn);       // the three answers' turns
const merged = await chat.continueFrom(question).merge(branches, merge);
merged.reads;          // the three turn ids: what this answer was made from
chat.head = merged;    // the conversation continues from the merge
```

### Python (the same, in a notebook)

```python
q = chat.turns[-1]
branches = [chat.continue_from(q).predict(message, lm=lm).turn for lm in MODELS]
merged = chat.continue_from(q).merge(branches, merge)
chat = chat.continue_from(merged)
chat("Thanks. Now in two sentences.")
```

**What it tests.** Parallel answers are branches from one turn, nothing more. A merge is a turn whose program is another AI function, with the branches as its input and their ids recorded. Every answer stays readable and ratable; the merge can be judged like any call (was the combination right?).

**What writing it revealed.**
- **A turn has one parent** (the question), so the tree stays a tree; `reads` is a second, non-structural link. This is the rule Chattering's `fanoutmerge.js` learned the hard way: a merge must never create a parent cycle.
- **The merge's program is not the conversation's program.** The merged turn is `merge`'s signature, but the conversation's next turn is `assistant`'s. lmcc renders only turns of the plan's own signature, so the next turn cannot see the merge as an earlier `assistant` turn. Proposed: a merged turn is also recorded as an `assistant` turn whose answer is the merge's output, marked `made_by: merge` so its rating and data go to `merge`, not `assistant`. To decide; the alternative (the next turn sees only the question) loses the merge.
- **Merging is deterministic given its inputs** except for the model's answer: re-running it makes a new turn, never rewrites one.
- **`Promise.all` and the three `requestId`s**: each branch is its own turn, so two clicks do not start six answers. The queue rule of vignette 2 applies per branch, not per conversation, or parallel answers would wait for each other. To state in the contract.
- Asking three models is a comprehension today; a `chat.continue_from(q).each(message, lm=MODELS)` returning the three turns may read better. To try in the word test.
- `predict(...).turn` makes a conversation's turn the same thing a call's `Prediction` already carries (lmcc's turn), so no second record type appears.

---

## Vignette 10 — A photo in the conversation

*A plant helper. The person sends a photo of a leaf, asks what is wrong, and asks a follow-up without sending the photo again. Tomorrow the conversation reopens, photo and all.*

### Python

```python
from functai import Image, ai

@ai
def plant_doctor(message: str, photo: Image | None = None) -> str:
    """Help with house plants. When there is a photo, say what you see before you guess."""
    ...

chat = plant_doctor.conversation("leaf", store="plants/")
chat("What's wrong with my monstera?", photo=Image("leaf.jpg"))
chat("Should I cut that leaf off?")                   # no photo: the earlier one is still in the conversation

chat.turns[0].inputs["photo"]
# Image(sha256:9c1e…, image/jpeg, 1.8 MB) — kept once in plants/files/, the turn holds its hash
```

### TypeScript

```ts
import { ai, t, image } from "functai";

const plantDoctor = ai("plant_doctor", {
  description: "Help with house plants. When there is a photo, say what you see before you guess.",
  input: { message: t.string(), photo: t.optional(t.image()) },
  output: t.string(),
});

const chat = await plantDoctor.conversation({ id: "leaf", store: "plants/" });
await chat({ message: "What's wrong with my monstera?", photo: await image.fromFile("leaf.jpg") });
await chat("Should I cut that leaf off?");
```

### R

```r
plant_doctor <- ai(reply ~ message + photo,
  "Help with house plants. When there is a photo, say what you see before you guess.",
  photo = optional(image()))

chat <- conversation(plant_doctor, id = "leaf", store = "plants/")
chat("What's wrong with my monstera?", photo = image_file("leaf.jpg"))
chat("Should I cut that leaf off?")

# The same type over a table: one call per row, one photo per row.
leaves |> mutate(diagnosis = plant_doctor("What is wrong?", photo = image_file(path)))
```

### Julia

```julia
@ai function plant_doctor(message::String; photo::Union{FunctAI.Image,Nothing} = nothing)::String
    "Help with house plants. When there is a photo, say what you see before you guess."
end

chat = Conversation(plant_doctor; id = "leaf", store = "plants/")
chat("What's wrong with my monstera?"; photo = FunctAI.Image("leaf.jpg"))
chat("Should I cut that leaf off?")
```

**What it tests.** An image is an input type, like `str` or a choice, declared in the signature. It works in one call, over a table, and in a conversation. It is stored once and referred to by its hash.

**What writing it revealed: what FunctAI's inputs need.**
1. **A media type in each language**: Python `functai.Image` (and `Audio`, `Document`), TypeScript `t.image()` with `image.fromFile` / `fromUrl` / `fromBytes`, R `image()` in the codebook with `image_file()` values, Julia `FunctAI.Image`. Each is lm15's image part underneath (lm15 has `image`, `audio`, `document` parts in every language).
2. **The signature's shape**: lmcc already has a field shape `{"media": "image"}` (kernel §2). FunctAI must write it for these types and read it back when loading a saved function, so the four languages agree.
3. **The version and the call log**: a media value is not JSON. The call log writes `{"$media": "image/jpeg", "sha256": "…", "bytes": 1843200}`, never the bytes, and a separate folder (or the store) keeps the file when content is logged. The sample input used for a function's version needs a fixed tiny image per type, in the contract, so every language computes the same version.
4. **Capabilities, before money**: a model that cannot see images refuses at bind (lmcc's capability check), naming the model and the input. Chattering's Pi router, which sends text only (`pirouter.js`), refuses the same way.
5. **Where the image goes in the prompt**: lmcc places a media field by the template; a layout written for text needs a place for it. The built-in layouts must place media inputs (after the text inputs, in their own part). To check in lmcc's `xml`, `chat` and `json` layouts.
6. **Memory cost**: every later turn re-sends the photo while it is in the context. `last_turns(n)` would keep it for n turns; a rule that keeps a turn's text but drops its media after the first answer (`drop_media_after=1`) is a candidate, and `saw` records whether the photo was sent.
7. **Rated rows and datasets**: `rated(plant_doctor)` gives a `photo` column of images (paths into the log's files); `evaluate` and the optimizers take them. A worked example with a photo costs image tokens on every call, so optimizers should say so.
8. **Tool results with images** (a tool that takes a screenshot) use the same type as outputs of tools; FunctAI must stop flattening tool results to text (`06`, section 4).
9. **Model outputs that are images** (generation) are a different feature: not proposed here.

---

## Vignette 11 — Calls inside calls: what is kept, remembered, shown

*A shop's support program. The module sorts the message and answers it. The answering helper has a tool, `order_status`, which itself calls an AI function that reads the carrier's raw tracking log. A refund tool needs a person's yes. Customers call the program as an endpoint; the owner watches everything in Chattering.*

The tree of one customer message:

```text
support (module)                  ← the turn: {message} → {reply}; all a customer ever sees
├── topic (AI)                    message → "shipping"
└── answer (AI, tools)            remembers its own earlier calls in this conversation
    ├── model step                asks for order_status("B-2210")
    ├── tool step: order_status
    │   └── read_tracking (AI)    a child call: raw log → {where, late_days}
    ├── model step                asks for refund("B-2210", 12.50)    ← effects: "changes"
    ├── tool step: refund         waits for a person's yes, then runs
    └── model step                the reply
```

### Python

```python
from typing import Literal, TypedDict

import functai
from functai import ai, module

class Tracking(TypedDict):
    where: str
    late_days: int

@ai
def read_tracking(log: str) -> Tracking:
    """From a carrier's raw tracking log: where the parcel is, and how many days late."""
    ...

@functai.tool(effects="reads")
def order_status(order: str) -> Tracking:
    """Where an order is, and how late."""
    return read_tracking(carrier_log(order))       # an AI function inside a tool: a child call

@functai.tool(effects="changes")
def refund(order: str, amount: float) -> str:
    """Refund part or all of an order."""
    return payments.refund(order, amount)

@ai
def topic(message: str) -> Literal["billing", "shipping", "product", "other"]:
    """What is the customer writing about?"""
    ...

@ai(tools=[order_status, refund])
def answer(message: str, topic: str) -> str:
    """Answer the customer kindly. Refund the shipping cost for parcels more than 5 days late."""
    ...

@module
def support(message: str) -> str:
    return answer(message, topic(message))

chat = support.conversation(customer_id, store=STORE, remembers={answer: "conversation"})
s = chat.stream("Where is B-2210? It's a week late.", approve="changes")
```

**1. What the turn keeps.** The turn is the module's boundary; the calls inside are links into the call tree, never copies.

```python
turn = chat.turns[-1]
turn.inputs, turn.outputs      # {'message': ...}, {'result': 'Your parcel is in Leeds ...'}
turn.call                      # the root call's id; every call inside hangs under it
turn.tree()
# support  1.9 s  $0.004
# ├─ topic → shipping
# └─ answer → 'Your parcel is in Leeds ...'
#    ├─ order_status(B-2210)
#    │  └─ read_tracking → {'where': 'Leeds depot', 'late_days': 7}
#    └─ refund(B-2210, 12.50)  approved by maxime
turn.cost                      # summed over the tree
```

**2. What a remembering helper sees next time.** By default, its own earlier calls as inputs and outputs only: no tool steps, no child calls.

```python
chat("Thanks. And will it arrive before Friday?")
chat.render("And before Friday?", call=answer)   # the request answer would get: its earlier turn, no tool steps

# To let the helper see what its tools returned before (costs tokens, shows internals to the model):
chat = support.conversation(customer_id, store=STORE,
                            remembers={answer: functai.remember("conversation", steps=True)})
```

**3. What the outside sees.** A served program shows its boundary; the owner sees the tree.

```bash
functai serve support/ --keys keys.txt
# a customer's stream: started, text (of the reply), approval (if it is theirs to give), done
# never: topic's answer, read_tracking's input (the carrier log), tool calls and results, hidden reasoning
```

```python
for e in s.events(): ...                 # the owner, in-process: every call in the tree (today's behaviour)
for e in s.events(view="outside"): ...   # what a caller of the program sees: the boundary only
```

**4. Rating the right level.** The reply was wrong because `read_tracking` misread the log.

```python
wrong = turn.find(read_tracking)[0]            # the inner call
functai.rate(wrong, "wrong", answer={"where": "Manchester hub", "late_days": 7})
functai.rate(turn, "wrong", note="said Leeds; it was in Manchester")

functai.rated(read_tracking)                   # the inner call: a row of read_tracking's own answer key
functai.rated(support)                         # the turn: a row of the program's, with its `earlier`
```

**5. A pause three levels down.** The refund needs a yes. The model step that asked for it is saved, so a yes given later (after a restart) continues from there without paying for that step again.

```python
turn.waiting
# [Approval(path='support/answer/refund', call='0192…', input={'order': 'B-2210', 'amount': 12.5})]
turn.approve(turn.waiting[0])
```

### TypeScript

```ts
const readTracking = ai("read_tracking", {
  description: "From a carrier's raw tracking log: where the parcel is, and how many days late.",
  input: { log: t.string() },
  output: t.object({ where: t.string(), lateDays: t.integer() }),
});

// A tool whose code calls an AI function: that call is a child of the tool step.
const orderStatus = tool("order_status",
  { description: "Where an order is, and how late.", input: { order: t.string() }, effects: "reads" },
  async ({ order }) => readTracking(await carrierLog(order)));

const answer = ai("answer", {
  description: "Answer the customer kindly. Refund the shipping cost for parcels more than 5 days late.",
  input: { message: t.string(), topic: t.string() },
  output: t.string(),
  tools: [orderStatus, refund],
});

const support = module("support", {
  input: { message: t.string() },
  output: t.string(),
  uses: [topic, answer],
}, async ({ message }) => answer({ message, topic: await topic(message) }));

const chat = await support.conversation({ id: customerId, store, remembers: { answer: "conversation" } });
const s = chat.stream("Where is B-2210? It's a week late.", { approve: "changes", requestId });

// Chattering: the owner's page gets everything; a customer's page the boundary only.
for await (const e of s.events({ view: isOwner ? "full" : "outside" })) send(e);
```

### Julia

```julia
@ai function read_tracking(log::String)
    "From a carrier's raw tracking log: where the parcel is, and how many days late."
    location::String = ai"where the parcel is"
    late_days::Int = ai"days late"
end

"Where an order is, and how late."
order_status(order::String) = read_tracking(carrier_log(order))

@ai tools = [tool(order_status; effects = :reads), tool(refund; effects = :changes)] function answer(message::String, topic::String)::String
    "Answer the customer kindly. Refund the shipping cost for parcels more than 5 days late."
end

@program support(message::String)::String = answer(message, string(topic(message)))

chat = Conversation(support; id = customer_id, store, remembers = (answer = :conversation,))
turn = chat("Where is B-2210? It's a week late."; approve = :changes)
rate(first(find(turn, read_tracking)), :wrong; answer = (location = "Manchester hub", late_days = 7))
```

R has no modules yet: until it does, an R function that calls AI functions logs each call as its own root.

**What it tests.** Three boundaries, one rule each. The **turn** is the program's inputs and outputs, with the calls inside linked, not copied. A helper's **memory** is its own inputs and outputs, unless steps are asked for. The **outside view** of a served or shared program is its boundary only. Every call in the tree stays ratable at its own level, and a pause anywhere in the tree is addressed by its path.

**What writing it revealed.**
- **A conversation needs its call tree kept.** The turn links to calls, so a conversation with a store must keep them even when `log_calls` is off: in the store, with the call log's record shape, so the Programs pages read both. A call is written once: to the store when there is one, else to the log. To state in the contract.
- **"Remember steps" is lmcc's rendering of a turn's steps**, chosen per helper: without steps, past turns are written as inputs then outputs; with steps, the tool calls and results come too. Child calls under a tool step are never written into the parent's prompt in either mode (lmcc renders one level).
- **The outside view hides more than inner calls:** also the boundary call's own tool calls, its thinking, and the reasons for its retries (they can quote a helper's reply). An outside caller sees `started`, the answer's `text`, an `approval` addressed to them, and `done` or `failed`, the latter with an error type but no message. Everything else is the owner's.
- **Errors cross the boundary too.** `read_tracking` failing with "could not read: …" (quoting the carrier log) must reach the customer as `failed {type: "ToolError"}`. Same rule as `log_content=False`.
- **An AI function as a tool** works through a tool's code, as here. Passing the AI function itself (`tools=[read_tracking]`, its signature becoming the tool's) should work too. Settings reach the child through the usual layers; a tool's AI function keeps its own model unless the caller's block sets one. To confirm that this is the least surprising rule.
- **A wrong outer answer says nothing about which helper was wrong.** Nothing is inferred: it is a row of `support`, and becomes a row of `read_tracking` only when someone rates that call. Optimizing the module (`support.opt`) already credits helpers from the outer score.
- **Approval paths** (`support/answer/refund`) are names, not ids, so a host can write rules before any call exists ("always ask for refund, anywhere"). Two calls of the same tool are told apart by call id.
- **Cost** is the tree's sum, including models called from inside tools.
- **TypeScript needs `module()` to take a declared interface** (vignette 6), or there is no boundary to show.

**Three more rules, from dspy-session's nested-session design** (`dspy_session/docs/v2-recursive-sessions.md`, §6.4 and §7):

**6. Memory within one message, and across messages.** A helper called once per message (a translator) wants its calls from earlier messages. A helper called in a loop (a step-by-step agent, a retry loop) must see its earlier steps *within* this message, and is confused by the steps of earlier messages.

```python
chat = agent.conversation(customer_id, store=STORE, remembers={
    answer: "conversation",   # its calls in earlier messages, and its earlier calls in this one
    thinker: "turn",          # its earlier calls in this message only; fresh at every message
})
```

`"conversation"` includes the current message's earlier calls: a helper called twice in one message sees its first call on the second.

**7. A helper remembers its own branch only.** A helper's memory is the calls it made under the turns from the root to the turn being continued, found through the call tree, never "every call it ever made in this conversation".

```python
chat("Where is B-2210?"); chat("And B-2211?"); chat("Refund B-2211, please.")
other = chat.continue_from(chat.turns[0])
other("Actually, cancel B-2210.")
other.render("…", call=answer)    # answer's earlier call under turn 1 only; nothing from turns 2 and 3
```

**8. Replaying a turn restores every helper's memory.** Evaluating or improving on a rated turn of `support` asks the whole module again. Each helper must then see what it saw that time, not what it remembers today, and the real conversation must not change.

```python
rows = functai.rated(support)        # each row: the message, the right reply, and `earlier`
functai.evaluate(better_support, rows)
# for each row: support runs with earlier = the row's earlier turns; answer is given the
# calls it saw that time (the `saw` of its original call); nothing is recorded in any conversation
```

This needs `saw` on **every** call inside a conversation, not only on the turn: the call log's record gains the ids of the calls that call was shown (a contract change, `calls.md`). A helper whose original call is not in the log (logging off, a store lost) cannot be replayed faithfully; evaluation says so and counts the row out, never guesses.

**What these revealed.**
- dspy-session keeps a snapshot of each helper's memory in every turn (`child_snapshots`). With the call tree and `saw` as ids, the same replay needs no copies: memory is found, not stored twice (lmcc F7's lesson).
- **Its argument for helpers remembering by default** (§6): remembering its own calls is the only kind of memory that works with every layout, even when helpers share no fields (a corrector, then a translator). It is the strongest case against our default of "nothing unless told" (question 2). Both agree on the rejected option: showing a helper the outer conversation matched by field names, which silently fills past turns with nothing when the fields differ ("an attractive nuisance", its words).
- **A remembering program inside another one.** A module that calls something that already has a conversation of its own (a sub-agent with persistent memory) is not covered by `remembers`. dspy-session offers four policies (inherit, keep, unwrap, error). Proposed here: refuse by default, naming the inner conversation; `remembers={sub: "own"}` makes it keep its own conversation, with its turns linked from the outer turn's tree. To decide.
- **Memory never enters code or tools as a value.** dspy-session's RLM notebook handed history to a code-running agent as a variable; every step failed ("Unsupported value type: History") and the agent retried 20 times. Memory reaches the model through the layout (turns) only; `earlier()` is data the program's own code passes on, as a declared input. A program whose memory cannot be laid out refuses before the first call.

---

## Decisions these vignettes took (reversible, stated)

1. **The noun is "conversation".** "Session" means the R session, a Pi session and an HTTP session; "thread" means an OS thread in Python; "chat" is a model client in ellmer. "Conversation" also matches Chattering and the call log's `caller.conversation` key.
2. **A conversation is opened from the program** (`fn.conversation`), and called like it. No separate "session object wraps a module" step.
3. **Branching is `continue_from`, and nothing is ever deleted.** Undo is continuing from an earlier turn.
4. **What the model sees is a rule** (`last_turns(n)`, with options to drop bulky inputs from past turns), and every turn records which turns it saw.
5. **Helpers remember nothing unless told** (vignette 6).
   When told, `"conversation"` covers earlier messages and this one, `"turn"` this message only; a helper remembers its own branch only; replaying a turn restores each helper's memory from the calls it saw (vignette 11).
6. **Tools: `approve` takes a function or a rule; undeclared effects count as "changes" for the rule.** No approval by default, so notebooks keep working as today.
7. **Stores are a folder by default, or two methods** (`append`, `read`), with `subscribe` optional.
8. **Removed**: `stateful=True`, `state_window`, `fn.history`, `fn.reset()`, `module.history`. There are no users to keep them for (FunctAI intent).

## Maxime's answers (2026-09-28)

Given in conversation; they settle `06` section 11 and reshape the questions below.

1. **Scope: FunctAI is the whole engine for AI programs, not a library under one.** It holds the program representation (nodes), the runtime that runs them, serving, and the isolation levels (ProgramIR's rungs), because describing and running programs is the point. It does not hold the host's product: identities, screens, IDEs, accounts. A conversation is part of the program's *data model and harness description*, not an app.
2. **Coding agents are the long-term target.** Conversations should one day be able to express Chattering's coding agents. They describe the harness of an AI program, agent or conversation; the IDE and GUI around it stay the host's.
3. **Python and TypeScript are built first.** R (tidyverse, domain workflows) and Julia keep shaping the design from the start: every surface is written for them in the vignettes, as the test of whether the API is accessible and delightful.
4. **Stores are pluggable and extensible, with a folder by default**, and designed for streaming: a store can broadcast events live and persist a stream while it is still being written (partial, mid-stream data), not only finished turns.
5. **One log or two** is not decided up front: whatever the rest of the design shows is best. Development cost is not a constraint; aim for the best design.
6. **Safety is expressible, light by default, easy to raise.** The language and the program representation can say everything (effects, isolation levels, postures), with defaults that stay out of the way of prototyping and research (R, Julia, notebooks). People shipping to production configure more. How strict a platform is (Chattering's guests, public endpoints) is the platform's policy.
7. **ProgramIR: borrow its best ideas, in FunctAI's own idiom.** Not its format as is (it was not finished); whatever fits FunctAI's intent and is delightful.
8. **The model sees the whole conversation by default.** Running out of context and being told is an acceptable surprise; a model silently missing context is not. Rules like `last_turns` stay available.
9. **Nested calls: leaning towards only the first layer in context**, not the whole recursion, but this must be an easy toggle, to explore and learn the right default.
10. **Policies such as queueing concurrent sends are the host's.** FunctAI's job is the machinery that makes every such policy expressible and settable; its defaults will emerge while building. The design work focuses on FunctAI's machinery and language, not on one application's rules.

Consequences for this note: the per-conversation defaults below (queueing, photos, helpers' memory) become settings with provisional defaults, not decisions to make now. `06` section 10's contradiction 1 (who owns serving and sandboxes) is resolved: FunctAI owns the mechanism, the host owns the policy.


1. **Default context.** Every earlier turn (grows until the model's limit, then fails), or `last_turns(20)` (forgets silently)? A third option: every turn, and a clear error with the fix when it no longer fits.
2. **Helpers' default memory** (vignette 6): nothing (proposed) or their own calls, as dspy-session.
3. **Two sends at once to one conversation**: queue (proposed) or refuse.
4. **Rows with earlier turns as worked examples** (vignette 5): skip (proposed) or find a way to show an example conversation.
5. **Summaries of old turns.** Not shown here. The honest form is an input the program declares (`notes: str`) that a summarizing AI function fills with what fell out of the window. Worth a vignette of its own?
6. **R conversations as closures** (reference semantics), unlike R AI functions (copies): acceptable, as in ellmer?
7. **Adding reasoning or a tool mid-conversation** (vignette 8): a rule in lmcc for absent hidden fields, or a conversion step in FunctAI?
8. **What the next turn sees after a merge** (vignette 9): the merge recorded also as the program's own turn (proposed), or only the question.
9. **Old photos in the context** (vignette 10): kept for as many turns as the rest, or dropped after the first answer by default.
10. **Calls inside calls** (vignette 11): helpers remember inputs and outputs only (proposed), or their tool steps too; outside callers see the boundary only (proposed); a tool's AI function keeps its own model unless the caller's block sets one (proposed).
11. **A remembering program called inside another** (vignette 11): refuse unless declared (proposed), or one of dspy-session's policies as the default.

## Not covered by these vignettes

Live (realtime voice) models, which fit only as recordings of finished turns; image generation as an output; durable runs beyond tools (a crash in the middle of a model answer), trust postures and sandboxes in detail, packaging code programs (lock files, platforms), watching external servers, the portable workflow language, and Chattering's coding-agent features (compaction, model changes, extensions, delegation).

## Next

1. The word test (`06`, section 13): the table at the top and vignette 1, given to three people new to FunctAI; every wrong guess is a naming bug.
2. Maxime's answers to the questions above and in `06`, section 11.
3. Then the contract text: conversation records, event `seq` and `turn`, tool effects and approval, the rated row's `earlier` and `conversation` columns, `serve.md`.
