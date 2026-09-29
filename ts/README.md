# functai for TypeScript and JavaScript

**Write a function's signature. A language model writes the body. You measure how well.**

```ts
import { ai, t } from "functai";

const mood = ai("mood", {
  description: "How does the customer feel about what they bought?",
  input: { review: t.string() },
  output: t.enum("happy", "unhappy", "mixed"),
});

await mood({ review: "It broke after one day and support never answered." });   // "unhappy"
await mood("It broke after one day.");               // one input: its value alone
```

The answer comes back as the type you asked for (here `"happy" | "unhappy"
| "mixed"`), checked against it: a reply that does not fit is asked again
once, then refused. The inputs are typed too: a missing, misspelled or
mistyped input is a compile error, not a paid call. The same function
written in Python, R or Julia has the same version, logs the same records
and saves to the same folder: this package follows the [FunctAI
contract](https://github.com/MaximeRivest/functai/tree/master/contract), like the
[Python](https://maximerivest.github.io/functai/python.html), [R](https://maximerivest.github.io/functai/r/index.html) and
[Julia](https://maximerivest.github.io/functai/julia/index.html) packages.

**Status: 0.1.0, not on npm yet.** The [API reference](https://maximerivest.github.io/functai/ts/api/index.html)
lists every export with its types.

## Install

Once it is published, `npm install functai` (Node 22.18+, Deno, Bun; in a
browser everything but the call log works). Until then, build it from a
checkout, as in [*Developing*](#developing), and install that folder into
your project: `npm install ../functai/ts`.

Set the key of the provider you call (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …). Any model
[lm15](https://github.com/lm15-dev/lm15-ts) reaches works:
`lm: "claude-haiku-4-5"`, `"gemini:gemini-2.5-flash"`, `"groq:openai/gpt-oss-120b"`.

## Writing functions

```ts
import { ai, t, describe, configure } from "functai";

configure({ lm: "gpt-4.1-mini", temperature: 0 });

const triage = ai("triage", {
  description: "Read the support ticket.",
  input: { ticket: describe(t.string(), "the customer's own words") },
  outputs: {                                   // several outputs: the last is the answer
    summary: t.string({ description: "one sentence, no names" }),
    minutes: t.integer(),
  },
});

const p = await triage.predict("I was charged twice for order B-2210.");
p.outputs;      // { summary: "...", minutes: 15 }
p.answer;       // 15
```

- **Shapes** are JSON Schema: `t.string()`, `t.integer()`, `t.number()`,
  `t.boolean()`, `t.enum(...)`, `t.list(...)`, `t.object({...})`,
  `t.record(...)`, `t.optional(...)`; or a schema from any library that
  implements [Standard Schema](https://standardschema.dev) with JSON
  Schema (zod 4, valibot, arktype, ...), read as the JSON Schema Python
  writes for the same type. A `description` on a field (or
  `describe(shape, text)`, or zod's `.describe`) is guidance the model reads.
  A builder's extra keys add to its shape (`t.list(t.string(), {
  description, minItems: 1 })`) and never replace what it makes (`type`,
  `items`, `properties`, …: a `TypeError`); a default is a value of its
  type (`t.string({ default: "kind" })`).
- **Names are data**: `toString`, `constructor` or `__proto__` is a field
  (an input or an output) like any other, and so is a member of a JSON
  value; a worked example that leaves one out is sent without it, and a
  reply that leaves such an output out is asked again, as any other. Give
  a `__proto__` member in an object as data (`JSON.parse`,
  `lmcc.parseJson`, `{ ["__proto__"]: v }`): the literal `{ __proto__: v }`
  sets the object's prototype instead.
- **Members keep their order**, as in Python: the messages a call sends,
  its record, its events, the worked examples a bootstrap records and a
  saved folder hold a value's members in the value's order, even names
  like `"10"` that JavaScript lists first. JavaScript's own tools lose that
  order before FunctAI sees it: an object literal and `JSON.parse` list
  `"10"` first, and so do `JSON.stringify` and `structuredClone`. Read
  JSON with `lmcc.parseJson` and write it with `lmcc.jsonText`; what
  FunctAI gives you (answers, predictions, events, rows) carries lmcc's
  record of the order (`lmcc.memberNames`). One exception: JSON that lm15
  writes (the `json` adapter's `response_format` schema, a tool's
  parameters, a `config`'s maps) reaches the provider in JavaScript's
  order, `"10"` before `"b"`, where Python sends `"b"` first: lm15 takes
  plain objects, which cannot hold another order. The same text in the
  messages (the `xml` adapter's schema) keeps it.
- **JSON FunctAI writes** (a text input given an object, the call log, a
  saved folder) is what `JSON.stringify` writes, members aside: a hole in
  an array is `null`, `new Number(42)` is `42`, and a value that holds
  itself is refused (`TypeError`). An integer past 2^53 (lmcc reads one
  as a `bigint`) is written and recorded as its digits.
- **A call** takes its options second: `await mood(input, { lm:
  "gpt-6-luna", signal })`: settings for that call only, and an
  `AbortSignal` that cancels it (`Cancelled`).
- **Optional inputs**: an input may be left out when its schema says so,
  as the schema's library means it: `t.optional(...)` and zod's
  `.optional()` are sent as `null` (as Python's `x: T | None = None`),
  zod's `.default(x)` as `x`; `.nullable()` must be given, null or not. A
  given value is checked by its schema before any call. With exactly one
  required input, its value alone is the call: `summarize("…")`.
- **The reply cache**: `cacheReplies: true` answers an identical request
  (model, messages, every setting sent) with the reply it got before,
  without a call; any store with `get`, `set` and `delete` (a `Map`,
  Redis, …) works instead of memory, and is given plain JSON.
  `clearCache()` empties the memory one. Unreadable replies are not kept.
  Off by default: it answers identical requests identically, samples
  included.
- **Many inputs**: `await mood.map(reviews, { concurrency: 8 })` gives
  every answer, in order, 8 calls at a time; it rejects with the first
  failure and starts no more. (`evaluate` keeps going and scores a failure 0.)
- **Settings**: `lm`, `temperature`, `maxTokens`, `adapter` (`"xml"`, the
  default; `"chat"`; `"json"`), `module: "cot"` (reasoning first),
  `tools`, `retries`, … on the function, in `configure(...)`, in
  `withSettings({...}, () => ...)` for a block of code, or on one call.
  `fn.using({...})` is a copy with other settings.
- `fn.render(...)` is the exact request, without sending it.
  `fn.version` names what the function sends besides its inputs.

## Tools

```ts
import { tool } from "functai";

const lookup = tool("lookup_order", { description: "Look up where an order is.", input: { order: t.string() } },
  ({ order }) => orders[order] ?? "unknown order");      // `order` is a string, from its shape
const support = ai("support", { description: "Answer the customer.", input: { message: t.string() }, tools: [lookup] });
await support("Where is my order A-1042?");
```

The model calls tools until it answers (at most `maxSteps`, default 8).

## Your code around AI functions

```ts
import { module } from "functai";

const reply = module("reply", {
  description: "Answer a ticket, or hand it to a person when it is long work.",
  input: { ticket: t.string(), minutes: t.integer({ default: 60 }) },
  output: t.string(),
  uses: [triage, support],
}, async ({ ticket, minutes }, { signal }) =>
  (await triage.predict(ticket, { signal })).outputs.minutes > minutes ? escalate(ticket) : support(ticket, { signal }));

await reply("I was charged twice for order B-2210.");
reply.interface;   // { description, inputs: [...], outputs: [...] }: the same JSON in every language
```

A module is your own code that calls AI functions (Python's `@module`,
Julia's `@program`). It declares what it takes and gives, as `ai()` does,
and every call is checked against that at both ends: a wrong input or
output is an `InterfaceError` (`code`, `field`), recorded like any failed
call (its record and events keep only the fields it declares: a value
given under another name is refused and never kept). Its code gets the
inputs by name (a left-out input takes its default; `{ shape, optional:
true }` with none stays out) and the call's `signal`; closing its stream
or aborting the signal ends the call `Cancelled`, even if the code returns
later. One output, whatever its name, is the value it returns; several are
a record by name. It is logged as one call, with the calls it makes as its
children, and its version changes when its code, its interface or
anything it `uses` changes. A field `t.opaque()` takes values with no
JSON form (a buffer, a class instance), unchecked; `t.json()` any JSON.
Every program has `.interface`; `checkInterface()` checks one.

## Streaming

```ts
for await (const piece of haiku.stream("the first snow")) process.stdout.write(piece);

const s = support.stream("Where is A-1042?");
for await (const e of s.events()) console.log(e.seq, e.kind);   // started, request, text, tool_call, tool_result, ..., done
await s;                                                        // the same value as calling it
```

A stream is the same call, watched: the same retries, tools and log line.
Its events are the call tree's log (format 2): each has its `tree`, its
position (`writer` and `seq`) and `after`, the position of the event before
it in the form you read, so a reader knows when it missed one. `request`
and `retry` start a field's text afresh (`s.text` is the answer so far).
`s.events({ form: "kept" })` gives what the log may keep (`logContent`),
`{ view }` a view of it (`views.boundary(callId)`), and `{ after }`
resumes after an event you have (`s.read(tree, after)` too).

A reader elsewhere (a page, another process) follows a log with
`Follower`: it drops stale and duplicate events, takes the next, rewinds
when a later writer continued the log, and says `"loss"` when it must read
again (`await reader.recover(tree, source)`, from the process or a store).

## Keeping calls while they run: observers and journals

```ts
configure({
  observers: [(e) => socket.send(JSON.stringify(e)), worker],   // the kept form of every event, each its own copy, off the call's turn
  journal: { store, mode: "required", timeout: 10_000 },       // or a store alone: best effort, the call never waits
});
await flush();                                                 // before the process ends: observers and journals catch up
```

An observer gets the kept form of every event in its scope, in order, each
its own copy: nothing it does to an event reaches the call, the journal or
another observer. A function is called soon after each event, from a queue
drained between the call's steps, never inside them; it still runs on
this thread, so heavy work belongs in a `Worker`: an object with
`postMessage` (a `Worker`, a `MessagePort`) is posted each event. An
observer that throws or rejects is warned about once and gets no more
events; one that falls 10,000 events behind loses events (it sees the gap
in `after`). Each observer has its own queue and its own share of the time
given to observers, so one that is slow falls behind (and loses events)
alone; the others beside it get every event. One limit: observers run on
this thread only between the call's steps, so more than 10,000 events
made in one synchronous burst (thousands of calls started at once) make
every observer lose events. Observers add up over every layer;
`configure({ observers })` replaces `configure`'s own list.

A journal keeps each call tree's kept log in a store while it is written,
with appends the store answers (`MemoryStore` here; any object with
`append` and `read`, and `claim` if a later writer may continue a log).
Each append is waited for at most `timeout` ms (default 30,000; its
`signal` aborts then) and sent again after no answer, backing off
(`retries`, `backoff`). A store that throws, never answers or answers
something else than `"kept"`, `"duplicate"` or a refusal never holds the
call nor reaches the process. A best-effort journal that fails is warned
about once per outage. When a round of resends gives up, the writer tries
again on its own later (after 1 s, then 2, 4, … up to 60 s apart, for as
long as the process lives; these tries never keep the process alive) and
at the next event: a tree's last events have no next event to carry them.
Time alone never makes it give up. Memory does: when all journal writers
together hold more than 100,000 events not confirmed, the one holding the
oldest gives up on its log (warned once per outage; the log is kept at
least up to the events it confirmed, and nothing more is sent to it),
then the next, so a store that stays down costs a bounded amount of
memory. A writer that gives up lets go at once: the append under way is
aborted (its `signal`) and no longer waited for, and no resend follows.
The copies that append gave the store are the store's: one that ignores
the abort and never answers keeps them.

`await flush()` sends what is still not confirmed once more, and says
`true` only when every journal confirmed every event it was sent (or
refused one: it is sent nothing more), every observer has its events, and
no writer gave up on events since the previous `flush()` (each loss makes
one `flush()` say `false`).

A required journal makes the call wait until its events are kept, before
its code runs, before each tool and before it returns (at most `timeout`
at each; cancelling the call stops the first two), and raises
`JournalError` when they are not: `journal-barrier` (the code or the tool
did not run), or `journal-end`, which holds the call's outcome
(`err.outcome`: what you would have got, or the error) and the position of
its end (`await err.settle({ signal: AbortSignal.timeout(5000) })` finds
out whether it was kept; the signal stops waiting for a store that does
not answer). A program's own settings cannot replace or remove a host's
journal (`journal-policy`).

## How often is it right?

```ts
import { evaluate } from "functai";

const ev = await evaluate(mood, [
  { review: "Love it.", result: "happy" },
  { review: "Broke in a day.", result: "unhappy" },
]);
ev.score;              // 1
String(ev);            // "exact_match: 1.00 (95% range 0.34 to 1.00), n=2"
```

Columns named like the inputs are the inputs; columns named like the
outputs are the right answers (or `expected: "category"`). Rows are typed:
a row missing an input, or an `expected` column the rows lack, is a
compile error. The range is a
95% interval (Wilson's for right-or-wrong).

## Making it better

```ts
import { labeledFewShot, bootstrapFewShot, gepa } from "functai";

const taught = labeledFewShot(mood, rows, { k: 8 });            // rows become worked examples
const better = await bootstrapFewShot(mood, rows, { teacher: "gpt-4.1" });   // runs that were right become examples
const { fn: learned, trials } = await gepa(mood, rows, { teacher: "gpt-6-sol" });   // the instruction, rewritten from mistakes
```

Each returns an improved copy with a new version; `mood` is unchanged.

`gepa` shows a stronger model the function's answers with feedback in
words ("wrong: the right answer is billing") and keeps the best
instruction it writes, chosen on rows it never shows the teacher. It is
Python's `gepa()`, R's and Julia's, the same algorithm with the same
prompts ([design/04-gepa.md](https://github.com/MaximeRivest/functai/blob/master/design/04-gepa.md) says how
it differs from the paper's).
Live, on refund decisions it took `gpt-5.4-nano` from 68% to 87% on rows
it never saw; on a split where the model already scored 83% it gained
little (87%, 8 answers fixed, 6 broken). Measure it on rows it never saw;
its own score flatters.

## The call log and ratings

```ts
configure({ logCalls: true });                  // or FUNCTAI_LOG_CALLS=1
const p = await mood.predict("Arrived late.");
rate(p, "wrong", { answer: "mixed" });         // a person's correction
rated(mood).rows;                              // rows with known answers, for evaluate and the optimizers
calls(mood);                                   // every call, typed: started, seconds, usage.inputTokens, caller, ...
```

Every call is one line of JSON in `~/.local/share/functai/calls`, the
folder Python, R and Julia write too; ratings are lines next to them. Each
language reads the others' calls and ratings (formats 1 and 2).

`logContent` says which values are written: `false`, or by field
(`{ transcript: false }`; `{ "*": false, question: true }` keeps only the
question). It only removes: a value is written only when no layer (the
program's own, `withSettings`, a call's options, `configure`,
`FUNCTAI_LOG_CONTENT=0`) drops it. A record without every value says so
(`content: false`, `omitted`) and keeps no request, reply or error message.
A misspelled field name in a program's own map is refused when it is
defined (`SettingError`).

## Running what another language saved

```ts
import { load, save } from "functai";

const mood = load("mood/");        // a folder Python's functai.save (or R, or Julia) wrote
await mood("Late, but fine.");
```

`load` checks the function sends exactly what it sent in Python and has
its version, and refuses (with the reason) what only the saving language
can run: a function with code of its own around the model, tools, a baked
model.
`save(fn, "folder/")` writes a TypeScript function the same way, with its
interface; `save(module, "folder/")` writes a module's interface and the AI
functions it uses (another language describes the module, and loads its
AI functions by key). `describeSaved("mood/")` says what a saved program (an AI
function or a module, in any language) takes and gives, without running
anything.

## Not here yet

Compared with the Python package: baking (training your own weights),
the instruction-search and random-search optimizers (`gepa` is here),
stateful memory (every call records what it saw: `[]`), escalation to a bigger model, reading tables other than
arrays of objects, signing in with a subscription (set a key), and loading
programs with code (only AI functions travel between languages). The
[home page](https://maximerivest.github.io/functai/#what-each-language-has) compares the four languages;
[design/01-many-languages.md](https://github.com/MaximeRivest/functai/blob/master/design/01-many-languages.md)
is the plan.

## Developing

functai follows lmcc's development closely (lmcc is on npm, but this
package can need what is not released yet), so `package.json` links a
checkout of lmcc beside this repository. It needs lmcc's decision D-58
(names are data, members keep their order): commit `3492090` or later
(lmcc 0.8.4 as published on npm lacks it, and functai refuses to start
on it). From `functai/ts`:

```bash
git clone https://github.com/MaximeRivest/lmcc ../../lmcc   # lmcc beside functai
(cd ../../lmcc/ts && npm install)
npm install
npm run check        # types
npm test             # offline tests, and every case of ../contract
npm run build        # dist/, what npm would ship
npm run docs         # the API reference, into docs-api/ (TypeDoc)
```

`../check` (from the repository) runs every language, then each against
the others. `node tools/generate.ts` refreshes the contract's data
in `src/generated/`; `tools/live.ts` calls real models (costs cents).
