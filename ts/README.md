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
mistyped input is a compile error, not a paid call. The same function written in Python has the same
version, logs the same records and saves to the same folder: this package
follows the [FunctAI contract](../contract/), like the
[Python package](../python/).

**Status: 0.1.0, not yet on npm.** It needs lmcc's TypeScript kernel,
which is not published yet either (see *Developing*).

## Install

```bash
npm install functai          # Node 22.18+, Deno, Bun; browsers (no call log there)
```

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
- **A call** takes its options second: `await mood(input, { lm:
  "gpt-6-luna", signal })`: settings for that call only, and an
  `AbortSignal` that cancels it (`Cancelled`).
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

## Streaming

```ts
for await (const piece of haiku.stream("the first snow")) process.stdout.write(piece);

const s = support.stream("Where is A-1042?");
for await (const e of s.events()) console.log(e.kind);   // started, text, tool_call, tool_result, ..., done
await s.result;                                          // the same value as calling it
```

A stream is the same call, watched: the same retries, tools and log line.

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
Python's `GEPA` and R's `gepa()`, the same algorithm with the same
prompts (`../design/04-gepa.md` says how it differs from the paper's).
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

Every call is one line of JSON in `~/.local/share/functai/calls` (the
folder Python writes too); ratings are lines next to them. Python and
TypeScript read each other's calls and ratings.

## Running what Python saved

```ts
import { load, save } from "functai";

const mood = load("mood/");        // a folder Python's functai.save wrote
await mood("Late, but fine.");
```

`load` checks the function sends exactly what it sent in Python and has
its version, and refuses (with the reason) what only Python can run: a
function with code of its own around the model, tools, a baked model.
`save(fn, "folder/")` writes a TypeScript function the same way.

## Not here yet

Compared with the Python package: baking (training your own weights),
`InstructionSearch` and the random-search optimizer (`gepa` is here), stateful memory,
escalation to a bigger model, reading tables other than arrays of objects,
the reply cache, and loading programs with code (only AI functions travel
between languages). See `../design/01-many-languages.md`.

## Developing

lmcc's TypeScript kernel is not on npm yet, so `package.json` links a
checkout next to this repository:

```bash
git clone https://github.com/MaximeRivest/lmcc ../../lmcc   # ../lmcc beside functai
(cd ../../lmcc/ts && npm install)
npm install
npm run check        # types
npm test             # offline tests, and every case of ../contract
```

`../check` (from the repository) runs Python and TypeScript, then each
against the other. `node tools/generate.ts` refreshes the contract's data
in `src/generated/`; `tools/live.ts` calls real models (costs cents).
