/** Calls, offline: a fake router stands in for every provider. */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as z from "zod";
import {
  ai, bootstrapFewShot, calls, Cancelled, configure, evaluate, fromManifest, labeledFewShot, module, rate, rated,
  StepLimit, t, tool, toManifest, withSettings, type StreamEvent,
} from "../src/index.ts";
import { FakeRouter, whole } from "./fake.ts";

// the tests read no settings from the environment they run in (an agent's caller, a log folder)
for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

const moodDef = {
  name: "mood",
  description: "How does the customer feel about what they bought?",
  inputs: { review: t.string() },
  output: t.enum("happy", "unhappy", "mixed"),
};
const text = (m: { parts: readonly unknown[] }) => (m.parts as { text?: string }[]).map((p) => p.text ?? "").join("");

test("a call sends the layout and returns the typed answer", async () => {
  const router = new FakeRouter(["<result>\nunhappy\n</result>"]);
  const mood = ai({ ...moodDef, router });
  const answer: "happy" | "unhappy" | "mixed" = await mood("Broke after a day.");
  assert.equal(answer, "unhappy");
  const [req] = router.requests;
  assert.match(req!.system as string, /^Function: mood\n\nHow does the customer feel/);
  assert.equal(text(req!.messages[0]!), "<review>\nBroke after a day.\n</review>\n");
  assert.equal(req!.config?.stop, undefined);                    // OpenAI's Responses API takes no stop sequences
  assert.equal(req!.model, "gpt-4.1-mini");
  const claude = new FakeRouter(["<result>\nunhappy\n</result>"]);
  await ai({ ...moodDef, router: claude, lm: "anthropic:claude-3-5-haiku" })("Broke.");
  assert.deepEqual(claude.requests[0]!.config?.stop, ["</result>"]);
});

test("inputs by name or by position; render shows the request without sending it", async () => {
  const router = new FakeRouter(["<result>\nParis\n</result>"]);
  const capital = ai({ name: "capital", description: "The capital city.", inputs: { country: t.string(), year: t.integer() }, router });
  const request = capital.render("France", 1900);
  assert.equal(text(request.messages[0]!), "<country>\nFrance\n</country>\n<year>\n1900\n</year>\n");
  assert.equal(router.requests.length, 0);
  assert.equal(await capital({ country: "France", year: 1900 }), "Paris");
  await assert.rejects(capital("a", 1, 2), /takes 2 input/);
});

test("an unreadable reply is asked again once, with lmcc's hint", async () => {
  const router = new FakeRouter(["I think they are sad.", "<result>\nunhappy\n</result>"]);
  const mood = ai({ ...moodDef, router });
  const p = await mood.predict("Broke.");
  assert.equal(p.answer, "unhappy");
  assert.equal(p.attempts, 2);
  const again = router.requests[1]!.messages;
  assert.equal(again.length, 3);
  assert.match(text(again[2]!), /^Your reply could not be read: .*\. Reply again, in exactly the form the instructions give\.$/s);
});

test("a value outside its type is unreadable too; with retries: 0 the refusal is the error", async () => {
  const router = new FakeRouter(["<result>\nfurious\n</result>", "<result>\nunhappy\n</result>"]);
  assert.equal(await ai({ ...moodDef, router })("x"), "unhappy");
  const strict = ai({ ...moodDef, router: new FakeRouter(["<result>\nfurious\n</result>"]), retries: 0 });
  await assert.rejects(strict("x"), (err: { code?: string }) => err.code === "parse-value" || err.code === "parse-choice");
});

test("records: several outputs, the last is the answer", async () => {
  const router = new FakeRouter(["<summary>\nCharged twice\n</summary>\n<result>\n30\n</result>"]);
  const triage = ai({
    name: "triage", description: "Read the ticket.", inputs: { ticket: t.string() },
    outputs: { summary: t.string(), result: t.integer({ description: "minutes to fix" }) }, router,
  });
  const p = await triage.predict("I was charged twice");
  assert.deepEqual(p.outputs, { summary: "Charged twice", result: 30 });
  assert.equal(p.answer, 30);
  assert.match(triage.instructions, /Output guidance:\n- result: minutes to fix$/);
});

test("tools run until the model answers; StepLimit after maxSteps", async () => {
  const lookup = tool({ name: "lookup_order", description: "Look up an order.", input: { order: t.string() } },
    ({ order }) => (order === "A-1" ? "stuck at the carrier" : "unknown"));
  const router = new FakeRouter([
    { calls: [{ id: "c1", name: "lookup_order", input: { order: "A-1" } }] },
    "<result>\nIt is stuck at the carrier.\n</result>",
  ]);
  const helper = ai({ name: "helper", description: "Help.", inputs: { question: t.string() }, tools: [lookup], router });
  assert.equal(await helper("Where is A-1?"), "It is stuck at the carrier.");
  assert.equal(router.requests[0]!.tools?.[0]?.name, "lookup_order");
  const result = router.requests[1]!.messages.at(-1)!;
  assert.match(JSON.stringify(result), /stuck at the carrier/);
  const loop = ai({ name: "helper", description: "Help.", inputs: { question: t.string() }, tools: [lookup], maxSteps: 2,
    router: new FakeRouter([], () => ({ calls: [{ id: "c", name: "lookup_order", input: { order: "B" } }] })) });
  await assert.rejects(loop("?"), StepLimit);
});

test("zod schemas are read as the shapes Python writes", () => {
  const P = z.object({ name: z.string(), age: z.number().int() });
  const withZod = ai({ name: "person", description: "Who?", inputs: { text: z.string().describe("a sentence") }, output: P });
  const withT = ai({ name: "person", description: "Who?", inputs: { text: t.string({ description: "a sentence" }) },
    output: t.object({ name: t.string(), age: t.integer() }) });
  assert.deepEqual(withZod.signature.fields, withT.signature.fields);
  assert.equal(withZod.version, withT.version);
  assert.equal(withZod.signature.fields[0]!.desc, "a sentence");
});

test("the version follows what is sent, not where it runs", () => {
  const a = ai({ ...moodDef });
  assert.match(a.version, /^sha256:[0-9a-f]{64}$/);
  assert.equal(a.using({ lm: "claude-haiku-4-5", temperature: 0.3 }).version, a.version);
  assert.notEqual(a.using({ adapter: "json" }).version, a.version);
  assert.notEqual(a.using({ module: "cot" }).version, a.version);
  const b = a.using({});
  b.demos = [{ review: "Great", result: "happy" }];
  assert.notEqual(b.version, a.version);
  b.demos = [];
  assert.equal(b.version, a.version);
  b.instructions = "Say how they feel.";
  assert.notEqual(b.version, a.version);
});

// ------------------------------------------------------------------ the call log

function logged(folder: string): Record<string, any>[] {
  const out: Record<string, any>[] = [];
  for (const day of readdirSync(folder)) {
    for (const f of readdirSync(join(folder, day))) {
      for (const line of readFileSync(join(folder, day, f), "utf8").split("\n")) if (line) out.push(JSON.parse(line));
    }
  }
  return out;
}

test("every call is a line in the log; ratings make rows with known answers", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-log-"));
  const router = new FakeRouter([], (req) => (text(req.messages[0]!).includes("charged") ? "<result>\nunhappy\n</result>" : "<result>\nhappy\n</result>"));
  const mood = ai({ ...moodDef, router, logCalls: folder, definedIn: "shop" });
  const p1 = await mood.predict("I was charged twice");
  const p2 = await withSettings({ caller: { kind: "test", user: "ana" } }, () => mood.predict("Lovely"));
  const [c1, c2] = logged(folder);
  assert.equal(c1!.functai_call, 1);
  assert.equal(c1!.id, p1.callId);
  assert.deepEqual(c1!.program.name, "mood");
  assert.equal(c1!.program.module, "shop");
  assert.equal(c1!.program.version, mood.version);
  assert.equal(c1!.program.signature, mood.signatureId);
  assert.equal(c1!.program.answer, "result");
  assert.deepEqual(c1!.inputs, { review: "I was charged twice" });
  assert.deepEqual(c1!.outputs, { result: "unhappy" });
  assert.deepEqual(c1!.sizes, { inputs: { review: 21 }, outputs: { result: 9 } });
  assert.match(c1!.started, /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$/);
  assert.equal(c1!.exchanges.length, 1);
  assert.equal(c1!.exchanges[0].provider, "openai");
  assert.deepEqual(c1!.usage, { input_tokens: 10, output_tokens: 5, total_tokens: 15 });
  assert.equal(c1!.process.language, "typescript");
  assert.deepEqual(c2!.caller, { kind: "test", user: "ana" });

  rate(p1, "wrong", { answer: "mixed", by: "ben", folder });
  rate(p2.callId, "right", { by: "ben", folder });
  const { rows, leftOut } = rated(mood, { folder });
  assert.deepEqual(rows.map((r) => [r["review"], r["result"], r["rating"]]), [["I was charged twice", "mixed", "wrong"], ["Lovely", "happy", "right"]]);
  assert.deepEqual(leftOut, { other_signature: 0, no_content: 0, no_answer: 0 });
  assert.equal(calls(mood, { folder }).length, 2);
});

test("logContent: false keeps sizes and tokens, never values; a failed call records its error", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-log-"));
  const mood = ai({ ...moodDef, router: new FakeRouter(["nope", "still nope"]), logCalls: folder, logContent: false });
  await assert.rejects(mood("secret"));
  const [c] = logged(folder);
  assert.equal(c!.content, false);
  assert.ok(!("inputs" in c!) && !("outputs" in c!));
  assert.equal(c!.sizes.inputs.review, 8);                     // the canonical JSON "secret", quotes included
  assert.equal(c!.error.type, "Refusal");
  assert.ok(!("message" in c!.error));
  assert.equal(c!.exchanges.length, 2);
  assert.ok(!("request" in c!.exchanges[0]));
});

test("a module's calls are its children", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-log-"));
  const mood = ai({ ...moodDef, router: new FakeRouter([], () => "<result>\nhappy\n</result>"), logCalls: folder });
  const both = module("both", async (a: string, b: string) => [await mood(a), await mood(b)], { uses: [mood], settings: { logCalls: folder } });
  assert.deepEqual(await both("x", "y"), ["happy", "happy"]);
  const recs = logged(folder);
  const parent = recs.find((r) => r.program.kind === "module")!;
  const children = recs.filter((r) => r.program.kind === "ai");
  assert.equal(children.length, 2);
  for (const c of children) assert.equal(c.parent, parent.id);
  assert.deepEqual(parent.outputs, { result: ["happy", "happy"] });
  assert.equal(parent.program.version, both.version);
});

// ------------------------------------------------------------------ streaming

test("a stream shows the answer as it is written, and ends with the same value", async () => {
  const reply = "<result>\nIt is stuck at the carrier.\n</result>";
  const f = ai({ name: "where", description: "Where is it?", inputs: { order: t.string() }, router: new FakeRouter([reply]) });
  const s = f.stream("A-1");
  const pieces: string[] = [];
  for await (const piece of s) pieces.push(piece);
  assert.ok(pieces.length > 3);
  assert.equal(pieces.join(""), "It is stuck at the carrier.");
  assert.equal(await s.result, "It is stuck at the carrier.");
  const kinds: string[] = [];
  for await (const e of s.events()) kinds.push(e.kind);
  assert.equal(kinds[0], "started");
  assert.equal(kinds.at(-1), "done");
  assert.ok(kinds.slice(1, -1).every((k) => k === "text"));
});

test("a reply that arrives whole is one text piece per field", async () => {
  const router = whole(new FakeRouter(["<summary>\nshort\n</summary>\n<result>\nlong answer\n</result>"]));
  const f = ai({ name: "two", description: "Two.", inputs: { x: t.string() }, outputs: { summary: t.string(), result: t.string() }, router: router as never });
  const s = f.stream("x");
  const events: StreamEvent[] = [];
  for await (const e of s.events()) events.push(e);
  const texts = events.filter((e) => e.kind === "text") as Extract<StreamEvent, { kind: "text" }>[];
  assert.deepEqual(texts.map((e) => [e.field, e.answer, e.text]), [["summary", false, "short"], ["result", true, "long answer"]]);
});

test("a retry voids the text before it; closing a stream cancels the call", async () => {
  const s = ai({ ...moodDef, router: new FakeRouter(["not tags", "<result>\nhappy\n</result>"]) }).stream("x");
  const kinds: string[] = [];
  for await (const e of s.events()) kinds.push(e.kind);
  assert.ok(kinds.includes("retry"));
  assert.equal(s.text, "happy");
  const slow = ai({ ...moodDef, router: new FakeRouter(["<result>\nhappy\n</result>"], null, "openai", 1) });
  const c = slow.stream("x");
  c.close();
  await assert.rejects(c.result, Cancelled);
});

// ------------------------------------------------------------------ evaluate and improve

const rows = [
  { review: "Broke in a day", result: "unhappy" },
  { review: "Love it", result: "happy" },
  { review: "Good but late", result: "mixed" },
  { review: "Terrible", result: "unhappy" },
];
const guess = (req: { messages: readonly { parts: readonly unknown[] }[] }) => {
  const last = text(req.messages.at(-1)!);
  return `<result>\n${last.includes("Love") ? "happy" : "unhappy"}\n</result>`;
};

test("evaluate: the score, its range, and every answer", async () => {
  const mood = ai({ ...moodDef, router: new FakeRouter([], guess) });
  const ev = await evaluate(mood, rows);
  assert.equal(ev.score, 0.75);
  assert.ok(ev.low! < 0.75 && ev.high! > 0.75);
  assert.deepEqual(ev.scores(), [1, 1, 0, 1]);
  assert.match(String(ev), /^exact_match: 0\.75 \(95% range/);
  const failing = ai({ ...moodDef, router: new FakeRouter([], () => "?"), retries: 0 });
  const bad = await evaluate(failing, rows.slice(0, 2));
  assert.equal(bad.score, 0);
  assert.ok(bad.rows.every((r) => r.error));
});

test("improving adds worked examples: a new version, the demos in the request", async () => {
  const router = new FakeRouter([], guess);
  const mood = ai({ ...moodDef, router });
  const labeled = labeledFewShot(mood, rows, { k: 2, seed: 1 });
  assert.equal(labeled.demos.length, 2);
  assert.notEqual(labeled.version, mood.version);
  assert.equal(mood.demos.length, 0);
  const request = labeled.render("new");
  assert.equal(request.messages.length, 5);
  const boot = await bootstrapFewShot(mood, rows, { maxBootstrapped: 2, maxLabeled: 3 });
  const demos = boot.demos as unknown as Record<string, unknown>[];
  assert.equal(demos.filter((d) => "steps" in d).length, 2);
  assert.equal(demos.length, 3);
  assert.equal(boot.render("new").messages.length, 7);
});

// ------------------------------------------------------------------ saved

test("a function saved here loads back with the same version and requests", () => {
  const mood = ai({ ...moodDef, definedIn: "shop", temperature: 0 });
  mood.demos = [{ review: "Broke", result: "unhappy" }];
  const manifest = JSON.parse(JSON.stringify(toManifest(mood)));
  assert.equal(manifest.language, "typescript");
  assert.equal(manifest.entry, "shop:mood");
  const again = fromManifest(manifest);
  assert.equal(again.version, mood.version);
  assert.equal(again.signatureId, mood.signatureId);
  assert.equal(again.module, "shop");
  assert.deepEqual(again.render("x"), mood.render("x"));
});

// ------------------------------------------------------------------ layouts

test("a template without an output pattern: the whole reply is the answer", async () => {
  const router = new FakeRouter(["  A short summary.  "]);
  const summarize = ai({
    name: "summarize", description: "Summarize.", inputs: { text: t.string() }, router,
    template: [{ role: "system", text: "You are terse. {instruction}" }, { role: "user", text: "Text: {text}" }],
  });
  assert.equal(await summarize("a long text"), "A short summary.");
  assert.equal(router.requests[0]!.system, "You are terse. Function: summarize\n\nSummarize.");
  assert.equal(text(router.requests[0]!.messages[0]!), "Text: a long text");
});

test("the json layout asks for one object and reads it; cot adds reasoning first", async () => {
  const router = new FakeRouter(['{"result": {"name": "Ana", "age": 31}}']);
  const person = ai({ name: "person", description: "Who?", inputs: { text: t.string() },
    output: t.object({ name: t.string(), age: t.integer() }), adapter: "json", router });
  assert.deepEqual(await person("Ana, 31."), { name: "Ana", age: 31 });
  assert.equal((router.requests[0]!.config?.responseFormat as { type?: string } | undefined)?.type, "json_schema");
  const r2 = new FakeRouter(["<reasoning>\n3 pens at 7 each\n</reasoning>\n<result>\n21\n</result>"]);
  const solve = ai({ name: "solve", description: "Solve it.", inputs: { problem: t.string() }, output: t.number(), module: "cot", router: r2 });
  const p = await solve.predict("7 pens at 3 dollars?");
  assert.equal(p.answer, 21);
  assert.equal(p.outputs["reasoning" as never], "3 pens at 7 each");
  assert.match(r2.requests[0]!.system as string, /Reason step by step in the 'reasoning' section/);
});
