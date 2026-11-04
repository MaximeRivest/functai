/** Calls, offline: a fake router stands in for every provider. */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as z from "zod";
import {
  ai, bootstrapFewShot, calls, Cancelled, compare, configure, describe, evaluate, fromManifest, InterfaceError, labeledFewShot, module, rate, rated,
  StepLimit, t, tool, toManifest, withSettings, type StreamEvent,
} from "../src/index.ts";
import { FakeRouter, whole } from "./fake.ts";
import { Message, Response } from "@lm15/lm15";
import * as lmcc from "lmcc";

// the tests read no settings from the environment they run in (an agent's caller, a log folder)
for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

const moodDef = {
  description: "How does the customer feel about what they bought?",
  input: { review: t.string() },
  output: t.enum("happy", "unhappy", "mixed"),
};
const text = (m: { parts: readonly unknown[] }) => (m.parts as { text?: string }[]).map((p) => p.text ?? "").join("");

test("a call sends the layout and returns the typed answer", async () => {
  const router = new FakeRouter(["<result>\nunhappy\n</result>"]);
  const mood = ai("mood", { ...moodDef, router });
  const answer: "happy" | "unhappy" | "mixed" = await mood("Broke after a day.");
  assert.equal(answer, "unhappy");
  const [req] = router.requests;
  assert.match(req!.system as string, /^Function: mood\n\nHow does the customer feel/);
  assert.equal(text(req!.messages[0]!), "<review>\nBroke after a day.\n</review>\n");
  assert.equal(req!.config?.stop, undefined);                    // OpenAI's Responses API takes no stop sequences
  assert.equal(req!.model, "gpt-4.1-mini");
  const claude = new FakeRouter(["<result>\nunhappy\n</result>"]);
  await ai("mood", { ...moodDef, router: claude, lm: "anthropic:claude-3-5-haiku" })("Broke.");
  assert.deepEqual(claude.requests[0]!.config?.stop, ["</result>"]);
});

test("compare says what differs when the rows are not the same", async () => {
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  const mood = ai("mood", { ...moodDef, router });
  const rows = [{ review: "Love it", mood: "happy" }, { review: "Broke", mood: "unhappy" }];
  const before = await evaluate(mood, rows, { expected: "mood", progress: false });
  const after = await evaluate(mood, rows.map((r) => ({ ...r, rules: "refunds are billing" })), { expected: "mood", progress: false });
  assert.throws(() => compare(before, after),
    /row 0 differs: column 'rules' only in the second\. Evaluate both versions on the very same table/);
});

test("a reply the provider stopped is never an answer, and is not asked again", async () => {
  const router = new FakeRouter([{ text: "", finish: "content_filter" }, "<result>\nhappy\n</result>"]);
  await assert.rejects(ai("mood", { ...moodDef, router, retries: 2 })("x"),
    (e: unknown) => lmcc.isRefusal(e) && (e as lmcc.Refusal).code === "parse-filtered"
      && (e as lmcc.Refusal).hint.includes("safety filter, or the model declining"));
  assert.equal(router.requests.length, 1);
  const reads = new FakeRouter([{ text: "<result>\nhappy\n</result>", finish: "content_filter" }]);
  await assert.rejects(ai("mood", { ...moodDef, router: reads })("x"), (e: unknown) => (e as lmcc.Refusal).code === "parse-filtered");
});

test("a model declining in its own words is lmcc's parse-filtered, and is not asked again", async () => {
  const declined = new Response({
    model: "m", message: Message.assistant([{ type: "refusal", text: "I can't." }] as never), finishReason: "stop" as never,
    usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
  });
  const router = new FakeRouter([declined, "<result>\nhappy\n</result>"]);
  await assert.rejects(ai("mood", { ...moodDef, router, retries: 2 })("x"),
    (e: unknown) => lmcc.isRefusal(e) && (e as lmcc.Refusal).code === "parse-filtered" && (e as lmcc.Refusal).hint.includes("I can't."));
  assert.equal(router.requests.length, 1);
});

test("an interrupted reply is sent again as it was: no correction, no larger budget", async () => {
  const router = new FakeRouter([{ text: "<result>\nhap", finish: "error" }, "<result>\nhappy\n</result>"]);
  assert.equal(await ai("mood", { ...moodDef, router })("x"), "happy");
  assert.equal(router.requests.length, 2);
  assert.deepEqual(router.requests[1]!.messages, router.requests[0]!.messages);
  assert.deepEqual(router.requests[1]!.config, router.requests[0]!.config);
  const twice = new FakeRouter([{ text: "<result>\nhap", finish: "error" }, { text: "<result>\nha", finish: "error" }]);
  await assert.rejects(ai("mood", { ...moodDef, router: twice })("x"),
    (e: unknown) => lmcc.isRefusal(e) && (e as lmcc.Refusal).code === "parse-interrupted");
  assert.equal(twice.requests.length, 2);
});

test("a picture given by its path is sent as its bytes, so a changed file is a new request", async () => {
  const dir = mkdtempSync(join(tmpdir(), "functai-media-"));
  const png = join(dir, "now.png");
  const A = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==", "base64");
  const B = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "base64");
  writeFileSync(png, A);
  const router = new FakeRouter([], () => "<result>\nyellow\n</result>");
  const colour = ai("colour", { description: "The main colour.", input: { picture: { media: "image" } as never }, output: t.string(),
    router, cacheReplies: join(dir, "replies.sqlite") });
  await colour({ picture: { media_type: "image/png", path: png } } as never);
  await colour({ picture: { media_type: "image/png", path: png } } as never);
  assert.equal(router.requests.length, 1);                                          // the same picture: from the cache
  const image = (router.requests[0]!.messages[0]!.parts as unknown as Rec[]).find((p) => p["type"] === "image")!;
  assert.equal(image["data"], A.toString("base64"));
  assert.equal(image["path"], undefined);
  writeFileSync(png, B);
  await colour({ picture: { media_type: "image/png", path: png } } as never);
  assert.equal(router.requests.length, 2);                                          // another picture, same name: asked
  await assert.rejects(colour({ picture: { media_type: "image/png", path: join(dir, "gone.png") } } as never),
    (e: unknown) => e instanceof InterfaceError && e.code === "interface-input" && /the file .*gone\.png cannot be read/.test(e.message));
  assert.equal(router.requests.length, 2);
});

test("a request to a provider that keeps requests says store: false, unless the settings say", async () => {
  const router = new FakeRouter(["<result>\nunhappy\n</result>", "<result>\nunhappy\n</result>", "<result>\nunhappy\n</result>"]);
  await ai("mood", { ...moodDef, router })("Broke.");
  await ai("mood", { ...moodDef, router, config: { store: true } })("Broke.");
  await ai("mood", { ...moodDef, router, lm: "anthropic:claude-3-5-haiku" })("Broke.");
  assert.deepEqual(router.requests.map((r) => r.config?.store), [false, true, undefined]);
});

test("inputs by name (one input: its value alone); render shows the request without sending it", async () => {
  const router = new FakeRouter(["<result>\nParis\n</result>", "<result>\nhappy\n</result>"]);
  const capital = ai("capital", { description: "The capital city.", input: { country: t.string(), year: t.integer() }, router });
  const request = await capital.render({ country: "France", year: 1900 });
  assert.equal(text(request.messages[0]!), "<country>\nFrance\n</country>\n<year>\n1900\n</year>\n");
  assert.equal(router.requests.length, 0);
  assert.equal(await capital({ country: "France", year: 1900 }), "Paris");
  await assert.rejects(capital("France" as never), /takes its inputs by name/);
  await assert.rejects(capital({ country: "France" } as never), /needs year/);
  await assert.rejects(capital({ country: "France", year: 1, month: 2 } as never), /has no input month/);
  const mood = ai("mood", { ...moodDef, router });
  assert.equal(await mood({ review: "Great." }), "happy");      // one input: by name, or its value alone
});

test("an unreadable reply is asked again once, with lmcc's hint", async () => {
  const router = new FakeRouter(["I think they are sad.", "<result>\nunhappy\n</result>"]);
  const mood = ai("mood", { ...moodDef, router });
  const p = await mood.predict("Broke.");
  assert.equal(p.answer, "unhappy");
  assert.equal(p.attempts, 2);
  const again = router.requests[1]!.messages;
  assert.equal(again.length, 3);
  assert.match(text(again[2]!), /^Your reply could not be read: .*\. Reply again, in exactly the form the instructions give\.$/s);
});

// functions.md, "When the reply cannot be read": a cut reply, all thinking and no answer.
function cutOffReply(request: { model: string }, adaptations: unknown[] = []): Response {
  return new Response({
    model: request.model, message: Message.assistant([{ type: "thinking", text: "still thinking" }] as never), finishReason: "length",
    usage: { inputTokens: 3, outputTokens: 900, totalTokens: 903, reasoningTokens: 900 }, adaptations: adaptations as never,
  });
}

test("a cut-off reply with a set budget is sent again with twice it", async () => {
  const router = new FakeRouter([], (req) => cutOffReply(req));
  const mood = ai("mood", { ...moodDef, router, maxTokens: 500, retries: 2 });
  await assert.rejects(mood("Broke."), (e: lmcc.Refusal) => {
    assert.equal(e.code, "parse-truncated");
    assert.ok(e.hint.endsWith("; the model spent 900 of its 900 output tokens thinking; raise maxTokens (it was 2000) or ask for less"), e.hint);
    return true;
  });
  assert.deepEqual(router.requests.map((r) => r.config?.maxTokens), [500, 1000, 2000]);
});

test("a cut-off reply without a budget is not sent again, and says what lm15 did", async () => {
  // The old rule re-sent it with 2048 after a reply that had 128000.
  const notes = [
    { field: "config.max_tokens", action: "defaulted", reason: "the Messages API requires max_tokens and none was set; the model's output ceiling was used", applied: 128000 },
    { field: "config.reasoning.thinking_budget", action: "dropped", reason: "budget_tokens is rejected by the API", asked: 32000 },
  ];
  const router = new FakeRouter([], (req) => cutOffReply(req, notes));
  const mood = ai("mood", { ...moodDef, router, retries: 2 });
  await assert.rejects(mood("Broke."), (e: lmcc.Refusal) => {
    assert.ok(e.hint.endsWith("; the model spent 900 of its 900 output tokens thinking; no maxTokens was set, and lm15 sent 128000, "
      + "the most it knows this model to allow: lower the reasoning effort or ask for less "
      + "(lm15 adapted the request: config.reasoning.thinking_budget dropped: budget_tokens is rejected by the API)"), e.hint);
    return true;
  });
  assert.equal(router.requests.length, 1);
  const own = new FakeRouter([], (req) => cutOffReply(req));
  await assert.rejects(ai("mood", { ...moodDef, router: own, retries: 2 })("Broke."), (e: lmcc.Refusal) =>
    e.hint.endsWith("; no maxTokens was set, so the provider used its own maximum: lower the reasoning effort or ask for less"));
  assert.equal(own.requests.length, 1);
});

test("a value outside its type is unreadable too; with retries: 0 the refusal is the error", async () => {
  const router = new FakeRouter(["<result>\nfurious\n</result>", "<result>\nunhappy\n</result>"]);
  assert.equal(await ai("mood", { ...moodDef, router })("x"), "unhappy");
  const strict = ai("mood", { ...moodDef, router: new FakeRouter(["<result>\nfurious\n</result>"]), retries: 0 });
  await assert.rejects(strict("x"), (err: { code?: string }) => err.code === "parse-value" || err.code === "parse-choice");
});

test("records: several outputs, the last is the answer", async () => {
  const router = new FakeRouter(["<summary>\nCharged twice\n</summary>\n<result>\n30\n</result>"]);
  const triage = ai("triage", {
    description: "Read the ticket.", input: { ticket: t.string() },
    outputs: { summary: t.string(), result: t.integer({ description: "minutes to fix" }) }, router,
  });
  const p = await triage.predict("I was charged twice");
  assert.deepEqual(p.outputs, { summary: "Charged twice", result: 30 });
  assert.equal(p.answer, 30);
  assert.match(triage.instructions, /Output guidance:\n- result: minutes to fix$/);
});

test("tools run until the model answers; StepLimit after maxSteps", async () => {
  const lookup = tool("lookup_order", { description: "Look up an order.", input: { order: t.string() } },
    ({ order }) => (order === "A-1" ? "stuck at the carrier" : "unknown"));
  const router = new FakeRouter([
    { calls: [{ id: "c1", name: "lookup_order", input: { order: "A-1" } }] },
    "<result>\nIt is stuck at the carrier.\n</result>",
  ]);
  const helper = ai("helper", { description: "Help.", input: { question: t.string() }, tools: [lookup], router });
  assert.equal(await helper("Where is A-1?"), "It is stuck at the carrier.");
  assert.equal(router.requests[0]!.tools?.[0]?.name, "lookup_order");
  const result = router.requests[1]!.messages.at(-1)!;
  assert.match(JSON.stringify(result), /stuck at the carrier/);
  const loop = ai("helper", { description: "Help.", input: { question: t.string() }, tools: [lookup], maxSteps: 2,
    router: new FakeRouter([], () => ({ calls: [{ id: "c", name: "lookup_order", input: { order: "B" } }] })) });
  await assert.rejects(loop("?"), StepLimit);
});

test("zod schemas are read as the shapes Python writes", () => {
  const P = z.object({ name: z.string(), age: z.number().int() });
  const withZod = ai("person", { description: "Who?", input: { text: z.string().describe("a sentence") }, output: P });
  const withT = ai("person", { description: "Who?", input: { text: t.string({ description: "a sentence" }) },
    output: t.object({ name: t.string(), age: t.integer() }) });
  assert.deepEqual(withZod.signature.fields, withT.signature.fields);
  assert.equal(withZod.version, withT.version);
  assert.equal(withZod.signature.fields[0]!.desc, "a sentence");
});

test("the version follows what is sent, not where it runs", () => {
  const a = ai("mood", { ...moodDef });
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
  const mood = ai("mood", { ...moodDef, router, logCalls: folder, definedIn: "shop" });
  const p1 = await mood.predict("I was charged twice");
  const p2 = await withSettings({ caller: { kind: "test", user: "ana" } }, () => mood.predict("Lovely"));
  const [c1, c2] = logged(folder);
  assert.equal(c1!.functai_call, 2);
  assert.equal(c1!.program.interface, mood.interfaceId);
  assert.deepEqual(c1!.saw, []);
  assert.match(c1!.exchanges[0].request_hash, /^sha256:[0-9a-f]{64}$/);
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
  assert.deepEqual(leftOut, { otherSignature: 0, noContent: 0, noAnswer: 0, noContext: 0 });
  assert.equal(calls(mood, { folder }).length, 2);
});

test("logContent: false keeps sizes and tokens, never values; a failed call records its error", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-log-"));
  const mood = ai("mood", { ...moodDef, router: new FakeRouter(["nope", "still nope"]), logCalls: folder, logContent: false });
  await assert.rejects(mood("secret"));
  const [c] = logged(folder);
  assert.equal(c!.content, false);
  assert.ok(!("inputs" in c!));
  assert.equal(c!.outputs, null);                              // failed before an answer (format 2 says so even without content)
  assert.deepEqual(c!.omitted, { inputs: ["review"], outputs: ["result"] });
  assert.equal(c!.sizes.inputs.review, 8);                     // the canonical JSON "secret", quotes included
  assert.equal(c!.error.type, "Refusal");
  assert.ok(!("message" in c!.error));
  assert.equal(c!.exchanges.length, 2);
  assert.ok(!("request" in c!.exchanges[0]));
});

test("a module's calls are its children", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-log-"));
  const mood = ai("mood", { ...moodDef, router: new FakeRouter([], () => "<result>\nhappy\n</result>"), logCalls: folder });
  const both = module("both", { input: { a: t.string(), b: t.string() }, output: t.list(t.string()), uses: [mood], logCalls: folder },
    async ({ a, b }) => [await mood(a), await mood(b)]);
  assert.deepEqual(await both({ a: "x", b: "y" }), ["happy", "happy"]);
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
  const f = ai("where", { description: "Where is it?", input: { order: t.string() }, router: new FakeRouter([reply]) });
  const s = f.stream("A-1");
  const pieces: string[] = [];
  for await (const piece of s) pieces.push(piece);
  assert.ok(pieces.length > 3);
  assert.equal(pieces.join(""), "It is stuck at the carrier.");
  assert.equal(await s.result, "It is stuck at the carrier.");
  const kinds: string[] = [];
  for await (const e of s.events()) kinds.push(e.kind);
  assert.equal(kinds[0], "started");
  assert.equal(kinds[1], "request");                            // a request empties the fields (format 2)
  assert.equal(kinds.at(-1), "done");
  assert.ok(kinds.slice(2, -1).every((k) => k === "text"));
});

test("a reply that arrives whole is one text piece per field", async () => {
  const router = whole(new FakeRouter(["<summary>\nshort\n</summary>\n<result>\nlong answer\n</result>"]));
  const f = ai("two", { description: "Two.", input: { x: t.string() }, outputs: { summary: t.string(), result: t.string() }, router: router as never });
  const s = f.stream("x");
  const events: StreamEvent[] = [];
  for await (const e of s.events()) events.push(e);
  const texts = events.filter((e) => e.kind === "text") as Extract<StreamEvent, { kind: "text" }>[];
  assert.deepEqual(texts.map((e) => [e.field, e.answer, e.text]), [["summary", false, "short"], ["result", true, "long answer"]]);
});

test("a retry voids the text before it; closing a stream cancels the call", async () => {
  const s = ai("mood", { ...moodDef, router: new FakeRouter(["not tags", "<result>\nhappy\n</result>"]) }).stream("x");
  const kinds: string[] = [];
  for await (const e of s.events()) kinds.push(e.kind);
  assert.ok(kinds.includes("retry"));
  assert.equal(s.text, "happy");
  const slow = ai("mood", { ...moodDef, router: new FakeRouter(["<result>\nhappy\n</result>"], null, "openai", 1) });
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
  const mood = ai("mood", { ...moodDef, router: new FakeRouter([], guess) });
  const ev = await evaluate(mood, rows);
  assert.equal(ev.score, 0.75);
  assert.ok(ev.low! < 0.75 && ev.high! > 0.75);
  assert.deepEqual(ev.scores(), [1, 1, 0, 1]);
  assert.match(String(ev), /^exact_match: 0\.75 \(95% range/);
  const failing = ai("mood", { ...moodDef, router: new FakeRouter([], () => "?"), retries: 0 });
  const bad = await evaluate(failing, rows.slice(0, 2));
  assert.equal(bad.score, 0);
  assert.ok(bad.rows.every((r) => r.error));
});

test("improving adds worked examples: a new version, the demos in the request", async () => {
  const router = new FakeRouter([], guess);
  const mood = ai("mood", { ...moodDef, router });
  const labeled = labeledFewShot(mood, rows, { k: 2, seed: 1 });
  assert.equal(labeled.demos.length, 2);
  assert.notEqual(labeled.version, mood.version);
  assert.equal(mood.demos.length, 0);
  const request = await labeled.render("new");
  assert.equal(request.messages.length, 5);
  const boot = await bootstrapFewShot(mood, rows, { maxBootstrapped: 2, maxLabeled: 3 });
  const demos = boot.demos as unknown as Record<string, unknown>[];
  assert.equal(demos.filter((d) => "steps" in d).length, 2);
  assert.equal(demos.length, 3);
  assert.equal((await boot.render("new")).messages.length, 7);
});

// ------------------------------------------------------------------ saved

test("a function saved here loads back with the same version and requests", async () => {
  const mood = ai("mood", { ...moodDef, definedIn: "shop", temperature: 0 });
  mood.demos = [{ review: "Broke", result: "unhappy" }];
  const manifest = JSON.parse(JSON.stringify(toManifest(mood)));
  assert.equal(manifest.language, "typescript");
  assert.equal(manifest.entry, "shop:mood");
  const again = fromManifest(manifest);
  assert.equal(again.version, mood.version);
  assert.equal(again.signatureId, mood.signatureId);
  assert.equal(again.module, "shop");
  assert.deepEqual(await again.render("x"), await mood.render("x"));
});

// ------------------------------------------------------------------ layouts

test("a template without an output pattern: the whole reply is the answer", async () => {
  const router = new FakeRouter(["  A short summary.  "]);
  const summarize = ai("summarize", {
    description: "Summarize.", input: { text: t.string() }, router,
    template: [{ role: "system", text: "You are terse. {instruction}" }, { role: "user", text: "Text: {text}" }],
  });
  assert.equal(await summarize("a long text"), "A short summary.");
  assert.equal(router.requests[0]!.system, "You are terse. Function: summarize\n\nSummarize.");
  assert.equal(text(router.requests[0]!.messages[0]!), "Text: a long text");
});

test("the json layout asks for one object and reads it; cot adds reasoning first", async () => {
  const router = new FakeRouter(['{"result": {"name": "Ana", "age": 31}}']);
  const person = ai("person", { description: "Who?", input: { text: t.string() },
    output: t.object({ name: t.string(), age: t.integer() }), adapter: "json", router });
  assert.deepEqual(await person("Ana, 31."), { name: "Ana", age: 31 });
  assert.equal((router.requests[0]!.config?.responseFormat as { type?: string } | undefined)?.type, "json_schema");
  const r2 = new FakeRouter(["<reasoning>\n3 pens at 7 each\n</reasoning>\n<result>\n21\n</result>"]);
  const solve = ai("solve", { description: "Solve it.", input: { problem: t.string() }, output: t.number(), module: "cot", router: r2 });
  const p = await solve.predict("7 pens at 3 dollars?");
  assert.equal(p.answer, 21);
  assert.equal(p.outputs["reasoning" as never], "3 pens at 7 each");
  assert.match(r2.requests[0]!.system as string, /Reason step by step in the 'reasoning' section/);
});

type Rec = Record<string, any>;

test("onUnreadable: record keeps an unreadable reply: no answer, the refusal said and recorded, a conversation goes on", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-log-"));
  const router = new FakeRouter(["no tags at all", { text: "", finish: "content_filter" }, "nope", "<result>\nhappy\n</result>"]);
  const keep = ai("mood", { ...moodDef, router, retries: 0, onUnreadable: "record", logCalls: folder });
  const p = await keep.predict("Where is it?");
  assert.equal(p.answer, undefined);
  assert.deepEqual(p.outputs, {});
  assert.equal(p.refusal?.code, "parse-missing-fields");
  assert.deepEqual((p.turn.meta as Rec)["refusal"], { ...p.refusal!.describe() });
  assert.equal(await keep("x"), undefined);                                   // a stopped reply is kept too
  const [rec, filtered] = logged(folder);
  assert.deepEqual(rec!.outputs, {});
  assert.ok(!("returned" in rec!));                                           // no answer, and no code changed one
  assert.equal(rec!.error.type, "Refusal");
  assert.equal(rec!.error.code, "parse-missing-fields");
  assert.equal(filtered!.error.code, "parse-filtered");
  // a conversation's turn is done with no outputs, and the next one goes on
  const chat = keep.conversation("kept");
  assert.equal(await chat("first"), undefined);
  assert.equal(await chat("second"), "happy");
  assert.deepEqual((await chat.turns()).map((turn) => turn.state), ["done", "done"]);
  // the default still raises; a provider error is never kept
  await assert.rejects(ai("mood", { ...moodDef, router: new FakeRouter(["nope"]), retries: 0 })("x"), (e: unknown) => lmcc.isRefusal(e));
  assert.throws(() => ai("mood", { ...moodDef, onUnreadable: "keep" as never }), /onUnreadable is "raise" or "record"/);
});

test("t.choice: what each answer means reaches the model as every language writes it", async () => {
  const router = new FakeRouter(["<urgency>\nhigh\n</urgency>\n<result>\nbilling\n</result>"]);
  const route = ai("route", {
    description: "Which team answers?", router,
    input: {
      message: { shape: t.string(), desc: "the customer's words" },
      channel: t.choice(["email", "chat"], { email: "written at leisure, often long", chat: "short,\n   often several messages" }),
    },
    outputs: {
      urgency: { shape: t.choice(["low", "high"], { high: "the customer is losing money or time now", low: "  " }), desc: "how soon to answer" },
      result: t.choice(["shipping", "billing"], { shipping: "where the parcel is, damage in transit", billing: "charges, refunds" }),
    },
  });
  const contract = JSON.parse(readFileSync(new URL("../../contract/cases/functions/18-choices-described-one-by-one.json", import.meta.url), "utf8"));
  assert.equal(route.version, contract.expect.version);
  const answer = await route({ message: "Charged twice", channel: "chat" });
  assert.equal(answer, "billing");
  assert.match(router.requests[0]!.system as string, /- channel: one of:\n {2}- email: written at leisure, often long\n {2}- chat: short, often several messages\n/);
  // the shape is t.enum's; inside a record, the meanings are the property's description
  assert.deepEqual(t.choice(["a", "b"]), t.enum("a", "b"));
  assert.deepEqual((t.object({ a: t.choice(["p", "q"], { q: "yes" }) }) as Rec)["properties"]["a"], { enum: ["p", "q"], type: "string", description: "one of:\n  - q: yes" });
  assert.throws(() => t.choice(["a", "b"], { c: "no" } as never), /"c" is given a meaning, but it is not one of the answers/);
  assert.throws(() => t.choice(["a", "a"]), /twice/);
});

test("describe() inside a shape is the shape with its description: the same request, the same version", async () => {
  // Written either way, a member's words are its schema's description (as R's record(x = described(...))
  // and Python's Field(description=...) write them). Before, t.object kept describe()'s { shape, desc }
  // as the member's schema: the model was sent it as is, and any answer fitted it.
  const reply = '<result>\n{"goal": "buy a gift", "tags": ["gift"], "level": "high", "who": null, "days": {"a": 2}}\n</result>';
  const urgency = () => t.choice(["low", "high"], { high: "losing money now" });
  const writtenRouter = new FakeRouter([reply]);
  const spelledRouter = new FakeRouter([reply]);
  const written = ai("visitor", {
    description: "Who is this visitor?", router: writtenRouter,
    input: { page: t.string() },
    output: t.object({
      goal: describe(t.string(), "the one thing they came to do"),
      tags: t.list(describe(t.string(), "a word for the page")),
      level: urgency(),
      who: t.optional(describe(t.string(), "who they are, if the page says")),
      days: t.record(describe(t.integer(), "whole days")),
    }),
  });
  const spelled = ai("visitor", {
    description: "Who is this visitor?", router: spelledRouter,
    input: { page: t.string() },
    output: t.object({
      goal: t.string({ description: "the one thing they came to do" }),
      tags: t.list(t.string({ description: "a word for the page" })),
      level: urgency(),
      who: t.optional(t.string({ description: "who they are, if the page says" })),
      days: t.record(t.integer({ description: "whole days" })),
    }),
  });
  const shape = (t.object({ goal: describe(t.string(), "the one thing they came to do") }) as Rec)["properties"]["goal"];
  assert.deepEqual(shape, { type: "string", description: "the one thing they came to do" });
  assert.equal(written.version, spelled.version);
  assert.deepEqual(written.interface, spelled.interface);
  const answer = await written({ page: "https://example.com" });
  await spelled({ page: "https://example.com" });
  assert.deepEqual(answer, { goal: "buy a gift", tags: ["gift"], level: "high", who: null, days: { a: 2 } });
  const sent = (router: FakeRouter) => JSON.stringify(router.requests[0]);
  assert.equal(sent(writtenRouter), sent(spelledRouter));
  assert.doesNotMatch(sent(writtenRouter), /"desc"|"shape"/);
  assert.match(sent(writtenRouter), /the one thing they came to do/);
  // a described choice: its words, then what its answers mean, as everywhere a choice is described (and
  // the one way to give a choice inside a record words of its own: t.choice takes no description)
  assert.equal((t.object({ level: describe(urgency(), "how soon to act") }) as Rec)["properties"]["level"]["description"], "how soon to act\n  - high: losing money now");
  // the words given to describe() are the member's, as at the top (a field's desc over its shape's description)
  assert.equal((t.list(describe(t.string({ description: "old" }), "new")) as Rec)["items"]["description"], "new");
});

test("apiKey and baseUrl: a key and an address of the function's own; the address is saved, the key never", async () => {
  const { createServer } = await import("node:http");
  const seen: { url: string; auth: string }[] = [];
  const server = createServer((req, res) => {
    seen.push({ url: req.url ?? "", auth: String(req.headers["authorization"] ?? "") });
    req.resume();
    req.on("end", () => { res.writeHead(400, { "content-type": "application/json" }); res.end('{"error":{"message":"seen"}}'); });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const url = `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`;
  try {
    const fn = ai("mood", { ...moodDef, lm: "openai:qwen3", apiKey: "sk-local", baseUrl: url, apiRetries: 0 });
    await assert.rejects(fn("x"));
    assert.equal(seen.length, 1);
    assert.match(seen[0]!.url, /^\/v1\//);                       // the request went to the function's own address
    assert.equal(seen[0]!.auth, "Bearer sk-local");                 // with its own key
    const manifest = toManifest(fn) as Rec;
    assert.equal(manifest["nodes"][manifest["entry"]]["ai"]["settings"]["base_url"], url);
    assert.ok(!JSON.stringify(manifest).includes("sk-local"));      // a key is never saved
    assert.equal(fromManifest(manifest).settings.baseUrl, url);
  } finally {
    server.close();
  }
  assert.throws(() => ai("mood", { ...moodDef, baseUrl: 3 as never }), /baseUrl is text/);
});
