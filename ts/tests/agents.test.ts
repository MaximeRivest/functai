/**
 * Long agents and pipelines: tools that answer with pictures, notes from
 * code, the step hook and its built-in plugins (limits, keepPictures), the
 * call log in a store of one's own, and results that come later; then the
 * contract's cases of them (../../contract/cases/agents), the same Python
 * passes. Offline: a fake model.
 */

import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { image, Part, type Request } from "@lm15/lm15";
import { ai, calls, ConversationError, image as fimage, keepPictures, later, limits, MemoryCalls, MemoryConversations, module, note, Plugin,
  PluginError, rate, rated, Spent, t, tool, Waiting, withSettings, keptLog } from "../src/index.ts";
import { clock, internals } from "../src/conversations.ts";
import { picturesToDrop, LAST_STEP, PICTURE_GONE } from "../src/builtins.ts";
import { resultOf, resultText } from "../src/results.ts";
import { FakeRouter, type Reply } from "./fake.ts";
import { cases } from "./cases.ts";

type Rec = Record<string, any>;
const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
const XML = (s: string) => `<result>\n${s}\n</result>`;
void fimage;

/** The tool results a request sends, as lm15 parts. */
const results = (req: Request): Rec[] => req.messages.flatMap((m) => m.parts).filter((p) => p.type === "tool_result") as unknown as Rec[];
const pictures = (req: Request) => results(req).reduce((n, p) => n + (p["content"] as Rec[]).filter((c) => c["type"] === "image").length, 0);
const texts = (part: Rec) => (part["content"] as Rec[]).filter((c) => c["type"] === "text").map((c) => c["text"]);

/** A model that asks for these tool calls, one per reply, then answers; offered no tool, it answers at once. */
function script(steps: [string, Rec][], answer: (req: Request) => string = () => "done") {
  return new FakeRouter([], (req): Reply => {
    const n = results(req).length;
    if (n < steps.length && req.tools?.length) return { calls: [{ id: `c${n + 1}`, name: steps[n]![0], input: steps[n]![1] }] };
    return XML(answer(req));
  });
}

const look = tool("look", { description: "A screenshot of a page, with its title.", input: { page: t.string() }, effects: "reads" },
  ({ page }) => [`page ${page}`, image({ data: PNG, mediaType: "image/png" })]);

const browser = (router: FakeRouter, extra: Rec = {}) =>
  ai("browse", { description: "Look at pages to do the task.", input: { task: t.string() }, output: t.string(), tools: [look],
    router: router as never, lm: "gpt-4.1-mini", ...extra });

const pages = (n: number): [string, Rec][] => Array.from({ length: n }, (_, i) => ["look", { page: String(i) }]);

// ------------------------------------------------------------------ pictures from tools

test("a tool answers with a picture: the model sees it, the event has its text and its parts", async () => {
  const router = script(pages(1));
  const s = browser(router).stream("find the price");
  const events: Rec[] = [];
  for await (const e of s.events()) events.push(e as Rec);
  assert.equal(await s, "done");
  const [part] = results(router.requests.at(-1)!);
  assert.deepEqual((part!["content"] as Rec[]).map((c) => c["type"]), ["text", "image"]);
  const res = events.find((e) => e["kind"] === "tool_result")!;
  assert.equal(res["output"], "page 0");
  assert.deepEqual(res["parts"], [{ type: "text", text: "page 0" }, { type: "image", media_type: "image/png", data: PNG }]);
});

test("a picture read from a file is given its bytes", () => {
  const dir = mkdtempSync(join(tmpdir(), "functai-agents-"));
  writeFileSync(join(dir, "shot.png"), Buffer.from(PNG, "base64"));
  const out = resultOf(image({ path: join(dir, "shot.png") }));
  assert.deepEqual(out, [{ type: "image", media_type: "image/png", data: PNG }]);
  assert.equal(resultOf({ a: 1 }), '{"a":1}');
  assert.equal(resultText([{ type: "text", text: "a" }, { type: "image", media_type: "image/png", data: PNG }, { type: "text", text: "b" }]), "a\nb");
});

test("a toolResult hook sees the parts and may change them", async () => {
  const seen: unknown[] = [];
  const p = new Plugin("caption").toolResult((r) => {
    seen.push(r.output);
    return { output: [...(r.output as Rec[]), "(the price is at the top right)"] };
  });
  const router = script(pages(1));
  assert.equal(await browser(router, { plugins: [p] })("find"), "done");
  assert.equal(((seen[0] as Rec[])[1]!)["type"], "image");
  assert.deepEqual(texts(results(router.requests.at(-1)!)[0]!), ["page 0", "(the price is at the top right)"]);
});

// ------------------------------------------------------------------ the step hook

test("the step hook sees each step, and can take the tools away with a reminder", async () => {
  const seen: unknown[] = [];
  const p = new Plugin("wrap").step((s) => {
    seen.push([s.step, s.maxSteps, s.results.map((r) => r.invocation), [...s.tools], s.spent.requests]);
    return s.step === 3 ? { tools: [], reminder: "Answer now." } : undefined;
  });
  const router = script(pages(10));
  assert.equal(await browser(router, { plugins: [p], maxSteps: 5 })("find"), "done");
  assert.deepEqual(seen, [[2, 5, [1], ["look"], 1], [3, 5, [1, 2], ["look"], 2]]);
  const last = router.requests.at(-1)!;
  assert.ok(!last.tools?.length);
  assert.equal(texts(results(last).at(-1)!).at(-1), "Answer now.");
});

test("earlier results shown otherwise: the request changes, the call's record keeps the change and every picture", async () => {
  const p = new Plugin("forget", { version: "2.0.0" }).step((s) => {
    const old = s.results.slice(0, -1).filter((r) => r.pictures);
    return old.length ? { replace: Object.fromEntries(old.map((r) => [r.invocation, "(an earlier screenshot)"])) } : undefined;
  });
  const router = script(pages(3));
  const log = new MemoryCalls();
  assert.equal(await browser(router, { plugins: [p], logCalls: log })("find"), "done");
  assert.deepEqual(router.requests.map(pictures), [0, 1, 1, 1]);
  const rec = log.read().find((r: Rec) => r["program"]["name"] === "browse") as Rec;
  const steps = (rec["changes"] as Rec[]).filter((c) => c["hook"] === "step");
  assert.deepEqual(steps[0], { plugin: "forget", version: "2.0.0", hook: "step", change: { replace: { "1": "(an earlier screenshot)" } } });
  assert.ok((rec["exchanges"] as Rec[]).every((ex) => "request_hash" in ex));
  const tools = (rec["steps"] as Rec[]).filter((st) => st["kind"] === "tool");
  assert.equal(tools[0]!["output"][1]["type"], "image");
});

test("a step handler stops the call; a change it cannot make is refused", async () => {
  const brake = new Plugin("brake").step((s) => s.step >= 3 ? { stop: "too many pages" } : undefined);
  await assert.rejects(browser(script(pages(10)), { plugins: [brake] })("find"),
    (e: unknown) => e instanceof PluginError && e.code === "plugin-stopped" && e.message.includes("too many pages"));
  const bad = new Plugin("bad").step(() => ({ tools: ["fly"] }));
  await assert.rejects(browser(script(pages(2)), { plugins: [bad] })("find"), (e: unknown) => e instanceof PluginError && e.code === "plugin-change");
  const wrong = new Plugin("wrong").step(() => ({ sections: ["no"] }));
  await assert.rejects(browser(script(pages(2)), { plugins: [wrong] })("find"), (e: unknown) => e instanceof PluginError && e.code === "plugin-change");
});

// ------------------------------------------------------------------ limits and keepPictures

test("limits wrap up at the last step instead of failing", async () => {
  const router = script(pages(10));
  assert.equal(await browser(router, { plugins: [limits({ seconds: 3600 })], maxSteps: 3 })("find"), "done");
  assert.equal(router.requests.length, 3);
  assert.equal(texts(results(router.requests.at(-1)!).at(-1)!).at(-1), LAST_STEP);
});

test("a token limit wraps the loop up, then stops the next call of the tree", async () => {
  const lim = limits({ tokens: 1 });
  const router = script(pages(10));
  const b = browser(router, { plugins: [lim] });
  assert.equal(await b("find"), "done");
  assert.equal(router.requests.length, 2);
  assert.ok(texts(results(router.requests.at(-1)!).at(-1)!).at(-1)!.startsWith("This task has reached its limit (1 tokens)"));
  const twice = module("twice", { input: { task: t.string() }, output: t.string() }, async ({ task }) => (await browser(script(pages(10)))(task)) + (await browser(script(pages(10)))(task)));
  await assert.rejects(withSettings({ plugins: [lim] }, () => twice({ task: "find" })),
    (e: unknown) => e instanceof PluginError && e.code === "plugin-stopped" && e.message.includes("limit reached: 1 tokens"));
});

test("a cost limit needs prices, refuses a model it cannot price, and passes one it can", async () => {
  assert.throws(() => limits({ cost: 1 }));
  assert.throws(() => limits({}));
  const priced = limits({ cost: 1, prices: { "other": { inputPerMillion: 1, outputPerMillion: 2 } } });
  await assert.rejects(browser(script(pages(1)), { plugins: [priced] })("find"), (e: unknown) => e instanceof PluginError && e.message.includes("no price is known for gpt-4.1-mini"));
  const flat = { input_per_million: 1, output_per_million: 2, cache_read_per_million: 1, cache_write_per_million: 1 };
  assert.equal(await browser(script(pages(1)), { plugins: [limits({ cost: 1000, prices: { "gpt-4.1-mini": flat }, then: "stop" })] })("find"), "done");
});

test("keepPictures shows only the newest, and drops in batches for the prompt cache", async () => {
  const r1 = script(pages(5));
  assert.equal(await browser(r1, { plugins: [keepPictures(2)] })("find"), "done");
  assert.deepEqual(r1.requests.map(pictures), [0, 1, 2, 2, 2, 2]);
  assert.deepEqual(texts(results(r1.requests.at(-1)!)[0]!), ["page 0", PICTURE_GONE]);
  const r2 = script(pages(7));
  assert.equal(await browser(r2, { plugins: [keepPictures(2, { atOnce: 3 })], maxSteps: 10 })("find"), "done");
  assert.deepEqual(r2.requests.map(pictures), [0, 1, 2, 3, 4, 2, 3, 4]);
});

// ------------------------------------------------------------------ notes

test("code says what happened: a note event, its data left out of a kept form that keeps no content", async () => {
  const write = ai("write", { input: { x: t.string() }, output: t.string(), router: new FakeRouter([XML("ok")]) as never, lm: "gpt-4.1-mini" });
  const pipeline = module("pipeline", { input: { url: t.string() }, output: t.string() }, async ({ url }) => {
    note("screen captured", { screen: 4 });
    return write({ x: url });
  });
  const s = pipeline.stream({ url: "a.com" });
  const events: Rec[] = [];
  for await (const e of s.events()) events.push(e as Rec);
  assert.equal(await s, "ok");
  const n = events.find((e) => e["kind"] === "note")!;
  assert.deepEqual([n["name"], n["data"], n["function"]], ["screen captured", { screen: 4 }, "pipeline"]);
  const kept = keptLog(events as never, Object.fromEntries(events.filter((e) => e["kind"] === "started").map((e) => [e["call"],
    { inputs: Object.fromEntries(Object.keys(e["inputs"]).map((k) => [k, false])), outputs: { result: true } }])) as never);
  const k = kept.find((e: Rec) => e["kind"] === "note") as Rec;
  assert.equal(k["name"], "screen captured");
  assert.ok(!("data" in k) && k["content"] === false);
  note("nothing watches");
  assert.throws(() => note(""));
});

// ------------------------------------------------------------------ the call log in a store

test("calls go to a store of one's own and are read back, rated; a store that fails never stops a call", async () => {
  const store = new MemoryCalls();
  const team = ai("team", { description: "Which team answers this message.", input: { message: t.string() }, output: t.string(),
    router: new FakeRouter([XML("billing")]) as never, lm: "gpt-4.1-mini", logCalls: store });
  const p = await team.predict({ message: "I was charged twice." });
  rate(p, "right", { folder: store });
  assert.deepEqual(store.read().map((r) => ["functai_call" in r, "functai_rating" in r]), [[true, false], [false, true]]);
  assert.equal(rated(team, { folder: store }).rows.length, 1);
  assert.equal(calls(team, { folder: store }).length, 1);
  let warned = false;
  const broken = { append: async () => { throw new Error("database away"); }, read: () => [] };
  const old = console.warn;
  console.warn = ((m: string) => { if (String(m).includes("could not log")) warned = true; }) as never;
  try {
    const f = ai("team2", { input: { message: t.string() }, output: t.string(), router: new FakeRouter([XML("billing")]) as never, lm: "gpt-4.1-mini", logCalls: broken });
    assert.equal(await f({ message: "x" }), "billing");
  } finally {
    console.warn = old;
  }
  assert.ok(warned);
});

// ------------------------------------------------------------------ results that come later

function harvesting() {
  const jobs: string[] = [];
  const start = tool("start_harvest", { description: "Visit a site in a cloud browser; its report comes back later.", input: { url: t.string() },
    effects: "changes" }, ({ url }) => {
    jobs.push(url);
    return later(`harvest:${jobs.length}`);
  });
  const weather = tool("weather", { description: "Today's weather.", input: { city: t.string() }, effects: "reads" }, ({ city }) => `sunny in ${city}`);
  return { jobs, start, weather };
}

const firstResult = (req: Request) => texts(results(req)[0]!).join("");

test("a turn waits for a later result, and goes on when it is delivered, from another process", async () => {
  const { jobs, start, weather } = harvesting();
  const router = script([["start_harvest", { url: "a.com" }], ["weather", { city: "Oslo" }]], (req) => "report: " + firstResult(req));
  const f = ai("researcher", { description: "Research the site.", input: { task: t.string() }, output: t.string(), tools: [start, weather],
    router: router as never, lm: "gpt-4.1-mini" });
  const store = new MemoryConversations();
  const chat = f.conversation("c1", { store });
  await assert.rejects(chat("look at a.com"), (e: unknown) => e instanceof Waiting && e.later.map((x) => x["key"]).join() === "harvest:1" && !e.approvals.length);
  assert.deepEqual(jobs, ["a.com"]);
  const asked = router.requests.length;
  const again = f.conversation("c1", { store });
  assert.equal(await again.deliver("harvest:1", ["3 pages", image({ data: PNG, mediaType: "image/png" })]), "report: 3 pages");
  assert.deepEqual(jobs, ["a.com"]);                                    // the tool did not run again
  assert.equal(router.requests.length - asked, 2);                       // no model answer paid twice
  assert.equal(await again.deliver("harvest:1", "late duplicate"), undefined);
  const kinds = (await store.read("c1", 0)).filter((r: Rec) => r["kind"] === "tool" || r["kind"] === "delivered").map((r: Rec) => [r["kind"], r["state"] ?? null]);
  assert.deepEqual(kinds, [["tool", "started"], ["tool", "later"], ["delivered", null], ["tool", "done"], ["tool", "done"]]);
});

test("a result delivered before the turn waits is used at once; outside a turn a later result is refused", async () => {
  const { jobs, start, weather } = harvesting();
  const f = ai("researcher", { description: "Research.", input: { task: t.string() }, output: t.string(), tools: [start, weather],
    router: script([["start_harvest", { url: "b.com" }]], (req) => "report: " + firstResult(req)) as never, lm: "gpt-4.1-mini" });
  const chat = f.conversation("c2", { store: new MemoryConversations() });
  await chat.deliver("harvest:1", "fast", { resume: false });
  assert.equal(await chat("look"), "report: fast");
  const g = ai("researcher", { description: "Research.", input: { task: t.string() }, output: t.string(), tools: [start, weather],
    router: script([["start_harvest", { url: "c.com" }]]) as never, lm: "gpt-4.1-mini" });
  await assert.rejects(g("look"), (e: unknown) => e instanceof ConversationError && e.code === "later-unavailable");
  assert.deepEqual(jobs, ["b.com", "c.com"]);
});

test("a waiting turn cannot resume before its result; its later event is shown", async () => {
  const { start, weather } = harvesting();
  const f = ai("researcher", { description: "Research.", input: { task: t.string() }, output: t.string(), tools: [start, weather],
    router: script([["start_harvest", { url: "d.com" }]]) as never, lm: "gpt-4.1-mini" });
  const chat = f.conversation("c3", { store: new MemoryConversations() });
  const s = chat.stream("look");
  const events: Rec[] = [];
  await assert.rejects((async () => { for await (const e of s.events()) events.push(e as Rec); await s; })(), (e: unknown) => e instanceof Waiting);
  assert.deepEqual(events.filter((e) => e["kind"] === "later").map((e) => [e["key"], e["name"], e["invocation"]]), [["harvest:1", "start_harvest", 1]]);
  const head = (await chat.turns()).at(-1)!;
  await assert.rejects(head.resume(), (e: unknown) => e instanceof ConversationError && e.code === "turn-state" && e.message.includes("harvest:1"));
  assert.equal((await head.later())[0]!["key"], "harvest:1");
});

// ------------------------------------------------------------------ the contract's cases (agents/)

const native = (v: unknown): unknown => Array.isArray(v) ? v.map(native) : v && typeof v === "object" ? Part.fromJSON(v as never) : v;

for (const [name, c] of cases("agents")) {
  test(`agents case ${name}`, async () => {
    if (c.kind === "result") {
      const result = resultOf(native(c.returned));
      assert.deepEqual(result, c.expect.result);
      assert.deepEqual(typeof result === "string" ? { output: result } : { output: resultText(result), parts: result }, c.expect.event);
    } else if (c.kind === "state") {
      const before = clock.now;
      clock.now = () => Date.parse(c.now) / 1000;
      try {
        const log = new internals.ConvLog().apply(c.records);
        const e = c.expect;
        assert.deepEqual(Object.fromEntries([...log.turns].map(([id, st]) => [id, st.state()])), e.states);
        assert.deepEqual(Object.fromEntries([...log.turns].filter(([, st]) => st.undelivered(log.delivered).length)
          .map(([id, st]) => [id, st.undelivered(log.delivered).map((x) => x["key"])])), e.later);
        assert.deepEqual(Object.fromEntries([...log.turns].filter(([, st]) => st.unfinished().length).map(([id, st]) => [id, st.unfinished().map((x) => x["invocation"])])), e.unfinished);
        assert.deepEqual(Object.fromEntries([...log.delivered].map(([k, r]) => [k, r["output"]])), e.delivered);
        assert.deepEqual(Object.fromEntries([...log.turns].filter(([, st]) => st.state() === "waiting")
          .map(([id, st]) => [id, !st.unanswered().length && !st.undelivered(log.delivered).length ? true : { refuses: "turn-state" }])), e.resumes);
      } finally {
        clock.now = before;
      }
    } else if (c.kind === "kept") {
      assert.deepEqual(keptLog(c.events, c.kept), c.expect.events);
    } else if (c.kind === "limits") {
      const plugin = limits(c.limits);
      const spent = new Spent(c.spent.exchanges, c.spent.seconds, c.spent.reported ?? 0);
      const at = c.at;
      const got = at.hook === "before_call"
        ? plugin.handlers.beforeCall![0]!({ spent, lm: at.lm })
        : plugin.handlers.step![0]!({ spent, step: at.step, maxSteps: at.max_steps, tools: at.tools, lastStep: at.step >= at.max_steps });
      assert.deepEqual(got ?? null, c.expect.change);
    } else if (c.kind === "spent") {
      const s = new Spent(c.exchanges, 0, c.reported ?? 0);
      assert.deepEqual(s.usage, c.expect.usage);
      assert.equal(s.tokens, c.expect.tokens);
      const cost = s.cost(c.prices);
      if (c.expect.cost === null) assert.equal(cost, null);
      else assert.ok(cost !== null && Math.abs(cost - c.expect.cost) < 1e-12, `${cost} != ${c.expect.cost}`);
    } else if (c.kind === "pictures") {
      assert.deepEqual(picturesToDrop(c.results, c.last, c.at_once), c.expect.replace);
    } else throw new Error(`a case of a kind this harness does not know: ${c.kind}`);
  });
}
