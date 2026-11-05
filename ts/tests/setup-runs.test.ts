/**
 * What a program runs on (contract/calls.md, *Setups*; design/18), through
 * real calls with a fake model: another model or a tool rewritten is another
 * setup of the same version; the wall a program runs behind; what code
 * spent; a saved folder's setup, and what differs between two.
 */

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { type Request } from "@lm15/lm15";
import { ai, delegate, keepPictures, limits, module, PluginError, ReachError, save, setupDiff, spend, t, tool, toManifest, withSettings } from "../src/index.ts";
import { FakeRouter, type Reply } from "./fake.ts";

type Rec = Record<string, any>;
const XML = (s: string) => `<result>\n${s}\n</result>`;
const results = (req: Request): Rec[] => req.messages.flatMap((m) => m.parts).filter((p) => p.type === "tool_result") as unknown as Rec[];
const LM = "openai:gpt-4.1-mini";

const mood = ai("mood", { description: "How does the writer feel?", input: { text: t.string() }, output: t.string() });

const look = (answer: string) => tool("look", {
  description: "Look at a page.", input: { page: t.string() }, effects: "reads", needs: { domains: ["shop.example"] },
}, ({ page }) => `page ${page}: ${answer}`);

const stockWith = (tl: ReturnType<typeof look>, router?: FakeRouter) => ai("stock", {
  description: "Is the item in stock?", input: { item: t.string() }, output: t.string(), tools: [tl],
  ...(router ? { router: router as never } : {}),
});

const records = (folder: string): Rec[] => readdirSync(folder, { recursive: true }).map(String).filter((f) => f.endsWith(".jsonl"))
  .flatMap((f) => readFileSync(join(folder, f), "utf8").trim().split("\n").map((l) => JSON.parse(l)));

test("another model is another setup of the same version; where it is reached is not in it", () => {
  const a = withSettings({ lm: LM }, () => mood.setup);
  assert.equal(withSettings({ lm: "gpt-4.1-mini" }, () => mood.setup), a);
  assert.notEqual(withSettings({ lm: "anthropic:claude-haiku-4-5" }, () => mood.setup), a);
  assert.notEqual(withSettings({ lm: LM, temperature: 0.2 }, () => mood.setup), a);
  assert.equal(withSettings({ lm: LM, baseUrl: "https://gw.example/v1", cacheReplies: true, apiRetries: 9 }, () => mood.setup), a);
});

test("code a version cannot read: the host names it (codeId), in every setup worked out under it, a module's too", () => {
  const both = module("both", { input: { a: t.string() }, output: t.string(), uses: [mood] }, async ({ a }) => mood(a));
  withSettings({ lm: LM }, () => {
    const plain = { mood: mood.setup, both: both.setup };
    assert.equal(mood.setupDocument()["code_id"], undefined);      // nothing said: setups as before
    withSettings({ codeId: "sha256:aaaa" }, () => {
      assert.equal(mood.setupDocument()["code_id"], "sha256:aaaa");
      assert.equal(both.setupDocument()["code_id"], "sha256:aaaa");
      assert.notEqual(mood.setup, plain.mood);
      assert.notEqual(both.setup, plain.both);
      assert.equal(mood.version, withSettings({ codeId: null }, () => mood.version));   // a setting, not the version
    });
    assert.notEqual(withSettings({ codeId: "sha256:aaaa" }, () => both.setup), withSettings({ codeId: "sha256:bbbb" }, () => both.setup));
    assert.equal(withSettings({ codeId: null }, () => mood.setup), plain.mood);
  });
  for (const bad of ["", "   ", "x".repeat(201), 7]) assert.throws(() => withSettings({ codeId: bad as never }, () => 0), /codeId/);
  const own = ai("own", { description: "d", input: { x: t.string() }, output: t.string(), codeId: "rev-12" } as never);
  assert.equal(withSettings({ lm: LM }, () => own.setupDocument()["code_id"]), "rev-12");
});

test("rewriting a tool is another setup of the same version; what its text cannot say, codeId says", () => {
  const spec = { description: "Look at a page.", input: { page: t.string() }, effects: "reads" as const };
  const a = stockWith(tool("look", spec, ({ page }) => `page ${page}: in stock`));
  const b = stockWith(tool("look", spec, ({ page }) => `page ${page}: sold out`));
  assert.equal(a.version, b.version);                             // what the model is told is the same
  withSettings({ lm: LM }, () => {
    assert.notEqual(a.setup, b.setup);
    const entry = (a.setupDocument()["tools"] as Rec)["look"];
    assert.equal(entry.code.language, "typescript");
    assert.match(entry.code.code, /^sha256:[0-9a-f]{64}$/);
    // a tool made by a factory: one text, two behaviours. The text alone cannot tell them apart; codeId can
    const made = (answer: string, codeId?: string) => tool("look", { ...spec, ...(codeId ? { codeId } : {}) }, ({ page }) => `page ${page}: ${answer}`);
    assert.equal(stockWith(made("in stock")).setup, stockWith(made("sold out")).setup);
    assert.notEqual(stockWith(made("in stock", "in-stock")).setup, stockWith(made("sold out", "sold-out")).setup);
    assert.equal((stockWith(made("x", "v3")).setupDocument()["tools"] as Rec)["look"].code.id, "v3");
  });
});

test("a program delegated to as a tool counts by its setup; escalation names its backup", () => {
  const caring = ai("caring", { description: "Answer kindly.", input: { message: t.string() }, output: t.string(), tools: [delegate(mood)] });
  withSettings({ lm: LM }, () => {
    assert.deepEqual((caring.setupDocument()["tools"] as Rec)["mood"], { setup: mood.setup });
    const careful = ai("careful", { description: "How does the writer feel?", input: { text: t.string() }, output: t.string(), escalateTo: mood });
    assert.deepEqual(careful.setupDocument()["escalate"], { to: { setup: mood.setup }, below: 0.9 });
  });
});

test("plugins count with their settings", () => {
  const a = withSettings({ lm: LM, plugins: [keepPictures(2)] }, () => {
    assert.deepEqual(mood.setupDocument()["plugins"], [{ name: "keep-pictures", version: "1.0.0", config: { last: 2, at_once: 1 } }]);
    return mood.setup;
  });
  assert.notEqual(withSettings({ lm: LM, plugins: [keepPictures(3)] }, () => mood.setup), a);
});

test("reach: open, a wall made from the needs, a wall that says what it lets through", () => {
  const stock = stockWith(look("in stock"));
  withSettings({ lm: LM }, () => assert.equal(stock.setupDocument()["reach"], "open"));
  withSettings({ lm: LM, wall: "srt" }, () => assert.deepEqual(stock.setupDocument()["reach"], { domains: ["shop.example"], secrets: [] }));
  withSettings({ lm: LM, wall: { name: "srt", domains: ["*.example", "shop.example"], limits: { seconds: 60 } } }, () =>
    assert.deepEqual(stock.setupDocument()["reach"], { domains: ["*.example", "shop.example"], secrets: [], limits: { seconds: 60 } }));
  assert.throws(() => tool("x", { input: {}, needs: { domains: ["not a host"] } }, () => ""), TypeError);
});

test("a wall that does not let through what a program needs refuses it before any request, recorded", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-setup-"));
  const router = new FakeRouter([], () => XML("yes"));
  const stock = stockWith(look("in stock"), router);
  await withSettings({ lm: LM, logCalls: folder, wall: { name: "srt", domains: ["api.other.example"] } }, async () => {
    await assert.rejects(stock({ item: "socks" }), (err: unknown) => err instanceof ReachError && err.code === "needs-unmet"
      && JSON.stringify(err.missing) === JSON.stringify({ domains: ["shop.example"], secrets: [] }));
  });
  assert.equal(router.requests.length, 0);                         // nothing sent
  const [rec] = records(folder);
  assert.equal(rec!.error.type, "ReachError");
  assert.deepEqual(rec!.placement, { wall: "srt" });
  await withSettings({ lm: LM, wall: { name: "srt", domains: ["*.example"] } }, async () => assert.equal(await stock({ item: "socks" }), "yes"));
});

test("a record says its setup and where it ran", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-setup-"));
  const m = ai("m", { description: "How does the writer feel?", input: { text: t.string() }, output: t.string(), router: new FakeRouter([XML("calm")]) as never });
  const setup = await withSettings({ lm: LM, logCalls: folder, baseUrl: "https://gw.example/v1", wall: "srt" }, async () => {
    await m({ text: "fine" });
    return m.setup;
  });
  const [rec] = records(folder);
  assert.equal(rec!.program.setup, setup);
  assert.deepEqual(rec!.placement, { wall: "srt", base_url: "https://gw.example/v1" });
  assert.ok(!("spent" in rec!));
});

test("what code spent is recorded and counted by limits", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-setup-"));
  const browse = tool("browse", { description: "Browse a page with a paid service.", input: { url: t.string() }, effects: "reads" }, () => {
    spend(0.04);
    spend(0.01);
    return "a page";
  });
  const router = new FakeRouter([], (req): Reply => {
    if (!results(req).length && req.tools?.length) return { calls: [{ id: "c1", name: "browse", input: { url: "https://a.example/" } }] };
    return XML("fine");
  });
  const walker = ai("walker", { description: "Walk the site.", input: { url: t.string() }, output: t.string(), tools: [browse], router: router as never });
  const study = module("study", { input: { url: t.string() }, output: t.string(), uses: [walker] }, async ({ url }) => {
    spend(0.2);
    return walker({ url });
  });
  await withSettings({ lm: LM, logCalls: folder }, async () => assert.equal(await study({ url: "https://a.example/" }), "fine"));
  const byName = Object.fromEntries(records(folder).map((r) => [r.program.name, r]));
  assert.deepEqual(byName["walker"].spent, { usd: 0.05 });
  assert.deepEqual(byName["study"].spent, { usd: 0.2 });
  const prices = { "openai:gpt-4.1-mini": { inputPerMillion: 0, outputPerMillion: 0 } };
  await withSettings({ lm: LM, plugins: [limits({ cost: 0.1, prices })] }, async () => {
    await assert.rejects(study({ url: "https://a.example/" }), (err: unknown) => err instanceof PluginError && err.code === "plugin-stopped" && /cost 0\.1/.test(err.message));
  });
  assert.throws(() => spend(-1), RangeError);
  spend(1);                                                        // outside any call: nothing
});

test("a module names each step it reaches, and its own needs", () => {
  const study = module("study", { input: { url: t.string() }, output: t.string(), uses: [mood], needs: { domains: ["api.browser-use.com"], secrets: ["BU_KEY"] } },
    async ({ url }) => mood({ text: url }));
  withSettings({ lm: LM, wall: "srt" }, () => {
    const doc = study.setupDocument();
    assert.equal(doc["version"], study.version);
    assert.deepEqual(Object.values(doc["ai"] as Rec), [mood.setup]);
    assert.deepEqual(doc["reach"], { domains: ["api.browser-use.com"], secrets: ["BU_KEY"] });
  });
});

test("a saved agent says its tools, its setup and its version; two folders are compared", () => {
  const stock = stockWith(look("in stock"));
  const answer = module("answer", { input: { item: t.string() }, output: t.string(), uses: [stock] }, async ({ item }) => stock({ item }));
  const a = withSettings({ lm: LM }, () => toManifest(answer));
  const b = withSettings({ lm: "anthropic:claude-haiku-4-5", temperature: 0.3 }, () => toManifest(answer));
  const node = Object.values(a["nodes"] as Rec).find((n: Rec) => n.kind === "module");
  assert.equal(node.version, answer.version);
  assert.equal(node.setup.version, answer.version);
  const agent = Object.values(a["nodes"] as Rec).find((n: Rec) => n.kind === "ai");
  const [entry] = agent.ai.tools;
  assert.deepEqual(entry.tool.name, "look");
  assert.equal(entry.code.language, "typescript");
  assert.deepEqual(entry.needs, { domains: ["shop.example"], secrets: [] });
  assert.equal(entry.effects, "reads");
  const diff = setupDiff(a, b);
  const key = Object.keys(a["nodes"] as Rec).find((k) => (a["nodes"] as Rec)[k].kind === "ai")!;
  assert.deepEqual(diff.find((d) => d.program === key && d.field === "model.provider"), { program: key, field: "model.provider", before: "openai", after: "anthropic" });
  assert.deepEqual(diff.find((d) => d.program === key && d.field === "config")?.after, { temperature: 0.3 });
  const folder = mkdtempSync(join(tmpdir(), "functai-saved-"));
  withSettings({ lm: LM }, () => save(answer, join(folder, "a")));
  assert.deepEqual(setupDiff(join(folder, "a"), a), []);
  withSettings({ lm: LM }, () => assert.deepEqual(setupDiff(answer, join(folder, "a")).filter((d) => d.program !== "answer"), []));
});
