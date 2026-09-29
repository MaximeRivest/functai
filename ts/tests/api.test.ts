/** The TypeScript API itself: one input's value alone, call options, cancelling, map. Offline. */

import assert from "node:assert/strict";
import { test } from "node:test";
import * as z from "zod";
import { ai, Cancelled, configure, t, tool } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

const echo = (router: FakeRouter) => ai("echo", { description: "Echo.", input: { text: t.string() }, router });
const lastText = (router: FakeRouter) => (router.requests.at(-1)!.messages.at(-1)!.parts as unknown as { text?: string }[]).map((p) => p.text ?? "").join("");

test("a call's own settings beat the function's, for that call only", async () => {
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const f = ai("echo", { description: "Echo.", input: { text: t.string() }, router, temperature: 0.5 });
  await f("a", { temperature: 0.1, lm: "gpt-4.1-nano" });
  assert.equal(router.requests[0]!.config?.temperature, 0.1);
  assert.equal(router.requests[0]!.model, "gpt-4.1-nano");
  await f("b");
  assert.equal(router.requests[1]!.config?.temperature, 0.5);
});

test("a signal cancels a call: before it starts, and while it waits", async () => {
  const f = echo(new FakeRouter([], () => "<result>\nok\n</result>"));
  await assert.rejects(f("x", { signal: AbortSignal.abort() }), Cancelled);
  const slow = { resolve: (m: string) => ({ provider: "openai", model: m }),
    complete: (_req: unknown, opts?: { signal?: AbortSignal }) => new Promise<never>((_, reject) => {
      opts?.signal?.addEventListener("abort", () => reject(new Error("aborted by the caller")), { once: true });
    }) };
  const g = ai("echo", { description: "Echo.", input: { text: t.string() }, router: slow as never });
  const c = new AbortController();
  const pending = g("x", { signal: c.signal });
  setTimeout(() => c.abort(), 5);
  await assert.rejects(pending, Cancelled);
});

test("map answers every item in order, a few at a time, and stops at the first failure", async () => {
  let inFlight = 0, most = 0;
  const router = new FakeRouter([], (req) => {
    const text = (req.messages.at(-1)!.parts as unknown as { text?: string }[]).map((p) => p.text ?? "").join("");
    return `<result>\n${text.includes("bad") ? "" : text.match(/<text>\n(.*)\n<\/text>/)![1]!.toUpperCase()}\n</result>`;
  });
  const orig = router.complete.bind(router);
  router.complete = async (req) => {
    inFlight++; most = Math.max(most, inFlight);
    await new Promise((r) => setTimeout(r, 2));
    try { return await orig(req); } finally { inFlight--; }
  };
  const f = echo(router);
  assert.deepEqual(await f.map(["a", "b", "c", "d", "e"], { concurrency: 2 }), ["A", "B", "C", "D", "E"]);
  assert.equal(most, 2);
  const strict = ai("strict", { description: "Echo.", input: { text: t.string() }, output: t.enum("A", "B"), router, retries: 0 });
  await assert.rejects(strict.map(["a", "bad", "b"], { concurrency: 1 }));
  assert.equal(lastText(router).includes("<text>\nb\n</text>"), false);   // nothing started after the failure
});

test("a tool's input is typed from its shape; zod works through Standard Schema", async () => {
  const seen: string[] = [];
  const lookup = tool("lookup_order", { description: "Look up an order.", input: { order: z.string() } },
    ({ order }) => { seen.push(order.toUpperCase()); return "stuck"; });
  assert.equal(lookup.name, "lookup_order");
  assert.deepEqual(lookup.parameters, { type: "object", properties: { order: { type: "string" } }, required: ["order"] });
  const router = new FakeRouter([{ calls: [{ id: "c1", name: "lookup_order", input: { order: "a-1" } }] }, "<result>\nStuck.\n</result>"]);
  const helper = ai("helper", { description: "Help.", input: { question: t.string() }, tools: [lookup], router });
  assert.equal(await helper("Where is a-1?"), "Stuck.");
  assert.deepEqual(seen, ["A-1"]);
});

test("an input may be left out when its schema allows it: sent as null, or as the schema's default", async () => {
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const f = ai("summarize", {
    description: "Summarize.",
    input: { text: t.string(), note: t.optional(t.string()), tone: z.string().default("plain"), max: z.number().optional(),
             lang: z.string().nullable() },
    router,
  });
  await f({ text: "hi", lang: null });
  assert.match(lastText(router), /<note>\nnull\n<\/note>/);
  assert.match(lastText(router), /<tone>\nplain\n<\/tone>/);            // the default, from the schema
  assert.match(lastText(router), /<max>\nnull\n<\/max>/);
  await assert.rejects(f({ text: "hi" } as never), /needs lang/);          // nullable is not optional: give it, null or not
  await assert.rejects(f({ text: "hi", lang: null, max: "ten" } as never), /input max: .*expected number/i);   // checked by its schema
  // the definition's (and the interface's) shape holds the default an input left out is sent with; the signature does not (functions.md)
  const shape = (n: string) => f.definition.inputs.find((x) => x.name === n)!.shape;
  const sent = (n: string) => f.signature.fields.find((x) => x.name === n)!.shape;
  assert.deepEqual(shape("max"), { anyOf: [{ type: "number" }, { type: "null" }], default: null });   // left out is null, and the shape says so
  assert.deepEqual(sent("max"), { anyOf: [{ type: "number" }, { type: "null" }] });
  assert.deepEqual(shape("tone"), { type: "string", default: "plain" });
  assert.deepEqual(sent("tone"), { type: "string" });
  assert.deepEqual(f.interface.inputs.map((x) => [x.name, x.optional ?? false]),
    [["text", false], ["note", true], ["tone", true], ["max", true], ["lang", false]]);
});

test("with one required input, its value alone is the call; the optional ones may be left out", async () => {
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const f = ai("summarize", { description: "Summarize.", input: { text: t.string(), note: t.optional(t.string()) }, router });
  assert.equal(await f("hello"), "ok");
  assert.match(lastText(router), /<text>\nhello\n<\/text>\n<note>\nnull\n<\/note>/);
  await f({ text: "hi", note: "be brief" });
  assert.match(lastText(router), /<note>\nbe brief\n<\/note>/);
  const all = ai("greet", { description: "Greet.", input: { name: t.optional(t.string()) }, router });
  assert.equal(await all(), "ok");                                          // every input optional: no argument at all
});
