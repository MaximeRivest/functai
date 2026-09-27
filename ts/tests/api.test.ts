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
