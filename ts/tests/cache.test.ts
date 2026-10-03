/** The reply cache, offline. */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ai, calls, clearCache, configure, t } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

const mood = (router: FakeRouter, more: Record<string, unknown> = {}) => ai("mood", {
  description: "How does the customer feel?", input: { review: t.string() }, output: t.enum("happy", "unhappy"), router, ...more,
});

test("off by default: every call is a model call", async () => {
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  const f = mood(router);
  await f("Love it.");
  await f("Love it.");
  assert.equal(router.requests.length, 2);
});

test("on: an identical request gets the reply it got before, logged as cached; another request is a model call", async () => {
  clearCache();
  const folder = mkdtempSync(join(tmpdir(), "functai-cache-"));
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  const f = mood(router, { cacheReplies: true, logCalls: folder });
  assert.equal(await f("Love it."), "happy");
  assert.equal(await f("Love it."), "happy");
  assert.equal(router.requests.length, 1);
  await f("Love it.", { temperature: 0.7 });                    // another setting sent: another request
  await f("Hate it.");
  assert.equal(router.requests.length, 3);
  const [first, second] = calls(f, { folder }).map((c) => (c.record["exchanges"] as Record<string, unknown>[])[0]!);
  assert.equal(first!["cached"], false);
  assert.equal(second!["cached"], true);
  assert.equal(second!["seconds"], 0);
  clearCache();
  await f("Love it.");
  assert.equal(router.requests.length, 4);
});

test("an unreadable reply is not kept: the next identical call asks the model again", async () => {
  clearCache();
  let n = 0;
  const router = new FakeRouter([], () => (++n === 1 ? "no tags at all" : "<result>\nunhappy\n</result>"));
  const f = mood(router, { cacheReplies: true, retries: 0 });
  await assert.rejects(f("Broke."));
  assert.equal(await f("Broke."), "unhappy");                   // not stuck on the bad reply
  assert.equal(await f("Broke."), "unhappy");
  assert.equal(router.requests.length, 2);
});

test("any store with get, set and delete: a Map, or one that keeps text; a store that fails is skipped", async () => {
  const store = new Map<string, unknown>();
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  const f = mood(router, { cacheReplies: store });
  await f("Fine.");
  await f("Fine.");
  assert.equal(router.requests.length, 1);
  const [key, value] = [...store.entries()][0]!;
  assert.match(key, /^sha256:[0-9a-f]{64}$/);                 // the contract's key (replies.md): every language finds it
  assert.deepEqual(JSON.parse(JSON.stringify(value)), value);  // plain JSON, for a store in another process

  const text = new Map<string, string>();
  const asText = { get: async (k: string) => text.get(k), set: async (k: string, v: unknown) => { text.set(k, JSON.stringify(v)); },
                   delete: async (k: string) => { text.delete(k); } };
  const g = mood(router, { cacheReplies: asText });
  await g("Good.");
  await g("Good.");
  assert.equal(router.requests.length, 2);

  const broken = { get: () => { throw new Error("down"); }, set: () => { throw new Error("down"); }, delete: () => undefined };
  const h = mood(router, { cacheReplies: broken });
  const warn = console.warn;
  const warned: string[] = [];
  console.warn = (m: string) => { warned.push(m); };
  try {
    assert.equal(await h("Nice."), "happy");                    // the call goes through
  } finally {
    console.warn = warn;
  }
  assert.match(warned.join("\n"), /reply cache could not read/);
});

test("a stream from the cache shows the whole answer, and ends with the same value", async () => {
  clearCache();
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  const f = mood(router, { cacheReplies: true });
  await f("Great.");
  const s = f.stream("Great.");
  let text = "";
  for await (const piece of s) text += piece;
  assert.equal(await s.result, "happy");
  assert.equal(text.trim(), "happy");
  assert.equal(router.requests.length, 1);
});
