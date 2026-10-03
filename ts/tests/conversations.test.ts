/** Conversations (contract/conversations.md), run against a fake model: turns, branches, queueing, stores, stopping. */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ai, calls, ConversationError, FolderStore, lastTurns, t } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

type Rec = Record<string, any>;
const folder = () => mkdtempSync(join(tmpdir(), "functai-conv-"));
const texts = (r: { messages: readonly unknown[] }) => (r.messages as { parts: { text?: string }[] }[]).map((m) => m.parts.map((p) => p.text ?? "").join(""));

function tutor(router: FakeRouter, extra: Rec = {}) {
  return ai("tutor", { description: "A patient tutor.", input: { message: t.string() }, output: t.string(), router: router as never, lm: "gpt-4.1-mini", ...extra });
}

test("a conversation remembers: each turn is shown the turns before it, and records what it saw", async () => {
  const router = new FakeRouter([], (_r, i) => `<result>\nanswer ${i + 1}\n</result>`);
  const logs = folder();
  const f = tutor(router, { logCalls: logs });
  const chat = f.conversation("alex");
  assert.equal(await chat("Hi, I'm Alex."), "answer 1");
  assert.equal(await chat("What is my name?"), "answer 2");
  assert.equal(await chat("And again?"), "answer 3");
  assert.equal(router.requests[0]!.messages.length, 1);
  assert.equal(router.requests[1]!.messages.length, 3);                 // the first turn, then the question
  assert.ok(texts(router.requests[1]!).join("|").includes("Hi, I'm Alex."));
  assert.equal(router.requests[2]!.messages.length, 5);
  const turns = await chat.turns();
  assert.deepEqual(turns.map((x) => x.result), ["answer 1", "answer 2", "answer 3"]);
  assert.deepEqual(turns.map((x) => x.parent), [null, turns[0]!.id, turns[1]!.id]);
  assert.deepEqual((await turns[2]!.saw()).map((x) => x.id), [turns[0]!.id, turns[1]!.id]);
  // the call log: each turn's call is the turn, with its conversation, its saw (saw_of keeps it short) and its steps
  const logged = calls(f, { folder: logs });
  assert.deepEqual(logged.map((c) => c.id), turns.map((x) => x.id));
  const third = logged[2]!.record;
  assert.deepEqual(third["conversation"], { id: "alex", turn: turns[2]!.id, parent: turns[1]!.id });
  assert.deepEqual(third["saw"], [{ saw_of: turns[1]!.id }, { call: turns[1]!.id, steps: true }]);
  assert.ok(Array.isArray(third["steps"]) && third["steps"].length === 1);
  // the program is unchanged: called on its own, it remembers nothing
  await f("Who am I?");
  assert.equal(router.requests[3]!.messages.length, 1);
});

test("a branch: continuing from an earlier turn shows that turn's path only; nothing is deleted", async () => {
  const router = new FakeRouter([], (_r, i) => `<result>\na${i + 1}\n</result>`);
  const chat = tutor(router).conversation("branchy");
  await chat("one");
  await chat("two");
  const [first] = await chat.turns();
  const other = chat.continueFrom(first!);
  await other("three");
  assert.equal(router.requests[2]!.messages.length, 3);                 // shown "one" only
  assert.ok(!texts(router.requests[2]!).join("|").includes("two"));
  assert.equal((await other.turns()).length, 2);
  assert.equal((await chat.allTurns()).length, 3);
  assert.deepEqual((await chat.turns()).map((x) => x.inputs["message"]), ["one", "two"]);   // the view opened by id: its own branch
});

test("the last turns only, and a field left out of earlier turns", async () => {
  const router = new FakeRouter([], (_r, i) => `<result>\na${i + 1}\n</result>`);
  const f = ai("reader", { input: { document: t.string(), question: t.string() }, output: t.string(), router: router as never, lm: "gpt-4.1-mini" });
  const chat = f.conversation("docs", { context: lastTurns(1, { without: ["document"] }) });
  await chat({ document: "DOC-ONE", question: "q1" });
  await chat({ document: "DOC-TWO", question: "q2" });
  await chat({ document: "DOC-THREE", question: "q3" });
  const sent = texts(router.requests[2]!).join("|");
  assert.ok(sent.includes("q2") && !sent.includes("q1"), sent);
  assert.ok(!sent.includes("DOC-TWO") && sent.includes("DOC-THREE"), sent);
});

test("two sends at once queue: the second continues from the first", async () => {
  const router = new FakeRouter([], (_r, i) => `<result>\nr${i + 1}\n</result>`);
  const chat = tutor(router).conversation("queue");
  const [a, b] = await Promise.all([chat("first"), chat("second")]);
  assert.deepEqual([a, b], ["r1", "r2"]);
  const turns = await chat.turns();
  assert.equal(turns.length, 2);
  assert.equal(turns[1]!.parent, turns[0]!.id);
});

test("the same request id twice is one turn", async () => {
  const router = new FakeRouter([], () => "<result>\nonce\n</result>");
  const chat = tutor(router).conversation("dupes");
  const s1 = chat.stream("hi", { requestId: "click-1" });
  const s2 = chat.stream("hi", { requestId: "click-1" });
  assert.equal(await s1, "once");
  assert.equal(await s2, "once");
  assert.equal(router.requests.length, 1);
  assert.equal((await s1.turn).id, (await s2.turn).id);
});

test("a folder store keeps a conversation another process (another store object) continues; files are the contract's", async () => {
  const where = folder();
  const router = new FakeRouter([], (_r, i) => `<result>\nr${i + 1}\n</result>`);
  const f = tutor(router);
  await f.conversation("kept", { store: new FolderStore(where) })("hello");
  const again = f.conversation("kept", { store: new FolderStore(where) });
  await again("still there?");
  assert.equal(router.requests[1]!.messages.length, 3);
  assert.deepEqual(readdirSync(join(where, "conversations")).sort(), ["kept.jsonl", "kept.lock"]);
  const records = readFileSync(join(where, "conversations", "kept.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.deepEqual(records.map((r: Rec) => r.kind), ["program", "turn", "lease", "ended", "turn", "lease", "ended"]);
  assert.deepEqual(records.map((r: Rec) => r.seq), [1, 2, 3, 4, 5, 6, 7]);
  // each turn's call tree, kept while it ran: another process can watch it
  const trees = readdirSync(join(where, "trees")).filter((x) => x.endsWith(".jsonl"));
  assert.equal(trees.length, 2);
  const turn = (await again.turns())[1]!;
  const kinds: string[] = [];
  for await (const e of turn.events()) kinds.push(e["kind"] as string);
  assert.deepEqual(kinds.slice(0, 2), ["started", "request"]);
  assert.equal(kinds.at(-1), "done");
});

test("sends: refuse says the conversation is busy; an id that is not a file name is refused", async () => {
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const slow = { resolve: (m: string) => router.resolve(m), complete: async (r: never) => { await gate; return router.complete(r); } };
  const chat = tutor(router, { router: slow }).conversation("busy", { sends: "refuse" });
  const first = chat("one");
  await new Promise((r) => setTimeout(r, 20));
  await assert.rejects(chat("two"), (e: unknown) => e instanceof ConversationError && e.code === "conversation-busy");
  release();
  assert.equal(await first, "ok");
  assert.throws(() => tutor(router).conversation("../etc/passwd"), (e: unknown) => e instanceof ConversationError && e.code === "conversation-id");
});

test("a turn is stopped from anywhere: it ends stopped", async () => {
  const router = new FakeRouter([], () => "<result>\nnever\n</result>");
  const hang = { resolve: (m: string) => router.resolve(m), complete: (_r: never, opts?: { signal?: AbortSignal }) => new Promise((_, reject) => opts?.signal?.addEventListener("abort", () => reject(new Error("aborted")))) };
  const chat = tutor(router, { router: hang }).conversation("stoppable");
  const s = chat.stream("think forever");
  const turn = await s.turn;
  await new Promise((r) => setTimeout(r, 30));
  await chat.stop(turn);
  await assert.rejects(s.result);
  assert.equal((await turn.refresh()).state, "stopped");
});
