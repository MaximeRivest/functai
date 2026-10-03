/** Tools that ask first (contract/tools.md) and plugins (contract/plugins.md), run against a fake model. */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ai, ApprovalError, calls, compaction, configure, delegate, earlier, FolderStore, module, Plugin, PluginError, remember, t, tool, Waiting,
  withSettings } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";
import { clock } from "../src/conversations.ts";
import { ConversationError } from "../src/errors.ts";

type Rec = Record<string, any>;
const folder = () => mkdtempSync(join(tmpdir(), "functai-tools-"));
const lastText = (r: { messages: readonly unknown[] }) => JSON.stringify(r.messages.at(-1));

function refunds() {
  const ran: Rec[] = [];
  const status = tool("order_status", { description: "Where an order is.", input: { order: t.string() }, effects: "reads" }, ({ order }) => `${order}: in Leeds`);
  const refund = tool("refund", { description: "Refund an order.", input: { order: t.string() }, effects: "changes" }, ({ order }) => {
    ran.push({ order });
    return `refunded ${order}`;
  });
  return { status, refund, ran };
}

/** A model that looks the order up, asks for a refund, then answers. */
function script() {
  return new FakeRouter([
    { calls: [{ id: "c1", name: "order_status", input: { order: "B-2210" } }] },
    { calls: [{ id: "c2", name: "refund", input: { order: "B-2210" } }] },
    "<result>\nDone: refunded.\n</result>",
  ]);
}

test("a plain call with an approval rule has nobody to ask: it refuses before the tool runs", async () => {
  const { status, refund, ran } = refunds();
  const f = ai("support", { input: { message: t.string() }, output: t.string(), tools: [status, refund], router: script() as never, lm: "gpt-4.1-mini", approve: "changes" });
  await assert.rejects(f("Refund B-2210"), (e: unknown) => e instanceof ApprovalError && e.code === "approval-required" && e.approval.path === "support/refund");
  assert.equal(ran.length, 0);
});

test("approve as a function: asked at once; a refusal is an answer the model sees, with the reason", async () => {
  const { status, refund, ran } = refunds();
  const router = script();
  const asked: Rec[] = [];
  const f = ai("support", { input: { message: t.string() }, output: t.string(), tools: [status, refund], router: router as never, lm: "gpt-4.1-mini",
    approve: (a) => { asked.push(a); return "not without a receipt"; } });
  assert.equal(await f("Refund B-2210"), "Done: refunded.");
  assert.deepEqual(asked.map((a) => [a.name, a.invocation, a.path, a.effects]), [["refund", 2, "support/refund", "changes"]]);   // reads: never asked
  assert.equal(ran.length, 0);
  assert.ok(lastText(router.requests[2]!).includes("The person did not allow this call. Reason: not without a receipt"), lastText(router.requests[2]!));
});

test("on a stream, the call waits for s.approve()", async () => {
  const { status, refund, ran } = refunds();
  const f = ai("support", { input: { message: t.string() }, output: t.string(), tools: [status, refund], router: script() as never, lm: "gpt-4.1-mini", approve: "changes" });
  const s = f.stream("Refund B-2210");
  const kinds: string[] = [];
  const watching = (async () => {
    for await (const e of s.events()) {
      kinds.push(e.kind);
      if (e.kind === "approval") s.approve();
    }
  })();
  assert.equal(await s, "Done: refunded.");
  await watching;
  assert.deepEqual(ran, [{ order: "B-2210" }]);
  assert.ok(kinds.indexOf("approval") < kinds.indexOf("approved"), kinds.join(","));
  assert.ok(kinds.indexOf("approved") < kinds.lastIndexOf("tool_result"));
});

test("in a conversation, the turn waits, saved; another process approves it and it resumes, paying for no answer twice and running no tool twice", async () => {
  const where = folder();
  const logs = folder();
  const { status, refund, ran } = refunds();
  const router = script();
  const f = ai("support", { input: { message: t.string() }, output: t.string(), tools: [status, refund], router: router as never, lm: "gpt-4.1-mini", logCalls: logs });
  const chat = f.conversation("ticket-1", { store: new FolderStore(where), approve: "changes" });
  const waited = await chat("Refund B-2210").then(() => null, (e: unknown) => e);
  assert.ok(waited instanceof Waiting, String(waited));
  const turn = (waited as Waiting).turn as { state: string; waiting: { name: string; invocation: number }[] };
  assert.equal(turn.state, "waiting");
  assert.deepEqual(turn.waiting.map((a) => [a.name, a.invocation]), [["refund", 2]]);
  assert.equal(router.requests.length, 2);
  assert.equal(ran.length, 0);
  assert.equal(calls(f, { folder: logs }).length, 0);                       // a waiting turn's call has not ended: no record yet
  // another process: a fresh store on the same folder
  const there = f.conversation("ticket-1", { store: new FolderStore(where), approve: "changes" });
  const pending = (await there.turns()).at(-1)!;
  assert.equal(await pending.approve(), "Done: refunded.");
  assert.equal(router.requests.length, 3);                                  // the two answers it had were replayed
  assert.deepEqual(ran, [{ order: "B-2210" }]);
  const final = (await there.turns()).at(-1)!;
  assert.equal(final.state, "done");
  const records = calls(f, { folder: logs });
  assert.equal(records.length, 1);
  assert.equal(records[0]!.record["writer"], 2);                            // continued by a later writer
  const cached = (records[0]!.record["exchanges"] as Rec[]).map((e) => e["cached"]);
  assert.deepEqual(cached, [true, true, false]);
});

test("a tool that changes things is recorded before it runs: after its process stops it may have run, and the turn goes on only when a person says what it did", async () => {
  const where = folder();
  const router = new FakeRouter([{ calls: [{ id: "c1", name: "send_email", input: { to: "ana" } }] }, "<result>\nsent\n</result>"]);
  let runs = 0;
  const send = tool("send_email", { input: { to: t.string() }, effects: "changes" }, () => {
    runs++;
    return new Promise<string>(() => undefined);                             // the process stops while it runs
  });
  const f = ai("mailer", { input: { request: t.string() }, output: t.string(), tools: [send], router: router as never, lm: "gpt-4.1-mini" });
  const first = f.conversation("mail", { store: new FolderStore(where) }).stream("Email Ana");
  first.result.catch(() => undefined);
  const turn = await first.turn;
  while (!new FolderStore(where).read("mail").some((r) => r["kind"] === "tool")) await new Promise((r) => setTimeout(r, 5));
  const before = clock.now;
  clock.now = () => Date.now() / 1000 + 120;                                 // its lease ran out: interrupted
  try {
    const there = f.conversation("mail", { store: new FolderStore(where) });
    const t2 = await there.turn(turn.id);
    assert.equal(t2.state, "interrupted");
    assert.deepEqual(t2.unfinished.map((x) => [x["name"], x["invocation"]]), [["send_email", 1]]);
    await assert.rejects(t2.resume(), (e: unknown) => e instanceof ConversationError && e.code === "turn-unfinished");
    assert.equal(await t2.resume({ results: { 1: "sent to ana" } }), "sent");
    assert.equal(runs, 1);                                                   // never run again on its own
    assert.equal(router.requests.length, 2);                                 // the first answer was replayed
    assert.ok(lastText(router.requests[1]!).includes("sent to ana"));
  } finally {
    clock.now = before;
  }
  await assert.rejects(first.result);                                        // the first process was taken over: it stops
});

test("plugins: before_call sections reach the instruction and are recorded; a tool_call block stops the tool; a request replaced is not replayable", async () => {
  const logs = folder();
  const { status, refund, ran } = refunds();
  const router = script();
  const careful = new Plugin("careful", { version: "1.2.0" }).beforeCall(() => ({ sections: ["Answer carefully."] }));
  const guard = new Plugin("guard").toolCall((x) => (x.name === "refund" ? { block: "refunds are closed today" } : undefined));
  const f = ai("support", { input: { message: t.string() }, output: t.string(), tools: [status, refund], router: router as never, lm: "gpt-4.1-mini",
    logCalls: logs, plugins: [careful] });
  await withSettings({ plugins: [guard] }, () => f("Refund B-2210"));
  assert.ok(/Function: support\n\nAnswer carefully\./.test(String(router.requests[0]!.system)), String(router.requests[0]!.system));
  assert.equal(ran.length, 0);
  assert.ok(lastText(router.requests[2]!).includes("This call was blocked (guard): refunds are closed today"));
  const rec = calls(f, { folder: logs })[0]!.record;
  assert.deepEqual((rec["changes"] as Rec[]).map((c) => [c["plugin"], c["hook"]]), [["careful", "before_call"], ["guard", "tool_call"]]);
  assert.equal((rec["changes"] as Rec[])[0]!["version"], "1.2.0");

  const rewrite = new Plugin("rewrite").request((e) => ({ ...e.request, messages: [...e.request.messages] }));
  const g = ai("plain", { input: { q: t.string() }, output: t.string(), router: new FakeRouter([], () => "<result>\nx\n</result>") as never, lm: "gpt-4.1-mini", logCalls: logs, plugins: [rewrite] });
  await g("hi");
  const r2 = calls(g, { folder: logs })[0]!.record;
  assert.equal(r2["replayable"], false);
  assert.ok(!("request_hash" in (r2["exchanges"] as Rec[])[0]!));
  assert.throws(() => new Plugin("Bad Name"), (e: unknown) => e instanceof PluginError && e.code === "plugin-name");
  assert.throws(() => new Plugin("x").on("beforecall" as never, () => undefined), (e: unknown) => e instanceof PluginError && e.code === "plugin-hook");
  const wrong = new Plugin("wrong").beforeCall(() => ({ block: "no" }) as never);
  await assert.rejects(g.using({ plugins: [wrong] })("hi"), (e: unknown) => e instanceof PluginError && e.code === "plugin-change");
});

test("compaction folds older turns into a summary that the next turns are shown, with only the turns after it", async () => {
  const router = new FakeRouter([], (_r, i) => `<result>\nr${i + 1}\n</result>`);
  const summaries: Rec[] = [];
  const f = ai("tutor", { input: { message: t.string() }, output: t.string(), router: router as never, lm: "gpt-4.1-mini" });
  const chat = f.conversation("long", { plugins: [compaction({ keep: 1, every: 2, summarize: (earlier, turns) => { summaries.push({ earlier, n: turns.length }); return `S${summaries.length}`; } })] });
  for (const m of ["m1", "m2", "m3", "m4"]) await chat(m);
  assert.deepEqual(summaries, [{ earlier: "", n: 2 }]);
  const system = String(router.requests[3]!.system);
  assert.ok(system.includes("Earlier in this conversation (2 turns, summarized):\nS1"), system);
  assert.equal(router.requests[3]!.messages.length, 3);                     // only m3 shown whole, then m4
  const last = (await chat.turns()).at(-1)!;
  assert.ok(last.state === "done");
});

test("delegation: another program as a tool, in a conversation of its own that follows the branch that asked", async () => {
  const inner = new FakeRouter([], (r) => `<result>\nresearched (${r.messages.length} messages)\n</result>`);
  const research = ai("research", { description: "Look things up.", input: { topic: t.string() }, output: t.string(), router: inner as never, lm: "gpt-4.1-mini" });
  const outer = new FakeRouter([
    { calls: [{ id: "d1", name: "research", input: { topic: "otters" } }] }, "<result>\nfirst\n</result>",
    { calls: [{ id: "d2", name: "research", input: { topic: "more otters" } }] }, "<result>\nsecond\n</result>",
  ]);
  const assistant = ai("assistant", { input: { request: t.string() }, output: t.string(), tools: [delegate(research)], router: outer as never, lm: "gpt-4.1-mini" });
  const chat = assistant.conversation("work");
  assert.equal(await chat("Tell me about otters"), "first");
  assert.equal(await chat("More"), "second");
  assert.deepEqual(inner.requests.map((r) => r.messages.length), [1, 3]);   // the delegate remembers what it was asked on this branch
  assert.ok(lastText(outer.requests[1]!).includes("researched (1 messages)"));
});

test("a module's conversation: its code reads earlier(); a helper remembers only when the conversation says so", async () => {
  const router = new FakeRouter([], (r, i) => (String(r.system).includes("Classify") ? "<result>\ntopic\n</result>" : `<result>\nanswer ${i}\n</result>`));
  const topic = ai("topic", { description: "Classify the message.", input: { message: t.string() }, output: t.string(), router: router as never, lm: "gpt-4.1-mini" });
  const answer = ai("answer", { description: "Answer the message.", input: { message: t.string(), topic: t.string() }, output: t.string(), router: router as never, lm: "gpt-4.1-mini" });
  const seen: number[] = [];
  const support = module("support", { input: { message: t.string() }, output: t.string(), uses: [topic, answer] }, async ({ message }) => {
    seen.push(earlier().length);
    return answer({ message, topic: await topic(message) });
  });
  const chat = support.conversation("s", { remembers: [[answer, remember("conversation")]] });
  await chat("one");
  await chat("two");
  assert.deepEqual(seen, [0, 1]);
  const answers = router.requests.filter((r) => String(r.system).includes("Answer"));
  const topics = router.requests.filter((r) => String(r.system).includes("Classify"));
  assert.deepEqual(answers.map((r) => r.messages.length), [1, 3]);          // answer remembers its own earlier call
  assert.deepEqual(topics.map((r) => r.messages.length), [1, 1]);           // topic remembers nothing
  assert.throws(() => support.conversation("bad", { remembers: [[ai("other", { input: { x: t.string() } }), "turn"]] }), /does not call/);
});

void configure;
