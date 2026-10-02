/**
 * Stage 1 through real calls (a fake model, no network): how a call makes its
 * events (the laws of streaming.md that no shared case can check yet), their
 * forms and views, observers, journals with a real tool loop, the policy
 * refusals, what the call log keeps, and modules.
 */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as z from "zod";
import {
  ai, Cancelled, configure, describeSaved, flush, EventUnknown, Follower, fromManifest, InterfaceError, JournalError, LoadRefused, MemoryStore,
  module, SettingError, stateOf, t, toManifest, tool, views, withSettings, type AppendAnswer, type Position, type StreamEvent,
} from "../src/index.ts";
import { passes } from "../src/schema.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

type Rec = Record<string, any>;
const all = async (it: AsyncIterable<StreamEvent>) => {
  const out: StreamEvent[] = [];
  for await (const e of it) out.push(e);
  return out;
};
function logged(folder: string): Rec[] {
  const out: Rec[] = [];
  for (const day of readdirSync(folder)) {
    for (const f of readdirSync(join(folder, day))) {
      for (const line of readFileSync(join(folder, day, f), "utf8").split("\n")) if (line) out.push(JSON.parse(line));
    }
  }
  return out;
}
const at = (e: StreamEvent): Position => ({ writer: e.writer, seq: e.seq });

const mood = (router: unknown, extra: Rec = {}) => ai("mood", {
  description: "How does the customer feel?", input: { review: t.string() }, output: t.enum("happy", "unhappy", "mixed"),
  router: router as never, ...extra,
});

test("a stream's events are format 2: numbered, chained, schema-valid; a request per exchange; the text is its latest request's", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-s1-"));
  const s = mood(new FakeRouter(["not tags", "<result>\nhappy\n</result>"]), { logCalls: folder }).stream("Lovely");
  const events = await all(s.events());
  assert.equal(await s, "happy");
  for (const e of events) assert.ok(passes("event", e), JSON.stringify(e));
  assert.deepEqual(events.map((e) => e.seq), events.map((_, i) => i + 1));                        // dense, one writer
  assert.deepEqual(events.map((e) => e.after), [null, ...events.slice(0, -1).map(at)]);          // each after its predecessor
  assert.ok(events.every((e, i) => i === 0 || e.at >= events[i - 1]!.at));
  assert.deepEqual(events.map((e) => e.kind).filter((k) => k !== "text"), ["started", "request", "retry", "request", "done"]);
  assert.deepEqual(stateOf(events).calls[s.callId!]!.fields, { result: "happy" });                // the retry voided "not tags"
  assert.equal(s.text, "happy");
  const [record] = logged(folder);
  assert.ok(passes("call", record), JSON.stringify(record).slice(0, 300));
  assert.equal(record!.exchanges.length, events.filter((e) => e.kind === "request").length);     // law 8
  assert.equal(events[0]!.tree, record!.id);
});

test("a stream opened inside a tree shows the tree's numbers, its own chain starting at null (law 7)", async () => {
  const inner: StreamEvent[] = [];
  const m = mood(new FakeRouter([], () => "<result>\nmixed\n</result>"));
  const both = module("both", { input: { a: t.string(), b: t.string() }, output: t.list(t.string()), uses: [m] }, async ({ a, b }) => {
    const first = await m(a);
    const s = m.stream(b);
    inner.push(...await all(s.events()));
    return [first, await s];
  });
  const outer = await all(both.stream({ a: "x", b: "y" }).events());
  const byPos = new Map(outer.map((e) => [`${e.writer}:${e.seq}`, e]));
  assert.equal(inner[0]!.after, null);
  assert.ok(inner[0]!.seq > 1);
  for (const e of inner) {
    const same = byPos.get(`${e.writer}:${e.seq}`)!;
    assert.equal(same.kind, e.kind);
    assert.equal(same.tree, e.tree);
  }
  assert.deepEqual(inner.slice(1).map((e) => e.after), inner.slice(0, -1).map(at));
  assert.equal(outer.at(-1)!.kind, "done");
  assert.deepEqual((outer.at(-1) as Rec).value, ["mixed", "mixed"]);
});

test("forms: the kept form follows logContent; a view shows a boundary; a stream resumes after an event of its form", async () => {
  const s = mood(new FakeRouter(["<result>\nhappy\n</result>"]), { logContent: { review: false } }).stream("my password is hunter2");
  const whole = await all(s.events());
  const kept = await all(s.events({ form: "kept" }));
  assert.equal((kept[0] as Rec).content, false);
  assert.deepEqual((kept[0] as Rec).omitted, { inputs: ["review"], outputs: [] });
  assert.ok(!JSON.stringify(kept).includes("hunter2"));
  assert.deepEqual(kept.map((e) => e.kind), whole.map((e) => e.kind));                           // the answer is kept: its text too
  const boundary = await all(s.events({ view: views.boundary(s.callId!) }));
  assert.deepEqual(boundary.map((e) => e.kind), ["started", "request", "done"]);                  // a view keeps requests
  assert.deepEqual(boundary.map((e) => e.after), [null, at(boundary[0]!), at(boundary[1]!)]);
  const after = at(whole[1]!);
  assert.deepEqual((await all(s.events({ after }))).map(at), whole.slice(2).map(at));
  assert.deepEqual(s.read(null, after), { events: whole.slice(2) });
  assert.deepEqual(s.read(null, { writer: 1, seq: 99 }), { refuses: "event-unknown" });
  await assert.rejects(all(s.events({ after: { writer: 2, seq: 1 } })), EventUnknown);
});

test("observers get the kept form of the calls in scope, added up over layers; one that fails is dropped with one warning", async () => {
  const seen: Rec = { host: [], block: [], own: [] };
  const warn = console.warn;
  const warned: string[] = [];
  console.warn = (m: string) => { warned.push(m); };
  const broken = () => { throw new Error("socket closed"); };
  configure({ observers: [(e) => { seen.host.push(e); }, broken] });
  try {
    const f = mood(new FakeRouter([], () => "<result>\nhappy\n</result>"), { observers: [(e: StreamEvent) => { seen.own.push(e); }], logContent: { review: false } });
    await withSettings({ observers: [(e) => { seen.block.push(e); }] }, () => f("secret words"));
    await f("again");
    assert.ok(await flush());                                   // observers are given events off the call's turn
  } finally {
    configure({ observers: undefined });
    console.warn = warn;
  }
  assert.equal(seen.host.length, seen.own.length);
  assert.equal(seen.block.length, seen.host.length / 2);                                          // the block's scope: the first call
  assert.ok(!JSON.stringify(seen.host).includes("secret"));
  assert.equal(warned.filter((w) => w.includes("observer broken failed")).length, 1);  // named, once
});

/** A tree with a tool: helper asks lookup_order, then answers. */
function helper(router: FakeRouter, ran: string[], extra: Rec = {}) {
  const lookup = tool("lookup_order", { description: "Look an order up.", input: { order: t.string() } }, ({ order }) => {
    ran.push(order);
    return "In Leeds, 7 days late.";
  });
  return ai("helper", { description: "Help.", input: { question: t.string() }, tools: [lookup], router, ...extra });
}
const toolReplies = () => new FakeRouter([
  { text: "", calls: [{ id: "call_1", name: "lookup_order", input: { order: "B-2210" } }] },
  "<result>\nIt is in Leeds, a week late.\n</result>",
]);

test("a required journal keeps the whole tree's kept log before each barrier; a follower from the store ends where the stream ends", async () => {
  const store = new MemoryStore();
  const ran: string[] = [];
  const f = helper(toolReplies(), ran, { journal: { store, mode: "required" } });
  const s = f.stream("Where is B-2210?");
  const kept = await all(s.events({ form: "kept" }));
  assert.equal(await s, "It is in Leeds, a week late.");
  assert.deepEqual(ran, ["B-2210"]);
  const tree = s.tree!;
  assert.deepEqual(store.events(tree), kept);
  assert.ok(store.finished(tree));
  const reader = new Follower({ form: "kept" });
  const reads = await reader.recover(tree, store);
  assert.deepEqual(reads.map((r) => r.after), [null]);
  assert.deepEqual(reader.state(tree), stateOf(kept));
});

/** A store that fails every append from the nth on. */
class FailingStore extends MemoryStore {
  sends = 0;
  from: number;
  answer: "down" | "conflict";
  constructor(from: number, answer: "down" | "conflict" = "down") {
    super();
    this.from = from;
    this.answer = answer;
  }
  override async append(events: readonly Rec[]): Promise<AppendAnswer> {
    if (++this.sends >= this.from) {
      if (this.answer === "down") throw new Error("unreachable");
      return { refuses: "event-conflict", event: { writer: events[0]!.writer, seq: events[0]!.seq } };
    }
    return this.appendNow(events);
  }
}

test("a required journal that does not keep the tool call: the tool does not run, JournalError journal-barrier, recorded", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-s1-"));
  const warn = console.warn;
  console.warn = () => undefined;
  try {
    const ran: string[] = [];
    // started, request, tool_call: the third send (the tool call's) and every one after it are refused
    const store = new FailingStore(3, "conflict");
    const f = helper(toolReplies(), ran, { journal: { store, mode: "required", batch: 1 }, logCalls: folder });
    const s = f.stream("Where is B-2210?");
    const events = await all(s.events());
    const err = await s.result.then(() => null, (e: unknown) => e);
    assert.deepEqual(ran, []);
    assert.ok(err instanceof JournalError);
    assert.equal(err.code, "journal-end");                     // the end cannot be kept either: refused
    assert.equal(err.journal, "refused");
    assert.ok("failed" in err.outcome! && err.outcome.failed instanceof JournalError && err.outcome.failed.code === "journal-barrier");
    assert.ok(!events.some((e) => e.kind === "failed"));       // the stream shows no end the journal does not hold
    const [record] = logged(folder);
    assert.deepEqual([record!.error.type, record!.error.code, record!.journal], ["JournalError", "journal-barrier", "refused"]);
  } finally {
    console.warn = warn;
  }
});

test("a required journal that does not answer the end: the call's value is in JournalError journal-end, settled by position", async () => {
  const warn = console.warn;
  console.warn = () => undefined;
  try {
    const store = new FailingStore(4);                         // started, request, text: the end's sends fail
    const f = mood(new FakeRouter(["<result>\nhappy\n</result>"], null, "openai", 100), { journal: { store, mode: "required", batch: 1 } });
    const events: StreamEvent[] = [];
    const s = f.stream("Lovely");
    const reading = (async () => { for await (const e of s.events()) events.push(e); })();
    const err = await s.result.then(() => null, (e: unknown) => e);
    await reading;
    assert.ok(err instanceof JournalError && err.code === "journal-end");
    assert.deepEqual(err.outcome, { done: "happy" });
    assert.equal(err.journal, "unknown");
    assert.equal(await err.settle(), "not-kept");
    assert.equal(err.event!.seq, events.length + 1);          // the end nobody was shown
  } finally {
    console.warn = warn;
  }
});

test("a best-effort journal never holds a call up; a program cannot replace a host's journal (journal-policy), nor set a required one inside a tree (journal-scope)", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-s1-"));
  const warn = console.warn;
  console.warn = () => undefined;
  const host = new MemoryStore();
  try {
    const down = new FailingStore(1);
    assert.equal(await mood(new FakeRouter(["<result>\nhappy\n</result>"]), { journal: down })("x"), "happy");
    configure({ journal: host });
    const mine = mood(new FakeRouter(["<result>\nhappy\n</result>"]), { journal: new MemoryStore(), logCalls: folder });
    await assert.rejects(mine("x"), (e: unknown) => e instanceof JournalError && e.code === "journal-policy");
    const [refused] = logged(folder);
    assert.deepEqual([refused!.error.type, refused!.error.code], ["JournalError", "journal-policy"]);
    assert.deepEqual(host.events(refused!.id).map((e) => e["kind"]), ["started", "failed"]);   // the host's journal has the refused tree
    configure({ journal: undefined });
    const inner = mood(new FakeRouter(["<result>\nhappy\n</result>"]), { journal: { store: new MemoryStore(), mode: "required" } });
    const outer = module("outer", { input: {}, output: t.string() }, () => inner("x"));
    await assert.rejects(outer({}), (e: unknown) => e instanceof JournalError && e.code === "journal-scope");
  } finally {
    configure({ journal: undefined });
    console.warn = warn;
  }
});

test("logContent only removes, over every layer; a program's own misspelt name, or a key that is not a name, refuses", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-s1-"));
  const f = ai("triage", {
    description: "Triage.", input: { transcript: t.string(), question: t.string() }, outputs: { summary: t.string(), result: t.string() },
    router: new FakeRouter([], () => "<reasoning>\nAna says so\n</reasoning>\n<summary>\nshort\n</summary>\n<result>\nno\n</result>") as never, logCalls: folder,
    logContent: { question: true }, module: "cot",
  });
  await withSettings({ logContent: { "*": false, question: true, result: true } }, () => f({ transcript: "Ana: red again", question: "Broken?" }));
  const [r] = logged(folder);
  assert.equal(r!.content, false);
  assert.deepEqual(r!.omitted, { inputs: ["transcript"], outputs: ["reasoning", "summary"] });
  assert.deepEqual(r!.inputs, { question: "Broken?" });
  assert.deepEqual(r!.outputs, { result: "no" });
  assert.ok(r!.exchanges.every((ex: Rec) => !("request" in ex) && !("request_hash" in ex)));
  assert.ok(passes("call", r));
  assert.throws(() => ai("f", { input: { transcript: t.string() }, logContent: { transcrpit: false } }),
    (e: unknown) => e instanceof SettingError && e.code === "log-content-field" && e.field === "transcrpit");
  assert.throws(() => withSettings({ logContent: { "#private": false } }, () => 0), (e: unknown) => e instanceof SettingError);
  process.env["FUNCTAI_LOG_CONTENT"] = " Off ";
  try {
    await f({ transcript: "x", question: "y" });
  } finally {
    delete process.env["FUNCTAI_LOG_CONTENT"];
  }
  assert.deepEqual(logged(folder)[1]!.omitted, { inputs: ["transcript", "question"], outputs: ["reasoning", "summary", "result"] });
});

test("a module checks its inputs and outputs on every call, records a refused call, and names what it declares", async () => {
  const folder = mkdtempSync(join(tmpdir(), "functai-s1-"));
  const answer = mood(new FakeRouter([], () => "<result>\nhappy\n</result>"));
  const support = module("support", {
    description: "Answer a customer.",
    input: { message: t.string(), tone: t.withDefault(t.string(), "kind"), order: z.string().optional() },
    outputs: { feeling: t.enum("happy", "unhappy", "mixed"), result: t.string() },
    uses: [answer], logCalls: folder,
  }, async ({ message, tone, order }, { signal }) => {
    const feeling = await answer(message, { signal });
    return { feeling, result: `${tone}: ${order ?? "no order"}` };
  });
  assert.deepEqual(await support("Where is it?"), { feeling: "happy", result: "kind: no order" });
  assert.deepEqual(support.interface.inputs.map((f) => [f.name, f.optional ?? false, f.shape["default"]]),
    [["message", false, undefined], ["tone", true, "kind"], ["order", true, undefined]]);
  await assert.rejects(support({ message: null } as never), (e: unknown) => e instanceof InterfaceError && e.code === "interface-input" && e.field === "message");
  const records = logged(folder);
  const mod = records.filter((r) => r.program.kind === "module");
  assert.deepEqual(mod[0]!.inputs, { message: "Where is it?", tone: "kind" });                   // left out with no default: absent
  assert.equal(mod[0]!.program.interface, support.interfaceId);
  assert.ok(!("signature" in mod[0]!.program));
  assert.deepEqual([mod[1]!.error.type, mod[1]!.error.code, mod[1]!.outputs], ["InterfaceError", "interface-input", null]);
  const liar = module("liar", { input: {}, outputs: { a: t.integer(), b: t.string() } }, () => ({ a: 1.5, b: "x" }) as never);
  await assert.rejects(liar({}), (e: unknown) => e instanceof InterfaceError && e.code === "interface-output" && e.field === "a");
  const other = module("support", { input: { message: t.string() }, output: t.string() }, async () => "");
  assert.notEqual(other.version, support.version);
  const c = new AbortController();
  let started = 0;
  const slow = module("slow", { input: {}, output: t.string() }, (_i, { signal }) => new Promise<string>((_r, reject) => {
    started++;
    signal.addEventListener("abort", () => reject(new Cancelled()), { once: true });
  }));
  const early = slow({}, { signal: c.signal });
  c.abort();
  await assert.rejects(early, Cancelled);
  assert.equal(started, 0);                                     // cancelled before its code started: it does not start
  const d = new AbortController();
  const pending = slow({}, { signal: d.signal });
  await new Promise((r) => setTimeout(r, 1));
  d.abort();
  await assert.rejects(pending, Cancelled);
  assert.equal(started, 1);
});

test("a follower of a live stream across a later writer rewinds; one that misses events recovers from the store", async () => {
  const store = new MemoryStore();
  const s = mood(new FakeRouter(["<result>\nhappy\n</result>"]), { journal: store }).stream("x");
  const whole = await all(s.events());
  await s;
  await new Promise((r) => setTimeout(r, 5));
  const reader = new Follower({ form: "live" });
  assert.equal(reader.receive(whole[0]!), "kept");
  assert.equal(reader.receive(whole[2]!), "loss");            // it missed one
  await reader.recover(s.tree!, store);                         // a live reader starts again from a store
  assert.deepEqual(reader.state(s.tree!), stateOf(store.events(s.tree!)));
});

test("a module saves with its interface and the AI functions it uses: described in any language, its AI functions loaded by key", () => {
  const answer = mood(new FakeRouter([]), { definedIn: "shop" });
  const support = module("support", { description: "Answer a customer.", input: { message: t.string() }, output: t.string(), uses: [answer], definedIn: "shop" },
    async ({ message }) => String(await answer(message)));
  const manifest = JSON.parse(JSON.stringify(toManifest(support)));
  assert.ok(passes("saved", manifest));
  assert.deepEqual(Object.keys(manifest.nodes), ["shop:support", "shop:mood"]);
  assert.deepEqual(describeSaved(manifest), support.interface);
  assert.deepEqual(describeSaved(manifest, { node: "shop:mood" }), answer.interface);
  assert.throws(() => fromManifest(manifest), (e: unknown) => e instanceof LoadRefused && e.code === "saved-not-ai");
  assert.equal(fromManifest(manifest, { node: "shop:mood" }).version, answer.version);
});
