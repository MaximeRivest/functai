/**
 * What goes wrong around a call, through real calls (a fake model, no
 * network): stores that throw, never answer, answer nonsense or mutate what
 * they are given; slow, failing, mutating and flooded observers; refused
 * inputs and policies; cancellation; names JavaScript treats specially;
 * unknown formats. Each test here was a defect a review found (its probe,
 * made permanent), or the guard beside one.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { getEventListeners } from "node:events";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";
import { MessageChannel } from "node:worker_threads";
import { test } from "node:test";
import { RateLimitError, responseToEvents, streamDelta, type Request, type StreamEvent as LmEvent } from "@lm15/lm15";
import * as z from "zod";
import {
  ai, Cancelled, checkInterface, configure, evaluate, flush, Follower, fromManifest, InterfaceError, JournalError, load, LoadRefused, MemoryStore, module, Prediction, replay,
  save, SettingError, settle as settleLog, t, toManifest, tool, withSettings, type AppendAnswer, type StandardSchemaLike, type EventStore, type Position, type ReadAnswer, type StreamEvent,
} from "../src/index.ts";
import { holdAtMost } from "../src/log.ts";
import { passes } from "../src/schema.ts";
import { jsonForm } from "../src/values.ts";
import { FakeRouter } from "./fake.ts";
import * as lmcc from "lmcc";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

type Rec = Record<string, any>;
const here = dirname(fileURLToPath(import.meta.url));
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
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
const folder = () => mkdtempSync(join(tmpdir(), "functai-fail-"));
/** Run `fn` with console.warn collected. */
async function quietly<T>(fn: (warned: string[]) => Promise<T>): Promise<T> {
  const warn = console.warn;
  const warned: string[] = [];
  console.warn = (m: string) => { warned.push(String(m)); };
  try {
    return await fn(warned);
  } finally {
    console.warn = warn;
  }
}
/** Unhandled rejections while `fn` runs (and a moment after). */
async function unhandledDuring(fn: () => Promise<void>): Promise<unknown[]> {
  const seen: unknown[] = [];
  const listener = (e: unknown) => { seen.push(e); };
  process.on("unhandledRejection", listener);
  try {
    await fn();
    await delay(30);
  } finally {
    process.removeListener("unhandledRejection", listener);
  }
  return seen;
}

const mood = (router: unknown, extra: Rec = {}) => ai("mood", {
  description: "How does the customer feel?", input: { review: t.string() }, output: t.enum("happy", "unhappy", "mixed"),
  router: router as never, ...extra,
});
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

/** A store whose appends are answered by `answer` (the n-th append, from 1, and its events). */
class Scripted implements EventStore {
  readonly inner = new MemoryStore();
  sends = 0;
  readonly at: number[] = [];
  readonly sizes: number[] = [];
  readonly signals: AbortSignal[] = [];
  private readonly answer: (n: number, events: readonly Rec[], inner: MemoryStore) => unknown;
  constructor(answer: (n: number, events: readonly Rec[], inner: MemoryStore) => unknown) {
    this.answer = answer;
  }
  /** Set: the store is well again, and keeps what it is sent. */
  healed = false;
  append(events: readonly Rec[], opts?: { signal?: AbortSignal }): Promise<AppendAnswer> {
    if (this.healed) return Promise.resolve(this.inner.appendNow(events));
    this.sends++;
    this.at.push(performance.now());
    this.sizes.push(events.length);
    if (opts?.signal) this.signals.push(opts.signal);
    return this.answer(this.sends, events, this.inner) as Promise<AppendAnswer>;
  }
  read(tree: string, after: Position | null): Promise<ReadAnswer> {
    return this.inner.read(tree, after);
  }
}
const never = () => new Promise<never>(() => undefined);
/**
 * Stores that were down are well again: `flush()` sends each writer's
 * events that were not confirmed (the tree's end among them), and says it
 * is all kept. It also leaves no writer holding events for the tests after.
 */
async function heal(...stores: Scripted[]): Promise<void> {
  for (const s of stores) s.healed = true;
  assert.ok(await flush(), "flush sends again what was not confirmed");
  for (const s of stores) for (const tree of s.inner.trees()) assert.ok(s.inner.finished(tree), "the tree's end is kept after all");
}

// ------------------------------------------------------------------ journals: liveness

test("a store that throws before returning a promise never spins the process: the call settles and timers fire (own process, time-limited)", () => {
  const run = spawnSync(process.execPath, ["--conditions=functai-source", join(here, "fixtures", "sync-throwing-store.ts")],
    { encoding: "utf8", timeout: 20_000 });
  assert.equal(run.signal, null, `killed after 20 s: the process hung (${run.stderr})`);
  assert.equal(run.status, 0, run.stderr);
  const got = JSON.parse(run.stdout.trim().split("\n").at(-1)!);
  assert.equal(got.out, "JournalError journal-end unknown journal-barrier");   // the start was not kept; nor was the end
  assert.equal(got.fired, true);
  assert.equal(got.ran, false);
  assert.ok(got.ms < 2000, `${got.ms} ms`);
});

test("a required journal that never answers: the start barrier gives up at its timeout, the code never runs, each append's signal aborts", async () => {
  await quietly(async () => {
    const store = new Scripted(never);
    let ran = false;
    const f = module("stalled", { input: {}, output: t.string(), journal: { store, mode: "required", timeout: 100 } }, () => {
      ran = true;
      return "ok";
    });
    const t0 = performance.now();
    const err = await f({}).then(() => null, (e: unknown) => e);
    const ms = performance.now() - t0;
    assert.ok(err instanceof JournalError && err.code === "journal-end" && err.journal === "unknown", String(err));
    assert.ok("failed" in err.outcome! && err.outcome.failed instanceof JournalError && err.outcome.failed.code === "journal-barrier");
    assert.equal(ran, false);
    assert.ok(ms < 1000, `${ms} ms`);                                    // start: 100 ms, end: 100 ms
    assert.ok(store.signals[0]!.aborted, "the append that did not answer in time is told so");
    await heal(store);
  });
});

test("cancelling a call waiting at the start barrier stops the wait: Cancelled with a healthy store, journal-end holding Cancelled with a dead one", async () => {
  await quietly(async () => {
    const slow = new Scripted(async (_n, events, inner) => {
      await delay(30);
      return inner.appendNow(events);
    });
    let ran = false;
    const code = () => {
      ran = true;
      return "ok";
    };
    const a = new AbortController();
    const f = module("slowly", { input: {}, output: t.string(), journal: { store: slow, mode: "required" } }, code);
    const pending = f({}, { signal: a.signal });
    setTimeout(() => a.abort(), 5);
    await assert.rejects(pending, Cancelled);                           // its end is kept a moment later, then it raises
    assert.equal(ran, false);
    assert.deepEqual(slow.inner.events(slow.inner.trees()[0]!).map((e) => e["kind"]), ["started", "failed"]);

    const dead = new Scripted(never);
    const b = new AbortController();
    const g = module("stalled", { input: {}, output: t.string(), journal: { store: dead, mode: "required", timeout: 150 } }, code);
    const t0 = performance.now();
    const waiting = g({}, { signal: b.signal }).then(() => null, (e: unknown) => e);
    setTimeout(() => b.abort(), 5);
    const err = await waiting;
    assert.ok(err instanceof JournalError && err.code === "journal-end", String(err));
    assert.ok("failed" in err.outcome! && err.outcome.failed instanceof Cancelled);
    assert.ok(performance.now() - t0 < 1000);
    assert.equal(ran, false);
    await heal(dead);
  });
});

test("a required journal that stops answering before a tool: the tool never runs, and the call ends within the timeouts; closing the stream there cancels it", async () => {
  await quietly(async () => {
    // started, request, then the tool call's append never answers
    const hang = () => new Scripted((n, events, inner) => (n >= 3 ? never() : Promise.resolve(inner.appendNow(events))));
    const ran: string[] = [];
    const store = hang();
    const err = await helper(toolReplies(), ran, { journal: { store, mode: "required", batch: 1, timeout: 100 } })("Where is B-2210?")
      .then(() => null, (e: unknown) => e);
    assert.deepEqual(ran, []);
    assert.ok(err instanceof JournalError && err.code === "journal-end");
    assert.ok("failed" in err.outcome! && (err.outcome.failed as JournalError).code === "journal-barrier");

    const other = hang();
    const s = helper(toolReplies(), ran, { journal: { store: other, mode: "required", batch: 1, timeout: 300 } }).stream("Where is B-2210?");
    for await (const e of s.events()) if (e.kind === "tool_call") s.close();
    const closed = await s.result.then(() => null, (e: unknown) => e);
    assert.deepEqual(ran, []);
    assert.ok(closed instanceof JournalError && "failed" in closed.outcome! && closed.outcome.failed instanceof Cancelled, String(closed));
    await heal(store, other);
  });
});

test("a required journal that never answers the end: journal-end unknown within its timeout, holding the value", async () => {
  await quietly(async () => {
    const store = new Scripted((_n, events, inner) =>
      events.some((e) => e["kind"] === "done") ? never() : Promise.resolve(inner.appendNow(events)));
    const f = mood(new FakeRouter(["<result>\nhappy\n</result>"]), { journal: { store, mode: "required", timeout: 100 } });
    const err = await f("Lovely").then(() => null, (e: unknown) => e);
    assert.ok(err instanceof JournalError && err.code === "journal-end" && err.journal === "unknown");
    assert.deepEqual(err.outcome, { done: "happy" });
    assert.equal(await err.settle(), "not-kept");
    await heal(store);
    assert.equal(await err.settle(), "kept");                           // the writer went on sending the end after the call raised
  });
});

test("a best-effort journal answered with nonsense never reaches the call nor the process; a required one takes it as a refusal", async () => {
  for (const answer of [{ refuses: "event-conflict" }, undefined, "ok", null, 42, { event: 1 }]) {
    await quietly(async (warned) => {
      const store = new Scripted(async () => answer);
      const unhandled = await unhandledDuring(async () => {
        assert.equal(await mood(new FakeRouter(["<result>\nhappy\n</result>"]), { journal: store })("x"), "happy");
        assert.ok(await flush());
      });
      assert.deepEqual(unhandled, [], JSON.stringify(answer));
      assert.equal(store.sends, 1, `nothing more is sent after a refusal (${JSON.stringify(answer)})`);
      // once for this store (earlier tests' dead stores may still be giving up in the background: "did not answer")
      assert.equal(warned.filter((w) => !w.includes("did not answer")).length, 1, warned.join("\n---\n"));
      const required = new Scripted(async () => answer);
      const err = await mood(new FakeRouter(["<result>\nhappy\n</result>"]), { journal: { store: required, mode: "required" } })("x")
        .then(() => null, (e: unknown) => e);
      assert.ok(err instanceof JournalError && err.code === "journal-end" && err.journal === "refused", JSON.stringify(answer));
      assert.ok("failed" in err.outcome! && (err.outcome.failed as JournalError).code === "journal-barrier");
    });
  }
});

test("resends back off: the second waits backoff, the third twice as long", async () => {
  await quietly(async () => {
    const store = new Scripted(async () => {
      throw new Error("down");
    });
    const f = module("m", { input: {}, output: t.string(), journal: { store, mode: "required", retries: 2, backoff: 40, timeout: 1000 } }, () => "ok");
    await assert.rejects(f({}), JournalError);
    const [a, b, c] = store.at;
    assert.ok(b! - a! >= 38, `${b! - a!}`);
    assert.ok(c! - b! >= 78, `${c! - b!}`);
    await heal(store);
  });
});

test("a journal sends what waits in batches (one append, several events), each append its own copies", async () => {
  const store = new Scripted(async (_n, events, inner) => {
    await delay(5);
    const answer = inner.appendNow(events);
    for (const e of events) (e as Rec)["_id"] = "a store's own mark";      // a store that marks what it is given (MongoDB's insert)
    if (_n === 2) throw new Error("the answer was lost");                    // then loses the answer: the resend must be the same events
    return answer;
  });
  const f = mood(new FakeRouter(["<result>\n" + "happy".repeat(1) + "\n</result>"], null, "openai", 1), { journal: { store, mode: "required" } });
  assert.equal(await f("x"), "happy");
  assert.ok(Math.max(...store.sizes) > 1, `appends of ${store.sizes.join(", ")} events`);
  const tree = store.inner.trees()[0]!;
  assert.ok(store.inner.finished(tree));
  assert.ok(store.inner.events(tree).every((e) => !("_id" in e)));
});

test("a barrier waits for its own event, not for a sibling's later events to drain", async () => {
  class Slow extends MemoryStore {
    override async append(events: readonly Rec[]) {
      await delay(5);
      return this.appendNow(events);
    }
  }
  const t0 = performance.now();
  const at: Record<string, number> = {};
  const lookup = tool("lookup", { description: "x", input: { order: t.string() } }, () => {
    at["tool"] = performance.now() - t0;
    return "ok";
  });
  const withTool = ai("withTool", { description: "x", input: { q: t.string() }, tools: [lookup],
    router: new FakeRouter([{ text: "", calls: [{ id: "c1", name: "lookup", input: { order: "1" } }] }, "<result>\ndone\n</result>"]) as never });
  class Chatty extends FakeRouter {
    override async *stream(r: Request): AsyncIterable<LmEvent> {
      for await (const e of super.stream(r)) {
        await delay(1);
        yield e;
      }
    }
  }
  const chatty = ai("chatty", { description: "x", input: { q: t.string() }, router: new Chatty([], () => "<result>\n" + "blah ".repeat(150) + "\n</result>", "openai", 3) as never });
  const both = module("both", { input: {}, output: t.list(t.string()), uses: [withTool, chatty], journal: { store: new Slow(), mode: "required", batch: 1 } },
    async () => Promise.all([chatty("x").then((v) => { at["chatty"] = performance.now() - t0; return v.slice(0, 5); }), withTool("y")]));
  await both({});
  assert.ok(at["tool"]! < at["chatty"]!, `tool ran at ${at["tool"]} ms, the sibling ended at ${at["chatty"]} ms`);
});

// ------------------------------------------------------------------ observers

test("a slow observer does not slow the call: it is given events after the call's own turn", async () => {
  const seen: string[] = [];
  const busy = (e: StreamEvent) => {
    const until = performance.now() + 50;
    while (performance.now() < until) { /* a synchronous socket, a heavy exporter */ }
    seen.push(e.kind);
  };
  const f = module("quick", { input: {}, output: t.string(), observers: [busy, busy] }, () => "ok");
  const t0 = performance.now();
  assert.equal(await f({}), "ok");
  const ms = performance.now() - t0;
  assert.ok(ms < 40, `${ms} ms`);
  assert.ok(await flush());
  assert.deepEqual(seen, ["started", "done"]);                 // the same observer twice is one observer
});

test("no receiver can change what another receives: each observer, the journal and the stream have their own copies", async () => {
  const inner = new MemoryStore();
  let open!: () => void;
  const gate = new Promise<void>((r) => { open = r; });
  const store: EventStore = { append: async (es) => { await gate; return inner.append(es); }, read: (tree, after) => inner.read(tree, after) };
  const second: StreamEvent[] = [];
  const f = module("mutate", {
    input: { payload: t.json(), hidden: t.string() }, output: t.string(), logContent: { hidden: false }, journal: store,
    observers: [(e) => { if (e.kind === "started") (e.inputs as Rec)["payload"].x = "MUTATED"; }, (e) => { second.push(e); }],
  }, ({ payload }) => (payload as Rec)["x"] as string);
  const s = f.stream({ payload: { x: "original" }, hidden: "secret" });
  assert.equal(await s, "original");
  await delay(10);                                              // the observers have their events; the journal still waits
  assert.equal(second.length, 2);
  open();
  assert.ok(await flush());
  assert.equal((second[0] as Rec).inputs.payload.x, "original");
  assert.equal((inner.events(s.tree!)[0] as Rec)["inputs"].payload.x, "original");
  assert.equal((s.read(s.tree, null) as Rec).events[0].inputs.payload.x, "original");
  assert.ok(!JSON.stringify([second, inner.events(s.tree!)]).includes("secret"));
});

test("each failing observer is warned about once, by name when it has one, and given no more events; others go on", async () => {
  await quietly(async (warned) => {
    const good: string[] = [];
    const f = module("m", {
      input: {}, output: t.string(),
      observers: [() => { throw new Error("socket closed"); }, () => Promise.reject(new Error("rejected")), (e) => { good.push(e.kind); }],
    }, () => "ok");
    await f({});
    await f({});
    assert.ok(await flush());
    await delay(5);
    assert.equal(warned.filter((w) => w.includes("failed")).length, 2, warned.join("\n"));
    assert.deepEqual(good, ["started", "done", "started", "done"]);
  });
});

test("an observer with postMessage (a Worker, a MessagePort) is posted each event: it handles them on its own thread", async () => {
  const { port1, port2 } = new MessageChannel();
  const got: StreamEvent[] = [];
  port2.on("message", (e: StreamEvent) => { got.push(e); });
  try {
    await module("m", { input: { q: t.string() }, output: t.string(), observers: [port1] }, ({ q }) => q)("hi");
    assert.ok(await flush());
    for (let i = 0; i < 50 && got.length < 2; i++) await delay(2);
    assert.deepEqual(got.map((e) => e.kind), ["started", "done"]);
    assert.ok(got.every((e) => passes("event", e)));
  } finally {
    port1.close();
    port2.close();
  }
});

test("an observer that falls 10,000 events behind loses events, is told once, and sees the gap in after", async () => {
  class Burst extends FakeRouter {
    override async *stream(r: Request): AsyncIterable<LmEvent> {
      const response = await this.complete(r);
      for (const e of responseToEvents(response)) {
        const d = e.type === "delta" ? (e.delta as { type: string; text?: string }) : null;
        if (d && d.type === "text" && d.text) {
          for (const ch of d.text) yield streamDelta({ ...d, text: ch } as never);   // all at once: no turn for anyone else
          continue;
        }
        yield e;
      }
    }
  }
  await quietly(async (warned) => {
    const got: StreamEvent[] = [];
    const f = mood(new Burst(["<result>\n" + "x".repeat(10_300) + "\n</result>"]), { observers: [(e: StreamEvent) => { got.push(e); }] });
    await f("x").catch(() => undefined);                                   // the answer does not fit the enum: no matter
    assert.ok(await flush());
    assert.ok(got.length === 10_000, `${got.length}`);
    assert.equal(warned.filter((w) => w.includes("behind")).length, 1);
  });
});

// ------------------------------------------------------------------ retention

test("a refused module input under another name is kept by no record, event, observer or journal, even under a host allowlist", async () => {
  const where = folder();
  const store = new MemoryStore();
  const seen: StreamEvent[] = [];
  const f = module("support", { input: { public: t.string() }, output: t.string(), logCalls: where, journal: { store, mode: "required" },
    observers: [(e) => { seen.push(e); }] }, ({ public: p }) => p);
  const s = withSettings({ logContent: { "*": false, public: true, result: true } }, () => f.stream({ public: "safe", extra: "SECRET" } as never));
  const whole = await all(s.events());
  await assert.rejects(s.result, (e: unknown) => e instanceof InterfaceError && e.code === "interface-input" && e.field === "extra");
  assert.ok(await flush());
  const [record] = logged(where);
  for (const [what, x] of [["record", record], ["observer", seen], ["journal", store.events(store.trees()[0]!)], ["stream", whole]] as const) {
    assert.ok(!JSON.stringify(x).includes("SECRET"), what);
  }
  assert.deepEqual(record!.inputs, { public: "safe" });
  assert.equal(record!.content, true);
  assert.ok(passes("call", record));
});

test("a refused input's value is never quoted in the error: it names the field and the kind of value", async () => {
  const where = folder();
  const f = module("m", { input: { pin: t.integer() }, output: t.string(), logCalls: where }, () => "x");
  const err = await f({ pin: "hunter2-not-a-number" } as never).then(() => null, (e: unknown) => e);
  assert.ok(err instanceof InterfaceError && err.field === "pin");
  assert.ok(!err.message.includes("hunter2"), err.message);
  assert.ok(err.message.includes("a string"), err.message);
  const outer = module("outer", { input: { note: t.string() }, output: t.string(), logCalls: where }, async () => f({ pin: "hunter2" } as never));
  await assert.rejects(outer({ note: "n" }));
  const errors = logged(where).map((r) => r.error);
  assert.equal(errors.length, 3);
  assert.ok(!JSON.stringify(errors).includes("hunter2"), "a parent whose content is whole keeps the child's message");
});

test("a saved function's own logContent is checked when it is loaded: a misspelt field refuses log-content-field", () => {
  const manifest = toManifest(ai("saved_policy", { input: { secret: t.string() }, output: t.string() })) as Rec;
  const node = manifest.nodes[manifest.entry].ai;
  node.settings.log_content = { secrett: false };
  assert.throws(() => fromManifest(manifest), (e: unknown) => e instanceof SettingError && e.code === "log-content-field" && e.field === "secrett");
  node.settings.log_content = { "input:secret": false };
  assert.throws(() => fromManifest(manifest), (e: unknown) => e instanceof SettingError && e.field === "input:secret");
  node.settings.log_content = { secret: false };
  assert.deepEqual(fromManifest(manifest).settings.logContent, { secret: false });
});

test("policy settings that could only fail later refuse where they are set: definitions, using, blocks, configure, a call's options", async () => {
  const m = module("m", { input: {}, output: t.string() }, () => "ok");
  await assert.rejects(m({}, { logContent: { "input:secret": false } }), (e: unknown) => e instanceof SettingError && e.field === "input:secret");
  await assert.rejects(m({}, { observers: "not a list" as never }), TypeError);
  await assert.rejects(m({}, { journal: {} as never }), TypeError);
  assert.throws(() => configure({ observers: [42 as never] }), TypeError);
  assert.throws(() => withSettings({ journal: { store: new MemoryStore(), mode: "sometimes" as never } }, () => 0), TypeError);
  assert.throws(() => withSettings({ journal: { store: new MemoryStore(), timeout: 0 } }, () => 0), TypeError);
  assert.throws(() => module("m", { input: {}, output: t.string(), journal: 5 as never }, () => ""), TypeError);
  assert.throws(() => m.using({ logContent: { typo: false } }), (e: unknown) => e instanceof SettingError && e.field === "typo");
  assert.throws(() => mood(new FakeRouter([]), { observers: [null] }), TypeError);
});

test("a tool loop's record holds every output the model gave, calls included, with its size; dropping a field drops it too", async () => {
  const where = folder();
  const ran: string[] = [];
  await helper(toolReplies(), ran, { logCalls: where })("Where is B-2210?");
  await helper(toolReplies(), ran, { logCalls: where, logContent: { question: false } })("Where is B-2210?");
  const [whole, partial] = logged(where);
  assert.deepEqual(whole!.outputs, { result: "It is in Leeds, a week late.", calls: [] });
  assert.equal(whole!.sizes.outputs.calls, 2);
  assert.ok(passes("call", whole));
  assert.deepEqual(partial!.omitted, { inputs: ["question"], outputs: ["calls"] });
  assert.deepEqual(partial!.outputs, { result: "It is in Leeds, a week late." });
  assert.equal(partial!.sizes.outputs.calls, 2);
});

test("a module given inputs it cannot bind (a value alone, two inputs) records the refused call, as any refusal", async () => {
  const where = folder();
  const seen: StreamEvent[] = [];
  const f = module("pair", { input: { a: t.string(), b: t.string() }, output: t.string(), logCalls: where, observers: [(e) => { seen.push(e); }] }, () => "ok");
  await assert.rejects(f(123 as never), (e: unknown) => e instanceof InterfaceError && e.code === "interface-input");
  assert.ok(await flush());
  assert.deepEqual(seen.map((e) => e.kind), ["started", "failed"]);
  const [record] = logged(where);
  assert.deepEqual([record!.error.type, record!.error.code, record!.outputs, record!.inputs], ["InterfaceError", "interface-input", null, {}]);
});

test("the record and the started event hold the inputs as given, when the call starts: not a schema's transform, nor what the code did to them", async () => {
  const where = folder();
  const seen: StreamEvent[] = [];
  const f = module("m", { input: { name: z.string().transform((s) => s.toUpperCase()), tags: t.list(t.string()) }, output: t.string(), logCalls: where,
    observers: [(e) => { seen.push(e); }] }, ({ name, tags }) => {
    tags.push("added by the code");
    return name;
  });
  assert.equal(await f({ name: "ana", tags: ["a"] }), "ANA");
  assert.ok(await flush());
  const [record] = logged(where);
  assert.deepEqual(record!.inputs, { name: "ana", tags: ["a"] });
  assert.deepEqual((seen[0] as Rec).inputs, record!.inputs);
});

// ------------------------------------------------------------------ declarations

test("a malformed declaration refuses interface-malformed, naming its field; an output cannot be optional, nor an AI function's opaque", () => {
  const malformed = (field: string) => (e: unknown) => e instanceof InterfaceError && e.code === "interface-malformed" && e.field === field;
  assert.throws(() => module("m", { input: { x: { shape: null as never } }, output: t.string() }, () => ""), malformed("x"));
  assert.throws(() => module("m", { input: { x: 5 as never }, output: t.string() }, () => ""), malformed("x"));
  assert.throws(() => module("m", { input: {}, outputs: { a: { shape: t.string(), optional: true }, b: t.string() } }, () => ({ a: "", b: "" }) as never), malformed("a"));
  assert.throws(() => ai("f", { input: { q: t.string() }, outputs: { frame: t.opaque() } }), malformed("frame"));
  assert.throws(() => ai("f", { input: { q: { shape: undefined as never } } }), malformed("q"));
});

test("every builder keeps the words given with it (none is silently dropped)", () => {
  const f = ai("f", { input: {
    a: t.list(t.string(), { description: "one per line" }), b: t.object({ id: t.integer() }, { description: "an order" }),
    c: t.json({ description: "each input by name" }), d: t.record(t.integer(), { description: "counts" }), e: t.string({ description: "text" }),
  } });
  assert.deepEqual(f.interface.inputs.map((x) => x.desc), ["one per line", "an order", "each input by name", "counts", "text"]);
});

test("one output, whatever its name, is the value; several are a record", async () => {
  const one = module("one", { input: {}, outputs: { count: t.integer() } }, () => 1);
  assert.equal(await one({}), 1);
  assert.equal(one.interface.outputs[0]!.name, "count");
});

// ------------------------------------------------------------------ validation and cancellation

const asyncSchema = (refuse: boolean): StandardSchemaLike<string, string> => ({
  "~standard": {
    vendor: "test",
    jsonSchema: { input: () => ({ type: "string" }) },
    validate: (v: unknown) => (v === undefined ? { issues: [{ message: "required" }] }
      : Promise.resolve(refuse ? { issues: [{ message: "bad" }] } : { value: v })),
  },
});

test("async validation never leaves a rejection unhandled: render refuses to wait, and a refusal beside another is the first in order", async () => {
  const unhandled = await unhandledDuring(async () => {
    const f = ai("async_shape", { input: { x: asyncSchema(true) }, output: t.string(), router: new FakeRouter([]) as never });
    assert.throws(() => f.render("x"), /validates asynchronously/);
    const throwing: StandardSchemaLike<string, string> = { "~standard": { vendor: "test", jsonSchema: { input: () => ({ type: "string" }) },
      validate: (v: unknown) => { if (v === undefined) return { issues: [{ message: "required" }] }; throw new Error("the validator broke"); } } };
    const g = ai("mixed", { input: { a: asyncSchema(true), b: throwing }, output: t.string(), router: new FakeRouter([]) as never });
    await assert.rejects(g({ a: "x", b: "y" }), (e: unknown) => e instanceof InterfaceError && e.field === "a");
    const h = module("mixed", { input: { a: asyncSchema(false), b: throwing }, output: t.string() }, () => "ok");
    await assert.rejects(h({ a: "x", b: "y" }), /the validator broke/);
  });
  assert.deepEqual(unhandled, []);
});

test("closing a module's stream while its code runs ends the call Cancelled, whatever the code returns after", async () => {
  const where = folder();
  let open!: () => void;
  const gate = new Promise<void>((r) => { open = r; });
  const f = module("slow", { input: {}, output: t.string(), logCalls: where }, async () => {
    await gate;
    return "ok";
  });
  const s = f.stream({});
  await delay(5);
  s.close();
  open();
  await assert.rejects(s.result, Cancelled);
  const events = await all(s.events());
  assert.equal(events.at(-1)!.kind, "failed");
  assert.equal(logged(where)[0]!.error.type, "Cancelled");
});

test("closing a module's stream while its inputs are parsed (an async schema) cancels it before its code runs", async () => {
  let open!: () => void;
  const gate = new Promise<void>((r) => { open = r; });
  let ran = false;
  const f = module("parsing", { input: { q: z.string().refine(async () => { await gate; return true; }) }, output: t.string() }, ({ q }) => {
    ran = true;
    return q;
  });
  const s = f.stream("x");
  await delay(5);
  s.close();
  open();
  await assert.rejects(s.result, Cancelled);
  assert.equal(ran, false);
});

test("a long-lived signal keeps no finished stream, and a retry's wait lets go of the caller's signal", async () => {
  const server = new AbortController();
  const quick = module("quick", { input: {}, output: t.string() }, () => "ok");
  for (let i = 0; i < 15; i++) await quick.stream({}, { signal: server.signal });
  const f = mood(new FakeRouter([], () => "<result>\nhappy\n</result>"));
  for (let i = 0; i < 15; i++) await f.stream("x", { signal: server.signal });
  assert.equal(getEventListeners(server.signal, "abort").length, 0);

  let failures = 1;
  const flaky = {
    resolve: (m: string) => ({ provider: "openai", model: m }),
    complete: async (r: Request) => {
      if (failures-- > 0) throw Object.assign(new RateLimitError("slow down"), { retryAfter: 0.001 });
      return new FakeRouter(["<result>\nhappy\n</result>"]).complete(r);
    },
  };
  const caller = new AbortController();
  assert.equal(await mood(flaky)("x", { signal: caller.signal }), "happy");
  assert.equal(getEventListeners(caller.signal, "abort").length, 0);
});

// ------------------------------------------------------------------ names JavaScript treats specially

test("names JavaScript treats specially are data: __proto__ in JSON, required toString, a keyword named constructor, fields named so", async () => {
  assert.equal(JSON.stringify(jsonForm(JSON.parse('{"__proto__":{"admin":true},"x":1}'))), '{"__proto__":{"admin":true},"x":1}');
  const needs = module("needs", { input: { x: { type: "object", required: ["toString"] } }, output: t.string() }, () => "accepted");
  await assert.rejects(needs({ x: {} }), (e: unknown) => e instanceof InterfaceError && e.field === "x");
  assert.equal(await needs({ x: { toString: 1 } }), "accepted");
  assert.throws(() => checkInterface({ description: "", inputs: [{ name: "x", shape: { type: "string", constructor: 42 } }], outputs: [{ name: "result", shape: {} }] }),
    (e: unknown) => e instanceof InterfaceError && e.code === "interface-malformed");
  const where = folder();
  const seen: StreamEvent[] = [];
  const odd = module("odd", {
    input: { constructor: t.string(), toString: t.string(), ["__proto__"]: t.string() } as never, output: t.string(), logCalls: where,
    logContent: { toString: false }, observers: [(e) => { seen.push(e); }],
  }, (i: Rec) => `${i["constructor"]} ${i["__proto__"]}`);
  const input = JSON.parse('{"constructor":"c","toString":"secret","__proto__":"p"}');
  assert.equal(await odd(input), "c p");
  assert.ok(await flush());
  const [record] = logged(where);
  assert.deepEqual(Object.keys(record!.inputs).sort(), ["__proto__", "constructor"]);
  assert.deepEqual(record!.omitted.inputs, ["toString"]);
  assert.equal(record!.sizes.inputs["__proto__"], 3);
  assert.ok(!JSON.stringify(seen).includes("secret"));
  await assert.rejects(odd({ constructor: "c", __proto__x: "p" } as never), (e: unknown) => e instanceof InterfaceError);
});

// Object's own members are names like any other. Before, an AI function bound and prepared its inputs by prototype
// lookups: a missing required toString or constructor was sent to the model as JavaScript function text.
const MEMBERS = ["toString", "constructor", "hasOwnProperty", "valueOf", "isPrototypeOf", "propertyIsEnumerable", "toLocaleString", "__defineGetter__"];
const sent = (r: unknown) => JSON.stringify(r);
const ok = () => new FakeRouter([], () => "<result>\nok\n</result>");

test("an AI function's input named like an Object member is sent as given, by name or alone, and render and a loaded copy send the same", async () => {
  for (const name of MEMBERS) {
    const router = ok();
    const f = ai("odd", { input: { [name]: t.string() }, output: t.string(), router: router as never });
    const given = { [name]: `VALUE-${name}` };
    assert.equal(await f(given as never), "ok");
    assert.equal(await f(`VALUE-${name}` as never), "ok");                  // its one input: the value alone
    const [byName, alone] = router.requests;
    assert.ok(sent(byName).includes(`<${name}>\\nVALUE-${name}\\n</${name}>`), sent(byName));
    assert.deepEqual(alone, byName);
    assert.deepEqual(f.render(given as never), byName);                     // what render shows is what is sent
    const loaded = fromManifest(JSON.parse(JSON.stringify(toManifest(f)))).using({ router: router as never });
    assert.equal(loaded.version, f.version);
    assert.equal(await loaded(given as never), "ok");
    assert.deepEqual(router.requests[2], byName);                           // saved and loaded: the very request
    assert.ok(!router.requests.some((r) => sent(r).includes("native code")));
  }
});

test("an AI function's required input named like an Object member, left out, is refused before anything is sent: by a call, render, a stream and a loaded copy", async () => {
  for (const name of MEMBERS) {
    const router = ok();
    const f = ai("odd", { input: { [name]: t.string() }, output: t.string(), router: router as never });
    const loaded = fromManifest(JSON.parse(JSON.stringify(toManifest(f)))).using({ router: router as never });
    const refused = (e: unknown) => e instanceof InterfaceError && e.code === "interface-input" && e.field === name;
    for (const g of [f, loaded]) {
      await assert.rejects(g({} as never), refused, name);
      await assert.rejects(g.predict({} as never), refused, name);
      assert.throws(() => g.render({} as never), refused, name);
      assert.throws(() => g.stream({} as never), refused, name);
      await assert.rejects(g.map([{}] as never), refused, name);
    }
    assert.equal(router.requests.length, 0, `nothing is sent (${name})`);
  }
});

test("an optional input named toString or constructor, left out, is sent with its default or null, never with an Object member", async () => {
  const router = ok();
  const f = ai("odd", { input: { q: t.string(), toString: t.string({ default: "kind" }), constructor: t.optional(t.string()) } as never, output: t.string(), router: router as never });
  assert.deepEqual(f.interface.inputs.map((i) => [i.name, i.optional ?? false, i.shape["default"]]), [["q", false, undefined], ["toString", true, "kind"], ["constructor", true, null]]);
  await f({ q: "x" } as never);
  await f("x" as never);
  const loaded = fromManifest(JSON.parse(JSON.stringify(toManifest(f)))).using({ router: router as never });
  await loaded({ q: "x" } as never);
  const [a, b, c] = router.requests;
  assert.ok(sent(a).includes("<toString>\\nkind\\n</toString>"), sent(a));
  assert.ok(!sent(a).includes("native code") && !sent(a).includes("function Object"), sent(a));
  assert.deepEqual(b, a);
  assert.deepEqual(c, a);
  assert.deepEqual(f.render({ q: "x" } as never), a);
});

test("worked examples and rows named like Object members: a demo's value is sent, and a row or demo without the column is never given Object's", async () => {
  const router = ok();
  const f = ai("odd", { input: { toString: t.string() }, output: t.string(), router: router as never, demos: [{ toString: "an example", result: "its answer" }] } as never);
  await f("x" as never);
  assert.ok(sent(router.requests[0]).includes("<toString>\\nan example\\n</toString>"), sent(router.requests[0]));
  const ev = await evaluate(f, [{ result: "ok" }, { toString: "y", result: "ok" }] as never);
  assert.match(ev.rows[0]!.error!, /InterfaceError: odd needs toString/);             // the row has no toString column
  assert.equal(ev.rows[1]!.error, null);
});

// A worked example may give some fields only. lmcc's TypeScript renderer once read a turn's values with `name in values`
// and `values[name]`, so a missing toString was Object's member; it reads own members now (D-58).
test("a worked example without an input or an output named like an Object member (or __proto__) is sent without it: the call succeeds, render and a loaded copy send the same", async () => {
  const names = [...MEMBERS, "__proto__"];
  for (const [i, name] of names.entries()) {
    const out = MEMBERS[(i + 1) % MEMBERS.length]!;                        // an output named like another member, missing from the demos too
    const f = ai("demo", { input: { [name]: t.string(), q: t.string() }, outputs: { [out]: t.string(), result: t.string() },
      demos: [{ inputs: {}, outputs: { result: "EXAMPLE" } }, { inputs: { q: "only q" }, outputs: { result: "EXAMPLE2" } }] } as never);
    const given = { [name]: "ACTUAL", q: "Q" };
    const answers = new FakeRouter([], () => `<${out}>\na\n</${out}>\n<result>\nok\n</result>`);
    const g = f.using({ router: answers as never });
    assert.equal(await g(given as never), "ok", name);
    const request = answers.requests[0]!;
    const text = sent(request);
    assert.ok(text.includes("EXAMPLE") && text.includes("EXAMPLE2") && text.includes("only q"), text);
    assert.ok(text.includes(`<${name}>\\nACTUAL\\n</${name}>`), text);
    assert.ok(!text.includes("native code") && !text.includes("[object"), text);
    assert.equal(text.split(`<${name}>`).length - 1, 1, `only the call gives ${name}: ${text}`);
    assert.deepEqual(g.render(given as never), request);
    const loaded = fromManifest(JSON.parse(JSON.stringify(toManifest(f)))).using({ router: answers as never });
    assert.equal(loaded.version, f.version);
    assert.equal(await loaded(given as never), "ok");
    assert.deepEqual(answers.requests[1], request);
  }
  // a recorded turn, used as it is, without its toString input and its valueOf output
  const answers = new FakeRouter([], () => "<valueOf>\na\n</valueOf>\n<result>\nok\n</result>");
  const r = ai("demo", { input: { toString: t.string(), q: t.string() }, outputs: { valueOf: t.string(), result: t.string() }, router: answers as never } as never);
  r.demos = [{ signature: lmcc.signatureFingerprint(r.signature), inputs: { q: "REC" }, steps: [{ kind: "model", outputs: { result: "RECORDED" } }], outputs: { result: "RECORDED" } }] as never;
  assert.equal(await r({ toString: "ACTUAL", q: "Q" } as never), "ok");
  const text = sent(answers.requests[0]);
  assert.ok(text.includes("<q>\\nREC\\n</q>") && text.includes("RECORDED") && !text.includes("native code"), text);
});

test("__proto__ is an input name like any other: supplied, alone, left out, optional, in a demo, saved and loaded from disk, the very request is sent", async () => {
  const P = "__proto__";
  const router = ok();
  const f = ai("proto", { input: { [P]: t.string() }, output: t.string(), router: router as never, demos: [{ inputs: { [P]: "DEMO" }, outputs: { result: "EXAMPLE" } }] } as never);
  const given = JSON.parse('{"__proto__":"ACTUAL"}');
  assert.deepEqual(f.interface.inputs.map((i) => i.name), [P]);
  assert.equal(await f(given), "ok");
  assert.equal(await f("ACTUAL" as never), "ok");                           // its one input: the value alone
  const [byName, alone] = router.requests;
  assert.ok(sent(byName).includes("<__proto__>\\nACTUAL\\n</__proto__>"), sent(byName));
  assert.ok(sent(byName).includes("<__proto__>\\nDEMO\\n</__proto__>"), sent(byName));
  assert.deepEqual(alone, byName);
  assert.deepEqual(f.render(given), byName);
  await assert.rejects(f({} as never), (e: unknown) => e instanceof InterfaceError && e.code === "interface-input" && e.field === P);
  const where = mkdtempSync(join(tmpdir(), "functai-proto-"));
  save(f, where);
  const loaded = load(where).using({ router: router as never });
  assert.equal(loaded.version, f.version);
  assert.equal(await loaded(given), "ok");
  assert.deepEqual(router.requests.at(-1), byName);
  assert.equal(router.requests.length, 3, "the call left out sent nothing");

  const opt = ai("proto", { input: { q: t.string(), [P]: t.string({ default: "kind" }) }, output: t.string(), router: router as never } as never);
  await opt({ q: "x" } as never);
  assert.ok(sent(router.requests.at(-1)).includes("<__proto__>\\nkind\\n</__proto__>"), sent(router.requests.at(-1)));
  const json = ai("proto", { input: { [P]: t.json() }, output: t.string(), router: router as never } as never);
  await json(JSON.parse('{"__proto__":{"a":1}}'));
  assert.ok(sent(router.requests.at(-1)).includes('<__proto__>\\n{\\n  \\"a\\": 1\\n}\\n</__proto__>'), sent(router.requests.at(-1)));
});

test("an output named __proto__ is read as an answer like any other: by the xml and json adapters, saved and loaded from disk, and asked again when a reply leaves it out", async () => {
  const P = "__proto__";
  for (const adapter of ["xml", "json"]) {
    const reply = (v: string) => (adapter === "xml" ? `<${P}>\n${v}\n</${P}>\n<result>\nok\n</result>` : `{"${P}": "${v}", "result": "ok"}`);
    const router = new FakeRouter([], () => reply("THE ANSWER"));
    const f = ai("proto", { input: { q: t.string() }, outputs: { [P]: t.string(), result: t.string() }, adapter, router: router as never } as never);
    const where = mkdtempSync(join(tmpdir(), "functai-proto-out-"));
    save(f, where);
    const loaded = load(where).using({ router: router as never });
    assert.equal(loaded.version, f.version);
    for (const g of [f, loaded]) {
      const pred = await g.predict("x" as never);
      assert.ok(Object.hasOwn(pred.outputs as object, P), adapter);
      assert.equal((pred.outputs as Rec)[P], "THE ANSWER", adapter);
      assert.equal(Object.getPrototypeOf(pred.outputs), Object.prototype, "the answer is a member, not the prototype");
    }
    assert.deepEqual(router.requests[1], router.requests[0], "the loaded copy sends the very request");
    // a reply that leaves it out is asked again, then refused: never read as nothing
    const missing = new FakeRouter([], () => (adapter === "xml" ? "<result>\nok\n</result>" : '{"result": "ok"}'));
    for (const g of [f, loaded]) {
      const before = missing.requests.length;
      await assert.rejects(g.using({ router: missing as never })("x" as never), (e: unknown) => (e as { code?: string }).code === "parse-missing-fields", adapter);
      assert.equal(missing.requests.length - before, 2, "asked again once");
    }
  }
  const m = module("proto", { input: { [P]: t.string() } as never, output: t.string() }, (i: Rec) => `got ${i[P]}`);
  assert.equal(await m(JSON.parse('{"__proto__":"it"}')), "got it");
  assert.equal(await m("alone" as never), "got alone");
});

// Outputs named like Object's members: a reply that leaves one out is asked again (Python does), never read as the
// member JavaScript would find on Object.prototype (an empty string once); a correct reply is read by every adapter.
test("an output named like an Object member, left out of a reply, is asked again and refused (parse-missing-fields), by every adapter, before and after save and load; a complete reply is read", async () => {
  const types: [string, unknown, unknown][] = [["string", t.string(), "A"], ["optional", t.optional(t.string()), "A"], ["integer", t.integer(), 7]];
  const spell = (adapter: string, fields: [string, unknown][]) => adapter === "json"
    ? `{${fields.map(([k, v]) => `"${k}": ${JSON.stringify(v)}`).join(", ")}}`
    : fields.map(([k, v]) => (adapter === "xml" ? `<${k}>\n${v}\n</${k}>` : `[[ ## ${k} ## ]]\n${v}`)).join("\n") + (adapter === "chat" ? "\n[[ ## completed ## ]]" : "");
  for (const name of ["toString", "valueOf", "constructor", "hasOwnProperty", "normal"]) {
    for (const [kind, type, value] of types) {
      for (const adapter of ["xml", "chat", "json"]) {
        const f = ai("member", { input: { q: t.string() }, outputs: { [name]: type, other: t.string() }, adapter } as never);
        const where = mkdtempSync(join(tmpdir(), "functai-member-out-"));
        save(f, where);
        for (const g of [f, load(where)]) {
          const at = `${name} ${kind} ${adapter}`;
          const router = new FakeRouter([], () => spell(adapter, [["other", "B"]]));
          await assert.rejects(g.using({ router: router as never })("x" as never),
            (e: unknown) => (e as { code?: string }).code === "parse-missing-fields" && (e as Error).message.includes(name), at);
          assert.equal(router.requests.length, 2, `${at}: asked again once`);
          const good = new FakeRouter([], () => spell(adapter, [[name, value], ["other", "B"]]));
          const pred = await g.using({ router: good as never }).predict("x" as never);
          assert.equal(good.requests.length, 1, `${at}: a complete reply is read at once`);
          assert.ok(Object.hasOwn(pred.outputs as object, name), at);
          assert.deepEqual([(pred.outputs as Rec)[name], (pred.outputs as Rec)["other"]], [value, "B"], at);
        }
      }
    }
  }
});

test("a member named __proto__ inside an input's value is sent as given: every adapter, render, a demo, a loaded copy, and the record", async () => {
  const where = folder();
  const answer: Record<string, string> = { xml: "<result>\nok\n</result>", chat: "[[ ## result ## ]]\nok\n\n[[ ## completed ## ]]", json: '{"result": "ok"}' };
  const values = [JSON.parse('{"__proto__":"v","a":"x"}'), JSON.parse('{"__proto__":{"admin":true},"a":"x"}'), [JSON.parse('{"a":"x","__proto__":"p"}')]];
  for (const shape of [t.json(), { type: "object" }, t.list(t.object({ a: t.string() }))]) {
    for (const adapter of ["xml", "chat", "json"]) {
      const router = new FakeRouter([], () => answer[adapter]!);
      const f = ai("nested", { input: { q: shape }, output: t.string(), adapter, router: router as never, logCalls: where } as never);
      const loaded = fromManifest(lmcc.parseJson(JSON.stringify(toManifest(f)))).using({ router: router as never });
      for (const q of values) {
        const before = router.requests.length;
        assert.equal(await f({ q } as never), "ok");
        const request = router.requests.at(-1)!;
        assert.ok(sent(request).includes("__proto__"), `${adapter}: ${sent(request)}`);
        assert.deepEqual(f.render({ q } as never), request);
        assert.equal(await loaded({ q } as never), "ok");
        assert.deepEqual(router.requests.at(-1), request);
        assert.equal(router.requests.length - before, 2);
      }
      const d = ai("nested", { input: { q: shape }, output: t.string(), adapter, router: router as never, demos: [{ inputs: { q: values[0] }, outputs: { result: "E" } }] } as never);
      await d({ q: { a: "y" } } as never);
      assert.equal(sent(router.requests.at(-1)).split("__proto__").length - 1, 1, "the demo's member is sent: " + sent(router.requests.at(-1)));
    }
  }
  const [record] = logged(where);
  assert.ok(Object.hasOwn(record!.inputs.q, "__proto__"));
  assert.equal(record!.inputs.q["__proto__"], "v", "the record keeps what the caller gave");
  const router = ok();
  const text = ai("nested", { input: { q: t.string() }, output: t.string(), router: router as never } as never);
  await text({ q: values[0] } as never);
  assert.ok(sent(router.requests.at(-1)).includes('\\"__proto__\\": \\"v\\"'), sent(router.requests.at(-1)));
});

test("a Standard Schema's JSON Schema keeps a property named __proto__ (normalizing it set the prototype instead)", async () => {
  const payload = z.object({ ["__proto__"]: z.string(), a: z.string() } as never);
  const m = module("m", { input: { payload }, output: t.string() }, () => "ok");
  const shape = m.interface.inputs[0]!.shape as Rec;
  assert.ok(Object.hasOwn(shape["properties"], "__proto__"), JSON.stringify(shape));
  assert.deepEqual(shape["required"], ["__proto__", "a"]);
  await assert.rejects(m({ payload: { a: "y" } } as never), (e: unknown) => e instanceof InterfaceError && e.field === "payload");
  assert.equal(await m({ payload: JSON.parse('{"__proto__":"x","a":"y"}') } as never), "ok");
});

test("a model's object answer missing a property named toString is not taken as fitting its shape", async () => {
  const router = new FakeRouter(['{"result": {}}', '{"result": {"toString": "fine"}}']);
  const f = ai("obj", { input: { q: t.string() }, output: t.object({ toString: t.string() }), adapter: "json", router: router as never } as never);
  assert.deepEqual(await f("x" as never), { toString: "fine" });
  assert.equal(router.requests.length, 2, "the first answer was asked again");
});

// ------------------------------------------------------------------ outcomes that are not errors

test("a call whose code rejects with no reason (Promise.reject(), throw undefined) fails as any other: failed kept and shown, a failed record, the same rejection", async () => {
  for (const reject of [() => Promise.reject(), () => { throw undefined; }, () => Promise.reject(null)]) {
    const where = folder();
    const store = new MemoryStore();
    const seen: StreamEvent[] = [];
    const f = module("nothing", { input: {}, output: t.string(), logCalls: where, journal: { store, mode: "required" }, observers: [(e) => { seen.push(e); }] },
      reject as never);
    const s = f.stream({});
    const events = await all(s.events());
    const why = await s.result.then(() => "resolved", (e: unknown) => e);
    const expected = reject.toString().includes("null") ? null : undefined;
    assert.equal(why, expected);
    assert.deepEqual(events.map((e) => e.kind), ["started", "failed"]);
    assert.ok(await flush());
    assert.deepEqual(seen.map((e) => e.kind), ["started", "failed"]);
    const tree = store.trees()[0]!;
    assert.deepEqual(store.events(tree).map((e) => e["kind"]), ["started", "failed"]);
    assert.ok(store.finished(tree));
    const [record] = logged(where);
    assert.equal(record!.outputs, null);
    assert.equal(record!.error.type, "Error");
    assert.equal(record!.error.message, String(expected));
    // inside a module: the parent's code gets the same rejection
    const inner = module("inner", { input: {}, output: t.string() }, reject as never);
    const outer = module("outer", { input: {}, output: t.string(), logCalls: where }, async () => {
      try {
        await inner({});
        return "resolved";
      } catch (e) {
        return `caught ${String(e)}`;
      }
    });
    assert.equal(await outer({}), `caught ${String(expected)}`);
  }
});

test("a tool that throws undefined is told to the model as an error; with toolErrors: raise, the call fails with it and is recorded", async () => {
  const lookup = tool("lookup", { description: "x", input: { q: t.string() } }, () => { throw undefined; });
  const replies = () => new FakeRouter([{ text: "", calls: [{ id: "c1", name: "lookup", input: { q: "1" } }] }, "<result>\ndone\n</result>"]);
  const router = replies();
  assert.equal(await ai("t", { input: { q: t.string() }, tools: [lookup], router: router as never })("x"), "done");
  assert.ok(sent(router.requests[1]).includes("error: Error: undefined"), sent(router.requests[1]));
  const where = folder();
  const raising = ai("t", { input: { q: t.string() }, tools: [lookup], router: replies() as never, toolErrors: "raise", logCalls: where });
  assert.equal(await raising("x").then(() => "resolved", (e: unknown) => e), undefined);
  assert.equal(logged(where)[0]!.error.message, "undefined");
});

// ------------------------------------------------------------------ inputs as given

/** A valid Standard Schema that changes its value in place, and returns that same object (allowed: nothing promises immutability). */
const inPlace = (async = false): StandardSchemaLike => ({
  "~standard": {
    version: 1, vendor: "probe", jsonSchema: { input: () => ({ type: "object", properties: { x: { type: "string" } } }) },
    validate(v: unknown) {
      if (v === undefined) return { issues: [{ message: "required" }] };
      (v as Rec)["x"] = "TRANSFORMED";
      return async ? Promise.resolve({ value: v }) : { value: v };
    },
  },
}) as never;

test("an AI function records its inputs as given, even when a schema changes them in place: started, record, observers and journal; the model gets the parsed value", async () => {
  for (const async of [false, true]) {
    const where = folder();
    const store = new MemoryStore();
    const seen: StreamEvent[] = [];
    const router = ok();
    const f = ai("mutates", { input: { payload: inPlace(async) }, output: t.string(), logCalls: where, router: router as never, journal: store, observers: [(e) => { seen.push(e); }] });
    const s = f.stream({ payload: { x: "ORIGINAL" } } as never);
    const events = await all(s.events());
    assert.equal(await s, "ok");
    assert.ok(await flush());
    const original = { payload: { x: "ORIGINAL" } };
    assert.deepEqual((events[0] as Rec)["inputs"], original);
    assert.deepEqual((seen[0] as Rec)["inputs"], original);
    assert.deepEqual(store.events(store.trees()[0]!)[0]!["inputs"], original);
    assert.deepEqual(logged(where)[0]!.inputs, original);
    assert.ok(sent(router.requests[0]).includes("TRANSFORMED"), "the model is sent what the schema parsed");
  }
});

test("a schema whose validation is a promise of another realm (a vm context, an iframe) is awaited, not taken as a result", async () => {
  const foreign = runInNewContext("(v) => Promise.resolve(v === undefined ? { issues: [{ message: 'required' }] } : { value: String(v).toUpperCase() })") as (v: unknown) => unknown;
  assert.equal(foreign("a") instanceof Promise, false, "the probe needs a promise this realm does not know");
  const schema = { "~standard": { version: 1, vendor: "probe", validate: foreign, jsonSchema: { input: () => ({ type: "string" }) } } } as never;
  const m = module("m", { input: { word: schema }, output: t.string() }, (i: Rec) => `got ${i["word"]}`);
  assert.equal(await m("hi" as never), "got HI");
  const router = ok();
  const f = ai("f", { input: { word: schema }, output: t.string(), router: router as never });
  assert.equal(await f("hi" as never), "ok");
  assert.ok(sent(router.requests[0]).includes("HI"));
  await assert.rejects(m({} as never), InterfaceError);
});

// ------------------------------------------------------------------ observers beside each other

test("a slow observer loses events alone: a fast one beside it gets every event", async () => {
  await quietly(async (warned) => {
    let slowGot = 0;
    let fastGot = 0;
    let running = true;
    const slow = function slowObserver() {
      slowGot++;
      const until = performance.now() + (running ? 1 : 0);          // slow while the call runs; its backlog then goes quickly
      while (performance.now() < until) { /* a synchronous exporter */ }
    };
    const fast = function fastObserver() { fastGot++; };
    const leaf = module("leaf", { input: {}, output: t.string() }, async () => "x");
    const outer = module("outer", { input: {}, output: t.string(), observers: [slow, fast] }, async () => {
      for (let i = 0; i < 350; i++) {
        await Promise.all(Array.from({ length: 25 }, () => leaf({})));
        await new Promise((r) => setImmediate(r));
      }
      return "ok";
    });
    await outer({});
    running = false;
    assert.ok(await flush({ timeout: 60_000 }));
    const made = 2 + 350 * 25 * 2;
    assert.equal(fastGot, made);
    assert.ok(slowGot < made, `the slow one got ${slowGot} of ${made}: this test needs it to fall behind`);
    assert.ok(warned.some((w) => w.includes("slowObserver") && w.includes("behind")));
    assert.ok(!warned.some((w) => w.includes("fastObserver")), warned.join("\n"));
  });
});

// ------------------------------------------------------------------ a best-effort journal's last events

test("a best-effort journal down for a moment still gets the tree's end: flush sends it again and says so; while it is down flush says false", async () => {
  await quietly(async () => {
    let down = true;
    const store = new Scripted((_n, events, inner) => (down ? Promise.reject(new Error("down")) : Promise.resolve(inner.appendNow(events))));
    const m = module("m", { input: {}, output: t.string(), journal: { store, backoff: 5 } }, () => "ok");
    assert.equal(await m({}), "ok");
    assert.equal(await flush({ timeout: 2000 }), false, "the journal holds events it did not confirm");
    down = false;
    assert.equal(await flush({ timeout: 2000 }), true);
    const tree = store.inner.trees()[0]!;
    assert.deepEqual(store.inner.events(tree).map((e) => e["kind"]), ["started", "done"]);

    // without flush: the writer tries again on its own a second later
    down = true;
    const g = module("g", { input: {}, output: t.string(), journal: { store, backoff: 5, retries: 0 } }, () => "ok");
    assert.equal(await g({}), "ok");
    await delay(50);
    down = false;
    await delay(1200);
    const trees = store.inner.trees();
    assert.equal(trees.length, 2);
    assert.ok(store.inner.finished(trees[1]!), "the end was sent again later, with no next event to carry it");
  });
});

test("a journal that stays down: the writer tries again on its own for as long as the process lives (1, 2, 4, … 60 s apart), gives nothing up for time, and keeps it all once the store is back", async (ctx) => {
  ctx.mock.timers.enable({ apis: ["setTimeout"] });
  await quietly(async (warned) => {
    const store = new Scripted(() => Promise.reject(new Error("down")));
    const m = module("m", { input: {}, output: t.string(), journal: { store, retries: 0, backoff: 0 } }, () => "ok");
    assert.equal(await m({}), "ok");
    const settle = () => new Promise((r) => setImmediate(r));
    for (let i = 0; i < 5; i++) await settle();
    const first = store.sends;                                    // its started, and its done (each tried once, as it came)
    const at: number[] = [];
    for (let second = 1; second <= 400; second++) {
      ctx.mock.timers.tick(1000);
      for (let i = 0; i < 3; i++) await settle();
      while (at.length < store.sends - first) at.push(second);
    }
    assert.deepEqual(at, [1, 3, 7, 15, 31, 63, 123, 183, 243, 303, 363], "later rounds 1, 2, 4, 8, 16, 32 s apart, then every 60 s");
    assert.ok(!warned.some((w) => w.includes("given up")), warned.join("\n"));
    store.healed = true;
    ctx.mock.timers.tick(60_000);
    for (let i = 0; i < 5; i++) await settle();
    const tree = store.inner.trees()[0]!;
    assert.deepEqual(store.inner.events(tree).map((e) => e["kind"]), ["started", "done"], "the next later round kept it all");
    assert.ok(await flush());
  });
});

test("a quiet tree through a store that fails fast for 36 s (one long model call): nothing is given up, its end is kept, and flush says true", async (ctx) => {
  ctx.mock.timers.enable({ apis: ["setTimeout"] });
  await quietly(async (warned) => {
    let down = true;
    const store = new Scripted((_n, events, inner) => (down ? Promise.reject(new Error("ECONNREFUSED")) : Promise.resolve(inner.appendNow(events))));
    let answer!: (v: string) => void;
    const m = module("m", { input: {}, output: t.string(), journal: { store, retries: 0, backoff: 0 } }, () => new Promise<string>((r) => { answer = r; }));
    const call = m({});
    const settle = () => new Promise((r) => setImmediate(r));
    for (let second = 1; second <= 40; second++) {
      if (second === 36) down = false;
      ctx.mock.timers.tick(1000);
      for (let i = 0; i < 3; i++) await settle();
    }
    answer("ok");                                                 // the model answers after 40 s
    assert.equal(await call, "ok");
    for (let i = 0; i < 5; i++) await settle();
    const tree = store.inner.trees()[0]!;
    assert.deepEqual(store.inner.events(tree).map((e) => e["kind"]), ["started", "done"]);
    assert.ok(!warned.some((w) => w.includes("given up")), warned.join("\n"));
    assert.ok(await flush());
  });
});

test("flush sends again a writer whose round was already under way when flush began, and failed: it says true once all is kept", async () => {
  await quietly(async () => {
    let mode: "down" | "slow" | "up" = "down";
    const store = new Scripted((_n, events, inner) => {
      if (mode === "down") return Promise.reject(new Error("down"));
      if (mode === "slow") {
        mode = "up";
        return delay(30).then(() => { throw new Error("down"); });
      }
      return Promise.resolve(inner.appendNow(events));
    });
    let answer!: (v: string) => void;
    const m = module("m", { input: {}, output: t.string(), journal: { store, retries: 0, backoff: 0 } }, () => new Promise<string>((r) => { answer = r; }));
    const call = m({});
    await delay(20);                                              // started: its round failed, and the writer holds it
    mode = "slow";
    answer("ok");
    assert.equal(await call, "ok");                               // done: a round begins now, and fails 30 ms later
    assert.equal(await flush({ timeout: 2000 }), true);
    const tree = store.inner.trees()[0]!;
    assert.deepEqual(store.inner.events(tree).map((e) => e["kind"]), ["started", "done"]);
  });
});

test("writers holding more unconfirmed events than the limit give up the oldest log first, warn once, and the next flush says false (once)", async () => {
  const before = holdAtMost(5);
  try {
    await quietly(async (warned) => {
      const store = new Scripted(() => Promise.reject(new Error("down")));
      const m = module("m", { input: {}, output: t.string(), journal: { store, retries: 0, backoff: 0 } }, () => "ok");
      for (let i = 0; i < 3; i++) assert.equal(await m({}), "ok");         // three trees, two events each: six held
      assert.equal(warned.filter((w) => w.includes("given up")).length, 1, warned.join("\n"));
      store.healed = true;
      assert.equal(await flush(), false, "a log was given up since the last flush");
      assert.equal(await flush(), true, "said once");
      const trees = store.inner.trees();
      assert.equal(trees.length, 2, "the oldest tree's log was given up; the two after it are kept");
      for (const tree of trees) assert.deepEqual(store.inner.events(tree).map((e) => e["kind"]), ["started", "done"]);
    });
  } finally {
    holdAtMost(before);
  }
});

// Astra, round 4 (S1): giving a log up dropped its events from the count, but the round sending them went on: its
// append was waited for (for ever, with timeout: Infinity), its signal never aborted, and it was sent again after its pause.
test("a writer that gives up its log ends the round under way: the append's signal aborts, its wait ends, nothing is sent again, and flush does not wait for it", async () => {
  const before = holdAtMost(1);
  try {
    await quietly(async (warned) => {
      const answers: Array<(a: AppendAnswer) => void> = [];
      const store = new Scripted(() => new Promise<AppendAnswer>((resolve) => { answers.push(resolve); }));
      const m = module("m", { input: {}, output: t.string(), journal: { store, timeout: Infinity, retries: 0, backoff: 0 } }, () => "ok");
      for (let i = 0; i < 4; i++) assert.equal(await m({}), "ok");
      await delay(20);
      assert.ok(warned.some((w) => w.includes("given up") && w.includes("aborted")), warned.join("\n"));
      const open = store.signals.filter((s) => !s.aborted);
      assert.ok(store.signals.length >= 2 && open.length <= 1, `every append of a log given up is aborted (${open.length} of ${store.signals.length} open)`);
      const sends = store.sends;
      for (const answer of answers) answer("kept");                   // late answers change nothing
      const started = performance.now();
      assert.equal(await flush({ timeout: 5000 }), false, "logs were given up");
      assert.ok(performance.now() - started < 2000, "flush waited for no writer that gave up");
      assert.equal(await flush({ timeout: 5000 }), true);
      assert.equal(store.sends, sends, "nothing sent again");
    });
  } finally {
    holdAtMost(before);
  }
  // given up during its pause between two sends of a round: no send follows
  const again = holdAtMost(1);
  try {
    await quietly(async () => {
      const store = new Scripted(() => Promise.reject(new Error("down")));
      const m = module("m", { input: {}, output: t.string(), journal: { store, timeout: 100, retries: 3, backoff: 40 } }, () => delay(10).then(() => "ok"));
      await m({});            // its started is sent and fails, and the round pauses 40 ms; its done comes: over the limit, the log is given up
      await delay(300);       // the round would have sent it three times more by now (at 40, 120 and 280 ms)
      assert.equal(store.sends, 1, "the round stopped in its pause when the log was given up");
      assert.equal(await flush({ timeout: 2000 }), false);
      assert.equal(await flush({ timeout: 2000 }), true);
      const sends = store.sends;
      await delay(200);
      assert.equal(store.sends, sends, "no later round");
    });
  } finally {
    holdAtMost(again);
  }
});

test("journal timeouts and backoffs past a timer's limit refuse where they are set (setTimeout would fire at once); settle can be stopped", async () => {
  const store = new MemoryStore();
  for (const bad of [{ timeout: 3_000_000_000 }, { backoff: 3_000_000_000 }, { timeout: 0 }, { backoff: -1 }]) {
    assert.throws(() => module("m", { input: {}, output: t.string(), journal: { store, ...bad } }, () => ""), TypeError, JSON.stringify(bad));
  }
  module("m", { input: {}, output: t.string(), journal: { store, timeout: Infinity, backoff: 2_147_483_647 } }, () => "");
  await quietly(async () => {
    const dead = new Scripted((_n, events, inner) => events.some((e) => e["kind"] === "done") ? never() : Promise.resolve(inner.appendNow(events)));
    const hung: EventStore = { append: (es, o) => dead.append(es as Rec[], o), read: () => never() };
    const f = module("m", { input: {}, output: t.string(), journal: { store: hung, mode: "required", timeout: 50 } }, () => "ok");
    const err = await f({}).then(() => null, (e: unknown) => e) as JournalError;
    assert.equal(err.code, "journal-end");
    await assert.rejects(err.settle({ signal: AbortSignal.timeout(50) }), (e: unknown) => (e as Error).name === "TimeoutError");
    await heal(dead);
  });
});

test("settle stops for an abort the store's read makes itself, at once or later, and leaves no listener behind", async () => {
  for (const synchronous of [true, false]) {
    const controller = new AbortController();
    const reason = new Error("transport closed");
    const source = {
      read: () => {
        if (synchronous) controller.abort(reason);
        else queueMicrotask(() => controller.abort(reason));
        return never();
      },
    };
    const outcome = await Promise.race([
      settleLog(source as never, "tree", { writer: 1, seq: 2 }, { signal: controller.signal }).then(() => "resolved", (e: unknown) => e),
      delay(500).then(() => "still waiting"),
    ]);
    assert.equal(outcome, reason, `synchronous: ${synchronous}`);
    assert.equal(getEventListeners(controller.signal, "abort").length, 0);
  }
  const controller = new AbortController();
  const throwing = { read: () => { throw new Error("no read"); } };
  await assert.rejects(settleLog(throwing as never, "tree", { writer: 1, seq: 2 }, { signal: controller.signal }), /no read/);
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

test("a builder's extra keys add to its shape and never replace it", () => {
  assert.throws(() => t.list(t.string(), { items: { type: "integer" } }), TypeError);
  assert.throws(() => t.string({ type: "integer" } as never), TypeError);
  assert.throws(() => t.object({ a: t.string() }, { required: [] }), TypeError);
  assert.throws(() => t.record(t.integer(), { additionalProperties: false }), TypeError);
  assert.deepEqual(t.list(t.string(), { description: "one per line", minItems: 1 }), { type: "array", items: { type: "string" }, description: "one per line", minItems: 1 });
});

// ------------------------------------------------------------------ formats

test("replaying or recovering stops at an event of a format this reader does not know", async () => {
  const started = { functai_event: 2, kind: "started", tree: "t", writer: 1, seq: 1, after: null, call: "t" };
  const future = { ...started, functai_event: 3, kind: "done", seq: 2, after: { writer: 1, seq: 1 }, value: "future" };
  const r = replay([started, future]);
  assert.equal(r.finished, false);
  assert.equal(r.stopped, true);
  const reader = new Follower({ form: "kept" });
  await reader.recover("t", { read: async () => ({ events: [started, future] }) as never });
  assert.equal(reader.stopped, true);
  assert.equal(reader.state("t").finished, false);
  assert.equal(reader.held("t").length, 1);
  await reader.recover("t", { read: async () => ({ events: [started] }) as never });     // a source it can read: it follows again
  assert.equal(reader.stopped, false);
  reader.forget("t");
  assert.deepEqual(reader.held("t"), []);
});

// ------------------------------------------------------------------ outcomes

test("a journal-end error holds what that caller would have got: the answer for fn(x) and a stream, the Prediction for predict", async () => {
  await quietly(async () => {
    const made: Scripted[] = [];
    const endless = () => {
      const s = new Scripted((_n, events, inner) => events.some((e) => e["kind"] === "done") ? never() : Promise.resolve(inner.appendNow(events)));
      made.push(s);
      return s;
    };
    const f = (store: EventStore) => ai("triage", { input: { q: t.string() }, outputs: { summary: t.string(), result: t.string() },
      router: new FakeRouter([], () => "<summary>\nshort\n</summary>\n<result>\nno\n</result>") as never, journal: { store, mode: "required", timeout: 50 } });
    const plain = await f(endless())("x").then(() => null, (e: unknown) => e) as JournalError;
    assert.deepEqual(plain.outcome, { done: "no" });
    const predicted = await f(endless()).predict("x").then(() => null, (e: unknown) => e) as JournalError;
    const p = (predicted.outcome as { done: Prediction }).done;
    assert.ok(p instanceof Prediction);
    assert.deepEqual(p.outputs, { summary: "short", result: "no" });
    const s = f(endless()).stream("x");
    const [answer, prediction] = await Promise.all([s.result.then(() => null, (e: unknown) => e), s.prediction.then(() => null, (e: unknown) => e)]) as JournalError[];
    assert.deepEqual(answer!.outcome, { done: "no" });
    assert.ok((prediction!.outcome as { done: unknown }).done instanceof Prediction);
    await heal(...made);
  });
});

test("a re-ask's exchange has the request hash of the rendered request it came from, as every exchange from it", async () => {
  const where = folder();
  await mood(new FakeRouter(["not tags", "<result>\nhappy\n</result>"]), { logCalls: where })("Lovely");
  const [record] = logged(where);
  assert.equal(record!.exchanges.length, 2);
  assert.equal(record!.exchanges[0].request_hash, record!.exchanges[1].request_hash);
  assert.notDeepEqual(record!.exchanges[0].request, record!.exchanges[1].request);
});
