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
import { MessageChannel } from "node:worker_threads";
import { test } from "node:test";
import { RateLimitError, responseToEvents, streamDelta, type Request, type StreamEvent as LmEvent } from "@lm15/lm15";
import * as z from "zod";
import {
  ai, Cancelled, checkInterface, configure, flush, Follower, fromManifest, InterfaceError, JournalError, MemoryStore, module, Prediction, replay,
  SettingError, t, toManifest, tool, withSettings, type AppendAnswer, type StandardSchemaLike, type EventStore, type Position, type ReadAnswer, type StreamEvent,
} from "../src/index.ts";
import { passes } from "../src/schema.ts";
import { jsonForm } from "../src/values.ts";
import { FakeRouter } from "./fake.ts";

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
  append(events: readonly Rec[], opts?: { signal?: AbortSignal }): Promise<AppendAnswer> {
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

// ------------------------------------------------------------------ journals: liveness

test("a store that throws before returning a promise never spins the process: the call settles and timers fire (own process, time-limited)", () => {
  const run = spawnSync(process.execPath, ["--conditions=functai-source", "--conditions=lmcc-source", join(here, "fixtures", "sync-throwing-store.ts")],
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
    const endless = () => new Scripted((_n, events, inner) =>
      events.some((e) => e["kind"] === "done") ? never() : Promise.resolve(inner.appendNow(events)));
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
