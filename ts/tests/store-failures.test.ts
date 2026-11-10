/**
 * A conversation's store that fails, at every point it is used: whichever
 * append or read fails, the turn ends for whoever awaits it (its answer, or
 * an error), and no rejection is left unhandled (in Node, one stops the
 * process).
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { ai, later, MemoryConversations, module, remember, t, tool } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

type Rec = Record<string, unknown>;

/** A store whose n-th append, or n-th read, fails; `slow`: it answers a moment later, as a database does. */
class Flaky extends MemoryConversations {
  appends = 0;
  reads = 0;
  readonly kinds: string[] = [];
  private readonly failAppend: number;
  private readonly failRead: number;
  private readonly slow: boolean;

  constructor(failAppend: number, failRead: number, slow: boolean) {
    super();
    this.failAppend = failAppend;
    this.failRead = failRead;
    this.slow = slow;
  }

  override append(conversation: string, records: readonly Rec[], opts: { expect?: number } = {}): never {
    const n = ++this.appends;
    this.kinds.push(records.map((r) => `${String(r["kind"])}${r["state"] ? `:${String(r["state"])}` : ""}`).join("+"));
    return (async () => {
      if (this.slow) await new Promise((r) => setTimeout(r, 1));
      if (n === this.failAppend) throw new Error(`the store is down (append ${n})`);
      return super.append(conversation, records, opts);
    })() as never;
  }

  override read(conversation: string, after = 0): never {
    const n = ++this.reads;
    return (async () => {
      if (n === this.failRead) throw new Error(`the store is down (read ${n})`);
      return super.read(conversation, after);
    })() as never;
  }
}

const send = () => tool("send", { input: { to: t.string() }, effects: "changes" }, () => "sent");
const asksToSend = () => new FakeRouter([{ calls: [{ id: "c1", name: "send", input: { to: "bob" } }] }, "<result>\ndone\n</result>"]);

const scenarios: Record<string, (store: Flaky) => Promise<unknown>> = {
  "a tool that reads, then one that changes things": (store) => {
    const router = new FakeRouter([{ calls: [{ id: "c1", name: "look", input: { q: "a" } }] }, { calls: [{ id: "c2", name: "send", input: { to: "bob" } }] },
      "<result>\ndone\n</result>"]);
    const look = tool("look", { input: { q: t.string() }, effects: "reads" }, () => "seen");
    return ai("agent", { input: { task: t.string() }, output: t.string(), tools: [look, send()], router: router as never, lm: "gpt-4.1-mini" })
      .conversation("c", { store })("go");
  },
  "a person is asked, and answers": async (store) => {
    const f = ai("agent", { input: { task: t.string() }, output: t.string(), tools: [send()], router: asksToSend() as never, lm: "gpt-4.1-mini", approve: "changes" });
    try {
      return await f.conversation("c", { store })("go");
    } catch (e) {
      const turn = (e as { turn?: { approve?(): Promise<unknown> } }).turn;
      if (turn?.approve) return turn.approve();
      throw e;
    }
  },
  "approve is a function": (store) =>
    ai("agent", { input: { task: t.string() }, output: t.string(), tools: [send()], router: asksToSend() as never, lm: "gpt-4.1-mini", approve: () => true })
      .conversation("c", { store })("go"),
  "a result that comes later, delivered": async (store) => {
    const router = new FakeRouter([{ calls: [{ id: "c1", name: "start", input: { url: "a" } }] }, "<result>\ndone\n</result>"]);
    const start = tool("start", { input: { url: t.string() }, effects: "changes" }, () => later("job:1"));
    const f = ai("agent", { input: { task: t.string() }, output: t.string(), tools: [start], router: router as never, lm: "gpt-4.1-mini" });
    await f.conversation("c", { store })("go").catch(() => undefined);                // it waits
    return f.conversation("c", { store }).deliver("job:1", "3 pages");
  },
  "a module whose helper the conversation remembers, streamed": async (store) => {
    const router = new FakeRouter([], () => "<result>\nok\n</result>");
    const helper = ai("helper", { input: { q: t.string() }, output: t.string(), router: router as never, lm: "gpt-4.1-mini" });
    const pipeline = module("pipeline", { input: { q: t.string() }, output: t.string(), uses: [helper] }, async ({ q }) => `${await helper({ q })}!`);
    const s = pipeline.conversation("c", { store, remembers: [[helper, remember("conversation")]] }).stream({ q: "one" });
    for await (const piece of s) void piece;
    return s.result;
  },
};

/** How a promise ends, within `ms`. */
const ending = (p: Promise<unknown>, ms: number): Promise<string> => Promise.race([
  p.then(() => "an answer", (e: unknown) => `an error: ${String((e as Error)?.message ?? e)}`),
  new Promise<string>((r) => { const timer = setTimeout(() => r("NEVER ENDED"), ms); (timer as { unref?: () => void }).unref?.(); }),
]);

for (const [name, scenario] of Object.entries(scenarios)) {
  test(`a store that fails at any point: the turn ends, and nothing is left unhandled (${name})`, async () => {
    const unhandled: string[] = [];
    const note = (e: unknown) => { unhandled.push(String((e as Error)?.message ?? e)); };
    const warn = console.warn;
    console.warn = () => undefined;                                           // each failure kept as a convenience is said once: not this test's output
    process.on("unhandledRejection", note);
    try {
      for (const slow of [false, true]) {
        const whole = new Flaky(0, 0, slow);
        assert.equal(await ending(scenario(whole), 5000), "an answer");       // nothing fails: it answers
        assert.ok(whole.appends >= 3 && whole.reads >= 1);
        for (const [what, total] of [["append", whole.appends], ["read", whole.reads]] as const) {
          for (let n = 1; n <= total; n++) {
            const store = what === "append" ? new Flaky(n, 0, slow) : new Flaky(0, n, slow);
            const how = await ending(scenario(store), 5000);
            await new Promise((r) => setTimeout(r, 5));                       // a rejection nobody awaited surfaces by now
            const where = `${what} ${n}${what === "append" ? ` (${whole.kinds[n - 1]})` : ""}${slow ? ", a slow store" : ""}`;
            assert.notEqual(how, "NEVER ENDED", where);
            assert.deepEqual(unhandled, [], where);
          }
        }
      }
    } finally {
      process.off("unhandledRejection", note);
      console.warn = warn;
    }
  });
}
