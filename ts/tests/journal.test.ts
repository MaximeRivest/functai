/**
 * The contract's journal-* and receivers-* cases (streaming.md, "Keeping a
 * log while it is written"), run through the runtime's own call lifecycle:
 * a call's events, a required journal's barriers and end, JournalError, the
 * record; and which observers and journal a tree gets from real settings
 * layers (a program's own, blocks, configure).
 */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  configure, JournalError, MemoryStore, module, t, withSettings, type AppendAnswer, type EventStore, type Observer, type Position,
  type ReadAnswer, type Settings, type StreamEvent,
} from "../src/index.ts";
import { errorJson } from "../src/calllog.ts";
import { runCall } from "../src/program.ts";
import { effective } from "../src/settings.ts";
import type { Node, TreeLog } from "../src/log.ts";
import { cases, pos, type Rec } from "./cases.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

/** A journal whose transport follows a case's script, and which records every send (cases/README.md, "journal"). */
class ScriptedStore implements EventStore {
  readonly inner = new MemoryStore();
  readonly trace: Rec[] = [];
  private i = 0;
  private readonly script: readonly string[];
  constructor(script: readonly string[]) {
    this.script = script;
  }

  private next(): string {
    return this.script[this.i++] ?? "ok";
  }

  /** Another writer claims the log ("claimed"), and ends it ("ended"). */
  private other(word: string, tree: string, fn: string): void {
    const claim = this.inner.claimNow(tree);
    this.trace.push({ other: word, writer: "writer" in claim ? claim.writer : null });
    if (word === "ended" && "writer" in claim) {
      const answer = this.inner.appendNow([{
        functai_event: 2, kind: "failed", tree, writer: claim.writer, seq: claim.after.seq + 1, after: claim.after,
        at: "2026-09-28T10:00:50.000000Z", call: tree, function: fn, error: { type: "Cancelled" },
      }]);
      assert.equal(answer, "kept");
    }
  }

  async append(events: readonly Rec[]): Promise<AppendAnswer> {
    const first = events[0]!;
    let word = this.next();
    while (word === "claimed" || word === "ended") {
      this.other(word, first["tree"], first["function"]);
      word = this.next();
    }
    const seq = first["seq"];
    if (word === "down") {
      this.trace.push({ seq, transport: word, answer: null });
      throw new Error("the journal is down");
    }
    if (word === "conflict") {
      this.trace.push({ seq, transport: word, answer: "event-conflict" });
      return { refuses: "event-conflict", event: pos(first) };
    }
    const answer = this.inner.appendNow(events);
    this.trace.push({ seq, transport: word, answer: word === "lost" ? null : typeof answer === "string" ? answer : answer.refuses });
    if (word === "lost") throw new Error("the answer was lost");
    return answer;
  }

  async read(tree: string, after: Position | null): Promise<ReadAnswer> {
    return this.inner.read(tree, after);
  }
}

const withoutMessage = (e: Rec | null) => (e ? Object.fromEntries(Object.entries(e).filter(([k]) => k !== "message")) : null);
const envelope = new Set(["functai_event", "kind", "tree", "writer", "seq", "after", "at", "call", "function"]);

// Each case runs twice: with `batch: 1` (every send compared with the case's trace) and with the default batch setting (the
// case's writer confirms each event before the next is made, so no batch forms: failure-paths.test.ts drives real batches).
for (const [name, c] of cases("events", "journal-")) for (const batch of [1, undefined]) {
  test(`events case ${name}${batch ? "" : " (default batch setting)"}`, async () => {
    const folder = mkdtempSync(join(tmpdir(), "functai-journal-"));
    const store = new ScriptedStore(c.script);
    const [started, ...rest] = c.events as Rec[];
    const end = rest.pop()!;
    const shown: StreamEvent[] = [];
    const made: StreamEvent[] = [];
    let log: TreeLog | null = null;
    const watcher = {
      signal: new AbortController().signal,
      receive: (e: StreamEvent, node: Node) => {
        shown.push(e);
        if (log) return;
        log = node.log;                                   // its started, made before anything runs
        made.push(e);
        const make = log.make.bind(log);                  // the events the writer makes after it (the end included, shown or not)
        log.make = (...args) => {
          const m = make(...args);
          made.push(m.event);
          return m;
        };
      },
    };
    const own: Settings = { journal: { store, mode: c.mode, retries: c.retries, ...(batch ? { batch } : {}) }, logCalls: folder };
    let caller: Rec;
    let error: unknown = null;
    try {
      const value = await runCall({
        program: () => started["program"], fields: { inputs: Object.keys(started["inputs"]), outputs: ["result"], added: [] },
        own, options: {}, settings: effective(own), stream: watcher, inputs: started["inputs"],
        body: async (call) => {
          const log = call.node.log;
          for (const e of rest) {
            await log.journalWriter!.idle();                // the case's writer confirms each event before the next is made
            const fields = Object.fromEntries(Object.entries(e).filter(([k]) => !envelope.has(k)));
            const m = call.event(e["kind"], fields);
            await log.journalWriter!.idle();
            if (e["kind"] === "tool_call") await log.barrier(m);
          }
          if (end["kind"] === "failed") throw Object.assign(new Error(end["error"]["message"]), { name: end["error"]["type"] });
          return { value: end["value"], outputs: { result: end["value"] } };
        },
      });
      caller = { returns: value };
    } catch (err) {
      error = err;
      caller = { raises: err instanceof JournalError && err.code === "journal-end"
        ? { type: "JournalError", code: err.code, journal: err.journal, event: err.event,
          outcome: "done" in err.outcome! ? err.outcome : { failed: withoutMessage(errorJson(err.outcome!.failed)) } }
        : withoutMessage(errorJson(err)) };
    }
    await (log as TreeLog | null)?.journalWriter!.idle();
    const tree = (log as TreeLog | null)?.tree ?? "";
    const ids = (x: unknown) => JSON.parse(JSON.stringify(x).replaceAll(tree, started["tree"]));
    const at = (e: Rec) => Object.fromEntries(Object.entries(e).filter(([k]) => k !== "at"));
    assert.deepEqual(ids(made.map((e) => at(e as Rec))), c.expect.log.map(at), "log");
    if (batch) assert.deepEqual(store.trace, c.expect.trace, "trace");  // one event per send, as the case's writer sends them
    assert.deepEqual(shown.map((e) => e.seq), c.expect.shown, "shown");
    assert.deepEqual({ events: store.inner.events(tree).map(pos), finished: store.inner.finished(tree) }, c.expect.kept, "kept");
    assert.deepEqual(ids(caller), c.expect.caller, "caller");
    const day = readdirSync(folder)[0]!;
    const record = JSON.parse(readFileSync(join(folder, day, readdirSync(join(folder, day))[0]!), "utf8").trim());
    assert.deepEqual({ error: withoutMessage(record.error), ...(record.journal ? { journal: record.journal } : {}) }, c.expect.record, "record");
    if (c.expect.settled) assert.equal(await (error as JournalError).settle(), c.expect.settled, "settled");
  });
}


/** A store that answers each append a moment later: a best-effort writer's call has ended before its end is kept. */
class SlowStore extends MemoryStore {
  override async append(events: readonly Rec[]): Promise<AppendAnswer> {
    await new Promise((r) => setTimeout(r, 1));
    return this.appendNow(events);
  }
}

for (const [name, c] of cases("events", "receivers-")) {
  test(`events case ${name}`, async () => {
    for (const scenario of c.scenarios) {
      const got: string[][] = [];                         // for each event, the observers given it, in order
      const observer = (n: string): Observer => Object.defineProperty((e: StreamEvent) => {
        (got[e.seq - 1] ??= []).push(n);
      }, "name", { value: n });
      const observers = new Map<string, Observer>();
      const stores = new Map<string, SlowStore>();
      const layerSettings = (layer: Rec): Settings => ({
        ...(layer.observers ? { observers: layer.observers.map((n: string) => observers.get(n) ?? observers.set(n, observer(n)).get(n)!) } : {}),
        ...("journal" in layer ? {
          journal: layer.journal === null ? null : {
            store: stores.get(layer.journal.name) ?? stores.set(layer.journal.name, new SlowStore()).get(layer.journal.name)!,
            mode: layer.journal.mode,
          },
        } : {}),
      });
      const layers = scenario.layers as Rec[];
      const own = layers.find((l) => l.where === "own");
      const blocks = layers.filter((l) => l.where === "block").reverse();      // outermost first
      const conf = layers.find((l) => l.where === "configure");
      const program = module("program", { input: {}, output: t.string(), ...(own ? layerSettings(own) : {}) }, () => "ok");
      configure(conf ? layerSettings(conf) : {});
      let outcome: unknown;
      try {
        const run = blocks.reduceRight<() => Promise<unknown>>((inner, b) => () => withSettings(layerSettings(b), inner), () => program({}));
        outcome = await run().then(() => "ok", (err: unknown) => err);
      } finally {
        configure({ observers: undefined, journal: undefined });
      }
      // a required journal's end is kept before the call returns; a best-effort one's a moment later
      const endedFirst = new Set([...stores].filter(([, s]) => s.trees().some((tree) => s.finished(tree))).map(([n]) => n));
      await new Promise((r) => setTimeout(r, 30));
      const kept = [...stores].filter(([, s]) => s.trees().length);
      assert.ok(kept.length <= 1, "one journal per tree");
      const journal = kept.length ? { name: kept[0]![0], mode: endedFirst.has(kept[0]![0]) ? "required" : "best-effort" } : null;
      const result = { observers: got[0] ?? [], journal };
      if (outcome instanceof JournalError) {
        if (got.length) assert.equal(got.length, 2, "a refused tree's log: its started and its failed");
        if (kept.length) assert.deepEqual(kept[0]![1].events(kept[0]![1].trees()[0]!).map((e) => e["kind"]), ["started", "failed"]);
        assert.deepEqual({ refuses: outcome.code, ...result }, scenario.expect, JSON.stringify(scenario.layers));
      } else {
        assert.equal(outcome, "ok");
        assert.deepEqual(result, scenario.expect, JSON.stringify(scenario.layers));
      }
    }
  });
}
