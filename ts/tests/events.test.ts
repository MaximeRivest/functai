/**
 * The contract's events/ cases (streaming.md) that are data: replay-*,
 * follow-*, kept-*, store-*. journal-* and receivers-* run a call, in
 * journal.test.ts.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { Follower, keptLog, MemoryStore, replay, resume, type EventSource, type Position, type ReadAnswer } from "../src/index.ts";
import { cases, pos, type Rec } from "./cases.ts";

const positions = (a: ReadAnswer | Rec): Rec => ("events" in a ? { events: (a.events as Rec[]).map(pos) } : a);

for (const [name, c] of cases("events", "replay-")) {
  test(`events case ${name}`, () => {
    assert.deepEqual(replay(c.events), c.expect);
    for (const r of c.resume) assert.deepEqual(positions(resume(c.events, r.after)), r.expect, JSON.stringify(r.after));
  });
}

for (const [name, c] of cases("events", "follow-")) {
  test(`events case ${name}`, async () => {
    const reader = new Follower({ form: c.recover?.reader ?? "live" });
    const results: string[] = [];
    for (const e of c.received) {
      const r = reader.receive(e);
      results.push(r);
      if (r === "unknown-format") break;
    }
    assert.deepEqual(results, c.expect.results);
    assert.deepEqual(reader.state(), c.expect.state);
    if (!c.recover) return;
    const tree = c.received[0].tree as string;
    const source: EventSource = {
      ...(c.recover.from === "store" ? {} : { writer: c.recover.from as number }),
      read: (_tree: string, after: Position | null) => resume(c.recover.source, after) as ReadAnswer,
    };
    const reads = await reader.recover(tree, source);
    assert.deepEqual({ reads: reads.map((r) => ({ after: r.after, expect: positions(r.answer) })), state: reader.state(tree) }, c.expect.recover);
  });
}

for (const [name, c] of cases("events", "kept-")) {
  test(`events case ${name}`, () => {
    assert.deepEqual(keptLog(c.events, c.kept), c.expect.events);
  });
}

for (const [name, c] of cases("events", "store-")) {
  test(`events case ${name}`, () => {
    const store = new MemoryStore();
    for (const step of c.steps) {
      if ("append" in step) {
        const answer = store.appendNow([step.append]);
        assert.deepEqual(typeof answer === "string" ? answer : answer.refuses, step.expect, JSON.stringify(step.append).slice(0, 200));
      } else if ("batch" in step) {
        assert.deepEqual(store.appendNow(step.batch), step.expect);
      } else {
        assert.deepEqual(store.claimNow(step.claim), step.expect);
      }
    }
    for (const r of c.reads) assert.deepEqual(positions(store.readNow(r.tree, r.after)), r.expect);
    const logs = Object.fromEntries(store.trees().map((tree) => [tree, {
      events: store.events(tree).map(pos), writer: store.writerOf(tree), finished: store.finished(tree),
    }]));
    assert.deepEqual({ logs }, c.expect);
  });
}
