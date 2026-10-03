/**
 * The contract's cases for stages 1.2 to 5, plugins and baking
 * (../../contract/cases: replies, conversations, tools, views, context,
 * plugins, baked), run through the TypeScript implementation: the same cases
 * Python, R and Julia pass.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { Request } from "@lm15/lm15";
import * as lmcc from "lmcc";
import { ai, configure, MemoryConversations, Plugin, t } from "../src/index.ts";
import { replyKey } from "../src/replies.ts";
import { clock, internals } from "../src/conversations.ts";
import { asks, denial } from "../src/tools.ts";
import { outside } from "../src/views.ts";
import { earlierOf, SawUnknown } from "../src/saw.ts";
import { Applied, beforeCall, inOrder, toolCall, toolResult } from "../src/plugins.ts";
import { bakeEntry, studentMessages } from "../src/bake.ts";
import { Call } from "../src/calllog.ts";
import { cases, tsFunction, type Rec } from "./cases.ts";
import { FakeRouter } from "./fake.ts";

const unix = (text: string) => Date.parse(text) / 1000;

for (const [name, c] of cases("replies")) {
  test(`replies case ${name}`, () => {
    assert.equal(replyKey(Request.fromJSON(c.request), c.replicate), c.expect.key);
  });
}

test("the reply key reads a number lm15 keeps as written as the number it is (0.0 is 0, as Python's canonical JSON writes it)", () => {
  const plain = Request.fromJSON({ model: "m", messages: [{ role: "user", parts: [{ type: "text", text: "x" }] }], config: { temperature: 0 } });
  const raw = Request.fromJSON(lmcc.parseJson('{"model": "m", "messages": [{"role": "user", "parts": [{"type": "text", "text": "x"}]}], "config": {"temperature": 0.0}}') as never);
  assert.equal(replyKey(raw), replyKey(plain));
});

for (const [name, c] of cases("conversations")) {
  test(`conversations case ${name}`, async () => {
    const before = clock.now;
    clock.now = () => unix(c.now);
    try {
      if (c.kind === "state") {
        const log = new internals.ConvLog().apply(c.records);
        const e = c.expect;
        assert.deepEqual(Object.fromEntries([...log.turns].map(([id, st]) => [id, st.state()])), e.states);
        assert.equal(log.head, e.head);
        const head = log.head === null ? null : log.turns.get(log.head)!;
        const state = head?.state() ?? null;
        if (e.next.refuses) assert.equal(state, "waiting");
        else if (e.next.waits) assert.ok(state === "running" && log.head === e.next.waits);
        else assert.equal(log.doneOn(log.head), e.next.parent);
        const waiting = Object.fromEntries([...log.turns].filter(([, st]) => st.unanswered().length).map(([id, st]) => [id, st.unanswered().map((a) => a["invocation"])]));
        assert.deepEqual(waiting, e.waiting);
        const unfinished = Object.fromEntries([...log.turns].filter(([, st]) => st.unfinished().length).map(([id, st]) => [id, st.unfinished().map((x) => x["invocation"])]));
        assert.deepEqual(unfinished, e.unfinished);
        return;
      }
      const tutor = ai("tutor", { description: "Tutor.", input: { message: t.string() }, output: t.string(), router: new FakeRouter() as never, lm: "gpt-4.1-mini" });
      const store = new MemoryConversations();
      store.append("c", c.records.map((r: Rec) => Object.fromEntries(Object.entries(r).filter(([k]) => k !== "seq"))));
      const rule = c.rule;
      const chat = tutor.conversation("c", { store, context: { last: rule.last, without: rule.without ?? [] } });
      const log = await chat.readLog();
      for (const d of log.programs.values()) d["signature"] = tutor.signatureId;     // the case's program is this one
      const ctx = await chat.contextOf(log, c.parent, {});
      const entries = ctx.turns.map((turn, i) => ({ call: ctx.ids[i], ...(((turn["steps"] ?? []) as unknown[]).length ? { steps: true } : {}) }));
      assert.deepEqual(ctx.finish(entries as Rec[]), c.expect.saw);
      assert.deepEqual(ctx.rows, c.expect.rows);
    } finally {
      clock.now = before;
    }
  });
}

for (const [name, c] of cases("tools")) {
  test(`tools case ${name}`, () => {
    if (c.kind === "asks") {
      const rule = c.rule === "function" ? () => true : c.rule;
      assert.deepEqual(c.approvals.map((a: Rec) => asks(rule, { name: a.name, path: a.path, effects: a.effects })), c.expect.asks);
    } else assert.equal(denial(c.reason), c.expect.output);
  });
}

for (const [name, c] of cases("views")) {
  test(`views case ${name}`, () => assert.deepEqual(outside(c.events, { answerFrom: c.answer_from }), c.expect.events));
}

for (const [name, c] of cases("context")) {
  test(`context case ${name}`, () => {
    if (c.expect.refuses) {
      assert.throws(() => earlierOf(c.records, c.call), (e: unknown) => e instanceof SawUnknown && e.code === c.expect.refuses);
      return;
    }
    const got = earlierOf(c.records, c.call);
    assert.deepEqual({ earlier: got["earlier"], conversation: got["conversation"] }, c.expect);
  });
}

/** lm15 settings as the cases write them (snake_case) and as TypeScript's are named. */
const camel = (k: string) => k.replace(/_([a-z])/g, (_, x: string) => x.toUpperCase());
const snake = (k: string) => k.replace(/[A-Z]/g, (x) => `_${x.toLowerCase()}`);

for (const [name, c] of cases("plugins")) {
  test(`plugins case ${name}`, async () => {
    if (c.kind === "order") {
      const made = new Map<string, Plugin>();
      const layers = c.layers.map((layer: Rec) => ({
        where: layer.where, plugins: layer.plugins.map((n: string) => made.get(n) ?? made.set(n, new Plugin(n)).get(n)!),
        ...(layer.where === "configure" && c.program_plugins === false ? { programPlugins: false } : {}),
      }));
      assert.deepEqual(inOrder(layers).map((p) => p.name), c.expect.order);
      return;
    }
    const hook = { before_call: "beforeCall", context: "context", tool_call: "toolCall", tool_result: "toolResult", turn_start: "turnStart" }[c.hook as string]!;
    let ran = 0;
    const plugins = c.changes.map((ch: Rec | null, i: number) => new Plugin(`p${i + 1}`).on(hook as never, () => {
      ran++;
      if (!ch) return null;
      return ch.settings ? { ...ch, settings: Object.fromEntries(Object.entries(ch.settings).map(([k, v]) => [camel(k), v])) } : ch;
    }));
    const { start, expect } = c;
    const read = { name: "read", description: "Read.", parameters: { type: "object" }, run: () => "r" };
    const write = { name: "write", description: "Write.", parameters: { type: "object" }, run: () => "w" };
    const f = ai("f", { description: "Answer.", input: { x: t.string() }, output: t.string(), tools: [read, write], router: new FakeRouter() as never, lm: "gpt-4.1-mini" });
    if (c.hook === "before_call") {
      const shaped = await beforeCall(plugins, { instruction: start.instruction, name: "f", program: f, inputs: { x: "1" }, settings: { lm: start.lm },
        tools: ["read", "write"], given: [], call: null });
      assert.deepEqual(shaped.sections, expect.sections);
      assert.equal(shaped.settings.lm, expect.lm);
      assert.equal(shaped.instruction ?? start.instruction, expect.instruction);
      assert.deepEqual(shaped.tools ?? ["read", "write"], expect.tools);
      const s = shaped.settings as Rec;
      const lm15 = { ...(s["config"] ?? {}), ...Object.fromEntries(["temperature", "maxTokens", "topP", "seed"].filter((k) => s[k] !== undefined).map((k) => [k, s[k]])) };
      assert.deepEqual(Object.fromEntries(Object.entries(lm15).map(([k, v]) => [snake(k), v])), expect.settings);
    } else if (c.hook === "context") {
      const store = new MemoryConversations();
      const recs: Rec[] = [{ functai_conversation: 1, kind: "program", at: "2026-09-30T10:00:00.000000Z", version: "v", name: "f", program_kind: "ai", module: "m", interface: f.interface, fields: [], answer: "result" }];
      let parent: string | null = null;
      for (const turn of start.keep) {
        recs.push({ functai_conversation: 1, kind: "turn", at: "2026-09-30T10:00:00.000000Z", turn, parent, program: "v", inputs: { x: turn } });
        recs.push({ functai_conversation: 1, kind: "ended", at: "2026-09-30T10:00:00.000000Z", turn, state: "done", outputs: { result: "r", photo: "P", notes: "N" } });
        parent = turn;
      }
      store.append("c", recs);
      configure({ plugins });
      try {
        const chat = f.conversation("c", { store });
        const ctx = await chat.contextOf(await chat.readLog(), parent, {});
        assert.deepEqual((ctx.recorded!["turns"] as string[]), expect.keep);
        assert.deepEqual(ctx.sections, expect.sections);
        assert.deepEqual(ctx.recorded!["without"], expect.without);
      } finally {
        configure({ plugins: null });
      }
    } else if (c.hook === "turn_start") {
      const g = ai("g", { description: "G.", input: { message: t.string(), tone: t.string() }, output: t.string(), router: new FakeRouter() as never, lm: "gpt-4.1-mini" });
      configure({ plugins });
      try {
        const chat = g.conversation("ts-case");
        const [got] = await (chat as unknown as { turnStart(i: unknown, p: Plugin[]): Promise<[Rec, Rec[]]> }).turnStart(start.inputs, plugins);
        assert.deepEqual(got, expect.inputs);
      } finally {
        configure({ plugins: null });
      }
    } else {
      const call = new Call(() => ({ name: "f", kind: "ai", module: "m", version: "v", interface: "i", answer: "result" }), undefined, null,
        { inputs: [], outputs: [], added: [] }, {}, {});
      const approval = { call: call.id, invocation: 1, id: "t1", name: "send", input: start.inputs ?? {}, effects: "changes" as const, path: "f/send", site: "f#1", plugin: "approval" };
      if (c.hook === "tool_call") {
        const [got, refused] = await toolCall(plugins, call, approval, {});
        if (expect.block) assert.ok(got === null && refused!.includes(expect.block), refused ?? "");
        else assert.deepEqual(got, expect.inputs);
      } else assert.equal(await toolResult(plugins, call, approval, {}, start.output), expect.output);
    }
    assert.equal(ran, expect.ran);
  });
}

for (const [name, c] of cases("baked")) {
  test(`baked case ${name}`, () => {
    const f = tsFunction(c.definition);
    const rows = c.rows.map((r: Rec) => ({ ...r.inputs, ...r.outputs }));
    const b = c.bake;
    const e = bakeEntry(f as never, { fixed: b.fixed ?? {}, derived: b.derived ?? {}, reasoning: b.reasoning === true, rows });
    const want = c.expect;
    const data = lmcc.signatureToDict(e.signature) as Rec;
    const fields = data.fields.map((x: Rec) => Object.fromEntries(Object.entries({ purpose: "plain", ...x }).filter(([k, v]) => k !== "type" && v !== null)));
    assert.deepEqual({ instructions: data.instructions, fields }, { instructions: want.signature.instructions, fields: want.signature.fields.map((x: Rec) => ({ purpose: "plain", ...x })) });
    assert.deepEqual(e.fixed, want.fixed);
    assert.deepEqual(e.derived, want.derived);
    c.rows.forEach((r: Rec, i: number) => {
      const [messages, reply] = studentMessages(e, { ...r.inputs, ...(b.fixed ?? {}) }, r.outputs);
      assert.deepEqual([...messages, { role: "assistant", content: reply }], want.examples[i].messages);
    });
  });
}

test("the contract has cases of every new kind", () => {
  for (const folder of ["replies", "conversations", "tools", "views", "context", "plugins", "baked"]) assert.ok(cases(folder).length, folder);
});

void Applied;
