/**
 * Measuring programs (contract/measures.md, design/20): every case in
 * contract/cases/measures/, then the same calls on four shapes of program,
 * offline (a fake router stands in for every provider).
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ai, compare, configure, evaluate, labeledFewShot, MemoryCalls, module, rate, rated, t, tool, withSettings,
  agreement, asCase, caseKey, criterion, draw, judgePairs, positions, prefer, rank, scoreboard, shownOrder, splitOf, winRate,
  MeasureError, paired,
} from "../src/index.ts";
import * as m from "../src/measures.ts";
import { ratedRows } from "../src/calllog.ts";
import { FakeRouter } from "./fake.ts";
import { cases, type Rec } from "./cases.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

// ------------------------------------------------------------------ the contract's cases

function close(got: unknown, want: unknown, where: string): void {
  if (want === null || typeof want === "string" || typeof want === "boolean") {
    assert.equal(got ?? null, want, where);
  } else if (typeof want === "number") {
    assert.equal(typeof got, "number", `${where}: ${JSON.stringify(got)} is not a number`);
    assert.ok(Math.abs((got as number) - want) <= 1e-9, `${where}: ${got} != ${want}`);
  } else if (Array.isArray(want)) {
    assert.ok(Array.isArray(got) && got.length === want.length, `${where}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`);
    want.forEach((w, i) => close((got as unknown[])[i], w, `${where}[${i}]`));
  } else {
    const g = got as Rec;
    assert.ok(g && typeof g === "object", `${where}: not an object`);
    assert.deepEqual(Object.keys(g).sort(), Object.keys(want as Rec).sort(), `${where}: keys`);
    for (const k of Object.keys(want as Rec)) close(g[k], (want as Rec)[k], `${where}.${k}`);
  }
}

const listOf = (cs: m.CaseValues) => [...cs].flatMap(([g, c]) => [...c].map(([k, v]) => ({ group: g, case: k, value: v })));

function runCase(c: Rec): unknown {
  switch (c.kind) {
    case "key-row": return { key: m.rowKey(c.row, c.inputs) };
    case "key-call": return { key: m.callKey(c.record) };
    case "split":
      try {
        return { splits: c.rows.map((r: Rec) => splitOf(r, c.inputs, { share: c.share, salt: c.salt })) };
      } catch (err) {
        if (err instanceof MeasureError) return { refuses: err.code };
        throw err;
      }
    case "draw": return { ids: draw(c.ids, c.n, c.seed) };
    case "ratings": return { counting: m.counting(c.records, c.criterion).map((j) => ({ id: j.id, calls: j.calls, item: j.item, value: j.value, source: j.source })) };
    case "values": {
      const v = m.valuesOf(c.records, c.program, c.criterion, c.source, { by: c.by, drawn: c.drawn });
      return { results: v.results, cases: listOf(v.cases) };
    }
    case "preferences": {
      const got = m.preferencesOf(c.records, c.program, c.criterion, c.source, c.a, c.b, { by: c.by, drawn: c.drawn });
      return { cases: [...got].map(([k, v]) => ({ case: k, value: v })) };
    }
    case "paired": return m.paired(c.a, c.b, c.margin, c.bounded);
    case "winrate": return m.winRateOf(c.values, c.margin);
    case "fragility": return { fragility: m.fragility(c.values) };
    case "ppi": return m.ppi(c.judge, c.people, c.bounded);
    case "ppi-paired": return m.ppiPaired(c.judge_a, c.judge_b, c.people_a, c.people_b, c.margin, c.bounded);
    case "agreement": return m.agreementOf(c.a, c.b, c.choice);
    case "self": return m.selfAgreementOf(c.per_case);
    case "rank": return m.bradleyTerry(c.versions, c.outcomes.map((o: Rec) => [o.case, o.first, o.second, o.first_share] as const));
    case "positions": return m.positionsOf(c.records.filter((r: Rec) => "functai_rating" in r), c.criterion, c.judge);
    case "scoreboard": return m.scoreboardOf(c.records, c.program, c.criteria, { by: c.by });
    case "compare": return m.compareOf(c.records, c.program, c.criterion, c.source, c.a, c.b, { by: c.by, drawn: c.drawn, margin: c.margin });
    case "disagreements": return { cases: m.disagreementsOf(c.records, c.program, c.criterion, c.source, { by: c.by }) };
    case "rated": {
      const [rows, left] = ratedRows(c.records.filter((r: Rec) => "functai_call" in r), c.records.filter((r: Rec) => "functai_rating" in r), c.rated);
      return { rows, left_out: left };
    }
    default: throw new Error(`a kind this harness does not know: ${c.kind}`);
  }
}

for (const [name, c] of cases("measures")) {
  test(`measures/${name}`, () => close(runCase(c), c.expect, name));
}

// ------------------------------------------------------------------ 1. known answers

const moodDef = { description: "The customer's overall mood: happy, unhappy or mixed.", input: { review: t.string() }, output: t.string() };
const reviewOf = (req: { messages: readonly { parts: readonly unknown[] }[] }) => {
  const text = (req.messages.at(-1)!.parts as { text?: string }[]).map((p) => p.text ?? "").join("");
  return /<review>\n([\s\S]*?)\n<\/review>/.exec(text)?.[1] ?? "";
};
const moodReply = (req: { messages: readonly { parts: readonly unknown[] }[] }) => {
  const r = reviewOf(req).toLowerCase();
  return `<result>\n${r.includes("but") ? "mixed" : r.includes("broke") ? "unhappy" : "happy"}\n</result>`;
};
const MOODS = [
  { review: "Love it. Works exactly as described.", result: "happy" },
  { review: "Broke after two days.", result: "unhappy" },
  { review: "Great sound, but the battery is weak.", result: "mixed" },
  { review: "Arrived early and works.", result: "happy" },
];

test("calls say their case and repeat, and reuse asks only what is missing", async () => {
  const router = new FakeRouter([], moodReply);
  const mood = ai("mood", { ...moodDef, router });
  const log = new MemoryCalls();
  const ev = await evaluate(mood, MOODS, { repeats: 2, reuse: log, progress: false });
  const recorded = log.read().filter((r: Rec) => "functai_call" in r) as Rec[];
  assert.equal(recorded.length, 8);
  assert.deepEqual(new Set(recorded.map((c) => c.case)), new Set(MOODS.map((r) => caseKey(mood, r))));
  assert.deepEqual(recorded.map((c) => c.repeat).sort(), [1, 1, 1, 1, 2, 2, 2, 2]);
  assert.equal(ev.score, 1);
  assert.equal(ev.summary[0]!.n, 4);                          // over cases, not runs
  assert.equal(router.requests.length, 8);
  const again = await evaluate(mood, MOODS, { repeats: 3, reuse: log, progress: false });
  assert.equal(router.requests.length, 12);                   // 8 reused, 4 asked
  assert.equal(again.rows.filter((r) => r.reused).length, 8);
  assert.ok(again.rows.every((r) => r.callId));
});

test("thirty cases right for both say how little they show", async () => {
  const router = new FakeRouter([], moodReply);
  const mood = ai("mood", { ...moodDef, router });
  const rows = Array.from({ length: 30 }, (_, i) => ({ review: `Love it, number ${i}.`, result: "happy" }));
  const [c] = compare(await evaluate(mood, rows, { progress: false }), await evaluate(mood, rows, { progress: false }), { margin: 0.05 });
  assert.equal(c!.diff, 0);
  assert.equal(c!.method, "mover-wilson");
  assert.ok(Math.abs(c!.low! + 0.11351339476145195) < 1e-12);  // the paired t said [0, 0]
  assert.equal(c!.outcome, "unclear");
});

test("compare pairs by case: rows in another order, a subset, a column that is not an input", async () => {
  const router = new FakeRouter([], moodReply);
  const mood = ai("mood", { ...moodDef, router });
  const before = await evaluate(mood, MOODS, { progress: false });
  const [shuffled] = compare(before, await evaluate(mood, [...MOODS].reverse().map((r) => ({ ...r, rules: "x" })), { progress: false }));
  assert.equal(shuffled!.n, 4);
  const [part] = compare(before, await evaluate(mood, MOODS.slice(0, 2), { progress: false }));
  assert.deepEqual([part!.n, part!.onlyBefore, part!.onlyAfter], [2, 2, 0]);
});

test("held-out rows never reach an optimizer", () => {
  const mood = ai("mood", { ...moodDef, router: new FakeRouter([], moodReply) });
  assert.throws(() => labeledFewShot(mood, MOODS.map((r, i) => ({ ...r, split: i === 0 ? "held_out" : "train" }))), /1 of the 4 rows are held out/);
});

test("split is stable, and keeps a conversation on one side", () => {
  const rows = Array.from({ length: 200 }, (_, i) => ({ q: `question ${i}` }));
  const sides = rows.map((r) => splitOf(r, ["q"], { share: 0.25 }));
  const held = sides.filter((s) => s === "held_out").length;
  assert.ok(held > 30 && held < 70);
  const turns = Array.from({ length: 50 }, (_, i) => ({ q: `turn ${i}`, conversation: `c${Math.floor(i / 5)}` }));
  const by = new Map<string, Set<string>>();
  for (const r of turns) by.set(r.conversation, (by.get(r.conversation) ?? new Set()).add(splitOf(r, ["q"], { share: 0.3 })));
  assert.ok([...by.values()].every((s) => s.size === 1));
  assert.throws(() => splitOf({ q: "x", split: "test" }, ["q"]), MeasureError);
});

test("a rating format 1 can say is format 1; another is format 2", async () => {
  const router = new FakeRouter([], moodReply);
  const log = new MemoryCalls();
  const mood = ai("mood", { ...moodDef, router });
  const p = await withSettings({ logCalls: log }, () => mood.predict({ review: "Broke after two days." }));
  const r1 = rate(p, "right", { by: "megan", blind: true, folder: log });
  const r2 = rate(p, "yes", { criterion: criterion("polite", { kind: "label", answers: { yes: 1, no: 0 } }), by: "megan", folder: log });
  assert.equal(r1.functai_rating, 1);
  assert.equal(r1.blind, true);
  assert.equal(r2.functai_rating, 2);
  assert.equal(r2.kind, "label");
  const { rows } = rated(mood, { records: log.read() as Rec[] });
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.result, "unhappy");
  const row = asCase(p, { records: log.read() as Rec[] });
  assert.equal(row.result, "unhappy");
  assert.equal(row.split, splitOf({ review: "Broke after two days." }, ["review"]));
});

// ------------------------------------------------------------------ 2. a study judged item by item

const SITES = Array.from({ length: 6 }, (_, i) => ({ url: `https://site${i}.example` }));
const realProblem = criterion("real_problem", { kind: "label", items: "/result", answers: { yes: 1, no: 0, unsure: null }, question: "Is this a real problem?" });
const problemReply = (req: { messages: readonly { parts: readonly unknown[] }[] }) => {
  const text = (req.messages.at(-1)!.parts as { text?: string }[]).map((p) => p.text ?? "").join("");
  const n = Number(/site(\d)/.exec(text)![1]);
  return `<result>\n${n % 2 ? "Slow search" : "Hidden menu"} on site ${n}\n</result>`;
};
const professor = ({ item }: { item?: unknown }) => (String((item as Rec).title).toLowerCase().includes("search") ? "yes" : "no");

test("a study on the scoreboard: people, people drawn, a judge, combined", async () => {
  const router = new FakeRouter([], problemReply);
  const findProblem = ai("find_problem", { description: "One usability problem a visitor would meet on this site.", input: { url: t.string() }, output: t.string(), router });
  const study = module("study", { input: { url: t.string() }, output: t.list(t.object({ title: t.string() })) },
    async ({ url }: { url: string }) => [{ title: await findProblem(url) }, { title: await findProblem(url) }]);
  const log = new MemoryCalls();
  const ev = await evaluate(study as never, SITES as never[], { criteria: [{ ...realProblem, judge: professor }], reuse: log, progress: false });
  assert.deepEqual(ev.metrics, ["real_problem"]);
  assert.ok(Math.abs(ev.score! - 0.5) < 1e-12);
  const all = log.read() as Rec[];
  const judged = all.filter((r) => r.functai_rating === 2);
  assert.equal(judged.length, 12);
  const studies = all.filter((r) => "functai_call" in r && r.program.name === "study");
  const drawn = draw(studies.map((c) => String(c.id)), 4, "draw-1");
  for (const id of drawn) {
    const c = studies.find((x) => x.id === id)!;
    (c.outputs.result as Rec[]).forEach((it, i) => rate(id, String(it.title).toLowerCase().includes("search") ? "yes" : "no",
      { criterion: realProblem, item: i, by: "megan", sample: "draw-1", blind: true, folder: log }));
  }
  const board = scoreboard(study as never, [realProblem], { records: log });
  assert.deepEqual(board.groups[0]!.measures.map((x) => x.source), ["people", "people drawn", "judge", "combined"]);
  assert.equal(board.groups[0]!.input_tokens, 6 * 2 * 10);    // the steps' tokens count
  const a = agreement(study as never, realProblem, "people", professor, { records: log });
  assert.equal(a.agree, 1);
  assert.equal(a.n, 8);
  // a second evaluation reuses the calls and the judge's ratings
  let asked = 0;
  const counting = (args: { item?: unknown }) => { asked++; return professor(args); };
  await evaluate(study as never, SITES as never[], { criteria: [{ ...realProblem, judge: counting }], reuse: log, progress: false });
  const askedOnce = asked;
  await evaluate(study as never, SITES as never[], { criteria: [{ ...realProblem, judge: counting }], reuse: log, progress: false });
  assert.equal(askedOnce, 12);
  assert.equal(asked, 12);
});

// ------------------------------------------------------------------ 3. a conversation's turns

test("a turn with earlier turns is its own case", () => {
  const first = { question: "Weather today?" };
  const later = { question: "And tomorrow?", earlier: [{ inputs: first, outputs: { result: "Sunny." } }], conversation: "c1" };
  assert.notEqual(caseKey(["question"], later), caseKey(["question"], { question: "And tomorrow?" }));
  assert.equal(caseKey(["question"], { ...first, earlier: [] }), caseKey(["question"], first));
});

// ------------------------------------------------------------------ 4. side by side, ranked

test("agents with tools: pairs judged both ways, a failure that loses on its own, and a ranking", async () => {
  const toolResults = (req: { messages: readonly { parts: readonly unknown[] }[] }) =>
    req.messages.flatMap((x) => x.parts as Rec[]).filter((p) => p.type === "tool_result").length;
  const router = new FakeRouter([], (req) => {
    const text = req.messages.flatMap((x) => x.parts as unknown as { text?: string }[]).map((p) => p.text ?? "").join("");
    if (!toolResults(req) && req.tools?.length) return { calls: [{ id: "c1", name: "lookup", input: { what: "the task" } }] };
    if (text.includes("fail") && req.model.endsWith("bad")) throw new Error("the model gave up");
    return `<result>\ndone by ${req.model}\n</result>`;
  });
  const lookup = tool("lookup", { description: "Look something up.", input: { what: t.string() }, effects: "reads" }, ({ what }) => `found ${what}`);
  const log = new MemoryCalls();
  const tasks = [...Array.from({ length: 5 }, (_, i) => ({ task: `task ${i}` })), { task: "fail please" }];
  const agents = Object.fromEntries(["good", "mid", "bad"].map((s) => [s, ai("agent", { description: "Do the task.", input: { task: t.string() }, output: t.string(), router, lm: `gpt-4.1-${s}`, apiRetries: 0, tools: [lookup] } as never)]));
  for (const a of Object.values(agents)) await evaluate(a as never, tasks as never[], { metric: () => 1, reuse: log, progress: false });
  const calls = (log.read() as Rec[]).filter((r) => "functai_call" in r);
  assert.ok(calls.filter((c) => c.error === null).every((c) => Array.isArray(c.steps) && c.steps.some((x: Rec) => x.kind === "tool")));
  const by = new Map(calls.map((c) => [`${c.program.setup} ${c.case}`, c] as const));
  const keys = [...new Set(calls.map((c) => String(c.case)))].sort();
  const pairs: [Rec, Rec][] = [];
  for (const k of keys) {
    const [g, mid, b] = ["good", "mid", "bad"].map((s) => by.get(`${(agents[s] as Rec).setup} ${k}`)!);
    for (const p of [[g, mid], [mid, b], [g, b]] as [Rec, Rec][]) if (p[0].error === null && p[1].error === null) pairs.push(p);
  }
  const rankOf: Rec = { good: 2, mid: 1, bad: 0 };
  const picker = ({ first, second }: { first?: unknown; second?: unknown }) => {
    const [x, y] = [first, second].map((v) => rankOf[String(v).split("-").at(-1)!] as number);
    return x! > y! ? "first" : y! > x! ? "second" : "first";
  };
  const written = await judgePairs(picker, pairs, undefined, { records: log });
  assert.equal(written.length, 2 * pairs.length);
  assert.equal(positions(picker, undefined, { records: log }).consistent, 1);
  const order = rank("agent", undefined, { records: log, source: picker });
  assert.deepEqual(order.order, ["good", "mid", "bad"].map((s) => (agents[s] as Rec).setup));
  const wr = winRate("agent", (agents.bad as Rec).setup, (agents.good as Rec).setup, undefined, { source: picker, records: log });
  assert.equal(wr.rate, 1);
  assert.equal(wr.n, 6);                                       // the bad agent's failure loses without a judge
});

test("prefer records a blind pick, shown in an order every language agrees on", async () => {
  const log = new MemoryCalls();
  const router = new FakeRouter([], moodReply);
  const a = await withSettings({ logCalls: log }, () => ai("mood", { ...moodDef, router }).predict({ review: "Love it." }));
  const b = await withSettings({ logCalls: log }, () => ai("mood", { ...moodDef, router, lm: "gpt-4.1" }).predict({ review: "Love it." }));
  const [first, second] = shownOrder(a, b, "queue-1");
  const r = prefer(first, second, second, { by: "megan", blind: true, folder: log });
  assert.deepEqual(r.calls, [first, second]);
  assert.equal(r.value, second);
  assert.deepEqual(shownOrder(b, a, "queue-1"), [first, second]);
});

test("a bounded paired t range stays within -1 and 1", () => {
  assert.ok(paired([0.5, 0.75], [1, 1], null, true).high! <= 1);
});

test("every record written passes the contract's schemas", async () => {
  const { passes } = await import("../src/schema.ts");
  const router = new FakeRouter([], moodReply);
  const log = new MemoryCalls();
  const mood = ai("mood", { ...moodDef, router });
  const other = ai("mood", { ...moodDef, router, lm: "gpt-4.1" });
  const politeJudge = ({ answer }: { answer?: unknown }) => (String(answer).length > 3 ? "yes" : "no");
  const polite = criterion("polite", { kind: "label", answers: { yes: 1, no: 0, unsure: null }, judge: politeJudge });
  await evaluate(mood, MOODS, { repeats: 2, reuse: log, criteria: [polite], progress: false });
  await evaluate(other, MOODS.slice(0, 1), { reuse: log, progress: false });
  const calls = (log.read() as Rec[]).filter((r) => "functai_call" in r);
  rate(calls[0]!.id, "right", { by: "megan", blind: true, folder: log });
  rate(calls[0]!.id, "unsure", { criterion: polite, by: "megan", sample: "d", blind: true, folder: log });
  prefer(calls[0]!.id, calls.at(-1)!.id, "tie", { by: "megan", blind: true, folder: log });
  await judgePairs(() => "second", [[calls[0]!, calls.at(-1)!]], undefined, { records: log });
  await m.judge(() => 4, [calls[0]!], criterion("clarity", { kind: "score", scale: [1, 5] }), { records: log });
  const kinds = new Set<string>();
  for (const r of log.read() as Rec[]) {
    if ("functai_call" in r) assert.ok(passes("call", r), `call ${r.program.name}`);
    else {
      assert.ok(passes("rating", r), JSON.stringify(r));
      kinds.add(`${r.functai_rating} ${r.kind ?? ""}`);
    }
  }
  assert.deepEqual([...kinds].sort(), ["1 ", "2 choice", "2 label", "2 score"]);
});
