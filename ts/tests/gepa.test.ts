/** GEPA, offline: a fake model answers by rules, and a fake teacher writes instructions. */

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import type { Request } from "@lm15/lm15";
import { ai, calls, configure, gepa, t, trials } from "../src/index.ts";
import { bestPair, fieldsText, frontier } from "../src/gepa.ts";
import { random } from "../src/optimize.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

const GOOD = "Use the labels exactly: booking, cancelation, information.";
const rows = [
  ["I need to reserve a room.", "booking"], ["How do I get there?", "information"], ["Cancel my reservation.", "cancelation"],
  ["Book me a suite.", "booking"], ["Please book a table for two.", "booking"], ["What time is breakfast?", "information"],
  ["I want to cancel tonight.", "cancelation"], ["Is there parking?", "information"],
].map(([query, intent]) => ({ query, intent }));

const lastText = (req: Request) => (req.messages[req.messages.length - 1]!.parts as unknown as { text?: string }[]).map((p) => p.text ?? "").join("");
const queryOf = (req: Request) => /<query>\n([\s\S]*?)\n<\/query>/.exec(lastText(req))?.[1] ?? "";
const reflecting = (req: Request) => String(req.system ?? "").includes("You improve the instruction");
const labelOf = (q: string) => (/reserve|book/i.test(q) ? "booking" : /cancel/i.test(q) ? "cancelation" : "information");
const classifier = (router: FakeRouter, more: Record<string, unknown> = {}) => ai({
  name: "intent", description: "Classify the user's intent.", inputs: { query: t.string() },
  output: t.enum("booking", "cancelation", "information"), router, ...more,
});

test("gepa rewrites the instruction from its mistakes, never showing the choosing rows", async () => {
  const router = new FakeRouter([], (req) => {
    if (reflecting(req)) return `<result>\n${GOOD}\n</result>`;
    const good = String(req.system ?? "").includes("Use the labels exactly");
    return `<result>\n${good ? labelOf(queryOf(req)) : "information"}\n</result>`;
  });
  const folder = mkdtempSync(join(tmpdir(), "gepa-"));
  const f = classifier(router, { logCalls: folder });
  const better = await gepa(f, rows, { expected: "intent", budget: 60, seed: 1 });
  assert.equal(better.instructions, GOOD);
  assert.notEqual(better.version, f.version);
  const shown = router.requests.filter(reflecting).map(lastText).join("\n");
  assert.match(shown, /wrong: the right answer is/);
  assert.match(shown, /- result: one of booking, cancelation, information/);
  // the choosing rows: the first half after the seeded shuffle
  const rng = random(1);
  const order = [...rows];
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j]!, order[i]!]; }
  for (const r of order.slice(0, 4)) assert.ok(!shown.includes(r.query), `${r.query} was shown`);
  const search = trials(better)!;
  const chosen = search.trials.filter((x) => x.chosen);
  assert.equal(chosen.length, 1);
  assert.equal(chosen[0]!.kind, "reflect");
  assert.equal(chosen[0]!.score, 1);
  assert.ok(search.calls <= 60);
  const log = calls(undefined, { folder });
  assert.ok(log.every((c) => (c["caller"] as Record<string, unknown>)?.["optimization"]), "every call is marked as part of the optimization");
  assert.ok(log.some((c) => (c["program"] as Record<string, unknown>)?.["name"] === "_reflect"));
});

test("gepa runs a row once per instruction, and keeps the written one when nothing beats it", async () => {
  const seen: string[] = [];
  const router = new FakeRouter([], (req) => {
    if (reflecting(req)) return "<result>\nStill vague.\n</result>";
    seen.push(`${req.system}|${queryOf(req)}`);
    return "<result>\ninformation\n</result>";
  });
  const f = classifier(router);
  const kept = await gepa(f, rows, { expected: "intent", budget: 40 });
  assert.equal(new Set(seen).size, seen.length);
  assert.equal(seen.length, trials(kept)!.calls);
  assert.equal(kept.version, f.version);
});

test("a proposal that copies an input is dropped, and the next reflection is told", async () => {
  const long = [
    ...[1, 2, 3, 4, 5, 6].map((i) => ({ query: `Hello there, I would like to know about option number ${i} please.`, intent: "information" })),
    ...[1, 2, 3, 4, 5, 6].map((i) => ({ query: `Please book the room number ${i} for the whole of next week.`, intent: "booking" })),
  ];
  const said: string[] = [];
  const router = new FakeRouter([], (req) => {
    if (reflecting(req)) {
      said.push(lastText(req));
      const q = /\n {2}query: ([^\n]*)/.exec(lastText(req))![1];
      return `<result>\nIf the message says '${q}', answer booking.\n</result>`;
    }
    return "<result>\ninformation\n</result>";
  });
  const f = classifier(router);
  const kept = await gepa(f, long, { expected: "intent", budget: 40 });
  assert.ok(trials(kept)!.trials.some((x) => x.note === "copied an input: dropped"));
  assert.ok(said.slice(1).some((s) => s.includes("dropped: it copied an input")));
  assert.equal(kept.instructions, f.instructions);
});

test("the frontier keeps candidates best somewhere and drops the dominated; pairs win different rows", () => {
  const scores = [[1, 0], [0, 1], [1, 1]];
  assert.deepEqual([...frontier(scores)], [[2, 2]]);
  assert.deepEqual(bestPair(scores.slice(0, 2), frontier(scores.slice(0, 2))), [0, 1]);
  assert.equal(bestPair(scores, frontier(scores)), null);
});

test("fields are described for the teacher in words, with their types (the same words as Python and R)", () => {
  const f = ai({
    name: "team", description: "Which team?",
    inputs: { message: { shape: t.string(), desc: "the customer's words" }, n: t.optional(t.integer()), tags: t.list(t.string()) },
    output: t.enum("a", "b"),
  });
  assert.equal(fieldsText(f),
    "Inputs:\n- message: text. the customer's words\n- n: a whole number, or nothing\n- tags: a list of text\nOutputs:\n- result: one of a, b");
});
