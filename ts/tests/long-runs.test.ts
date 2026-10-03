/** Stages 1.2, 3 and 5 against a fake model: the reply cache on disk, serving and remote, rated turns asked again, escalation, baking. */

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { Response } from "@lm15/lm15";
import { ai, baked, bakeExamples, calls, compare, DiskReplies, evaluate, exportExamples, inspectHistory, module, phistory, pruneCalls,
  quotesFound, rate, rated, remote, serve, Service, split, t, tool } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

type Rec = Record<string, any>;
const folder = () => mkdtempSync(join(tmpdir(), "functai-long-"));
const mood = (router: unknown, extra: Rec = {}) => ai("mood", { description: "How does the customer feel?", input: { review: t.string() },
  output: t.enum("happy", "unhappy"), router: router as never, lm: "gpt-4.1-mini", ...extra });

test("the reply cache on disk: a second process gets the kept reply; replicate is another answer; a call whose log drops a field is not written", async () => {
  const path = join(folder(), "replies.sqlite");
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  await mood(router, { cacheReplies: path })("Love it.");
  await mood(router, { cacheReplies: new DiskReplies(path) })("Love it.");    // another store on the same file: another process
  assert.equal(router.requests.length, 1);
  await mood(router, { cacheReplies: path, replicate: 1 })("Love it.");
  assert.equal(router.requests.length, 2);
  const disk = new DiskReplies(path);
  assert.equal(disk.size, 2);
  await mood(router, { cacheReplies: path, logContent: { review: false } })("Private.");
  assert.equal(disk.size, 2, "a call that may not keep its input never reaches the disk");
  assert.equal((Number(readFileSync(path).length) > 0), true);
});

test("one flight per key: calls of one request at once ask the model once", async () => {
  let release!: () => void;
  const gate = new Promise<void>((r) => { release = r; });
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  const slow = { resolve: (m: string) => router.resolve(m), complete: async (r: never) => { await gate; return router.complete(r); } };
  const path = join(folder(), "replies.sqlite");
  const f = mood(slow, { cacheReplies: path });
  const all = Promise.all([f("Same."), f("Same."), f("Same.")]);
  setTimeout(release, 30);
  assert.deepEqual(await all, ["happy", "happy", "happy"]);
  assert.equal(router.requests.length, 1);
});

test("mapSettled goes on past a failure; inspectHistory and phistory show what was sent", async () => {
  const router = new FakeRouter([], (r) => (JSON.stringify(r.messages).includes("bad") ? "nonsense" : "<result>\nhappy\n</result>"));
  const f = mood(router, { retries: 0 });
  const out = await f.mapSettled(["good", "bad", "fine"], { concurrency: 2, progress: false });
  assert.deepEqual(out.map((r) => r.status), ["fulfilled", "rejected", "fulfilled"]);
  assert.ok(inspectHistory(3).length === 3);
  assert.match(phistory(), /mood → gpt-4\.1-mini/);
});

test("serving: interface, call, a stream's outside view, keys; remote is a program again, one call tree across two logs", async () => {
  const serverLogs = folder();
  const callerLogs = folder();
  const router = new FakeRouter([], () => "<result>\nhappy\n</result>");
  const served = mood(router, { logCalls: serverLogs });
  const server = await serve(served, { port: 0, keys: ["s3cret"] });
  const port = (server.address() as { port: number }).port;
  try {
    const base = `http://127.0.0.1:${port}`;
    assert.equal((await fetch(`${base}/interface`)).status, 401);
    const described = await (await fetch(`${base}/interface`, { headers: { authorization: "Bearer s3cret" } })).json() as Rec;
    assert.equal(described.functai_interface, 1);
    assert.equal(described.version, served.version);
    const bad = await fetch(`${base}/call`, { method: "POST", headers: { authorization: "Bearer s3cret" }, body: JSON.stringify({ inputs: {} }) });
    assert.equal(bad.status, 422);
    const sse = await (await fetch(`${base}/stream`, { method: "POST", headers: { authorization: "Bearer s3cret" }, body: JSON.stringify({ inputs: { review: "ok" } }) })).text();
    assert.match(sse, /^id: 1-1\nevent: started\n/);
    assert.match(sse, /event: done\n/);
    const team = await remote(base, { key: "s3cret" });
    assert.equal(await team.using({ logCalls: callerLogs })({ review: "Love it." }), "happy");
    const mine: Rec = calls(undefined, { folder: callerLogs })[0]!.record;
    const theirs: Rec = calls(undefined, { folder: serverLogs }).at(-1)!.record;
    assert.equal(mine["program"]["kind"], "remote");
    assert.equal(mine["program"]["version"], served.version);
    assert.equal(theirs["parent"], mine["id"]);                              // one call tree across two logs
    const ev = await evaluate(team, [{ review: "Lovely.", result: "happy" }, { review: "Awful.", result: "unhappy" }], { progress: false });
    assert.equal(ev.score, 0.5);                                             // a served program is evaluated as a local one
    const conv = await fetch(`${base}/conversations/c1/turns`, { method: "POST", headers: { authorization: "Bearer s3cret" }, body: JSON.stringify({ inputs: { review: "hi" }, wait: true }) });
    assert.equal(conv.status, 201);
    const turn = await conv.json() as Rec;
    assert.equal(turn.state, "done");
    assert.equal(turn.value, "happy");
  } finally {
    server.close();
  }
  await assert.rejects(serve(served, { host: "0.0.0.0", port: 0 }), /serve-keys|anyone on the network/);
  const opaque = module("o", { input: { x: t.opaque() }, output: t.string() }, () => "x");
  assert.throws(() => new Service(opaque as never), /no JSON form/);
});

test("rated turns keep their earlier turns, and evaluating asks each one again with them", async () => {
  const logs = folder();
  const router = new FakeRouter([], (_r, i) => `<result>\n${i % 2 ? "unhappy" : "happy"}\n</result>`);
  const f = mood(router, { logCalls: logs });
  const chat = f.conversation("c");
  await chat("first");
  await chat("second");
  const turns = await chat.turns();
  rate(turns[1]!.id, "wrong", { answer: "happy", folder: logs, by: "ana" });
  const { rows, leftOut } = rated(f, { folder: logs });
  assert.equal(leftOut.noContext, 0);
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!["conversation"], "c");
  assert.deepEqual((rows[0]!["earlier"] as Rec[]).map((e) => e["inputs"]), [{ review: "first" }]);
  const ev = await evaluate(f, rows as never, { progress: false });
  const asked = router.requests.at(-1)!;
  assert.equal(asked.messages.length, 3, "asked again with its earlier turn");
  const again = calls(f, { folder: logs }).at(-1)!.record;
  assert.deepEqual(again["saw"], [{ saw_of: turns[1]!.id }]);
  assert.equal(ev.rows.length, 1);
  const [train, held] = split(rows, { test: 0.5 });
  assert.equal(train.length + held.length, 1);
  assert.equal(compare(ev, ev)[0]!.diff, 0);
});

/** A TypeSafe-style reply: the answer as data, with the model's probability for each answer. */
const measured = (answer: string, p: number) => (r: { model: string }) => Response.fromJSON({
  model: r.model, finish_reason: "stop", usage: { input_tokens: 3, output_tokens: 1 },
  message: { role: "assistant", parts: [{ type: "data", value: { result: answer }, probabilities: { result: { happy: p, unhappy: 1 - p } }, method: "provider_classification" }] },
} as never);

test("escalation: a first model less sure than escalateBelow hands the question to another; the record says so", async () => {
  const logs = folder();
  const sure = new FakeRouter([], measured("happy", 0.97) as never, "typesafe");
  assert.equal(await mood(sure, { lm: "typesafe:jev-latest", escalateTo: "gpt-4.1-mini" })("fine"), "happy");
  assert.equal(sure.requests.length, 1);
  const unsure = new FakeRouter([], ((r: { model: string }, i: number) => (i === 0 ? measured("happy", 0.55)(r) : "<result>\nunhappy\n</result>")) as never, "typesafe");
  const f = mood(unsure, { lm: "typesafe:jev-latest", escalateTo: "openai:gpt-4.1-mini", logCalls: logs });
  const p = await f.predict("it broke, but support was quick");
  assert.equal(p.answer, "unhappy");
  assert.equal(p.first!.answer, "happy");
  assert.equal(unsure.requests.length, 2);
  assert.ok(unsure.requests[1]!.model.endsWith("gpt-4.1-mini"));
  const rec = calls(f, { folder: logs })[0]!.record;
  assert.equal(rec["escalated"], true);
  assert.equal((rec["exchanges"] as unknown[]).length, 2);
  // an AI function as the target: a call of its own, inside this one
  const r2 = new FakeRouter([], ((r: { model: string }, i: number) => (i === 0 ? measured("happy", 0.5)(r) : "<result>\nunhappy\n</result>")) as never, "typesafe");
  const careful = ai("careful_mood", { description: "Read carefully.", input: { review: t.string() }, output: t.enum("happy", "unhappy"), router: r2 as never, lm: "openai:gpt-4.1-mini" });
  assert.equal(await mood(r2, { lm: "typesafe:jev-latest", escalateTo: careful })("hm"), "unhappy");
  // no probabilities: escalation cannot decide, and says so
  await assert.rejects(mood(new FakeRouter([], () => "<result>\nhappy\n</result>"), { escalateTo: "gpt-4.1" })("x"), /measures its confidence/);
});

test("baking: the examples every trainer reads, and a student served elsewhere, called as it was trained", async () => {
  const where = folder();
  const summarize = ai("label", { description: "Label the message.", input: { message: t.string() }, output: t.string(), lm: "gpt-4.1-mini" });
  const rows = [{ message: "a", result: "A" }, { message: "b", result: "B" }];
  const table = bakeExamples(summarize as never, rows);
  assert.equal(table.length, 2);
  assert.equal(table[0]!.messages.at(-1)!["role"], "assistant");
  const path = await exportExamples(join(where, "ex.jsonl"), summarize as never, rows);
  const meta = JSON.parse(readFileSync(path + ".meta.json", "utf8"));
  assert.equal(meta.functai_examples, 1);
  const dir = join(where, "baked");
  mkdirSync(dir);
  writeFileSync(join(dir, "baked.json"), JSON.stringify({ functai_baked: 2, kind: "generative", name: "lbl", student: "s", functions: meta.functions }));
  const seen: Rec[] = [];
  const http = await import("node:http");
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      seen.push(JSON.parse(body));
      res.end(JSON.stringify({ model: "lbl", choices: [{ message: { content: "<result>\nA\n</result>" }, finish_reason: "stop" }], usage: {} }));
    });
  }).listen(0);
  await new Promise((r) => server.once("listening", r));
  try {
    const student = baked(dir, { url: `http://127.0.0.1:${(server.address() as { port: number }).port}/v1` });
    assert.equal(await summarize.using({ lm: student as never })("a"), "A");
    assert.deepEqual(seen[0]!.messages, table[0]!.messages.slice(0, -1));     // called with the very messages it was trained on
    assert.equal(seen[0]!.chat_template_kwargs.enable_thinking, false);
  } finally {
    server.close();
  }
});

test("quotesFound, and pruneCalls keeps what ratings need", async () => {
  assert.deepEqual(quotesFound("The parcel left Leeds on Monday. It was delayed by snow.", ["“It was delayed by snow”", "It was lost"]), [true, false]);
  const logs = folder();
  mkdirSync(join(logs, "2020-01-01"));
  const call = (id: string) => ({ functai_call: 2, id, parent: null, root: id, program: { name: "x" }, started: "2020-01-01T00:00:00.000000Z", saw: [] });
  writeFileSync(join(logs, "2020-01-01", "h.jsonl"), [call("a"), call("b"), { functai_rating: 1, id: "r", call: "a", at: "2020-01-01T00:00:01.000000Z", verdict: "right" }].map((r) => JSON.stringify(r)).join("\n") + "\n");
  const got = pruneCalls({ olderThan: "30d", folder: logs });
  assert.deepEqual(got, { days: 1, calls: 1, kept: 1 });
  assert.ok(!existsSync(join(logs, "2020-01-01")));
  assert.equal(readdirSync(logs).filter((f) => f.startsWith("kept-")).length, 1);
  assert.equal(calls(undefined, { folder: logs }).length, 1);
});

void module;
void tool;
