/**
 * JSON FunctAI writes itself, and data it hands lm15.
 *
 * - The writer (values.ts `writeData`) writes what `JSON.stringify` writes,
 *   members in the value's order: every array index (a hole is `null`), a
 *   boxed primitive as its value, a value that holds itself refused. Through
 *   requests, call log lines and saved folders.
 * - What FunctAI gives lm15 (a request with its `response_format` schema and
 *   tool parameters, a `Config`, a saved `config`, a cached reply) carries
 *   no record of member order: lm15 refuses an object with a symbol key.
 * - An integer past 2^53 in an answer is recorded as the integer it is.
 *
 * Through public calls, with a fake model.
 */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as lmcc from "lmcc";
import { Request, stringifyJson, type Request as Req } from "@lm15/lm15";
import * as bridge from "lmcc/lm15";
import { ai, configure, flush, fromManifest, load, save, t, toManifest, tool, type StreamEvent } from "../src/index.ts";
import { writeData } from "../src/values.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

type Rec = Record<string, any>;
const folder = () => mkdtempSync(join(tmpdir(), "functai-data-"));
function logLines(where: string): string[] {
  const out: string[] = [];
  for (const day of readdirSync(where)) for (const f of readdirSync(join(where, day))) out.push(...readFileSync(join(where, day, f), "utf8").split("\n").filter(Boolean));
  return out;
}
/** The text of every part of a request's messages. */
const texts = (r: Req) => r.messages.flatMap((m) => m.parts.map((p) => (p as { text?: string }).text ?? "")).join("\n");
/** Every symbol-keyed member anywhere in a value (lm15 refuses an object that has one). */
function symbolKeys(value: unknown, seen = new Set<object>()): string[] {
  if (value === null || typeof value !== "object" || seen.has(value)) return [];
  seen.add(value);
  const own = Object.getOwnPropertySymbols(value).map(String);
  return [...own, ...Reflect.ownKeys(value).filter((k) => typeof k === "string").flatMap((k) => symbolKeys((value as Rec)[k as string], seen))];
}

/* eslint-disable no-sparse-arrays */
const ARRAYS: [string, unknown][] = [
  ["a leading hole", [, "B"]],
  ["a middle hole", ["A", , "C"]],
  ["a trailing hole", ["A", ,]],
  ["only holes", new Array(2)],
  ["holes, nested", { a: [, 1], b: [[, ,], "x"] }],
  ["dense, with null", [null, "B"]],
  ["dense, with undefined", [undefined, "B"]],
  ["dense", ["A", "B"]],
];

const BOXED: [string, unknown][] = [
  ["a Number", new Number(42)],
  ["a String", new String("x")],
  ["a Boolean", new Boolean(false)],
  ["nested", { n: new Number(1.5), s: [new String("y")], b: { c: new Boolean(true) } }],
];
const BOXED_BY_TOJSON: [string, unknown] = ["from toJSON", { v: { toJSON: () => new Number(3) } }];

// ------------------------------------------------------------------ the writer

// Astra, round 5 (B1, S1): the writer iterated arrays with forEach, which skips holes, so `[, "B"]` was written `[,"B"]`
// (not JSON), `["A", ,]` lost its last element and `new Array(2)` both; and it wrote `new Number(42)` as `{}`.
test("the writer writes what JSON.stringify writes: every array index, a hole as null; a boxed primitive as its value", () => {
  for (const [label, value] of [...ARRAYS, ...BOXED, BOXED_BY_TOJSON]) {
    for (const indent of [0, 2]) assert.equal(writeData(value, indent), JSON.stringify(value, null, indent), `${label}, indent ${indent}`);
  }
  assert.equal(writeData(Object(12345678901234567890n)), "12345678901234567890", "a boxed bigint as its digits, as a bigint is");
  const shared = { a: 1 };
  assert.equal(writeData([shared, { again: shared }]), JSON.stringify([shared, { again: shared }]), "a value met twice, not inside itself, is written twice");
});

// Opus, round 5 (minor): a value that holds itself overflowed the stack; JSON.stringify refuses it with a TypeError.
test("the writer refuses a value that holds itself with JSON.stringify's TypeError, and a call given one sends nothing", async () => {
  const cyclic: Rec = { a: 1 };
  cyclic["self"] = [cyclic];
  assert.throws(() => writeData(cyclic), { name: "TypeError", message: /circular structure/ });
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const f = ai("cyclic", { input: { text: t.string() }, output: t.string(), router: router as never } as never);
  await assert.rejects(f({ text: cyclic } as never), { name: "TypeError", message: /circular structure/ });
  assert.equal(router.requests.length, 0);
});

test("arrays with holes given to a text input: the request, the call log line and a saved worked example hold null where JSON does, and load", async () => {
  for (const [label, value] of ARRAYS) {
    const where = folder();
    const router = new FakeRouter([], () => "<result>\nok\n</result>");
    const f = ai("sparse", { input: { text: t.string() }, output: t.string(), router: router as never, logCalls: where } as never);
    await f({ text: value } as never);
    assert.ok(texts(router.requests[0]!).includes(`<text>\n${JSON.stringify(value, null, 2)}\n</text>`), `${label}: ${texts(router.requests[0]!)}`);
    assert.ok(await flush());
    const [line] = logLines(where);
    assert.deepEqual(JSON.parse(line!)["inputs"]["text"], JSON.parse(JSON.stringify(value)), `${label}: the log line is JSON and holds the nulls: ${line}`);

    f.demos = [{ inputs: { text: value }, outputs: { result: "ok" } }] as never;
    const saved = folder();
    save(f, saved);
    const file = JSON.parse(readFileSync(join(saved, "functai.json"), "utf8"));   // JSON any reader takes
    const node = file["nodes"][file["entry"]]["ai"];
    assert.deepEqual(node["state"]["demos"][0]["inputs"]["text"], JSON.parse(JSON.stringify(value)), label);
    const loaded = load(saved).using({ router: router as never });
    assert.equal(loaded.version, f.version, label);
    await f({ text: "x" } as never);
    await loaded({ text: "x" } as never);
    assert.deepEqual(router.requests.at(-1), router.requests.at(-2), `${label}: the loaded copy sends the very request`);
    assert.ok(texts(router.requests.at(-1)!).includes(JSON.stringify(value, null, 2)), `${label}: the worked example is sent as JSON writes it`);
  }
});

test("boxed primitives given to a text input are sent and recorded as their values, nested ones too", async () => {
  for (const [label, value] of BOXED) {
    const where = folder();
    const router = new FakeRouter([], () => "<result>\nok\n</result>");
    const f = ai("boxed", { input: { text: t.string() }, output: t.string(), router: router as never, logCalls: where } as never);
    await f({ text: value } as never);
    const written = typeof value === "object" && value !== null && !(value instanceof String) ? JSON.stringify(value, null, 2) : String(value);
    assert.ok(texts(router.requests[0]!).includes(`<text>\n${written}\n</text>`), `${label}: ${texts(router.requests[0]!)}`);
    assert.ok(await flush());
    assert.deepEqual(JSON.parse(logLines(where)[0]!)["inputs"]["text"], JSON.parse(JSON.stringify(value)), label);
  }
});

// ------------------------------------------------------------------ what lm15 is given

// Opus, round 5 (B1): lmcc records member order under a symbol wherever an integer-like name follows another, and lm15
// refused an object with a symbol key, so a json-adapter function whose output shape came from lmcc.parseJson (or a
// saved folder) threw a TypeError at every render and call, and so did a tool with such parameters. The record is now
// lm15's (lmcc D-59): an lm15 that keeps member order sends it; one that does not (1.0.0-rc.2) is given plain copies.

/** `wanted` (a piece of the request lm15 writes) holds where lm15 keeps member order; else no record reaches lm15. */
function sentInOrder(request: Req, wanted: string, jsOrder: string): void {
  const text = stringifyJson(Request.toJSON(request));
  if (bridge.lm15KeepsOrder) assert.ok(text.includes(wanted), `sent in the value's order (${wanted}): ${text}`);
  else {
    assert.deepEqual(symbolKeys(request), []);
    assert.ok(text.includes(jsOrder), `an lm15 without the record sends JavaScript's order (${jsOrder}): ${text}`);
  }
}
const ORDERED_SHAPE = '{"type": "object", "properties": {"b": {"type": "integer"}, "10": {"type": "integer"}}, "required": ["b", "10"], "additionalProperties": false}';

test("an output shape whose members lmcc keeps in order: the json adapter renders, calls, saves and loads; the schema is sent in its order", async () => {
  const router = new FakeRouter([], () => '{"result": {"b": 1, "10": 2}}');
  const f = ai("shaped", { input: { q: t.string() }, output: lmcc.parseJson(ORDERED_SHAPE), adapter: "json", router: router as never } as never);
  const rendered = f.render({ q: "x" } as never);
  const answer = await f({ q: "x" } as never) as unknown as Rec;
  assert.deepEqual(lmcc.memberNames(answer), ["b", "10"], "the answer keeps the reply's order");
  const request = router.requests[0]!;
  assert.deepEqual(Request.toJSON(rendered), Request.toJSON(request));
  sentInOrder(request, '"properties":{"b":{"type":"integer"},"10":', '"properties":{"10":{"type":"integer"},"b":');
  const schema = (Request.toJSON(request) as Rec)["config"]["response_format"]["schema"]["properties"]["result"];
  assert.deepEqual(new Set(Object.keys(schema["properties"])), new Set(["b", "10"]));
  assert.deepEqual(schema["required"], ["b", "10"], "an array keeps its order");
  const where = folder();
  save(f, where);
  const loaded = load(where).using({ router: router as never });
  assert.equal(loaded.version, f.version);
  await loaded({ q: "x" } as never);
  assert.deepEqual(Request.toJSON(router.requests[1]!), Request.toJSON(request), "the loaded copy sends the very request");
  assert.equal(stringifyJson(Request.toJSON(router.requests[1]!)), stringifyJson(Request.toJSON(request)), "byte for byte, order included");
  // text FunctAI and lmcc write keep the order: the xml adapter's schema in the instructions
  const xml = new FakeRouter([], () => '<result>\n{"b": 1, "10": 2}\n</result>');
  await f.using({ adapter: "xml", router: xml as never })({ q: "x" } as never);
  const system = String(xml.requests[0]!.system);
  assert.ok(system.indexOf('"b"') < system.indexOf('"10"'), system);
});

test("a tool whose parameters lmcc keeps in order: rendered and called with native tool calling; the parameters are sent in their order", async () => {
  const seen: unknown[] = [];
  const look = tool("look", { input: { x: lmcc.parseJson(ORDERED_SHAPE) as never } }, (input) => { seen.push(input); return "found"; });
  const router = new FakeRouter([{ calls: [{ id: "c1", name: "look", input: { x: { b: 1, 10: 2 } } }] }, "<result>\ndone\n</result>"]);
  const f = ai("helper", { input: { q: t.string() }, output: t.string(), tools: [look], router: router as never } as never);
  f.render({ q: "x" } as never);
  assert.equal(await f({ q: "x" } as never), "done");
  assert.equal(router.requests.length, 2);
  for (const r of router.requests) sentInOrder(r, '"properties":{"b":{"type":"integer"},"10":', '"properties":{"10":{"type":"integer"},"b":');
  const params = (Request.toJSON(router.requests[0]!) as Rec)["tools"][0]["parameters"];
  assert.deepEqual(new Set(Object.keys(params["properties"]["x"]["properties"])), new Set(["b", "10"]));
  assert.equal(seen.length, 1);
});

test("a Config holding members lmcc keeps in order, given in settings or read from a saved folder, is sent in its order", async () => {
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const extensions = lmcc.parseJson('{"b": 1, "10": 2}');
  const f = ai("configured", { input: { q: t.string() }, output: t.string(), config: { extensions }, router: router as never } as never);
  await f({ q: "x" } as never);
  sentInOrder(router.requests[0]!, '"extensions":{"b":1,"10":2}', '"extensions":{"10":2,"b":1}');
  assert.deepEqual((Request.toJSON(router.requests[0]!) as Rec)["config"]["extensions"], { b: 1, 10: 2 });
  // a folder whose config another language wrote in the value's order ("b" before "10"), read with lmcc's parser
  const manifest = lmcc.parseJson(writeData(toManifest(f))) as Rec;
  manifest["nodes"][manifest["entry"]]["ai"]["config"] = lmcc.parseJson('{"extensions": {"b": 1, "10": 2}}');
  const loaded = fromManifest(manifest).using({ router: router as never });
  await loaded({ q: "x" } as never);
  assert.deepEqual(Request.toJSON(router.requests[1]!), Request.toJSON(router.requests[0]!));
  sentInOrder(router.requests[1]!, '"extensions":{"b":1,"10":2}', '"extensions":{"10":2,"b":1}');
});

test("a cached reply a store keeps as text is read by lm15: the tool call's input reaches the tool in the order written", async () => {
  const text = new Map<string, string>();
  // another writer kept the reply in the value's order; JSON.parse would have lost it
  const store = {
    get: (k: string) => text.get(k),
    set: (k: string, v: unknown) => { text.set(k, stringifyJson(v).replace('{"10":2,"b":1}', '{"b":1,"10":2}')); },
    delete: (k: string) => { text.delete(k); },
  };
  const seen: unknown[] = [];
  const look = tool("look", { input: { x: t.json() } }, (input) => { seen.push((input as Rec)["x"]); return "found"; });
  const router = new FakeRouter([], (_r, i) => i % 2 === 0 ? { calls: [{ id: "c1", name: "look", input: { x: { b: 1, 10: 2 } } }] } : "<result>\ndone\n</result>");
  const f = ai("cached-text", { input: { q: t.string() }, output: t.string(), tools: [look], cacheReplies: store, router: router as never } as never);
  assert.equal(await f({ q: "x" } as never), "done");
  assert.ok([...text.values()].some((v) => v.includes('{"b":1,"10":2}')), "the test's premise: the kept text holds that order");
  const calls = router.requests.length;
  assert.equal(await f({ q: "x" } as never), "done");
  assert.equal(router.requests.length, calls, "answered from the cache");
  assert.deepEqual(lmcc.memberNames(seen.at(-1) as object), bridge.lm15KeepsOrder ? ["b", "10"] : ["10", "b"]);
});

test("a cached reply a store hands back as lmcc parsed it (members in the order written) answers the call", async () => {
  const text = new Map<string, string>();
  // a store that keeps text another writer wrote, members in the value's order, and reads it with lmcc's parser
  const store = {
    get: (k: string) => text.has(k) ? lmcc.parseJson(text.get(k)!.replace('{"10":2,"b":1}', '{"b":1,"10":2}')) : undefined,
    set: (k: string, v: unknown) => { text.set(k, lmcc.jsonText(v)); },
    delete: (k: string) => { text.delete(k); },
  };
  const look = tool("look", { input: { x: t.json() } }, () => "found");
  const router = new FakeRouter([], (_r, i) => i % 2 === 0 ? { calls: [{ id: "c1", name: "look", input: { x: { b: 1, 10: 2 } } }] } : "<result>\ndone\n</result>");
  const f = ai("cached", { input: { q: t.string() }, output: t.string(), tools: [look], cacheReplies: store, router: router as never } as never);
  const warn = console.warn;
  const warned: string[] = [];
  console.warn = (m: string) => { warned.push(m); };
  try {
    assert.equal(await f({ q: "x" } as never), "done");
    assert.ok([...text.values()].some((v) => v.includes('{"10":2,"b":1}')), "the test's premise: a reply holds such a member");
    assert.equal(await f({ q: "x" } as never), "done");
  } finally {
    console.warn = warn;
  }
  assert.equal(router.requests.length, 2, "both replies came from the store");
  assert.deepEqual(warned, []);
});

// ------------------------------------------------------------------ integers past 2^53 in answers

// Opus, round 5 (S1): such an integer had no JSON form in the record: an integer answer was a description, and a JSON
// answer holding one was recorded as {"$type": "Object", "$repr": "[object Object]"}.
test("an integer past 2^53 in an answer is recorded as that integer: the call log line, the done event", async () => {
  for (const [label, shape, reply, recorded] of [
    ["integer", t.integer(), "12345678901234567890", '"outputs":{"result":12345678901234567890}'],
    ["json", t.json(), '{"id": 12345678901234567890, "b": 1, "10": 2}', '"outputs":{"result":{"id":12345678901234567890,"b":1,"10":2}}'],
  ] as const) {
    const where = folder();
    const seen: StreamEvent[] = [];
    const router = new FakeRouter([], () => `<result>\n${reply}\n</result>`);
    const f = ai("big", { input: { q: t.string() }, output: shape, router: router as never, logCalls: where, observers: [(e: StreamEvent) => { seen.push(e); }] } as never);
    const answer = await f({ q: "x" } as never);
    assert.ok(await flush());
    const [line] = logLines(where);
    assert.ok(line!.includes(recorded), `${label}: ${line}`);
    const done = (seen as Rec[]).find((e) => e["kind"] === "done")!;
    assert.equal(lmcc.jsonText(done["value"]), lmcc.jsonText(answer), label);
  }
});
