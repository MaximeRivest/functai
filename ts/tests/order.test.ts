/**
 * Names are data and members keep their order, as in Python (lmcc D-58):
 * an input or an output named `__proto__` is a field like any other, in a
 * call, a worked example a bootstrap records, a saved folder; and every
 * JSON value keeps its members in the value's order, integer-like names
 * (`"10"`, which JavaScript lists first) included, in what is sent, what
 * the call log and the events hold, and what a saved folder holds. Through
 * public calls, with a fake model; the requests compared are those sent.
 */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as lmcc from "lmcc";
import {
  ai, bootstrapFewShot, configure, flush, fromManifest, labeledFewShot, load, MemoryStore, rate, rated, save, t, toManifest, type StreamEvent,
} from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

type Rec = Record<string, any>;
const P = "__proto__";
const sent = (r: unknown) => JSON.stringify(r);
const folder = () => mkdtempSync(join(tmpdir(), "functai-order-"));
/** The text of every line of a call log folder. */
function logText(where: string): string[] {
  const out: string[] = [];
  for (const day of readdirSync(where)) for (const f of readdirSync(join(where, day))) out.push(...readFileSync(join(where, day, f), "utf8").split("\n").filter(Boolean));
  return out;
}
/** Whether `text` holds `first` before `second`. */
const before = (text: string, first: string, second: string) => text.includes(first) && text.includes(second) && text.indexOf(first) < text.indexOf(second);
const ordered = () => lmcc.parseJson('{"b": 1, "10": 2}') as Rec;

// ------------------------------------------------------------------ __proto__ in worked examples a bootstrap records

// Both reviewers, round 4: the bootstrap recorded its turns through lmcc's Turn.toJSON, which wrote `out[k] = v`, so an
// input named __proto__ vanished from the learned example (Python keeps it), and the saved folder lost it too.
test("bootstrapFewShot keeps an input named __proto__ in the worked examples it records: sent, saved and loaded from disk, under another adapter", async () => {
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const f = ai("proto", { input: { [P]: t.string(), q: t.string() }, output: t.string(), router: router as never } as never);
  const row = JSON.parse('{"__proto__": "ROWP", "q": "QQ", "result": "ok"}');
  const g = await bootstrapFewShot(f, [row], { maxBootstrapped: 1, maxLabeled: 0 });
  const [demo] = g.demos as unknown as Rec[];
  assert.ok(Object.hasOwn(demo!["inputs"], P), JSON.stringify(demo));
  assert.equal(demo!["inputs"][P], "ROWP");
  assert.deepEqual(lmcc.memberNames(demo!["inputs"]), [P, "q"]);
  const given = JSON.parse('{"__proto__": "CALL", "q": "x"}');
  await g(given);
  const request = sent(router.requests.at(-1));
  assert.ok(request.includes("<__proto__>\\nROWP\\n</__proto__>") && request.includes("<__proto__>\\nCALL\\n</__proto__>"), request);
  const where = folder();
  save(g, where);
  assert.ok(readFileSync(join(where, "functai.json"), "utf8").includes('"__proto__": "ROWP"'));
  const loaded = load(where).using({ router: router as never });
  assert.equal(loaded.version, g.version);
  await loaded(given);
  assert.deepEqual(router.requests.at(-1), router.requests.at(-2), "the loaded copy sends the very request");
  for (const adapter of ["json", "chat"]) {
    const answer = adapter === "json" ? '{"result": "ok"}' : "[[ ## result ## ]]\nok\n\n[[ ## completed ## ]]";
    const other = new FakeRouter([], () => answer);
    await g.using({ adapter, router: other as never })(given);
    await loaded.using({ adapter, router: other as never })(given);
    assert.ok(sent(other.requests[0]).includes("ROWP"), `${adapter}: ${sent(other.requests[0])}`);
    assert.deepEqual(other.requests[1], other.requests[0], adapter);
  }
  const pred = await g.predict(given);
  assert.ok(Object.hasOwn(pred.turn.toJSON().inputs, P), "a prediction's turn keeps it in its JSON");
});

test("bootstrapFewShot keeps a member named __proto__ inside an output it records: sent under another adapter, saved and loaded", async () => {
  const value = JSON.parse('{"__proto__": "v", "a": "x"}');
  const router = new FakeRouter([], () => '{"result": {"__proto__": "v", "a": "x"}}');
  const f = ai("nested", { input: { q: t.string() }, output: t.json(), adapter: "json", router: router as never } as never);
  const g = await bootstrapFewShot(f, [{ q: "QQ", result: value }] as never, { maxBootstrapped: 1, maxLabeled: 0 });
  const [demo] = g.demos as unknown as Rec[];
  assert.ok(Object.hasOwn(demo!["outputs"]["result"], P), JSON.stringify(demo));
  const where = folder();
  save(g, where);
  const loaded = load(where);
  const xml = new FakeRouter([], () => "<result>\n{}\n</result>");
  await g.using({ adapter: "xml", router: xml as never })("x" as never);
  await loaded.using({ adapter: "xml", router: xml as never })("x" as never);
  assert.ok(sent(xml.requests[0]).includes('\\"__proto__\\": \\"v\\"'), sent(xml.requests[0]));
  assert.deepEqual(xml.requests[1], xml.requests[0]);
});

// ------------------------------------------------------------------ member order

test("an answer's members keep the reply's order, integer-like names included: the prediction, the call log line, the events, the journal", async () => {
  const where = folder();
  const store = new MemoryStore();
  const seen: StreamEvent[] = [];
  const router = new FakeRouter([], () => '{"result": {"b": 1, "10": 2}}');
  const f = ai("obj", { input: { q: t.string() }, output: t.json(), adapter: "json", router: router as never, logCalls: where,
    journal: { store }, observers: [(e: StreamEvent) => { seen.push(e); }] } as never);
  const answer = await f("x" as never) as unknown as Rec;
  assert.deepEqual(lmcc.memberNames(answer), ["b", "10"]);
  assert.ok(await flush());
  const [line] = logText(where);
  assert.ok(line!.includes('"outputs":{"result":{"b":1,"10":2}}'), line);
  const done = (events: readonly Rec[]) => events.find((e) => e["kind"] === "done")!;
  assert.deepEqual(lmcc.memberNames(done(seen as never)["value"]), ["b", "10"], "the observer's event");
  assert.deepEqual(lmcc.memberNames(done(store.events(store.trees()[0]!))["value"]), ["b", "10"], "the journal's event");
});

test("an input's members keep the value's order: in the request (JSON and text inputs), the record and the started event", async () => {
  const where = folder();
  const seen: StreamEvent[] = [];
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const f = ai("obj", { input: { data: t.json(), text: t.string() }, output: t.string(), router: router as never, logCalls: where, observers: [(e: StreamEvent) => { seen.push(e); }] } as never);
  await f({ data: ordered(), text: ordered() } as never);
  const request = sent(router.requests[0]);
  const [data, text] = request.split("<text>");
  assert.ok(before(data!, '\\"b\\": 1', '\\"10\\": 2'), request);
  assert.ok(before(text!, '\\"b\\": 1', '\\"10\\": 2'), request);
  assert.deepEqual(await f.render({ data: ordered(), text: ordered() } as never), router.requests[0]);
  await flush();
  const [line] = logText(where);
  assert.ok(line!.includes('"data":{"b":1,"10":2}'), line);
  assert.deepEqual(lmcc.memberNames((seen[0] as Rec)["inputs"]["data"]), ["b", "10"]);
});

test("worked examples keep their values' order: labeled and bootstrapped, sent, written to a saved folder, and sent the same once loaded", async () => {
  const router = new FakeRouter([], () => '{"result": {"b": 1, "10": 2}}');
  const f = ai("obj", { input: { data: t.json() }, output: t.json(), adapter: "json", router: router as never } as never);
  const rows = [lmcc.parseJson('{"data": {"y": 1, "3": 2}, "result": {"b": 1, "10": 2}}') as Rec];
  for (const g of [labeledFewShot(f, rows as never, { k: 1 }), await bootstrapFewShot(f, rows as never, { maxBootstrapped: 1, maxLabeled: 0 })]) {
    const xml = new FakeRouter([], () => "<result>\n{}\n</result>");
    const h = g.using({ adapter: "xml", router: xml as never });
    await h({ data: {} } as never);
    const request = sent(xml.requests[0]);
    assert.ok(before(request, '\\"y\\": 1', '\\"3\\": 2') && before(request, '\\"b\\": 1', '\\"10\\": 2'), request);
    const where = folder();
    save(h, where);
    const saved = readFileSync(join(where, "functai.json"), "utf8");
    assert.ok(before(saved, '"y": 1', '"3": 2') && before(saved, '"b": 1', '"10": 2'), saved);
    const loaded = load(where).using({ router: xml as never });
    assert.equal(loaded.version, h.version);
    await loaded({ data: {} } as never);
    assert.deepEqual(xml.requests[1], xml.requests[0], "the loaded copy sends the very request");
  }
});

test("rows read back from the call log keep their values' order", async () => {
  const where = folder();
  const router = new FakeRouter([], () => '{"result": {"b": 1, "10": 2}}');
  const f = ai("obj", { input: { data: t.json() }, output: t.json(), adapter: "json", router: router as never, logCalls: where } as never);
  const pred = await f.predict({ data: ordered() } as never);
  rate(pred, "right", { folder: where });
  const { rows } = rated(f, { folder: where }) as { rows: Rec[] };
  assert.deepEqual(lmcc.memberNames(rows[0]!["data"]), ["b", "10"]);
  assert.deepEqual(lmcc.memberNames(rows[0]!["result"]), ["b", "10"]);
});

// A structured clone or JSON.parse loses the order lmcc records (and JSON.stringify writes JavaScript's): in src/, data
// is copied, parsed and written by values.ts's copyData, parseData and writeData. The two lines below read lm15's own
// JSON (a cached reply, a config), which lm15 rebuilds into its own objects.
test("src/ copies, parses and writes data only through values.ts (copyData, parseData, writeData)", () => {
  const src = join(import.meta.dirname, "..", "src");
  // (the served form's own script runs in the caller's browser)
  const allowed = ["JSON.parse(stringifyJson(Config.toJSON(", "try{inputs[k]=JSON.parse(v)}"];
  const found: string[] = [];
  for (const file of readdirSync(src).filter((f) => f.endsWith(".ts") && f !== "values.ts")) {
    readFileSync(join(src, file), "utf8").split("\n").forEach((line, i) => {
      if (/^\s*(\*|\/\/)/.test(line)) return;
      if (/structuredClone\(|JSON\.parse\(|\bstringifyJson\((?!Config)/.test(line) && !allowed.some((a) => line.includes(a))) found.push(`${file}:${i + 1}: ${line.trim()}`);
    });
  }
  assert.deepEqual(found, []);
});

// Found comparing with Python (round 5): the loader rebuilt a saved signature without its fields' type names (`dict`,
// `str`), so its fingerprint was not the saved one, and a recorded turn (a Python bootstrap's) was written again from
// its values instead of replayed as the model wrote it: another request than Python's, and `saved-differs` on load.
test("a loaded function keeps its fields' saved type names: its signature's fingerprint is the saved one, and a recorded turn is replayed as it was written", async () => {
  for (const cot of [false, true]) {
    const f = ai("typed", { input: { data: t.json() }, output: t.json(), module: cot ? "cot" : "predict" } as never);
    const manifest = lmcc.parseJson(JSON.stringify(toManifest(f))) as Rec;
    const node = manifest["nodes"][manifest["entry"]]["ai"];
    for (const field of node["signature"]["fields"]) field["type"] = field["name"] === "reasoning" ? "str" : "dict";
    const saved = lmcc.signatureFingerprint(lmcc.signatureFromDict(node["signature"]));
    const outputs = lmcc.parseJson(cot ? '{"reasoning": "R", "result": {"b": 1, "10": 2}}' : '{"result": {"b": 1, "10": 2}}');
    const text = (cot ? "<reasoning>\nR\n</reasoning>\n" : "") + '<result>\n{"b": 1, "10": 2}\n</result>';
    node["state"]["demos"] = [{ signature: saved, inputs: { data: { y: 1 } }, steps: [{ kind: "model", outputs, message: { role: "assistant", parts: [{ type: "text", text }] } }], outputs }];
    node["fingerprints"] = { signature: saved, requests: [] };                 // its probes are not checked here: what it sends is
    delete node["version"];
    const loaded = fromManifest(manifest);
    assert.equal(lmcc.signatureFingerprint(loaded.signature), saved, `cot: ${cot}`);
    const request = sent(await loaded.render({ data: {} } as never));
    assert.ok(request.includes(JSON.stringify(text).slice(1, -1)), `the recorded reply, as written (cot: ${cot}): ${request}`);
  }
});
