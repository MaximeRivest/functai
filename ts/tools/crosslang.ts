/**
 * The TypeScript half of ../../tools/crosslang.py (run by it, with its
 * working folder): load what Python saved, define the same function here,
 * log and rate into the same folder, and read the ratings back.
 */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Request, stringifyJson } from "@lm15/lm15";
import * as lmcc from "lmcc";
import { spawn } from "node:child_process";
import { ai, describe, load, LoadRefused, rate, rated, remote, save, serve, t } from "../src/index.ts";
import { artifacts } from "../src/files.ts";
import { FakeRouter } from "../tests/fake.ts";

const work = process.argv[2]!;
const python = JSON.parse(readFileSync(join(work, "python.json"), "utf8")) as Record<string, { version: string; signature: string; request: unknown; inputs: Record<string, unknown> }>;

// 1. what Python saved loads here, and sends the same bytes
for (const [name, want] of Object.entries(python)) {
  if (name === "rounded") {
    assert.throws(() => load(join(work, "saved", name)), (e: unknown) => e instanceof LoadRefused && e.code === "saved-code");
    console.log(`  ok    ${name}: refused (saved-code: it runs Python code of its own)`);
    continue;
  }
  const fn = load(join(work, "saved", name));
  assert.equal(fn.version, want.version, `${name}: version`);
  assert.equal(fn.signatureId, want.signature, `${name}: signature`);
  const mine = lmcc.canonicalJson(JSON.parse(stringifyJson(Request.toJSON(await fn.render(want.inputs)))));
  assert.equal(mine, lmcc.canonicalJson(want.request as lmcc.Json), `${name}: the request`);
  if (name === "route") {                  // the settings saved with it travel too (its address, its rule)
    assert.equal(fn.settings.baseUrl, "http://127.0.0.1:8000/v1");
    assert.equal(fn.settings.onUnreadable, "record");
  }
  console.log(`  ok    ${name}: saved in Python, loaded here, same version and same request${name === "route" ? ", baseUrl and onUnreadable" : ""}`);
}

// 6. a function reading files (Attachments' artifacts), saved in Python: loads here, and sends the same bytes
const pythonFiles = JSON.parse(readFileSync(join(work, "python-files.json"), "utf8")) as typeof python;
for (const [name, want] of Object.entries(pythonFiles)) {
  const fn = load(join(work, "saved", name));
  assert.equal(fn.version, want.version, `${name}: version`);
  assert.equal(fn.signatureId, want.signature, `${name}: signature`);
  const mine = lmcc.canonicalJson(JSON.parse(stringifyJson(Request.toJSON(await fn.render(want.inputs)))));
  assert.equal(mine, lmcc.canonicalJson(want.request as lmcc.Json), `${name}: the request`);
  console.log(`  ok    ${name}: reads files, saved in Python, loaded here, same version and same request (each page's picture a picture)`);
}

// 2. the same function, written here
const mood = ai("mood", {
  description: "How does the customer feel about what they bought?", definedIn: "shop",
  input: { review: t.string() }, output: t.enum("happy", "unhappy", "mixed"), temperature: 0, lm: "gpt-4.1-mini",
  router: new FakeRouter([], () => "<result>\nmixed\n</result>"), logCalls: join(work, "log"),
});
assert.equal(mood.version, python["mood"]!.version);
assert.equal(mood.signatureId, python["mood"]!.signature);
console.log("  ok    mood written in TypeScript has Python's version and signature");
const route = ai("route", {
  description: "Which team answers?", definedIn: "shop", lm: "gpt-4.1-mini", baseUrl: "http://127.0.0.1:8000/v1", onUnreadable: "record",
  input: {
    message: describe(t.string(), "the customer's words"),
    channel: t.choice(["email", "chat"], { email: "written at leisure, often long", chat: "short, often several messages" }),
  },
  outputs: {
    urgency: describe(t.choice(["low", "high"], { high: "the customer is losing money or time now" }), "how soon to answer"),
    result: t.choice(["shipping", "billing"], { shipping: "where the parcel is, damage in transit", billing: "charges, refunds" }),
  },
});
assert.equal(route.version, python["route"]!.version, "route: version");
assert.equal(route.signatureId, python["route"]!.signature, "route: signature");
console.log("  ok    route written in TypeScript (t.choice, each answer described) has Python's version and signature");

// 6. a function reading files that Python serves, called here: the files travel as their wire form, a failed one
// is refused here before it is sent
const filesServed = JSON.parse(readFileSync(join(work, "served-files.json"), "utf8")) as { url: string };
const summarizeThere = await remote(filesServed.url);
assert.equal(await summarizeThere({ report: pythonFiles["summarize"]!.inputs["report"] } as never), "mixed");
await assert.rejects(summarizeThere({ report: [{ text: "", images: [], audio: [], video: [],
  meta: { source: "scan.pdf", error: { code: "missing-dependency", message: "needs pypdf" } } }] } as never),
(e: unknown) => (e as { code?: string }).code === "interface-input");
console.log("  ok    a function reading files Python serves is called here: the files as their wire form, a failed file refused before sending");

// 6. a picture and files, written here: Python's versions and signatures
const colour = ai("colour", { description: "The main colour of the picture.", definedIn: "shop", lm: "gpt-4.1-mini", temperature: 0,
  input: { picture: t.image() }, output: t.string() });
assert.equal(colour.version, python["colour"]!.version, "colour: version");
assert.equal(colour.signatureId, python["colour"]!.signature, "colour: signature");
const summarize = ai("summarize", { description: "The report in three sentences.", definedIn: "shop", lm: "gpt-4.1-mini", temperature: 0,
  input: { report: artifacts() }, output: t.string() });
assert.equal(summarize.version, pythonFiles["summarize"]!.version, "summarize: version");
assert.equal(summarize.signatureId, pythonFiles["summarize"]!.signature, "summarize: signature");
console.log("  ok    colour (t.image()) and summarize (artifacts()) written in TypeScript have Python's versions and signatures");

// 5. what TypeScript saves, Python loads (it checks the version and the request this gives)
const savedHere: Record<string, unknown> = {};
for (const [name, fn] of [["mood", mood], ["route", route], ["colour", colour], ["summarize", summarize]] as const) {
  save(fn as never, join(work, "typescript-saved", name));
  const inputs = (python[name] ?? pythonFiles[name])!.inputs;
  savedHere[name] = { version: fn.version, inputs, request: JSON.parse(stringifyJson(Request.toJSON(await fn.render(inputs as never)))) };
}
writeFileSync(join(work, "typescript-saved.json"), stringifyJson(savedHere));

// 3. one log: log and rate here, then read Python's calls and ratings with ours
const p = await mood.predict("Arrived broken, but support was great.");
rate(p, "wrong", { answer: "unhappy", by: "ben", folder: join(work, "log") });
const { rows } = rated(mood, { folder: join(work, "log") });
writeFileSync(join(work, "typescript-rated.json"), JSON.stringify(rows));

// 4. stages 1.2 to 5 with Python: continue Python's conversation, answer from Python's disk cache, call Python's
// server; serve a program Python calls
const log = join(work, "log");
const tutor = ai("tutor", { description: "Tutor.", definedIn: "shop", input: { message: t.string() }, output: t.string(), lm: "gpt-4.1-mini",
  router: new FakeRouter([], (r) => `<result>\ntypescript ${r.messages.length}\n</result>`), logCalls: log });
const chat = tutor.conversation("lesson-ts", { store: join(work, "conversations") });
const answer = await chat("Is it 5/6?");
assert.equal(answer, "typescript 5", `the third turn saw both of Python's: ${answer}`);
rate((await chat.turns()).at(-1)!.id, "right", { by: "ana", folder: log });
console.log("  ok    a conversation Python started in a folder continues here: the third turn saw Python's two");

const noCalls = { resolve: (m: string) => ({ provider: "openai", model: m }), complete: async () => { throw new Error("typescript: the request should have been answered from Python's disk cache"); } };
const kept = await mood.using({ router: noCalls as never, cacheReplies: join(work, "replies.sqlite"), logCalls: false })("Kept for later.");
assert.equal(kept, "unhappy", `the cached reply: ${kept}`);
console.log("  ok    a reply Python kept in the disk cache answers the same request here, with no model call");

const served = JSON.parse(readFileSync(join(work, "served.json"), "utf8")) as { url: string };
const far = await remote(served.url);
const got = await far.using({ logCalls: log })({ review: "I was charged twice." });
assert.equal(got, "mixed", `the served answer: ${got}`);
console.log("  ok    a program Python serves is called here with remote (logged here, kind remote)");

const server = await serve(mood.using({ router: new FakeRouter([], () => "<result>\nhappy\n</result>") as never, logCalls: log }), { port: 0 });
writeFileSync(join(work, "ts-served.json"), JSON.stringify({ url: `http://127.0.0.1:${(server.address() as { port: number }).port}` }));
const python_ = join(import.meta.dirname, "..", "..", "python", ".venv", "bin", "python");
// not spawnSync: the server answers on this process's event loop while Python asks
const asked = await new Promise<{ status: number | null; stdout: string; stderr: string }>((resolve) => {
  const child = spawn(python_, ["-c", `
import json, sys, functai
url = json.load(open(sys.argv[1]))["url"]
far = functai.remote(url)
with functai.configure(log_calls=sys.argv[2]):
    print(far(review="Lovely."))
`, join(work, "ts-served.json"), log]);
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (d) => { stdout += d; });
  child.stderr.on("data", (d) => { stderr += d; });
  child.on("close", (status) => resolve({ status, stdout, stderr }));
});
server.close();
assert.equal(asked.status, 0, asked.stderr);
assert.equal(asked.stdout.trim(), "happy", asked.stdout);
console.log("  ok    a program TypeScript serves is called from Python with remote: one call tree across the two logs");
