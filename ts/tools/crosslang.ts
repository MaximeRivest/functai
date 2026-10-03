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
import { spawnSync } from "node:child_process";
import { ai, load, LoadRefused, rate, rated, remote, serve, t } from "../src/index.ts";
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
  console.log(`  ok    ${name}: saved in Python, loaded here, same version and same request`);
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
const asked = spawnSync(python_, ["-c", `
import json, sys, functai
url = json.load(open(sys.argv[1]))["url"]
far = functai.remote(url)
with functai.configure(log_calls=sys.argv[2]):
    print(far(review="Lovely."))
`, join(work, "ts-served.json"), log], { encoding: "utf8" });
server.close();
assert.equal(asked.status, 0, asked.stderr);
assert.equal(asked.stdout.trim(), "happy", asked.stdout);
console.log("  ok    a program TypeScript serves is called from Python with remote: one call tree across the two logs");
