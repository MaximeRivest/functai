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
import { ai, load, LoadRefused, rate, rated, t } from "../src/index.ts";
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
