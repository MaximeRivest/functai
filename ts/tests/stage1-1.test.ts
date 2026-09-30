/**
 * Stage 1.1 (design/09): what the shared cases do not reach from TypeScript.
 * A default given as a function counts in the version by its code; a saved
 * folder keeps it; a rating with no person named is made under the account;
 * a record names the lmcc and lm15 that made it.
 */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import * as lmcc from "lmcc";
import { ai, configure, flush, fromManifest, module, rate, rated, t, toManifest } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

type Rec = Record<string, any>;
for (const k of ["FUNCTAI_CALLER", "FUNCTAI_LOG_CALLS", "FUNCTAI_LOG_CONTENT"]) delete process.env[k];
configure({ lm: "gpt-4.1-mini", logCalls: false });

const folder = () => mkdtempSync(join(tmpdir(), "functai-s11-"));
function lines(where: string): Rec[] {
  const out: Rec[] = [];
  for (const day of readdirSync(where)) {
    for (const f of readdirSync(join(where, day))) {
      for (const line of readFileSync(join(where, day, f), "utf8").split("\n")) if (line) out.push(JSON.parse(line));
    }
  }
  return out;
}

function plan(day: string, tone: string) {
  const today = () => day;
  return ai("plan", {
    description: "Plan the day.",
    input: { notes: t.string(), day: t.string({ default: () => today() } as never), tone: t.string({ default: tone }) },
    output: t.string(),
  });
}

test("a default given as a function counts by its code: the same version every day, another for another value", () => {
  const monday = plan("2026-09-28", "kind"), tuesday = plan("2026-09-29", "kind"), formal = plan("2026-09-28", "formal");
  assert.notEqual(monday.interface.inputs[1]!.shape["default"], tuesday.interface.inputs[1]!.shape["default"]);
  assert.equal(monday.version, tuesday.version);
  assert.notEqual(monday.version, formal.version);
  assert.equal(monday.interfaceId, formal.interfaceId);
});

test("a default given as a function is computed at each call that leaves the input out", async () => {
  let n = 0;
  const router = new FakeRouter([], () => "<result>\nok\n</result>");
  const f = ai("count", { input: { a: t.string(), n: t.integer({ default: () => ++n } as never) }, output: t.string(), router: router as never } as never);
  await f({ a: "x" } as never);
  await f({ a: "x" } as never);
  const sent = router.requests.map((r) => JSON.stringify(r.messages));
  assert.ok(sent[0]!.includes("<n>\\n2\\n</n>") && sent[1]!.includes("<n>\\n3\\n</n>"), sent.join(" | "));
});

test("a saved folder keeps a default's code, and the loaded function has the saved version", () => {
  const fn = plan("2026-09-30", "kind");
  const manifest = toManifest(fn) as Rec;
  const node = manifest.nodes[manifest.entry];
  assert.deepEqual(node.defaults, { day: { code: "today()" } });
  assert.equal(fromManifest(lmcc.parseJson(JSON.stringify(manifest))).version, fn.version);
});

test("a rating with no person named is made under the account; two such ratings on one call are both kept, disputed", async () => {
  const where = folder();
  const router = new FakeRouter([], () => "<result>\nbilling\n</result>");
  const team = ai("team", { input: { message: t.string() }, output: t.string(), router: router as never, logCalls: where } as never);
  const p = await team.predict({ message: "charged twice" } as never);
  assert.ok(await flush());
  const first = rate(p, "right", { folder: where });
  rate(p, "wrong", { answer: "shipping", folder: where });
  assert.ok(!("by" in first) && typeof first["account"] === "string");
  assert.equal(rate(p, "right", { by: "ana", folder: where })["by"], "ana");
  const { rows } = rated(team, { folder: where, by: undefined });
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!["disputed"], true);
});

test("a record names the lmcc and lm15 that made it", async () => {
  const where = folder();
  const m = module("m", { input: {}, output: t.string(), logCalls: where }, () => "ok");
  await m({});
  assert.ok(await flush());
  const [rec] = lines(where);
  assert.equal(rec!["process"]["lmcc"], lmcc.VERSION);
  assert.ok(typeof rec!["process"]["lm15"] === "string" && rec!["process"]["lm15"].length);
});
