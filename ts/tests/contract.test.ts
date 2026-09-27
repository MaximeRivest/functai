/**
 * The contract every language shares (../../contract), held by the
 * TypeScript implementation: the same cases Python passes.
 */

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import * as lmcc from "lmcc";
import { ai, exactMatch, fromManifest, interval, LoadRefused, type Tool } from "../src/index.ts";
import { ratedRows } from "../src/calllog.ts";
import { adjustSettings, capabilities, refusedSettings } from "../src/models.ts";
import { REGISTRY, resolveAdapter } from "../src/layouts.ts";
import { generate, OUT, CONTRACT } from "../tools/generate.ts";

type Rec = Record<string, any>;
const cases = (folder: string): [string, Rec][] =>
  readdirSync(join(CONTRACT, "cases", folder)).filter((f) => f.endsWith(".json")).sort()
    .map((f) => [f.replace(/\.json$/, ""), JSON.parse(readFileSync(join(CONTRACT, "cases", folder, f), "utf8"))]);
const read = (p: string) => JSON.parse(readFileSync(join(CONTRACT, p), "utf8"));

test("the contract's data in src/generated is up to date", () => {
  assert.equal(readFileSync(OUT, "utf8"), generate(), "run: node tools/generate.ts");
});

for (const name of ["xml", "chat", "json"]) {
  test(`the ${name} layout is the contract's`, () => {
    const mine = lmcc.dump(resolveAdapter(name), REGISTRY) as Rec;
    const theirs = read(`layouts/${name}.json`);
    delete mine["versions"]["kernel"];
    delete theirs["versions"]["kernel"];
    assert.deepEqual(mine, theirs);
  });
}

test("capabilities follow the contract's table", () => {
  const table = read("models.json");
  for (const p of table.native.providers) {
    const caps = capabilities(p, "some-model");
    assert.equal(caps["stop_sequences"], !table.native.no_stop_sequences.includes(p));
    assert.equal(caps["native_function_calling"], true);
  }
  assert.equal(capabilities("anthropic", "claude-sonnet-4-5")["native_reasoning"], true);
  assert.equal(capabilities("anthropic", "claude-3-5-haiku")["assistant_prefill"], true);
  assert.deepEqual(capabilities("claude-code", "claude-sonnet-4-5"), capabilities("anthropic", "claude-sonnet-4-5"));
  assert.equal(capabilities("groq", "x")["native_function_calling"], true);
  assert.equal(capabilities("ollama", "x")["native_function_calling"], false);
  assert.deepEqual(capabilities("typesafe", "x"), { native_structured_output: true });
});

// ------------------------------------------------------------------ functions

/** A contract definition written the way a TypeScript user writes it. */
function tsFunction(d: Rec) {
  const field = (f: Rec) => (f.desc ? { shape: f.shape, desc: f.desc } : f.shape);
  const tools: Tool[] = (d.tools ?? []).map((t: Rec) => ({ ...t, run: () => "" }));
  const settings: Rec = {};
  if (d.settings.adapter) settings.adapter = d.settings.adapter;
  if (d.settings.module) settings.module = d.settings.module;
  if (d.settings.include_fn_name_in_instructions === false) settings.includeFnName = false;
  const fn = ai(d.name, {
    description: d.description,
    input: Object.fromEntries(d.inputs.map((f: Rec) => [f.name, field(f)])),
    outputs: Object.fromEntries(d.outputs.map((f: Rec) => [f.name, field(f)])),
    ...(tools.length ? { tools } : {}), ...settings,
  } as never);
  fn.loadState(d.state);
  return fn;
}

function withoutType(signature: lmcc.Signature): Rec {
  const data = lmcc.signatureToDict(signature) as Rec;
  return {
    instructions: data.instructions,
    fields: data.fields.map((f: Rec) => {
      const out: Rec = {};
      for (const [k, v] of Object.entries(f)) if (k !== "type" && v !== null) out[k] = v;
      out.purpose ??= "plain";
      return out;
    }),
  };
}

for (const [name, c] of cases("functions")) {
  test(`function case ${name}`, () => {
    const fn = tsFunction(c.definition);
    const want = { ...c.expect.signature, fields: c.expect.signature.fields.map((f: Rec) => ({ ...f, purpose: f.purpose ?? "plain" })) };
    assert.deepEqual(withoutType(fn.signature), want);
    const request = fn.probeRequest(c.expect.sample);
    assert.deepEqual(JSON.parse(JSON.stringify(request)), c.expect.request);
    assert.equal(lmcc.sha256(request), c.expect.request_hash);
    assert.equal(fn.version, c.expect.version);
    assert.equal(fn.signatureId, c.expect.signature_id);
  });
}

test("the contract has function cases", () => assert.ok(cases("functions").length >= 10));

// ------------------------------------------------------------------ scores

for (const [name, c] of cases("scores")) {
  test(`score case ${name}`, () => {
    if (c.kind === "interval") {
      const got = interval(c.values);
      for (const k of ["mean", "low", "high"] as const) {
        if (c.expect[k] === null) assert.equal(got[k], null, k);
        else assert.ok(Math.abs(got[k]! - c.expect[k]) < 1e-12, `${k}: ${got[k]} vs ${c.expect[k]}`);
      }
      return;
    }
    const keys = Object.keys(c.prediction).filter((k) => k in c.answers);
    const got = exactMatch(c.answers, Object.fromEntries(keys.map((k) => [k, c.prediction[k]])));
    assert.deepEqual(got, c.expect);
  });
}

// ------------------------------------------------------------------ rated

for (const [name, c] of cases("rated")) {
  test(`rated case ${name}`, () => {
    const calls = c.records.filter((r: Rec) => "functai_call" in r);
    const ratings = c.records.filter((r: Rec) => "functai_rating" in r);
    const [rows, left] = ratedRows(calls, ratings, c.rated);
    assert.deepEqual(rows, c.expect.rows);
    assert.deepEqual(left, c.expect.left_out);
  });
}

// ------------------------------------------------------------------ saved

for (const [name, c] of cases("saved")) {
  test(`saved case ${name}`, () => {
    if (c.expect.refuses) {
      assert.throws(() => fromManifest(c.manifest, { node: c.node ?? undefined }),
        (err: unknown) => err instanceof LoadRefused && err.code === c.expect.refuses);
      return;
    }
    const fn = fromManifest(c.manifest, { node: c.node ?? undefined });
    const want = c.expect.loads;
    assert.equal(fn.name, want.name);
    assert.equal(fn.module, want.module);
    assert.equal(fn.version, want.version);
    assert.equal(fn.signatureId, want.signature_id);
  });
}

test("a sampling the model does not take is left out, with one warning", () => {
  const warn = console.warn;
  const said: string[] = [];
  console.warn = (m: string) => { said.push(m); };
  try {
    assert.deepEqual(adjustSettings({ temperature: 1 }, "openai", "gpt-6-luna"), { temperature: 1 });
    assert.deepEqual(adjustSettings({ temperature: 0, topP: 0.5 }, "openai", "gpt-6-luna"), { temperature: null, topP: null });
    adjustSettings({ temperature: 0, topP: 0.5 }, "openai", "gpt-6-luna");
    assert.equal(said.filter((m) => m.includes("does not take temperature, top_p")).length, 1);
    assert.equal(adjustSettings({ temperature: 0 }, "claude-code", "claude-sonnet-5").temperature, null);
    assert.deepEqual(adjustSettings({ temperature: 0 }, "anthropic", "claude-haiku-4-5"), { temperature: 0 });
    assert.deepEqual(refusedSettings("gemini", "gemini-3.8-flash"), []);
  } finally {
    console.warn = warn;
  }
});
