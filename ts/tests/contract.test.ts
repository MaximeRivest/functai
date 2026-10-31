/**
 * The contract every language shares (../../contract), held by the
 * TypeScript implementation: the same cases Python passes.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import * as lmcc from "lmcc";
import { describeSaved, exactMatch, FilesUnavailable, fromManifest, interval, LoadRefused } from "../src/index.ts";
import { checkInputs, InterfaceError } from "../src/interface.ts";
import { ratedRows } from "../src/calllog.ts";
import { MODELS } from "../src/generated/contract.ts";
import { adjustSettings, capabilities, NO_STORAGE, PROBE, refusedSettings } from "../src/models.ts";
import { Request, stringifyJson } from "@lm15/lm15";
import { FakeRouter } from "./fake.ts";
import { REGISTRY, resolveAdapter } from "../src/layouts.ts";
import { generate, OUT, CONTRACT } from "../tools/generate.ts";

import { cases, tsFunction, type Rec } from "./cases.ts";
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
    for (const sent of c.sends ?? []) assert.deepEqual(sends(fn, sent.inputs), sent.expect, JSON.stringify(sent.inputs).slice(0, 200));
  });
}

/**
 * What a function sends for these inputs, rendered as its probe is (functions/ and files/ `sends`): its inputs
 * bound as a call binds them (programs.md), then the request; or the refusal binding gives, before any request.
 */
function sends(fn: ReturnType<typeof tsFunction>, inputs: Rec): Rec {
  let bound: Rec;
  try {
    bound = checkInputs(fn.interface, inputs, fn.name);
  } catch (err) {
    if (err instanceof InterfaceError) return { refuses: err.code, field: err.field };
    throw err;
  }
  return { request: JSON.parse(JSON.stringify(fn.probeRequest(bound))) };
}

test("the contract has function cases", () => assert.ok(cases("functions").length >= 10));

// ------------------------------------------------------------------ files

// Files (functions.md, "Pictures, sounds, documents and files") are read through Attachments (attachments-dev, plugged
// in by functai/files). With it, each case is defined here and loaded from the folder another language saved, and
// both send what the case says; without it, both refuse files-unavailable and send nothing.
const files = await import("../src/files.ts").catch(() => null);

for (const [name, c] of cases("files")) {
  test(`files case ${name}${files ? "" : " (without Attachments: files-unavailable)"}`, () => {
    const manifest = { ...c.manifest, language: "python" };
    if (!files) {
      assert.throws(() => fromManifest(manifest, { node: c.node }), (e: unknown) => e instanceof FilesUnavailable && e.code === "files-unavailable");
      assert.throws(() => tsFunction(c.definition), (e: unknown) => e instanceof FilesUnavailable);
      return;
    }
    const defined = tsFunction(c.definition);
    const loaded = fromManifest(manifest, { node: c.node });
    for (const fn of [defined, loaded]) {
      const want = { ...c.expect.signature, fields: c.expect.signature.fields.map((f: Rec) => ({ ...f, purpose: f.purpose ?? "plain" })) };
      assert.deepEqual(withoutType(fn.signature), want);
      assert.deepEqual(JSON.parse(JSON.stringify(fn.probeRequest(c.expect.sample))), c.expect.request);
      assert.equal(lmcc.sha256(fn.probeRequest(c.expect.sample)), c.expect.request_hash);
      assert.equal(fn.version, c.expect.version);
      assert.equal(fn.signatureId, c.expect.signature_id);
      for (const sent of c.sends) assert.deepEqual(sends(fn, sent.inputs), sent.expect, JSON.stringify(sent.inputs).slice(0, 200));
    }
  });
}

test("the contract has files cases", () => assert.ok(cases("files").length >= 2));

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
  test(`saved case ${name}`, async () => {
    const node = c.node ?? undefined;
    if (c.expect.describe.refuses) {
      assert.throws(() => describeSaved(c.manifest, { node }), (err: unknown) => err instanceof LoadRefused && err.code === c.expect.describe.refuses);
    } else {
      assert.deepEqual(describeSaved(c.manifest, { node }), c.expect.describe.interface);
    }
    if (c.expect.refuses) {
      assert.throws(() => fromManifest(c.manifest, { node }),
        (err: unknown) => err instanceof LoadRefused && err.code === c.expect.refuses);
      return;
    }
    const fn = fromManifest(c.manifest, { node });
    const want = c.expect.loads;
    assert.equal(fn.name, want.name);
    assert.equal(fn.module, want.module);
    assert.equal(fn.version, want.version);
    assert.equal(fn.signatureId, want.signature_id);
    const saved = c.manifest.nodes[c.node ?? c.manifest.entry].ai;
    assert.deepEqual(saved.probes.map((p: Rec) => lmcc.sha256(fn.probeRequest(p))), want.requests, "requests");
    for (const send of c.expect.sends ?? []) {
      // a real call of the loaded function, under the probe facts (no sampling: where a version runs is not what it sends)
      const router = new FakeRouter([], () => "<result>\nok\n</result>", "probe");
      await fn.using({ lm: "probe", router: router as never, capabilities: PROBE, temperature: null, maxTokens: null, topP: null, seed: null, logCalls: false })(send.inputs);
      const sent = JSON.parse(stringifyJson(Request.toJSON(router.requests[0]!)));
      assert.equal(lmcc.sha256(sent), send.request_hash, JSON.stringify(send.inputs));
    }
  });
}

test("a sampling the model does not take is left out, with one warning", () => {
  const warn = console.warn;
  const said: string[] = [];
  console.warn = (m: string) => { said.push(m); };
  try {
    assert.deepEqual(adjustSettings({ temperature: 1 }, "openai", "gpt-6-luna"), { temperature: 1, config: { store: false } });
    assert.deepEqual(adjustSettings({ temperature: 0, topP: 0.5 }, "openai", "gpt-6-luna"),
      { temperature: null, topP: null, config: { store: false } });
    adjustSettings({ temperature: 0, topP: 0.5 }, "openai", "gpt-6-luna");
    assert.equal(said.filter((m) => m.includes("does not take temperature, top_p")).length, 1);
    assert.equal(adjustSettings({ temperature: 0 }, "claude-code", "claude-sonnet-5").temperature, null);
    assert.deepEqual(adjustSettings({ temperature: 0 }, "anthropic", "claude-haiku-4-5"), { temperature: 0 });
    assert.deepEqual(refusedSettings("gemini", "gemini-3.8-flash"), []);
  } finally {
    console.warn = warn;
  }
});

test("a provider that keeps requests is asked not to, unless the settings say", () => {
  assert.deepEqual([...NO_STORAGE], [...MODELS.no_storage.providers]);
  assert.deepEqual(adjustSettings({}, "openai", "gpt-4.1-mini"), { config: { store: false } });
  assert.deepEqual(adjustSettings({ config: { store: true } }, "openai", "gpt-4.1-mini"), { config: { store: true } });
  assert.deepEqual(adjustSettings({ config: { seed: 1 } }, "openai", "gpt-4.1-mini"), { config: { seed: 1, store: false } });
  for (const provider of ["openai-chat", "anthropic", "gemini", "openai-codex", "claude-code"]) {
    assert.deepEqual(adjustSettings({}, provider, "m"), {});
  }
});
