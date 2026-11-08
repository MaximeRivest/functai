/**
 * The Tinker client against tools/fake_tinker.py (held to the official
 * Python SDK): sessions, numbered requests, futures, protobuf, pictures,
 * sampling, checkpoints, storage and billing.
 */

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { deflateSync } from "node:zlib";
import { Tinker, TinkerError, wireDatum } from "../src/tinker.ts";
import { picture } from "../src/students.ts";
import { rlDatum, sftDatum } from "../src/training.ts";
import { fakeMissing, startFake, type Fake } from "./fake-tinker.ts";

const missing = fakeMissing();
let fake: Fake;
let tk: Tinker;

before(async () => {
  if (missing) return;
  fake = await startFake();
  tk = new Tinker({ apiKey: "tml-test", baseUrl: fake.url, billingRetryMs: 50, heartbeatMs: 200 });
});

after(() => {
  tk?.close();
  fake?.stop();
});

function png(w: number, h: number): Uint8Array {
  const crc = (b: Buffer) => {
    let c = ~0;
    for (const x of b) {
      c ^= x;
      for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
    }
    return ~c >>> 0;
  };
  const chunk = (kind: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const kd = Buffer.concat([Buffer.from(kind), data]);
    const c = Buffer.alloc(4);
    c.writeUInt32BE(crc(kd));
    return Buffer.concat([len, kd, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  const raw = Buffer.concat(Array.from({ length: h }, () => Buffer.concat([Buffer.from([0]), Buffer.alloc(w, 0x80)])));
  return new Uint8Array(Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]));
}

const PROMPT = [11, 12, 13, 14];
const REPLY = [21, 22, 23, 99];

test("a supervised loop learns, its steps sent in order without waiting between them", { skip: missing ?? false }, async () => {
  await fake.config({ try_again: 2 });
  const run = await tk.createModel("Qwen/Qwen3-8B", { rank: 8 });
  const d = wireDatum(sftDatum([{ text: [...PROMPT, ...REPLY] }], PROMPT.length));
  const losses: number[] = [];
  for (let step = 0; step < 6; step++) {
    const fb = run.forwardBackward([d], "cross_entropy");          // both sent before either is awaited
    const op = run.optimStep({ learning_rate: 1e-4 });
    const out = await fb;
    await op;
    const lp = out.outputs[0]!.logprobs as number[];
    losses.push(-lp.slice(PROMPT.length - 1).reduce((a, b) => a + b, 0));
  }
  assert.ok(losses.at(-1)! < losses[0]! / 4, `${losses}`);
  const sampler = await run.sampler();
  const r = await sampler.sample([{ tokens: PROMPT }], { numSamples: 3, maxTokens: 8, temperature: 0, stop: [99] });
  assert.deepEqual(r.sequences.map((s) => s.tokens), [REPLY, REPLY, REPLY]);
  assert.equal(r.sequences[0]!.stopReason, "stop");
  const lp = await sampler.logprobs([{ tokens: [...PROMPT, ...REPLY] }]);
  assert.ok(Number.isNaN(lp[0]!) && lp[PROMPT.length]! > -1, `${lp}`);
  const state = await run.saveState("s1");
  const weights = await run.saveForSampler("w1");
  assert.match(state, /^tinker:\/\/.+\/weights\/s1$/);
  const again = await tk.loadState(state, { optimizer: true });
  const out2 = await again.forwardBackward([d], "cross_entropy", { forwardOnly: true });
  const loss2 = -(out2.outputs[0]!.logprobs as number[]).slice(PROMPT.length - 1).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(loss2 - losses.at(-1)!) < losses[0]! / 4, `the state continues where it was (${loss2})`);
  const saved = await tk.sampler({ path: weights });
  assert.deepEqual((await saved.sample([{ tokens: PROMPT }], { maxTokens: 8, temperature: 0, stop: [99] })).sequences[0]!.tokens, REPLY);
  const st = await fake.state();
  const runState = st["runs"][run.modelId];
  assert.equal(runState["steps"], 6);
  assert.ok(st["runs"][again.modelId]["loaded_from"] === state);
  await fake.config({ try_again: 0 });
});

test("a reinforcement step: sampled answers with their log-probabilities and advantages", { skip: missing ?? false }, async () => {
  const run = await tk.createModel("Qwen/Qwen3-8B", { rank: 8 });
  const sampler = await run.sampler();
  const r = await sampler.sample([{ tokens: PROMPT }], { numSamples: 4, maxTokens: 4, temperature: 1, seed: 1 });
  assert.equal(r.sequences.length, 4);
  const data = r.sequences.map((s, i) => wireDatum(rlDatum([{ text: PROMPT }], s.tokens, s.logprobs!, i % 2 ? 1 : -1)));
  const out = await run.forwardBackward(data, "importance_sampling");
  await run.optimStep({ learning_rate: 1e-4 });
  assert.equal(out.outputs.length, 4);
  assert.ok(typeof out.metrics["loss:sum"] === "number");
  await assert.rejects(run.forwardBackward([wireDatum({ modelInput: [{ text: [1, 2] }], lossFnInputs: { target_tokens: [2, 3], weights: [1, 1], advantages: [0, 1], logprobs: [0, 0] } })], "importance_sampling"),
    (e: unknown) => e instanceof TinkerError && e.status === 400 && /weights/.test(e.detail));
});

test("pictures: the right token count is trained on; a wrong one is refused as Tinker refuses it", { skip: missing ?? false }, async () => {
  const run = await tk.createModel("thinkingmachines/Inkling-2-Small", { rank: 8 });
  const pic = picture(png(80, 41));
  const ok = sftDatum([{ text: [1, 2] }, { picture: 0, tokens: 4 }, { text: [3, 4, 5, 9] }], 9);
  assert.deepEqual(ok.lossFnInputs["weights"], [0, 0, 0, 0, 0, 0, 0, 0, 1]);
  const out = await run.forwardBackward([wireDatum(ok, [pic])], "cross_entropy");
  assert.equal((out.outputs[0]!.logprobs as number[]).length, 9);
  const bad = { modelInput: [{ text: [1, 2] }, { picture: 0, tokens: 6 }, { text: [3] }], lossFnInputs: { target_tokens: Array(9).fill(0), weights: Array(9).fill(0) } };
  await assert.rejects(run.forwardBackward([wireDatum(bad, [pic])], "cross_entropy"), (e: unknown) => e instanceof TinkerError && /token count mismatch/.test(e.detail));
});

test("a large batch goes in several requests, numbered in order", { skip: missing ?? false }, async () => {
  const run = await tk.createModel("Qwen/Qwen3-8B", { rank: 4 });
  const d = wireDatum(sftDatum([{ text: [1, 2, 3] }], 1));
  const out = await run.forwardBackward(Array(1100).fill(d), "cross_entropy");
  assert.equal(out.outputs.length, 1100);
  await run.optimStep({ learning_rate: 1e-4 });
  const st = await fake.state();
  assert.equal(st["runs"][run.modelId]["seq"], 3);            // two forward_backward requests, then the step
});

test("decision probabilities: target log-probabilities, row i scoring position i + 1", { skip: missing ?? false }, async () => {
  const sampler = await tk.sampler({ base: "Qwen/Qwen3-8B" });
  const rows = [[-1, -1], [-1, -1], [5, 6]];
  const r = await sampler.sample([{ tokens: [1, 2, 3, 5] }], { maxTokens: 1, targetPromptLogprobs: rows });
  assert.equal(r.targetPromptLogprobs!.length, 3);
  assert.ok(Number.isNaN(r.targetPromptLogprobs![0]![0]!));
  assert.ok(Math.abs(r.targetPromptLogprobs![2]![0]! - r.targetPromptLogprobs![2]![1]!) < 1e-6);    // untrained: equally likely
});

test("a payment Tinker has not seen yet (402) is waited out; checkpoints, storage and billing are read", { skip: missing ?? false }, async () => {
  await fake.config({ billing_blocked: 2 });
  const run = await tk.createModel("Qwen/Qwen3-8B", { rank: 4 });
  const path = await run.saveForSampler("keep", { ttlSeconds: 3600 });
  const cks = await tk.checkpoints();
  assert.ok(cks.some((c) => c.tinker_path === path && c.checkpoint_type === "sampler" && c.expires_at));
  await tk.setTtl(path, null);
  assert.equal((await tk.checkpoints()).find((c) => c.tinker_path === path)!.expires_at, null);
  const before = await tk.storage();
  await tk.deleteCheckpoint(path);
  const after = await tk.storage();
  assert.ok(after.gigabytes < before.gigabytes);
  const bill = await tk.billed(new Date(Date.now() - 3600_000));
  assert.ok(bill.total > 0 && Object.keys(bill.bySession).length >= 1);
  const st = await fake.state();
  assert.ok(Object.values(st["sessions"]).some((s: any) => s["heartbeats"] >= 1), "the session is kept alive");
});

test("no key: said before anything is sent", () => {
  assert.throws(() => new Tinker({ apiKey: "" }), /TINKER_API_KEY/);
  assert.throws(() => new Tinker({ apiKey: "sk-123" }), /not a Tinker key/);
});
