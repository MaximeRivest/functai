/**
 * Baking on Tinker from TypeScript, end to end against tools/fake_tinker.py
 * (whose model is a lookup table that learns): the plan and its price, each
 * method, checks choosing the checkpoint and temperature, pictures, resuming,
 * and the model called through `fn.using({ lm })`.
 */

import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deflateSync } from "node:zlib";
import { after, before, test } from "node:test";
import { ai, t } from "../src/index.ts";
import { bakeEntry, studentPlan } from "../src/bake.ts";
import { studentConversation } from "../src/students.ts";
import { bake, planBake, rewards, TinkerRunner } from "../src/train.ts";
import { Tinker } from "../src/tinker.ts";
import { criterion } from "../src/measures.ts";
import { passes } from "../src/schema.ts";
import { bakedId } from "../src/setup.ts";
import { CONTRACT } from "../tools/generate.ts";
import { fakeMissing, startFake, type Fake } from "./fake-tinker.ts";

type Rec = Record<string, any>;
const missing = fakeMissing();
const skip = missing ?? false;
const CASES = join(CONTRACT, "cases", "training");
let fake: Fake;
let home: string;
let files: string;            // a toy student, Qwen-shaped
let pictureFiles: string;     // a toy student of Inkling's family (pictures by the 40 px grid)
const opts = () => ({ tinker: { apiKey: "tml-test", baseUrl: fake.url, billingRetryMs: 20 }, log: false as const });

before(async () => {
  if (missing) return;
  home = mkdtempSync(join(tmpdir(), "functai-train-"));
  process.env["XDG_CACHE_HOME"] = join(home, "cache");
  process.env["FUNCTAI_PRICES"] = "off";
  const tok = JSON.parse(readFileSync(join(CASES, "toy-tokenizer.json"), "utf8")) as Rec;
  tok["added_tokens"].push({ id: 298, content: "<|unused_200054|>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true });
  files = join(home, "toy");
  mkdirSync(files, { recursive: true });
  writeFileSync(join(files, "tokenizer.json"), JSON.stringify(tok));
  writeFileSync(join(files, "chat_template.jinja"), readFileSync(join(CASES, "qwen-like.jinja"), "utf8"));
  pictureFiles = join(home, "inkling-toy");
  mkdirSync(pictureFiles, { recursive: true });
  writeFileSync(join(pictureFiles, "tokenizer.json"), JSON.stringify(tok));
  writeFileSync(join(pictureFiles, "chat_template.jinja"), readFileSync(join(CASES, "inkling-like.jinja"), "utf8").replace("<|pic|>", "<|unused_200054|>"));
  writeFileSync(join(pictureFiles, "processor_config.json"), JSON.stringify({ image_processor: { size: { width: 40, height: 40 } } }));
  const price = { prefill: "$0.30", cached_prefill: "$0.03", sample: "$1.00", train: "$1.50", context: "8K" };
  mkdirSync(join(home, "cache", "functai", "prices"), { recursive: true });
  writeFileSync(join(home, "cache", "functai", "prices", "tinker-models.json"), JSON.stringify([
    { tinker_id: "toy/Toy-1", ...price }, { tinker_id: "thinkingmachines/Inkling-Toy", ...price }]));
  fake = await startFake();
  const model = { vocab: 299, prefill: 0.3, cached: 0.03, sample: 1.0, train: 1.5, context: 8192, state_gb: 0.01, sampler_gb: 0.005 };
  await fake.config({ models: { "toy/Toy-1": { ...model, patch: null }, "thinkingmachines/Inkling-Toy": { ...model, patch: 40 } } });
});

after(() => fake?.stop());

const capital = ai("capital", { description: "The capital city of the country.", input: { country: t.string() }, output: t.string() });
const PAIRS = [["France", "Paris"], ["Japan", "Tokyo"], ["Peru", "Lima"], ["Kenya", "Nairobi"], ["Chile", "Santiago"]];
const rows = (n = 4) => PAIRS.flatMap(([country, result]) => Array.from({ length: n }, () => ({ country, result })));
const test5 = PAIRS.map(([country, result]) => ({ country, result }));

test("the plan prices the bake and spends nothing", { skip }, async () => {
  const before = (await fake.state())["sessions"];
  const plan = await planBake(capital, rows(), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder: join(home, "plan") });
  const text = String(plan);
  assert.match(text, /bake capital → toy\/Toy-1 on Tinker \(sft, LoRA r32\)/);
  assert.match(text, /20 train · 5 held out/);
  assert.ok(Number(plan.dollars["total"]) > 0, text);
  assert.match(text, /are sent to Thinking Machines/);
  assert.deepEqual((await fake.state())["sessions"], before, "no session, no request");
});

test("sft: the student learns the answers; the checks choose the step and the temperature; calls run on Tinker", { skip }, async () => {
  const folder = join(home, "sft");
  const model = await bake(capital, rows(), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder });
  const meta = model.meta as Rec;
  assert.ok(passes("baked", meta), "baked.json passes the contract's schema");
  assert.equal(meta["weights"]["service"], "tinker");
  assert.match(meta["weights"]["uri"], /sampler_weights\/best-step/);
  assert.match(meta["weights"]["state_uri"], /weights\/best-step-\d+-state$/, "the state of the weights it answers with");
  assert.equal(meta["template"]["source"], "files");
  assert.equal(meta["training"]["best"]["score"], 1);
  assert.equal(meta["sampling"]["temperature"], 0, "the lowest temperature wins a tie");
  const checks = meta["training"]["checks"] as Rec[];
  assert.equal(checks[0]["step"], 0);
  assert.ok(checks[0]["scores"]["0"] < 1, "untrained, it does not know");
  assert.equal(await capital.using({ lm: model as never })({ country: "Peru" }), "Lima");
  // records name it by its weights on Tinker (calls.md, Setups): two bakes, two names
  assert.deepEqual(capital.using({ lm: model as never }).setupDocument()["model"], { baked: bakedId(meta) });
  assert.notEqual(bakedId(meta), bakedId({ ...meta, weights: { ...meta["weights"], uri: meta["weights"]["uri"] + "x" } }));
  const metrics = readFileSync(join(folder, "metrics.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  const losses = metrics.filter((r) => r["loss"] !== undefined).map((r) => r["loss"]);
  assert.ok(losses.at(-1) < losses[0] / 3, `${losses}`);
  const st = await fake.state();
  const cks = Object.keys(st["checkpoints"]);
  assert.ok(!cks.some((p) => /weights\/step-/.test(p)), "intermediate states are deleted once the run ends");
  assert.equal(cks.filter((p) => /sampler_weights\/best-step/.test(p)).length >= 1, true);
  // the same bake again finds it done
  const again = await bake(capital, rows(), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder });
  assert.equal((again.meta as Rec)["weights"]["uri"], meta["weights"]["uri"]);
});

test("criteria decide by their own name, not by exact match on rows that know their answers", { skip }, async () => {
  const onlyLima = criterion("only_lima", { kind: "verdict", judge: ({ answer }) => answer === "Lima" });
  const model = await bake(capital, rows(), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, criteria: [onlyLima], folder: join(home, "criteria"), epochs: 2 });
  const tr = (model.meta as Rec)["training"];
  assert.ok(Math.abs(tr["best"]["score"] - 0.2) < 1e-9, "one case of five says Lima: the criterion's 0.2, not exact match's 1");
  assert.deepEqual(Object.keys(tr["criteria"]), ["only_lima"]);
});

test("a run that fails resumes from its last saved state", { skip }, async () => {
  const folder = join(home, "resume");
  const o = { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder, epochs: 4, evalEvery: 100 };
  // the fifth training request fails: steps 1 to 4 are done, a state was saved at step 2 (every quarter of 8 steps)
  const earlier = new Set(Object.keys((await fake.state())["runs"]));
  const failing = bake(capital, rows(), { ...o, batch: 8 });
  const watch = setInterval(async () => {
    const st = await fake.state();
    const run = Object.entries(st["runs"]).find(([id, r]: [string, any]) => !earlier.has(id) && r["steps"] >= 4) as [string, Rec] | undefined;
    if (run) {
      clearInterval(watch);
      await fake.config({ fail: { "/api/v1/forward_backward": 1 } });
    }
  }, 5);
  await assert.rejects(failing);
  clearInterval(watch);
  const run = JSON.parse(readFileSync(join(folder, "run.json"), "utf8"));
  assert.equal(run["state"], "failed");
  assert.ok(run["states"].length >= 1, "a state was saved before the failure");
  const model = await bake(capital, rows(), { ...o, batch: 8 });
  const done = JSON.parse(readFileSync(join(folder, "run.json"), "utf8"));
  assert.equal(done["state"], "done");
  assert.equal((model.meta as Rec)["training"]["steps"], 8);
  const st = await fake.state();
  assert.ok(Object.values(st["runs"]).some((r: any) => /weights\/step-/.test(r["loaded_from"] ?? "")), "the second run loaded the saved state");
});

test("a run asked to stop saves its state at the end of the step and goes on from that step when baked again", { skip }, async () => {
  const folder = join(home, "stop");
  const o = { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder, epochs: 4, evalEvery: 100, batch: 8 };
  // 8 steps, a state every 2: asked to stop during step 3, it saves step 3 (not only step 2) and stops there
  const stop = new AbortController();
  const earlier = new Set(Object.keys((await fake.state())["runs"]));
  const watch = setInterval(async () => {
    const run = Object.entries((await fake.state())["runs"]).find(([id, r]: [string, any]) => !earlier.has(id) && r["steps"] >= 2) as [string, Rec] | undefined;
    if (run) { clearInterval(watch); stop.abort(); }
  }, 5);
  await assert.rejects(bake(capital, rows(), { ...o, signal: stop.signal }), (e: any) => e.code === "bake-stopped" && /stopped after step \d of 8/.test(e.message));
  clearInterval(watch);
  const run = JSON.parse(readFileSync(join(folder, "run.json"), "utf8"));
  assert.equal(run["state"], "stopped");
  assert.equal(run["states"].at(-1)["step"], run["step"], "the state of the very step it stopped after is saved");
  assert.ok(run["step"] < 8);
  const stoppedAt = run["step"];
  const model = await bake(capital, rows(), o);
  assert.equal(JSON.parse(readFileSync(join(folder, "run.json"), "utf8"))["state"], "done");
  assert.equal((model.meta as Rec)["training"]["steps"], 8);
  const lines = readFileSync(join(folder, "metrics.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((x) => x["loss"] !== undefined);
  assert.deepEqual(lines.map((x) => x["step"]), [1, 2, 3, 4, 5, 6, 7, 8].filter((n) => n <= stoppedAt || n > stoppedAt), "no step trained twice");
  assert.equal(lines.length, 8, "each step once");
});

test("spent: what Tinker billed this run's sessions, not the account's other work; none while its records are not in", { skip }, async () => {
  const folder = join(home, "billed");
  const model = await bake(capital, rows(), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder, epochs: 1 });
  const spent = (model.meta as Rec)["training"]["spent"];
  const sessions: string[] = JSON.parse(readFileSync(join(folder, "run.json"), "utf8"))["sessions"];
  assert.equal(sessions.length, 1);
  const usage: Rec[] = (await fake.state())["usage"];
  const ours = usage.filter((u) => sessions.includes(u["session_id"])).reduce((a, u) => a + u["estimated_cost_usd"], 0);
  const all = usage.reduce((a, u) => a + u["estimated_cost_usd"], 0);
  assert.ok(ours > 0 && ours < all, "earlier tests' bakes spent on the same account");
  assert.ok(Math.abs(spent["billed"] - ours) < 1e-12, `${spent["billed"]} vs ${ours}`);
  assert.match(spent["billed_through"], /^\d{4}-\d\d-\d\dT\d\d:00:00Z$/);
  await fake.config({ billing_lag: true });
  try {
    const later = await bake(capital, rows(), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder: join(home, "billed-later"), epochs: 1 });
    const s2 = (later.meta as Rec)["training"]["spent"];
    assert.equal(s2["billed"], null, "not billed yet is not $0");
    assert.equal(s2["billed_through"], null);
    assert.ok(s2["estimated"] > 0);
  } finally {
    await fake.config({ billing_lag: false });
  }
});

test("a check whose calls fail in Tinker stops the bake with Tinker's error: a failure is not a score of 0", { skip }, async () => {
  const folder = join(home, "check-fails");
  await fake.config({ fail: { "/api/v1/asample": 1 } });             // the first check's first call
  try {
    await assert.rejects(bake(capital, rows(), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder }),
      (err: Error) => err.name === "TinkerError" && /failed on purpose/.test(err.message));
  } finally {
    await fake.config({ fail: {} });
  }
  assert.equal(JSON.parse(readFileSync(join(folder, "run.json"), "utf8"))["state"], "failed");
});

test("rl: a reward moves the student towards rewarded answers; groups that teach nothing are skipped", { skip }, async () => {
  // Peru's answer split between Lima and Cusco; checked on France, which it learns: the best check is a trained one
  const mixed = [...Array.from({ length: 4 }, () => ({ country: "Peru", result: "Lima" })), ...Array.from({ length: 4 }, () => ({ country: "Peru", result: "Cusco" })),
    ...Array.from({ length: 4 }, () => ({ country: "France", result: "Paris" }))];
  const start = await bake(capital, mixed, { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: [{ country: "France", result: "Paris" }], folder: join(home, "rl-start"), epochs: 4, batch: 4, lr: 4e-5, evalEvery: 100 });
  const tk = new Tinker({ apiKey: "tml-test", baseUrl: fake.url });
  const student = (start.runner as TinkerRunner).student;
  const plan = studentPlan(bakeEntry(capital as never));
  const { messages } = studentConversation(plan.render(plan.turn({ country: "Peru" })).request("student") as Rec);
  const prompt = student.render(messages).prompt;
  /** The share of 100 answers (seeded) that say Lima. */
  const share = async (uri: string) => {
    const r = await (await tk.sampler({ path: uri })).sample(prompt.map((c) => ({ tokens: (c as { text: number[] }).text })), { numSamples: 100, temperature: 1, maxTokens: 32, stop: [student.stopToken], seed: 7 });
    return r.sequences.filter((q) => student.readReply(q.tokens).includes("Lima")).length / 100;
  };
  const isLima = criterion("is_lima", { kind: "verdict", judge: ({ answer }) => answer === "Lima" });
  const rl = await bake(capital, [{ country: "Peru" }, { country: "Peru", result: "Lima" }], { ...opts(), method: "rl", start, reward: isLima, group: 8, epochs: 6, batch: 1,
    test: [{ country: "Peru", result: "Lima" }], folder: join(home, "rl"), temperatures: [1] });
  assert.equal((rl.meta as Rec)["training"]["method"], "rl");
  assert.deepEqual((rl.meta as Rec)["training"]["trained_on"], ["is_lima"]);
  const metrics = readFileSync(join(home, "rl", "metrics.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l));
  assert.ok(metrics.some((r) => typeof r["agreement"] === "number"), "how alike the answers are is measured at checks");
  const before = await share((start.meta as Rec)["weights"]["uri"]);
  const after = await share((rl.meta as Rec)["weights"]["uri"]);
  tk.close();
  assert.ok(before > 0 && before < 0.5, `the start answers Lima only sometimes (${before})`);
  assert.ok(after > before + 0.5, `Lima ${before} → ${after}`);
});

test("rl with a group reward (the energy score) runs, judged by its own score at checks", { skip }, async () => {
  const start = await bake(capital, rows(2), { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder: join(home, "energy-start"), epochs: 1, lr: 3e-5 });
  const distance = (a: unknown, b: unknown) => (a === b ? 0 : 1);
  const model = await bake(capital, rows(2), { ...opts(), method: "rl", start, reward: rewards.energy(distance), group: 4, epochs: 1, test: test5, folder: join(home, "energy"), temperatures: [1] });
  const checks = (model.meta as Rec)["training"]["checks"] as Rec[];
  assert.ok(checks.length >= 2 && checks.every((c) => typeof c["scores"]["1"] === "number"));
});

test("prefer: DPO from pairs moves the student to the chosen answer", { skip }, async () => {
  const mixed = [...Array.from({ length: 3 }, () => ({ country: "Peru", result: "Lima" })), ...Array.from({ length: 3 }, () => ({ country: "Peru", result: "Cusco" })),
    ...Array.from({ length: 3 }, () => ({ country: "France", result: "Paris" }))];
  const start = await bake(capital, mixed, { ...opts(), student: "toy/Toy-1", studentFiles: { dir: files }, test: [{ country: "France", result: "Paris" }], folder: join(home, "dpo-start"), epochs: 2, lr: 4e-5 });
  const pairs = Array.from({ length: 4 }, () => ({ inputs: { country: "Peru" }, chosen: { result: "Lima" }, rejected: { result: "Cusco" } }));
  const model = await bake(capital, [], { ...opts(), method: "prefer", start, pairs, beta: 1, lr: 5e-4, epochs: 3, batch: 2, test: [{ country: "Peru", result: "Lima" }], folder: join(home, "dpo") });
  const metrics = readFileSync(join(home, "dpo", "metrics.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((r) => r["loss"] !== undefined);
  assert.ok(metrics.at(-1)["loss"] < metrics[0]["loss"], `${metrics.map((r) => r["loss"])}`);
  assert.equal(await capital.using({ lm: model as never, temperature: 0 } as never)({ country: "Peru" }), "Lima");
});

test("decision: one forward pass gives each answer's probability; calls answer the likeliest", { skip }, async () => {
  const mood = ai("mood", { description: "The mood of the message.", input: { text: t.string() }, output: t.choice(["happy", "sad"]) });
  const data = [["What a lovely day", "happy"], ["I lost my keys", "sad"], ["We won!", "happy"], ["It rained all week", "sad"]]
    .flatMap(([text, result]) => Array.from({ length: 3 }, () => ({ text, result })));
  const model = await bake(mood, data, { ...opts(), method: "decision", student: "toy/Toy-1", studentFiles: { dir: files }, test: data.slice(0, 12).filter((_, i) => i % 3 === 0), folder: join(home, "decision"), epochs: 3 });
  assert.deepEqual((model.meta as Rec)["training"]["decision"]["choices"], ["happy", "sad"]);
  assert.equal(await mood.using({ lm: model as never })({ text: "I lost my keys" }), "sad");
  assert.equal(await mood.using({ lm: model as never })({ text: "We won!" }), "happy");
  const { Request } = await import("@lm15/lm15");
  const req = Request.fromJSON({ model: "student", messages: [{ role: "user", parts: [{ type: "text", text: "<text>\nWe won!\n</text>\n" }] }] } as never);
  const probs = await (model.runner as TinkerRunner).probabilities(req);
  assert.ok(Math.abs(probs.reduce((a, p) => a + p.probability, 0) - 1) < 1e-9);
});

test("distill: on-policy against a teacher sharing the tokenizer; a teacher with another tokenizer is refused", { skip }, async () => {
  const model = await bake(capital, rows(1), { ...opts(), method: "distill", student: "toy/Toy-1", teacher: "toy/Toy-1", studentFiles: { dir: files }, test: test5, folder: join(home, "distill"), epochs: 1, batch: 5, group: 2, maxTokens: 16 });
  assert.equal((model.meta as Rec)["training"]["method"], "distill");
  const other = join(home, "other-tokenizer");
  mkdirSync(other, { recursive: true });
  const tok = JSON.parse(readFileSync(join(files, "tokenizer.json"), "utf8")) as Rec;
  tok["added_tokens"].push({ id: 299, content: "<|other|>", single_word: false, lstrip: false, rstrip: false, normalized: false, special: true });
  writeFileSync(join(other, "tokenizer.json"), JSON.stringify(tok));
  writeFileSync(join(other, "chat_template.jinja"), readFileSync(join(files, "chat_template.jinja"), "utf8"));
  await assert.rejects(bake(capital, rows(1), { ...opts(), method: "distill", student: "toy/Toy-1", teacher: "Qwen/Qwen3-8B", studentFiles: (id) => ({ dir: id === "toy/Toy-1" ? files : other }), test: test5, folder: join(home, "distill-bad"), epochs: 1, maxTokens: 16 }),
    /tokenizer is not/);
});

function png(w: number, h: number): string {
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
  const raw = Buffer.concat(Array.from({ length: h }, (_, y) => Buffer.concat([Buffer.from([0]), Buffer.alloc(w, (y * 7) % 256)])));
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]).toString("base64");
}

test("pictures: a function with a picture input trains with each picture's token count, and is called with it", { skip }, async () => {
  const where = ai("where", { description: "The corner the target is in.", input: { screen: t.image(), target: t.string() }, output: t.choice(["top", "bottom"]) });
  const screens = [[png(80, 41), "top"], [png(120, 40), "bottom"], [png(41, 81), "top"]] as const;
  const data = screens.flatMap(([data, result]) => Array.from({ length: 3 }, () => ({ screen: { media_type: "image/png", data }, target: "OK", result })));
  const model = await bake(where, data, { ...opts(), student: "thinkingmachines/Inkling-Toy", studentFiles: { dir: pictureFiles }, test: data.filter((_, i) => i % 3 === 0), folder: join(home, "pictures"), epochs: 3 });
  assert.equal((model.meta as Rec)["template"]["picture"]["rule"], "grid");
  assert.equal(await where.using({ lm: model as never })({ screen: { media_type: "image/png", data: screens[1][0] } as never, target: "OK" }), "bottom");
});
