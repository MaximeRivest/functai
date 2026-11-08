/**
 * Training on Tinker for real, from TypeScript (costs a few cents).
 *
 *     export TINKER_API_KEY=tml-...
 *     cd ts && node --conditions=functai-source tools/tinker-live.ts          # plans and prices, nothing spent
 *     cd ts && node --conditions=functai-source tools/tinker-live.ts --yes    # do it
 *
 * First the client alone (the session, the models, one supervised step and a
 * sample: what proves Tinker takes this client's requests), then a small
 * function trained on Qwen3.5-4B, called, one rl step, a decision model;
 * every checkpoint made is deleted at the end.
 */
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ai, t } from "../src/index.ts";
import { criterion } from "../src/measures.ts";
import { Student } from "../src/students.ts";
import { bake, planBake } from "../src/train.ts";
import { Tinker, wireChunks, wireDatum } from "../src/tinker.ts";
import { sftDatum } from "../src/training.ts";

const STUDENT = process.env["TINKER_STUDENT"] ?? "Qwen/Qwen3.5-4B";
const spend = process.argv.includes("--yes");
if (!process.env["TINKER_API_KEY"]) throw new Error("set TINKER_API_KEY first");
const home = mkdtempSync(join(tmpdir(), "functai-tinker-live-"));

const capital = ai("capital", { description: "The capital city of the country, in one word.", input: { country: t.string() }, output: t.string() });
const mood = ai("mood", { description: "The mood of the message.", input: { text: t.string() }, output: t.choice(["happy", "sad"]) });
const pairs = [["France", "Paris"], ["Japan", "Tokyo"], ["Peru", "Lima"], ["Kenya", "Nairobi"], ["Chile", "Santiago"], ["Canada", "Ottawa"], ["Egypt", "Cairo"], ["Spain", "Madrid"]];
const rows = [...pairs, ...pairs].map(([country, result]) => ({ country, result }));
const test = [{ country: "Peru", result: "Lima" }, { country: "Japan", result: "Tokyo" }];
const moods = ([["What a lovely day", "happy"], ["I lost my keys", "sad"], ["We won the match!", "happy"], ["It rained all week", "sad"]] as const)
  .flatMap(([text, result]) => [0, 1, 2].map(() => ({ text, result })));
const exact = criterion("exact", { kind: "verdict", judge: ({ answer, expected }) => String(answer).trim() === expected });

const plan = await planBake(capital, rows, { student: STUDENT, test, criteria: [exact], epochs: 2, folder: join(home, "sft") });
const dplan = await planBake(mood, moods, { student: STUDENT, method: "decision", test: moods.slice(0, 4), epochs: 2, folder: join(home, "decision") });
console.log(String(plan), "\n");
console.log(`estimated: sft ≈ $${Number(plan.dollars["total"]).toFixed(3)}, decision ≈ $${Number(dplan.dollars["total"]).toFixed(3)} (rl adds a few cents)`);
if (!spend) {
  console.log("nothing spent; run with --yes to do it");
  process.exit(0);
}

// 1. the client alone: does Tinker take its requests?
const tk = new Tinker();
console.log("models:", (await tk.models()).length);
const st = await Student.load(STUDENT);
const r = st.render([{ role: "user", content: "Say hi." }], [], "hi");
const run = await tk.createModel(STUDENT, { rank: 8 });
const out = await run.forwardBackward([wireDatum(sftDatum(r.full!, r.replyStart))], "cross_entropy");
await run.optimStep({ learning_rate: 1e-5 });
const sampler = await run.sampler();
const sample = await sampler.sample(wireChunks(r.prompt, []), { maxTokens: 8, temperature: 0, stop: [st.stopToken] });
console.log("client: forward_backward", out.outputs.length, "datum; sampled", JSON.stringify(st.readReply(sample.sequences[0]!.tokens)));
tk.close();

// 2. bakes
const s1 = await plan.run();
console.log("sft:", await capital.using({ lm: s1 as never })({ country: "Peru" }), JSON.stringify((s1.meta as never as Record<string, unknown>)["training"]));
const rl = await bake(capital, rows, { method: "rl", start: s1, reward: exact, test, epochs: 1, batch: 4, group: 4, temperatures: [1], folder: join(home, "rl") });
console.log("rl:", await capital.using({ lm: rl as never })({ country: "Japan" }));
const d = await dplan.run();
console.log("decision:", await mood.using({ lm: d as never })({ text: "I lost my keys" }));
const cleanup = new Tinker();
let n = 0;
for (const m of [s1, rl, d]) for (const k of ["uri", "state_uri"]) {
  const p = ((m.meta as Record<string, Record<string, string>>)["weights"]!)[k];
  if (p) {
    await cleanup.deleteCheckpoint(p);
    n++;
  }
}
cleanup.close();
console.log(`deleted ${n} checkpoints; run folders in ${home}`);
