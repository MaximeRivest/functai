/**
 * contract/cases/training/: a student's tokens from its files, pictures,
 * reading a reply, training examples, advantages, the energy score, losses
 * as weights, choosing. The same cases Python passes.
 */

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { CONTRACT } from "../tools/generate.ts";
import { BakeError } from "../src/errors.ts";
import { gridTokens, picture, pictureSize, qwenTokens, Student, type PictureRule } from "../src/students.ts";
import { advantages, choose, decisionWeights, dpo, energy, klAdjusted, rlDatum, sftDatum } from "../src/training.ts";

type Rec = Record<string, any>;
const DIR = join(CONTRACT, "cases", "training");
const read = (name: string) => JSON.parse(readFileSync(join(DIR, name), "utf8")) as Rec;
const files = readdirSync(DIR).filter((f) => f.endsWith(".json") && f !== "toy-tokenizer.json").sort();

/** Numbers equal to 1e-9, structures otherwise exactly. */
function close(actual: unknown, expected: unknown, where = ""): void {
  if (typeof expected === "number" && typeof actual === "number") {
    assert.ok(Math.abs(actual - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), `${where}: ${actual} != ${expected}`);
  } else if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual) && actual.length === expected.length, `${where}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
    expected.forEach((e, i) => close((actual as unknown[])[i], e, `${where}[${i}]`));
  } else if (expected && typeof expected === "object") {
    assert.ok(actual && typeof actual === "object", `${where}: ${JSON.stringify(actual)}`);
    assert.deepEqual(Object.keys(actual as object).sort(), Object.keys(expected).sort(), where);
    for (const [k, v] of Object.entries(expected)) close((actual as Rec)[k], v, `${where}.${k}`);
  } else {
    assert.deepEqual(actual, expected, where);
  }
}

const toy = read("toy-tokenizer.json");
const student = (template: string, rule: PictureRule | null, effort = 0) =>
  Student.fromFiles("toy", { tokenizer: toy, template: readFileSync(join(DIR, template), "utf8"), picture: rule }, { effort });

test("the training cases are all read", () => {
  assert.ok(files.length >= 19, `${files.length} cases`);
});

for (const f of files.filter((x) => x.startsWith("tokens-"))) {
  test(`training/${f}`, async () => {
    const c = read(f);
    const st = await student(c["template"], c["picture"], c["effort"]);
    const pics = (c["pictures"] as string[]).map((b) => picture(new Uint8Array(Buffer.from(b, "base64"))));
    if (c["expect"]["refuses"]) {
      assert.throws(() => st.render(c["messages"], pics, c["reply"]), (e: unknown) => e instanceof BakeError && (e as BakeError).code === c["expect"]["refuses"]);
      return;
    }
    const r = st.render(c["messages"], pics, c["reply"]);
    const e = c["expect"];
    assert.deepEqual(st.kwargs, e["kwargs"]);
    assert.equal(st.lead, e["lead"]);
    assert.equal(st.end, e["end"]);
    assert.equal(st.stopToken, e["stop_token"]);
    assert.equal(st.endHead, e["end_head"]);
    assert.deepEqual(r.prompt, e["prompt"]);
    assert.deepEqual(r.full, e["full"]);
    assert.equal(r.replyStart, e["reply_start"]);
    assert.equal(r.prefixFallback, e["prefix_fallback"] === true);
  });
}

test("training/pictures-01-sizes", () => {
  for (const p of read("pictures-01-sizes.json")["pictures"]) {
    assert.deepEqual(pictureSize(new Uint8Array(Buffer.from(p["data"], "base64"))), p["expect"]);
  }
});

test("training/pictures-02-grid", () => {
  const c = read("pictures-02-grid.json");
  for (const s of c["sizes"]) assert.equal(gridTokens(s["width"], s["height"], c["patch"]), s["expect"], JSON.stringify(s));
});

test("training/pictures-03-qwen", () => {
  const c = read("pictures-03-qwen.json");
  for (const s of c["sizes"]) {
    assert.equal(qwenTokens(s["width"], s["height"], c["patch"], c["merge"], c["min_pixels"], c["max_pixels"]), s["expect"], JSON.stringify(s));
  }
});

test("training/reply-01-reading", async () => {
  const c = read("reply-01-reading.json");
  for (const g of c["generated"]) {
    const st = await student(g["template"], null, c["effort"]);
    assert.equal(st.readReply(g["generated"]), g["expect"]);
  }
});

const datum = (d: { modelInput: unknown; lossFnInputs: unknown }) => ({ model_input: d.modelInput, loss_fn_inputs: d.lossFnInputs });

test("training/datums-01-sft and datums-02-sft-picture", () => {
  for (const f of ["datums-01-sft.json", "datums-02-sft-picture.json"]) {
    const c = read(f);
    close(datum(sftDatum(c["chunks"], c["reply_start"], c["weight"])), c["expect"], f);
  }
});

test("training/datums-03-rl", () => {
  const c = read("datums-03-rl.json");
  close(datum(rlDatum(c["prompt"], c["sampled"], c["logprobs"], c["advantage"])), c["expect"]);
});

test("training/advantages-01-groups", () => {
  for (const g of read("advantages-01-groups.json")["groups"]) close(advantages(g["rewards"], g["normalize"]), g["expect"], JSON.stringify(g["rewards"]));
});

test("training/advantages-02-kl", () => {
  const c = read("advantages-02-kl.json");
  close(klAdjusted(c["advantages"], c["deltas"], c["beta"]), c["expect"]);
});

test("training/energy-01", () => {
  for (const g of read("energy-01.json")["groups"]) close(energy(g["readable"], g["d_ref"], g["d_pair"]), g["expect"], JSON.stringify(g["readable"]));
});

test("training/weights-01-dpo", () => {
  const c = read("weights-01-dpo.json");
  close(dpo(c["pairs"], c["beta"]), c["expect"]);
});

test("training/weights-02-decisions", () => {
  for (const d of read("weights-02-decisions.json")["decisions"]) close(decisionWeights(d["logprobs"], d["answer"], d["target"]), d["expect"]);
});

test("training/choosing-01", () => {
  const c = read("choosing-01.json");
  close(choose(c["checks"]), c["expect"]);
});
