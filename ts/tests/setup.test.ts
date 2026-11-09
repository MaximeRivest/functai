/**
 * The contract's setup/ cases (contract/calls.md, *Setups*): an AI
 * function's setup document and its hash, under settings, plugins and walls.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { keepPictures, limits, withSettings, type Settings } from "../src/index.ts";
import { cases, tsFunction, type Rec } from "./cases.ts";
import { bakedId } from "../src/setup.ts";

/** The contract's setting names, as TypeScript spells them. */
const NAMES: Record<string, string> = {
  max_tokens: "maxTokens", top_p: "topP", max_steps: "maxSteps", tool_errors: "toolErrors", on_unreadable: "onUnreadable",
  base_url: "baseUrl", api_retries: "apiRetries", api_max_wait: "apiMaxWait", cache_replies: "cacheReplies",
  log_content: "logContent", escalate_to: "escalateTo", escalate_below: "escalateBelow", code_id: "codeId",
};

const builtin = (p: Rec) => p.builtin === "limits"
  ? limits(p.args)
  : keepPictures(p.args.last ?? 3, { atOnce: p.args.at_once ?? 1 });

const setupCases = cases("setup");

for (const [name, c] of setupCases) {
  test(`setup case ${name}`, () => {
    const fn = tsFunction(c.definition);
    const settings = Object.fromEntries(Object.entries(c.settings as Rec).map(([k, v]) => [NAMES[k] ?? k, v])) as Settings;
    if (c.plugins.length) settings.plugins = c.plugins.map(builtin);
    withSettings(settings, () => {
      assert.deepEqual(fn.setupDocument(), c.expect.document);
      assert.equal(fn.setup, c.expect.setup);
    });
  });
}

test("the contract has setup cases", () => assert.ok(setupCases.length >= 12));

test("a baked model is named by its files; weights on a service by their URI too (calls.md, Setups)", () => {
  // sha256 of {"model/a.safetensors":"ab"} and of {"hashes":{},"uri":"tinker://run-1:train:0/sampler_weights/best-step-000004"}, written by hand
  assert.equal(bakedId({ hashes: { "model/a.safetensors": "ab" }, weights: { form: "merged", path: "model" } }), "sha256:1a84d571669514bafda4965197c2a4fd2f5a3740b380564d5681927ce738902d");
  const remote = { hashes: {}, weights: { form: "remote", service: "tinker", uri: "tinker://run-1:train:0/sampler_weights/best-step-000004" } };
  assert.equal(bakedId(remote), "sha256:5de301cc192f33bcbf9cefe4c2f64d1b20499863787fe9ffb43a0220e1f382ea");
  assert.notEqual(bakedId({ ...remote, weights: { ...remote.weights, uri: "tinker://run-2:train:0/sampler_weights/best-step-000004" } }), "sha256:5de301cc192f33bcbf9cefe4c2f64d1b20499863787fe9ffb43a0220e1f382ea");
});
