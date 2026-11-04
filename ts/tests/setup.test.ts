/**
 * The contract's setup/ cases (contract/calls.md, *Setups*): an AI
 * function's setup document and its hash, under settings, plugins and walls.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { keepPictures, limits, withSettings, type Settings } from "../src/index.ts";
import { cases, tsFunction, type Rec } from "./cases.ts";

/** The contract's setting names, as TypeScript spells them. */
const NAMES: Record<string, string> = {
  max_tokens: "maxTokens", top_p: "topP", max_steps: "maxSteps", tool_errors: "toolErrors", on_unreadable: "onUnreadable",
  base_url: "baseUrl", api_retries: "apiRetries", api_max_wait: "apiMaxWait", cache_replies: "cacheReplies",
  log_content: "logContent", escalate_to: "escalateTo", escalate_below: "escalateBelow",
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

test("the contract has setup cases", () => assert.ok(setupCases.length >= 11));
