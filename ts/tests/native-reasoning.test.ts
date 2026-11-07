/** A reasoning output on a model that thinks natively (contract/functions.md, *Reasoning*): the request asks for the thinking and its summary, and the function's own `reasoning` setting chooses the effort. */

import assert from "node:assert/strict";
import { test } from "node:test";
import { Message, Response, type Request as ModelRequest } from "@lm15/lm15";
import { ai, t } from "../src/index.ts";
import { capabilities } from "../src/models.ts";
import { FakeRouter } from "./fake.ts";

type Rec = Record<string, any>;

const thinker = () => new FakeRouter([], (request: ModelRequest) => new Response({
  model: request.model,
  message: Message.assistant([{ type: "thinking", text: "Twelve pens leave 14 for two notebooks." }, { type: "text", text: "<result>\n12\n</result>" }] as never),
  finishReason: "stop" as never, usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
}), "anthropic");
const solve = (router: FakeRouter, extra: Rec = {}) => ai("solve", { description: "Solve the puzzle.", input: { puzzle: t.string() }, output: t.number(),
  lm: "anthropic:claude-opus-5-5", module: "cot", router: router as never, ...extra });
const sent = (router: FakeRouter) => router.requests.at(-1)!.config?.reasoning;

test("Claude 5 thinks natively", () => {
  for (const model of ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-fable-5-1", "claude-mythos-5", "claude-opus-4-8", "claude-sonnet-4-6"]) {
    const caps = capabilities("anthropic", model);
    assert.ok(caps["native_reasoning"] && !caps["assistant_prefill"], model);
  }
});

test("the thinking and its summary are asked for, and read into the reasoning output", async () => {
  const router = thinker();
  const p = await (solve(router) as any).predict({ puzzle: "x" });
  assert.equal(p.outputs.reasoning, "Twelve pens leave 14 for two notebooks.");
  assert.equal(p.outputs.result, 12);
  assert.deepEqual(sent(router), { effort: "medium", summary: "auto" });
});

test("the function's own reasoning setting chooses the effort, never a conflict with the layout's", async () => {
  for (const [reasoning, wanted] of [
    [{ effort: "high" }, { effort: "high", summary: "auto" }],
    [{ effort: "low", summary: "detailed" }, { effort: "low", summary: "detailed" }],
    [{ effort: "high", thinkingBudget: 4096 }, { effort: "high", thinkingBudget: 4096, summary: "auto" }],
    [{ effort: "off" }, { effort: "off" }],
  ] as const) {
    const router = thinker();
    const f = solve(router, { config: { reasoning } });
    await (f as any).predict({ puzzle: "x" });
    assert.deepEqual(sent(router), wanted, JSON.stringify(reasoning));
    assert.deepEqual((await (f as any).render({ puzzle: "x" })).config?.reasoning, wanted);      // what render shows is sent
  }
});
