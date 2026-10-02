/**
 * functai for TypeScript against real models (costs cents). Not run by `npm test`.
 *
 *     set -a; source ~/Projects/lm15-dev/.env; set +a
 *     node --conditions=functai-source tools/live.ts [model ...]
 */
import { ai, evaluate, t, tool } from "../src/index.ts";

const models = process.argv.slice(2).length ? process.argv.slice(2) : ["gpt-4.1-mini", "claude-haiku-4-5", "gemini:gemini-2.5-flash"];
let failed = 0;
const check = async (what: string, f: () => Promise<unknown>) => {
  const t0 = Date.now();
  try {
    const out = await f();
    console.log(`  ok    ${what} (${((Date.now() - t0) / 1000).toFixed(1)} s): ${JSON.stringify(out).slice(0, 120)}`);
  } catch (err) {
    failed++;
    console.log(`  FAIL  ${what}: ${(err as Error).name}: ${(err as Error).message.slice(0, 400)}`);
  }
};

for (const lm of models) {
  console.log(lm);
  const mood = ai("mood", { description: "How does the customer feel about what they bought?", input: { review: t.string() },
    output: t.enum("happy", "unhappy", "mixed"), lm, temperature: 0 });
  await check("a choice", () => mood("It broke after one day and support never answered."));
  await check("a record, json layout", () => ai("person", { description: "Who is described?", input: { text: t.string() },
    output: t.object({ name: t.string(), age: t.integer() }), adapter: "json", lm })("Ana turned 31 last week."));
  await check("reasoning first (cot)", () => ai("solve", { description: "Solve the word problem.", input: { problem: t.string() },
    output: t.number(), module: "cot", lm, maxTokens: 4000 })("A pen costs 3 dollars. How much do 7 pens cost?"));
  const lookup = tool("lookup_order", { description: "Look up where an order is.", input: { order: t.string() } },
    ({ order }) => (order === "A-1042" ? "stuck at the carrier since Monday" : "unknown order"));
  await check("a tool", () => ai("support", { description: "Answer the customer, looking up their order.", input: { message: t.string() },
    tools: [lookup], lm })("Where is my order A-1042?"));
  await check("streaming", async () => {
    const s = ai("haiku", { description: "A haiku about the topic.", input: { topic: t.string() }, lm }).stream("the first snow");
    let pieces = 0;
    let text = "";
    for await (const piece of s) { pieces++; text += piece; }
    const value = await s.result;
    if (text.trim() !== String(value).trim()) throw new Error(`pieces ${JSON.stringify(text)} != value ${JSON.stringify(value)}`);
    return { pieces, value };
  });
  await check("evaluate", async () => {
    const ev = await evaluate(mood, [
      { review: "Love it, works perfectly.", result: "happy" },
      { review: "Broke in a day.", result: "unhappy" },
      { review: "Good product but arrived late and damaged box.", result: "mixed" },
    ]);
    return String(ev);
  });
}
process.exit(failed ? 1 : 0);
