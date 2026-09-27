/**
 * What the type checker must see in functai's API. Never run: `npm run check`
 * (tsc) checks it. Each `@ts-expect-error` is an error the checker must
 * report; if one stops being an error, tsc fails on the unused directive.
 */

import * as z from "zod";
import { ai, evaluate, gepa, labeledFewShot, t, tool, type Prediction } from "../src/index.ts";

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
const is = <T extends true>(_: T) => undefined;

const team = ai("team", {
  description: "Which team should answer this customer message?",
  input: { message: t.string(), urgent: t.boolean() },
  output: t.enum("shipping", "billing"),
});
const mood = ai("mood", { description: "How does the customer feel?", input: { review: z.string() }, output: z.enum(["happy", "unhappy"]) });

export async function calls() {
  is<Equal<Awaited<ReturnType<typeof team>>, "shipping" | "billing">>(true);
  is<Equal<Awaited<ReturnType<typeof mood>>, "happy" | "unhappy">>(true);
  await team({ message: "x", urgent: false });
  await mood("It broke.");                                      // one input: its value alone
  await mood({ review: "It broke." });
  await team({ message: "x", urgent: true }, { lm: "gpt-6-luna", signal: AbortSignal.timeout(5000) });
  const p: Prediction = await team.predict({ message: "x", urgent: false });
  void p;
  const answers: ("happy" | "unhappy")[] = await mood.map(["a", "b"], { concurrency: 4 });
  void answers;

  // @ts-expect-error: a function with two inputs takes them by name
  await team("x");
  // @ts-expect-error: an input is missing
  await team({ message: "x" });
  // @ts-expect-error: the wrong type
  await team({ message: 1, urgent: false });
  // @ts-expect-error: no such input
  await team({ message: "x", urgent: false, extra: 1 });
  // @ts-expect-error: the wrong type, one input
  await mood(42);
  // @ts-expect-error: a setting that does not exist
  await mood("x", { temprature: 0 });
}

export async function rows() {
  const train = [{ message: "Charged twice.", urgent: false, category: "billing" as const }];
  await evaluate(team, train, { expected: "category" });
  await labeledFewShot(team, train, { k: 1, expected: "category" });
  const { fn, trials } = await gepa(team, train, { expected: "category", teacher: "gpt-6-sol" });
  is<Equal<typeof fn, typeof team>>(true);
  void trials;

  // @ts-expect-error: a misspelled input column
  await evaluate(team, [{ mesage: "typo", urgent: false, category: "billing" }]);
  // @ts-expect-error: expected names a column the rows do not have
  await evaluate(team, train, { expected: "categroy" });
  // @ts-expect-error: the same, in gepa
  await gepa(team, train, { expected: "categroy" });
}

export function tools() {
  tool("lookup_order", { input: { order: t.string(), n: t.integer() } }, ({ order, n }) => order.toUpperCase().repeat(n));
  // @ts-expect-error: `order` is text, not a number
  tool("lookup_order", { input: { order: t.string() } }, ({ order }) => order.toFixed());
}
