/**
 * What the type checker must see in functai's API. Never run: `npm run check`
 * (tsc) checks it. Each `@ts-expect-error` is an error the checker must
 * report; if one stops being an error, tsc fails on the unused directive.
 */

import * as z from "zod";
import { ai, evaluate, gepa, labeledFewShot, module, t, tool, type Prediction } from "../src/index.ts";

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

export async function optionalInputs() {
  const summarize = ai("summarize", {
    description: "Summarize.",
    input: { text: t.string(), note: t.optional(t.string()), tone: z.string().default("plain"), lang: z.string().nullable() },
  });
  await summarize({ text: "x", lang: null });                     // note and tone may be left out
  await summarize({ text: "x", lang: "fr", note: "short", tone: "warm" });
  const greet = ai("greet", { description: "Greet.", input: { name: t.optional(t.string()) } });
  await greet();                                                  // every input optional
  const brief = ai("brief", { description: "Brief.", input: { text: t.string(), note: t.optional(t.string()) } });
  await brief("text alone");                                      // one required input: its value alone

  // @ts-expect-error: nullable is not optional
  await summarize({ text: "x" });
  // @ts-expect-error: tone is text
  await summarize({ text: "x", lang: null, tone: 3 });
  // @ts-expect-error: two required inputs: by name
  await summarize("x");
  // @ts-expect-error: a function with a required input needs an argument
  await brief();
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

export async function modules() {
  const answer = ai("answer", { description: "Answer.", input: { message: t.string(), topic: t.string() } });
  const support = module("support", {
    description: "Answer a customer's message.",
    input: { message: t.string(), tone: t.withDefault(t.string(), "kind"), order: z.string().optional(), frame: t.opaque<Map<string, number>>() },
    output: t.string(),
    uses: [answer],
  }, async ({ message, tone, order, frame }, { signal }) => {
    is<Equal<typeof tone, string>>(true);                          // a default: always there in the code
    is<Equal<typeof order, string | undefined>>(true);             // optional with no default: may be absent
    is<Equal<typeof frame, Map<string, number>>>(true);
    return answer({ message: `${tone}: ${message} ${order ?? ""} ${frame.size}` , topic: "x" }, { signal });
  });
  is<Equal<Awaited<ReturnType<typeof support>>, string>>(true);
  await support({ message: "Hi", frame: new Map() });              // tone and order may be left out
  const triage = module("triage", { input: { ticket: t.string() }, outputs: { team: t.enum("billing", "shipping"), minutes: t.integer() } },
    ({ ticket }) => ({ team: ticket ? "billing" as const : "shipping" as const, minutes: 5 }));
  const got = await triage("I was charged twice.");                // one required input: its value alone
  is<Equal<typeof got, { team: "billing" | "shipping"; minutes: number }>>(true);
  for await (const e of triage.stream("x").events({ form: "kept" })) {
    if (e.kind === "text") e.field.toUpperCase();                  // events narrow by kind
    if (e.kind === "request") e.request.toFixed();
  }

  // @ts-expect-error: a module declares what it returns
  module("m", { input: {} }, () => "x");
  // @ts-expect-error: it returns what it declares
  module("m", { input: {}, output: t.integer() }, () => "x");
  // @ts-expect-error: message is required
  await support({ frame: new Map() });
  // @ts-expect-error: several outputs are returned by name
  module("m", { input: {}, outputs: { a: t.string(), b: t.integer() } }, () => ({ a: "x" }));
}
