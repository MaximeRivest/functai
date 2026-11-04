/**
 * Notes a program declares (contract/programs.md, *Notes*) and the answer so
 * far (contract/streaming.md, *The answer so far*), through real calls with a
 * fake model: a module's and a tool's notes, checked where they are made;
 * the outside view; a choice that shows only once it is whole.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { type Request } from "@lm15/lm15";
import { ai, describe, interfaceSignature, InterfaceError, module, note, outsideNotes, t, tool } from "../src/index.ts";
import { FakeRouter, type Reply } from "./fake.ts";

type Rec = Record<string, any>;
const XML = (s: string) => `<result>\n${s}\n</result>`;
const results = (req: Request): Rec[] => req.messages.flatMap((m) => m.parts).filter((p) => p.type === "tool_result") as unknown as Rec[];

const Page = t.object({ url: t.string(), title: t.string() });

const screenshot = tool("screenshot", {
  description: "Take a screenshot of a page.", input: { url: t.string() }, effects: "reads",
  notes: { "page captured": { shape: Page, outside: true } },
}, ({ url }) => {
  note("page captured", { url, title: "Home" });
  return `captured ${url}`;
});

const router = () => new FakeRouter([], (req): Reply => {
  if (!results(req).length && req.tools?.length) return { calls: [{ id: "c1", name: "screenshot", input: { url: "https://a.example/" } }] };
  return XML("The fees appear late.");
});

const walker = (r: FakeRouter) => ai("walker", {
  description: "Walk the site and say what is wrong with it.", input: { url: t.string() }, output: t.string(),
  tools: [screenshot], router: r as never, lm: "gpt-4.1-mini",
});

const study = (r: FakeRouter) => {
  const w = walker(r);
  return module("study", {
    input: { url: t.string() }, output: t.string(), uses: [w],
    notes: {
      stage: { shape: t.choice(["walking", "writing"]), desc: "where the walk is", latest: true, outside: true },
      "cost so far": t.number(),
    },
  }, async ({ url }) => {
    note("stage", "walking");
    const found = await w({ url });
    note("cost so far", 0.01);
    note("stage", "writing");
    return found;
  });
};

test("an AI function declares its tools' notes; notes are not in the signature", () => {
  const w = walker(router());
  assert.deepEqual(w.interface.notes, [{ name: "page captured", outside: true, shape: {
    type: "object", properties: { url: { type: "string" }, title: { type: "string" } }, required: ["url", "title"] } }]);
  const { notes: _n, ...plain } = w.interface;
  assert.equal(interfaceSignature(w.interface), interfaceSignature(plain));
});

test("a module declares its notes in order, with their words and flags", () => {
  const notes = study(router()).interface.notes!;
  assert.deepEqual(notes.map((n) => n.name), ["stage", "cost so far"]);
  assert.deepEqual(notes[0], { name: "stage", shape: { enum: ["walking", "writing"], type: "string" }, desc: "where the walk is",
    latest: true, outside: true });
  assert.deepEqual(notes[1], { name: "cost so far", shape: { type: "number" } });
  // describe() words reach a note as they reach a field
  const m = module("m", { input: {}, output: t.string(), notes: { stage: describe(t.string(), "where it is") } }, () => "");
  assert.deepEqual(m.interface.notes, [{ name: "stage", shape: { type: "string" }, desc: "where it is" }]);
});

test("two tools that declare one note differently are refused; so is a note that is not JSON", () => {
  const other = tool("other", { input: { url: t.string() }, notes: { "page captured": t.string() } }, () => "");
  assert.throws(() => ai("both", { description: "Both.", input: { url: t.string() }, output: t.string(), tools: [screenshot, other] }),
    (err: unknown) => err instanceof InterfaceError && err.code === "interface-malformed" && err.field === "page captured");
  assert.throws(() => module("m", { input: {}, output: t.string(), notes: { frame: t.opaque() } }, () => ""),
    (err: unknown) => err instanceof InterfaceError && err.field === "frame");
  assert.throws(() => module("m", { input: {}, output: t.string(), notes: { stage: { shape: t.string(), optional: true } as never } }, () => ""),
    (err: unknown) => err instanceof InterfaceError && err.field === "stage");
});

test("a note is checked where it is made", async () => {
  const renamed = module("renamed", { input: {}, output: t.string(), notes: { stage: t.choice(["walking", "writing"]) } }, () => {
    note("stage changed", "walking");
    return "x";
  });
  const wrong = module("wrong", { input: {}, output: t.string(), notes: { stage: t.choice(["walking", "writing"]) } }, () => {
    note("stage", "resting");
    return "x";
  });
  const free = module("free", { input: {}, output: t.string() }, () => {
    note("anything at all", { x: 1 });
    return "x";
  });
  await assert.rejects(renamed({}), (err: unknown) => err instanceof InterfaceError && err.code === "interface-note" && err.field === "stage changed");
  await assert.rejects(wrong({}), (err: unknown) => err instanceof InterfaceError && err.code === "interface-note" && err.field === "stage");
  assert.equal(await free({}), "x");                   // a program that declares none refuses none
  note("stage", "outside any call: nothing");           // outside any call, nothing is checked
});

test("the outside view shows only the notes the outermost program declares outside", async () => {
  const s = study(router()).stream({ url: "https://a.example/" });
  const shown: Rec[] = [];
  for await (const e of s.events({ view: "outside" })) shown.push(e as Rec);
  assert.equal(await s.result, "The fees appear late.");
  assert.deepEqual(shown.filter((e) => e.kind === "note").map((e) => [e.name, e.data, e.function]),
    [["stage", "walking", "study"], ["stage", "writing", "study"]]);
  const whole: Rec[] = [];
  for await (const e of s.events()) whole.push(e as Rec);
  assert.deepEqual(new Set(whole.filter((e) => e.kind === "note").map((e) => e.name)), new Set(["stage", "page captured", "cost so far"]));
  assert.deepEqual(outsideNotes(study(router())), ["stage"]);
});

test("a choice in the answer so far shows only once it is whole", async () => {
  const reply = XML('[{"title": "Fees appear late", "severity": "medium"}, {"title": "No search", "severity": "high"}]');
  const findings = ai("findings", {
    description: "What is wrong with this page?", input: { page: t.string() },
    output: t.list(t.object({ title: t.string(), severity: t.choice(["low", "medium", "high"]) })),
    router: new FakeRouter([reply]) as never, lm: "gpt-4.1-mini",
  });
  const s = findings.stream("checkout");
  const seen: unknown[] = [];
  for await (const e of s.events()) if ((e as Rec).kind === "text") seen.push(s.partial);
  const severities = seen.flatMap((p) => (Array.isArray(p) ? p : [])).filter((f: Rec) => "severity" in f).map((f: Rec) => f.severity);
  assert.ok(severities.length && severities.every((x) => x === "medium" || x === "high"), JSON.stringify(severities));
  assert.ok(seen.some((p) => Array.isArray(p) && p.some((f: Rec) => f.title?.startsWith("Fees app") && !("severity" in f))));
  const result = await s.result;
  assert.deepEqual(s.partial, result);
});
