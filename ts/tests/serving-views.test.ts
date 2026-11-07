/** What a served program's callers see (contract/serving.md, *What callers see*): the outside view by default, or the view its owner chooses. */

import assert from "node:assert/strict";
import { test } from "node:test";
import { Message, Response, type Request as ModelRequest } from "@lm15/lm15";
import { ai, Service, t, type StreamEvent } from "../src/index.ts";
import { FakeRouter } from "./fake.ts";

type Rec = Record<string, any>;

/** A model that thinks before it answers (its thinking comes as its own channel, as Claude's and Gemini's do). */
const thinker = () => new FakeRouter([], (request: ModelRequest) => new Response({
  model: request.model,
  message: Message.assistant([{ type: "thinking", text: "They sound pleased." }, { type: "text", text: "<result>\nhappy\n</result>" }] as never),
  finishReason: "stop" as never, usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
}));
const mood = (router: unknown, extra: Rec = {}) => ai("mood", { description: "How does the customer feel?", input: { review: t.string() },
  output: t.enum("happy", "unhappy"), router: router as never, lm: "gpt-4.1-mini", ...extra });

const post = (svc: Service, path: string, body: unknown) =>
  svc.fetch(new Request(`http://localhost${path}`, { method: "POST", body: JSON.stringify(body) }));
const get = (svc: Service, path: string) => svc.fetch(new Request(`http://localhost${path}`));

/** The events of a Server-Sent Events body. */
async function sse(r: globalThis.Response): Promise<Rec[]> {
  assert.equal(r.status, 200);
  return (await r.text()).split("\n\n").filter(Boolean).map((block) => JSON.parse(block.split("\n").find((l) => l.startsWith("data: "))!.slice(6)));
}
const kinds = (events: Rec[]) => [...new Set(events.map((e) => e["kind"] as string))].sort();

test("callers see the outside view unless the owner chooses another: no thinking, no model stream, no error message", async () => {
  const svc = new Service(mood(thinker()));
  assert.equal(svc.describe()["view"], "outside");
  const events = await sse(await post(svc, "/stream", { inputs: { review: "Lovely." } }));
  assert.deepEqual(kinds(events), ["done", "request", "started", "text"]);
  const failing = new Service(mood(new FakeRouter([], () => { throw new Error("the key sk-SECRET was refused"); })));
  const r = await post(failing, "/call", { inputs: { review: "x" } });
  assert.equal(r.status, 500);
  assert.ok(!(await r.text()).includes("SECRET"));
});

test("view: full shows callers everything the owner sees: thinking, the model's stream, the outputs FunctAI adds, an error's message", async () => {
  const svc = new Service(mood(thinker()), { view: "full" });
  assert.equal(svc.describe()["view"], "full");
  const events = await sse(await post(svc, "/stream", { inputs: { review: "Lovely." } }));
  assert.deepEqual(kinds(events), ["done", "piece", "request", "started", "text", "thinking"]);
  assert.equal(events.filter((e) => e["kind"] === "thinking").map((e) => e["text"]).join(""), "They sound pleased.");
  assert.deepEqual(events.filter((e) => e["kind"] === "piece").map((e) => e["lm15"]["type"]), ["start", "end"]);
  assert.equal(events.at(-2)!["lm15"]["finish_reason"], "stop");        // why the reply stopped, as lm15 said it

  const cot = new Service(mood(new FakeRouter([], () => "<reasoning>\nThey say lovely.\n</reasoning>\n<result>\nhappy\n</result>"), { module: "cot" }), { view: "full" });
  const called = await (await post(cot, "/call", { inputs: { review: "Lovely." } })).json() as Rec;
  assert.deepEqual(called["outputs"], { reasoning: "They say lovely.", result: "happy" });
  const outside = new Service(mood(new FakeRouter([], () => "<reasoning>\nThey say lovely.\n</reasoning>\n<result>\nhappy\n</result>"), { module: "cot" }));
  assert.deepEqual((await (await post(outside, "/call", { inputs: { review: "Lovely." } })).json() as Rec)["outputs"], { result: "happy" });

  const failing = new Service(mood(new FakeRouter([], () => { throw new Error("the key sk-SECRET was refused"); })), { view: "full" });
  const r = await post(failing, "/call", { inputs: { review: "x" } });
  assert.equal(r.status, 500);
  assert.match(((await r.json()) as Rec)["error"]["message"], /sk-SECRET was refused/);
});

test("view: kept shows what the owner's log keeps: with a field not kept, no thinking and no piece of the model's stream", async () => {
  const svc = new Service(mood(thinker(), { logContent: { review: false } }), { view: "kept" });
  const events = await sse(await post(svc, "/stream", { inputs: { review: "my password is hunter2" } }));
  assert.deepEqual(kinds(events), ["done", "request", "started", "text"]);
  assert.ok(!JSON.stringify(events).includes("hunter2"));
  const whole = new Service(mood(thinker()), { view: "kept" });
  assert.ok(kinds(await sse(await post(whole, "/stream", { inputs: { review: "Lovely." } }))).includes("thinking"));
});

test("a view of the owner's own: the answer and the thinking, nothing else; a call it shows keeps its start, requests and end", async () => {
  const view = { calls: (s: StreamEvent) => s.kind === "started" && s.parent === null, shows: (e: StreamEvent) => e.kind === "thinking" || e.kind === "text" ? e : null };
  const svc = new Service(mood(thinker()), { view });
  assert.equal(svc.describe()["view"], "own");
  const events = await sse(await post(svc, "/stream", { inputs: { review: "Lovely." } }));
  assert.deepEqual(kinds(events), ["done", "request", "started", "text", "thinking"]);
  assert.deepEqual(events.map((e) => e["after"]), [null, ...events.slice(0, -1).map((e) => ({ writer: e["writer"], seq: e["seq"] }))]);
  assert.throws(() => new Service(mood(thinker()), { view: "everything" as never }), /callers see a view/);
  assert.throws(() => new Service(mood(thinker()), { view: { shows: "thinking" } as never }), /callers see a view/);
});

test("a turn's events come from the store's kept form: full gives it whole, outside its boundary, an own view what it shows", async () => {
  const program = mood(thinker());
  const ends: Rec = {};
  for (const [name, view] of [["outside", "outside"], ["full", "full"], ["own", { shows: (e: StreamEvent) => e.kind === "piece" ? null : e }]] as const) {
    const svc = new Service(program, { view: view as never });
    const made = await (await post(svc, "/conversations/c1/turns", { inputs: { review: "Lovely." }, wait: true })).json() as Rec;
    assert.equal(made["state"], "done");
    ends[name] = kinds(await sse(await get(svc, `/conversations/c1/turns/${made["turn"]}/events`)));
  }
  assert.deepEqual(ends["outside"], ["done", "request", "started", "text"]);
  assert.deepEqual(ends["full"], ["done", "piece", "request", "started", "text", "thinking"]);
  assert.deepEqual(ends["own"], ["done", "request", "started", "text", "thinking"]);
});
